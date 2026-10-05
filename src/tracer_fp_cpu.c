/*
    tracer_fp_cpu.c

    Tracer CPU solve core calling batched solver
*/

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "FP_Coef.h"
#include "tracer_fp_coef.h"
#include "tracer_fp.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1.0e3 * (double)ts.tv_sec + 1.0e-6 * (double)ts.tv_nsec;
}

static int cpu_secondary_refresh_each_fp_step(void)
{
    static int initialized = 0;
    static int enabled = 0;
    const char *env;

    if (initialized) return enabled;

    env = getenv("CROMA_CPU_SECONDARY_REFRESH");
    enabled = (env != 0 &&
               (strcmp(env, "fpstep") == 0 ||
                strcmp(env, "step") == 0 ||
                strcmp(env, "1") == 0 ||
                strcmp(env, "true") == 0 ||
                strcmp(env, "TRUE") == 0 ||
                strcmp(env, "yes") == 0 ||
                strcmp(env, "YES") == 0 ||
                strcmp(env, "on") == 0 ||
                strcmp(env, "ON") == 0));
    initialized = 1;
    return enabled;
}

static void refresh_secondary_sources(int ncell,
                                      const CRspectrum *crp_grid,
                                      const double *n_gas,
                                      const double *qepri_batch,
                                      const double *fqe_flat,
                                      const int *np_min_qe,
                                      const double *crp_state,
                                      TracerFpGpuTimes *times,
                                      TracerFpCpuWs *ws)
{
    const double t0 = now_ms();

    prepare_secondary_sources_cell_major(ncell,
                                         n_gas,
                                         crp_state,
                                         crp_grid->dp,
                                         np_min_qe,
                                         fqe_flat,
                                         qepri_batch,
                                         ws->source.qe_integral,
                                         ws->source.inje);
    times->secondary_ms += now_ms() - t0;
}

int tracer_fp_cpu_threads(void)
{
#ifdef _OPENMP
    {
        const int nthr = omp_get_max_threads();
        return (nthr > 0) ? nthr : 1;
    }
#else
    return 1;
#endif
}

int tracer_fp_cpu_evolve(int ncell,
                         int nstep,
                         double dt,
                         int use_windowed_reacc,
                         int on_start_step,
                         int on_end_step,
                         const CRspectrum *crp_grid,
                         const CRspectrum *cre_grid,
                         const double *n_gas,
                         const double *kbt,
                         const double *b_field,
                         const double *divv_gyr,
                         const double *l_turb_mpc,
                         const double *dv_imc,
                         const double *cs,
                         const double *beta_pl,
                         const double *rad_ic_batch,
                         const double *rad_ic_m1,
                         const double *rad_ic_p1,
                         const double *qpi_batch,
                         const double *qepri_batch,
                         const double *fqe_flat,
                         const int *np_min_qe,
                         const double *tracer_mass,
                         const unsigned char *disable_adiabatic,
                         double *crp_state,
                         double *cre_state,
                         TracerFpGpuTimes *times,
                         TracerFpCpuWs *ws)
{
    FpChangCooperSolveBatchInput solve_in;
    FpChangCooperSolveBatchCpuTimes solve_times;
    double t0, t1;
    int istep;
    int ierr = -1;
    const int refresh_secondary_each_step = cpu_secondary_refresh_each_fp_step();

    memset(&solve_in, 0, sizeof(solve_in));
    if (tracer_fp_prepare_coeff_batches(ncell, dt,
                                        crp_grid, cre_grid,
                                        n_gas, kbt, b_field, divv_gyr,
                                        l_turb_mpc, dv_imc, cs, beta_pl,
                                        rad_ic_batch, rad_ic_m1, rad_ic_p1,
                                        qpi_batch, qepri_batch,
                                        fqe_flat, np_min_qe,
                                        tracer_mass, disable_adiabatic,
                                        crp_state, cre_state,
                                        times, ws) != 0) {
        goto cleanup;
    }

    solve_in.nsys = ncell;
    solve_in.dt = dt;

    if (!refresh_secondary_each_step) {
        /* Refresh once per coefficient batch. The batch caller re-enters this
         * routine at coefficient-interpolation segment boundaries. */
        refresh_secondary_sources(ncell, crp_grid, n_gas, qepri_batch,
                                  fqe_flat, np_min_qe, crp_state, times, ws);
    }

    for (istep = 0; istep < nstep; istep++) {
        const int use_on_coeff =
            !use_windowed_reacc ||
            (istep >= on_start_step && istep < on_end_step);

        solve_in.nrow = np;
        solve_in.a_batch = use_on_coeff ? ws->cc.ccp_a : ws->cc.ccp_a_off;
        solve_in.b_batch = use_on_coeff ? ws->cc.ccp_b : ws->cc.ccp_b_off;
        solve_in.c_batch = use_on_coeff ? ws->cc.ccp_c : ws->cc.ccp_c_off;
        solve_in.source_batch = qpi_batch;

        t0 = now_ms();
        if (solve_cc_batch_cpu_timed(&solve_in, crp_state, &solve_times) != 0) {
            goto cleanup;
        }
        t1 = now_ms();
        times->solve_ms += t1 - t0;
        times->solve_alloc_ms += solve_times.alloc_ms;
        times->solve_rhs_ms += solve_times.rhs_ms;
        times->solve_tridiag_ms += solve_times.tridiag_ms;

        if (refresh_secondary_each_step) {
            refresh_secondary_sources(ncell, crp_grid, n_gas, qepri_batch,
                                      fqe_flat, np_min_qe, crp_state, times, ws);
        }

        solve_in.nrow = npe;
        solve_in.a_batch = use_on_coeff ? ws->cc.cce_a : ws->cc.cce_a_off;
        solve_in.b_batch = use_on_coeff ? ws->cc.cce_b : ws->cc.cce_b_off;
        solve_in.c_batch = use_on_coeff ? ws->cc.cce_c : ws->cc.cce_c_off;
        solve_in.source_batch = ws->source.inje;

        t0 = now_ms();
        if (solve_cc_batch_cpu_timed(&solve_in, cre_state, &solve_times) != 0) {
            goto cleanup;
        }
        t1 = now_ms();
        times->solve_ms += t1 - t0;
        times->solve_alloc_ms += solve_times.alloc_ms;
        times->solve_rhs_ms += solve_times.rhs_ms;
        times->solve_tridiag_ms += solve_times.tridiag_ms;
    }

    ierr = 0;

cleanup:
    return ierr;
}
