/*
    tracer_fp_coef.c
    
    K. Nishiwaki, 2026-06-25
    - FP coeff and CC coeff utilities
    
*/
   

#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tracer_fp_coef.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1.0e3 * (double)ts.tv_sec + 1.0e-6 * (double)ts.tv_nsec;
}

static int alloc_doubles(double **ptr, size_t count)
{
    *ptr = (double *)malloc(count * sizeof(double));
    return (*ptr != 0) ? 0 : -1;
}

static int alloc_uchars(unsigned char **ptr, size_t count)
{
    *ptr = (unsigned char *)malloc(count * sizeof(unsigned char));
    return (*ptr != 0) ? 0 : -1;
}

int tracer_fp_snapshot_ws_ensure(TracerFpSnapshotWs *ws,
                                 int ncell)
{
    if (ws == 0 || ncell <= 0) return -1;
    if (ws->capacity_ncell >= ncell) return 0;

    tracer_fp_snapshot_ws_free(ws);

    if (alloc_doubles(&ws->loss.crp_radpm1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->loss.crp_radpp1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->loss.cre_radpm1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->loss.cre_radpp1, (size_t)ncell) != 0) {
        tracer_fp_snapshot_ws_free(ws);
        return -1;
    }

    ws->capacity_ncell = ncell;
    return 0;
}

void tracer_fp_snapshot_ws_free(TracerFpSnapshotWs *ws)
{
    if (ws == 0) return;

    free(ws->loss.crp_radp); free(ws->loss.crp_tloss); free(ws->loss.crp_invtloss);
    free(ws->loss.cre_radp); free(ws->loss.cre_tloss); free(ws->loss.cre_invtloss);
    free(ws->loss.crp_radpm1); free(ws->loss.crp_radpp1);
    free(ws->loss.cre_radpm1); free(ws->loss.cre_radpp1);
    memset(ws, 0, sizeof(*ws));
}

int tracer_fp_gpu_host_ws_ensure(TracerFpGpuHostWs *ws,
                                 int ncell)
{
    if (ws == 0 || ncell <= 0) return -1;
    if (ws->capacity_ncell >= ncell) return 0;

    tracer_fp_gpu_host_ws_free(ws);

    if (alloc_doubles(&ws->bucket.b_dyn, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bucket.logb, (size_t)ncell) != 0) {
        tracer_fp_gpu_host_ws_free(ws);
        return -1;
    }

    ws->capacity_ncell = ncell;
    return 0;
}

void tracer_fp_gpu_host_ws_free(TracerFpGpuHostWs *ws)
{
    if (ws == 0) return;

    free(ws->bucket.b_dyn);
    free(ws->bucket.logb);
    memset(ws, 0, sizeof(*ws));
}

int tracer_fp_cpu_ws_ensure(TracerFpCpuWs *ws,
                            int ncell)
{
    const size_t np_batch = (size_t)ncell * (size_t)np;
    const size_t npe_batch = (size_t)ncell * (size_t)npe;

    if (ws == 0 || ncell <= 0) return -1;
    if (ws->capacity_ncell >= ncell) return 0;

    tracer_fp_cpu_ws_free(ws);

    if (alloc_doubles(&ws->bg_curr.n_gas, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_curr.kbt, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_curr.b_field, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_curr.divv, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_curr.lturb, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_curr.dv, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_curr.cs, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_curr.beta, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_curr.rad_ic, npe_batch) != 0 ||
        alloc_doubles(&ws->bg_curr.rad_ic_m1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_curr.rad_ic_p1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.n_gas, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.kbt, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.b_field, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.divv, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.lturb, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.dv, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.cs, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.beta, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.rad_ic, npe_batch) != 0 ||
        alloc_doubles(&ws->bg_next.rad_ic_m1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_next.rad_ic_p1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.n_gas, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.kbt, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.b_field, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.divv, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.lturb, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.dv, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.cs, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.beta, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.rad_ic, npe_batch) != 0 ||
        alloc_doubles(&ws->bg_interp.rad_ic_m1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->bg_interp.rad_ic_p1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->state.qpi, np_batch) != 0 ||
        alloc_doubles(&ws->state.qepri, npe_batch) != 0 ||
        alloc_doubles(&ws->state.crp, np_batch) != 0 ||
        alloc_doubles(&ws->state.cre, npe_batch) != 0 ||
        alloc_doubles(&ws->state.mass_msun, (size_t)ncell) != 0 ||
        alloc_uchars(&ws->state.disable_adiabatic, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.crp_radp, np_batch) != 0 ||
        alloc_doubles(&ws->coeff.crp_tloss, np_batch) != 0 ||
        alloc_doubles(&ws->coeff.crp_invtloss, np_batch) != 0 ||
        alloc_doubles(&ws->coeff.cre_radp, npe_batch) != 0 ||
        alloc_doubles(&ws->coeff.cre_tloss, npe_batch) != 0 ||
        alloc_doubles(&ws->coeff.cre_invtloss, npe_batch) != 0 ||
        alloc_doubles(&ws->coeff.dpp, np_batch) != 0 ||
        alloc_doubles(&ws->coeff.dppe, npe_batch) != 0 ||
        alloc_doubles(&ws->coeff.dpp_off, np_batch) != 0 ||
        alloc_doubles(&ws->coeff.dppe_off, npe_batch) != 0 ||
        alloc_doubles(&ws->coeff.crp_radpm1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.crp_radpp1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.cre_radpm1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.cre_radpp1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.dppm1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.dppp1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.dppem1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.dppep1, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.dppm1_off, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.dppp1_off, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.dppem1_off, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->coeff.dppep1_off, (size_t)ncell) != 0 ||
        alloc_doubles(&ws->cc.ccp_a, np_batch) != 0 ||
        alloc_doubles(&ws->cc.ccp_b, np_batch) != 0 ||
        alloc_doubles(&ws->cc.ccp_c, np_batch) != 0 ||
        alloc_doubles(&ws->cc.cce_a, npe_batch) != 0 ||
        alloc_doubles(&ws->cc.cce_b, npe_batch) != 0 ||
        alloc_doubles(&ws->cc.cce_c, npe_batch) != 0 ||
        alloc_doubles(&ws->cc.ccp_a_off, np_batch) != 0 ||
        alloc_doubles(&ws->cc.ccp_b_off, np_batch) != 0 ||
        alloc_doubles(&ws->cc.ccp_c_off, np_batch) != 0 ||
        alloc_doubles(&ws->cc.cce_a_off, npe_batch) != 0 ||
        alloc_doubles(&ws->cc.cce_b_off, npe_batch) != 0 ||
        alloc_doubles(&ws->cc.cce_c_off, npe_batch) != 0 ||
        alloc_doubles(&ws->source.qe_integral, npe_batch) != 0 ||
        alloc_doubles(&ws->source.inje, npe_batch) != 0) {
        tracer_fp_cpu_ws_free(ws);
        return -1;
    }

    ws->capacity_ncell = ncell;
    return 0;
}

void tracer_fp_cpu_ws_free(TracerFpCpuWs *ws)
{
    if (ws == 0) return;

    free(ws->bg_curr.n_gas); free(ws->bg_curr.kbt); free(ws->bg_curr.b_field); free(ws->bg_curr.divv);
    free(ws->bg_curr.lturb); free(ws->bg_curr.dv); free(ws->bg_curr.cs); free(ws->bg_curr.beta);
    free(ws->bg_curr.rad_ic); free(ws->bg_curr.rad_ic_m1); free(ws->bg_curr.rad_ic_p1);
    free(ws->bg_next.n_gas); free(ws->bg_next.kbt); free(ws->bg_next.b_field); free(ws->bg_next.divv);
    free(ws->bg_next.lturb); free(ws->bg_next.dv); free(ws->bg_next.cs); free(ws->bg_next.beta);
    free(ws->bg_next.rad_ic); free(ws->bg_next.rad_ic_m1); free(ws->bg_next.rad_ic_p1);
    free(ws->bg_interp.n_gas); free(ws->bg_interp.kbt); free(ws->bg_interp.b_field); free(ws->bg_interp.divv);
    free(ws->bg_interp.lturb); free(ws->bg_interp.dv); free(ws->bg_interp.cs); free(ws->bg_interp.beta);
    free(ws->bg_interp.rad_ic); free(ws->bg_interp.rad_ic_m1); free(ws->bg_interp.rad_ic_p1);
    free(ws->state.qpi); free(ws->state.qepri); free(ws->state.crp); free(ws->state.cre);
    free(ws->state.mass_msun);
    free(ws->state.disable_adiabatic);
    tracer_fp_snapshot_ws_free(&ws->snapshot);
    free(ws->coeff.crp_radp); free(ws->coeff.crp_tloss); free(ws->coeff.crp_invtloss);
    free(ws->coeff.cre_radp); free(ws->coeff.cre_tloss); free(ws->coeff.cre_invtloss);
    free(ws->coeff.dpp); free(ws->coeff.dppe); free(ws->coeff.dpp_off); free(ws->coeff.dppe_off);
    free(ws->coeff.crp_radpm1); free(ws->coeff.crp_radpp1); free(ws->coeff.cre_radpm1); free(ws->coeff.cre_radpp1);
    free(ws->coeff.dppm1); free(ws->coeff.dppp1); free(ws->coeff.dppem1); free(ws->coeff.dppep1);
    free(ws->coeff.dppm1_off); free(ws->coeff.dppp1_off); free(ws->coeff.dppem1_off); free(ws->coeff.dppep1_off);
    free(ws->cc.ccp_a); free(ws->cc.ccp_b); free(ws->cc.ccp_c);
    free(ws->cc.cce_a); free(ws->cc.cce_b); free(ws->cc.cce_c);
    free(ws->cc.ccp_a_off); free(ws->cc.ccp_b_off); free(ws->cc.ccp_c_off);
    free(ws->cc.cce_a_off); free(ws->cc.cce_b_off); free(ws->cc.cce_c_off);
    free(ws->source.qe_integral); free(ws->source.inje);
    memset(ws, 0, sizeof(*ws));
}

int tracer_fp_prepare_coeff_batches(int ncell,
                                    double dt,
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
                                    const double *crp_state,
                                    const double *cre_state,
                                    TracerFpGpuTimes *times,
                                    TracerFpCpuWs *ws)
{
    FpCoeffBatchInput coeff_in;
    FpCoeffCellMajorOutput coeff_out;
    FpChangCooperCoeffCellMajorInput cc_in;
    FpChangCooperCoeffBatchOutput cc_out;
    double t0, t1;

    if (tracer_fp_cpu_ws_ensure(ws, ncell) != 0) {
        return -1;
    }

    memset(&coeff_in, 0, sizeof(coeff_in));
    memset(&coeff_out, 0, sizeof(coeff_out));
    memset(&cc_in, 0, sizeof(cc_in));
    memset(&cc_out, 0, sizeof(cc_out));

    coeff_in.ncell = ncell;
    coeff_in.z = 0.0;
    coeff_in.dt_gyr = dt;
    coeff_in.epmax = 1.0e7;
    coeff_in.n_gas = n_gas;
    coeff_in.kbt = kbt;
    coeff_in.b_field = b_field;
    coeff_in.divv_gyr = divv_gyr;
    coeff_in.l_turb_mpc = l_turb_mpc;
    coeff_in.dv_imc = dv_imc;
    coeff_in.cs = cs;
    coeff_in.beta_pl = beta_pl;
    coeff_in.tracer_mass_msun = tracer_mass;
    coeff_in.disable_adiabatic = disable_adiabatic;
    coeff_in.crp_state_cell_major = crp_state;
    coeff_in.cre_state_cell_major = cre_state;
    coeff_in.rad_ic_batch = rad_ic_batch;
    coeff_in.rad_ic_m1 = rad_ic_m1;
    coeff_in.rad_ic_p1 = rad_ic_p1;

    coeff_out.loss.crp_radp_batch = ws->coeff.crp_radp;
    coeff_out.loss.crp_tloss_batch = ws->coeff.crp_tloss;
    coeff_out.loss.crp_invtloss_batch = ws->coeff.crp_invtloss;
    coeff_out.loss.cre_radp_batch = ws->coeff.cre_radp;
    coeff_out.loss.cre_tloss_batch = ws->coeff.cre_tloss;
    coeff_out.loss.cre_invtloss_batch = ws->coeff.cre_invtloss;
    coeff_out.loss.crp_radpm1 = ws->coeff.crp_radpm1;
    coeff_out.loss.crp_radpp1 = ws->coeff.crp_radpp1;
    coeff_out.loss.cre_radpm1 = ws->coeff.cre_radpm1;
    coeff_out.loss.cre_radpp1 = ws->coeff.cre_radpp1;
    coeff_out.dpp_batch = ws->coeff.dpp;
    coeff_out.dppe_batch = ws->coeff.dppe;
    coeff_out.dpp_off_batch = ws->coeff.dpp_off;
    coeff_out.dppe_off_batch = ws->coeff.dppe_off;
    coeff_out.dppm1 = ws->coeff.dppm1;
    coeff_out.dppp1 = ws->coeff.dppp1;
    coeff_out.dppem1 = ws->coeff.dppem1;
    coeff_out.dppep1 = ws->coeff.dppep1;
    coeff_out.dppm1_off = ws->coeff.dppm1_off;
    coeff_out.dppp1_off = ws->coeff.dppp1_off;
    coeff_out.dppem1_off = ws->coeff.dppem1_off;
    coeff_out.dppep1_off = ws->coeff.dppep1_off;

    t0 = now_ms();
    if (prepare_fp_coefficients_cell_major_cpu(&coeff_in, crp_grid, cre_grid, &coeff_out) != 0) {
        return -1;
    }

    cc_in.ncell = ncell;
    cc_in.dt = dt;
    cc_in.crp_radp_batch = ws->coeff.crp_radp;
    cc_in.crp_tloss_batch = ws->coeff.crp_tloss;
    cc_in.dpp_batch = ws->coeff.dpp;
    cc_in.qpi_batch = qpi_batch;
    cc_in.cre_radp_batch = ws->coeff.cre_radp;
    cc_in.cre_tloss_batch = ws->coeff.cre_tloss;
    cc_in.dppe_batch = ws->coeff.dppe;
    cc_in.inje_batch = ws->source.inje;
    cc_in.crp_radpm1 = ws->coeff.crp_radpm1;
    cc_in.crp_radpp1 = ws->coeff.crp_radpp1;
    cc_in.cre_radpm1 = ws->coeff.cre_radpm1;
    cc_in.cre_radpp1 = ws->coeff.cre_radpp1;
    cc_in.dppm1 = ws->coeff.dppm1;
    cc_in.dppp1 = ws->coeff.dppp1;
    cc_in.dppem1 = ws->coeff.dppem1;
    cc_in.dppep1 = ws->coeff.dppep1;

    cc_out.ccp_a_batch = ws->cc.ccp_a;
    cc_out.ccp_b_batch = ws->cc.ccp_b;
    cc_out.ccp_c_batch = ws->cc.ccp_c;
    cc_out.cce_a_batch = ws->cc.cce_a;
    cc_out.cce_b_batch = ws->cc.cce_b;
    cc_out.cce_c_batch = ws->cc.cce_c;

    if (prepare_cc_coeff_cell_major(&cc_in, crp_grid, cre_grid, &cc_out) != 0) {
        return -1;
    }

    cc_in.dpp_batch = ws->coeff.dpp_off;
    cc_in.dppe_batch = ws->coeff.dppe_off;
    cc_in.dppm1 = ws->coeff.dppm1_off;
    cc_in.dppp1 = ws->coeff.dppp1_off;
    cc_in.dppem1 = ws->coeff.dppem1_off;
    cc_in.dppep1 = ws->coeff.dppep1_off;
    cc_out.ccp_a_batch = ws->cc.ccp_a_off;
    cc_out.ccp_b_batch = ws->cc.ccp_b_off;
    cc_out.ccp_c_batch = ws->cc.ccp_c_off;
    cc_out.cce_a_batch = ws->cc.cce_a_off;
    cc_out.cce_b_batch = ws->cc.cce_b_off;
    cc_out.cce_c_batch = ws->cc.cce_c_off;

    if (prepare_cc_coeff_cell_major(&cc_in, crp_grid, cre_grid, &cc_out) != 0) {
        return -1;
    }
    t1 = now_ms();
    if (times != 0) {
        times->coeff_ms += t1 - t0;
    }

    return 0;
}
