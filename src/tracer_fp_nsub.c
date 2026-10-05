/*
    tracer_fp_nsub.c

    K. Nishiwaki, 2026-06-18
    - compute number of solver time steps in each snapshot
    - determined by CFL of cooling time at min/max momentum
    - n_sub is also used for load-balancing
*/



#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "FP_Coef.h"
#include "tracer_fp_nsub.h"

static const double kTracerGyr = 3.1536e16;
static const double kTracerMpc = 3.0857e24;
#define TRACER_FP_NSUB_OMP_MIN_CELL 64

static int should_parallel_cells(int ncell)
{
#ifdef _OPENMP
    int threshold = TRACER_FP_NSUB_OMP_MIN_CELL;
    const int nthr = omp_get_max_threads();
    if (omp_in_parallel()) return 0;
    if (nthr > 1) threshold *= nthr;
    return ncell >= threshold;
#else
    return ncell >= TRACER_FP_NSUB_OMP_MIN_CELL;
#endif
}

static double quantize_dt_sub(double dt_sub_target)
{
    static const double kMantissaLadder[] = {
        1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.4, 8.0
    };
    const int nlevel = (int)(sizeof(kMantissaLadder) / sizeof(kMantissaLadder[0]));
    double exponent10, scale, normalized;
    double chosen;
    int i;

    if (!(dt_sub_target > 0.0) || !isfinite(dt_sub_target)) return dt_sub_target;

    exponent10 = floor(log10(dt_sub_target));
    scale = pow(10.0, exponent10);
    normalized = dt_sub_target / scale;
    chosen = kMantissaLadder[0];

    for (i = 0; i < nlevel; i++) {
        if (kMantissaLadder[i] <= normalized) {
            chosen = kMantissaLadder[i];
        } else {
            break;
        }
    }

    return chosen * scale;
}

static double quantize_on_fraction(double on_fraction)
{
    static const double kOnFractionBins[] = {
        1.0 / 8.0,
        1.0 / 4.0,
        1.0 / 3.0,
        1.0 / 2.0,
        1.0
    };
    const int nbins = (int)(sizeof(kOnFractionBins) / sizeof(kOnFractionBins[0]));
    double chosen;
    double best_distance;
    int i;

    if (!(on_fraction > 0.0) || !isfinite(on_fraction)) return 1.0;
    if (on_fraction >= 1.0) return 1.0;

    chosen = kOnFractionBins[0];
    best_distance = fabs(on_fraction - chosen);
    for (i = 1; i < nbins; i++) {
        const double distance = fabs(on_fraction - kOnFractionBins[i]);
        if (distance < best_distance) {
            chosen = kOnFractionBins[i];
            best_distance = distance;
        }
    }
    return chosen;
}

static int compute_nsubsteps_core(int ntracer,
                                            double dt_snap,
                                            const CRspectrum *crp_grid,
                                            const CRspectrum *cre_grid,
                                            const double *crp_radp_lo,
                                            const double *crp_radp_hi,
                                            const double *cre_radp_lo,
                                            const double *cre_radp_hi,
                                            const double *l_turb_mpc,
                                            const double *dv_imc,
                                            int *nsubsteps,
                                            int *target_nsubsteps,
                                            int *n_onsteps,
                                            unsigned char *capped_flags,
                                            int *max_nsubsteps,
                                            int *capped_count_out,
                                            int *max_raw_nsub_out)
{
    int itr;
    int max_steps = 0;
    int capped_count = 0;
    int max_raw_nsub = 0;

    if (ntracer <= 0 || dt_snap <= 0.0 || crp_grid == 0 || cre_grid == 0 ||
        crp_radp_lo == 0 || crp_radp_hi == 0 || cre_radp_lo == 0 || cre_radp_hi == 0 ||
        nsubsteps == 0 ||
        n_onsteps == 0 || l_turb_mpc == 0 || dv_imc == 0) {
        return -1;
    }

    for (itr = 0; itr < ntracer; itr++) {
        const double tcool_e0 = cre_grid->p[0] / cre_radp_lo[itr];
        const double tcool_p0 = crp_grid->p[0] / crp_radp_lo[itr];
        const double tcool_eN = cre_grid->p[npe - 1] / cre_radp_hi[itr];
        const double tcool_pN = crp_grid->p[np - 1] / crp_radp_hi[itr];
        const double tcool_min = fmin(fabs(tcool_p0),
                                      fmin(fabs(tcool_eN),
                                           fmin(fabs(tcool_e0), fabs(tcool_pN))));
        int nsub = 10;
        int n_on = nsub;
        const double safety = tracer_nsub_safety;
        double dt_sub_target = dt_snap / 10.0;
        double dt_sub_used = dt_sub_target;
        double target_nsub = 10.0;
        double raw_nsub = 10.0;
        double on_fraction = 1.0;

        if (isfinite(tcool_min) && tcool_min > 0.0) {
            const double dt_sub_stability = safety * tcool_min;
            if (dt_sub_stability < dt_sub_target) dt_sub_target = dt_sub_stability;
        }
        if (dt_sub_target <= 0.0 || !isfinite(dt_sub_target)) {
            dt_sub_target = dt_snap / 10.0;
        }
        target_nsub = ceil(dt_snap / dt_sub_target);
        if (!isfinite(target_nsub) || target_nsub > 1000000.0) target_nsub = 1000000.0;
        if (target_nsub < 10.0) target_nsub = 10.0;
        dt_sub_used = quantize_dt_sub(dt_sub_target);
        if (!(dt_sub_used > 0.0) || !isfinite(dt_sub_used) || dt_sub_used > dt_sub_target) {
            dt_sub_used = dt_sub_target;
        }

        raw_nsub = ceil(dt_snap / dt_sub_used);
        if (!isfinite(raw_nsub) || raw_nsub > 1000000.0) raw_nsub = 1000000.0;
        if (raw_nsub < 10.0) raw_nsub = 10.0;
        nsub = (int)raw_nsub;
        if (nsub > max_raw_nsub) max_raw_nsub = nsub;
        if (tracer_nsub_max > 0 && nsub > tracer_nsub_max) {
            nsub = tracer_nsub_max;
            capped_count++;
            if (capped_flags != 0) capped_flags[itr] = 1;
        }

        if (reacc_window_mode == REACC_WINDOW_CENTERED && f_eddy > 0.0 && dv_imc[itr] > 0.0) {
            const double t_eddy = l_turb_mpc[itr] * kTracerMpc / dv_imc[itr] / kTracerGyr;
            if (f_eddy * t_eddy < dt_snap) {
                on_fraction = quantize_on_fraction(f_eddy * t_eddy / dt_snap);
            }
        }
        n_on = (int)floor((double)nsub * on_fraction + 0.5);
        if (n_on < 1) n_on = 1;
        if (n_on > nsub) n_on = nsub;

        nsubsteps[itr] = nsub;
        if (target_nsubsteps != 0) target_nsubsteps[itr] = (int)target_nsub;
        n_onsteps[itr] = n_on;
        if (nsub > max_steps) max_steps = nsub;
    }

    if (max_nsubsteps != 0) *max_nsubsteps = max_steps;
    if (capped_count_out != 0) *capped_count_out = capped_count;
    if (max_raw_nsub_out != 0) *max_raw_nsub_out = max_raw_nsub;
    return 0;
}

int tracer_fp_compute_nsubsteps(int ntracer,
                                double dt_snap,
                                const CRspectrum *crp_grid,
                                const CRspectrum *cre_grid,
                                const double *crp_radp_batch,
                                const double *cre_radp_batch,
                                const double *l_turb_mpc,
                                const double *dv_imc,
                                int *nsubsteps,
                                int *target_nsubsteps,
                                int *n_onsteps,
                                unsigned char *capped_flags,
                                int *max_nsubsteps,
                                int *capped_count_out,
                                int *max_raw_nsub_out)
{
    return compute_nsubsteps_core(
        ntracer, dt_snap, crp_grid, cre_grid,
        crp_radp_batch,
        crp_radp_batch + (size_t)(np - 1) * (size_t)ntracer,
        cre_radp_batch,
        cre_radp_batch + (size_t)(npe - 1) * (size_t)ntracer,
        l_turb_mpc, dv_imc,
        nsubsteps, target_nsubsteps, n_onsteps, capped_flags,
        max_nsubsteps, capped_count_out, max_raw_nsub_out);
}

int tracer_fp_compute_nsubsteps_from_endpoints(int ntracer,
                                               double dt_snap,
                                               const CRspectrum *crp_grid,
                                               const CRspectrum *cre_grid,
                                               const double *crp_radp_lo,
                                               const double *crp_radp_hi,
                                               const double *cre_radp_lo,
                                               const double *cre_radp_hi,
                                               const double *l_turb_mpc,
                                               const double *dv_imc,
                                               int *nsubsteps,
                                               int *target_nsubsteps,
                                               int *n_onsteps,
                                               unsigned char *capped_flags,
                                               int *max_nsubsteps,
                                               int *capped_count_out,
                                               int *max_raw_nsub_out)
{
    return compute_nsubsteps_core(
        ntracer, dt_snap, crp_grid, cre_grid,
        crp_radp_lo, crp_radp_hi, cre_radp_lo, cre_radp_hi,
        l_turb_mpc, dv_imc,
        nsubsteps, target_nsubsteps, n_onsteps, capped_flags,
        max_nsubsteps, capped_count_out, max_raw_nsub_out);
}

int tracer_fp_prepare_nsub_loss_endpoints(int ntracer,
                                          const CRspectrum *crp_grid,
                                          const CRspectrum *cre_grid,
                                          const double *n_gas,
                                          const double *kbt,
                                          const double *b_field,
                                          const double *divv_gyr,
                                          const double *rad_ic_batch,
                                          double *crp_radp_lo,
                                          double *crp_radp_hi,
                                          double *cre_radp_lo,
                                          double *cre_radp_hi)
{
    int itr;

    if (ntracer <= 0 || crp_grid == 0 || cre_grid == 0 ||
        n_gas == 0 || kbt == 0 || b_field == 0 || divv_gyr == 0 ||
        rad_ic_batch == 0 ||
        crp_radp_lo == 0 || crp_radp_hi == 0 ||
        cre_radp_lo == 0 || cre_radp_hi == 0) {
        return -1;
    }

    #pragma omp parallel for schedule(static) if(should_parallel_cells(ntracer))
    for (itr = 0; itr < ntracer; itr++) {
        crp_radp_lo[itr] = b_Coulomb_p(n_gas[itr], crp_grid->p[0], kbt[itr]) +
                           adiabatic_divv(divv_gyr[itr], crp_grid->p[0]);
        crp_radp_hi[itr] = b_Coulomb_p(n_gas[itr], crp_grid->p[np - 1], kbt[itr]) +
                           adiabatic_divv(divv_gyr[itr], crp_grid->p[np - 1]);
        cre_radp_lo[itr] = b_synch(cre_grid->p[0], b_field[itr]) +
                           rad_ic_batch[itr] +
                           b_Coulomb_e(n_gas[itr], cre_grid->p[0]) +
                           adiabatic_divv(divv_gyr[itr], cre_grid->p[0]);
        cre_radp_hi[itr] = b_synch(cre_grid->p[npe - 1], b_field[itr]) +
                           rad_ic_batch[(size_t)(npe - 1) * (size_t)ntracer + (size_t)itr] +
                           b_Coulomb_e(n_gas[itr], cre_grid->p[npe - 1]) +
                           adiabatic_divv(divv_gyr[itr], cre_grid->p[npe - 1]);
    }

    return 0;
}
