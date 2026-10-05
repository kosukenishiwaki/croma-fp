/*
    tracer_fp_synch.c

    K. Nishiwaki, 2026-06-18
    - setup synchrotron tables with binned B-field, pitch-angle, and frequency.
    
*/

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "FP_Coef.h"
#include "params.h"
#include "Synchrotron.h"
#include "tracer_fp.h"
#include "tracer_fp_synch.h"

static const double kTracerSynchLogBPad = 0.25;

static void build_pointer_view_2d(double **rows,
                                            double *storage,
                                            int n0,
                                            int n1)
{
    int i;
    for (i = 0; i < n0; i++) {
        rows[i] = storage + (size_t)i * (size_t)n1;
    }
}

static void build_pointer_view_3d(double ***planes,
                                            double **rows,
                                            double *storage,
                                            int n0,
                                            int n1,
                                            int n2)
{
    int i, j;
    for (i = 0; i < n0; i++) {
        planes[i] = rows + (size_t)i * (size_t)n1;
        for (j = 0; j < n1; j++) {
            planes[i][j] = storage + ((size_t)i * (size_t)n1 + (size_t)j) * (size_t)n2;
        }
    }
}

void tracer_fp_synch_data_reset(SynchData *synch)
{
    if (synch == 0) return;
    memset(synch, 0, sizeof(*synch));
    synch->ntheta_pitch = tracer_synch_ntheta_pitch;
    synch->nx_tab = 2750;
    synch->xmin = -25.0;
    synch->xmax = 2.5;
    synch->dxtab = (synch->xmax - synch->xmin) / (double)synch->nx_tab;
}

int tracer_fp_synch_data_alloc(SynchData *synch, int nfreq)
{
    if (synch == 0 || nfreq <= 0) return -1;
    synch->ntheta_pitch = tracer_synch_ntheta_pitch;

    synch->logx_tab = (double *)calloc((size_t)synch->nx_tab, sizeof(double));
    synch->logfx_tab = (double *)calloc((size_t)synch->nx_tab, sizeof(double));
    synch->fx_tab = (double *)calloc((size_t)synch->nx_tab, sizeof(double));
    synch->theta = (double *)calloc((size_t)synch->ntheta_pitch, sizeof(double));
    synch->dtheta = (double *)calloc((size_t)synch->ntheta_pitch, sizeof(double));
    synch->pitch_weight = (double *)calloc((size_t)synch->ntheta_pitch, sizeof(double));
    synch->nus = (double *)calloc((size_t)nfreq, sizeof(double));
    synch->nu_crit_storage = (double *)calloc((size_t)npe * (size_t)synch->ntheta_pitch, sizeof(double));
    synch->logy_storage = (double *)calloc((size_t)nfreq * (size_t)npe * (size_t)synch->ntheta_pitch, sizeof(double));
    synch->nu_crit_rows = (double **)calloc((size_t)npe, sizeof(double *));
    synch->logy_rows = (double **)calloc((size_t)nfreq * (size_t)npe, sizeof(double *));
    synch->logy_planes = (double ***)calloc((size_t)nfreq, sizeof(double **));

    if (synch->logx_tab == 0 || synch->logfx_tab == 0 || synch->fx_tab == 0 ||
        synch->theta == 0 || synch->dtheta == 0 || synch->pitch_weight == 0 ||
        synch->nus == 0 || synch->nu_crit_storage == 0 || synch->logy_storage == 0 ||
        synch->nu_crit_rows == 0 || synch->logy_rows == 0 || synch->logy_planes == 0) {
        return -1;
    }

    return 0;
}

int tracer_fp_synch_data_init_tables(SynchData *synch,
                                     int nfreq,
                                     const CRspectrum *cre_grid)
{
    int i, k;

    if (synch == 0 || cre_grid == 0 || nfreq <= 0) return -1;

    for (i = 0; i < synch->nx_tab; i++) synch->logx_tab[i] = synch->xmin + synch->dxtab * (double)i;
    SYN_logFx_table(synch->nx_tab, synch->logx_tab, synch->logfx_tab);
    for (i = 0; i < synch->nx_tab; i++) synch->fx_tab[i] = pow(10.0, synch->logfx_tab[i]);

    for (k = 0; k < synch->ntheta_pitch; k++) {
        synch->theta[k] = (k + 0.5) * M_PI / (double)synch->ntheta_pitch;
        synch->dtheta[k] = M_PI / (double)synch->ntheta_pitch;
    }
    SYN_pitch_weight_table(synch->theta, synch->dtheta, synch->pitch_weight);

    {
        const double dnu = (nu_max_s - nu_min_s) / (double)nfreq;
        double a = nu_min_s - dnu;
        for (i = 0; i < nfreq; i++) {
            a += dnu;
            synch->nus[i] = pow(10.0, a);
        }
    }

    build_pointer_view_2d(synch->nu_crit_rows, synch->nu_crit_storage, npe, synch->ntheta_pitch);
    build_pointer_view_3d(synch->logy_planes, synch->logy_rows, synch->logy_storage,
                                    nfreq, npe, synch->ntheta_pitch);
    SYN_nu_crit_subB(synch->nu_crit_rows, synch->theta, (double *)cre_grid->p);
    SYN_logy_table(synch->logy_planes, synch->nu_crit_rows, nfreq, synch->nus, 1.0e-6);

    return 0;
}

void tracer_fp_synch_data_release(SynchData *synch)
{
    if (synch == 0) return;
    free(synch->logx_tab);
    free(synch->logfx_tab);
    free(synch->fx_tab);
    free(synch->theta);
    free(synch->dtheta);
    free(synch->pitch_weight);
    free(synch->nus);
    free(synch->nu_crit_storage);
    free(synch->logy_storage);
    free(synch->nu_crit_rows);
    free(synch->logy_rows);
    free(synch->logy_planes);
    free(synch->kernel_table);
    tracer_fp_synch_data_reset(synch);
}

void tracer_fp_logb_range_from_snapshot(int ntracer,
                                        const double *b_dyn,
                                        double *logb,
                                        double *logb_min_out,
                                        double *logb_max_out,
                                        int *nout_of_range_out)
{
    int itr;
    double logb_min = 0.0;
    double logb_max = 0.0;
    int nout_of_range = 0;

    if (logb_min_out) *logb_min_out = 0.0;
    if (logb_max_out) *logb_max_out = 0.0;
    if (nout_of_range_out) *nout_of_range_out = 0;
    if (ntracer <= 0 || b_dyn == 0 || logb == 0) return;

    for (itr = 0; itr < ntracer; itr++) {
        double bval = b_dyn[itr];
        if (!(bval > 0.0) || !isfinite(bval)) bval = 1.0e-30;
        logb[itr] = log10(bval / 1.0e-6);
        if (logb[itr] < tracer_synch_logb_min || logb[itr] > tracer_synch_logb_max) {
            nout_of_range++;
        }
        if (itr == 0 || logb[itr] < logb_min) logb_min = logb[itr];
        if (itr == 0 || logb[itr] > logb_max) logb_max = logb[itr];
    }

    if (logb_min_out) *logb_min_out = logb_min;
    if (logb_max_out) *logb_max_out = logb_max;
    if (nout_of_range_out) *nout_of_range_out = nout_of_range;
}

int tracer_fp_refresh_synch_kernel_table(SynchData *synch,
                                         int nfreq,
                                         double snapshot_logb_min,
                                         double snapshot_logb_max,
                                         double **kernel_table_io)
{
    FpSynchEmissionBatchInput table_in;
    double *kernel_table;
    double logb_min;
    double logb_max;
    double dlogb;

    if (synch == 0 || kernel_table_io == 0) return -1;

    if (adaptive_synch_logb) {
        logb_min = snapshot_logb_min - kTracerSynchLogBPad;
        logb_max = snapshot_logb_max + kTracerSynchLogBPad;
        if (logb_min < tracer_synch_logb_min) logb_min = tracer_synch_logb_min;
        if (logb_max > tracer_synch_logb_max) logb_max = tracer_synch_logb_max;
        if (!(logb_max > logb_min)) {
            logb_min -= 0.5;
            logb_max += 0.5;
        }
    } else {
        logb_min = tracer_synch_logb_min;
        logb_max = tracer_synch_logb_max;
    }

    if (*kernel_table_io != 0 &&
        logb_min >= synch->table_logb_min &&
        logb_max <= synch->table_logb_max) {
        return 0;
    }

    dlogb = (logb_max - logb_min) / (double)(TRACER_FP_SYNCH_LOGB_BINS - 1);
    if (!(dlogb > 0.0)) return -1;

    kernel_table = (double *)calloc((size_t)TRACER_FP_SYNCH_LOGB_BINS * (size_t)nfreq * (size_t)npe,
                                    sizeof(double));
    if (kernel_table == 0) return -1;

    memset(&table_in, 0, sizeof(table_in));
    table_in.nfreq = nfreq;
    table_in.nx_tab = synch->nx_tab;
    table_in.ntheta_pitch = synch->ntheta_pitch;
    table_in.nlogb = TRACER_FP_SYNCH_LOGB_BINS;
    table_in.xmin = synch->xmin;
    table_in.logb_min = logb_min;
    table_in.inv_dlogb = 1.0 / dlogb;
    table_in.fx_tab = synch->fx_tab;
    table_in.logfx_tab = synch->logfx_tab;
    table_in.logx_tab = synch->logx_tab;
    table_in.logy = synch->logy_planes;
    table_in.pitch_weight = synch->pitch_weight;

    if (prepare_synch_pitch_kernel_table(&table_in, kernel_table) != 0) {
        free(kernel_table);
        return -1;
    }

    free(*kernel_table_io);
    *kernel_table_io = kernel_table;
    synch->table_logb_min = logb_min;
    synch->table_logb_max = logb_max;
    return 0;
}

int tracer_fp_prepare_ic_cooling_snapshot(int ntracer,
                                          int allow_snapshot_interp,
                                          double z_curr,
                                          double z_next,
                                          const CRspectrum *cre_grid,
                                          double *rad_ic_zero,
                                          double *rad_ic_m1_zero,
                                          double *rad_ic_p1_zero,
                                          double *rad_ic_row_next,
                                          double *rad_ic_m1_next_val,
                                          double *rad_ic_p1_next_val)
{
    double rad_ic_row[npe];
    double rad_ic_m1_val = 0.0;
    double rad_ic_p1_val = 0.0;
    int itr;

    if (ntracer <= 0 || cre_grid == 0 || rad_ic_zero == 0 || rad_ic_m1_zero == 0 ||
        rad_ic_p1_zero == 0 || rad_ic_row_next == 0 ||
        rad_ic_m1_next_val == 0 || rad_ic_p1_next_val == 0) {
        return -1;
    }

    if (prepare_ic_cooling_row(z_curr, cre_grid,
                               rad_ic_row,
                               &rad_ic_m1_val,
                               &rad_ic_p1_val) != 0) {
        return -1;
    }
    if (allow_snapshot_interp) {
        if (prepare_ic_cooling_row(z_next, cre_grid,
                                   rad_ic_row_next,
                                   rad_ic_m1_next_val,
                                   rad_ic_p1_next_val) != 0) {
            return -1;
        }
    } else {
        memcpy(rad_ic_row_next, rad_ic_row, (size_t)npe * sizeof(double));
        *rad_ic_m1_next_val = rad_ic_m1_val;
        *rad_ic_p1_next_val = rad_ic_p1_val;
    }

    for (itr = 0; itr < ntracer; itr++) {
        int je;
        for (je = 0; je < npe; je++) {
            rad_ic_zero[(size_t)je * (size_t)ntracer + (size_t)itr] = rad_ic_row[je];
        }
        rad_ic_m1_zero[itr] = rad_ic_m1_val;
        rad_ic_p1_zero[itr] = rad_ic_p1_val;
    }

    return 0;
}
