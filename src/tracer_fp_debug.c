/*
    K. Nishiwaki, 2026-06-18
    - debug utilities for tracer_fp 
    - "synthetic" functions for artificially generating mock tracer data.
*/


#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "params.h"
#include "tracer_fp_background.h"
#include "tracer_fp_debug.h"

static const double kTracerMpc = 3.0857e24;
static const double kTracerKpc = 3.0857e21;

static double lerp_linear(double x0, double x1, double t)
{
    return x0 + (x1 - x0) * t;
}

static double lerp_log10(double x0, double x1, double t)
{
    return pow(10.0, lerp_linear(log10(x0), log10(x1), t));
}

void tracer_fp_fill_timeline_synthetic(double *dt_snap,
                                       double *z_snap,
                                       int nsnap)
{
    int isnap;

    if (dt_snap == 0 || z_snap == 0 || nsnap <= 0) return;
    for (isnap = 0; isnap < nsnap; isnap++) {
        const double ts = (nsnap > 1) ? (double)isnap / (double)(nsnap - 1) : 0.0;
        dt_snap[isnap] = lerp_linear(0.012, 0.045,
                                               0.5 * ts + 0.25 * sin(2.0 * M_PI * ts) + 0.25);
        z_snap[isnap] = lerp_linear(0.45, 0.02, ts);
    }
}

int tracer_fp_fill_background_snapshot_synthetic(int isnap,
                                                 const double *z_snap,
                                                 int ntracer,
                                                 const long int *tracer_ids,
                                                 int ntracer_global,
                                                 int nsnap,
                                                 TracerFpBackgroundSlot *slot)
{
    const double target_l_turb_cm = 150.0 * kTracerKpc;
    const double ts = (nsnap > 1) ? (double)isnap / (double)(nsnap - 1) : 0.0;
    TracerDataHistory history_view;
    int itr;

    if (z_snap == 0 || slot == 0 || ntracer <= 0 || tracer_ids == 0) return -1;

    history_view.n_gas = slot->n_gas;
    history_view.kbt = slot->kbt;
    history_view.b_field = slot->b_field;
    history_view.divv = slot->divv;
    history_view.l_turb = slot->l_turb;
    history_view.dv_imc = slot->dv_imc;
    history_view.cs = slot->cs;
    history_view.beta_pl = slot->beta_pl;

    for (itr = 0; itr < ntracer; itr++) {
        const long int global_tracer = tracer_ids[itr];
        const double tr = (ntracer_global > 1)
            ? (double)global_tracer / (double)(ntracer_global - 1) : 0.0;
        const double mix = 0.55 * tr + 0.45 * ts;
        const double wobble = 0.5 + 0.5 * sin(2.0 * M_PI * (0.17 * (double)global_tracer + ts));
        const double n_gas = lerp_log10(3.0e-5, 3.0e-3, fmin(1.0, fmax(0.0, mix)));
        const double density = n_gas * (1.6e-24 * mu_mol);
        const double temp_K = lerp_linear(2.5e7, 2.1e8, 0.7 * tr + 0.3 * wobble);
        const double b_sim_G = lerp_log10(0.4e-6, 18.0e-6, 0.4 * tr + 0.6 * wobble);
        const double divv = 0.08 * sin(2.0 * M_PI * ts) * (0.3 + tr);
        const double l_turb_cm =
            lerp_log10(3.0e-2, 4.0e-1, 0.6 * tr + 0.4 * ts) * kTracerMpc;
        const double dv_target = lerp_linear(90.0e5, 1.2e8, 0.65 * wobble + 0.35 * tr);
        const double curl_v = dv_target / l_turb_cm;
        const FpBackgroundCellInput bg_in = {
            .density_gcc = density,
            .temp_K = temp_K,
            .bx_G = b_sim_G,
            .by_G = 0.0,
            .bz_G = 0.0,
            .divv_gyr = divv,
            .curl_v_s = curl_v,
            .curl_v_prev_s = curl_v,
            .curl_v_next_s = curl_v,
            .z_prev = z_snap[isnap],
            .z_curr = z_snap[isnap],
            .z_next = z_snap[isnap],
            .l_turb_cm = l_turb_cm,
            .target_l_turb_cm = target_l_turb_cm,
            .apply_temp_floor = (isnap > 0),
            .allow_curl_interp = 0
        };
        FpBackgroundCellOutput bg_out;

        if (prepare_background_cell(&bg_in, &bg_out) != 0) {
            tracer_zero_background_history_slot((size_t)itr, &history_view);
            continue;
        }

        tracer_store_background_history_slot((size_t)itr, &bg_out, &history_view);
    }
    return 0;
}

long tracer_fp_debug_target_global_id(void)
{
    static int initialized = 0;
    static long target_id = -1;

    if (!initialized) {
        const char *env = getenv("TRACER_FP_DEBUG_TRACER_ID");
        initialized = 1;
        if (env != 0 && *env != '\0') {
            char *endptr = 0;
            long parsed = strtol(env, &endptr, 10);
            if (endptr != env) target_id = parsed;
        }
    }
    return target_id;
}

int tracer_fp_debug_dump_full_bucket(void)
{
    static int initialized = 0;
    static int enabled = 0;

    if (!initialized) {
        const char *env = getenv("TRACER_FP_DEBUG_FULL_BUCKET");
        initialized = 1;
        if (env != 0 && *env != '\0' && strcmp(env, "0") != 0) enabled = 1;
    }
    return enabled;
}

void tracer_fp_debug_print_cre_sample(const char *tag,
                                      long int global_tracer,
                                      int snapshot_1based,
                                      const double *cre_row)
{
    static const int probe_bins[] = {0, 10, 30, 50, 80, 100, 110, 120, npe - 1};
    const int nprobe = (int)(sizeof(probe_bins) / sizeof(probe_bins[0]));
    int i;

    if (tag == 0 || cre_row == 0) return;

    fprintf(stderr, "[debug tracer %ld] %s snapshot=%d", global_tracer, tag, snapshot_1based);
    for (i = 0; i < nprobe; i++) {
        const int bin = probe_bins[i];
        if (bin >= 0 && bin < npe) {
            fprintf(stderr, " b%d=%e", bin, cre_row[bin]);
        }
    }
    fprintf(stderr, "\n");
}

int tracer_fp_debug_report_row_change(const char *tag,
                                      long int global_tracer,
                                      int snapshot_1based,
                                      int bucket_1based,
                                      const double *before_row,
                                      const double *after_row)
{
    static const int probe_bins[] = {0, 10, 30, 50, 80, 90, 100, 110, 120, npe - 1};
    const int nprobe = (int)(sizeof(probe_bins) / sizeof(probe_bins[0]));
    int changed = 0;
    int i;

    if (tag == 0 || before_row == 0 || after_row == 0) return 0;
    if (memcmp(before_row, after_row, (size_t)npe * sizeof(double)) == 0) return 0;

    fprintf(stderr,
            "[debug tracer %ld] %s snapshot=%d bucket=%d CRE row changed unexpectedly\n",
            global_tracer, tag, snapshot_1based, bucket_1based);
    for (i = 0; i < nprobe; i++) {
        const int bin = probe_bins[i];
        if (bin >= 0 && bin < npe && before_row[bin] != after_row[bin]) {
            fprintf(stderr,
                    "[debug tracer %ld]   bin=%d before=%e after=%e rel=%e\n",
                    global_tracer, bin, before_row[bin], after_row[bin],
                    (before_row[bin] != 0.0)
                        ? fabs((after_row[bin] - before_row[bin]) / before_row[bin]) : 0.0);
            changed = 1;
        }
    }
    if (!changed) {
        for (i = 0; i < npe; i++) {
            if (before_row[i] != after_row[i]) {
                fprintf(stderr,
                        "[debug tracer %ld]   first_diff bin=%d before=%e after=%e rel=%e\n",
                        global_tracer, i, before_row[i], after_row[i],
                        (before_row[i] != 0.0)
                            ? fabs((after_row[i] - before_row[i]) / before_row[i]) : 0.0);
                break;
            }
        }
    }
    return 1;
}
