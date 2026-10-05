/*
    tracer_fp_batch.c
    
    K. Nishiwaki, 2026-06-25
    - Tracer CPU batch orchestration 
    - pack per-bucket state/background into workspaces
    - interpolate snapshot backgrounds for coefficient segments
    - execute bucket solves with optional outer OpenMP strategy
*/

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "FP_Coef.h"
#include "tracer_fp_batch.h"
#include "tracer_fp_coef.h"
#include "tracer_fp_setup.h"

#define TRACER_FP_CPU_OMP_MIN_PACK 512
#define TRACER_FP_CPU_OMP_MIN_INTERP 4096

typedef struct {
    int tracer;
    int thread;
    long long weight;
} TracerBatchTask;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1.0e3 * (double)ts.tv_sec + 1.0e-6 * (double)ts.tv_nsec;
}

static int cpu_direct_tacc_on(double requested_t_acc_gyr)
{
    return resolve_momentumdiff_model(FP_MOMENTUMDIFF_MODEL_AUTO) ==
               FP_MOMENTUMDIFF_MODEL_DIRECT_TACC &&
           isfinite(requested_t_acc_gyr) && requested_t_acc_gyr > 0.0 &&
           isfinite(t_off_gyr) && t_off_gyr > 0.0 &&
           isfinite(t_acc_off_gyr) && t_acc_off_gyr > 0.0;
}

static double cpu_direct_tacc_value(double time_gyr,
                                                    double requested_t_acc_gyr)
{
    if (cpu_direct_tacc_on(requested_t_acc_gyr) &&
        time_gyr < t_off_gyr) {
        return t_acc_off_gyr;
    }
    return requested_t_acc_gyr;
}

static void split_range(int total_fp_steps,
                                        int nsegment,
                                        int isegment,
                                        int *fp_begin_out,
                                        int *fp_end_out)
{
    int fp_begin = 0;
    int fp_end = 0;

    if (nsegment > 0 && isegment >= 0) {
        fp_begin = (int)(((long long)isegment * (long long)total_fp_steps) / (long long)nsegment);
        fp_end = (int)(((long long)(isegment + 1) * (long long)total_fp_steps) / (long long)nsegment);
    }

    if (fp_begin_out) *fp_begin_out = fp_begin;
    if (fp_end_out) *fp_end_out = fp_end;
}

static int should_parallel_loop(int n)
{
#ifdef _OPENMP
    if (omp_in_parallel()) return 0;
#endif
    return n > 0;
}

static void interp_array(int n,
                                         const double *curr,
                                         const double *next,
                                         double alpha,
                                         double *out)
{
    int i;

    if (n <= 0 || curr == 0 || out == 0) return;
    if (next == 0 || alpha <= 0.0) {
        memcpy(out, curr, (size_t)n * sizeof(double));
        return;
    }
    if (alpha >= 1.0) {
        memcpy(out, next, (size_t)n * sizeof(double));
        return;
    }

    #pragma omp parallel for schedule(static) if(should_parallel_loop(n) && n >= TRACER_FP_CPU_OMP_MIN_INTERP)
    for (i = 0; i < n; i++) {
        out[i] = curr[i] + alpha * (next[i] - curr[i]);
    }
}

static long long task_weight(int count,
                                             int nstep,
                                             int base_nstep,
                                             int coeff_interp_segments,
                                             TracerFpIntegrationMode integration_mode)
{
    const int pipeline_nstep =
        (integration_mode == TRACER_FP_INTEGRATION_MULTIRATE && base_nstep > 0)
        ? base_nstep : nstep;
    return (long long)count *
           (long long)((pipeline_nstep > 0) ? pipeline_nstep : 1) *
           (long long)((coeff_interp_segments > 0) ? coeff_interp_segments : 1);
}

static int use_outer_omp(int bucket_count)
{
    const int nthr = tracer_fp_cpu_threads();
#ifdef _OPENMP
    if (omp_in_parallel()) return 0;
#endif
    return (nthr > 1 && bucket_count >= nthr);
}

static void accum_times(TracerFpGpuTimes *dst,
                                        const TracerFpGpuTimes *src)
{
    if (dst == 0 || src == 0) return;
    dst->loss_prepass_ms += src->loss_prepass_ms;
    dst->nsub_estimate_ms += src->nsub_estimate_ms;
    dst->bucket_build_ms += src->bucket_build_ms;
    dst->gpu_group_build_ms += src->gpu_group_build_ms;
    dst->pack_ms += src->pack_ms;
    dst->bucket_host_ms += src->bucket_host_ms;
    dst->interp_ms += src->interp_ms;
    dst->snapshot_prep_ms += src->snapshot_prep_ms;
    dst->coeff_ms += src->coeff_ms;
    dst->secondary_ms += src->secondary_ms;
    dst->solve_ms += src->solve_ms;
    dst->solve_alloc_ms += src->solve_alloc_ms;
    dst->solve_rhs_ms += src->solve_rhs_ms;
    dst->solve_tridiag_ms += src->solve_tridiag_ms;
    dst->synch_table_ms += src->synch_table_ms;
    dst->synch_ms += src->synch_ms;
    dst->ic_ms += src->ic_ms;
    dst->gamma_ms += src->gamma_ms;
    dst->neutrino_ms += src->neutrino_ms;
    dst->cuda_setup_ms += src->cuda_setup_ms;
    dst->cuda_h2d_ms += src->cuda_h2d_ms;
    dst->cuda_d2h_ms += src->cuda_d2h_ms;
    dst->cuda_other_ms += src->cuda_other_ms;
    dst->cuda_total_ms += src->cuda_total_ms;
    dst->input_read_ms += src->input_read_ms;
    dst->bg_prepare_ms += src->bg_prepare_ms;
    dst->tracer_mass_ms += src->tracer_mass_ms;
    dst->output_write_ms += src->output_write_ms;
    dst->output_sync_ms += src->output_sync_ms;
    dst->checkpoint_ms += src->checkpoint_ms;
    dst->restart_ms += src->restart_ms;
    dst->total_ms += src->total_ms;
    dst->input_read_calls += src->input_read_calls;
    dst->input_selected_runs += src->input_selected_runs;
    dst->output_write_calls += src->output_write_calls;
    dst->output_sync_calls += src->output_sync_calls;
    dst->checkpoint_calls += src->checkpoint_calls;
    dst->gpu_pipeline_calls += src->gpu_pipeline_calls;
    dst->gpu_pipeline_cells += src->gpu_pipeline_cells;
    dst->gpu_pipeline_fp_steps += src->gpu_pipeline_fp_steps;
    dst->gpu_pipeline_cell_steps += src->gpu_pipeline_cell_steps;
}

int tracer_fp_batch_pack_bucket(int bucket_count,
                                const int *bucket_indices,
                                int coeff_interp_segments,
                                int ntracer_stride,
                                const double *n_gas_snap,
                                const double *kbt_snap,
                                const double *b_field_snap,
                                const double *divv_snap,
                                const double *lturb_snap,
                                const double *dv_snap,
                                const double *cs_snap,
                                const double *beta_snap,
                                const double *rad_ic_zero,
                                const double *rad_ic_m1_zero,
                                const double *rad_ic_p1_zero,
                                const double *n_gas_next_snap,
                                const double *kbt_next_snap,
                                const double *b_field_next_snap,
                                const double *divv_next_snap,
                                const double *lturb_next_snap,
                                const double *dv_next_snap,
                                const double *cs_next_snap,
                                const double *beta_next_snap,
                                const double *rad_ic_row_next,
                                double rad_ic_m1_next_val,
                                double rad_ic_p1_next_val,
                                const double *qpi_batch,
                                const double *qepri_batch,
                                const double *tracer_mass,
                                const unsigned char *disable_adiabatic,
                                const double *crp_state,
                                const double *cre_state,
                                const double *b_dyn,
                                const double *logb,
                                TracerFpCpuWs *cpu_ws,
                                TracerFpGpuHostWs *gpu_ws,
                                TracerFpGpuTimes *times)
{
    double t0_pack;
    int idx;

    if (bucket_count <= 0 || bucket_indices == 0 || ntracer_stride <= 0 ||
        n_gas_snap == 0 || kbt_snap == 0 || b_field_snap == 0 || divv_snap == 0 ||
        lturb_snap == 0 || dv_snap == 0 || cs_snap == 0 || beta_snap == 0 ||
        rad_ic_zero == 0 || rad_ic_m1_zero == 0 || rad_ic_p1_zero == 0 ||
        tracer_mass == 0 || crp_state == 0 || cre_state == 0 || cpu_ws == 0 || times == 0) {
        return -1;
    }
    if (qpi_batch == 0 || qepri_batch == 0) {
        return -1;
    }
    if (coeff_interp_segments > 1 &&
        (n_gas_next_snap == 0 || kbt_next_snap == 0 || b_field_next_snap == 0 ||
         divv_next_snap == 0 || lturb_next_snap == 0 || dv_next_snap == 0 ||
         cs_next_snap == 0 || beta_next_snap == 0 || rad_ic_row_next == 0)) {
        return -1;
    }
    if ((b_dyn != 0 || logb != 0 || gpu_ws != 0) &&
        (b_dyn == 0 || logb == 0 || gpu_ws == 0)) {
        return -1;
    }
    if (tracer_fp_cpu_ws_ensure(cpu_ws, bucket_count) != 0) {
        return -1;
    }
    if (gpu_ws != 0 && tracer_fp_gpu_host_ws_ensure(gpu_ws, bucket_count) != 0) {
        return -1;
    }

    t0_pack = now_ms();
    #pragma omp parallel for schedule(static) if(should_parallel_loop(bucket_count) && bucket_count >= TRACER_FP_CPU_OMP_MIN_PACK)
    for (idx = 0; idx < bucket_count; idx++) {
        const int src = bucket_indices[idx];
        const size_t src_p_off = (size_t)src * (size_t)np;
        const size_t src_e_off = (size_t)src * (size_t)npe;
        const size_t dst_p_off = (size_t)idx * (size_t)np;
        const size_t dst_e_off = (size_t)idx * (size_t)npe;
        int je;

        cpu_ws->bg_curr.n_gas[idx] = n_gas_snap[src];
        cpu_ws->bg_curr.kbt[idx] = kbt_snap[src];
        cpu_ws->bg_curr.b_field[idx] = b_field_snap[src];
        cpu_ws->bg_curr.divv[idx] = divv_snap[src];
        cpu_ws->bg_curr.lturb[idx] = lturb_snap[src];
        cpu_ws->bg_curr.dv[idx] = dv_snap[src];
        cpu_ws->bg_curr.cs[idx] = cs_snap[src];
        cpu_ws->bg_curr.beta[idx] = beta_snap[src];
        for (je = 0; je < npe; je++) {
            cpu_ws->bg_curr.rad_ic[(size_t)je * (size_t)bucket_count + (size_t)idx] =
                rad_ic_zero[(size_t)je * (size_t)ntracer_stride + (size_t)src];
        }
        cpu_ws->bg_curr.rad_ic_m1[idx] = rad_ic_m1_zero[src];
        cpu_ws->bg_curr.rad_ic_p1[idx] = rad_ic_p1_zero[src];

        memcpy(cpu_ws->state.qpi + dst_p_off, qpi_batch + src_p_off, (size_t)np * sizeof(double));
        memcpy(cpu_ws->state.qepri + dst_e_off, qepri_batch + src_e_off, (size_t)npe * sizeof(double));
        memcpy(cpu_ws->state.crp + dst_p_off, crp_state + src_p_off, (size_t)np * sizeof(double));
        memcpy(cpu_ws->state.cre + dst_e_off, cre_state + src_e_off, (size_t)npe * sizeof(double));
        cpu_ws->state.mass_msun[idx] = tracer_mass[src];
        cpu_ws->state.disable_adiabatic[idx] =
            (disable_adiabatic != 0) ? disable_adiabatic[src] : 0;

        if (gpu_ws != 0) {
            gpu_ws->bucket.b_dyn[idx] = b_dyn[src];
            gpu_ws->bucket.logb[idx] = logb[src];
        }

        if (coeff_interp_segments > 1) {
            cpu_ws->bg_next.n_gas[idx] = n_gas_next_snap[src];
            cpu_ws->bg_next.kbt[idx] = kbt_next_snap[src];
            cpu_ws->bg_next.b_field[idx] = b_field_next_snap[src];
            cpu_ws->bg_next.divv[idx] = divv_next_snap[src];
            cpu_ws->bg_next.lturb[idx] = lturb_next_snap[src];
            cpu_ws->bg_next.dv[idx] = dv_next_snap[src];
            cpu_ws->bg_next.cs[idx] = cs_next_snap[src];
            cpu_ws->bg_next.beta[idx] = beta_next_snap[src];
            for (je = 0; je < npe; je++) {
                cpu_ws->bg_next.rad_ic[(size_t)je * (size_t)bucket_count + (size_t)idx] =
                    rad_ic_row_next[je];
            }
            cpu_ws->bg_next.rad_ic_m1[idx] = rad_ic_m1_next_val;
            cpu_ws->bg_next.rad_ic_p1[idx] = rad_ic_p1_next_val;
        }
    }
    times->pack_ms += now_ms() - t0_pack;
    return 0;
}

void tracer_fp_batch_scatter_bucket_state(int bucket_count,
                                          const int *bucket_indices,
                                          double *crp_state,
                                          double *cre_state,
                                          const TracerFpCpuWs *cpu_ws,
                                          TracerFpGpuTimes *times)
{
    double t0_pack;
    int idx;

    if (bucket_count <= 0 || bucket_indices == 0 || crp_state == 0 ||
        cre_state == 0 || cpu_ws == 0 || times == 0) {
        return;
    }

    t0_pack = now_ms();
    #pragma omp parallel for schedule(static) if(should_parallel_loop(bucket_count) && bucket_count >= TRACER_FP_CPU_OMP_MIN_PACK)
    for (idx = 0; idx < bucket_count; idx++) {
        const int dst = bucket_indices[idx];
        const size_t dst_p_off = (size_t)dst * (size_t)np;
        const size_t dst_e_off = (size_t)dst * (size_t)npe;
        const size_t src_p_off = (size_t)idx * (size_t)np;
        const size_t src_e_off = (size_t)idx * (size_t)npe;

        memcpy(crp_state + dst_p_off, cpu_ws->state.crp + src_p_off, (size_t)np * sizeof(double));
        memcpy(cre_state + dst_e_off, cpu_ws->state.cre + src_e_off, (size_t)npe * sizeof(double));
    }
    times->pack_ms += now_ms() - t0_pack;
}

int tracer_fp_batch_pack_group_current(int group_bucket_count,
                                       const int *group_bucket_ids,
                                       const TracerBucket *bucket,
                                       int ntracer_stride,
                                       const double *n_gas_snap,
                                       const double *kbt_snap,
                                       const double *b_field_snap,
                                       const double *divv_snap,
                                       const double *lturb_snap,
                                       const double *dv_snap,
                                       const double *cs_snap,
                                       const double *beta_snap,
                                       const double *rad_ic_zero,
                                       const double *rad_ic_m1_zero,
                                       const double *rad_ic_p1_zero,
                                       const double *n_gas_next_snap,
                                       const double *kbt_next_snap,
                                       const double *b_field_next_snap,
                                       const double *divv_next_snap,
                                       const double *lturb_next_snap,
                                       const double *dv_next_snap,
                                       const double *cs_next_snap,
                                       const double *beta_next_snap,
                                       const double *rad_ic_row_next,
                                       double rad_ic_m1_next_val,
                                       double rad_ic_p1_next_val,
                                       const double *qpi_batch,
                                       const double *qepri_batch,
                                       const double *tracer_mass,
                                       const unsigned char *disable_adiabatic,
                                       const double *crp_state,
                                       const double *cre_state,
                                       const double *b_dyn,
                                       const double *logb,
                                       int *cell_offsets_out,
                                       int *rad_ic_offsets_out,
                                       TracerFpCpuWs *cpu_ws,
                                       TracerFpGpuHostWs *gpu_ws,
                                       TracerFpGpuTimes *times)
{
    double t0_pack;
    int ibucket_local;
    int total_cells = 0;
    int cell_offset = 0;
    int rad_ic_offset = 0;

    if (group_bucket_count <= 0 || group_bucket_ids == 0 || bucket == 0 ||
        ntracer_stride <= 0 || n_gas_snap == 0 || kbt_snap == 0 ||
        b_field_snap == 0 || divv_snap == 0 || lturb_snap == 0 ||
        dv_snap == 0 || cs_snap == 0 || beta_snap == 0 ||
        rad_ic_zero == 0 || rad_ic_m1_zero == 0 || rad_ic_p1_zero == 0 ||
        tracer_mass == 0 || crp_state == 0 || cre_state == 0 || cell_offsets_out == 0 ||
        rad_ic_offsets_out == 0 || cpu_ws == 0 || times == 0) {
        return -1;
    }
    if (qpi_batch == 0 || qepri_batch == 0) {
        return -1;
    }
    if ((b_dyn != 0 || logb != 0 || gpu_ws != 0) &&
        (b_dyn == 0 || logb == 0 || gpu_ws == 0)) {
        return -1;
    }

    for (ibucket_local = 0; ibucket_local < group_bucket_count; ibucket_local++) {
        const int bucket_id = group_bucket_ids[ibucket_local];
        const int bucket_count = bucket->member_counts[bucket_id];
        if (bucket_count <= 0) return -1;
        total_cells += bucket_count;
    }

    if (tracer_fp_cpu_ws_ensure(cpu_ws, total_cells) != 0 ||
        (gpu_ws != 0 && tracer_fp_gpu_host_ws_ensure(gpu_ws, total_cells) != 0)) {
        return -1;
    }

    t0_pack = now_ms();
    for (ibucket_local = 0; ibucket_local < group_bucket_count; ibucket_local++) {
        const int bucket_id = group_bucket_ids[ibucket_local];
        const int bucket_count = bucket->member_counts[bucket_id];
        const int bucket_offset = bucket->member_offsets[bucket_id];
        const int *bucket_indices = bucket->indices + (size_t)bucket_offset;
        int idx;

        cell_offsets_out[ibucket_local] = cell_offset;
        rad_ic_offsets_out[ibucket_local] = rad_ic_offset;

        #pragma omp parallel for schedule(static) if(should_parallel_loop(bucket_count) && bucket_count >= TRACER_FP_CPU_OMP_MIN_PACK)
        for (idx = 0; idx < bucket_count; idx++) {
            const int src = bucket_indices[idx];
            const int dst_cell = cell_offset + idx;
            const size_t src_p_off = (size_t)src * (size_t)np;
            const size_t src_e_off = (size_t)src * (size_t)npe;
            const size_t dst_p_off = (size_t)dst_cell * (size_t)np;
            const size_t dst_e_off = (size_t)dst_cell * (size_t)npe;
            int je;

            cpu_ws->bg_curr.n_gas[dst_cell] = n_gas_snap[src];
            cpu_ws->bg_curr.kbt[dst_cell] = kbt_snap[src];
            cpu_ws->bg_curr.b_field[dst_cell] = b_field_snap[src];
            cpu_ws->bg_curr.divv[dst_cell] = divv_snap[src];
            cpu_ws->bg_curr.lturb[dst_cell] = lturb_snap[src];
            cpu_ws->bg_curr.dv[dst_cell] = dv_snap[src];
            cpu_ws->bg_curr.cs[dst_cell] = cs_snap[src];
            cpu_ws->bg_curr.beta[dst_cell] = beta_snap[src];
            for (je = 0; je < npe; je++) {
                cpu_ws->bg_curr.rad_ic[rad_ic_offset + (size_t)je * (size_t)bucket_count + (size_t)idx] =
                    rad_ic_zero[(size_t)je * (size_t)ntracer_stride + (size_t)src];
            }
            cpu_ws->bg_curr.rad_ic_m1[dst_cell] = rad_ic_m1_zero[src];
            cpu_ws->bg_curr.rad_ic_p1[dst_cell] = rad_ic_p1_zero[src];
            if (n_gas_next_snap != 0 && kbt_next_snap != 0 && b_field_next_snap != 0 &&
                divv_next_snap != 0 && lturb_next_snap != 0 && dv_next_snap != 0 &&
                cs_next_snap != 0 && beta_next_snap != 0 && rad_ic_row_next != 0) {
                cpu_ws->bg_next.n_gas[dst_cell] = n_gas_next_snap[src];
                cpu_ws->bg_next.kbt[dst_cell] = kbt_next_snap[src];
                cpu_ws->bg_next.b_field[dst_cell] = b_field_next_snap[src];
                cpu_ws->bg_next.divv[dst_cell] = divv_next_snap[src];
                cpu_ws->bg_next.lturb[dst_cell] = lturb_next_snap[src];
                cpu_ws->bg_next.dv[dst_cell] = dv_next_snap[src];
                cpu_ws->bg_next.cs[dst_cell] = cs_next_snap[src];
                cpu_ws->bg_next.beta[dst_cell] = beta_next_snap[src];
                for (je = 0; je < npe; je++) {
                    cpu_ws->bg_next.rad_ic[rad_ic_offset + (size_t)je * (size_t)bucket_count + (size_t)idx] =
                        rad_ic_row_next[je];
                }
                cpu_ws->bg_next.rad_ic_m1[dst_cell] = rad_ic_m1_next_val;
                cpu_ws->bg_next.rad_ic_p1[dst_cell] = rad_ic_p1_next_val;
            }

            memcpy(cpu_ws->state.qpi + dst_p_off, qpi_batch + src_p_off, (size_t)np * sizeof(double));
            memcpy(cpu_ws->state.qepri + dst_e_off, qepri_batch + src_e_off, (size_t)npe * sizeof(double));
            memcpy(cpu_ws->state.crp + dst_p_off, crp_state + src_p_off, (size_t)np * sizeof(double));
            memcpy(cpu_ws->state.cre + dst_e_off, cre_state + src_e_off, (size_t)npe * sizeof(double));
            cpu_ws->state.mass_msun[dst_cell] = tracer_mass[src];
            cpu_ws->state.disable_adiabatic[dst_cell] =
                (disable_adiabatic != 0) ? disable_adiabatic[src] : 0;
            if (gpu_ws != 0) {
                gpu_ws->bucket.b_dyn[dst_cell] = b_dyn[src];
                gpu_ws->bucket.logb[dst_cell] = logb[src];
            }
        }

        cell_offset += bucket_count;
        rad_ic_offset += bucket_count * npe;
    }

    times->pack_ms += now_ms() - t0_pack;
    return 0;
}

void tracer_fp_batch_scatter_group_state(int group_bucket_count,
                                         const int *group_bucket_ids,
                                         const TracerBucket *bucket,
                                         const int *cell_offsets,
                                         double *crp_state,
                                         double *cre_state,
                                         const TracerFpCpuWs *cpu_ws,
                                         TracerFpGpuTimes *times)
{
    double t0_pack;
    int ibucket_local;

    if (group_bucket_count <= 0 || group_bucket_ids == 0 || bucket == 0 ||
        cell_offsets == 0 || crp_state == 0 || cre_state == 0 ||
        cpu_ws == 0 || times == 0) {
        return;
    }

    t0_pack = now_ms();
    for (ibucket_local = 0; ibucket_local < group_bucket_count; ibucket_local++) {
        const int bucket_id = group_bucket_ids[ibucket_local];
        const int bucket_count = bucket->member_counts[bucket_id];
        const int bucket_offset = bucket->member_offsets[bucket_id];
        const int *bucket_indices = bucket->indices + (size_t)bucket_offset;
        const int src_cell_offset = cell_offsets[ibucket_local];
        int idx;

        #pragma omp parallel for schedule(static) if(should_parallel_loop(bucket_count) && bucket_count >= TRACER_FP_CPU_OMP_MIN_PACK)
        for (idx = 0; idx < bucket_count; idx++) {
            const int dst = bucket_indices[idx];
            const int src_cell = src_cell_offset + idx;
            const size_t dst_p_off = (size_t)dst * (size_t)np;
            const size_t dst_e_off = (size_t)dst * (size_t)npe;
            const size_t src_p_off = (size_t)src_cell * (size_t)np;
            const size_t src_e_off = (size_t)src_cell * (size_t)npe;

            memcpy(crp_state + dst_p_off, cpu_ws->state.crp + src_p_off, (size_t)np * sizeof(double));
            memcpy(cre_state + dst_e_off, cpu_ws->state.cre + src_e_off, (size_t)npe * sizeof(double));
        }
    }
    times->pack_ms += now_ms() - t0_pack;
}

static int run_bucket_outer_omp(int bucket_count,
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
                                                TracerFpGpuTimes *times)
{
    const int enable_outer_omp = use_outer_omp(bucket_count);
    int ierr_any = 0;

    if (bucket_count <= 0 || crp_grid == 0 || cre_grid == 0 || n_gas == 0 || kbt == 0 ||
        b_field == 0 || divv_gyr == 0 || l_turb_mpc == 0 || dv_imc == 0 || cs == 0 ||
        beta_pl == 0 || rad_ic_batch == 0 || rad_ic_m1 == 0 || rad_ic_p1 == 0 ||
        qpi_batch == 0 || qepri_batch == 0 || fqe_flat == 0 || np_min_qe == 0 ||
        tracer_mass == 0 || crp_state == 0 || cre_state == 0 || times == 0) {
        return -1;
    }

    if (!enable_outer_omp) {
        TracerFpCpuWs serial_ws;
        memset(&serial_ws, 0, sizeof(serial_ws));
        ierr_any = tracer_fp_cpu_evolve(bucket_count, nstep, dt,
                                        use_windowed_reacc, on_start_step, on_end_step,
                                        crp_grid, cre_grid,
                                        n_gas, kbt, b_field, divv_gyr,
                                        l_turb_mpc, dv_imc, cs, beta_pl,
                                        rad_ic_batch, rad_ic_m1, rad_ic_p1,
                                        qpi_batch, qepri_batch, fqe_flat, np_min_qe,
                                        tracer_mass,
                                        disable_adiabatic,
                                        crp_state, cre_state, times, &serial_ws);
        tracer_fp_cpu_ws_free(&serial_ws);
        return ierr_any;
    }

    #pragma omp parallel if(enable_outer_omp)
    {
        TracerFpGpuTimes thread_times;
        TracerFpCpuWs thread_ws;
        int idx;

        memset(&thread_times, 0, sizeof(thread_times));
        memset(&thread_ws, 0, sizeof(thread_ws));

        #pragma omp for schedule(static)
        for (idx = 0; idx < bucket_count; idx++) {
            double rad_ic_local[npe];
            int je;
            for (je = 0; je < npe; je++) {
                rad_ic_local[je] = rad_ic_batch[(size_t)je * (size_t)bucket_count + (size_t)idx];
            }

            if (tracer_fp_cpu_evolve(1, nstep, dt,
                                     use_windowed_reacc, on_start_step, on_end_step,
                                     crp_grid, cre_grid,
                                     n_gas + idx, kbt + idx, b_field + idx, divv_gyr + idx,
                                     l_turb_mpc + idx, dv_imc + idx, cs + idx, beta_pl + idx,
                                     rad_ic_local, rad_ic_m1 + idx, rad_ic_p1 + idx,
                                     qpi_batch + (size_t)idx * (size_t)np,
                                     qepri_batch + (size_t)idx * (size_t)npe,
                                     fqe_flat, np_min_qe,
                                     tracer_mass + idx,
                                     disable_adiabatic != 0 ? disable_adiabatic + idx : 0,
                                     crp_state + (size_t)idx * (size_t)np,
                                     cre_state + (size_t)idx * (size_t)npe,
                                     &thread_times, &thread_ws) != 0) {
                #pragma omp critical
                {
                    ierr_any = -1;
                }
            }
        }

        #pragma omp critical
        {
            accum_times(times, &thread_times);
        }
        tracer_fp_cpu_ws_free(&thread_ws);
    }

    return (ierr_any != 0) ? ierr_any : 0;
}

static int compare_tasks_desc(const void *lhs,
                                              const void *rhs)
{
    const TracerBatchTask *a = (const TracerBatchTask *)lhs;
    const TracerBatchTask *b = (const TracerBatchTask *)rhs;

    if (a->weight < b->weight) return 1;
    if (a->weight > b->weight) return -1;
    if (a->tracer < b->tracer) return -1;
    if (a->tracer > b->tracer) return 1;
    return 0;
}

int tracer_fp_batch_execute_bucket(int bucket_count,
                                   const int *bucket_indices,
                                   int bucket_nstep,
                                   int bucket_n_on,
                                   int bucket_base_nstep,
                                   int bucket_fp_cadence,
                                   TracerFpIntegrationMode integration_mode,
                                   int coeff_interp_segments,
                                   int allow_snapshot_interp,
                                   int ntracer_stride,
                                   double elapsed_gyr,
                                   double dt_snap,
                                   double z_curr,
                                   double z_next,
                                   double requested_t_acc_direct_gyr,
                                   const CRspectrum *crp_grid,
                                   const CRspectrum *cre_grid,
                                   const double *n_gas_snap,
                                   const double *kbt_snap,
                                   const double *b_field_snap,
                                   const double *divv_snap,
                                   const double *lturb_snap,
                                   const double *dv_snap,
                                   const double *cs_snap,
                                   const double *beta_snap,
                                   const double *rad_ic_zero,
                                   const double *rad_ic_m1_zero,
                                   const double *rad_ic_p1_zero,
                                   const double *n_gas_next_snap,
                                   const double *kbt_next_snap,
                                   const double *b_field_next_snap,
                                   const double *divv_next_snap,
                                   const double *lturb_next_snap,
                                   const double *dv_next_snap,
                                   const double *cs_next_snap,
                                   const double *beta_next_snap,
                                   const double *rad_ic_row_next,
                                   double rad_ic_m1_next_val,
                                   double rad_ic_p1_next_val,
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
    const int pipeline_nstep =
        (integration_mode == TRACER_FP_INTEGRATION_MULTIRATE && bucket_base_nstep > 0)
        ? bucket_base_nstep : bucket_nstep;
    const int bucket_on_start = (bucket_nstep - bucket_n_on) / 2;
    const int bucket_on_end = bucket_on_start + bucket_n_on;
    const double base_dt = dt_snap / (double)pipeline_nstep;
    int iseg;

    if (bucket_count <= 0 || bucket_indices == 0 || bucket_nstep <= 0 ||
        pipeline_nstep <= 0 || ntracer_stride <= 0 || ws == 0) {
        return -1;
    }
    (void)bucket_fp_cadence;
    (void)allow_snapshot_interp;
    (void)z_curr;
    (void)z_next;
    if (tracer_fp_batch_pack_bucket(bucket_count, bucket_indices,
                                    coeff_interp_segments, ntracer_stride,
                                    n_gas_snap, kbt_snap, b_field_snap, divv_snap,
                                    lturb_snap, dv_snap, cs_snap, beta_snap,
                                    rad_ic_zero, rad_ic_m1_zero, rad_ic_p1_zero,
                                    n_gas_next_snap, kbt_next_snap, b_field_next_snap,
                                    divv_next_snap, lturb_next_snap, dv_next_snap,
                                    cs_next_snap, beta_next_snap,
                                    rad_ic_row_next, rad_ic_m1_next_val, rad_ic_p1_next_val,
                                    qpi_batch, qepri_batch, tracer_mass,
                                    disable_adiabatic,
                                    crp_state, cre_state,
                                    0, 0,
                                    ws, 0, times) != 0) {
        return -1;
    }

    for (iseg = 0; iseg < coeff_interp_segments; iseg++) {
        const double alpha = (coeff_interp_segments > 1)
            ? ((double)iseg + 0.5) / (double)coeff_interp_segments : 0.0;
        const double *n_gas_coeff_ptr = ws->bg_curr.n_gas;
        const double *kbt_coeff_ptr = ws->bg_curr.kbt;
        const double *b_field_coeff_ptr = ws->bg_curr.b_field;
        const double *divv_coeff_ptr = ws->bg_curr.divv;
        const double *lturb_coeff_ptr = ws->bg_curr.lturb;
        const double *dv_coeff_ptr = ws->bg_curr.dv;
        const double *cs_coeff_ptr = ws->bg_curr.cs;
        const double *beta_coeff_ptr = ws->bg_curr.beta;
        const double *rad_ic_coeff_ptr = ws->bg_curr.rad_ic;
        const double *rad_ic_m1_coeff_ptr = ws->bg_curr.rad_ic_m1;
        const double *rad_ic_p1_coeff_ptr = ws->bg_curr.rad_ic_p1;
        int seg_fp_begin = 0;
        int seg_fp_end = 0;
        int seg_fp_steps;

        split_range(bucket_nstep, coeff_interp_segments, iseg,
                                    &seg_fp_begin, &seg_fp_end);
        seg_fp_steps = seg_fp_end - seg_fp_begin;
        if (seg_fp_steps <= 0) continue;

        if (coeff_interp_segments > 1) {
            const double t0_interp = now_ms();
            interp_array(bucket_count, ws->bg_curr.n_gas, ws->bg_next.n_gas, alpha, ws->bg_interp.n_gas);
            interp_array(bucket_count, ws->bg_curr.kbt, ws->bg_next.kbt, alpha, ws->bg_interp.kbt);
            interp_array(bucket_count, ws->bg_curr.b_field, ws->bg_next.b_field, alpha, ws->bg_interp.b_field);
            interp_array(bucket_count, ws->bg_curr.divv, ws->bg_next.divv, alpha, ws->bg_interp.divv);
            interp_array(bucket_count, ws->bg_curr.lturb, ws->bg_next.lturb, alpha, ws->bg_interp.lturb);
            interp_array(bucket_count, ws->bg_curr.dv, ws->bg_next.dv, alpha, ws->bg_interp.dv);
            interp_array(bucket_count, ws->bg_curr.cs, ws->bg_next.cs, alpha, ws->bg_interp.cs);
            interp_array(bucket_count, ws->bg_curr.beta, ws->bg_next.beta, alpha, ws->bg_interp.beta);
            interp_array(bucket_count * npe, ws->bg_curr.rad_ic, ws->bg_next.rad_ic, alpha, ws->bg_interp.rad_ic);
            interp_array(bucket_count, ws->bg_curr.rad_ic_m1, ws->bg_next.rad_ic_m1, alpha, ws->bg_interp.rad_ic_m1);
            interp_array(bucket_count, ws->bg_curr.rad_ic_p1, ws->bg_next.rad_ic_p1, alpha, ws->bg_interp.rad_ic_p1);
            n_gas_coeff_ptr = ws->bg_interp.n_gas;
            kbt_coeff_ptr = ws->bg_interp.kbt;
            b_field_coeff_ptr = ws->bg_interp.b_field;
            divv_coeff_ptr = ws->bg_interp.divv;
            lturb_coeff_ptr = ws->bg_interp.lturb;
            dv_coeff_ptr = ws->bg_interp.dv;
            cs_coeff_ptr = ws->bg_interp.cs;
            beta_coeff_ptr = ws->bg_interp.beta;
            rad_ic_coeff_ptr = ws->bg_interp.rad_ic;
            rad_ic_m1_coeff_ptr = ws->bg_interp.rad_ic_m1;
            rad_ic_p1_coeff_ptr = ws->bg_interp.rad_ic_p1;
            times->interp_ms += now_ms() - t0_interp;
        }

        {
            const int seg_on_begin = (bucket_on_start > seg_fp_begin) ? bucket_on_start : seg_fp_begin;
            int seg_on_end = (bucket_on_end < seg_fp_end) ? bucket_on_end : seg_fp_end;
            int seg_use_windowed_reacc;
            int seg_on_start_local;
            int seg_on_end_local;
            const double seg_time_mid = elapsed_gyr +
                ((double)seg_fp_begin + 0.5 * (double)seg_fp_steps) * base_dt;

            if (seg_on_end < seg_on_begin) seg_on_end = seg_on_begin;
            seg_use_windowed_reacc = ((seg_on_end - seg_on_begin) < seg_fp_steps) ? 1 : 0;
            seg_on_start_local = seg_on_begin - seg_fp_begin;
            seg_on_end_local = seg_on_start_local + (seg_on_end - seg_on_begin);

            t_acc_direct_gyr =
                cpu_direct_tacc_value(seg_time_mid, requested_t_acc_direct_gyr);
            if (run_bucket_outer_omp(bucket_count, seg_fp_steps,
                                                     dt_snap / (double)bucket_nstep,
                                                     seg_use_windowed_reacc,
                                                     seg_on_start_local, seg_on_end_local,
                                                     crp_grid, cre_grid,
                                                     n_gas_coeff_ptr, kbt_coeff_ptr,
                                                     b_field_coeff_ptr, divv_coeff_ptr,
                                                     lturb_coeff_ptr, dv_coeff_ptr,
                                                     cs_coeff_ptr, beta_coeff_ptr,
                                                     rad_ic_coeff_ptr, rad_ic_m1_coeff_ptr,
                                                     rad_ic_p1_coeff_ptr,
                                                     ws->state.qpi, ws->state.qepri,
                                                     fqe_flat, np_min_qe,
                                                     ws->state.mass_msun,
                                                     ws->state.disable_adiabatic,
                                                     ws->state.crp, ws->state.cre,
                                                     times) != 0) {
                return -1;
            }
        }
        t_acc_direct_gyr = requested_t_acc_direct_gyr;
    }

    tracer_fp_batch_scatter_bucket_state(bucket_count, bucket_indices,
                                         crp_state, cre_state, ws, times);

    return 0;
}

int tracer_fp_batch_execute_parallel(int ntracer,
                                     const int *nsubsteps,
                                     const int *n_onsteps,
                                     const int *base_nsteps,
                                     const int *fp_cadence_steps,
                                     TracerFpIntegrationMode integration_mode,
                                     int allow_snapshot_interp,
                                     double elapsed_gyr,
                                     double dt_snap,
                                     double z_curr,
                                     double z_next,
                                     double requested_t_acc_direct_gyr,
                                     const CRspectrum *crp_grid,
                                     const CRspectrum *cre_grid,
                                     const double *n_gas_snap,
                                     const double *kbt_snap,
                                     const double *b_field_snap,
                                     const double *divv_snap,
                                     const double *lturb_snap,
                                     const double *dv_snap,
                                     const double *cs_snap,
                                     const double *beta_snap,
                                     const double *rad_ic_zero,
                                     const double *rad_ic_m1_zero,
                                     const double *rad_ic_p1_zero,
                                     const double *n_gas_next_snap,
                                     const double *kbt_next_snap,
                                     const double *b_field_next_snap,
                                     const double *divv_next_snap,
                                     const double *lturb_next_snap,
                                     const double *dv_next_snap,
                                     const double *cs_next_snap,
                                     const double *beta_next_snap,
                                     const double *rad_ic_row_next,
                                     double rad_ic_m1_next_val,
                                     double rad_ic_p1_next_val,
                                     const double *qpi_batch,
                                     const double *qepri_batch,
                                     const double *fqe_flat,
                                     const int *np_min_qe,
                                     const double *tracer_mass,
                                     const unsigned char *disable_adiabatic,
                                     double *crp_state,
                                     double *cre_state,
                                     TracerFpGpuTimes *times,
                                     int *bucket_calls_out,
                                     int *min_bucket_size_out,
                                     int *max_bucket_size_out)
{
    TracerBatchTask *tasks = 0;
    long long *thread_loads = 0;
    int *thread_counts = 0;
    int nthreads;
    int itr;
    int ierr_any = 0;
    int bucket_calls_total = 0;
    int min_bucket_size_total = 0;
    int max_bucket_size_total = 0;

    if (ntracer <= 0 || nsubsteps == 0 || n_onsteps == 0 || base_nsteps == 0 ||
        fp_cadence_steps == 0 || crp_grid == 0 || cre_grid == 0 ||
        n_gas_snap == 0 || kbt_snap == 0 || b_field_snap == 0 || divv_snap == 0 ||
        lturb_snap == 0 || dv_snap == 0 || cs_snap == 0 || beta_snap == 0 ||
        rad_ic_zero == 0 || rad_ic_m1_zero == 0 || rad_ic_p1_zero == 0 ||
        n_gas_next_snap == 0 || kbt_next_snap == 0 || b_field_next_snap == 0 ||
        divv_next_snap == 0 || lturb_next_snap == 0 || dv_next_snap == 0 ||
        cs_next_snap == 0 || beta_next_snap == 0 || rad_ic_row_next == 0 ||
        qpi_batch == 0 || qepri_batch == 0 || fqe_flat == 0 || np_min_qe == 0 ||
        tracer_mass == 0 ||
        crp_state == 0 || cre_state == 0 || times == 0) {
        return -1;
    }

    nthreads = tracer_fp_cpu_threads();
    if (nthreads < 1) nthreads = 1;

    tasks = (TracerBatchTask *)calloc((size_t)ntracer, sizeof(*tasks));
    thread_loads = (long long *)calloc((size_t)nthreads, sizeof(*thread_loads));
    thread_counts = (int *)calloc((size_t)nthreads, sizeof(*thread_counts));
    if (tasks == 0 || thread_loads == 0 || thread_counts == 0) {
        free(tasks);
        free(thread_loads);
        free(thread_counts);
        return -1;
    }

    for (itr = 0; itr < ntracer; itr++) {
        const int coeff_interp_segments =
            tracer_fp_choose_coeff_interp_segments(nsubsteps[itr], allow_snapshot_interp);
        tasks[itr].tracer = itr;
        tasks[itr].thread = 0;
        tasks[itr].weight = task_weight(1,
                                                        nsubsteps[itr],
                                                        base_nsteps[itr],
                                                        coeff_interp_segments,
                                                        integration_mode);
    }

    qsort(tasks, (size_t)ntracer, sizeof(*tasks), compare_tasks_desc);
    for (itr = 0; itr < ntracer; itr++) {
        int best_thread = 0;
        int ithread;
        for (ithread = 1; ithread < nthreads; ithread++) {
            if (thread_loads[ithread] < thread_loads[best_thread]) best_thread = ithread;
        }
        tasks[itr].thread = best_thread;
        thread_loads[best_thread] += tasks[itr].weight;
        thread_counts[best_thread]++;
    }

    #pragma omp parallel if(nthreads > 1)
    {
        TracerFpGpuTimes thread_times;
        TracerFpCpuWs thread_ws;
        const int tid =
#ifdef _OPENMP
            omp_get_thread_num();
#else
            0;
#endif
        const int local_count = thread_counts[tid];
        int *local_tracers = 0;
        int *local_bucket_steps = 0;
        int *local_bucket_onsteps = 0;
        int *local_bucket_base_steps = 0;
        int *local_bucket_fp_cadences = 0;
        int *local_tracer_bucket_ids = 0;
        int *local_bucket_member_offsets = 0;
        int *local_bucket_member_counts = 0;
        int *local_bucket_member_write = 0;
        int *local_bucket_indices = 0;
        int local_nbuckets = 0;
        int local_bucket_calls = 0;
        int local_min_bucket_size = 0;
        int local_max_bucket_size = 0;
        int local_error = 0;
        int local_itr = 0;
        int itask;

        memset(&thread_times, 0, sizeof(thread_times));
        memset(&thread_ws, 0, sizeof(thread_ws));

        if (local_count > 0) {
            const double t0_bucket_build = now_ms();
            local_tracers = (int *)calloc((size_t)local_count, sizeof(*local_tracers));
            local_bucket_steps = (int *)calloc((size_t)local_count, sizeof(*local_bucket_steps));
            local_bucket_onsteps = (int *)calloc((size_t)local_count, sizeof(*local_bucket_onsteps));
            local_bucket_base_steps = (int *)calloc((size_t)local_count, sizeof(*local_bucket_base_steps));
            local_bucket_fp_cadences = (int *)calloc((size_t)local_count, sizeof(*local_bucket_fp_cadences));
            local_tracer_bucket_ids = (int *)calloc((size_t)local_count, sizeof(*local_tracer_bucket_ids));
            local_bucket_member_offsets = (int *)calloc((size_t)local_count, sizeof(*local_bucket_member_offsets));
            local_bucket_member_counts = (int *)calloc((size_t)local_count, sizeof(*local_bucket_member_counts));
            local_bucket_member_write = (int *)calloc((size_t)local_count, sizeof(*local_bucket_member_write));
            local_bucket_indices = (int *)calloc((size_t)local_count, sizeof(*local_bucket_indices));

            if (local_tracers == 0 || local_bucket_steps == 0 || local_bucket_onsteps == 0 ||
                local_bucket_base_steps == 0 || local_bucket_fp_cadences == 0 ||
                local_tracer_bucket_ids == 0 || local_bucket_member_offsets == 0 ||
                local_bucket_member_counts == 0 || local_bucket_member_write == 0 ||
                local_bucket_indices == 0) {
                local_error = -1;
            } else {
                for (itask = 0; itask < ntracer; itask++) {
                    if (tasks[itask].thread != tid) continue;
                    local_tracers[local_itr++] = tasks[itask].tracer;
                }

                for (local_itr = 0; local_itr < local_count; local_itr++) {
                    const int src = local_tracers[local_itr];
                    int bucket_id = -1;
                    int ibucket;

                    for (ibucket = 0; ibucket < local_nbuckets; ibucket++) {
                        if (local_bucket_steps[ibucket] == nsubsteps[src] &&
                            local_bucket_onsteps[ibucket] == n_onsteps[src] &&
                            local_bucket_base_steps[ibucket] == base_nsteps[src] &&
                            local_bucket_fp_cadences[ibucket] == fp_cadence_steps[src]) {
                            bucket_id = ibucket;
                            break;
                        }
                    }
                    if (bucket_id < 0) {
                        bucket_id = local_nbuckets;
                        local_bucket_steps[bucket_id] = nsubsteps[src];
                        local_bucket_onsteps[bucket_id] = n_onsteps[src];
                        local_bucket_base_steps[bucket_id] = base_nsteps[src];
                        local_bucket_fp_cadences[bucket_id] = fp_cadence_steps[src];
                        local_nbuckets++;
                    }
                    local_tracer_bucket_ids[local_itr] = bucket_id;
                    local_bucket_member_counts[bucket_id]++;
                }

                {
                    int bucket_offset = 0;
                    int ibucket;
                    for (ibucket = 0; ibucket < local_nbuckets; ibucket++) {
                        local_bucket_member_offsets[ibucket] = bucket_offset;
                        local_bucket_member_write[ibucket] = bucket_offset;
                        bucket_offset += local_bucket_member_counts[ibucket];
                    }
                }

                for (local_itr = 0; local_itr < local_count; local_itr++) {
                    const int bucket_id = local_tracer_bucket_ids[local_itr];
                    local_bucket_indices[local_bucket_member_write[bucket_id]++] =
                        local_tracers[local_itr];
                }
                thread_times.bucket_build_ms += now_ms() - t0_bucket_build;

                {
                    int ibucket;
                    for (ibucket = 0; ibucket < local_nbuckets; ibucket++) {
                        const int bucket_count = local_bucket_member_counts[ibucket];
                        const int coeff_interp_segments =
                            tracer_fp_choose_coeff_interp_segments(local_bucket_steps[ibucket],
                                                                   allow_snapshot_interp);
                        const int *bucket_members =
                            local_bucket_indices + (size_t)local_bucket_member_offsets[ibucket];

                        if (bucket_count <= 0) continue;
                        if (local_min_bucket_size == 0 || bucket_count < local_min_bucket_size) {
                            local_min_bucket_size = bucket_count;
                        }
                        if (bucket_count > local_max_bucket_size) {
                            local_max_bucket_size = bucket_count;
                        }

                        if (tracer_fp_batch_execute_bucket(
                                bucket_count, bucket_members,
                                local_bucket_steps[ibucket],
                                local_bucket_onsteps[ibucket],
                                local_bucket_base_steps[ibucket],
                                local_bucket_fp_cadences[ibucket],
                                integration_mode, coeff_interp_segments,
                                allow_snapshot_interp, ntracer,
                                elapsed_gyr, dt_snap, z_curr, z_next,
                                requested_t_acc_direct_gyr,
                                crp_grid, cre_grid,
                                n_gas_snap, kbt_snap, b_field_snap, divv_snap,
                                lturb_snap, dv_snap, cs_snap, beta_snap,
                                rad_ic_zero, rad_ic_m1_zero, rad_ic_p1_zero,
                                n_gas_next_snap, kbt_next_snap, b_field_next_snap, divv_next_snap,
                                lturb_next_snap, dv_next_snap, cs_next_snap, beta_next_snap,
                                rad_ic_row_next, rad_ic_m1_next_val, rad_ic_p1_next_val,
                                qpi_batch, qepri_batch, fqe_flat, np_min_qe,
                                tracer_mass,
                                disable_adiabatic,
                                crp_state, cre_state, &thread_times, &thread_ws) != 0) {
                            local_error = -1;
                            break;
                        }
                        local_bucket_calls++;
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (local_error != 0) {
                ierr_any = -1;
            } else if (ierr_any == 0) {
                accum_times(times, &thread_times);
                bucket_calls_total += local_bucket_calls;
                if (local_min_bucket_size > 0 &&
                    (min_bucket_size_total == 0 || local_min_bucket_size < min_bucket_size_total)) {
                    min_bucket_size_total = local_min_bucket_size;
                }
                if (local_max_bucket_size > max_bucket_size_total) {
                    max_bucket_size_total = local_max_bucket_size;
                }
            }
        }

        tracer_fp_cpu_ws_free(&thread_ws);
        free(local_tracers);
        free(local_bucket_steps);
        free(local_bucket_onsteps);
        free(local_bucket_base_steps);
        free(local_bucket_fp_cadences);
        free(local_tracer_bucket_ids);
        free(local_bucket_member_offsets);
        free(local_bucket_member_counts);
        free(local_bucket_member_write);
        free(local_bucket_indices);
    }

    free(tasks);
    free(thread_loads);
    free(thread_counts);

    if (ierr_any != 0) return -1;
    if (bucket_calls_out) *bucket_calls_out = bucket_calls_total;
    if (min_bucket_size_out) *min_bucket_size_out = min_bucket_size_total;
    if (max_bucket_size_out) *max_bucket_size_out = max_bucket_size_total;
    return 0;
}
