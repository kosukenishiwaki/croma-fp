/*
    tracer_fp_solve.c

    Snapshot-loop solver for tracer_fp.
*/

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <hdf5.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "CONSTANTS.h"
#include "COSFUNC.h"
#include "DSA_MODELS.h"
#include "EMISSION.h"
#include "FP_Coef.h"
#include "READFILE.h"
#include "Synchrotron.h"
#include "tracer_fp_background.h"
#include "tracer_fp_bucket.h"
#include "tracer_fp_cr_init.h"
#include "tracer_fp_coef.h"
#include "tracer_fp_debug.h"
#include "tracer_fp_dsa.h"
#include "tracer_fp_loadbalance.h"
#include "tracer_fp_nsub.h"
#include "tracer_fp_output.h"
#include "tracer_fp_restart.h"
#include "tracer_fp_setup.h"
#include "tracer_fp_solve.h"
#include "tracer_fp_synch.h"
#include "read_grid_hdf5.h"
#ifdef FP_USE_CUDA_BACKEND
#include "fp_cuda_backend.h"
#define TRACER_FP_PROGNAME "tracer_fp_cuda.out"
#else
#define TRACER_FP_PROGNAME "tracer_fp_cpu.out"
#endif

#define TRACER_FP_CPU_OMP_MIN_INTERP 4096

static const int kTracerSynchLogBBins = 512;
static const double kTracerSynchLogBHardMin = -5.0;
static const double kTracerSynchLogBHardMax = 3.0;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1.0e3 * (double)ts.tv_sec + 1.0e-6 * (double)ts.tv_nsec;
}

static double epoch_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return 1.0e3 * (double)ts.tv_sec + 1.0e-6 * (double)ts.tv_nsec;
}

static int bucket_hist_bin(int bucket_count)
{
    if (bucket_count < 32) return 0;
    if (bucket_count < 128) return 1;
    if (bucket_count < 512) return 2;
    if (bucket_count < 2048) return 3;
    if (bucket_count < 8192) return 4;
    return 5;
}

static void segment_fp_range(int total_fp_steps,
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
    #pragma omp parallel for schedule(static) if(n >= TRACER_FP_CPU_OMP_MIN_INTERP)
    for (i = 0; i < n; i++) {
        out[i] = curr[i] + alpha * (next[i] - curr[i]);
    }
}

#ifdef FP_USE_CUDA_BACKEND
static int cuda_group_scratch_ensure(FpCudaPipelineInput **inputs,
                                     FpCoeffBatchInput **coeff_inputs,
                                     int **cell_offsets,
                                     int **rad_ic_offsets,
                                     int required_capacity,
                                     int *capacity_io)
{
    if (inputs == 0 || coeff_inputs == 0 ||
        cell_offsets == 0 || rad_ic_offsets == 0 ||
        capacity_io == 0 || required_capacity <= 0) {
        return -1;
    }
    if (*capacity_io >= required_capacity) return 0;

    free(*inputs);
    free(*coeff_inputs);
    free(*cell_offsets);
    free(*rad_ic_offsets);

    *inputs = (FpCudaPipelineInput *)calloc((size_t)required_capacity, sizeof(FpCudaPipelineInput));
    *coeff_inputs = (FpCoeffBatchInput *)calloc((size_t)required_capacity, sizeof(FpCoeffBatchInput));
    *cell_offsets = (int *)calloc((size_t)required_capacity, sizeof(int));
    *rad_ic_offsets = (int *)calloc((size_t)required_capacity, sizeof(int));
    if (*inputs == 0 || *coeff_inputs == 0 ||
        *cell_offsets == 0 || *rad_ic_offsets == 0) {
        free(*inputs); *inputs = 0;
        free(*coeff_inputs); *coeff_inputs = 0;
        free(*cell_offsets); *cell_offsets = 0;
        free(*rad_ic_offsets); *rad_ic_offsets = 0;
        *capacity_io = 0;
        return -1;
    }

    *capacity_io = required_capacity;
    return 0;
}

static FpCudaTransportMode cuda_transport_mode(void)
{
    static int initialized = 0;
    static FpCudaTransportMode mode = FP_CUDA_TRANSPORT_FUSED_SUBSTEPS;

    if (!initialized) {
        const char *spec = getenv("CROMA_CUDA_TRANSPORT");

        if (spec == 0) {
            spec = getenv("FP_GPU_CUDA_TRANSPORT");
        }

        if (spec != 0) {
            if (strcmp(spec, "fused") == 0 ||
                strcmp(spec, "fused_substeps") == 0) {
                mode = FP_CUDA_TRANSPORT_FUSED_SUBSTEPS;
            } else if (strcmp(spec, "legacy") == 0) {
                mode = FP_CUDA_TRANSPORT_LEGACY;
            }
        }
        initialized = 1;
    }
    return mode;
}

static int cuda_device_bg_interp_enabled(void)
{
    static int initialized = 0;
    static int enabled = 1;

    if (!initialized) {
        const char *spec = getenv("CROMA_CUDA_DEVICE_BG_INTERP");
        if (spec != 0) {
            if (strcmp(spec, "0") == 0 ||
                strcmp(spec, "off") == 0 ||
                strcmp(spec, "false") == 0 ||
                strcmp(spec, "no") == 0) {
                enabled = 0;
            }
        }
        initialized = 1;
    }
    return enabled;
}
#endif

static void record_bucket_metrics(int bucket_count,
                                  int *total_bucket_calls,
                                  long long *total_bucket_cells,
                                  int *min_bucket_size,
                                  int *max_bucket_size_seen,
                                  long long *bucket_hist_counts)
{
    int hist_bin;

    if (bucket_count <= 0) return;
    if (total_bucket_calls) (*total_bucket_calls)++;
    if (total_bucket_cells) *total_bucket_cells += (long long)bucket_count;
    if (min_bucket_size && (*min_bucket_size == 0 || bucket_count < *min_bucket_size)) {
        *min_bucket_size = bucket_count;
    }
    if (max_bucket_size_seen && bucket_count > *max_bucket_size_seen) {
        *max_bucket_size_seen = bucket_count;
    }
    if (bucket_hist_counts) {
        hist_bin = bucket_hist_bin(bucket_count);
        bucket_hist_counts[hist_bin]++;
    }
}

int tracer_fp_solve(TracerFpSolveCtx *ctx,
                    TracerFpBgWin *bg_window,
                    TracerFpOutState *output_state,
                    TracerStep *step,
                    TracerBucket *bucket,
                    TracerFpGpuTimes *times,
                    SynchData *synch,
                    TracerFpCpuWs *cpu_ws,
                    void *cuda_workspace,
                    CRspectrum *crp_grid,
                    CRspectrum *cre_grid,
                    DSAGrid *dsa_grid,
                    int start_snapshot,
                    int nsnap,
                    int ntracer,
                    int ntracer_global,
                    int nfreq,
                    TracerFpInputMode input_mode,
                    TracerFpBackgroundMode background_mode,
                    int mpi_rank,
                    int use_cuda_backend,
                    TracerFpFileOutputMode file_output_mode,
                    TracerFpWriteBufferMode write_buffer_mode,
                    int mapped_chunk_snapshots,
                    int debug_target_local,
                    long int debug_target_global,
                    long int tracer_start,
                    TracerDsaInjectionMode dsa_injection_mode,
                    TracerDsaReaccMode dsa_reacc_mode,
                    double requested_t_acc_direct_gyr)
{
    const TracerFpOutputSchedule *schedule = ctx->schedule;
    const int *source_offsets = ctx->source_offsets;
    const long int *tracer_ids = ctx->tracer_ids;
    FILE *input_read_fp = ctx->input_read_fp;
    int *input_read_order = ctx->input_read_order;
    const int local_rank = ctx->local_rank;
    const char *hostname = ctx->hostname;
    const char *checkpoint_dir_effective = ctx->checkpoint_dir;
    double *dt_snap = ctx->dt_snap;
    double *z_snap = ctx->z_snap;
    double elapsed_gyr = *ctx->elapsed_gyr;
    TracerFpBackgroundSlot *bg_slots = bg_window->bg_slots;
    TracerFpRawBackgroundSlot *raw_slots = bg_window->raw_slots;
    TracerFpHdf5Meta hdf5_meta = *bg_window->hdf5_meta;
    int bg_curr_slot = *bg_window->bg_curr;
    int bg_next_slot = *bg_window->bg_next;
    int raw_prev_slot = *bg_window->raw_prev;
    int raw_curr_slot = *bg_window->raw_curr;
    int raw_next_slot = *bg_window->raw_next;
    TracerStep step_local = *step;
    TracerBucket bucket_local = *bucket;
    TracerFpGpuTimes times_local = *times;
    SynchData synch_local = *synch;
    TracerFpCpuWs cpu_workspace_local = *cpu_ws;
    unsigned char *dsa_disable_adiabatic_local = 0;
#ifdef FP_USE_CUDA_BACKEND
    FpCudaPipelineWorkspace *cuda_workspace_local = (FpCudaPipelineWorkspace *)cuda_workspace;
    FpCudaPipelineInput *gpu_group_inputs_local = 0;
    FpCoeffBatchInput *gpu_group_coeff_inputs_local = 0;
    int *gpu_group_cell_offsets_local = 0;
    int *gpu_group_rad_ic_offsets_local = 0;
    int gpu_group_launch_capacity = 0;
#endif
    TracerFpMappedOutput ne_output_local = *output_state->ne_output;
    TracerFpMappedOutput np_output_local = *output_state->np_output;
    TracerFpMappedOutput epssyn_output_local = *output_state->epssyn_output;
    TracerFpMappedOutput epsic_output_local = *output_state->epsic_output;
    TracerFpMappedOutput epsgamma_output_local = *output_state->epsgamma_output;
    TracerFpMappedOutput epsnu_output_local = *output_state->epsnu_output;
    TracerFpTileOutput ne_tile_output_local = *output_state->ne_tile_output;
    TracerFpTileOutput np_tile_output_local = *output_state->np_tile_output;
    TracerFpTileOutput epssyn_tile_output_local = *output_state->epssyn_tile_output;
    TracerFpTileOutput epsic_tile_output_local = *output_state->epsic_tile_output;
    TracerFpTileOutput epsgamma_tile_output_local = *output_state->epsgamma_tile_output;
    TracerFpTileOutput epsnu_tile_output_local = *output_state->epsnu_tile_output;
    CRspectrum crp_grid_local = *crp_grid;
    CRspectrum cre_grid_local = *cre_grid;
    DSAGrid dsa_grid_local = *dsa_grid;
    double *beta_p = ctx->beta_p;
    double *gamma2e = ctx->gamma2e;
    double *tracer_mass = ctx->tracer_mass;
    double *rad_ic_zero = ctx->rad_ic_zero;
    double *rad_ic_m1_zero = ctx->rad_ic_m1_zero;
    double *rad_ic_p1_zero = ctx->rad_ic_p1_zero;
    double *eps_syn = ctx->eps_syn;
    double *eps_ic = ctx->eps_ic;
    double *eps_gamma = ctx->eps_gamma;
    double *eps_nu = ctx->eps_nu;
    double *b_dyn = ctx->b_dyn;
    double *logb = ctx->logb;
    double *fqe_flat = ctx->fqe_flat;
    double *fic_flat = ctx->fic_flat;
    double *fga_flat = ctx->fga_flat;
    double *fnu_flat = ctx->fnu_flat;
    int *np_min_qe = ctx->np_min_qe;
    double *qpi_batch = ctx->qpi_batch;
    double *qepri_batch = ctx->qepri_batch;
    double *crp_state = ctx->crp_state;
    double *cre_state = ctx->cre_state;
    int *tracer_id_core = ctx->tracer_id_core;
    double **ne_buffer_core = output_state->ne_buffer_core;
    double **np_buffer_core = output_state->np_buffer_core;
    double **epssyn_buffer_core = output_state->epssyn_buffer_core;
    double **epsic_buffer_core = output_state->epsic_buffer_core;
    double **epsgamma_buffer_core = output_state->epsgamma_buffer_core;
    double **epsnu_buffer_core = output_state->epsnu_buffer_core;
    double **ne_chunk_core = output_state->ne_chunk_core;
    double **np_chunk_core = output_state->np_chunk_core;
    double **epssyn_chunk_core = output_state->epssyn_chunk_core;
    double **epsic_chunk_core = output_state->epsic_chunk_core;
    double **epsgamma_chunk_core = output_state->epsgamma_chunk_core;
    double **epsnu_chunk_core = output_state->epsnu_chunk_core;
    int mapped_chunk_start_snapshot = *output_state->chunk_start;
    int mapped_chunk_count = *output_state->chunk_count;
    int mapped_chunk_cr_base_slot = *output_state->chunk_cr_base;
    int mapped_chunk_cr_slots = *output_state->chunk_cr_slots;
    FILE *bucketstats_top_fp = output_state->bucket_top_fp;
    FILE *bucketstats_rank_fp = output_state->bucket_rank_fp;
    int emitted_synch = *ctx->emitted_synch;
    int emitted_gamma = *ctx->emitted_gamma;
    int emitted_neutrino = *ctx->emitted_neutrino;
    int last_synch_snapshot = *ctx->last_synch_snap;
    int total_dsa_injected = *ctx->dsa_injected;
    int total_bucket_calls = *ctx->bucket_calls;
    long long total_bucket_cells = *ctx->bucket_cells;
    int min_bucket_size = *ctx->bucket_min;
    int max_bucket_size_seen = *ctx->bucket_max;
    int max_bucket_count_per_snapshot = *ctx->max_buckets_per_snap;
    int max_nsubsteps = *ctx->max_nsubsteps;
    long long total_capped_tracer_snapshots = *ctx->capped_tracer_snaps;
    int max_raw_nsubsteps = *ctx->max_raw_nsub;
    long long total_gpu_pipeline_calls = *ctx->gpu_pipeline_calls;
    long long total_gpu_pipeline_cells = *ctx->gpu_pipeline_cells;
    long long total_gpu_pipeline_fp_steps = *ctx->gpu_pipeline_fp_steps;
    long long total_gpu_pipeline_cell_steps = *ctx->gpu_pipeline_cell_steps;
    long long total_gpu_group_count_est = *ctx->gpu_group_count_est;
    int max_gpu_groups_per_snapshot = *ctx->max_gpu_groups_per_snap;
    long long *bucket_hist_counts = ctx->bucket_hist_counts;
    long long *runtime_snapshot_nsub_local = ctx->runtime_nsub;
    long long *runtime_snapshot_target_nsub_local = ctx->runtime_target_nsub;
    int world_size = 1;
    const int log_root = (mpi_rank == 0);
    const int has_crp_emission = (seed_cr_species != SEED_CR_SPECIES_ELECTRON_ONLY) ? 1 : 0;
    const int ic_enabled = tracer_write_ic_output;
    const int gamma_enabled = has_crp_emission && tracer_write_gamma_output;
    const int neutrino_enabled = has_crp_emission && tracer_write_neutrino_output;
    const int write_output_files = (file_output_mode == TRACER_FP_OUTPUT_WRITE);
    const int use_tile_output =
        (write_output_files && write_buffer_mode == TRACER_FP_WRITE_BUFFER_TILE);
    const int use_mapped_output =
        (write_output_files && write_buffer_mode == TRACER_FP_WRITE_BUFFER_MAPPED);
    const int use_mapped_chunk = (use_mapped_output && mapped_chunk_snapshots > 1);
    const int write_crp_output = (tracer_write_crp_output != 0);
    const int bucket_stats_only = (file_output_mode == TRACER_FP_OUTPUT_BUCKET_STATS);
    const int checkpoint_enabled = (tracer_checkpoint_interval > 0);
    const int direct_tacc_schedule_enabled =
        (resolve_momentumdiff_model(FP_MOMENTUMDIFF_MODEL_AUTO) ==
             FP_MOMENTUMDIFF_MODEL_DIRECT_TACC &&
         isfinite(requested_t_acc_direct_gyr) && requested_t_acc_direct_gyr > 0.0 &&
         isfinite(t_off_gyr) && t_off_gyr > 0.0 &&
         isfinite(t_acc_off_gyr) && t_acc_off_gyr > 0.0);
    const size_t np_batch = (size_t)ntracer * (size_t)np;
    const size_t npe_batch = (size_t)ntracer * (size_t)npe;
    double debug_target_cre_before_bucket[npe];
    TracerGpuBucketGroups gpu_bucket_groups_local;
    /* Keep the grouped CUDA path available in-tree, but default to the
     * simpler bucket-by-bucket execution route. */
    const int enable_cuda_bucket_group_path = 0;
    const int use_cuda_rank_state_path = (use_cuda_backend != 0);
#ifdef FP_USE_CUDA_BACKEND
    const int enable_cuda_device_bg_interp =
        (use_cuda_backend != 0) ? cuda_device_bg_interp_enabled() : 0;
#else
    const int enable_cuda_device_bg_interp = 0;
#endif
    int isnap;
    int itr;
    int i;
    int idx;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    tracer_gpu_bucket_groups_reset(&gpu_bucket_groups_local);
    dsa_disable_adiabatic_local = (unsigned char *)calloc((size_t)ntracer, sizeof(unsigned char));
    if (dsa_disable_adiabatic_local == 0) {
        goto cleanup;
    }
    if (use_cuda_backend && enable_cuda_bucket_group_path &&
        tracer_gpu_bucket_groups_alloc(&gpu_bucket_groups_local, ntracer) != 0) {
        goto cleanup;
    }

#include "tracer_fp_solve.inc"

    tracer_gpu_bucket_groups_release(&gpu_bucket_groups_local);
    free(dsa_disable_adiabatic_local);
#ifdef FP_USE_CUDA_BACKEND
    free(gpu_group_inputs_local);
    free(gpu_group_coeff_inputs_local);
    free(gpu_group_cell_offsets_local);
    free(gpu_group_rad_ic_offsets_local);
#endif
    *ctx->elapsed_gyr = elapsed_gyr;
    *bg_window->hdf5_meta = hdf5_meta;
    *bg_window->bg_curr = bg_curr_slot;
    *bg_window->bg_next = bg_next_slot;
    *bg_window->raw_prev = raw_prev_slot;
    *bg_window->raw_curr = raw_curr_slot;
    *bg_window->raw_next = raw_next_slot;
    *step = step_local;
    *bucket = bucket_local;
    *times = times_local;
    *synch = synch_local;
    *cpu_ws = cpu_workspace_local;
    *output_state->ne_output = ne_output_local;
    *output_state->np_output = np_output_local;
    *output_state->epssyn_output = epssyn_output_local;
    *output_state->epsic_output = epsic_output_local;
    *output_state->epsgamma_output = epsgamma_output_local;
    *output_state->epsnu_output = epsnu_output_local;
    *output_state->ne_tile_output = ne_tile_output_local;
    *output_state->np_tile_output = np_tile_output_local;
    *output_state->epssyn_tile_output = epssyn_tile_output_local;
    *output_state->epsic_tile_output = epsic_tile_output_local;
    *output_state->epsgamma_tile_output = epsgamma_tile_output_local;
    *output_state->epsnu_tile_output = epsnu_tile_output_local;
    *crp_grid = crp_grid_local;
    *cre_grid = cre_grid_local;
    *dsa_grid = dsa_grid_local;
    *output_state->chunk_start = mapped_chunk_start_snapshot;
    *output_state->chunk_count = mapped_chunk_count;
    *output_state->chunk_cr_base = mapped_chunk_cr_base_slot;
    *output_state->chunk_cr_slots = mapped_chunk_cr_slots;
    *ctx->emitted_synch = emitted_synch;
    *ctx->emitted_gamma = emitted_gamma;
    *ctx->emitted_neutrino = emitted_neutrino;
    *ctx->last_synch_snap = last_synch_snapshot;
    *ctx->dsa_injected = total_dsa_injected;
    *ctx->bucket_calls = total_bucket_calls;
    *ctx->bucket_cells = total_bucket_cells;
    *ctx->bucket_min = min_bucket_size;
    *ctx->bucket_max = max_bucket_size_seen;
    *ctx->max_buckets_per_snap = max_bucket_count_per_snapshot;
    *ctx->max_nsubsteps = max_nsubsteps;
    *ctx->capped_tracer_snaps = total_capped_tracer_snapshots;
    *ctx->max_raw_nsub = max_raw_nsubsteps;
    *ctx->gpu_pipeline_calls = total_gpu_pipeline_calls;
    *ctx->gpu_pipeline_cells = total_gpu_pipeline_cells;
    *ctx->gpu_pipeline_fp_steps = total_gpu_pipeline_fp_steps;
    *ctx->gpu_pipeline_cell_steps = total_gpu_pipeline_cell_steps;
    *ctx->gpu_group_count_est = total_gpu_group_count_est;
    *ctx->max_gpu_groups_per_snap = max_gpu_groups_per_snapshot;
    times_local.gpu_pipeline_calls = total_gpu_pipeline_calls;
    times_local.gpu_pipeline_cells = total_gpu_pipeline_cells;
    times_local.gpu_pipeline_fp_steps = total_gpu_pipeline_fp_steps;
    times_local.gpu_pipeline_cell_steps = total_gpu_pipeline_cell_steps;
    return 0;

cleanup:
    tracer_gpu_bucket_groups_release(&gpu_bucket_groups_local);
    free(dsa_disable_adiabatic_local);
#ifdef FP_USE_CUDA_BACKEND
    free(gpu_group_inputs_local);
    free(gpu_group_coeff_inputs_local);
    free(gpu_group_cell_offsets_local);
    free(gpu_group_rad_ic_offsets_local);
#endif
    *cpu_ws = cpu_workspace_local;
    return 1;
}
