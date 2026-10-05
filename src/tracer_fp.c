/*
    tracer_fp.c

    K. Nishiwaki, 2026-06-18
    - Main Fokker-Planck driver for tracer mode.

*/


#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
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
#include "tracer_fp_alloc.h"
#include "tracer_fp_background.h"
#include "tracer_fp_bucket.h"
#include "tracer_fp_cr_init.h"
#include "tracer_fp_cr_source.h"
#include "tracer_fp_data.h"
#include "tracer_fp_debug.h"
#include "tracer_fp_dsa.h"
#include "tracer_fp_entry_helpers.h"
#include "tracer_fp_loadbalance.h"
#include "tracer_fp_nsub.h"
#include "tracer_fp_output.h"
#include "tracer_fp_restart.h"
#include "tracer_fp_selection.h"
#include "tracer_fp_solve.h"
#include "tracer_fp_setup.h"
#include "tracer_fp_step.h"
#include "tracer_fp_synch.h"
#include "tracer_fp.h"
#include "read_grid_hdf5.h"
#ifdef FP_USE_CUDA_BACKEND
#include "fp_cuda_backend.h"
#define TRACER_FP_PROGNAME "tracer_fp_cuda.out"
#else
#define TRACER_FP_PROGNAME "tracer_fp_cpu.out"
#endif

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

static void init_test_params(void)
{
    pmin = -1.0;
    pmax = 8.0;
    pemin = -0.5;
    pemax = 6.0;
}

static const char *planned_cuda_solver_name(int nrow)
{
#ifdef FP_USE_CUDA_BACKEND
    return fp_cuda_tridiag_solver_name(nrow);
#else
    (void)nrow;
    return "Thomas";
#endif
}

static const char *accel_model_name(void)
{
    return momentumdiff_model_name(FP_MOMENTUMDIFF_MODEL_AUTO);
}

static const char *primary_electron_injection_name(void)
{
    return (steady_primary_electron_injection != 0) ? "steady" : "one_shot";
}

static void print_physics_setup(TracerDsaInjectionMode dsa_injection_mode,
                                TracerDsaReaccMode dsa_reacc_mode,
                                int dsa_zero_seed_proton_note,
                                double requested_t_acc_direct_gyr)
{
    printf("  accel model         : %s\n", accel_model_name());
    printf("  primary e injection : %s\n", primary_electron_injection_name());
    printf("  DSA injection       : %s\n",
           tracer_fp_dsa_injection_mode_name(dsa_injection_mode));
    printf("  DSA species         : %s\n", DSAInjectSpeciesSpec);
    printf("  DSA reacceleration  : %s\n",
           tracer_fp_dsa_reacc_mode_name(dsa_reacc_mode));
    if (dsa_zero_seed_proton_note) {
        printf("  proton initial mode : zero state; DSA-only population\n");
    }
    if (resolve_momentumdiff_model(FP_MOMENTUMDIFF_MODEL_AUTO) ==
            FP_MOMENTUMDIFF_MODEL_DIRECT_TACC &&
        isfinite(requested_t_acc_direct_gyr) && requested_t_acc_direct_gyr > 0.0 &&
        isfinite(t_off_gyr) && t_off_gyr > 0.0 &&
        isfinite(t_acc_off_gyr) && t_acc_off_gyr > 0.0) {
        printf("  direct t_acc quiet  : %.6e Gyr until %.6e Gyr\n",
               t_acc_off_gyr, t_off_gyr);
        printf("  direct t_acc active : %.6e Gyr after %.6e Gyr\n",
               requested_t_acc_direct_gyr, t_off_gyr);
    }
    printf("  reacc window mode   : %s\n",
           (reacc_window_mode == REACC_WINDOW_CENTERED) ? "centered off/on/off" : "always-on");
    printf("  nsub safety         : %.6g\n", tracer_nsub_safety);
    printf("  nsub cap            : %d\n", tracer_nsub_max);
    printf("  coeff min steps     : %d\n", coeff_interp_min_steps);
    printf("  coeff max segments  : %d\n", coeff_interp_max_segments);
}

static void print_seed_cr_setup(const CRspectrum *crp_grid,
                                const CRspectrum *cre_grid)
{
    double *crp_template = 0;
    double *cre_template = 0;
    double num_p = 0.0;
    double num_e = 0.0;
    double energy_p = 0.0;
    double energy_e = 0.0;
    double phi_p_effective = phi_CRp;
    double phi_e_effective = phi_CRe;
    const int seed_p = (seed_cr_species == SEED_CR_SPECIES_PROTON_ONLY ||
                        seed_cr_species == SEED_CR_SPECIES_ELECTRON_PROTON);
    const int seed_e = (seed_cr_species == SEED_CR_SPECIES_ELECTRON_ONLY ||
                        seed_cr_species == SEED_CR_SPECIES_ELECTRON_PROTON);

    if (!seed_p) phi_p_effective = 0.0;
    if (!seed_e) phi_e_effective = 0.0;

    crp_template = (double *)calloc((size_t)np, sizeof(double));
    cre_template = (double *)calloc((size_t)npe, sizeof(double));
    if (crp_template != 0 && cre_template != 0) {
        spectrum_temprate(delta_CR_inj,
                          (CRspectrum *)crp_grid,
                          (CRspectrum *)cre_grid,
                          crp_template,
                          cre_template,
                          &num_p,
                          &energy_p,
                          &num_e,
                          &energy_e);
        if (initial_cr_norm_mode == INITIAL_CR_NORM_ENERGY_RATIO &&
            seed_p && seed_e &&
            num_p > 0.0 && num_e > 0.0 &&
            energy_p > 0.0 && energy_e > 0.0) {
            const double mean_energy_p = energy_p / num_p;
            const double mean_energy_e = energy_e / num_e;
            if (initial_cr_energy_anchor == INITIAL_CR_ENERGY_ANCHOR_ELECTRON) {
                phi_p_effective =
                    initial_cr_energy_ratio_p_to_e * phi_CRe *
                    mean_energy_e / mean_energy_p;
            } else {
                phi_e_effective =
                    phi_CRp * mean_energy_p /
                    (initial_cr_energy_ratio_p_to_e * mean_energy_e);
            }
        }
    }

    printf("  seed CR species     : %s\n", seed_cr_species_spec);
    printf("  seed phi_CRe        : configured=%.6e effective=%.6e\n",
           phi_CRe, phi_e_effective);
    printf("  seed phi_CRp        : configured=%.6e effective=%.6e\n",
           phi_CRp, phi_p_effective);
    printf("  seed norm mode      : %s\n", initial_cr_norm_mode_spec);
    if (initial_cr_norm_mode == INITIAL_CR_NORM_ENERGY_RATIO) {
        printf("  seed energy anchor  : %s\n", initial_cr_energy_anchor_spec);
        printf("  seed energy ratio   : CRp/CRe=%.6e\n",
               initial_cr_energy_ratio_p_to_e);
    }

    free(crp_template);
    free(cre_template);
}

typedef struct {
    TracerFpMappedOutput ne;
    TracerFpMappedOutput crp;
    TracerFpMappedOutput epssyn;
    TracerFpMappedOutput epsic;
    TracerFpMappedOutput epsgamma;
    TracerFpMappedOutput epsnu;
    TracerFpTileOutput ne_tile;
    TracerFpTileOutput np_tile;
    TracerFpTileOutput epssyn_tile;
    TracerFpTileOutput epsic_tile;
    TracerFpTileOutput epsgamma_tile;
    TracerFpTileOutput epsnu_tile;
    double **ne_buf;
    double **np_buf;
    double **epssyn_buf;
    double **epsic_buf;
    double **epsgamma_buf;
    double **epsnu_buf;
    double **ne_chunk;
    double **np_chunk;
    double **epssyn_chunk;
    double **epsic_chunk;
    double **epsgamma_chunk;
    double **epsnu_chunk;
    int *tracer_id_core;
    FILE *bucket_top_fp;
    FILE *bucket_rank_fp;
    FILE *input_read_fp;
    FILE *dsa_reacc_debug_fp;
    char dir[MAX_LINE_LENGTH];
    char checkpoint_dir[MAX_LINE_LENGTH];
} TracerFpDriverOutputState;

typedef struct {
    int emitted_synch;
    int emitted_gamma;
    int emitted_neutrino;
    int last_synch_snapshot;
    int total_dsa_injected;
    int total_bucket_calls;
    long long total_bucket_cells;
    int min_bucket_size;
    int max_bucket_size;
    int max_bucket_count;
    int max_nsubsteps;
    long long capped_tracer_snapshots;
    int max_raw_nsubsteps;
    long long gpu_pipeline_calls;
    long long gpu_pipeline_cells;
    long long gpu_pipeline_fp_steps;
    long long gpu_pipeline_cell_steps;
    long long gpu_group_count_est;
    int max_gpu_groups_per_snap;
    long long bucket_hist_counts[TRACER_FP_BUCKET_HIST_NBIN];
} TracerFpDriverStats;

typedef struct {
    int chunk_snapshots;
    int use_mapped_chunk;
    int start_snapshot;
    int count;
    int cr_base_slot;
    int cr_slots;
} TracerFpMappedChunkState;

typedef struct {
    double t0_ms;
    double t1_ms;
    double ms;
    double max_ms;
    double min_ms;
    double sum_ms;
    double mean_ms;
} TracerFpWallTimes;

static void selected_offset_bounds(long int tracer_start,
                                   const int *selected_offsets,
                                   int nselected,
                                   long int *offset_min_out,
                                   long int *offset_max_out,
                                   long int *offset_span_out)
{
    long int offset_min = 0;
    long int offset_max = 0;
    long int offset_span = 0;

    if (selected_offsets != 0 && nselected > 0) {
        offset_min = tracer_start + (long int)selected_offsets[0];
        offset_max = tracer_start + (long int)selected_offsets[0];
        for (int i = 1; i < nselected; i++) {
            const long int offset = tracer_start + (long int)selected_offsets[i];
            if (offset < offset_min) offset_min = offset;
            if (offset > offset_max) offset_max = offset;
        }
        offset_span = offset_max - offset_min + 1;
    }

    if (offset_min_out != 0) *offset_min_out = offset_min;
    if (offset_max_out != 0) *offset_max_out = offset_max;
    if (offset_span_out != 0) *offset_span_out = offset_span;
}

static int open_input_read_log(TracerFpDriverOutputState *out,
                               int mpi_rank,
                               int local_rank,
                               const char *hostname)
{
    char path[MAX_LINE_LENGTH];

    if (out == 0) return -1;
    if (out->dir[0] == '\0') {
        if (output_dir[0] != '\0') {
            snprintf(out->dir, sizeof(out->dir), "%s", output_dir);
        } else {
            snprintf(out->dir, sizeof(out->dir), "tracer_fp_gpu_output");
        }
    }
    if (ensure_output_dir(out->dir) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to create output directory '%s'\n",
                out->dir);
        return -1;
    }
    snprintf(path, sizeof(path), "%s/input_read_rank%03d.tsv", out->dir, mpi_rank);
    out->input_read_fp = fopen(path, "w");
    if (out->input_read_fp == 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to open input read log '%s'\n",
                path);
        return -1;
    }
    fprintf(out->input_read_fp,
            "rank\tlocal_rank\thostname\tread_order\tphase\tsnapshot_index\tphysical_snapshot\tz\tread_start_epoch_ms\tread_end_epoch_ms\tread_ms\tlocal_ntracer\tselected_count\toffset_runs\toffset_min_run_len\toffset_max_run_len\toffset_max_gap\toffset_min\toffset_max\toffset_span\n");
    fflush(out->input_read_fp);
    (void)local_rank;
    (void)hostname;
    return 0;
}

static int open_dsa_reacc_debug_log(TracerFpDriverOutputState *out,
                                    int mpi_rank)
{
    char path[MAX_LINE_LENGTH];

    if (out == 0) return -1;
    if (out->dir[0] == '\0') {
        if (output_dir[0] != '\0') {
            snprintf(out->dir, sizeof(out->dir), "%s", output_dir);
        } else {
            snprintf(out->dir, sizeof(out->dir), "tracer_fp_gpu_output");
        }
    }
    if (ensure_output_dir(out->dir) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to create output directory '%s'\n",
                out->dir);
        return -1;
    }
    snprintf(path, sizeof(path), "%s/dsa_reacc_debug_rank%03d.tsv", out->dir, mpi_rank);
    out->dsa_reacc_debug_fp = fopen(path, "w");
    if (out->dsa_reacc_debug_fp == 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to open DSA reacc debug log '%s'\n",
                path);
        return -1;
    }
    fprintf(out->dsa_reacc_debug_fp,
            "rank\tsnapshot\ttracer_id\tinjection_mode\treacc_mode\tshock_source\t"
            "side_code\tdsa_species\tmach\tcompression_ratio\trho1_cgs\trho2_cgs\t"
            "u1_cms\tkinetic_flux_cgs\tmass_cgs\tvolume_upstream_cgs\t"
            "volume_downstream_cgs\tshock_area_dt_cm2_s\tflux_budget_erg\t"
            "eta_cap\tbudget_erg\treacc_energy_erg\t"
            "nonadiabatic_energy_erg\tbudget_ratio\treacc_scale\t"
            "seed_p_energy_erg\tseed_e_energy_erg\tconv_p_energy_erg\t"
            "conv_e_energy_erg\tad_p_energy_erg\tad_e_energy_erg\t"
            "nonad_p_energy_erg\tnonad_e_energy_erg\tcap_hit\t"
            "inject_proton\tinject_electron\n");
    fflush(out->dsa_reacc_debug_fp);
    return 0;
}

static void log_input_read(FILE *fp,
                           int mpi_rank,
                           int local_rank,
                           const char *hostname,
                           int *read_order,
                           const char *phase,
                           int snapshot_index,
                           int physical_snapshot,
                           double z_snapshot,
                           double read_start_epoch_ms,
                           double read_end_epoch_ms,
                           double read_ms,
                           int ntracer,
                           int selected_count,
                           int offset_runs,
                           int offset_min_run_len,
                           int offset_max_run_len,
                           int offset_max_gap,
                           long int offset_min,
                           long int offset_max,
                           long int offset_span)
{
    int order = 0;

    if (read_order != 0) {
        order = *read_order;
        *read_order = order + 1;
    }
    if (fp == 0) return;
    fprintf(fp,
            "%d\t%d\t%s\t%d\t%s\t%d\t%d\t%.8e\t%.3f\t%.3f\t%.6f\t%d\t%d\t%d\t%d\t%d\t%d\t%ld\t%ld\t%ld\n",
            mpi_rank, local_rank, (hostname != 0 && *hostname != '\0') ? hostname : "unknown",
            order, (phase != 0) ? phase : "unknown",
            snapshot_index, physical_snapshot, z_snapshot,
            read_start_epoch_ms, read_end_epoch_ms, read_ms,
            ntracer, selected_count, offset_runs, offset_min_run_len,
            offset_max_run_len, offset_max_gap, offset_min, offset_max, offset_span);
    fflush(fp);
}

static int load_hdf5_raw_snapshot_logged(const TracerFpHdf5Meta *meta,
                                         int physical_snapshot,
                                         int snapshot_index,
                                         const char *phase,
                                         long int tracer_start,
                                         const int *source_offsets,
                                         int ntracer,
                                         TracerFpRawBackgroundSlot *raw,
                                         TracerFpGpuTimes *times,
                                         FILE *input_read_fp,
                                         int mpi_rank,
                                         int local_rank,
                                         const char *hostname,
                                         int *read_order,
                                         int selected_count,
                                         int offset_runs,
                                         int offset_min_run_len,
                                         int offset_max_run_len,
                                         int offset_max_gap,
                                         long int offset_min,
                                         long int offset_max,
                                         long int offset_span,
                                         const double *z_snap,
                                         int nsnap)
{
    const double z_snapshot =
        (z_snap != 0 && snapshot_index >= 0 && snapshot_index < nsnap) ? z_snap[snapshot_index] : 0.0;
    const double t0_read = now_ms();
    const double t0_epoch = epoch_ms();
    double read_ms;
    double t1_epoch;

    if (tracer_load_hdf5_raw_snapshot_slice(meta, physical_snapshot,
                                            tracer_start, source_offsets, ntracer,
                                            raw) != 0) {
        return -1;
    }
    read_ms = now_ms() - t0_read;
    t1_epoch = epoch_ms();
    if (times != 0) {
        times->input_read_ms += read_ms;
        times->input_read_calls++;
        times->input_selected_runs += selected_count;
        times->input_offset_runs += offset_runs;
        if (offset_min_run_len > 0 &&
            (times->input_offset_min_run_len == 0 ||
             offset_min_run_len < times->input_offset_min_run_len)) {
            times->input_offset_min_run_len = offset_min_run_len;
        }
        if (offset_max_run_len > times->input_offset_max_run_len) {
            times->input_offset_max_run_len = offset_max_run_len;
        }
        if (offset_max_gap > times->input_offset_max_gap) {
            times->input_offset_max_gap = offset_max_gap;
        }
    }
    log_input_read(input_read_fp, mpi_rank, local_rank, hostname,
                   read_order, phase,
                   snapshot_index, physical_snapshot, z_snapshot,
                   t0_epoch, t1_epoch, read_ms,
                   ntracer, selected_count, offset_runs,
                   offset_min_run_len, offset_max_run_len, offset_max_gap,
                   offset_min, offset_max, offset_span);
    return 0;
}

static int run_tracer_driver(const TracerFpSelection *selection,
                             const TracerFpConfig *config)
{
    if (selection == 0 || config == 0) return 1;

    /* Expand the immutable selection/config inputs first so the rest of the
     * driver can work with plain local variables. */
    const int ntracer = selection->ntracer;
    const long int tracer_start = selection->tracer_start;
    const int source_ntracer = selection->source_ntracer;
    const int ntracer_global = selection->ntracer_global;
    const long int *tracer_ids = selection->tracer_ids;
    const int *source_offsets = selection->source_offsets;
    const long long *estimate_snapshot_nsub = selection->estimate_snapshot_nsub;
    const long long *estimate_snapshot_target_nsub = selection->estimate_snapshot_target_nsub;
    const int skip_heavy = selection->skip_heavy;
    const int mpi_rank = config->mpi_rank;
    const int mpi_size = config->mpi_size;
    const int nsnap = config->nsnap;
    const int nfreq = config->nfreq;
    const char *output_spec = config->output_spec;
    const TracerFpOutputSchedule *schedule = config->schedule;
    const TracerFpBackendMode backend_mode = config->backend_mode;
    const TracerFpInputMode input_mode = config->input_mode;
    const char *params_file = config->params_file;
    const TracerFpFileOutputMode file_output_mode = config->file_output_mode;
    const TracerFpWriteBufferMode write_buffer_mode = config->write_buffer_mode;
    const TracerFpIntegrationMode integration_mode = config->integration_mode;
    const TracerFpBackgroundMode background_mode = config->background_mode;
    const TracerFpOpenmpInfo *omp_info = config->omp_info;
    const int local_rank = config->local_rank;
    double load_balance_ms = config->load_balance_ms;
    const size_t np_batch = (size_t)ntracer * (size_t)np;
    const size_t npe_batch = (size_t)ntracer * (size_t)npe;
    const size_t bg_slot_storage_size = 2u * (size_t)ntracer;
    CRspectrum crp_grid, cre_grid;
    double beta_p[np], gamma2e[npe];
    double *dt_snap = 0, *z_snap = 0;
    TracerDataHistory bg_history;
    TracerFpBackgroundSlot bg_slots[2];
    int bg_curr_slot = 0;
    int bg_next_slot = 1;
    TracerFpHdf5Meta hdf5_meta;
    TracerDataStorage raw_storage;
    TracerFpRawBackgroundSlot raw_slots[3];
    int raw_prev_slot = 0;
    int raw_curr_slot = 1;
    int raw_next_slot = 2;
    double *rad_ic_zero = 0, *rad_ic_m1_zero = 0, *rad_ic_p1_zero = 0;
    double *crp_state = 0, *cre_state = 0;
    double *qpi_batch = 0, *qepri_batch = 0;
    double *tracer_mass = 0;
    DSAGrid dsa_grid;
    TracerDsaInjectionMode dsa_injection_mode = TRACER_DSA_INJECTION_OFF;
    TracerDsaReaccMode dsa_reacc_mode = TRACER_DSA_REACC_OFF;
    double *fqe_flat = 0;
    double *fic_flat = 0;
    double *fga_flat = 0;
    double *fnu_flat = 0;
    int *np_min_qe = 0;
    SynchData synch;
    double *eps_syn = 0, *eps_ic = 0, *eps_gamma = 0, *eps_nu = 0, *b_dyn = 0, *logb = 0;
    TracerStep step;
    TracerBucket bucket;
    long long *runtime_snapshot_nsub_local = 0, *runtime_snapshot_nsub_global = 0;
    long long *runtime_snapshot_target_nsub_local = 0, *runtime_snapshot_target_nsub_global = 0;
    TracerFpDriverOutputState out;
    TracerFpGpuTimes times;
    TracerFpDriverStats stats;
    TracerFpMappedChunkState chunk;
    TracerFpWallTimes wall;
    int use_cuda_backend = 0;
    TracerFpCpuWs cpu_workspace;
#ifdef FP_USE_CUDA_BACKEND
    FpCudaPipelineWorkspace *cuda_workspace = 0;
#endif
    int has_crp_emission = (seed_cr_species != SEED_CR_SPECIES_ELECTRON_ONLY) ? 1 : 0;
    int ic_enabled = tracer_write_ic_output;
    int gamma_enabled = has_crp_emission && tracer_write_gamma_output;
    int neutrino_enabled = has_crp_emission && tracer_write_neutrino_output;
    int dsa_zero_seed_proton_note = 0;
    const int write_output_files = (file_output_mode == TRACER_FP_OUTPUT_WRITE);
    int write_crp_output = (tracer_write_crp_output != 0);
    const int use_buffered_output = (write_output_files &&
                                     write_buffer_mode == TRACER_FP_WRITE_BUFFER_BUFFERED);
    const int use_mapped_output = (write_output_files &&
                                   write_buffer_mode == TRACER_FP_WRITE_BUFFER_MAPPED);
    const int use_tile_output = (write_output_files &&
                                 write_buffer_mode == TRACER_FP_WRITE_BUFFER_TILE);
    const int bucket_stats_only = (file_output_mode == TRACER_FP_OUTPUT_BUCKET_STATS);
    const int restart_enabled = (tracer_restart_dir[0] != '\0');
    const int checkpoint_enabled = (tracer_checkpoint_interval > 0);
    const long debug_target_global = tracer_fp_debug_target_global_id();
    const int log_root = (mpi_rank == 0);
    int debug_target_local = -1;
    double debug_target_cre_before_bucket[npe];
    double elapsed_gyr = 0.0;
    double requested_t_acc_direct_gyr = 0.0;
    int world_size = 1;
    int isnap, itr, i, k;
    int start_snapshot = 0;
    int input_read_order = 0;
    int selected_runs = 0;
    int selected_offset_runs = 0;
    int selected_min_run_len = 0;
    int selected_max_run_len = 0;
    int selected_max_gap = 0;
    long int selected_offset_min = 0;
    long int selected_offset_max = 0;
    long int selected_offset_span = 0;
    char hostname[256];
    int ierr = 1;

    memset(&out, 0, sizeof(out));
    memset(&stats, 0, sizeof(stats));
    memset(&chunk, 0, sizeof(chunk));
    memset(&wall, 0, sizeof(wall));
    stats.last_synch_snapshot = -1;
    tracer_fp_map_reset(&out.ne);
    tracer_fp_map_reset(&out.crp);
    tracer_fp_map_reset(&out.epssyn);
    tracer_fp_map_reset(&out.epsic);
    tracer_fp_map_reset(&out.epsgamma);
    tracer_fp_map_reset(&out.epsnu);
    tracer_fp_tile_reset(&out.ne_tile);
    tracer_fp_tile_reset(&out.np_tile);
    tracer_fp_tile_reset(&out.epssyn_tile);
    tracer_fp_tile_reset(&out.epsic_tile);
    tracer_fp_tile_reset(&out.epsgamma_tile);
    tracer_fp_tile_reset(&out.epsnu_tile);

    if (ntracer <= 0) {
        if (!log_root) return 0;
        printf(TRACER_FP_PROGNAME " driver\n");
        printf("  physics mode        : tracer snapshots + N_subloop buckets\n");
        printf("  background mode     : %s\n", tracer_fp_background_mode_name(background_mode));
        printf("  input mode          : %s\n", tracer_fp_input_mode_name(input_mode));
        printf("  local ntracer       : 0\n");
        printf("  global ntracer      : %d\n", ntracer_global);
        printf("  local id range      : [%ld, %ld)\n", tracer_start, tracer_start + (long int)source_ntracer);
        printf("  status              : idle rank\n");
        return 0;
    }

    if (tracer_ids == 0 || source_offsets == 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": missing tracer selection arrays\n");
        return 1;
    }
    for (itr = 0; itr < ntracer; itr++) {
        if (tracer_ids[itr] == debug_target_global) {
            debug_target_local = itr;
            break;
        }
    }

    memset(&crp_grid, 0, sizeof(crp_grid));
    memset(&cre_grid, 0, sizeof(cre_grid));
    memset(&times, 0, sizeof(times));
    memset(&cpu_workspace, 0, sizeof(cpu_workspace));
    tracer_fp_synch_data_reset(&synch);
    tracer_step_reset(&step);
    tracer_bucket_reset(&bucket);
    tracer_data_history_reset(&bg_history);
    memset(bg_slots, 0, sizeof(bg_slots));
    memset(&hdf5_meta, 0, sizeof(hdf5_meta));
    tracer_data_storage_reset(&raw_storage);
    memset(raw_slots, 0, sizeof(raw_slots));
    memset(&dsa_grid, 0, sizeof(dsa_grid));
    out.dir[0] = '\0';
    out.checkpoint_dir[0] = '\0';
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        strncpy(hostname, "unknown", sizeof(hostname) - 1);
        hostname[sizeof(hostname) - 1] = '\0';
    } else {
        hostname[sizeof(hostname) - 1] = '\0';
    }
    wall.t0_ms = now_ms();
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Restart/checkpoint require file-backed output so resumed runs can keep
     * appending by absolute snapshot slot without rebuilding prior outputs. */
    if ((restart_enabled || checkpoint_enabled) &&
        !(use_mapped_output || use_tile_output)) {
        fprintf(stderr,
                TRACER_FP_PROGNAME ": checkpoint/restart requires file_output_mode=write and write_buffer_mode=mapped|tile\n");
        return 1;
    }


    init_test_params();
    if (params_file != 0 && *params_file != '\0') {
        read_param_file_noMPI(params_file);
    }
    if (finalize_dpp_mode_config(0) != SUCCESS) {
        return 1;
    }
    has_crp_emission = (seed_cr_species != SEED_CR_SPECIES_ELECTRON_ONLY) ? 1 : 0;
    ic_enabled = tracer_write_ic_output;
    gamma_enabled = has_crp_emission && tracer_write_gamma_output;
    neutrino_enabled = has_crp_emission && tracer_write_neutrino_output;
    write_crp_output = (tracer_write_crp_output != 0);
    chunk.chunk_snapshots = tracer_write_buffer_chunk_snapshots;
    if (chunk.chunk_snapshots < 1) chunk.chunk_snapshots = 1;
    if (chunk.chunk_snapshots > nsnap) chunk.chunk_snapshots = nsnap;
    chunk.use_mapped_chunk = use_mapped_output && chunk.chunk_snapshots > 1;

    if (log_root) {
        printf("  setup               : allocating tracer work arrays (ntracer=%d, nsnap=%d, nfreq=%d)\n",
               ntracer, nsnap, nfreq);
        printf("  setup               : synch theta bins = %d\n", tracer_synch_ntheta_pitch);
        printf("  setup               : magnetic field mode = %s\n", bfield_mode_spec);
        if (eta_dpp_cap > 0.0) {
            printf("  setup               : eta_dpp_cap = %.6g\n", eta_dpp_cap);
        } else {
            printf("  setup               : eta_dpp_cap = off\n");
        }
        if (write_output_files) {
            printf("  setup               : write buffer mode = %s\n",
                   tracer_fp_write_buffer_mode_name(write_buffer_mode));
            printf("  setup               : write CRP output = %s\n",
                   write_crp_output ? "yes" : "no");
            printf("  setup               : write IC output = %s\n",
                   ic_enabled ? "yes" : "no");
            printf("  setup               : write gamma output = %s\n",
                   gamma_enabled ? "yes" : "no");
            printf("  setup               : write neutrino output = %s\n",
                   neutrino_enabled ? "yes" : "no");
            printf("  setup               : output units = %s\n",
                   tracer_output_per_cc ? "physical_per_cc" : "per_tracer");
            if (use_mapped_output) {
                printf("  setup               : write buffer chunk snapshots = %d\n",
                       chunk.chunk_snapshots);
            }
        }
        fflush(stdout);
    }

    requested_t_acc_direct_gyr = t_acc_direct_gyr;
    if (tracer_fp_parse_dsa_injection_mode(DSAInjectionModeSpec, &dsa_injection_mode) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid DSAInjectionMode '%s'\n",
                DSAInjectionModeSpec);
        goto cleanup;
    }
    if (tracer_fp_parse_dsa_reacc_mode(DSAReaccModeSpec, &dsa_reacc_mode) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid DSAReaccMode '%s'\n",
                DSAReaccModeSpec);
        goto cleanup;
    }
    if (dsa_reacc_mode != TRACER_DSA_REACC_OFF &&
        dsa_injection_mode == TRACER_DSA_INJECTION_OFF) {
        fprintf(stderr, TRACER_FP_PROGNAME ": DSAReaccMode requires DSAInjectionMode=tracer_state or tracer_source\n");
        goto cleanup;
    }
    dsa_zero_seed_proton_note =
        ((dsa_injection_mode != TRACER_DSA_INJECTION_OFF ||
          dsa_reacc_mode != TRACER_DSA_REACC_OFF) &&
         seed_cr_species == SEED_CR_SPECIES_ELECTRON_ONLY) ? 1 : 0;
    if (log_root) {
        printf("  physics             :\n");
        print_physics_setup(dsa_injection_mode, dsa_reacc_mode,
                            dsa_zero_seed_proton_note,
                            requested_t_acc_direct_gyr);
        fflush(stdout);
    }

    /*  Set up momentum bins  */
    momentum_bin(&crp_grid, &cre_grid, beta_p, gamma2e);
    if (log_root) {
        print_seed_cr_setup(&crp_grid, &cre_grid);
        fflush(stdout);
    }

    /* Allocate every per-rank work buffer up front so the snapshot loop can
     * stay focused on bucket assembly, solve, and output. */
    dt_snap = (double *)calloc((size_t)nsnap, sizeof(double));
    z_snap = (double *)calloc((size_t)nsnap, sizeof(double));
    if (tracer_data_history_alloc(&bg_history, bg_slot_storage_size) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": background history allocation failure\n");
        goto cleanup;
    }
    rad_ic_zero = (double *)calloc((size_t)ntracer * (size_t)npe, sizeof(double));
    rad_ic_m1_zero = (double *)calloc((size_t)ntracer, sizeof(double));
    rad_ic_p1_zero = (double *)calloc((size_t)ntracer, sizeof(double));
    crp_state = (double *)calloc(np_batch, sizeof(double));
    cre_state = (double *)calloc(npe_batch, sizeof(double));
    qpi_batch = (double *)calloc(np_batch, sizeof(double));
    qepri_batch = (double *)calloc(npe_batch, sizeof(double));
    tracer_mass = (double *)calloc((size_t)ntracer, sizeof(double));
    fqe_flat = (double *)calloc((size_t)npe * (size_t)np, sizeof(double));
    fic_flat = ic_enabled ? (double *)calloc((size_t)bins_IC * (size_t)npe, sizeof(double)) : 0;
    fga_flat = gamma_enabled ? (double *)calloc((size_t)bins_gamma * (size_t)np, sizeof(double)) : 0;
    fnu_flat = neutrino_enabled ? (double *)calloc((size_t)bins_nu * (size_t)np, sizeof(double)) : 0;
    np_min_qe = (int *)calloc((size_t)npe, sizeof(int));
    if (tracer_fp_synch_data_alloc(&synch, nfreq) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": synch allocation failure\n");
        goto cleanup;
    }
    eps_syn = (double *)calloc((size_t)ntracer * (size_t)nfreq, sizeof(double));
    eps_ic = ic_enabled ? (double *)calloc((size_t)ntracer * (size_t)bins_IC, sizeof(double)) : 0;
    eps_gamma = gamma_enabled ? (double *)calloc((size_t)ntracer * (size_t)bins_gamma, sizeof(double)) : 0;
    eps_nu = neutrino_enabled ? (double *)calloc((size_t)ntracer * (size_t)bins_nu, sizeof(double)) : 0;
    b_dyn = (double *)calloc((size_t)ntracer, sizeof(double));
    logb = (double *)calloc((size_t)ntracer, sizeof(double));
    if (tracer_step_alloc(&step, ntracer) != 0 ||
        tracer_bucket_alloc(&bucket, ntracer) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": step/bucket allocation failure\n");
        goto cleanup;
    }
    runtime_snapshot_nsub_local = (long long *)calloc((size_t)((nsnap > 0) ? nsnap : 1),
                                                      sizeof(long long));
    runtime_snapshot_target_nsub_local = (long long *)calloc((size_t)((nsnap > 0) ? nsnap : 1),
                                                             sizeof(long long));
    if (log_root) {
        runtime_snapshot_nsub_global = (long long *)calloc((size_t)((nsnap > 0) ? nsnap : 1),
                                                           sizeof(long long));
        runtime_snapshot_target_nsub_global = (long long *)calloc((size_t)((nsnap > 0) ? nsnap : 1),
                                                                  sizeof(long long));
    }
    out.tracer_id_core = (int *)calloc((size_t)ntracer, sizeof(int));
    if (use_buffered_output) {
        out.ne_buf = allocate2DArray(ntracer, (nsnap + 1) * npe);
        out.np_buf = write_crp_output ? allocate2DArray(ntracer, (nsnap + 1) * np) : 0;
        out.epssyn_buf = allocate2DArray(ntracer, nsnap * nfreq);
        out.epsic_buf = ic_enabled ? allocate2DArray(ntracer, nsnap * bins_IC) : 0;
        out.epsgamma_buf = gamma_enabled ? allocate2DArray(ntracer, nsnap * bins_gamma) : 0;
        out.epsnu_buf = neutrino_enabled ? allocate2DArray(ntracer, nsnap * bins_nu) : 0;
    }
    if (dt_snap == 0 || z_snap == 0 ||
        rad_ic_zero == 0 || rad_ic_m1_zero == 0 || rad_ic_p1_zero == 0 ||
        crp_state == 0 || cre_state == 0 ||
        qpi_batch == 0 || qepri_batch == 0 || tracer_mass == 0 ||
        fqe_flat == 0 || np_min_qe == 0 ||
        (ic_enabled && fic_flat == 0) ||
        (gamma_enabled && fga_flat == 0) ||
        (neutrino_enabled && fnu_flat == 0) ||
        eps_syn == 0 ||
        (ic_enabled && eps_ic == 0) ||
        (gamma_enabled && eps_gamma == 0) ||
        (neutrino_enabled && eps_nu == 0) ||
        b_dyn == 0 || logb == 0 ||
        runtime_snapshot_nsub_local == 0 || runtime_snapshot_target_nsub_local == 0 ||
        (log_root && (runtime_snapshot_nsub_global == 0 || runtime_snapshot_target_nsub_global == 0)) ||
        out.tracer_id_core == 0 ||
        (use_buffered_output &&
         (out.ne_buf == 0 || (write_crp_output && out.np_buf == 0) ||
          out.epssyn_buf == 0 ||
          (ic_enabled && out.epsic_buf == 0) ||
          (gamma_enabled && out.epsgamma_buf == 0) ||
          (neutrino_enabled && out.epsnu_buf == 0)))) {
        fprintf(stderr, TRACER_FP_PROGNAME ": allocation failure\n");
        goto cleanup;
    }

    if (dsa_injection_mode == TRACER_DSA_INJECTION_TRACER_STATE ||
        dsa_injection_mode == TRACER_DSA_INJECTION_TRACER_SOURCE ||
        dsa_reacc_mode != TRACER_DSA_REACC_OFF) {
        if (dsa_init_grid(&dsa_grid, &crp_grid, &cre_grid, -3.0, -3.0) == 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to initialize DSA grid\n");
            goto cleanup;
        }
    }


    /*  MHD data of initial & next snapshots (for interpolation)  */
    tracer_bind_background_slot(ntracer, 0, &bg_history, &bg_slots[0]);
    tracer_bind_background_slot(ntracer, 1, &bg_history, &bg_slots[1]);


    /* Build the initial background window and tracer metadata once before the
     * snapshot loop. HDF5 mode populates the raw/background rings; synthetic
     * mode fabricates the same state directly in memory. */
    if (input_mode == TRACER_FP_INPUT_HDF5) {
        selected_runs = tracer_fp_count_selected(source_offsets, ntracer);
        tracer_fp_selected_offset_stats(source_offsets, ntracer,
                                        0,
                                        &selected_offset_runs,
                                        &selected_min_run_len,
                                        &selected_max_run_len,
                                        &selected_max_gap);
        selected_offset_bounds(tracer_start, source_offsets, ntracer,
                               &selected_offset_min,
                               &selected_offset_max,
                               &selected_offset_span);
        if (log_root) {
            printf("  data load           : initializing HDF5 metadata and background ring\n");
            fflush(stdout);
        }
        if (open_input_read_log(&out, mpi_rank, local_rank, hostname) != 0) {
            goto cleanup;
        }
        if (DSADebug && dsa_reacc_mode != TRACER_DSA_REACC_OFF &&
            open_dsa_reacc_debug_log(&out, mpi_rank) != 0) {
            goto cleanup;
        }
        if (tracer_fp_hdf5_init(params_file, ntracer, ntracer_global, nsnap,
                                      background_mode, &raw_storage, raw_slots,
                                      &hdf5_meta, dt_snap, z_snap) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to initialize HDF5 tracer background setup\n");
            goto cleanup;
        }
        if (log_root &&
            tracer_fp_write_hdf5_layout_log(out.dir, nsnp_i) != 0) {
            fprintf(stderr,
                    TRACER_FP_PROGNAME ": warning failed to write input_hdf5_layout.tsv for snapshot %d\n",
                    nsnp_i);
        }

        if (background_mode == TRACER_FP_BACKGROUND_FROZEN) {
            if (log_root) {
                printf("  data load           : reading frozen snapshot nsnp=%d\n", nsnp_i);
                fflush(stdout);
            }
            if (load_hdf5_raw_snapshot_logged(&hdf5_meta, nsnp_i, 0,
                                              "initial_frozen",
                                              tracer_start, source_offsets, ntracer,
                                              &raw_slots[0], &times,
                                              out.input_read_fp, mpi_rank,
                                              local_rank, hostname,
                                              &input_read_order,
                                              selected_runs,
                                              selected_offset_runs,
                                              selected_min_run_len,
                                              selected_max_run_len,
                                              selected_max_gap,
                                              selected_offset_min,
                                              selected_offset_max,
                                              selected_offset_span,
                                              z_snap, nsnap) != 0) {
                goto cleanup;
            }
            {
                const double t0_bg = now_ms();
                if (tracer_prepare_background_from_raw(ntracer,
                                                       &raw_slots[0], &raw_slots[0], &raw_slots[0],
                                                       z_snap[0], z_snap[0], z_snap[0],
                                                       0, 0,
                                                       &bg_slots[bg_curr_slot]) != 0) {
                    goto cleanup;
                }
                times.bg_prepare_ms += now_ms() - t0_bg;
            }
            tracer_copy_background_slot(ntracer, &bg_slots[bg_next_slot], &bg_slots[bg_curr_slot]);
        } else {
            if (log_root) {
                printf("  data load           : reading initial evolving snapshots nsnp=%d..%d\n",
                       nsnp_i, nsnp_i + ((nsnap > 2) ? 2 : (nsnap - 1)));
                fflush(stdout);
            }
            if (load_hdf5_raw_snapshot_logged(&hdf5_meta, nsnp_i + 0, 0,
                                              "initial_prev",
                                              tracer_start, source_offsets, ntracer,
                                              &raw_slots[raw_prev_slot], &times,
                                              out.input_read_fp, mpi_rank,
                                              local_rank, hostname,
                                              &input_read_order,
                                              selected_runs,
                                              selected_offset_runs,
                                              selected_min_run_len,
                                              selected_max_run_len,
                                              selected_max_gap,
                                              selected_offset_min,
                                              selected_offset_max,
                                              selected_offset_span,
                                              z_snap, nsnap) != 0) {
                goto cleanup;
            }
            if (nsnap > 1 &&
                load_hdf5_raw_snapshot_logged(&hdf5_meta, nsnp_i + 1, 1,
                                              "initial_curr",
                                              tracer_start, source_offsets, ntracer,
                                              &raw_slots[raw_curr_slot], &times,
                                              out.input_read_fp, mpi_rank,
                                              local_rank, hostname,
                                              &input_read_order,
                                              selected_runs,
                                              selected_offset_runs,
                                              selected_min_run_len,
                                              selected_max_run_len,
                                              selected_max_gap,
                                              selected_offset_min,
                                              selected_offset_max,
                                              selected_offset_span,
                                              z_snap, nsnap) != 0) {
                goto cleanup;
            }
            if (nsnap > 2 &&
                load_hdf5_raw_snapshot_logged(&hdf5_meta, nsnp_i + 2, 2,
                                              "initial_next",
                                              tracer_start, source_offsets, ntracer,
                                              &raw_slots[raw_next_slot], &times,
                                              out.input_read_fp, mpi_rank,
                                              local_rank, hostname,
                                              &input_read_order,
                                              selected_runs,
                                              selected_offset_runs,
                                              selected_min_run_len,
                                              selected_max_run_len,
                                              selected_max_gap,
                                              selected_offset_min,
                                              selected_offset_max,
                                              selected_offset_span,
                                              z_snap, nsnap) != 0) {
                goto cleanup;
            }
            {
                const double t0_bg = now_ms();
                if (tracer_prepare_background_from_raw(ntracer,
                                                       &raw_slots[raw_prev_slot],
                                                       &raw_slots[raw_prev_slot],
                                                       (nsnap > 1) ? &raw_slots[raw_curr_slot]
                                                                   : &raw_slots[raw_prev_slot],
                                                       z_snap[0], z_snap[0],
                                                       (nsnap > 1) ? z_snap[1] : z_snap[0],
                                                       0, 0,
                                                       &bg_slots[bg_curr_slot]) != 0) {
                    goto cleanup;
                }
                times.bg_prepare_ms += now_ms() - t0_bg;
            }
            if (nsnap > 1) {
                const double t0_bg = now_ms();
                if (tracer_prepare_background_from_raw(ntracer,
                                                       &raw_slots[raw_prev_slot],
                                                       &raw_slots[raw_curr_slot],
                                                       (nsnap > 2) ? &raw_slots[raw_next_slot]
                                                                   : &raw_slots[raw_curr_slot],
                                                       z_snap[0], z_snap[1],
                                                       (nsnap > 2) ? z_snap[2] : z_snap[1],
                                                       1, (nsnap > 2),
                                                       &bg_slots[bg_next_slot]) != 0) {
                    goto cleanup;
                }
                times.bg_prepare_ms += now_ms() - t0_bg;
            } else {
                tracer_copy_background_slot(ntracer, &bg_slots[bg_next_slot], &bg_slots[bg_curr_slot]);
            }
        }
        if (log_root) {
            printf("  data load           : reading initial tracer masses\n");
            fflush(stdout);
        }
        {
            const double t0_mass = now_ms();
            if (load_tracer_initial_mass_hdf5(tracer_mass, ntracer, tracer_start, source_offsets, ntracer_global,
                                              params_file, z_snap[0]) != 0) {
                goto cleanup;
            }
            times.tracer_mass_ms += now_ms() - t0_mass;
        }
    } else {
        if (log_root) {
            printf("  data load           : generating synthetic tracer timeline/background\n");
            fflush(stdout);
        }
        for (itr = 0; itr < ntracer; itr++) {
            const long int global_tracer = tracer_ids[itr];
            const double frac = (ntracer_global > 1)
                ? (double)global_tracer / (double)(ntracer_global - 1) : 0.0;
            tracer_mass[itr] =
                pow(10.0, log10(3.0e7) + frac * (log10(3.0e9) - log10(3.0e7)));
        }
        tracer_fp_fill_timeline_synthetic(dt_snap, z_snap, nsnap);
        if (tracer_fp_fill_background_snapshot_synthetic(0, z_snap, ntracer,
                                                         tracer_ids, ntracer_global, nsnap,
                                                         &bg_slots[bg_curr_slot]) != 0) {
            goto cleanup;
        }
        if (background_mode == TRACER_FP_BACKGROUND_FROZEN || nsnap == 1) {
            tracer_copy_background_slot(ntracer, &bg_slots[bg_next_slot], &bg_slots[bg_curr_slot]);
        } else if (tracer_fp_fill_background_snapshot_synthetic(1, z_snap, ntracer,
                                                                tracer_ids, ntracer_global, nsnap,
                                                                &bg_slots[bg_next_slot]) != 0) {
            goto cleanup;
        }
    }
    if (params_file != 0 && *params_file != '\0') {
        (void)read_param_file_noMPI(params_file);
    }
    if (log_root) {
        printf("  data load           : complete\n");
        fflush(stdout);
    }

    /* Output setup owns file paths, mapped/tile handles, and any chunk buffers
     * needed by the chosen write mode. */
    {
        TracerFpOutputs output_files;
        TracerFpOutputCfg output_cfg;

        memset(&output_files, 0, sizeof(output_files));
        memset(&output_cfg, 0, sizeof(output_cfg));
        output_files.ne_output = &out.ne;
        output_files.np_output = &out.crp;
        output_files.epssyn_output = &out.epssyn;
        output_files.epsic_output = &out.epsic;
        output_files.epsgamma_output = &out.epsgamma;
        output_files.epsnu_output = &out.epsnu;
        output_files.ne_tile_output = &out.ne_tile;
        output_files.np_tile_output = &out.np_tile;
        output_files.epssyn_tile_output = &out.epssyn_tile;
        output_files.epsic_tile_output = &out.epsic_tile;
        output_files.epsgamma_tile_output = &out.epsgamma_tile;
        output_files.epsnu_tile_output = &out.epsnu_tile;
        output_files.ne_buffer_core = &out.ne_buf;
        output_files.np_buffer_core = &out.np_buf;
        output_files.epssyn_buffer_core = &out.epssyn_buf;
        output_files.epsic_buffer_core = &out.epsic_buf;
        output_files.epsgamma_buffer_core = &out.epsgamma_buf;
        output_files.epsnu_buffer_core = &out.epsnu_buf;
        output_files.ne_chunk_core = &out.ne_chunk;
        output_files.np_chunk_core = &out.np_chunk;
        output_files.epssyn_chunk_core = &out.epssyn_chunk;
        output_files.epsic_chunk_core = &out.epsic_chunk;
        output_files.epsgamma_chunk_core = &out.epsgamma_chunk;
        output_files.epsnu_chunk_core = &out.epsnu_chunk;
        output_files.bucketstats_top_fp = &out.bucket_top_fp;
        output_files.bucketstats_rank_fp = &out.bucket_rank_fp;
        output_cfg.output_dir = out.dir;
        output_cfg.output_dir_size = sizeof(out.dir);
        output_cfg.checkpoint_dir = out.checkpoint_dir;
        output_cfg.checkpoint_dir_size = sizeof(out.checkpoint_dir);
        output_cfg.tracer_id_core = out.tracer_id_core;
        output_cfg.tracer_ids = tracer_ids;
        output_cfg.ntracer = ntracer;
        output_cfg.nsnap = nsnap;
        output_cfg.nfreq = nfreq;
        output_cfg.mpi_rank = mpi_rank;
        output_cfg.write_output_files = write_output_files;
        output_cfg.write_crp_output = write_crp_output;
        output_cfg.ic_enabled = ic_enabled;
        output_cfg.gamma_enabled = gamma_enabled;
        output_cfg.neutrino_enabled = neutrino_enabled;
        output_cfg.use_tile_output = use_tile_output;
        output_cfg.use_mapped_output = use_mapped_output;
        output_cfg.use_mapped_chunk = chunk.use_mapped_chunk;
        output_cfg.mapped_chunk_snapshots = chunk.chunk_snapshots;
        output_cfg.bucket_stats_only = bucket_stats_only;
        output_cfg.checkpoint_enabled = checkpoint_enabled;
        output_cfg.restart_enabled = restart_enabled;
        output_cfg.configured_output_dir = output_dir;
        output_cfg.log_root = log_root;
        if (tracer_fp_outputs_init(&output_files, &output_cfg) != 0) {
            goto cleanup;
        }
    }

                if (log_root && debug_target_global >= 0) {
                    fprintf(stderr,
                    "[debug electron grid] pemin=%e CRe_p0=%e CRe_p1=%e CRe_p2=%e CRe_pN=%e npe=%d\n",
                    pemin,
                    cre_grid.p[0],
                    cre_grid.p[1],
                    cre_grid.p[2],
                    cre_grid.p[npe - 1],
                    npe);
                }


    /* Set initial CR spectrum */
    tracer_fp_init_state(crp_state, cre_state, qpi_batch, qepri_batch,
                         &crp_grid, &cre_grid, tracer_mass, z_snap[0],
                         ntracer, tracer_start, ntracer_global, debug_target_global);


    /* Restart restores the CR state and then rebuilds the two-snapshot
     * background window so the rolling solve can resume mid-stream. */
    if (restart_enabled) {
        int completed_snapshot = -1;
        const double t0_restart = now_ms();
        TracerFpCkptRead checkpoint_read;
        TracerFpRestartWin restart_window;

        memset(&checkpoint_read, 0, sizeof(checkpoint_read));
        checkpoint_read.restart_dir = tracer_restart_dir;
        checkpoint_read.mpi_rank = mpi_rank;
        checkpoint_read.world_size = world_size;
        checkpoint_read.ntracer = ntracer;
        checkpoint_read.ntracer_global = ntracer_global;
        checkpoint_read.nsnap = nsnap;
        checkpoint_read.nfreq = nfreq;
        checkpoint_read.tracer_start = tracer_start;
        checkpoint_read.tracer_ids = out.tracer_id_core;
        checkpoint_read.crp_state = crp_state;
        checkpoint_read.cre_state = cre_state;
        checkpoint_read.completed_snapshot_out = &completed_snapshot;
        if (tracer_fp_ckpt_read(&checkpoint_read) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to read restart checkpoint from '%s'\n",
                    tracer_restart_dir);
            goto cleanup;
        }
        times.restart_ms += now_ms() - t0_restart;
        start_snapshot = completed_snapshot + 1;
        if (log_root) {
            printf("  restart             : loaded checkpoint from %s (completed snapshot=%d, start=%d)\n",
                   tracer_restart_dir, completed_snapshot + 1, start_snapshot + 1);
            fflush(stdout);
        }
        memset(&restart_window, 0, sizeof(restart_window));
        restart_window.start_snapshot = start_snapshot;
        restart_window.nsnap = nsnap;
        restart_window.input_mode = input_mode;
        restart_window.background_mode = background_mode;
        restart_window.ntracer = ntracer;
        restart_window.ntracer_global = ntracer_global;
        restart_window.tracer_start = tracer_start;
        restart_window.source_offsets = source_offsets;
        restart_window.tracer_ids = tracer_ids;
        restart_window.z_snap = z_snap;
        restart_window.hdf5_meta = &hdf5_meta;
        restart_window.raw_slots = raw_slots;
        restart_window.raw_prev_slot = &raw_prev_slot;
        restart_window.raw_curr_slot = &raw_curr_slot;
        restart_window.raw_next_slot = &raw_next_slot;
        restart_window.bg_slots = bg_slots;
        restart_window.bg_curr_slot = bg_curr_slot;
        restart_window.bg_next_slot = bg_next_slot;
        if (tracer_fp_restart_bg(&restart_window) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to prepare restart background window\n");
            goto cleanup;
        }
    }
    if (write_output_files && !restart_enabled &&
        tracer_fp_should_emit_snapshot(schedule, 0, nsnap)) {
        /* Match tracer_and_emission buffer semantics: slot 0 stores the
         * initial CR state before the snapshot loop advances anything. */
        if (use_tile_output) {
            if (tracer_fp_tile_write_cr(&out.ne_tile, &out.np_tile, ntracer, 0,
                                        cre_state, crp_state,
                                        raw_slots[raw_curr_slot].rho, tracer_mass,
                                        tracer_output_per_cc, write_crp_output) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": failed to write tile initial CR state (%s)\n",
                        strerror(errno));
                goto cleanup;
            }
        } else if (use_mapped_output) {
            if (chunk.use_mapped_chunk) {
                chunk.start_snapshot = 0;
                chunk.cr_base_slot = 0;
                chunk.cr_slots = 1;
                populate_output_buffers(out.ne_chunk, out.np_chunk, ntracer, 0,
                                        cre_state, crp_state,
                                        raw_slots[raw_curr_slot].rho, tracer_mass,
                                        tracer_output_per_cc);
            } else {
                if (tracer_fp_write_cr_map(&out.ne, &out.crp, ntracer, 0,
                                           cre_state, crp_state,
                                           raw_slots[raw_curr_slot].rho, tracer_mass,
                                           tracer_output_per_cc) != 0) {
                    fprintf(stderr, TRACER_FP_PROGNAME ": failed to write mapped initial CR state (%s)\n",
                            strerror(errno));
                    goto cleanup;
                }
            }
        } else {
            populate_output_buffers(out.ne_buf, out.np_buf, ntracer, 0, cre_state, crp_state,
                                    raw_slots[raw_curr_slot].rho, tracer_mass,
                                    tracer_output_per_cc);
        }
    }
                if (log_root && debug_target_local >= 0) {
                    const size_t debug_e_off = (size_t)debug_target_local * (size_t)npe;
                    tracer_fp_debug_print_cre_sample("initial", debug_target_global, 0,
                                                    cre_state + debug_e_off);
    }

    /* kernels for hadronic chain */
    {
        const double t0_secondary = now_ms();
        prepare_secondary_kernel_flat(&crp_grid, &cre_grid, beta_p, fqe_flat, np_min_qe);
        times.secondary_ms += now_ms() - t0_secondary;
    }
    if (gamma_enabled) {
        prepare_gamma_kernel_flat(&crp_grid, bins_gamma, E_gamma_min, E_gamma_max, fga_flat);
    }
    if (neutrino_enabled) {
        prepare_neutrino_kernel_flat(&crp_grid, bins_nu, E_nu_min, E_nu_max, fnu_flat);
    }

    if (tracer_fp_synch_data_init_tables(&synch, nfreq, &cre_grid) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to initialize synch tables\n");
        goto cleanup;
    }

    /* The backend decision is per driver run, not per bucket, so every
     * snapshot in this invocation uses the same execution path. */
    if (backend_mode != TRACER_FP_BACKEND_CPU) {
#ifdef FP_USE_CUDA_BACKEND
        if (cuda_pipeline_is_available()) {
            use_cuda_backend = 1;
            cuda_workspace = cuda_pipeline_workspace_create();
            if (cuda_workspace == 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": CUDA workspace allocation failed\n");
                goto cleanup;
            }
        } else if (backend_mode == TRACER_FP_BACKEND_CUDA) {
            fprintf(stderr, TRACER_FP_PROGNAME ": CUDA backend unavailable in this build\n");
            goto cleanup;
        }
#else
        if (backend_mode == TRACER_FP_BACKEND_CUDA) {
            fprintf(stderr, TRACER_FP_PROGNAME ": CUDA backend requested but this binary was built without CUDA support\n");
            goto cleanup;
        }
#endif
    }

                    if (log_root) {
                        printf("  solve backend(plan) : proton:%s electron:%s\n",
                            use_cuda_backend ? planned_cuda_solver_name(np)
                                                : cpu_tridiag_solver_name(),
                            use_cuda_backend ? planned_cuda_solver_name(npe)
                                                : cpu_tridiag_solver_name());
                        fflush(stdout);
                    }


    /*  Solve snapshot loop  */
    {
        TracerFpSolveCtx solve_ctx;
        TracerFpBgWin bg_window;
        TracerFpOutState output_state;

        /* The solver takes a condensed view of the driver state: immutable
         * scalar/config data in solve_ctx, rolling background slots in
         * bg_window, and output handles/buffers in output_state. */
        memset(&solve_ctx, 0, sizeof(solve_ctx));
        memset(&bg_window, 0, sizeof(bg_window));
        memset(&output_state, 0, sizeof(output_state));
        solve_ctx.schedule = schedule;
        solve_ctx.source_offsets = source_offsets;
        solve_ctx.tracer_ids = tracer_ids;
        solve_ctx.checkpoint_dir = out.checkpoint_dir;
        solve_ctx.dt_snap = dt_snap;
        solve_ctx.z_snap = z_snap;
        solve_ctx.elapsed_gyr = &elapsed_gyr;
        solve_ctx.beta_p = beta_p;
        solve_ctx.gamma2e = gamma2e;
        solve_ctx.tracer_mass = tracer_mass;
        solve_ctx.rad_ic_zero = rad_ic_zero;
        solve_ctx.rad_ic_m1_zero = rad_ic_m1_zero;
        solve_ctx.rad_ic_p1_zero = rad_ic_p1_zero;
        solve_ctx.eps_syn = eps_syn;
        solve_ctx.eps_ic = eps_ic;
        solve_ctx.eps_gamma = eps_gamma;
        solve_ctx.eps_nu = eps_nu;
        solve_ctx.b_dyn = b_dyn;
        solve_ctx.logb = logb;
        solve_ctx.fqe_flat = fqe_flat;
        solve_ctx.fic_flat = fic_flat;
        solve_ctx.fga_flat = fga_flat;
        solve_ctx.fnu_flat = fnu_flat;
        solve_ctx.np_min_qe = np_min_qe;
        solve_ctx.qpi_batch = qpi_batch;
        solve_ctx.qepri_batch = qepri_batch;
        solve_ctx.crp_state = crp_state;
        solve_ctx.cre_state = cre_state;
        solve_ctx.tracer_id_core = out.tracer_id_core;
        solve_ctx.emitted_synch = &stats.emitted_synch;
        solve_ctx.emitted_gamma = &stats.emitted_gamma;
        solve_ctx.emitted_neutrino = &stats.emitted_neutrino;
        solve_ctx.last_synch_snap = &stats.last_synch_snapshot;
        solve_ctx.dsa_injected = &stats.total_dsa_injected;
        solve_ctx.bucket_calls = &stats.total_bucket_calls;
        solve_ctx.bucket_cells = &stats.total_bucket_cells;
        solve_ctx.bucket_min = &stats.min_bucket_size;
        solve_ctx.bucket_max = &stats.max_bucket_size;
        solve_ctx.max_buckets_per_snap = &stats.max_bucket_count;
        solve_ctx.max_nsubsteps = &stats.max_nsubsteps;
        solve_ctx.capped_tracer_snaps = &stats.capped_tracer_snapshots;
        solve_ctx.max_raw_nsub = &stats.max_raw_nsubsteps;
        solve_ctx.gpu_pipeline_calls = &stats.gpu_pipeline_calls;
        solve_ctx.gpu_pipeline_cells = &stats.gpu_pipeline_cells;
        solve_ctx.gpu_pipeline_fp_steps = &stats.gpu_pipeline_fp_steps;
        solve_ctx.gpu_pipeline_cell_steps = &stats.gpu_pipeline_cell_steps;
        solve_ctx.gpu_group_count_est = &stats.gpu_group_count_est;
        solve_ctx.max_gpu_groups_per_snap = &stats.max_gpu_groups_per_snap;
        solve_ctx.bucket_hist_counts = stats.bucket_hist_counts;
        solve_ctx.runtime_nsub = runtime_snapshot_nsub_local;
        solve_ctx.runtime_target_nsub = runtime_snapshot_target_nsub_local;
        bg_window.bg_slots = bg_slots;
        bg_window.raw_slots = raw_slots;
        bg_window.hdf5_meta = &hdf5_meta;
        bg_window.bg_curr = &bg_curr_slot;
        bg_window.bg_next = &bg_next_slot;
        bg_window.raw_prev = &raw_prev_slot;
        bg_window.raw_curr = &raw_curr_slot;
        bg_window.raw_next = &raw_next_slot;
        output_state.ne_output = &out.ne;
        output_state.np_output = &out.crp;
        output_state.epssyn_output = &out.epssyn;
        output_state.epsic_output = &out.epsic;
        output_state.epsgamma_output = &out.epsgamma;
        output_state.epsnu_output = &out.epsnu;
        output_state.ne_tile_output = &out.ne_tile;
        output_state.np_tile_output = &out.np_tile;
        output_state.epssyn_tile_output = &out.epssyn_tile;
        output_state.epsic_tile_output = &out.epsic_tile;
        output_state.epsgamma_tile_output = &out.epsgamma_tile;
        output_state.epsnu_tile_output = &out.epsnu_tile;
        output_state.ne_buffer_core = out.ne_buf;
        output_state.np_buffer_core = out.np_buf;
        output_state.epssyn_buffer_core = out.epssyn_buf;
        output_state.epsic_buffer_core = out.epsic_buf;
        output_state.epsgamma_buffer_core = out.epsgamma_buf;
        output_state.epsnu_buffer_core = out.epsnu_buf;
        output_state.ne_chunk_core = out.ne_chunk;
        output_state.np_chunk_core = out.np_chunk;
        output_state.epssyn_chunk_core = out.epssyn_chunk;
        output_state.epsic_chunk_core = out.epsic_chunk;
        output_state.epsgamma_chunk_core = out.epsgamma_chunk;
        output_state.epsnu_chunk_core = out.epsnu_chunk;
        output_state.chunk_start = &chunk.start_snapshot;
        output_state.chunk_count = &chunk.count;
        output_state.chunk_cr_base = &chunk.cr_base_slot;
        output_state.chunk_cr_slots = &chunk.cr_slots;
        output_state.bucket_top_fp = out.bucket_top_fp;
        output_state.bucket_rank_fp = out.bucket_rank_fp;
        solve_ctx.input_read_fp = out.input_read_fp;
        solve_ctx.input_read_order = &input_read_order;
        solve_ctx.dsa_reacc_debug_fp = out.dsa_reacc_debug_fp;
        solve_ctx.local_rank = local_rank;
        solve_ctx.hostname = hostname;
        solve_ctx.input_selected_count = selected_runs;
        solve_ctx.input_offset_runs = selected_offset_runs;
        solve_ctx.input_offset_min_run_len = selected_min_run_len;
        solve_ctx.input_offset_max_run_len = selected_max_run_len;
        solve_ctx.input_offset_max_gap = selected_max_gap;
        solve_ctx.input_offset_min = selected_offset_min;
        solve_ctx.input_offset_max = selected_offset_max;
        solve_ctx.input_offset_span = selected_offset_span;
        if (tracer_fp_solve(&solve_ctx,
                            &bg_window, &output_state,
                            &step, &bucket, &times, &synch, &cpu_workspace,
#ifdef FP_USE_CUDA_BACKEND
                            cuda_workspace,
#else
                            0,
#endif
                            &crp_grid, &cre_grid, &dsa_grid,
                            start_snapshot, nsnap, ntracer, ntracer_global, nfreq,
                            input_mode, background_mode, mpi_rank,
                            use_cuda_backend, file_output_mode, write_buffer_mode,
                            chunk.chunk_snapshots, debug_target_local,
                            debug_target_global, tracer_start,
                            dsa_injection_mode, dsa_reacc_mode,
                            requested_t_acc_direct_gyr) != 0) {
            goto cleanup;
        }
    }

    /* After the snapshot loop, reduce per-rank runtime counters so the root
     * summary can compare estimated and actual N_subloop work. */
    times.total_ms = times.loss_prepass_ms + times.nsub_estimate_ms +
                     times.bucket_build_ms + times.bucketstats_top_ms +
                     times.gpu_group_build_ms +
                     times.pack_ms + times.bucket_host_ms +
                     times.interp_ms + times.snapshot_prep_ms +
                     times.coeff_ms + times.secondary_ms +
                     times.solve_ms + times.synch_table_ms +
                     times.synch_ms + times.ic_ms +
                     times.gamma_ms + times.neutrino_ms +
                     times.cuda_setup_ms + times.cuda_h2d_ms +
                     times.cuda_d2h_ms + times.cuda_other_ms +
                     times.input_read_ms + times.bg_prepare_ms +
                     times.tracer_mass_ms + times.output_write_ms +
                     times.output_sync_ms + times.checkpoint_ms +
                     times.restart_ms;
    wall.t1_ms = now_ms();
    wall.ms = wall.t1_ms - wall.t0_ms;
    MPI_Allreduce(&wall.ms, &wall.max_ms, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&wall.ms, &wall.min_ms, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&wall.ms, &wall.sum_ms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    wall.mean_ms = wall.sum_ms / (double)world_size;
    MPI_Reduce(runtime_snapshot_nsub_local, runtime_snapshot_nsub_global, nsnap,
               MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(runtime_snapshot_target_nsub_local, runtime_snapshot_target_nsub_global, nsnap,
               MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    {
        static const char *const bucket_hist_labels[TRACER_FP_BUCKET_HIST_NBIN] = {
            "<32", "32-127", "128-511", "512-2047", "2048-8191", ">=8192"
        };
        long long bucket_calls_local = (long long)stats.total_bucket_calls;
        long long global_bucket_calls = 0;
        long long global_bucket_cells = 0;
        long long global_gpu_pipeline_calls = 0;
        long long global_gpu_pipeline_cells = 0;
        long long global_gpu_pipeline_fp_steps = 0;
        long long global_gpu_pipeline_cell_steps = 0;
        long long global_gpu_group_count_est = 0;
        long long global_bucket_hist_counts[TRACER_FP_BUCKET_HIST_NBIN];
        int global_max_gpu_groups_per_snap = 0;

        memset(global_bucket_hist_counts, 0, sizeof(global_bucket_hist_counts));
        MPI_Reduce(&bucket_calls_local, &global_bucket_calls, 1,
                   MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.total_bucket_cells, &global_bucket_cells, 1,
                   MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.gpu_pipeline_calls, &global_gpu_pipeline_calls, 1,
                   MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.gpu_pipeline_cells, &global_gpu_pipeline_cells, 1,
                   MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.gpu_pipeline_fp_steps, &global_gpu_pipeline_fp_steps, 1,
                   MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.gpu_pipeline_cell_steps, &global_gpu_pipeline_cell_steps, 1,
                   MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.gpu_group_count_est, &global_gpu_group_count_est, 1,
                   MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stats.max_gpu_groups_per_snap, &global_max_gpu_groups_per_snap, 1,
                   MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(stats.bucket_hist_counts, global_bucket_hist_counts,
                   TRACER_FP_BUCKET_HIST_NBIN, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (log_root) {
            long long runtime_global_sum_nsub = 0;
            long long runtime_global_target_sum_nsub = 0;
            long long estimate_global_sum_nsub = 0;
            long long estimate_global_target_sum_nsub = 0;
            TracerFpStageTimes stage_times;
            int ihist;

            tracer_fp_stage_times_from_raw(&times, load_balance_ms, &stage_times);

            for (i = 0; i < nsnap; i++) {
                runtime_global_sum_nsub += runtime_snapshot_nsub_global[i];
                runtime_global_target_sum_nsub += runtime_snapshot_target_nsub_global[i];
                if (estimate_snapshot_nsub != 0) {
                    estimate_global_sum_nsub += estimate_snapshot_nsub[i];
                }
                if (estimate_snapshot_target_nsub != 0) {
                    estimate_global_target_sum_nsub += estimate_snapshot_target_nsub[i];
                }
            }
            printf(TRACER_FP_PROGNAME " driver\n");
            printf("  target backend      : %s\n", use_cuda_backend ? "NVIDIA CUDA" : "CPU reference");
            printf("  requested backend   : %s\n", tracer_fp_backend_name(backend_mode));
            if (!use_cuda_backend) {
                printf("  CPU bucket path     : cpu workspace + thread-local OpenMP\n");
            }
            printf("  physics mode        : tracer snapshots + N_subloop buckets\n");
            printf("  background mode     : %s\n", tracer_fp_background_mode_name(background_mode));
            printf("  input mode          : %s\n", tracer_fp_input_mode_name(input_mode));
            if (dsa_injection_mode == TRACER_DSA_INJECTION_TRACER_STATE ||
                dsa_injection_mode == TRACER_DSA_INJECTION_TRACER_SOURCE ||
                dsa_reacc_mode != TRACER_DSA_REACC_OFF) {
                printf("  DSA injected tracers: %d\n", stats.total_dsa_injected);
            }
            if (input_mode == TRACER_FP_INPUT_HDF5 && params_file != 0) {
                printf("  params file         : %s\n", params_file);
            }
            printf("  radiation           : synch%s%s%s\n",
                   ic_enabled ? " + IC" : "",
                   gamma_enabled ? " + gamma" : "",
                   neutrino_enabled ? " + neutrino" : "");
            printf("  synch logB mode     : %s\n", adaptive_synch_logb ? "adaptive" : "fixed");
            printf("  synch logB range    : [%.3f, %.3f]\n",
                   tracer_synch_logb_min, tracer_synch_logb_max);
            printf("  ntracer             : %d\n", ntracer);
            printf("  global ntracer      : %d\n", ntracer_global);
            printf("  local id range      : [%ld, %ld)\n", tracer_start, tracer_start + (long int)source_ntracer);
            if (skip_heavy) {
                printf("  hetero skip heavy   : on\n");
                printf("  active bulk tracers : %d\n", ntracer);
                printf("  skipped heavy local : %d\n", source_ntracer - ntracer);
            }
            printf("  runtime steps       : %d\n", nsnap);
            printf("  nfreq               : %d\n", nfreq);
            printf("  output spec         : %s\n", output_spec);
            printf("  synch outputs made  : %d\n", stats.emitted_synch);
            if (ic_enabled) {
                printf("  IC outputs made     : %d\n", stats.emitted_synch);
            }
            if (gamma_enabled) {
                printf("  gamma outputs made  : %d\n", stats.emitted_gamma);
            }
            if (neutrino_enabled) {
                printf("  neutrino outputs made: %d\n", stats.emitted_neutrino);
            }
            if (stats.last_synch_snapshot > 0) {
                printf("  last synch snapshot : %d\n", stats.last_synch_snapshot);
            }
            printf("  bucket calls        : %d\n", stats.total_bucket_calls);
            if (stats.total_bucket_calls > 0) {
                printf("  avg bucket size     : %.2f\n",
                       (double)stats.total_bucket_cells / (double)stats.total_bucket_calls);
                printf("  min/max bucket size : %d / %d\n",
                       stats.min_bucket_size, stats.max_bucket_size);
            }
            printf("  bucket calls(global): %lld\n", global_bucket_calls);
            if (global_bucket_calls > 0) {
                printf("  avg bucket size(g)  : %.2f\n",
                       (double)global_bucket_cells / (double)global_bucket_calls);
            }
            printf("  bucket hist(global) :");
            for (ihist = 0; ihist < TRACER_FP_BUCKET_HIST_NBIN; ihist++) {
                printf(" %s=%lld", bucket_hist_labels[ihist], global_bucket_hist_counts[ihist]);
            }
            printf("\n");
            printf("  max buckets/snap    : %d\n", stats.max_bucket_count);
            printf("  max N_subloop       : %d\n", stats.max_nsubsteps);
            printf("  runtime sum_nsub    : %lld\n", runtime_global_sum_nsub);
            printf("  runtime target_nsub : %lld\n", runtime_global_target_sum_nsub);
            if (estimate_snapshot_nsub != 0) {
                printf("  estimate sum_nsub   : %lld\n", estimate_global_sum_nsub);
                if (estimate_global_sum_nsub != 0) {
                    printf("  runtime/estimate    : %.6f\n",
                           (double)runtime_global_sum_nsub / (double)estimate_global_sum_nsub);
                }
            }
            if (estimate_snapshot_target_nsub != 0) {
                printf("  estimate target_nsub: %lld\n", estimate_global_target_sum_nsub);
                if (estimate_global_target_sum_nsub != 0) {
                    printf("  runtime/est target  : %.6f\n",
                           (double)runtime_global_target_sum_nsub / (double)estimate_global_target_sum_nsub);
                }
            }
            if (tracer_nsub_max > 0) {
                int capped_unique_tracers = 0;
                for (i = 0; i < ntracer; i++) {
                    if (step.capped_flags != 0 && step.capped_flags[i]) capped_unique_tracers++;
                }
                printf("  max raw N_subloop   : %d\n", stats.max_raw_nsubsteps);
                printf("  capped tracer-snaps : %lld\n", stats.capped_tracer_snapshots);
                printf("  capped tracers      : %d\n", capped_unique_tracers);
            }
            printf("  loss prepass time   : %.3f ms\n", times.loss_prepass_ms);
            printf("  nsub estimate time  : %.3f ms\n", times.nsub_estimate_ms);
            printf("  bucket build time   : %.3f ms\n", times.bucket_build_ms);
            printf("  bucketstats top time: %.3f ms\n", times.bucketstats_top_ms);
            printf("  gpu group build time: %.3f ms\n", times.gpu_group_build_ms);
            printf("  host pack/scatter   : %.3f ms\n", times.pack_ms);
            printf("  bucket host time    : %.3f ms\n", times.bucket_host_ms);
            printf("  coeff interp time   : %.3f ms\n", times.interp_ms);
            printf("  snapshot prep time  : %.3f ms\n", times.snapshot_prep_ms);
            printf("  coeff prep time     : %.3f ms\n", times.coeff_ms);
            printf("  secondary time      : %.3f ms\n", times.secondary_ms);
            printf("  FP solve time       : %.3f ms\n", times.solve_ms);
            printf("  solve alloc time    : %.3f ms\n", times.solve_alloc_ms);
            printf("  solve rhs time      : %.3f ms\n", times.solve_rhs_ms);
            printf("  solve tridiag time  : %.3f ms\n", times.solve_tridiag_ms);
            printf("  synch table time    : %.3f ms\n", times.synch_table_ms);
            printf("  synch time          : %.3f ms\n", times.synch_ms);
            if (ic_enabled) {
                printf("  IC time             : %.3f ms\n", times.ic_ms);
            }
            printf("  input read time     : %.3f ms\n", times.input_read_ms);
            printf("  bg prepare time     : %.3f ms\n", times.bg_prepare_ms);
            printf("  tracer mass time    : %.3f ms\n", times.tracer_mass_ms);
            printf("  output write time   : %.3f ms\n", times.output_write_ms);
            printf("  output sync time    : %.3f ms\n", times.output_sync_ms);
            printf("  checkpoint time     : %.3f ms\n", times.checkpoint_ms);
            printf("  restart time        : %.3f ms\n", times.restart_ms);
            printf("  input read calls    : %d\n", times.input_read_calls);
            printf("  input selected runs : %d\n", times.input_selected_runs);
            printf("  input offset runs   : %lld\n", times.input_offset_runs);
            printf("  input min run len   : %d\n", times.input_offset_min_run_len);
            printf("  input max run len   : %d\n", times.input_offset_max_run_len);
            printf("  input max gap       : %d\n", times.input_offset_max_gap);
            printf("  output write calls  : %d\n", times.output_write_calls);
            printf("  output sync calls   : %d\n", times.output_sync_calls);
            printf("  checkpoint calls    : %d\n", times.checkpoint_calls);
            if (gamma_enabled) {
                printf("  gamma time          : %.3f ms\n", times.gamma_ms);
            }
            if (neutrino_enabled) {
                printf("  neutrino time       : %.3f ms\n", times.neutrino_ms);
            }
            printf("  stage IO            : %.3f ms\n", stage_times.io_ms);
            printf("  stage scheduling    : %.3f ms\n", stage_times.scheduling_ms);
            printf("  stage host          : %.3f ms\n", stage_times.host_ms);
            printf("  stage coeff prep    : %.3f ms\n", stage_times.coeffprep_ms);
            printf("  stage secondary     : %.3f ms\n", stage_times.secondary_ms);
            printf("  stage solve         : %.3f ms\n", stage_times.solve_ms);
            printf("  stage emission      : %.3f ms\n", stage_times.emission_ms);
            printf("  stage backend       : %.3f ms\n", stage_times.backend_ms);
            printf("  stage total(+LB)    : %.3f ms\n", stage_times.total_ms);
            if (use_cuda_backend) {
                if (global_gpu_group_count_est > 0 || stats.gpu_group_count_est > 0) {
                    printf("  GPU group est(local): %lld\n", stats.gpu_group_count_est);
                    printf("  GPU group est(global): %lld\n", global_gpu_group_count_est);
                    printf("  GPU avg buckets/group: %.2f\n",
                           (double)global_bucket_calls / (double)global_gpu_group_count_est);
                    printf("  GPU avg cells/group : %.2f\n",
                           (double)global_bucket_cells / (double)global_gpu_group_count_est);
                    printf("  GPU avg calls/group : %.2f\n",
                           (double)global_gpu_pipeline_calls / (double)global_gpu_group_count_est);
                    printf("  GPU max groups/snap : %d\n", global_max_gpu_groups_per_snap);
                }
                printf("  GPU pipeline calls  : %lld\n", stats.gpu_pipeline_calls);
                printf("  GPU pipeline calls(g): %lld\n", global_gpu_pipeline_calls);
                if (global_gpu_pipeline_calls > 0) {
                    printf("  GPU avg cells/call  : %.2f\n",
                           (double)global_gpu_pipeline_cells / (double)global_gpu_pipeline_calls);
                    printf("  GPU avg fpstep/call : %.2f\n",
                           (double)global_gpu_pipeline_fp_steps / (double)global_gpu_pipeline_calls);
                }
                if (global_gpu_pipeline_cell_steps > 0 && global_gpu_pipeline_cells > 0) {
                    printf("  GPU avg nsub/cell   : %.2f\n",
                           (double)global_gpu_pipeline_cell_steps / (double)global_gpu_pipeline_cells);
                }
                printf("  CUDA setup time     : %.3f ms\n", times.cuda_setup_ms);
                printf("  CUDA H2D time       : %.3f ms\n", times.cuda_h2d_ms);
                printf("  CUDA D2H time       : %.3f ms\n", times.cuda_d2h_ms);
                printf("  CUDA other time     : %.3f ms\n", times.cuda_other_ms);
                printf("  CUDA backend total  : %.3f ms\n", times.cuda_total_ms);
            }
            printf("  load balance time   : %.3f ms\n", load_balance_ms);
            printf("  total staged time   : %.3f ms\n", times.total_ms);
            printf("  wall time           : %.3f ms\n", wall.ms);
            printf("  wall time min/avg/max : %.3f / %.3f / %.3f ms\n",
                   wall.min_ms, wall.mean_ms, wall.max_ms);
            printf("  local throughput    : %.3f tracer/s\n",
                   (wall.ms > 0.0) ? (1.0e3 * (double)ntracer / wall.ms) : 0.0);
            printf("  global throughput   : %.3f tracer/s\n",
                   (wall.max_ms > 0.0) ? (1.0e3 * (double)ntracer_global / wall.max_ms) : 0.0);
            printf("  output files        : %s\n",
                   write_output_files ? out.dir :
                   (bucket_stats_only ? "disabled (bucket stats only)" : "disabled (debug mode)"));
        }
    }

    if (out.dir[0] != '\0') {
        TracerFpRunInfo run_artifacts;

        memset(&run_artifacts, 0, sizeof(run_artifacts));
        run_artifacts.output_dir = out.dir;
        run_artifacts.mpi_rank = mpi_rank;
        run_artifacts.log_root = log_root;
        run_artifacts.world_size = world_size;
        run_artifacts.mpi_size = mpi_size;
        run_artifacts.global_ntracer = (long int)ntracer_global;
        run_artifacts.local_ntracer = ntracer;
        run_artifacts.nsnap = nsnap;
        run_artifacts.input_mode = input_mode;
        run_artifacts.file_output_mode = file_output_mode;
        run_artifacts.integration_mode = integration_mode;
        run_artifacts.background_mode = background_mode;
        run_artifacts.omp_info = omp_info;
        run_artifacts.times = &times;
        run_artifacts.load_balance_ms = load_balance_ms;
        for (i = 0; i < nsnap; i++) {
            run_artifacts.runtime_sum_nsub_local += runtime_snapshot_nsub_local[i];
            run_artifacts.runtime_target_nsub_local += runtime_snapshot_target_nsub_local[i];
        }
        run_artifacts.wall_ms = wall.ms;
        run_artifacts.wall_ms_min = wall.min_ms;
        run_artifacts.wall_ms_mean = wall.mean_ms;
        run_artifacts.wall_ms_max = wall.max_ms;
        tracer_fp_write_artifacts(&run_artifacts,
                                      estimate_snapshot_nsub,
                                      runtime_snapshot_nsub_global,
                                      estimate_snapshot_target_nsub,
                                      runtime_snapshot_target_nsub_global);
    }
    if (log_root && !bucket_stats_only) {
        printf("  sample CRp(tracer0,bin0..2)   : %.6e %.6e %.6e\n",
               crp_state[0], crp_state[1], crp_state[2]);
        printf("  sample CRe(tracer0,bin1..3)   : %.6e %.6e %.6e\n",
               cre_state[1], cre_state[2], cre_state[3]);
    }

    if (log_root && !bucket_stats_only && debug_target_local >= 0) {
        const double *debug_cre = cre_state + (size_t)debug_target_local * (size_t)npe;
        double min_cre = 1.0e300;
        double max_cre = -1.0e300;
        int nneg = 0;
        int jmin = -1;
        int jmax = -1;
        int j;

        for (j = 1; j < npe; j++) {
            const double v = debug_cre[j];

            if (v < min_cre) {
                min_cre = v;
                jmin = j;
            }
            if (v > max_cre) {
                max_cre = v;
                jmax = j;
            }
            if (v < 0.0) {
                nneg++;
            }
        }

        printf("  debug CRe tracer%ld stats (j>=1): pe_min=%e min=%.6e(j=%d) max=%.6e(j=%d) nneg=%d/%d\n",
            debug_target_global, pemin, min_cre, jmin, max_cre, jmax, nneg, npe - 1);

        printf("  debug CRe tracer%ld bins : "
            "j1=%.6e j2=%.6e j3=%.6e j4=%.6e j5=%.6e "
            "j10=%.6e j20=%.6e j50=%.6e j80=%.6e j127=%.6e\n",
            debug_target_global,
            debug_cre[1],
            debug_cre[2],
            debug_cre[3],
            debug_cre[4],
            debug_cre[5],
            debug_cre[10],
            debug_cre[20],
            debug_cre[50],
            debug_cre[80],
            debug_cre[127]);
    }

    if (log_root && !bucket_stats_only) {
        if (stats.emitted_synch > 0) {
            printf("  sample synch(tracer0,freq0..2): %.6e %.6e %.6e\n",
                   eps_syn[0], eps_syn[1], eps_syn[2]);
        } else {
            printf("  sample synch(tracer0,freq0..2): not emitted\n");
        }
    }
    if (log_root && ic_enabled) {
        if (stats.emitted_synch > 0) {
            printf("  sample IC(tracer0,bin0..2): %.6e %.6e %.6e\n",
                   eps_ic[0], eps_ic[1], eps_ic[2]);
        } else {
            printf("  sample IC(tracer0,bin0..2): not emitted\n");
        }
    }
    if (log_root && gamma_enabled) {
        if (stats.emitted_gamma > 0) {
            printf("  sample gamma(tracer0,bin0..2): %.6e %.6e %.6e\n",
                   eps_gamma[0], eps_gamma[1], eps_gamma[2]);
        } else {
            printf("  sample gamma(tracer0,bin0..2): not emitted\n");
        }
    }
    if (log_root && neutrino_enabled) {
        if (stats.emitted_neutrino > 0) {
            printf("  sample neutrino(tracer0,bin0..2): %.6e %.6e %.6e\n",
                   eps_nu[0], eps_nu[1], eps_nu[2]);
        } else {
            printf("  sample neutrino(tracer0,bin0..2): not emitted\n");
        }
    }

    if (out.dir[0] != '\0' && out.tracer_id_core != 0) {
        if (log_root) {
            printf("  output              : writing tracer IDs to %s\n", out.dir);
            fflush(stdout);
        }
        {
            const double t0_output = now_ms();
            times.output_write_calls++;
            tracer_fp_write_ids(out.dir, ntracer, mpi_rank, out.tracer_id_core);
            times.output_write_ms += now_ms() - t0_output;
        }
    }

    if (write_output_files) {
        if (log_root) {
            printf("  output              : %s CR/synch files\n",
                   use_tile_output ? "finalizing tile" :
                   (use_mapped_output ? "finalizing mapped" : "writing buffered"));
            fflush(stdout);
        }
        {
            /* Finalization closes tile/mapped outputs or flushes buffered
             * arrays, and also emits the capped-tracer list when enabled. */
            const double t0_output = now_ms();
            TracerFpOutputs output_files;
            TracerFpOutputDone finalize_cfg;

            memset(&output_files, 0, sizeof(output_files));
            memset(&finalize_cfg, 0, sizeof(finalize_cfg));
            output_files.ne_output = &out.ne;
            output_files.np_output = &out.crp;
            output_files.epssyn_output = &out.epssyn;
            output_files.epsic_output = &out.epsic;
            output_files.epsgamma_output = &out.epsgamma;
            output_files.epsnu_output = &out.epsnu;
            output_files.ne_tile_output = &out.ne_tile;
            output_files.np_tile_output = &out.np_tile;
            output_files.epssyn_tile_output = &out.epssyn_tile;
            output_files.epsic_tile_output = &out.epsic_tile;
            output_files.epsgamma_tile_output = &out.epsgamma_tile;
            output_files.epsnu_tile_output = &out.epsnu_tile;
            output_files.ne_buffer_core = &out.ne_buf;
            output_files.np_buffer_core = &out.np_buf;
            output_files.epssyn_buffer_core = &out.epssyn_buf;
            output_files.epsic_buffer_core = &out.epsic_buf;
            output_files.epsgamma_buffer_core = &out.epsgamma_buf;
            output_files.epsnu_buffer_core = &out.epsnu_buf;
            finalize_cfg.output_dir = out.dir;
            finalize_cfg.ntracer = ntracer;
            finalize_cfg.nsnap = nsnap;
            finalize_cfg.nfreq = nfreq;
            finalize_cfg.mpi_rank = mpi_rank;
            finalize_cfg.write_crp_output = write_crp_output;
            finalize_cfg.ic_enabled = ic_enabled;
            finalize_cfg.gamma_enabled = gamma_enabled;
            finalize_cfg.neutrino_enabled = neutrino_enabled;
            finalize_cfg.use_tile_output = use_tile_output;
            finalize_cfg.use_mapped_output = use_mapped_output;
            finalize_cfg.total_capped_tracer_snapshots = stats.capped_tracer_snapshots;
            finalize_cfg.tracer_id_core = out.tracer_id_core;
            finalize_cfg.capped_flags = step.capped_flags;
            if (tracer_fp_outputs_done(&output_files, &finalize_cfg) != 0) {
                goto cleanup;
            }
            if (use_tile_output || use_mapped_output) {
                times.output_sync_ms += now_ms() - t0_output;
                times.output_sync_calls++;
            } else {
                times.output_write_ms += now_ms() - t0_output;
                times.output_write_calls++;
            }
            if (tracer_nsub_max > 0 && stats.capped_tracer_snapshots > 0) {
                times.output_write_calls++;
            }
        }
        if (log_root) {
            printf("  output              : complete\n");
            fflush(stdout);
        }
    }

    ierr = 0;

cleanup:
    if (out.input_read_fp != 0) {
        fclose(out.input_read_fp);
        out.input_read_fp = 0;
    }
    if (out.dsa_reacc_debug_fp != 0) {
        fclose(out.dsa_reacc_debug_fp);
        out.dsa_reacc_debug_fp = 0;
    }
#ifdef FP_USE_CUDA_BACKEND
    if (cuda_workspace) cuda_pipeline_workspace_destroy(cuda_workspace);
#endif
    tracer_fp_cpu_ws_free(&cpu_workspace);
    tracer_free_hdf5_meta(&hdf5_meta);
    dsa_free_grid(&dsa_grid);
    tracer_data_storage_release(&raw_storage);
    free(dt_snap); free(z_snap);
    tracer_data_history_release(&bg_history);
    free(rad_ic_zero); free(rad_ic_m1_zero); free(rad_ic_p1_zero);
    free(crp_state); free(cre_state);
    free(qpi_batch); free(qepri_batch);
    free(tracer_mass);
    free(fqe_flat); free(fic_flat); free(fga_flat); free(fnu_flat); free(np_min_qe);
    tracer_fp_synch_data_release(&synch);
    free(eps_syn); free(eps_ic); free(eps_gamma); free(eps_nu); free(b_dyn); free(logb);
    {
        TracerFpOutputs output_files;

        memset(&output_files, 0, sizeof(output_files));
        output_files.ne_output = &out.ne;
        output_files.np_output = &out.crp;
        output_files.epssyn_output = &out.epssyn;
        output_files.epsic_output = &out.epsic;
        output_files.epsgamma_output = &out.epsgamma;
        output_files.epsnu_output = &out.epsnu;
        output_files.ne_tile_output = &out.ne_tile;
        output_files.np_tile_output = &out.np_tile;
        output_files.epssyn_tile_output = &out.epssyn_tile;
        output_files.epsic_tile_output = &out.epsic_tile;
        output_files.epsgamma_tile_output = &out.epsgamma_tile;
        output_files.epsnu_tile_output = &out.epsnu_tile;
        output_files.ne_buffer_core = &out.ne_buf;
        output_files.np_buffer_core = &out.np_buf;
        output_files.epssyn_buffer_core = &out.epssyn_buf;
        output_files.epsic_buffer_core = &out.epsic_buf;
        output_files.epsgamma_buffer_core = &out.epsgamma_buf;
        output_files.epsnu_buffer_core = &out.epsnu_buf;
        output_files.ne_chunk_core = &out.ne_chunk;
        output_files.np_chunk_core = &out.np_chunk;
        output_files.epssyn_chunk_core = &out.epssyn_chunk;
        output_files.epsic_chunk_core = &out.epsic_chunk;
        output_files.epsgamma_chunk_core = &out.epsgamma_chunk;
        output_files.epsnu_chunk_core = &out.epsnu_chunk;
        output_files.bucketstats_top_fp = &out.bucket_top_fp;
        output_files.bucketstats_rank_fp = &out.bucket_rank_fp;
        tracer_fp_outputs_free(&output_files, ntracer);
    }
    free(out.tracer_id_core);
    tracer_step_release(&step);
    tracer_bucket_release(&bucket);
    free(runtime_snapshot_nsub_local); free(runtime_snapshot_nsub_global);
    free(runtime_snapshot_target_nsub_local); free(runtime_snapshot_target_nsub_global);

    return ierr;
}

int tracer_fp(int argc, char **argv)
{
    TracerFpState state;
    TracerFpSelectionState selection_state;
    int ierr;

    tracer_fp_state_init(&state);
    tracer_fp_selection_state_init(&selection_state);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &state.world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.world_size);
    state.mpi_rank = state.world_rank;
    state.mpi_size = state.world_size;

    if (tracer_fp_parse_cli(argc, argv, &state) != 0 ||
        tracer_fp_setup_prepare(&state) != 0) {
        tracer_fp_state_release(&state);
        MPI_Finalize();
        return 1;
    }

    tracer_fp_setup_print(&state);

    if (state.file_output_mode == TRACER_FP_OUTPUT_LOAD_ESTIMATE) {
        const double lb_t0 = now_ms();
        if (state.world_size == 1) {
            ierr = tracer_fp_load_est_mode(&state);
        } else {
            ierr = tracer_fp_load_est_actual_mode(&state, &selection_state);
            if (ierr == 0) {
                tracer_fp_mapping_print(&state, &selection_state);
                if (state.world_rank == 0) {
                    printf("  load_estimate       : actual-rank planning complete, exiting before FP solve\n");
                    fflush(stdout);
                }
            }
        }
        state.load_balance_ms = now_ms() - lb_t0;
        if (state.world_rank == 0) {
            if (state.world_size > 1) {
                printf("  load_estimate alloc : %.3f ms\n", state.load_estimate_alloc_ms);
                printf("  load_estimate nsub  : %.3f ms\n", state.load_estimate_nsub_ms);
                printf("  load_estimate reduce: %.3f ms\n", state.load_estimate_reduce_ms);
                printf("  load_estimate plan  : %.3f ms\n", state.load_estimate_plan_ms);
                printf("  load_estimate select: %.3f ms\n", state.load_estimate_selection_ms);
            }
            printf("  load_estimate time  : %.3f ms\n", state.load_balance_ms);
            fflush(stdout);
        }
        tracer_fp_selection_state_release(&selection_state);
        tracer_fp_state_release(&state);
        MPI_Finalize();
        return ierr;
    }
    if (tracer_fp_selection_prepare(&state, &selection_state) != 0 ||
        tracer_fp_backend_bind(&state) != 0) {
        tracer_fp_selection_state_release(&selection_state);
        tracer_fp_state_release(&state);
        MPI_Finalize();
        return 1;
    }

    tracer_fp_mapping_print(&state, &selection_state);

    {
        const TracerFpSelection selection = tracer_fp_selection_make(&state, &selection_state);
        const TracerFpConfig config = tracer_fp_config_make(&state);
        ierr = run_tracer_driver(&selection, &config);
    }
    tracer_fp_selection_state_release(&selection_state);
    tracer_fp_state_release(&state);
    MPI_Finalize();
    return ierr;
}

#ifndef TRACER_FP_NO_STANDALONE_MAIN
int main(int argc, char **argv)
{
    return tracer_fp(argc, argv);
}
#endif
