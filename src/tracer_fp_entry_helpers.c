/*
    tracer_fp_entry_helpers.c

    
*/

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "FP_Coef.h"
#include "READFILE.h"
#include "params.h"
#include "read_grid_hdf5.h"
#include "tracer_fp_background.h"
#include "tracer_fp_debug.h"
#include "tracer_fp_entry_helpers.h"
#include "tracer_fp_loadbalance.h"
#include "tracer_fp_output.h"
#include "tracer_fp_selection.h"
#include "tracer_fp_setup.h"
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

static size_t count_or_one(int n)
{
    return (size_t)((n > 0) ? n : 1);
}

static int alloc_active_buffers(TracerFpSelectionState *selection, size_t cap)
{
    selection->active_global_ids = (long int *)calloc(cap, sizeof(long int));
    selection->active_source_offsets = (int *)calloc(cap, sizeof(int));
    return (selection->active_global_ids != 0 && selection->active_source_offsets != 0) ? 0 : -1;
}

static int alloc_load_estimates(const TracerFpState *state,
                                          TracerFpSelectionState *selection)
{
    selection->sum_nsub_local =
        (long long *)calloc(count_or_one(state->local_count), sizeof(long long));
    selection->est_nsub_local =
        (long long *)calloc(count_or_one(state->nsnap), sizeof(long long));
    selection->est_target_local =
        (long long *)calloc(count_or_one(state->nsnap), sizeof(long long));
    if (state->world_rank == 0) {
        selection->est_nsub_global =
            (long long *)calloc(count_or_one(state->nsnap), sizeof(long long));
        selection->est_target_global =
            (long long *)calloc(count_or_one(state->nsnap), sizeof(long long));
    }
    if (selection->sum_nsub_local == 0 || selection->est_nsub_local == 0 ||
        selection->est_target_local == 0 ||
        (state->world_rank == 0 &&
         (selection->est_nsub_global == 0 || selection->est_target_global == 0))) {
        return -1;
    }
    return 0;
}

static int local_rank_from_env(int mpi_rank)
{
    const char *text = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    char *endptr = 0;
    long parsed;

    if (text == 0 || *text == '\0') text = getenv("SLURM_LOCALID");
    if (text == 0 || *text == '\0') text = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (text == 0 || *text == '\0') return mpi_rank;

    parsed = strtol(text, &endptr, 10);
    if (endptr == text || *endptr != '\0' || parsed < 0 || parsed > 2147483647L) {
        return mpi_rank;
    }
    return (int)parsed;
}

void tracer_fp_state_init(TracerFpState *state)
{
    memset(state, 0, sizeof(*state));
    state->nfreq = MAX_SYNCH_FREQ_BINS;
    state->params_file = "params_run00.txt";
    state->background_spec = 0;
    state->background_mode = TRACER_FP_BACKGROUND_EVOLVING;
    state->mpi_size = 1;
    state->world_size = 1;
    state->job_comm = MPI_COMM_WORLD;
    state->bound_device = -1;
    tracer_fp_init_openmp_info(&state->omp_info);
}

void tracer_fp_selection_state_init(TracerFpSelectionState *selection)
{
    memset(selection, 0, sizeof(*selection));
}

void tracer_fp_state_release(TracerFpState *state)
{
    tracer_fp_lb_plan_free(&state->lb_plan);
    tracer_fp_free_output_schedule(&state->schedule);
    if (state->job_parallel_active) MPI_Comm_free(&state->job_comm);
}

void tracer_fp_selection_state_release(TracerFpSelectionState *selection)
{
    free(selection->heavy_local_mask);
    free(selection->sum_nsub_local);
    free(selection->est_nsub_local);
    free(selection->est_target_local);
    free(selection->est_nsub_global);
    free(selection->est_target_global);
    free(selection->active_global_ids);
    free(selection->active_source_offsets);
}

int tracer_fp_parse_cli(int argc, char **argv, TracerFpState *state)
{
    if (argc > 1) state->params_file = argv[1];
    if (argc > 2) {
        state->background_spec = argv[2];
        state->background_cli_override = 1;
    }

    if (argc > 3) {
        if (state->world_rank == 0) {
            fprintf(stderr, "usage: %s [params_file] [background_mode]\n", argv[0]);
            fprintf(stderr, "  background_mode  : evolving | frozen\n");
            fprintf(stderr, "  params default   : frozen_background = 0|1\n");
            fprintf(stderr, "  file_output_mode is configured only in params\n");
        }
        return 1;
    }
    if (state->background_cli_override &&
        (strcmp(state->background_spec, "nowrite") == 0 ||
         strcmp(state->background_spec, "write") == 0 ||
         strcmp(state->background_spec, "bucketstats") == 0 ||
         strcmp(state->background_spec, "bucket_stats") == 0 ||
         strcmp(state->background_spec, "bucket") == 0 ||
         strcmp(state->background_spec, "load_estimate") == 0 ||
         strcmp(state->background_spec, "loadestimate") == 0 ||
         strcmp(state->background_spec, "estimate") == 0)) {
        if (state->world_rank == 0) {
            fprintf(stderr,
                    TRACER_FP_PROGNAME ": CLI file_output_mode override has been removed; "
                    "set file_output_mode in '%s'\n",
                    state->params_file);
        }
        return 1;
    }
    return 0;
}

int tracer_fp_setup_prepare(TracerFpState *state)
{
    const char *file_output_spec = 0;
    const char *write_buffer_spec = 0;
    const char *integration_spec = 0;

    if (read_param_file_noMPI(state->params_file) != SUCCESS) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to read params file '%s'\n",
                state->params_file);
        return 1;
    }
    state->background_spec = state->background_cli_override ?
        state->background_spec :
        ((frozen_background != 0) ? "frozen" : "evolving");
    if (tracer_fp_parse_background_mode(state->background_spec, &state->background_mode) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid background_mode '%s'\n",
                state->background_spec);
        return 1;
    }
    file_output_spec = tracer_file_output_spec;
    if (tracer_fp_parse_file_output_mode(file_output_spec, &state->file_output_mode) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid file_output_mode '%s'\n",
                file_output_spec);
        return 1;
    }
    write_buffer_spec = tracer_write_buffer_mode_spec;
    if (tracer_fp_parse_write_buffer_mode(write_buffer_spec, &state->write_buffer_mode) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid write_buffer_mode '%s'\n",
                write_buffer_spec);
        return 1;
    }
    if (N_TRACERS <= 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": params file '%s' must define N_TRACERS > 0\n",
                state->params_file);
        return 1;
    }
    if (synch_nfreq <= 0 || synch_nfreq > MAX_SYNCH_FREQ_BINS) {
        fprintf(stderr,
                TRACER_FP_PROGNAME
                ": params file '%s' must define synch_nfreq/nfreq in [1, %d]\n",
                state->params_file, MAX_SYNCH_FREQ_BINS);
        return 1;
    }
    if (tracer_fp_parse_backend_mode(tracer_backend_spec, &state->backend_mode) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid backend '%s' in '%s'\n",
                tracer_backend_spec, state->params_file);
        return 1;
    }
    if (tracer_fp_parse_input_mode(tracer_input_mode_spec, &state->input_mode) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid input_mode '%s' in '%s'\n",
                tracer_input_mode_spec, state->params_file);
        return 1;
    }
    tracer_fp_configure_openmp(openmp_enabled, openmp_threads, &state->omp_info);
    state->nfreq = synch_nfreq;
    state->output_spec = tracer_synch_output_spec;
    integration_spec = tracer_integration_mode_spec;
    if (tracer_fp_parse_integration_mode(integration_spec, &state->integration_mode) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid integration_mode '%s' in '%s'\n",
                integration_spec, state->params_file);
        return 1;
    }
    if (state->integration_mode == TRACER_FP_INTEGRATION_MULTIRATE) {
        if (state->world_rank == 0) {
            fprintf(stderr,
                    TRACER_FP_PROGNAME ": integration_mode=multirate is no longer used in tracer_fp; "
                    "falling back to quantized\n");
        }
        state->integration_mode = TRACER_FP_INTEGRATION_QUANTIZED;
    }
    if (tracer_checkpoint_interval < 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": checkpoint_interval must be >= 0\n");
        return 1;
    }
    if (load_balancing && hetero_skip_heavy) {
        if (state->world_rank == 0) {
            fprintf(stderr,
                    TRACER_FP_PROGNAME ": load_balancing=1 is incompatible with hetero_skip_heavy=1\n");
        }
        return 1;
    }
    if (tracer_fp_resolve_job_parallel(state->world_rank, state->world_size,
                                       job_count, job_index,
                                       job_parallel_enabled,
                                       &state->effective_num_run, &state->effective_run,
                                       &state->mpi_rank, &state->mpi_size,
                                       &state->job_parallel_active, &state->job_comm) != 0) {
        if (state->world_rank == 0) {
            fprintf(stderr,
                    TRACER_FP_PROGNAME ": invalid job parallel configuration "
                    "(world_size=%d, job_count=%d, job_index=%d, "
                    "job_parallel_enabled=%d, CROMA_NUM_JOBS/CROMA_JOB_SIZE)\n",
                    state->world_size, job_count, job_index, job_parallel_enabled);
        }
        return 1;
    }
    job_count = state->effective_num_run;
    job_index = state->effective_run;
    NUM_RUN = job_count;
    RUN = job_index;
    if (load_balancing && state->job_parallel_active) {
        if (state->world_rank == 0) {
            fprintf(stderr,
                    TRACER_FP_PROGNAME
                    ": load_balancing=1 is not supported with job_parallel; "
                    "disable one of them\n");
        }
        return 1;
    }
    if (load_balancing && state->mpi_size <= 1) {
        if (state->world_rank == 0) {
            printf("  load balance        : disabled automatically for mpi_size=%d\n",
                   state->mpi_size);
            fflush(stdout);
        }
        load_balancing = 0;
    }
    if (tracer_fp_run_part(N_TRACERS, job_count, job_index,
                                        &state->run_start, &state->run_count_long) != 0) {
        fprintf(stderr,
                TRACER_FP_PROGNAME
                ": invalid run partition (N_TRACERS=%ld, job_count=%d, job_index=%d)\n",
                N_TRACERS, job_count, job_index);
        return 1;
    }
    state->run_split_active = (job_count > 1 || job_index > 0);
    state->ntracer = (int)state->run_count_long;
    if (tracer_fp_runtime_steps(state->input_mode, state->background_mode,
                                            state->params_file, &state->nsnap,
                                            &state->frozen_runtime_override,
                                            &state->frozen_runtime_dt_gyr) != 0) {
        fprintf(stderr,
                TRACER_FP_PROGNAME ": failed to derive runtime steps from '%s' "
                "(input_mode=%s, background_mode=%s, nsnp_i=%d, nsnp_f=%d)\n",
                state->params_file, tracer_fp_input_mode_name(state->input_mode),
                tracer_fp_background_mode_name(state->background_mode), nsnp_i, nsnp_f);
        return 1;
    }
    if (tracer_debug_max_snapshots > 0 && state->nsnap > tracer_debug_max_snapshots) {
        state->nsnap = tracer_debug_max_snapshots;
    }
    if (tracer_fp_parse_output_schedule(state->output_spec, state->nsnap, &state->schedule) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid synch_output_spec '%s'\n",
                state->output_spec);
        return 1;
    }
    if (state->schedule.physical_min >= 0 && state->schedule.physical_max >= 0 &&
        state->schedule.physical_min > state->schedule.physical_max) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid output nsnp range [%d, %d]\n",
                state->schedule.physical_min, state->schedule.physical_max);
        return 1;
    }
    if (output_dir[0] != '\0') {
        snprintf(state->output_dir, sizeof(state->output_dir), "%s", output_dir);
    } else {
        snprintf(state->output_dir, sizeof(state->output_dir), "tracer_fp_gpu_output");
    }
    return 0;
}

void tracer_fp_setup_print(const TracerFpState *state)
{
    if (state->world_rank != 0) return;
    printf(TRACER_FP_PROGNAME " setup\n");
    printf("  params file         : %s\n", state->params_file);
    printf("  file output mode    : %s\n", tracer_fp_file_output_mode_name(state->file_output_mode));
    printf("  write buffer mode   : %s\n", tracer_fp_write_buffer_mode_name(state->write_buffer_mode));
    printf("  write CRP output    : %s\n", tracer_write_crp_output ? "yes" : "no");
    printf("  write gamma output  : %s\n", tracer_write_gamma_output ? "yes" : "no");
    printf("  write neutrino output: %s\n", tracer_write_neutrino_output ? "yes" : "no");
    printf("  output units        : %s\n", tracer_output_per_cc ? "physical_per_cc" : "per_tracer");
    if (state->write_buffer_mode == TRACER_FP_WRITE_BUFFER_MAPPED) {
        printf("  write chunk snaps   : %d\n", tracer_write_buffer_chunk_snapshots);
    }
    printf("  background mode     : %s\n", tracer_fp_background_mode_name(state->background_mode));
    printf("  magnetic field mode : %s\n", bfield_mode_spec);
    printf("  input mode          : %s\n", tracer_fp_input_mode_name(state->input_mode));
    if (eta_dpp_cap > 0.0) {
        printf("  eta_dpp_cap         : %.6g\n", eta_dpp_cap);
    } else {
        printf("  eta_dpp_cap         : off\n");
    }
    printf("  synch theta bins    : %d\n", tracer_synch_ntheta_pitch);
    if (state->file_output_mode == TRACER_FP_OUTPUT_BUCKET_STATS) {
        printf("  bucketstats top frac: %.6g\n", tracer_bucketstats_top_frac);
    }
    if (tracer_debug_max_snapshots > 0) {
        printf("  debug max snapshots : %d\n", tracer_debug_max_snapshots);
    }
    if (tracer_checkpoint_dir[0] != '\0') {
        printf("  checkpoint dir      : %s\n", tracer_checkpoint_dir);
    }
    if (tracer_restart_dir[0] != '\0') {
        printf("  restart dir         : %s\n", tracer_restart_dir);
    }
    printf("  runtime snapshots   : %d\n", state->nsnap);
    if (state->schedule.physical_min >= 0 || state->schedule.physical_max >= 0) {
        printf("  output nsnp range   : ");
        if (state->schedule.physical_min >= 0) {
            printf("%d", state->schedule.physical_min);
        } else {
            printf("-inf");
        }
        printf("..");
        if (state->schedule.physical_max >= 0) {
            printf("%d", state->schedule.physical_max);
        } else {
            printf("+inf");
        }
        printf("\n");
    }
    printf("  global ntracer      : %ld\n", N_TRACERS);
    printf("  OpenMP compiled     : %s\n", state->omp_info.compiled ? "yes" : "no");
    fflush(stdout);
}

int tracer_fp_load_est_mode(const TracerFpState *state)
{
    return tracer_fp_load_est(state->run_start, state->ntracer, state->nsnap,
                                       state->input_mode, state->background_mode,
                                       state->params_file, state->output_dir,
                                       load_estimate_virtual_ranks,
                                       load_balance_top_frac);
}

int tracer_fp_load_est_actual_mode(TracerFpState *state,
                                   TracerFpSelectionState *selection)
{
    double t0 = 0.0;
    double local_times[5];
    double max_times[5];

    grid_cell_range((long int)state->ntracer, state->mpi_rank, state->mpi_size,
                    &state->local_start, &state->local_count_long);
    state->local_start += state->run_start;
    state->local_count = (int)state->local_count_long;
    state->driver_count = state->run_split_active ? (int)N_TRACERS : state->ntracer;
    selection->driver_start = state->local_start;
    selection->source_count = state->local_count;

    t0 = now_ms();
    if (alloc_load_estimates(state, selection) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": load-estimate allocation failure\n");
        return 1;
    }
    state->load_estimate_alloc_ms = now_ms() - t0;
    if (state->world_rank == 0) {
        printf("  load estimate       : estimating costs on actual MPI ranks (nsnap=%d)\n",
               state->nsnap);
        fflush(stdout);
    }
    t0 = now_ms();
    if (tracer_fp_nsub_estimate(state->local_count, state->local_start, state->ntracer,
                                state->nsnap, state->input_mode, state->background_mode,
                                state->params_file, selection->sum_nsub_local,
                                selection->est_nsub_local,
                                selection->est_target_local) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": actual-rank load-estimate pass failed\n");
        return 1;
    }
    state->load_estimate_nsub_ms = now_ms() - t0;

    t0 = now_ms();
    MPI_Reduce(selection->est_nsub_local, selection->est_nsub_global,
               state->nsnap, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(selection->est_target_local, selection->est_target_global,
               state->nsnap, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    state->load_estimate_reduce_ms = now_ms() - t0;

    if (state->world_rank == 0) {
        long long global_estimate_sum = 0;
        long long global_estimate_target_sum = 0;
        int isnap;
        for (isnap = 0; isnap < state->nsnap; isnap++) {
            global_estimate_sum += selection->est_nsub_global[isnap];
            global_estimate_target_sum += selection->est_target_global[isnap];
        }
        printf("  load estimate sum   : global sum_nsub=%lld target_sum_nsub=%lld\n",
               global_estimate_sum, global_estimate_target_sum);
        fflush(stdout);
    }

    t0 = now_ms();
    if (load_balancing) {
        if (tracer_fp_lb_plan_build(state->run_start, state->local_start,
                                    state->ntracer, state->mpi_rank, state->mpi_size,
                                    state->local_count, selection->sum_nsub_local,
                                    load_balance_top_frac, state->output_dir,
                                    &state->lb_plan) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to build actual-rank load-balance plan\n");
            return 1;
        }
    } else {
        if (tracer_fp_lb_report_baseline(state->run_start, state->local_start,
                                         state->ntracer, state->mpi_rank, state->mpi_size,
                                         state->local_count, selection->sum_nsub_local,
                                         load_balance_top_frac, state->output_dir) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to write actual-rank baseline planning report\n");
            return 1;
        }
    }
    state->load_estimate_plan_ms = now_ms() - t0;

    t0 = now_ms();
    selection->active_local = state->local_count;
    selection->active_global = state->ntracer;
    selection->driver_start = state->local_start;
    selection->source_count = state->local_count;

    local_times[0] = state->load_estimate_alloc_ms;
    local_times[1] = state->load_estimate_nsub_ms;
    local_times[2] = state->load_estimate_reduce_ms;
    local_times[3] = state->load_estimate_plan_ms;
    local_times[4] = now_ms() - t0;
    MPI_Reduce(local_times, max_times, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (state->world_rank == 0) {
        state->load_estimate_alloc_ms = max_times[0];
        state->load_estimate_nsub_ms = max_times[1];
        state->load_estimate_reduce_ms = max_times[2];
        state->load_estimate_plan_ms = max_times[3];
        state->load_estimate_selection_ms = max_times[4];
    }
    return 0;
}

int tracer_fp_selection_prepare(TracerFpState *state,
                                TracerFpSelectionState *selection)
{
    grid_cell_range((long int)state->ntracer, state->mpi_rank, state->mpi_size,
                    &state->local_start, &state->local_count_long);
    state->local_start += state->run_start;
    state->local_count = (int)state->local_count_long;
    state->driver_count = state->run_split_active ? (int)N_TRACERS : state->ntracer;
    selection->driver_start = state->local_start;
    selection->source_count = state->local_count;

    if (hetero_skip_heavy) {
        selection->heavy_local_mask =
            (unsigned char *)calloc(count_or_one(state->local_count),
                                    sizeof(unsigned char));
        if (selection->heavy_local_mask == 0 ||
            alloc_active_buffers(selection, count_or_one(state->local_count)) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": tracer selection allocation failure\n");
            return 1;
        }
        if (hetero_heavy_id_file[0] == '\0') {
            fprintf(stderr, TRACER_FP_PROGNAME ": hetero_skip_heavy=1 requires hetero_heavy_id_file\n");
            return 1;
        }
        if (tracer_fp_heavy_mask_load(hetero_heavy_id_file, state->local_start,
                                            state->local_count, selection->heavy_local_mask,
                                            &selection->heavy_local) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to read heavy tracer IDs from '%s'\n",
                    hetero_heavy_id_file);
            return 1;
        }
        tracer_fp_local_select(state->local_start, state->local_count,
                                        selection->heavy_local_mask, 1,
                                        selection->active_global_ids,
                                        selection->active_source_offsets,
                                        &selection->active_local);
    } else if (load_balancing) {
        const double lb_t0 = now_ms();
        size_t local_cap = count_or_one(state->local_count) +
            (size_t)((int)ceil((double)state->ntracer * load_balance_top_frac)) + 1u;

        if (alloc_load_estimates(state, selection) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": load-balance estimate allocation failure\n");
            return 1;
        }
        if (state->world_rank == 0) {
            printf("  load balance        : estimating costs before FP solve (nsnap=%d)\n",
                   state->nsnap);
            fflush(stdout);
        }
        if (tracer_fp_nsub_estimate(state->local_count, state->local_start, state->ntracer,
                                        state->nsnap, state->input_mode, state->background_mode,
                                        state->params_file, selection->sum_nsub_local,
                                        selection->est_nsub_local,
                                        selection->est_target_local) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": load-balance estimate pass failed\n");
            return 1;
        }
        if (state->world_rank == 0) {
            printf("  load balance        : estimate pass complete\n");
            fflush(stdout);
        }
        MPI_Reduce(selection->est_nsub_local, selection->est_nsub_global,
                   state->nsnap, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(selection->est_target_local, selection->est_target_global,
                   state->nsnap, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (state->world_rank == 0) {
            long long global_estimate_sum = 0;
            long long global_estimate_target_sum = 0;
            int isnap;
            for (isnap = 0; isnap < state->nsnap; isnap++) {
                global_estimate_sum += selection->est_nsub_global[isnap];
                global_estimate_target_sum += selection->est_target_global[isnap];
            }
            printf("  load balance est    : global sum_nsub=%lld target_sum_nsub=%lld\n",
                   global_estimate_sum, global_estimate_target_sum);
            fflush(stdout);
        }
        if (tracer_fp_lb_plan_build(state->run_start, state->local_start,
                                              state->ntracer, state->mpi_rank, state->mpi_size,
                                              state->local_count, selection->sum_nsub_local,
                                              load_balance_top_frac, state->output_dir,
                                              &state->lb_plan) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to build load-balance plan\n");
            return 1;
        }
        {
            int planned_active = 0;
            if (tracer_fp_lb_active_capacity(state->run_start, state->local_start,
                                             state->local_count, &state->lb_plan,
                                             state->mpi_rank, &planned_active) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": failed to size load-balanced tracer selection\n");
                return 1;
            }
            if (planned_active > 0) local_cap = count_or_one(planned_active);
        }
        if (alloc_active_buffers(selection, local_cap) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": balanced selection allocation failure\n");
            return 1;
        }
        selection->heavy_local =
            tracer_fp_lb_select(state->run_start, state->local_start,
                                               state->local_count, &state->lb_plan, state->mpi_rank,
                                               selection->active_global_ids,
                                               selection->active_source_offsets,
                                               &selection->active_local);
        if (selection->heavy_local < 0) {
            fprintf(stderr,
                    TRACER_FP_PROGNAME ": failed to build balanced local tracer selection\n");
            return 1;
        }
        if (state->world_rank == 0) {
            printf("  load balance        : balanced tracer selection ready\n");
            fflush(stdout);
        }
        state->load_balance_ms = now_ms() - lb_t0;
        if (state->world_rank == 0) {
            printf("  load balance time   : %.3f ms\n", state->load_balance_ms);
            fflush(stdout);
        }
        selection->driver_start = state->run_start;
        selection->source_count = state->ntracer;
    } else {
        if (alloc_active_buffers(selection, count_or_one(state->local_count)) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": tracer selection allocation failure\n");
            return 1;
        }
        tracer_fp_local_select(state->local_start, state->local_count, 0, 0,
                                        selection->active_global_ids,
                                        selection->active_source_offsets,
                                        &selection->active_local);
    }

    MPI_Allreduce(&selection->active_local, &selection->active_global,
                  1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&selection->heavy_local, &selection->heavy_global,
                  1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return 0;
}

int tracer_fp_backend_bind(TracerFpState *state)
{
#ifdef FP_USE_CUDA_BACKEND
    if (state->backend_mode != TRACER_FP_BACKEND_CPU && cuda_pipeline_is_available()) {
        typedef struct {
            int world_rank;
            int local_rank;
            int device_count;
            int bound_device;
            int current_device;
            char device_id[64];
            char hostname[256];
            char visible_devices[256];
        } TracerFpCudaBindInfo;

        char hostname[256];
        const char *visible_devices = getenv("CUDA_VISIBLE_DEVICES");
        int irank;
        TracerFpCudaBindInfo local_info;
        TracerFpCudaBindInfo *all_info = 0;

        state->local_rank = local_rank_from_env(state->world_rank);
        if (cuda_bind_local_rank(state->local_rank, &state->bound_device) != 0 &&
            state->backend_mode == TRACER_FP_BACKEND_CUDA) {
            fprintf(stderr,
                    TRACER_FP_PROGNAME ": rank %d failed to bind CUDA device for local rank %d\n",
                    state->world_rank, state->local_rank);
            return 1;
        }
        if (cuda_get_binding_state(&state->cuda_device_count,
                                   &state->cuda_current_device) != 0) {
            state->cuda_device_count = -1;
            state->cuda_current_device = -1;
        }
        if (cuda_get_current_device_id(state->cuda_device_id,
                                       (int)sizeof(state->cuda_device_id)) != 0) {
            strncpy(state->cuda_device_id, "unknown", sizeof(state->cuda_device_id) - 1);
            state->cuda_device_id[sizeof(state->cuda_device_id) - 1] = '\0';
        }
        if (gethostname(hostname, sizeof(hostname)) != 0) {
            strncpy(hostname, "unknown", sizeof(hostname) - 1);
            hostname[sizeof(hostname) - 1] = '\0';
        } else {
            hostname[sizeof(hostname) - 1] = '\0';
        }

        memset(&local_info, 0, sizeof(local_info));
        local_info.world_rank = state->world_rank;
        local_info.local_rank = state->local_rank;
        local_info.device_count = state->cuda_device_count;
        local_info.bound_device = state->bound_device;
        local_info.current_device = state->cuda_current_device;
        strncpy(local_info.device_id, state->cuda_device_id, sizeof(local_info.device_id) - 1);
        strncpy(local_info.hostname, hostname, sizeof(local_info.hostname) - 1);
        if (visible_devices != 0 && *visible_devices != '\0') {
            strncpy(local_info.visible_devices, visible_devices,
                    sizeof(local_info.visible_devices) - 1);
        } else {
            strncpy(local_info.visible_devices, "(unset)",
                    sizeof(local_info.visible_devices) - 1);
        }

        for (irank = 0; irank < state->world_size; irank++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (state->world_rank == irank) {
                printf(TRACER_FP_PROGNAME " CUDA bind"
                       " rank=%d local_rank=%d host=%s visible=%s"
                       " device_count=%d bound=%d current=%d device_id=%s\n",
                       state->world_rank,
                       state->local_rank,
                       hostname,
                       (visible_devices != 0 && *visible_devices != '\0') ? visible_devices : "(unset)",
                       state->cuda_device_count,
                       state->bound_device,
                       state->cuda_current_device,
                       state->cuda_device_id);
                fflush(stdout);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (state->world_rank == 0) {
            all_info = (TracerFpCudaBindInfo *)calloc((size_t)state->world_size, sizeof(*all_info));
        }
        MPI_Gather(&local_info, (int)sizeof(local_info), MPI_BYTE,
                   all_info, (int)sizeof(local_info), MPI_BYTE,
                   0, MPI_COMM_WORLD);
        if (state->world_rank == 0 && all_info != 0) {
            int conflict_found = 0;
            int warned = 0;
            int i, j;
            for (i = 0; i < state->world_size; i++) {
                for (j = i + 1; j < state->world_size; j++) {
                    if (strcmp(all_info[i].hostname, all_info[j].hostname) != 0) continue;
                    if (strcmp(all_info[i].device_id, all_info[j].device_id) != 0) continue;
                    conflict_found = 1;
                    warned = 1;
                    fprintf(stderr,
                            TRACER_FP_PROGNAME ": warning likely GPU contention: "
                            "rank %d and rank %d share host=%s physical_gpu=%s\n",
                            all_info[i].world_rank, all_info[j].world_rank,
                            all_info[i].hostname, all_info[i].device_id);
                }
            }
            if (warned) fflush(stderr);
            if (conflict_found && one_rank_one_gpu != 0) {
                fprintf(stderr,
                        TRACER_FP_PROGNAME ": one_rank_one_gpu=1 but at least two ranks share one visible GPU; aborting\n");
                fflush(stderr);
            }
        }
        {
            int abort_flag = 0;
            if (state->world_rank == 0 && all_info != 0) {
                int i, j;
                for (i = 0; i < state->world_size && !abort_flag; i++) {
                    for (j = i + 1; j < state->world_size; j++) {
                        if (strcmp(all_info[i].hostname, all_info[j].hostname) != 0) continue;
                        if (strcmp(all_info[i].device_id, all_info[j].device_id) != 0) continue;
                        if (one_rank_one_gpu != 0) {
                            abort_flag = 1;
                            break;
                        }
                    }
                }
            }
            MPI_Bcast(&abort_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            free(all_info);
            if (abort_flag) return 1;
        }
    }
#endif
    return 0;
}

void tracer_fp_mapping_print(const TracerFpState *state,
                             const TracerFpSelectionState *selection)
{
    if (state->world_rank != 0) return;
    printf(TRACER_FP_PROGNAME " MPI mapping\n");
    if (state->backend_mode != TRACER_FP_BACKEND_CPU) {
        printf("  cuda devices       : visible=%d current=%d physical=%s\n",
               state->cuda_device_count, state->cuda_current_device,
               state->cuda_device_id);
    }
    printf("  mpi size           : %d\n", state->world_size);
    if (state->job_parallel_active) {
        printf("  job parallel       : job=%d/%d  job_rank=%d  job_size=%d\n",
               job_index, job_count, state->mpi_rank, state->mpi_size);
    }
    printf("  local rank         : %d\n", state->local_rank);
    printf("  global ntracer     : %ld\n", N_TRACERS);
    if (state->run_split_active) {
        printf("  run partition      : job_index=%d / %d\n", job_index, job_count);
        printf("  run tracer range   : [%ld, %ld)\n",
               state->run_start, state->run_start + state->run_count_long);
    }
    printf("  local ntracer      : %d\n", state->local_count);
    printf("  local id range     : [%ld, %ld)\n",
           state->local_start, state->local_start + state->local_count_long);
    if (load_balancing) {
        printf("  load balancing     : 1\n");
        printf("  heavy top frac     : %.4f\n", load_balance_top_frac);
        printf("  LB chunk size      : %d\n", load_balance_chunk_size);
        printf("  LB diagnostics     : %d\n", load_balance_diagnostics);
        printf("  local bulk/heavy   : %d / %d\n",
               selection->active_local - selection->heavy_local,
               selection->heavy_local);
        printf("  global heavy       : %d\n", selection->heavy_global);
    }
    if (hetero_skip_heavy) {
        printf("  hetero skip heavy  : 1\n");
        printf("  local bulk/heavy   : %d / %d\n",
               selection->active_local, selection->heavy_local);
        printf("  global bulk/heavy  : %d / %d\n",
               selection->active_global, selection->heavy_global);
    }
    if (state->frozen_runtime_override > 0) {
        printf("  frozen runtime     : nstep=%d  dt=%.6e Gyr  total=%.6e Gyr\n",
               state->frozen_runtime_override, state->frozen_runtime_dt_gyr,
               state->frozen_runtime_dt_gyr * (double)state->frozen_runtime_override);
    }
}

TracerFpSelection tracer_fp_selection_make(const TracerFpState *state,
                                           const TracerFpSelectionState *selection_state)
{
    TracerFpSelection selection;
    memset(&selection, 0, sizeof(selection));
    selection.ntracer = selection_state->active_local;
    selection.tracer_start = selection_state->driver_start;
    selection.source_ntracer = selection_state->source_count;
    selection.ntracer_global = state->driver_count;
    selection.tracer_ids = selection_state->active_global_ids;
    selection.source_offsets = selection_state->active_source_offsets;
    selection.estimate_snapshot_nsub = selection_state->est_nsub_global;
    selection.estimate_snapshot_target_nsub = selection_state->est_target_global;
    selection.skip_heavy = (hetero_skip_heavy != 0);
    return selection;
}

TracerFpConfig tracer_fp_config_make(const TracerFpState *state)
{
    TracerFpConfig config;
    memset(&config, 0, sizeof(config));
    config.mpi_rank = state->world_rank;
    config.mpi_size = state->mpi_size;
    config.local_rank = state->local_rank;
    config.nsnap = state->nsnap;
    config.nfreq = state->nfreq;
    config.output_spec = state->output_spec;
    config.schedule = &state->schedule;
    config.backend_mode = state->backend_mode;
    config.input_mode = state->input_mode;
    config.params_file = state->params_file;
    config.file_output_mode = state->file_output_mode;
    config.write_buffer_mode = state->write_buffer_mode;
    config.integration_mode = state->integration_mode;
    config.background_mode = state->background_mode;
    config.omp_info = &state->omp_info;
    config.load_balance_ms = state->load_balance_ms;
    return config;
}
