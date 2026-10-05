/*
    tracer_fp_setup.c  (sorry for this confusing name)

    K. Nishiwaki, 2026-06-18
    - setup global conditions for this job/run
    - recognize OpenMP/output/backend/DSA setttings
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tracer_fp_setup.h"

static int env_int(const char *name, int *value_out)
{
    const char *text = getenv(name);
    char *end = 0;
    long value;

    if (value_out == 0) return 0;
    if (text == 0 || *text == '\0') return 0;

    value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < -2147483647L || value > 2147483647L) {
        return -1;
    }

    *value_out = (int)value;
    return 1;
}

static int env_int_with_legacy(const char *name, const char *legacy_name, int *value_out)
{
    int value = 0;
    int state = env_int(name, &value);

    if (state != 0) {
        if (value_out != 0) *value_out = value;
        return state;
    }
    return env_int(legacy_name, value_out);
}

static int parse_positive_int(const char *text, int *value)
{
    char *endptr = 0;
    long parsed;

    if (text == 0 || *text == '\0' || value == 0) return -1;
    parsed = strtol(text, &endptr, 10);
    if (endptr == text || *endptr != '\0' || parsed <= 0 || parsed > 2147483647L) return -1;
    *value = (int)parsed;
    return 0;
}

static int parse_nonnegative_int(const char *text, int *value)
{
    char *endptr = 0;
    long parsed;

    if (text == 0 || *text == '\0' || value == 0) return -1;
    parsed = strtol(text, &endptr, 10);
    if (endptr == text || *endptr != '\0' || parsed < 0 || parsed > 2147483647L) return -1;
    *value = (int)parsed;
    return 0;
}

int tracer_fp_choose_coeff_interp_segments(int effective_nsub,
                                           int allow_snapshot_interp)
{
    int segments = 1;
    int max_allowed;
    int max_segments = coeff_interp_max_segments;

    if (!allow_snapshot_interp || effective_nsub <= 1) return 1;
    if (coeff_interp_min_steps <= 0) return 1;
    if (max_segments < 1) max_segments = 1;

    max_allowed = effective_nsub / coeff_interp_min_steps;
    if (max_allowed < 1) return 1;

    while (segments <= 1073741823 &&
           segments * 2 <= max_segments &&
           segments * 2 <= max_allowed) {
        segments *= 2;
    }
    if (segments > effective_nsub) segments = effective_nsub;
    if (segments < 1) segments = 1;
    return segments;
}

void tracer_fp_init_openmp_info(TracerFpOpenmpInfo *info)
{
    if (info == 0) return;
    memset(info, 0, sizeof(*info));
    info->requested_threads = 1;
    info->max_threads = 1;
    info->num_procs = 1;
}

void tracer_fp_configure_openmp(int use_omp_param,
                                int num_threads_param,
                                TracerFpOpenmpInfo *info)
{
    int env_threads = 0;
    int env_state = 0;

    tracer_fp_init_openmp_info(info);
    if (info == 0) return;

    info->use_omp_param = use_omp_param;
    info->param_threads = num_threads_param;
    env_state = env_int("OMP_NUM_THREADS", &env_threads);
    if (env_state > 0 && env_threads > 0) {
        info->env_threads = env_threads;
    }

#ifdef _OPENMP
    info->compiled = 1;
    info->num_procs = omp_get_num_procs();
    info->dynamic_enabled = omp_get_dynamic();
    if (info->env_threads <= 0 && use_omp_param && num_threads_param > 0) {
        omp_set_num_threads(num_threads_param);
    }
    info->max_threads = omp_get_max_threads();
    if (info->env_threads > 0) {
        info->requested_threads = info->env_threads;
    } else if (use_omp_param && num_threads_param > 0) {
        info->requested_threads = num_threads_param;
    } else {
        info->requested_threads = info->max_threads;
    }
#else
    if (info->env_threads > 0) {
        info->requested_threads = info->env_threads;
    } else if (use_omp_param && num_threads_param > 0) {
        info->requested_threads = num_threads_param;
    }
#endif
}

int tracer_fp_should_log_root_only(void)
{
    int initialized = 0;
    int rank = 0;

    MPI_Initialized(&initialized);
    if (!initialized) return 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return (rank == 0);
}

int tracer_fp_resolve_job_parallel(int world_rank,
                                   int world_size,
                                   int param_job_count,
                                   int param_job_index,
                                   int param_job_parallel_enabled,
                                   int *num_run_out,
                                   int *run_out,
                                   int *job_rank_out,
                                   int *job_size_out,
                                   int *job_parallel_out,
                                   MPI_Comm *job_comm_out)
{
    int env_num_jobs = 0;
    int env_job_size = 0;
    int env_num_jobs_state;
    int env_job_size_state;
    int num_jobs = 1;
    int job_size = world_size;
    int job_id = 0;
    int color = 0;
    MPI_Comm job_comm = MPI_COMM_WORLD;

    if (num_run_out == 0 || run_out == 0 || job_rank_out == 0 ||
        job_size_out == 0 || job_parallel_out == 0 || job_comm_out == 0) {
        return -1;
    }

    *num_run_out = param_job_count;
    *run_out = param_job_index;
    *job_rank_out = world_rank;
    *job_size_out = world_size;
    *job_parallel_out = 0;
    *job_comm_out = MPI_COMM_WORLD;

    env_num_jobs_state = env_int_with_legacy("CROMA_NUM_JOBS", "FP_GPU_NUM_JOBS", &env_num_jobs);
    env_job_size_state = env_int_with_legacy("CROMA_JOB_SIZE", "FP_GPU_JOB_SIZE", &env_job_size);
    if (env_num_jobs_state < 0 || env_job_size_state < 0) return -1;
    if (env_num_jobs_state > 0 && env_job_size_state > 0) return -1;

    if (env_job_size_state > 0) {
        if (env_job_size <= 0 || world_size % env_job_size != 0) return -1;
        job_size = env_job_size;
        num_jobs = world_size / job_size;
        if (param_job_count > 1 && param_job_count != num_jobs) return -1;
    } else if (env_num_jobs_state > 0) {
        if (env_num_jobs <= 0 || world_size % env_num_jobs != 0) return -1;
        num_jobs = env_num_jobs;
        job_size = world_size / num_jobs;
        if (param_job_count > 1 && param_job_count != num_jobs) return -1;
    } else if (param_job_parallel_enabled != 0 ||
               (param_job_count > 1 && param_job_index < 0)) {
        if (param_job_count <= 1 || world_size % param_job_count != 0) return -1;
        num_jobs = param_job_count;
        job_size = world_size / num_jobs;
    } else {
        return 0;
    }

    job_id = world_rank / job_size;
    color = job_id;
    if (MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &job_comm) != MPI_SUCCESS) {
        return -1;
    }

    if (MPI_Comm_rank(job_comm, job_rank_out) != MPI_SUCCESS ||
        MPI_Comm_size(job_comm, job_size_out) != MPI_SUCCESS) {
        MPI_Comm_free(&job_comm);
        return -1;
    }

    *num_run_out = num_jobs;
    *run_out = job_id;
    *job_parallel_out = 1;
    *job_comm_out = job_comm;
    return 0;
}

void tracer_fp_free_output_schedule(TracerFpOutputSchedule *schedule)
{
    if (schedule == 0) return;
    free(schedule->selected_steps);
    schedule->selected_steps = 0;
    schedule->nselected_steps = 0;
}

int tracer_fp_parse_output_schedule(const char *spec,
                                    int nsnap,
                                    TracerFpOutputSchedule *schedule)
{
    if (schedule == 0 || nsnap <= 0) return -1;

    memset(schedule, 0, sizeof(*schedule));
    schedule->emit_all_steps = 1;
    schedule->physical_min = tracer_output_nsnp_min;
    schedule->physical_max = tracer_output_nsnp_max;

    if (spec == 0 || *spec == '\0' || strcmp(spec, "all") == 0) return 0;
    if (strcmp(spec, "none") == 0) {
        schedule->emit_all_steps = 0;
        return 0;
    }
    if (strcmp(spec, "final") == 0) {
        schedule->emit_all_steps = 0;
        schedule->emit_final_only = 1;
        return 0;
    }
    if (strncmp(spec, "every:", 6) == 0) {
        int stride = 0;
        if (parse_positive_int(spec + 6, &stride) != 0) return -1;
        schedule->emit_all_steps = 0;
        schedule->emit_stride = stride;
        return 0;
    }
    if (strncmp(spec, "nsnp>=", 6) == 0) {
        int nsnp_min = -1;
        if (parse_nonnegative_int(spec + 6, &nsnp_min) != 0) return -1;
        schedule->physical_min = nsnp_min;
        return 0;
    }
    if (strncmp(spec, "nsnp>", 5) == 0) {
        int nsnp_min = -1;
        if (parse_nonnegative_int(spec + 5, &nsnp_min) != 0) return -1;
        schedule->physical_min = nsnp_min + 1;
        return 0;
    }
    if (strncmp(spec, "nsnp<=", 6) == 0) {
        int nsnp_max = -1;
        if (parse_nonnegative_int(spec + 6, &nsnp_max) != 0) return -1;
        schedule->physical_max = nsnp_max;
        return 0;
    }
    if (strncmp(spec, "nsnp<", 5) == 0) {
        int nsnp_max = -1;
        if (parse_nonnegative_int(spec + 5, &nsnp_max) != 0) return -1;
        schedule->physical_max = nsnp_max - 1;
        return 0;
    }
    if (strncmp(spec, "nsnp:", 5) == 0) {
        const char *range = spec + 5;
        const char *dash = strchr(range, '-');
        int nsnp_min = -1;
        int nsnp_max = -1;

        if (dash == 0) return -1;
        if (dash > range) {
            char lower[32];
            const size_t len = (size_t)(dash - range);
            if (len >= sizeof(lower)) return -1;
            memcpy(lower, range, len);
            lower[len] = '\0';
            if (parse_nonnegative_int(lower, &nsnp_min) != 0) return -1;
        }
        if (*(dash + 1) != '\0') {
            if (parse_nonnegative_int(dash + 1, &nsnp_max) != 0) return -1;
        }
        schedule->physical_min = nsnp_min;
        schedule->physical_max = nsnp_max;
        return 0;
    }
    if (strncmp(spec, "steps:", 6) == 0) {
        const char *cursor = spec + 6;
        int count = 1;
        int *steps = 0;
        char *copy = 0;
        char *token = 0;
        char *saveptr = 0;
        int i = 0;

        if (*cursor == '\0') return -1;
        while (*cursor != '\0') {
            if (*cursor == ',') count++;
            cursor++;
        }

        copy = strdup(spec + 6);
        if (copy == 0) return -1;
        steps = (int *)calloc((size_t)count, sizeof(int));
        if (steps == 0) {
            free(copy);
            return -1;
        }

        token = strtok_r(copy, ",", &saveptr);
        while (token != 0) {
            if (parse_positive_int(token, &steps[i]) != 0 || steps[i] > nsnap) {
                free(copy);
                free(steps);
                return -1;
            }
            i++;
            token = strtok_r(0, ",", &saveptr);
        }
        free(copy);

        schedule->emit_all_steps = 0;
        schedule->selected_steps = steps;
        schedule->nselected_steps = i;
        return 0;
    }

    return -1;
}

int tracer_fp_should_emit_snapshot(const TracerFpOutputSchedule *schedule,
                                   int snap_index,
                                   int nsnap)
{
    int i;
    const int step_1based = snap_index + 1;
    const int physical_snapshot = nsnp_i + snap_index;

    if (schedule == 0 || nsnap <= 0) return 0;
    if (schedule->physical_min >= 0 && physical_snapshot < schedule->physical_min) return 0;
    if (schedule->physical_max >= 0 && physical_snapshot > schedule->physical_max) return 0;
    if (schedule->emit_all_steps) return 1;
    if (schedule->emit_final_only) return step_1based == nsnap;
    if (schedule->emit_stride > 0) return (step_1based % schedule->emit_stride) == 0;
    for (i = 0; i < schedule->nselected_steps; i++) {
        if (schedule->selected_steps[i] == step_1based) return 1;
    }
    return 0;
}

int tracer_fp_has_later_emit_snapshot(const TracerFpOutputSchedule *schedule,
                                      int snap_index,
                                      int nsnap)
{
    int next;

    if (schedule == 0 || nsnap <= 0) return 0;
    for (next = snap_index + 1; next < nsnap; next++) {
        if (tracer_fp_should_emit_snapshot(schedule, next, nsnap)) return 1;
    }
    return 0;
}

const char *tracer_fp_backend_name(TracerFpBackendMode mode)
{
    if (mode == TRACER_FP_BACKEND_CPU) return "cpu";
    if (mode == TRACER_FP_BACKEND_CUDA) return "cuda";
    return "auto";
}

int tracer_fp_parse_backend_mode(const char *spec, TracerFpBackendMode *mode)
{
    if (spec == 0 || *spec == '\0' || strcmp(spec, "auto") == 0) {
        *mode = TRACER_FP_BACKEND_AUTO;
        return 0;
    }
    if (strcmp(spec, "cpu") == 0) {
        *mode = TRACER_FP_BACKEND_CPU;
        return 0;
    }
    if (strcmp(spec, "cuda") == 0) {
        *mode = TRACER_FP_BACKEND_CUDA;
        return 0;
    }
    return -1;
}

const char *tracer_fp_input_mode_name(TracerFpInputMode mode)
{
    if (mode == TRACER_FP_INPUT_HDF5) return "tracer-hdf5";
    return "synthetic";
}

int tracer_fp_parse_input_mode(const char *spec, TracerFpInputMode *mode)
{
    if (spec == 0 || *spec == '\0' || strcmp(spec, "synthetic") == 0 ||
        strcmp(spec, "test") == 0 || strcmp(spec, "dummy") == 0) {
        *mode = TRACER_FP_INPUT_SYNTHETIC;
        return 0;
    }
    if (strcmp(spec, "hdf5") == 0 || strcmp(spec, "tracer-hdf5") == 0 ||
        strcmp(spec, "real") == 0) {
        *mode = TRACER_FP_INPUT_HDF5;
        return 0;
    }
    return -1;
}

const char *tracer_fp_dsa_injection_mode_name(TracerDsaInjectionMode mode)
{
    if (mode == TRACER_DSA_INJECTION_TRACER_SOURCE) return "tracer_source";
    if (mode == TRACER_DSA_INJECTION_TRACER_STATE) return "tracer_state";
    return "off";
}

int tracer_fp_parse_dsa_injection_mode(const char *spec, TracerDsaInjectionMode *mode)
{
    if (mode == 0) return -1;
    if (spec == 0 || *spec == '\0' || strcmp(spec, "off") == 0 ||
        strcmp(spec, "none") == 0 || strcmp(spec, "0") == 0) {
        *mode = TRACER_DSA_INJECTION_OFF;
        return 0;
    }
    if (strcmp(spec, "tracer_state") == 0 || strcmp(spec, "state") == 0 ||
        strcmp(spec, "1") == 0) {
        *mode = TRACER_DSA_INJECTION_TRACER_STATE;
        return 0;
    }
    if (strcmp(spec, "tracer_source") == 0 || strcmp(spec, "source") == 0 ||
        strcmp(spec, "rate") == 0 || strcmp(spec, "2") == 0) {
        *mode = TRACER_DSA_INJECTION_TRACER_SOURCE;
        return 0;
    }
    return -1;
}

const char *tracer_fp_dsa_reacc_mode_name(TracerDsaReaccMode mode)
{
    if (mode == TRACER_DSA_REACC_DRURY83) return "convolution";
    if (mode == TRACER_DSA_REACC_POSITIVE_DELTA) return "positive_delta";
    return "off";
}

int tracer_fp_parse_dsa_reacc_mode(const char *spec, TracerDsaReaccMode *mode)
{
    if (mode == 0) return -1;
    if (spec == 0 || *spec == '\0' || strcmp(spec, "off") == 0 ||
        strcmp(spec, "none") == 0 || strcmp(spec, "0") == 0) {
        *mode = TRACER_DSA_REACC_OFF;
        return 0;
    }
    if (strcmp(spec, "positive_delta") == 0 || strcmp(spec, "delta") == 0 ||
        strcmp(spec, "1") == 0) {
        *mode = TRACER_DSA_REACC_POSITIVE_DELTA;
        return 0;
    }
    if (strcmp(spec, "convolution") == 0 || strcmp(spec, "conv") == 0 ||
        strcmp(spec, "drury83") == 0 || strcmp(spec, "drury") == 0 ||
        strcmp(spec, "2") == 0) {
        *mode = TRACER_DSA_REACC_DRURY83;
        return 0;
    }
    return -1;
}

const char *tracer_fp_file_output_mode_name(TracerFpFileOutputMode mode)
{
    if (mode == TRACER_FP_OUTPUT_WRITE) return "write";
    if (mode == TRACER_FP_OUTPUT_BUCKET_STATS) return "bucketstats";
    if (mode == TRACER_FP_OUTPUT_LOAD_ESTIMATE) return "load_estimate";
    return "nowrite";
}

int tracer_fp_parse_file_output_mode(const char *spec, TracerFpFileOutputMode *mode)
{
    if (spec == 0 || *spec == '\0' || strcmp(spec, "nowrite") == 0 ||
        strcmp(spec, "debug") == 0 || strcmp(spec, "none") == 0) {
        *mode = TRACER_FP_OUTPUT_NOWRITE;
        return 0;
    }
    if (strcmp(spec, "write") == 0 || strcmp(spec, "files") == 0) {
        *mode = TRACER_FP_OUTPUT_WRITE;
        return 0;
    }
    if (strcmp(spec, "bucketstats") == 0 || strcmp(spec, "bucket_stats") == 0 ||
        strcmp(spec, "stats") == 0) {
        *mode = TRACER_FP_OUTPUT_BUCKET_STATS;
        return 0;
    }
    if (strcmp(spec, "load_estimate") == 0 || strcmp(spec, "loadestimate") == 0 ||
        strcmp(spec, "lb_estimate") == 0) {
        *mode = TRACER_FP_OUTPUT_LOAD_ESTIMATE;
        return 0;
    }
    return -1;
}

const char *tracer_fp_write_buffer_mode_name(TracerFpWriteBufferMode mode)
{
    if (mode == TRACER_FP_WRITE_BUFFER_TILE) return "tile";
    if (mode == TRACER_FP_WRITE_BUFFER_MAPPED) return "mapped";
    return "buffered";
}

int tracer_fp_parse_write_buffer_mode(const char *spec, TracerFpWriteBufferMode *mode)
{
    if (mode == 0) return -1;
    if (spec == 0 || *spec == '\0' || strcmp(spec, "buffered") == 0 ||
        strcmp(spec, "legacy") == 0 || strcmp(spec, "memory") == 0) {
        *mode = TRACER_FP_WRITE_BUFFER_BUFFERED;
        return 0;
    }
    if (strcmp(spec, "mapped") == 0 || strcmp(spec, "mmap") == 0 ||
        strcmp(spec, "filebacked") == 0 || strcmp(spec, "file_backed") == 0) {
        *mode = TRACER_FP_WRITE_BUFFER_MAPPED;
        return 0;
    }
    if (strcmp(spec, "tile") == 0 || strcmp(spec, "tiled") == 0 ||
        strcmp(spec, "slab") == 0 || strcmp(spec, "snapshot_major") == 0) {
        *mode = TRACER_FP_WRITE_BUFFER_TILE;
        return 0;
    }
    return -1;
}

const char *tracer_fp_integration_mode_name(TracerFpIntegrationMode mode)
{
    if (mode == TRACER_FP_INTEGRATION_MULTIRATE) return "multirate";
    return "quantized";
}

int tracer_fp_parse_integration_mode(const char *spec, TracerFpIntegrationMode *mode)
{
    if (spec == 0 || *spec == '\0' || strcmp(spec, "quantized") == 0 ||
        strcmp(spec, "bucket") == 0) {
        *mode = TRACER_FP_INTEGRATION_QUANTIZED;
        return 0;
    }
    if (strcmp(spec, "multirate") == 0 || strcmp(spec, "mr") == 0) {
        *mode = TRACER_FP_INTEGRATION_MULTIRATE;
        return 0;
    }
    return -1;
}

const char *tracer_fp_background_mode_name(TracerFpBackgroundMode mode)
{
    if (mode == TRACER_FP_BACKGROUND_FROZEN) return "frozen";
    return "evolving";
}

int tracer_fp_parse_background_mode(const char *spec, TracerFpBackgroundMode *mode)
{
    if (spec == 0 || *spec == '\0' || strcmp(spec, "evolving") == 0 ||
        strcmp(spec, "live") == 0 || strcmp(spec, "history") == 0) {
        *mode = TRACER_FP_BACKGROUND_EVOLVING;
        return 0;
    }
    if (strcmp(spec, "frozen") == 0 || strcmp(spec, "fixed") == 0 ||
        strcmp(spec, "single") == 0) {
        *mode = TRACER_FP_BACKGROUND_FROZEN;
        return 0;
    }
    return -1;
}
