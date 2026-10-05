#ifndef INCLUDED_tracer_fp_entry_helpers_h_
#define INCLUDED_tracer_fp_entry_helpers_h_

#include <mpi.h>

#include "tracer_fp.h"

typedef struct {
    int ntracer;
    int nsnap;
    int nfreq;
    const char *params_file;
    const char *background_spec;
    int background_cli_override;
    const char *output_spec;
    TracerFpOutputSchedule schedule;
    TracerFpBackendMode backend_mode;
    TracerFpInputMode input_mode;
    TracerFpFileOutputMode file_output_mode;
    TracerFpWriteBufferMode write_buffer_mode;
    TracerFpIntegrationMode integration_mode;
    TracerFpBackgroundMode background_mode;
    int mpi_rank;
    int mpi_size;
    int world_rank;
    int world_size;
    int job_parallel_active;
    int effective_num_run;
    int effective_run;
    MPI_Comm job_comm;
    int local_rank;
    int bound_device;
    int cuda_device_count;
    int cuda_current_device;
    char cuda_device_id[64];
    long int local_start;
    long int local_count_long;
    long int run_start;
    long int run_count_long;
    int local_count;
    int frozen_runtime_override;
    double frozen_runtime_dt_gyr;
    int run_split_active;
    int driver_count;
    char output_dir[MAX_LINE_LENGTH];
    TracerLoadBalancePlan lb_plan;
    double load_balance_ms;
    double load_estimate_alloc_ms;
    double load_estimate_nsub_ms;
    double load_estimate_reduce_ms;
    double load_estimate_plan_ms;
    double load_estimate_selection_ms;
    TracerFpOpenmpInfo omp_info;
} TracerFpState;

typedef struct {
    int active_local;
    int active_global;
    int heavy_local;
    int heavy_global;
    unsigned char *heavy_local_mask;
    long long *sum_nsub_local;
    long long *est_nsub_local;
    long long *est_target_local;
    long long *est_nsub_global;
    long long *est_target_global;
    long int *active_global_ids;
    int *active_source_offsets;
    long int driver_start;
    int source_count;
} TracerFpSelectionState;

void tracer_fp_state_init(TracerFpState *state);
void tracer_fp_selection_state_init(TracerFpSelectionState *selection);
void tracer_fp_state_release(TracerFpState *state);
void tracer_fp_selection_state_release(TracerFpSelectionState *selection);
int tracer_fp_parse_cli(int argc, char **argv, TracerFpState *state);
int tracer_fp_setup_prepare(TracerFpState *state);
void tracer_fp_setup_print(const TracerFpState *state);
int tracer_fp_load_est_mode(const TracerFpState *state);
int tracer_fp_load_est_actual_mode(TracerFpState *state,
                                   TracerFpSelectionState *selection);
int tracer_fp_selection_prepare(TracerFpState *state,
                                TracerFpSelectionState *selection);
int tracer_fp_backend_bind(TracerFpState *state);
void tracer_fp_mapping_print(const TracerFpState *state,
                             const TracerFpSelectionState *selection);
TracerFpSelection tracer_fp_selection_make(const TracerFpState *state,
                                           const TracerFpSelectionState *selection_state);
TracerFpConfig tracer_fp_config_make(const TracerFpState *state);

#endif /* INCLUDED_tracer_fp_entry_helpers_h_ */
