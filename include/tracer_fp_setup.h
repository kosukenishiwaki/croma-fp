#ifndef INCLUDED_tracer_fp_setup_h_
#define INCLUDED_tracer_fp_setup_h_

#include <mpi.h>

#include "tracer_fp.h"

void tracer_fp_init_openmp_info(TracerFpOpenmpInfo *info);
void tracer_fp_configure_openmp(int use_omp_param,
                                int num_threads_param,
                                TracerFpOpenmpInfo *info);
int tracer_fp_choose_coeff_interp_segments(int effective_nsub,
                                           int allow_snapshot_interp);
int tracer_fp_should_log_root_only(void);
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
                                   MPI_Comm *job_comm_out);

void tracer_fp_free_output_schedule(TracerFpOutputSchedule *schedule);
int tracer_fp_parse_output_schedule(const char *spec,
                                    int nsnap,
                                    TracerFpOutputSchedule *schedule);
int tracer_fp_should_emit_snapshot(const TracerFpOutputSchedule *schedule,
                                   int snap_index,
                                   int nsnap);
int tracer_fp_has_later_emit_snapshot(const TracerFpOutputSchedule *schedule,
                                      int snap_index,
                                      int nsnap);

const char *tracer_fp_backend_name(TracerFpBackendMode mode);
int tracer_fp_parse_backend_mode(const char *spec, TracerFpBackendMode *mode);

const char *tracer_fp_input_mode_name(TracerFpInputMode mode);
int tracer_fp_parse_input_mode(const char *spec, TracerFpInputMode *mode);

const char *tracer_fp_file_output_mode_name(TracerFpFileOutputMode mode);
int tracer_fp_parse_file_output_mode(const char *spec, TracerFpFileOutputMode *mode);

const char *tracer_fp_write_buffer_mode_name(TracerFpWriteBufferMode mode);
int tracer_fp_parse_write_buffer_mode(const char *spec, TracerFpWriteBufferMode *mode);

const char *tracer_fp_integration_mode_name(TracerFpIntegrationMode mode);
int tracer_fp_parse_integration_mode(const char *spec, TracerFpIntegrationMode *mode);

const char *tracer_fp_background_mode_name(TracerFpBackgroundMode mode);
int tracer_fp_parse_background_mode(const char *spec, TracerFpBackgroundMode *mode);

const char *tracer_fp_dsa_injection_mode_name(TracerDsaInjectionMode mode);
int tracer_fp_parse_dsa_injection_mode(const char *spec, TracerDsaInjectionMode *mode);

const char *tracer_fp_dsa_reacc_mode_name(TracerDsaReaccMode mode);
int tracer_fp_parse_dsa_reacc_mode(const char *spec, TracerDsaReaccMode *mode);

#endif
