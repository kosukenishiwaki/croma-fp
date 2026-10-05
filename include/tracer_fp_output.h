#ifndef INCLUDED_tracer_fp_output_h_
#define INCLUDED_tracer_fp_output_h_

#include <stdio.h>

#include "tracer_fp.h"

void output_tracerid_node(char *strout, int n_tracer_node, int mpi_rank, int *id_thread);
void output_CRspectrum_buffer_node(char *strout,
                                   int n_tracer_node,
                                   int size_per_tracer,
                                   int sign,
                                   int mpi_rank,
                                   double **n_buffer);

typedef struct {
    TracerFpMappedOutput *ne_output;
    TracerFpMappedOutput *np_output;
    TracerFpMappedOutput *epssyn_output;
    TracerFpMappedOutput *epsic_output;
    TracerFpMappedOutput *epsgamma_output;
    TracerFpMappedOutput *epsnu_output;
    TracerFpTileOutput *ne_tile_output;
    TracerFpTileOutput *np_tile_output;
    TracerFpTileOutput *epssyn_tile_output;
    TracerFpTileOutput *epsic_tile_output;
    TracerFpTileOutput *epsgamma_tile_output;
    TracerFpTileOutput *epsnu_tile_output;
    double ***ne_buffer_core;
    double ***np_buffer_core;
    double ***epssyn_buffer_core;
    double ***epsic_buffer_core;
    double ***epsgamma_buffer_core;
    double ***epsnu_buffer_core;
    double ***ne_chunk_core;
    double ***np_chunk_core;
    double ***epssyn_chunk_core;
    double ***epsic_chunk_core;
    double ***epsgamma_chunk_core;
    double ***epsnu_chunk_core;
    FILE **bucketstats_top_fp;
    FILE **bucketstats_rank_fp;
} TracerFpOutputs;

typedef struct {
    char *output_dir;
    size_t output_dir_size;
    char *checkpoint_dir;
    size_t checkpoint_dir_size;
    int *tracer_id_core;
    const long int *tracer_ids;
    int ntracer;
    int nsnap;
    int nfreq;
    int mpi_rank;
    int write_output_files;
    int write_crp_output;
    int ic_enabled;
    int gamma_enabled;
    int neutrino_enabled;
    int use_tile_output;
    int use_mapped_output;
    int use_mapped_chunk;
    int mapped_chunk_snapshots;
    int bucket_stats_only;
    int checkpoint_enabled;
    int restart_enabled;
    const char *configured_output_dir;
    int log_root;
} TracerFpOutputCfg;

typedef struct {
    const char *output_dir;
    int ntracer;
    int nsnap;
    int nfreq;
    int mpi_rank;
    int write_crp_output;
    int ic_enabled;
    int gamma_enabled;
    int neutrino_enabled;
    int use_tile_output;
    int use_mapped_output;
    int total_capped_tracer_snapshots;
    const int *tracer_id_core;
    const unsigned char *capped_flags;
} TracerFpOutputDone;

typedef struct {
    const char *output_dir;
    int mpi_rank;
    int log_root;
    int world_size;
    int mpi_size;
    long int global_ntracer;
    int local_ntracer;
    int nsnap;
    TracerFpInputMode input_mode;
    TracerFpFileOutputMode file_output_mode;
    TracerFpIntegrationMode integration_mode;
    TracerFpBackgroundMode background_mode;
    const TracerFpOpenmpInfo *omp_info;
    const TracerFpGpuTimes *times;
    double load_balance_ms;
    long long runtime_sum_nsub_local;
    long long runtime_target_nsub_local;
    double wall_ms;
    double wall_ms_min;
    double wall_ms_mean;
    double wall_ms_max;
} TracerFpRunInfo;

int ensure_output_dir(const char *path);

void tracer_fp_map_reset(TracerFpMappedOutput *out);
void tracer_fp_tile_reset(TracerFpTileOutput *out);
int tracer_fp_map_sync(const TracerFpMappedOutput *out);
int tracer_fp_tile_sync(const TracerFpTileOutput *out);
int tracer_fp_write_cr_map(TracerFpMappedOutput *ne_out,
                           TracerFpMappedOutput *np_out,
                           int nlocal,
                           int snapshot_slot,
                           const double *cre_state,
                           const double *crp_state,
                           const double *rho_gcc,
                           const double *tracer_mass,
                           int output_per_cc);
int tracer_fp_write_emit_map(TracerFpMappedOutput *out,
                             int nlocal,
                             int snapshot_index,
                             int nbin,
                             const double *emission_cell_major,
                             const double *rho_gcc,
                             const double *tracer_mass,
                             int output_per_cc);
void tracer_fp_zero_rows(double **buffer, int nrow, size_t ncol);
int tracer_fp_tile_open(TracerFpTileOutput *out,
                               const char *output_dir,
                               const char *stem,
                               int mpi_rank,
                               size_t nslab,
                               size_t nrow,
                               size_t ncol,
                               int resume_existing);
int tracer_fp_tile_write(TracerFpTileOutput *out,
                         size_t slab,
                         const double *values);
int tracer_fp_tile_write_cr(TracerFpTileOutput *ne_out,
                            TracerFpTileOutput *np_out,
                            int nlocal,
                            size_t slab,
                            const double *cre_state,
                            const double *crp_state,
                            const double *rho_gcc,
                            const double *tracer_mass,
                            int output_per_cc,
                            int write_crp_output);
int tracer_fp_tile_write_emit(TracerFpTileOutput *out,
                              int nlocal,
                              size_t slab,
                              int nbin,
                              const double *emission_cell_major,
                              const double *rho_gcc,
                              const double *tracer_mass,
                              int output_per_cc);
int tracer_fp_tile_close(TracerFpTileOutput *out);
int tracer_fp_flush_map_chunk(TracerFpMappedOutput *ne_out,
                                        TracerFpMappedOutput *np_out,
                                        TracerFpMappedOutput *epssyn_out,
                                        TracerFpMappedOutput *epsic_out,
                                        TracerFpMappedOutput *epsgamma_out,
                                        TracerFpMappedOutput *epsnu_out,
                                        int ic_enabled,
                                        int gamma_enabled,
                                        int neutrino_enabled,
                                        int nlocal,
                                        int nfreq,
                                        int chunk_start_snapshot,
                                        int chunk_count,
                                        int chunk_cr_base_slot,
                                        int chunk_cr_slots,
                                        double **ne_chunk,
                                        double **np_chunk,
                                        double **epssyn_chunk,
                                        double **epsic_chunk,
                                        double **epsgamma_chunk,
                                        double **epsnu_chunk);
int tracer_fp_map_open(TracerFpMappedOutput *out,
                                 const char *output_dir,
                                 const char *stem,
                                 int mpi_rank,
                                 int nrow,
                                 size_t ncol,
                                 int resume_existing);
int tracer_fp_map_close(TracerFpMappedOutput *out);
int tracer_fp_outputs_init(TracerFpOutputs *files,
                           const TracerFpOutputCfg *cfg);
void tracer_fp_outputs_free(TracerFpOutputs *files,
                               int ntracer);
int tracer_fp_outputs_done(TracerFpOutputs *files,
                               const TracerFpOutputDone *cfg);

int tracer_fp_write_rank_log(const char *output_dir,
                                    int mpi_rank,
                                    int world_size,
                                    int mpi_size,
                                    long int global_ntracer,
                                    int local_ntracer,
                                    int nsnap,
                                    TracerFpInputMode input_mode,
                                    TracerFpFileOutputMode file_output_mode,
                                    TracerFpIntegrationMode integration_mode,
                                    TracerFpBackgroundMode background_mode,
                                    const TracerFpOpenmpInfo *omp_info,
                                    const TracerFpGpuTimes *times,
                                    double load_balance_ms,
                                    long long runtime_sum_nsub_local,
                                    long long runtime_target_nsub_local,
                                    double wall_ms,
                                    double max_wall_ms,
                                    double mean_wall_ms,
                                    double min_wall_ms);
int tracer_fp_write_summary(const char *output_dir,
                                int world_size,
                                int mpi_size,
                                long int global_ntracer,
                                int runtime_steps,
                                TracerFpInputMode input_mode,
                                TracerFpFileOutputMode file_output_mode,
                                TracerFpIntegrationMode integration_mode,
                                TracerFpBackgroundMode background_mode,
                                const TracerFpOpenmpInfo *omp_info,
                                double load_balance_ms,
                                double wall_ms_min,
                                double wall_ms_mean,
                                double wall_ms_max,
                                double global_throughput);
int tracer_fp_write_nsub_log(const char *output_dir,
                                         int nsnap,
                                         const long long *estimate_nsub,
                                         const long long *runtime_nsub,
                                         const long long *estimate_target_nsub,
                                         const long long *runtime_target_nsub);
void tracer_fp_write_ids(const char *output_dir,
                                int ntracer,
                                int mpi_rank,
                                const int *tracer_id_core);
void tracer_fp_write_artifacts(const TracerFpRunInfo *cfg,
                                   const long long *estimate_nsub,
                                   const long long *runtime_nsub,
                                   const long long *estimate_target_nsub,
                                   const long long *runtime_target_nsub);

void populate_output_buffers(double **ne_buffer,
                             double **np_buffer,
                             int nlocal,
                             int snapshot_slot,
                             const double *cre_state,
                             const double *crp_state,
                             const double *rho_gcc,
                             const double *tracer_mass,
                             int output_per_cc);
void populate_emission_buffer(double **buffer,
                              int nlocal,
                              int snapshot_index,
                              int nbin,
                              const double *emission_cell_major,
                              const double *rho_gcc,
                              const double *tracer_mass,
                              int output_per_cc);
int output_flagged_tracer_ids(const char *strout,
                              int nlocal,
                              int mpi_rank,
                              const int *tracer_ids,
                              const unsigned char *flags);

#endif
