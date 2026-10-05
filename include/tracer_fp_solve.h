#ifndef INCLUDED_tracer_fp_solve_h_
#define INCLUDED_tracer_fp_solve_h_

#include <stdio.h>

#include "tracer_fp.h"
#include "tracer_fp_background.h"
#include "tracer_fp_bucket.h"
#include "tracer_fp_output.h"
#include "tracer_fp_step.h"
#include "tracer_fp_synch.h"

typedef struct {
    const TracerFpOutputSchedule *schedule;
    const int *source_offsets;
    const long int *tracer_ids;
    FILE *input_read_fp;
    int *input_read_order;
    FILE *dsa_reacc_debug_fp;
    int local_rank;
    const char *hostname;
    int input_selected_count;
    int input_offset_runs;
    int input_offset_min_run_len;
    int input_offset_max_run_len;
    int input_offset_max_gap;
    long int input_offset_min;
    long int input_offset_max;
    long int input_offset_span;
    const char *checkpoint_dir;
    double *dt_snap;
    double *z_snap;
    double *elapsed_gyr;
    double *beta_p;
    double *gamma2e;
    double *tracer_mass;
    double *rad_ic_zero;
    double *rad_ic_m1_zero;
    double *rad_ic_p1_zero;
    double *eps_syn;
    double *eps_ic;
    double *eps_gamma;
    double *eps_nu;
    double *b_dyn;
    double *logb;
    double *fqe_flat;
    double *fic_flat;
    double *fga_flat;
    double *fnu_flat;
    int *np_min_qe;
    double *qpi_batch;
    double *qepri_batch;
    double *crp_state;
    double *cre_state;
    int *tracer_id_core;
    int *emitted_synch;
    int *emitted_gamma;
    int *emitted_neutrino;
    int *last_synch_snap;
    int *dsa_injected;
    int *bucket_calls;
    long long *bucket_cells;
    int *bucket_min;
    int *bucket_max;
    int *max_buckets_per_snap;
    int *max_nsubsteps;
    long long *capped_tracer_snaps;
    int *max_raw_nsub;
    long long *gpu_pipeline_calls;
    long long *gpu_pipeline_cells;
    long long *gpu_pipeline_fp_steps;
    long long *gpu_pipeline_cell_steps;
    long long *gpu_group_count_est;
    int *max_gpu_groups_per_snap;
    long long *bucket_hist_counts;
    long long *runtime_nsub;
    long long *runtime_target_nsub;
} TracerFpSolveCtx;

typedef struct {
    TracerFpBackgroundSlot *bg_slots;
    TracerFpRawBackgroundSlot *raw_slots;
    TracerFpHdf5Meta *hdf5_meta;
    int *bg_curr;
    int *bg_next;
    int *raw_prev;
    int *raw_curr;
    int *raw_next;
} TracerFpBgWin;

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
    double **ne_buffer_core;
    double **np_buffer_core;
    double **epssyn_buffer_core;
    double **epsic_buffer_core;
    double **epsgamma_buffer_core;
    double **epsnu_buffer_core;
    double **ne_chunk_core;
    double **np_chunk_core;
    double **epssyn_chunk_core;
    double **epsic_chunk_core;
    double **epsgamma_chunk_core;
    double **epsnu_chunk_core;
    int *chunk_start;
    int *chunk_count;
    int *chunk_cr_base;
    int *chunk_cr_slots;
    FILE *bucket_top_fp;
    FILE *bucket_rank_fp;
} TracerFpOutState;

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
                    double requested_t_acc_direct_gyr);

#endif
