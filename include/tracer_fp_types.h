#ifndef INCLUDED_tracer_fp_types_h_
#define INCLUDED_tracer_fp_types_h_

#include <stddef.h>

#include "params.h"

enum { TRACER_FP_BUCKET_HIST_NBIN = 6 };

typedef struct {
    double loss_prepass_ms;
    double nsub_estimate_ms;
    double bucket_build_ms;
    double bucketstats_top_ms;
    double gpu_group_build_ms;
    double pack_ms;
    double bucket_host_ms;
    double interp_ms;
    double snapshot_prep_ms;
    double coeff_ms;
    double secondary_ms;
    double solve_ms;
    double solve_alloc_ms;
    double solve_rhs_ms;
    double solve_tridiag_ms;
    double synch_table_ms;
    double synch_ms;
    double ic_ms;
    double gamma_ms;
    double neutrino_ms;
    double cuda_setup_ms;
    double cuda_h2d_ms;
    double cuda_d2h_ms;
    double cuda_other_ms;
    double cuda_total_ms;
    double input_read_ms;
    double bg_prepare_ms;
    double tracer_mass_ms;
    double output_write_ms;
    double output_sync_ms;
    double checkpoint_ms;
    double restart_ms;
    double total_ms;
    int input_read_calls;
    int input_selected_runs;
    long long input_offset_runs;
    int input_offset_min_run_len;
    int input_offset_max_run_len;
    int input_offset_max_gap;
    int output_write_calls;
    int output_sync_calls;
    int checkpoint_calls;
    long long gpu_pipeline_calls;
    long long gpu_pipeline_cells;
    long long gpu_pipeline_fp_steps;
    long long gpu_pipeline_cell_steps;
} TracerFpGpuTimes;

typedef struct {
    double io_ms;
    double scheduling_ms;
    double host_ms;
    double coeffprep_ms;
    double secondary_ms;
    double solve_ms;
    double transport_ms;
    double emission_ms;
    double backend_ms;
    double total_ms;
} TracerFpStageTimes;

static void tracer_fp_stage_times_from_raw(const TracerFpGpuTimes *times,
                                           double load_balance_ms,
                                           TracerFpStageTimes *stage)
{
    if (stage == 0) return;

    stage->io_ms = 0.0;
    stage->scheduling_ms = 0.0;
    stage->host_ms = 0.0;
    stage->coeffprep_ms = 0.0;
    stage->secondary_ms = 0.0;
    stage->solve_ms = 0.0;
    stage->transport_ms = 0.0;
    stage->emission_ms = 0.0;
    stage->backend_ms = 0.0;
    stage->total_ms = 0.0;

    if (times == 0) return;

    stage->io_ms = times->input_read_ms + times->bg_prepare_ms +
                   times->tracer_mass_ms + times->output_write_ms +
                   times->output_sync_ms + times->checkpoint_ms +
                   times->restart_ms;
    stage->scheduling_ms = times->loss_prepass_ms + times->nsub_estimate_ms +
                           times->bucket_build_ms + times->bucketstats_top_ms +
                           times->gpu_group_build_ms + load_balance_ms;
    stage->host_ms = times->pack_ms + times->bucket_host_ms + times->interp_ms +
                     times->snapshot_prep_ms;
    stage->coeffprep_ms = times->coeff_ms;
    stage->secondary_ms = times->secondary_ms;
    stage->solve_ms = times->solve_ms;
    stage->transport_ms = stage->coeffprep_ms + stage->secondary_ms + stage->solve_ms;
    stage->emission_ms = times->synch_table_ms + times->synch_ms + times->ic_ms +
                         times->gamma_ms + times->neutrino_ms;
    stage->backend_ms = times->cuda_setup_ms + times->cuda_h2d_ms +
                        times->cuda_d2h_ms + times->cuda_other_ms;
    stage->total_ms = stage->io_ms + stage->scheduling_ms + stage->host_ms +
                      stage->transport_ms + stage->emission_ms +
                      stage->backend_ms;
}

typedef struct {
    int compiled;
    int use_omp_param;
    int param_threads;
    int env_threads;
    int requested_threads;
    int max_threads;
    int num_procs;
    int dynamic_enabled;
} TracerFpOpenmpInfo;

typedef struct {
    long int tracer_id;
    int nsub;
    int target_nsub;
    double n_gas;
    double kbt;
    double b_field;
    double divv;
    double lturb;
    double dv;
    double beta;
} TracerBucketStatsRecord;

typedef struct {
    long int tracer_id;
    long long sum_nsub;
    int target_rank;
    int count;
} TracerLoadBalanceRecord;

typedef struct {
    int enabled;
    int nheavy;
    int chunk_size;
    TracerLoadBalanceRecord *records;
} TracerLoadBalancePlan;

typedef struct {
    int emit_final_only;
    int emit_all_steps;
    int emit_stride;
    int physical_min;
    int physical_max;
    int *selected_steps;
    int nselected_steps;
} TracerFpOutputSchedule;

typedef struct {
    int ntracer;
    long int tracer_start;
    int source_ntracer;
    int ntracer_global;
    const long int *tracer_ids;
    const int *source_offsets;
    const long long *estimate_snapshot_nsub;
    const long long *estimate_snapshot_target_nsub;
    int skip_heavy;
} TracerFpSelection;

typedef enum {
    TRACER_FP_BACKEND_AUTO = 0,
    TRACER_FP_BACKEND_CPU = 1,
    TRACER_FP_BACKEND_CUDA = 2
} TracerFpBackendMode;

typedef enum {
    TRACER_FP_INPUT_SYNTHETIC = 0,
    TRACER_FP_INPUT_HDF5 = 1
} TracerFpInputMode;

typedef enum {
    TRACER_FP_OUTPUT_NOWRITE = 0,
    TRACER_FP_OUTPUT_WRITE = 1,
    TRACER_FP_OUTPUT_BUCKET_STATS = 2,
    TRACER_FP_OUTPUT_LOAD_ESTIMATE = 3
} TracerFpFileOutputMode;

typedef enum {
    TRACER_FP_WRITE_BUFFER_BUFFERED = 0,
    TRACER_FP_WRITE_BUFFER_MAPPED = 1,
    TRACER_FP_WRITE_BUFFER_TILE = 2
} TracerFpWriteBufferMode;

typedef enum {
    TRACER_FP_INTEGRATION_QUANTIZED = 0,
    TRACER_FP_INTEGRATION_MULTIRATE = 1
} TracerFpIntegrationMode;

typedef enum {
    TRACER_FP_BACKGROUND_EVOLVING = 0,
    TRACER_FP_BACKGROUND_FROZEN = 1
} TracerFpBackgroundMode;

typedef struct {
    int mpi_rank;
    int mpi_size;
    int local_rank;
    int nsnap;
    int nfreq;
    const char *output_spec;
    const TracerFpOutputSchedule *schedule;
    TracerFpBackendMode backend_mode;
    TracerFpInputMode input_mode;
    const char *params_file;
    TracerFpFileOutputMode file_output_mode;
    TracerFpWriteBufferMode write_buffer_mode;
    TracerFpIntegrationMode integration_mode;
    TracerFpBackgroundMode background_mode;
    const TracerFpOpenmpInfo *omp_info;
    double load_balance_ms;
} TracerFpConfig;

typedef struct {
    int ntheta_pitch;
    int nx_tab;
    double xmin;
    double xmax;
    double dxtab;
    double *logx_tab;
    double *logfx_tab;
    double *fx_tab;
    double *theta;
    double *dtheta;
    double *pitch_weight;
    double *nus;
    double *nu_crit_storage;
    double *logy_storage;
    double **nu_crit_rows;
    double **logy_rows;
    double ***logy_planes;
    double *kernel_table;
    double table_logb_min;
    double table_logb_max;
} SynchData;

typedef struct {
    double *n_gas;
    double *kbt;
    double *b_field;
    double *divv;
    double *l_turb;
    double *dv_imc;
    double *cs;
    double *beta_pl;
} TracerFpBackgroundSlot;

typedef struct {
    double *n_gas;
    double *kbt;
    double *b_field;
    double *divv;
    double *l_turb;
    double *dv_imc;
    double *cs;
    double *beta_pl;
} TracerDataHistory;

typedef struct {
    double *temp;
    double *rho;
    double *bx;
    double *by;
    double *bz;
    double *divv;
    double *rotv;
    double *lturb;
    double *mach;
    double *prestemp;
    double *presden;
    double *pre_density_cgs;
    double *upstream_speed_cgs;
    double *shock_side_code;
    double *dsa_trigger;
    double *dsa_mach;
    double *dsa_pre_density;
    double *dsa_kinetic_energy_flux_cgs;
} TracerDataStorage;

typedef struct {
    double *temp;
    double *rho;
    double *bx;
    double *by;
    double *bz;
    double *divv;
    double *rotv;
    double *lturb;
    double *mach;
    double *prestemp;
    double *presden;
    double *pre_density_cgs;
    double *upstream_speed_cgs;
    double *shock_side_code;
    double *dsa_trigger;
    double *dsa_mach;
    double *dsa_pre_density;
    double *dsa_kinetic_energy_flux_cgs;
    int has_shock;
    int has_dsa;
} TracerFpRawBackgroundSlot;

typedef enum {
    TRACER_DSA_INJECTION_OFF = 0,
    TRACER_DSA_INJECTION_TRACER_STATE = 1,
    TRACER_DSA_INJECTION_TRACER_SOURCE = 2
} TracerDsaInjectionMode;

typedef enum {
    TRACER_DSA_REACC_OFF = 0,
    TRACER_DSA_REACC_POSITIVE_DELTA = 1,
    TRACER_DSA_REACC_DRURY83 = 2
} TracerDsaReaccMode;

typedef struct {
    double *z_full;
    double *z_nxt_full;
    double *c_dens;
    double *c_velo;
    double *c_mag;
} TracerFpHdf5Meta;

typedef struct {
    double **rows;
    double *base;
    size_t nrow;
    size_t ncol;
    size_t nbyte;
    int fd;
    int enabled;
    char path[MAX_LINE_LENGTH];
} TracerFpMappedOutput;

typedef struct {
    size_t nslab;
    size_t nrow;
    size_t ncol;
    size_t nbyte;
    int fd;
    int enabled;
    char path[MAX_LINE_LENGTH];
} TracerFpTileOutput;

typedef struct {
    char magic[16];
    int version;
    int mpi_rank;
    int world_size;
    int ntracer;
    int ntracer_global;
    int nsnap;
    int nfreq;
    int np_bins;
    int npe_bins;
    int nsnp_i_value;
    int nsnp_f_value;
    int completed_snapshot;
    long int tracer_start;
} TracerFpCheckpointHeader;

#endif
