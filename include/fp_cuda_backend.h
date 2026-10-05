#ifndef INCLUDED_fp_cuda_backend_h_
#define INCLUDED_fp_cuda_backend_h_

#include "fp_shared_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FpCudaPipelineWorkspace FpCudaPipelineWorkspace;

typedef struct {
    int emit_final_only;
    int emit_all_steps;
    int emit_stride;
    int physical_min;
    int physical_max;
    const int *selected_steps;
    int nselected_steps;
} FpCudaOutputSchedule;

typedef enum {
    FP_CUDA_TRANSPORT_LEGACY = 0,
    FP_CUDA_TRANSPORT_FUSED_SUBSTEPS = 1
} FpCudaTransportMode;

typedef struct {
    int ncell;
    int nstep;
    int fp_cadence_steps;
    int diff_cadence_steps;
    int adv_cadence_steps;
    int use_windowed_reacc;
    int on_start_step;
    int on_end_step;
    int nfreq;
    int nx_tab;

    double dt;
    double epmax;
    double xmin;
    double psi_value;
    double mach_limit_value;
    FpCudaTransportMode transport_mode;

    const CRspectrum *crp_grid;
    const CRspectrum *cre_grid;
    const FpCoeffBatchInput *coeff_in;
    const FpSynchEmissionBatchInput *synch_in; /* optional; only needed when transport emits synch */
    const FpGammaEmissionBatchInput *gamma_in;
    const FpNeutrinoEmissionBatchInput *neutrino_in;

    const double *qpi_batch;      /* [ncell * np ] cell-major */
    const double *qepri_batch;    /* [ncell * npe] cell-major */
    const double *beta_p;         /* [np] */
    const double *crp_init;       /* [ncell * np ] cell-major */
    const double *cre_init;       /* [ncell * npe] cell-major */
    double *crp_state_out;        /* optional output slice */
    double *cre_state_out;        /* optional output slice */
    double *eps_syn_out;          /* optional output slice */
    double *eps_gamma_out;        /* optional output slice */
    double *eps_nu_out;           /* optional output slice */
    const int *rank_state_indices; /* optional [ncell] host local tracer indices */
    int gather_rank_state;        /* load scratch state/source rows from rank-global device arrays */
    int scatter_rank_state;       /* store scratch CR rows into rank-global device arrays */
    FpCudaPipelineWorkspace *workspace; /* optional reusable CUDA workspace */
} FpCudaPipelineInput;

typedef struct {
    double setup_ms;
    double h2d_ms;
    double device_interp_ms;
    double coeff_ms;
    double secondary_ms;
    double solve_ms;
    double diff_ms;
    double adv_ms;
    double synch_ms;
    double gamma_ms;
    double neutrino_ms;
    double d2h_ms;
    double other_ms;
    double total_ms;
    int emitted_synch;
    int emitted_gamma;
    int emitted_neutrino;
    int last_synch_step;
    int used_pcr_proton;
    int used_pcr_electron;
} FpCudaPipelineTimes;

int cuda_pipeline_is_available(void);
int cuda_bind_local_rank(int local_rank, int *device_out);
int cuda_get_binding_state(int *device_count_out, int *current_device_out);
int cuda_get_current_device_id(char *device_id_out, int device_id_out_len);

int prepare_synch_emission_cuda_batch(const FpSynchEmissionBatchInput *in,
                                              const CRspectrum *cre_grid,
                                              double *eps_syn_out,
                                              double *elapsed_ms);

int prepare_gamma_emission_cuda_batch(const FpGammaEmissionBatchInput *in,
                                              const CRspectrum *crp_grid,
                                              double *eps_gamma_out,
                                              double *elapsed_ms);

int prepare_neutrino_emission_cuda_batch(const FpNeutrinoEmissionBatchInput *in,
                                                 const CRspectrum *crp_grid,
                                                 double *eps_nu_out,
                                                 double *elapsed_ms);

int solve_cc_cuda_batch(const FpChangCooperSolveBatchInput *in,
                        const double *state_init,
                        double *state_out,
                        double *elapsed_ms,
                        int *used_pcr_out);

int fp_cuda_tridiag_uses_pcr(int nrow);
const char *fp_cuda_tridiag_solver_name(int nrow);

FpCudaPipelineWorkspace *cuda_pipeline_workspace_create(void);
void cuda_pipeline_workspace_destroy(FpCudaPipelineWorkspace *workspace);

int cuda_rank_state_upload(FpCudaPipelineWorkspace *workspace,
                           int ncell,
                           const double *qpi,
                           const double *qepri,
                           const double *crp,
                           const double *cre,
                           double *elapsed_ms);

int cuda_rank_state_download(FpCudaPipelineWorkspace *workspace,
                             int ncell,
                             double *crp,
                             double *cre,
                             double *elapsed_ms);

int run_tracer_pipeline(const FpCudaPipelineInput *in,
                        const FpCudaOutputSchedule *schedule,
                        double *crp_state_out,
                        double *cre_state_out,
                        double *eps_syn_out,
                        double *eps_gamma_out,
                        double *eps_nu_out,
                        FpCudaPipelineTimes *times);

int run_tracer_pipeline_group(const FpCudaPipelineInput *inputs,
                              int ninput,
                              const FpCudaOutputSchedule *schedule,
                              FpCudaPipelineTimes *times);

int run_pipeline_cuda(const FpCudaPipelineInput *in,
                      const FpCudaOutputSchedule *schedule,
                      double *crp_state_out,
                      double *cre_state_out,
                      double *eps_syn_out,
                      double *eps_gamma_out,
                      double *eps_nu_out,
                      FpCudaPipelineTimes *times);

#ifdef __cplusplus
}
#endif

#endif
