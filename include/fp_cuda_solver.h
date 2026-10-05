#ifndef INCLUDED_fp_cuda_solver_h_
#define INCLUDED_fp_cuda_solver_h_

#include "fp_cuda_backend.h"

int fp_cuda_solve_device_batch(int nsys,
                               int nrow,
                               double dt,
                               const double *d_a,
                               const double *d_b,
                               const double *d_c,
                               const double *d_source,
                               double *d_state,
                               const char *label,
                               int *used_pcr_out);

typedef struct {
    int nsys;
    int nstep;
    int use_windowed_reacc;
    int on_start_step;
    int on_end_step;
    double dt;
    const double *d_ccp_a;
    const double *d_ccp_b;
    const double *d_ccp_c;
    const double *d_ccp_a_off;
    const double *d_ccp_b_off;
    const double *d_ccp_c_off;
    const double *d_cce_a;
    const double *d_cce_b;
    const double *d_cce_c;
    const double *d_cce_a_off;
    const double *d_cce_b_off;
    const double *d_cce_c_off;
    const double *d_qpi_cell;
    const double *d_qepri_cell;
    const double *d_n_gas;
    const double *d_crp_dp;
    const int *d_np_min_qe;
    const double *d_fqe_flat;
    double *d_qe_integral_cell;
    double *d_inje_cell;
    double *d_crp_state;
    double *d_cre_state;
} FpCudaFusedTransportInput;

int fp_cuda_run_fused_transport(const FpCudaFusedTransportInput *in,
                                const char *label,
                                int *used_pcr_proton_out,
                                int *used_pcr_electron_out);

#endif
