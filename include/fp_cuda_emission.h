#ifndef INCLUDED_fp_cuda_emission_h_
#define INCLUDED_fp_cuda_emission_h_

#include "fp_cuda_backend.h"

typedef struct {
    int ncell;
    int nfreq;
    int nx_tab;
    int ntheta_pitch;
    double xmin;
    int nlogb;
    double logb_min;
    double inv_dlogb;
    const double *d_fx_tab;
    const double *d_logx_tab;
    const double *d_lognu_syn;
    const double *d_lognu_crit;
    const double *d_pitch_kernel_table;
    const double *d_b_dyn;
    const double *d_logb;
    const double *d_cre_state;
    const double *d_cre_dp;
    const double *d_pitch_weight;
    double *d_eps_out;
} FpCudaSynchDeviceInput;

typedef struct {
    int ncell;
    int nbins;
    double egamma_min;
    double egamma_max;
    const double *d_n_gas;
    const double *d_crp_state;
    const double *d_crp_dp;
    const double *d_beta_p;
    const double *d_fga_flat;
    double *d_eps_out;
} FpCudaGammaDeviceInput;

typedef struct {
    int ncell;
    int nbins;
    double enu_min;
    double enu_max;
    const double *d_n_gas;
    const double *d_crp_state;
    const double *d_crp_dp;
    const double *d_beta_p;
    const double *d_fnu_flat;
    double *d_eps_out;
} FpCudaNeutrinoDeviceInput;

int fp_cuda_emit_synch_device(const FpCudaSynchDeviceInput *in,
                              double *elapsed_ms);
int fp_cuda_emit_gamma_device(const FpCudaGammaDeviceInput *in,
                              double *elapsed_ms);
int fp_cuda_emit_neutrino_device(const FpCudaNeutrinoDeviceInput *in,
                                 double *elapsed_ms);

#endif
