#ifndef INCLUDED_fp_cuda_coeff_h_
#define INCLUDED_fp_cuda_coeff_h_

#include "fp_shared_core.h"

typedef struct {
    int ncell;
    int cell_blocks;
    int cell_threads;
    int active_model;
    double epmax;
    double psi_value;
    double mach_limit_value;
    int ttd_tacc_model;
    double t_acc_direct_gyr;
    double eta_dpp_cap_value;
    const double *d_n_gas;
    const double *d_kbt;
    const double *d_b_field;
    const double *d_divv;
    const double *d_rad_ic_batch;
    const double *d_rad_ic_m1;
    const double *d_rad_ic_p1;
    const double *d_l_turb_mpc;
    const double *d_dv_imc;
    const double *d_cs;
    const double *d_beta_pl;
    const double *d_crp_p;
    const double *d_crp_e;
    const double *d_crp_dp;
    const double *d_crp_p2;
    const double *d_crp_exp_cut;
    const double *d_crp_sigma_pp;
    const double *d_crp_sigmoid_pp;
    const double *d_cre_p;
    const double *d_cre_dp;
    const double *d_cre_p2;
    const double *d_cre_exp_cut;
    const double *d_tracer_mass_msun;
    const double *d_crp_state;
    const double *d_cre_state;
    double crp_pm1;
    double crp_pp1;
    double cre_pm1;
    double cre_pp1;
    int crp_j_pp;
    double crp_pm1_p2;
    double crp_pp1_p2;
    double crp_pm1_exp_cut;
    double crp_pp1_exp_cut;
    double cre_pm1_p2;
    double cre_pp1_p2;
    double cre_pm1_exp_cut;
    double cre_pp1_exp_cut;
    double *d_crp_radp;
    double *d_crp_tloss;
    double *d_cre_radp;
    double *d_cre_tloss;
    double *d_crp_radpm1;
    double *d_crp_radpp1;
    double *d_cre_radpm1;
    double *d_cre_radpp1;
    double *d_dpp;
    double *d_dppm1;
    double *d_dppp1;
    double *d_dppe;
    double *d_dppem1;
    double *d_dppep1;
    double *d_dpp_off;
    double *d_dppm1_off;
    double *d_dppp1_off;
    double *d_dppe_off;
    double *d_dppem1_off;
    double *d_dppep1_off;
} FpCudaCoeffPrepareInput;

typedef struct {
    int ncell;
    int cell_blocks;
    int cell_threads;
    double dt;
    const double *d_crp_p;
    const double *d_crp_dp;
    const double *d_cre_p;
    const double *d_cre_dp;
    double crp_pm1;
    double crp_pp1;
    double cre_pm1;
    double cre_pp1;
    const double *d_crp_radp;
    const double *d_crp_tloss;
    const double *d_dpp;
    const double *d_cre_radp;
    const double *d_cre_tloss;
    const double *d_dppe;
    const double *d_crp_radpm1;
    const double *d_crp_radpp1;
    const double *d_cre_radpm1;
    const double *d_cre_radpp1;
    const double *d_dppm1;
    const double *d_dppp1;
    const double *d_dppem1;
    const double *d_dppep1;
    double *d_ccp_a;
    double *d_ccp_b;
    double *d_ccp_c;
    double *d_cce_a;
    double *d_cce_b;
    double *d_cce_c;
} FpCudaCcBuildInput;

int fp_cuda_prepare_coeff_terms(const FpCudaCoeffPrepareInput *in,
                                const char *losses_label,
                                const char *model_label,
                                const char *off_label);

int fp_cuda_build_cc_coeffs(const FpCudaCcBuildInput *in,
                            const char *label);

#endif
