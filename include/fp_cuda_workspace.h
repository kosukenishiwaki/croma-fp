#ifndef INCLUDED_fp_cuda_workspace_h_
#define INCLUDED_fp_cuda_workspace_h_

#include <vector>

#include <cuda_runtime.h>

#include "fp_cuda_backend.h"

struct FpCudaWorkspaceScratch {
    int capacity_ncell = 0;
    int capacity_nfreq = 0;
    int capacity_nx_tab = 0;

    double *d_n_gas = nullptr, *d_kbt = nullptr, *d_b_field = nullptr, *d_divv = nullptr;
    double *d_l_turb_mpc = nullptr, *d_dv_imc = nullptr, *d_cs = nullptr, *d_beta_pl = nullptr;
    double *d_tracer_mass_msun = nullptr;
    double *d_rad_ic_batch = nullptr, *d_rad_ic_m1 = nullptr, *d_rad_ic_p1 = nullptr;
    double *d_qpi_cell = nullptr, *d_qepri_cell = nullptr, *d_qe_integral_cell = nullptr, *d_inje_cell = nullptr;
    double *d_crp_state = nullptr, *d_cre_state = nullptr;
    double *d_crp_radp = nullptr, *d_crp_tloss = nullptr;
    double *d_cre_radp = nullptr, *d_cre_tloss = nullptr;
    double *d_crp_radpm1 = nullptr, *d_crp_radpp1 = nullptr, *d_cre_radpm1 = nullptr, *d_cre_radpp1 = nullptr;
    double *d_dpp = nullptr, *d_dppe = nullptr, *d_dpp_off = nullptr, *d_dppe_off = nullptr;
    double *d_dppm1 = nullptr, *d_dppp1 = nullptr, *d_dppem1 = nullptr, *d_dppep1 = nullptr;
    double *d_dppm1_off = nullptr, *d_dppp1_off = nullptr, *d_dppem1_off = nullptr, *d_dppep1_off = nullptr;
    double *d_ccp_a = nullptr, *d_ccp_b = nullptr, *d_ccp_c = nullptr;
    double *d_cce_a = nullptr, *d_cce_b = nullptr, *d_cce_c = nullptr;
    double *d_ccp_a_off = nullptr, *d_ccp_b_off = nullptr, *d_ccp_c_off = nullptr;
    double *d_cce_a_off = nullptr, *d_cce_b_off = nullptr, *d_cce_c_off = nullptr;
    double *d_b_dyn = nullptr, *d_logb = nullptr;
    double *d_eps = nullptr;
    int *d_bucket_indices = nullptr;

    double *d_bg_n_gas_curr = nullptr, *d_bg_n_gas_next = nullptr;
    double *d_bg_kbt_curr = nullptr, *d_bg_kbt_next = nullptr;
    double *d_bg_b_field_curr = nullptr, *d_bg_b_field_next = nullptr;
    double *d_bg_divv_curr = nullptr, *d_bg_divv_next = nullptr;
    double *d_bg_l_turb_curr = nullptr, *d_bg_l_turb_next = nullptr;
    double *d_bg_dv_curr = nullptr, *d_bg_dv_next = nullptr;
    double *d_bg_cs_curr = nullptr, *d_bg_cs_next = nullptr;
    double *d_bg_beta_curr = nullptr, *d_bg_beta_next = nullptr;
    double *d_bg_rad_ic_curr = nullptr, *d_bg_rad_ic_next = nullptr;
    double *d_bg_rad_ic_m1_curr = nullptr, *d_bg_rad_ic_m1_next = nullptr;
    double *d_bg_rad_ic_p1_curr = nullptr, *d_bg_rad_ic_p1_next = nullptr;
};

struct FpCudaWorkspaceConstCache {
    const CRspectrum *crp_grid_key = nullptr;
    const CRspectrum *cre_grid_key = nullptr;
    const double *beta_p_key = nullptr;
    const double *fx_tab_key = nullptr;
    const double *logx_tab_key = nullptr;
    const double *pitch_kernel_table_key = nullptr;
    const double *pitch_weight_key = nullptr;
    const double *theta_key = nullptr;
    const double *nus_key = nullptr;
    int ntheta_pitch_key = 0;
    int pitch_kernel_nlogb_key = 0;
    double pitch_kernel_logb_min_key = 0.0;
    double pitch_kernel_inv_dlogb_key = 0.0;

    double *d_crp_p = nullptr, *d_crp_e = nullptr, *d_crp_dp = nullptr;
    double *d_crp_p2 = nullptr, *d_crp_exp_cut = nullptr;
    double *d_crp_sigma_pp = nullptr, *d_crp_sigmoid_pp = nullptr;
    double *d_cre_p = nullptr, *d_cre_dp = nullptr;
    double *d_cre_p2 = nullptr, *d_cre_exp_cut = nullptr;
    int crp_j_pp = 0;
    double crp_pm1_p2 = 0.0, crp_pp1_p2 = 0.0;
    double crp_pm1_exp_cut = 0.0, crp_pp1_exp_cut = 0.0;
    double cre_pm1_p2 = 0.0, cre_pp1_p2 = 0.0;
    double cre_pm1_exp_cut = 0.0, cre_pp1_exp_cut = 0.0;
    double *d_fqe_flat = nullptr;
    int *d_np_min_qe = nullptr;
    double *d_fx_tab = nullptr, *d_logx_tab = nullptr, *d_lognu_syn = nullptr, *d_lognu_crit = nullptr, *d_pitch_weight = nullptr;
    double *d_pitch_kernel_table = nullptr;
};

struct FpCudaWorkspaceEvents {
    cudaEvent_t coeff_start = nullptr, coeff_stop = nullptr;
    cudaEvent_t interp_start = nullptr, interp_stop = nullptr;
    cudaEvent_t secondary_start = nullptr, secondary_stop = nullptr;
    cudaEvent_t solve_start = nullptr, solve_stop = nullptr;
    cudaEvent_t synch_start = nullptr, synch_stop = nullptr;
};

struct FpCudaWorkspaceHostCache {
    std::vector<double> fqe_flat;
    std::vector<int> np_min_qe;
    double secondary_setup_ms = 0.0;
    std::vector<double> crp_p2;
    std::vector<double> crp_exp_cut;
    std::vector<double> crp_sigma_pp;
    std::vector<double> crp_sigmoid_pp;
    std::vector<double> cre_p2;
    std::vector<double> cre_exp_cut;
    std::vector<double> lognu_syn;
    std::vector<double> nu_crit_storage;
    std::vector<double *> nu_crit_rows;
    std::vector<double> lognu_crit;
    std::vector<double> merged_rad_ic_batch;
};

struct FpCudaWorkspaceResidentState {
    int state_valid = 0;
    int ncell = 0;
};

struct FpCudaWorkspaceRankState {
    int capacity_ncell = 0;
    int ncell = 0;
    int state_valid = 0;
    double *d_qpi = nullptr;
    double *d_qepri = nullptr;
    double *d_crp = nullptr;
    double *d_cre = nullptr;
};

struct FpCudaWorkspaceBgInterpCache {
    int valid = 0;
    int ncell = 0;
    const double *n_gas = nullptr, *n_gas_next = nullptr;
    const double *kbt = nullptr, *kbt_next = nullptr;
    const double *b_field = nullptr, *b_field_next = nullptr;
    const double *divv = nullptr, *divv_next = nullptr;
    const double *l_turb = nullptr, *l_turb_next = nullptr;
    const double *dv = nullptr, *dv_next = nullptr;
    const double *cs = nullptr, *cs_next = nullptr;
    const double *beta = nullptr, *beta_next = nullptr;
    const double *rad_ic = nullptr, *rad_ic_next = nullptr;
    const double *rad_ic_m1 = nullptr, *rad_ic_m1_next = nullptr;
    const double *rad_ic_p1 = nullptr, *rad_ic_p1_next = nullptr;
};

struct FpCudaPipelineWorkspace {
    FpCudaWorkspaceScratch scratch;
    FpCudaWorkspaceConstCache cache;
    FpCudaWorkspaceEvents events;
    FpCudaWorkspaceHostCache host;
    FpCudaWorkspaceResidentState resident;
    FpCudaWorkspaceRankState rank;
    FpCudaWorkspaceBgInterpCache bg_interp;
};

void workspace_realloc_simple_buffers(FpCudaPipelineWorkspace *ws,
                                      int ncell,
                                      int nfreq,
                                      int nx_tab);

void workspace_ensure_simple_constants(FpCudaPipelineWorkspace *ws,
                                       const FpCudaPipelineInput *in);

#endif
