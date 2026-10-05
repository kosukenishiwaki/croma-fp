#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "fp_cuda_workspace.h"
#include "FP_Coef.h"
#include "Synchrotron.h"

namespace {

static void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(2);
    }
}

template <typename T>
static void ensure_device_buffer(T **ptr,
                                           size_t count,
                                           const char *what)
{
    if (*ptr != nullptr || count == 0) return;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T)), what);
}

template <typename T>
static void free_device_buffer(T **ptr)
{
    if (*ptr != nullptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
    }
}

static void release_cell_scratch(FpCudaWorkspaceScratch *scratch)
{
    free_device_buffer(&scratch->d_n_gas);
    free_device_buffer(&scratch->d_kbt);
    free_device_buffer(&scratch->d_b_field);
    free_device_buffer(&scratch->d_divv);
    free_device_buffer(&scratch->d_l_turb_mpc);
    free_device_buffer(&scratch->d_dv_imc);
    free_device_buffer(&scratch->d_cs);
    free_device_buffer(&scratch->d_beta_pl);
    free_device_buffer(&scratch->d_tracer_mass_msun);
    free_device_buffer(&scratch->d_rad_ic_batch);
    free_device_buffer(&scratch->d_rad_ic_m1);
    free_device_buffer(&scratch->d_rad_ic_p1);
    free_device_buffer(&scratch->d_qpi_cell);
    free_device_buffer(&scratch->d_qepri_cell);
    free_device_buffer(&scratch->d_qe_integral_cell);
    free_device_buffer(&scratch->d_inje_cell);
    free_device_buffer(&scratch->d_crp_state);
    free_device_buffer(&scratch->d_cre_state);
    free_device_buffer(&scratch->d_crp_radp);
    free_device_buffer(&scratch->d_crp_tloss);
    free_device_buffer(&scratch->d_cre_radp);
    free_device_buffer(&scratch->d_cre_tloss);
    free_device_buffer(&scratch->d_crp_radpm1);
    free_device_buffer(&scratch->d_crp_radpp1);
    free_device_buffer(&scratch->d_cre_radpm1);
    free_device_buffer(&scratch->d_cre_radpp1);
    free_device_buffer(&scratch->d_dpp);
    free_device_buffer(&scratch->d_dppe);
    free_device_buffer(&scratch->d_dpp_off);
    free_device_buffer(&scratch->d_dppe_off);
    free_device_buffer(&scratch->d_dppm1);
    free_device_buffer(&scratch->d_dppp1);
    free_device_buffer(&scratch->d_dppem1);
    free_device_buffer(&scratch->d_dppep1);
    free_device_buffer(&scratch->d_dppm1_off);
    free_device_buffer(&scratch->d_dppp1_off);
    free_device_buffer(&scratch->d_dppem1_off);
    free_device_buffer(&scratch->d_dppep1_off);
    free_device_buffer(&scratch->d_ccp_a);
    free_device_buffer(&scratch->d_ccp_b);
    free_device_buffer(&scratch->d_ccp_c);
    free_device_buffer(&scratch->d_cce_a);
    free_device_buffer(&scratch->d_cce_b);
    free_device_buffer(&scratch->d_cce_c);
    free_device_buffer(&scratch->d_ccp_a_off);
    free_device_buffer(&scratch->d_ccp_b_off);
    free_device_buffer(&scratch->d_ccp_c_off);
    free_device_buffer(&scratch->d_cce_a_off);
    free_device_buffer(&scratch->d_cce_b_off);
    free_device_buffer(&scratch->d_cce_c_off);
    free_device_buffer(&scratch->d_b_dyn);
    free_device_buffer(&scratch->d_logb);
    free_device_buffer(&scratch->d_eps);
    free_device_buffer(&scratch->d_bucket_indices);
    free_device_buffer(&scratch->d_bg_n_gas_curr); free_device_buffer(&scratch->d_bg_n_gas_next);
    free_device_buffer(&scratch->d_bg_kbt_curr); free_device_buffer(&scratch->d_bg_kbt_next);
    free_device_buffer(&scratch->d_bg_b_field_curr); free_device_buffer(&scratch->d_bg_b_field_next);
    free_device_buffer(&scratch->d_bg_divv_curr); free_device_buffer(&scratch->d_bg_divv_next);
    free_device_buffer(&scratch->d_bg_l_turb_curr); free_device_buffer(&scratch->d_bg_l_turb_next);
    free_device_buffer(&scratch->d_bg_dv_curr); free_device_buffer(&scratch->d_bg_dv_next);
    free_device_buffer(&scratch->d_bg_cs_curr); free_device_buffer(&scratch->d_bg_cs_next);
    free_device_buffer(&scratch->d_bg_beta_curr); free_device_buffer(&scratch->d_bg_beta_next);
    free_device_buffer(&scratch->d_bg_rad_ic_curr); free_device_buffer(&scratch->d_bg_rad_ic_next);
    free_device_buffer(&scratch->d_bg_rad_ic_m1_curr); free_device_buffer(&scratch->d_bg_rad_ic_m1_next);
    free_device_buffer(&scratch->d_bg_rad_ic_p1_curr); free_device_buffer(&scratch->d_bg_rad_ic_p1_next);
}

static void release_const_cache(FpCudaWorkspaceConstCache *cache)
{
    free_device_buffer(&cache->d_crp_p);
    free_device_buffer(&cache->d_crp_e);
    free_device_buffer(&cache->d_crp_dp);
    free_device_buffer(&cache->d_crp_p2);
    free_device_buffer(&cache->d_crp_exp_cut);
    free_device_buffer(&cache->d_crp_sigma_pp);
    free_device_buffer(&cache->d_crp_sigmoid_pp);
    free_device_buffer(&cache->d_cre_p);
    free_device_buffer(&cache->d_cre_dp);
    free_device_buffer(&cache->d_cre_p2);
    free_device_buffer(&cache->d_cre_exp_cut);
    free_device_buffer(&cache->d_fqe_flat);
    free_device_buffer(&cache->d_np_min_qe);
    free_device_buffer(&cache->d_fx_tab);
    free_device_buffer(&cache->d_logx_tab);
    free_device_buffer(&cache->d_lognu_syn);
    free_device_buffer(&cache->d_lognu_crit);
    free_device_buffer(&cache->d_pitch_weight);
    free_device_buffer(&cache->d_pitch_kernel_table);
}

static void release_rank_state(FpCudaWorkspaceRankState *rank)
{
    free_device_buffer(&rank->d_qpi);
    free_device_buffer(&rank->d_qepri);
    free_device_buffer(&rank->d_crp);
    free_device_buffer(&rank->d_cre);
    *rank = FpCudaWorkspaceRankState{};
}

static void release_events(FpCudaWorkspaceEvents *events)
{
    if (events->coeff_start) cudaEventDestroy(events->coeff_start);
    if (events->coeff_stop) cudaEventDestroy(events->coeff_stop);
    if (events->interp_start) cudaEventDestroy(events->interp_start);
    if (events->interp_stop) cudaEventDestroy(events->interp_stop);
    if (events->secondary_start) cudaEventDestroy(events->secondary_start);
    if (events->secondary_stop) cudaEventDestroy(events->secondary_stop);
    if (events->solve_start) cudaEventDestroy(events->solve_start);
    if (events->solve_stop) cudaEventDestroy(events->solve_stop);
    if (events->synch_start) cudaEventDestroy(events->synch_start);
    if (events->synch_stop) cudaEventDestroy(events->synch_stop);
    *events = FpCudaWorkspaceEvents{};
}

}  // namespace

void workspace_realloc_simple_buffers(FpCudaPipelineWorkspace *ws,
                                      int ncell,
                                      int nfreq,
                                      int nx_tab)
{
    FpCudaWorkspaceScratch &scratch = ws->scratch;
    FpCudaWorkspaceConstCache &cache = ws->cache;
    const size_t np_batch = (size_t)ncell * (size_t)np;
    const size_t npe_batch = (size_t)ncell * (size_t)npe;

    if (scratch.capacity_ncell < ncell) {
        release_cell_scratch(&scratch);
        scratch.capacity_ncell = ncell;
        ws->resident = FpCudaWorkspaceResidentState{};
        ws->bg_interp = FpCudaWorkspaceBgInterpCache{};
    }

    if (scratch.capacity_nfreq < nfreq) {
        free_device_buffer(&scratch.d_eps);
        free_device_buffer(&cache.d_lognu_syn);
        free_device_buffer(&cache.d_pitch_kernel_table);
        scratch.capacity_nfreq = nfreq;
        cache.nus_key = nullptr;
        cache.pitch_kernel_table_key = nullptr;
        cache.pitch_kernel_nlogb_key = 0;
    }

    if (scratch.capacity_nx_tab < nx_tab) {
        free_device_buffer(&cache.d_fx_tab);
        free_device_buffer(&cache.d_logx_tab);
        scratch.capacity_nx_tab = nx_tab;
        cache.fx_tab_key = nullptr;
        cache.logx_tab_key = nullptr;
    }

    ensure_device_buffer(&scratch.d_n_gas, (size_t)scratch.capacity_ncell, "alloc ws d_n_gas");
    ensure_device_buffer(&scratch.d_kbt, (size_t)scratch.capacity_ncell, "alloc ws d_kbt");
    ensure_device_buffer(&scratch.d_b_field, (size_t)scratch.capacity_ncell, "alloc ws d_b_field");
    ensure_device_buffer(&scratch.d_divv, (size_t)scratch.capacity_ncell, "alloc ws d_divv");
    ensure_device_buffer(&scratch.d_l_turb_mpc, (size_t)scratch.capacity_ncell, "alloc ws d_l_turb_mpc");
    ensure_device_buffer(&scratch.d_dv_imc, (size_t)scratch.capacity_ncell, "alloc ws d_dv_imc");
    ensure_device_buffer(&scratch.d_cs, (size_t)scratch.capacity_ncell, "alloc ws d_cs");
    ensure_device_buffer(&scratch.d_beta_pl, (size_t)scratch.capacity_ncell, "alloc ws d_beta_pl");
    ensure_device_buffer(&scratch.d_tracer_mass_msun, (size_t)scratch.capacity_ncell, "alloc ws d_tracer_mass_msun");
    ensure_device_buffer(&scratch.d_rad_ic_batch, (size_t)scratch.capacity_ncell * (size_t)npe, "alloc ws d_rad_ic_batch");
    ensure_device_buffer(&scratch.d_rad_ic_m1, (size_t)scratch.capacity_ncell, "alloc ws d_rad_ic_m1");
    ensure_device_buffer(&scratch.d_rad_ic_p1, (size_t)scratch.capacity_ncell, "alloc ws d_rad_ic_p1");
    ensure_device_buffer(&scratch.d_qpi_cell, (size_t)scratch.capacity_ncell * (size_t)np, "alloc ws d_qpi_cell");
    ensure_device_buffer(&scratch.d_qepri_cell, (size_t)scratch.capacity_ncell * (size_t)npe, "alloc ws d_qepri_cell");
    ensure_device_buffer(&scratch.d_qe_integral_cell, npe_batch, "alloc ws d_qe_integral_cell");
    ensure_device_buffer(&scratch.d_inje_cell, npe_batch, "alloc ws d_inje_cell");
    ensure_device_buffer(&scratch.d_crp_state, np_batch, "alloc ws d_crp_state");
    ensure_device_buffer(&scratch.d_cre_state, npe_batch, "alloc ws d_cre_state");
    ensure_device_buffer(&scratch.d_crp_radp, np_batch, "alloc ws d_crp_radp");
    ensure_device_buffer(&scratch.d_crp_tloss, np_batch, "alloc ws d_crp_tloss");
    ensure_device_buffer(&scratch.d_cre_radp, npe_batch, "alloc ws d_cre_radp");
    ensure_device_buffer(&scratch.d_cre_tloss, npe_batch, "alloc ws d_cre_tloss");
    ensure_device_buffer(&scratch.d_crp_radpm1, (size_t)scratch.capacity_ncell, "alloc ws d_crp_radpm1");
    ensure_device_buffer(&scratch.d_crp_radpp1, (size_t)scratch.capacity_ncell, "alloc ws d_crp_radpp1");
    ensure_device_buffer(&scratch.d_cre_radpm1, (size_t)scratch.capacity_ncell, "alloc ws d_cre_radpm1");
    ensure_device_buffer(&scratch.d_cre_radpp1, (size_t)scratch.capacity_ncell, "alloc ws d_cre_radpp1");
    ensure_device_buffer(&scratch.d_dpp, np_batch, "alloc ws d_dpp");
    ensure_device_buffer(&scratch.d_dppe, npe_batch, "alloc ws d_dppe");
    ensure_device_buffer(&scratch.d_dpp_off, np_batch, "alloc ws d_dpp_off");
    ensure_device_buffer(&scratch.d_dppe_off, npe_batch, "alloc ws d_dppe_off");
    ensure_device_buffer(&scratch.d_dppm1, (size_t)scratch.capacity_ncell, "alloc ws d_dppm1");
    ensure_device_buffer(&scratch.d_dppp1, (size_t)scratch.capacity_ncell, "alloc ws d_dppp1");
    ensure_device_buffer(&scratch.d_dppem1, (size_t)scratch.capacity_ncell, "alloc ws d_dppem1");
    ensure_device_buffer(&scratch.d_dppep1, (size_t)scratch.capacity_ncell, "alloc ws d_dppep1");
    ensure_device_buffer(&scratch.d_dppm1_off, (size_t)scratch.capacity_ncell, "alloc ws d_dppm1_off");
    ensure_device_buffer(&scratch.d_dppp1_off, (size_t)scratch.capacity_ncell, "alloc ws d_dppp1_off");
    ensure_device_buffer(&scratch.d_dppem1_off, (size_t)scratch.capacity_ncell, "alloc ws d_dppem1_off");
    ensure_device_buffer(&scratch.d_dppep1_off, (size_t)scratch.capacity_ncell, "alloc ws d_dppep1_off");
    ensure_device_buffer(&scratch.d_ccp_a, np_batch, "alloc ws d_ccp_a");
    ensure_device_buffer(&scratch.d_ccp_b, np_batch, "alloc ws d_ccp_b");
    ensure_device_buffer(&scratch.d_ccp_c, np_batch, "alloc ws d_ccp_c");
    ensure_device_buffer(&scratch.d_cce_a, npe_batch, "alloc ws d_cce_a");
    ensure_device_buffer(&scratch.d_cce_b, npe_batch, "alloc ws d_cce_b");
    ensure_device_buffer(&scratch.d_cce_c, npe_batch, "alloc ws d_cce_c");
    ensure_device_buffer(&scratch.d_ccp_a_off, np_batch, "alloc ws d_ccp_a_off");
    ensure_device_buffer(&scratch.d_ccp_b_off, np_batch, "alloc ws d_ccp_b_off");
    ensure_device_buffer(&scratch.d_ccp_c_off, np_batch, "alloc ws d_ccp_c_off");
    ensure_device_buffer(&scratch.d_cce_a_off, npe_batch, "alloc ws d_cce_a_off");
    ensure_device_buffer(&scratch.d_cce_b_off, npe_batch, "alloc ws d_cce_b_off");
    ensure_device_buffer(&scratch.d_cce_c_off, npe_batch, "alloc ws d_cce_c_off");
    ensure_device_buffer(&scratch.d_b_dyn, (size_t)scratch.capacity_ncell, "alloc ws d_b_dyn");
    ensure_device_buffer(&scratch.d_logb, (size_t)scratch.capacity_ncell, "alloc ws d_logb");
    ensure_device_buffer(&scratch.d_eps, (size_t)scratch.capacity_ncell * (size_t)scratch.capacity_nfreq, "alloc ws d_eps");
    ensure_device_buffer(&scratch.d_bucket_indices, (size_t)scratch.capacity_ncell, "alloc ws d_bucket_indices");

    ensure_device_buffer(&cache.d_fx_tab, (size_t)scratch.capacity_nx_tab, "alloc ws d_fx_tab");
    ensure_device_buffer(&cache.d_logx_tab, (size_t)scratch.capacity_nx_tab, "alloc ws d_logx_tab");
    ensure_device_buffer(&cache.d_lognu_syn, (size_t)scratch.capacity_nfreq, "alloc ws d_lognu_syn");
}

void workspace_ensure_simple_constants(FpCudaPipelineWorkspace *ws,
                                       const FpCudaPipelineInput *in)
{
    FpCudaWorkspaceScratch &scratch = ws->scratch;
    FpCudaWorkspaceConstCache &cache = ws->cache;
    FpCudaWorkspaceHostCache &host = ws->host;
    const FpSynchEmissionBatchInput *synch_in = in->synch_in;
    constexpr double p_cut = 1.0;
    constexpr double pe_cut = 1.0;

    host.secondary_setup_ms = 0.0;
    if (cache.d_crp_p == nullptr || cache.d_crp_p2 == nullptr ||
        cache.d_crp_exp_cut == nullptr || cache.d_crp_sigma_pp == nullptr ||
        cache.d_crp_sigmoid_pp == nullptr || cache.crp_grid_key != in->crp_grid) {
        const double *crp_sigma_pp = nullptr;
        const double *crp_sigmoid_pp = nullptr;
        ensure_device_buffer(&cache.d_crp_p, (size_t)np, "alloc ws d_crp_p");
        ensure_device_buffer(&cache.d_crp_e, (size_t)np, "alloc ws d_crp_e");
        ensure_device_buffer(&cache.d_crp_dp, (size_t)np, "alloc ws d_crp_dp");
        ensure_device_buffer(&cache.d_crp_p2, (size_t)np, "alloc ws d_crp_p2");
        ensure_device_buffer(&cache.d_crp_exp_cut, (size_t)np, "alloc ws d_crp_exp_cut");
        ensure_device_buffer(&cache.d_crp_sigma_pp, (size_t)np, "alloc ws d_crp_sigma_pp");
        ensure_device_buffer(&cache.d_crp_sigmoid_pp, (size_t)np, "alloc ws d_crp_sigmoid_pp");
        host.crp_p2.resize((size_t)np);
        host.crp_exp_cut.resize((size_t)np);
        host.crp_sigma_pp.resize((size_t)np);
        host.crp_sigmoid_pp.resize((size_t)np);
        crp_sigma_pp = momentumdiff_crp_sigma_pp_cache(in->crp_grid);
        crp_sigmoid_pp = momentumdiff_crp_sigmoid_pp_cache(in->crp_grid);
        cache.crp_j_pp = momentumdiff_crp_j_pp_cache(in->crp_grid);
        for (int jp = 0; jp < np; jp++) {
            const double p = in->crp_grid->p[jp];
            host.crp_p2[(size_t)jp] = p * p;
            host.crp_exp_cut[(size_t)jp] = std::exp(-p_cut / p);
            host.crp_sigma_pp[(size_t)jp] = crp_sigma_pp[jp];
            host.crp_sigmoid_pp[(size_t)jp] = crp_sigmoid_pp[jp];
        }
        cache.crp_pm1_p2 = in->crp_grid->pm1 * in->crp_grid->pm1;
        cache.crp_pp1_p2 = in->crp_grid->pp1 * in->crp_grid->pp1;
        cache.crp_pm1_exp_cut = std::exp(-p_cut / in->crp_grid->pm1);
        cache.crp_pp1_exp_cut = std::exp(-p_cut / in->crp_grid->pp1);
        cuda_check(cudaMemcpy(cache.d_crp_p, in->crp_grid->p, (size_t)np * sizeof(double), cudaMemcpyHostToDevice), "copy ws crp_p");
        cuda_check(cudaMemcpy(cache.d_crp_e, in->crp_grid->E, (size_t)np * sizeof(double), cudaMemcpyHostToDevice), "copy ws crp_e");
        cuda_check(cudaMemcpy(cache.d_crp_dp, in->crp_grid->dp, (size_t)np * sizeof(double), cudaMemcpyHostToDevice), "copy ws crp_dp");
        cuda_check(cudaMemcpy(cache.d_crp_p2, host.crp_p2.data(), (size_t)np * sizeof(double), cudaMemcpyHostToDevice), "copy ws crp_p2");
        cuda_check(cudaMemcpy(cache.d_crp_exp_cut, host.crp_exp_cut.data(), (size_t)np * sizeof(double), cudaMemcpyHostToDevice), "copy ws crp_exp_cut");
        cuda_check(cudaMemcpy(cache.d_crp_sigma_pp, host.crp_sigma_pp.data(), (size_t)np * sizeof(double), cudaMemcpyHostToDevice), "copy ws crp_sigma_pp");
        cuda_check(cudaMemcpy(cache.d_crp_sigmoid_pp, host.crp_sigmoid_pp.data(), (size_t)np * sizeof(double), cudaMemcpyHostToDevice), "copy ws crp_sigmoid_pp");
        cache.crp_grid_key = in->crp_grid;
        cache.beta_p_key = nullptr;
    }
    if (cache.d_cre_p == nullptr || cache.d_cre_p2 == nullptr ||
        cache.d_cre_exp_cut == nullptr || cache.cre_grid_key != in->cre_grid) {
        ensure_device_buffer(&cache.d_cre_p, (size_t)npe, "alloc ws d_cre_p");
        ensure_device_buffer(&cache.d_cre_dp, (size_t)npe, "alloc ws d_cre_dp");
        ensure_device_buffer(&cache.d_cre_p2, (size_t)npe, "alloc ws d_cre_p2");
        ensure_device_buffer(&cache.d_cre_exp_cut, (size_t)npe, "alloc ws d_cre_exp_cut");
        host.cre_p2.resize((size_t)npe);
        host.cre_exp_cut.resize((size_t)npe);
        for (int je = 0; je < npe; je++) {
            const double p = in->cre_grid->p[je];
            host.cre_p2[(size_t)je] = p * p;
            host.cre_exp_cut[(size_t)je] = std::exp(-pe_cut / p);
        }
        cache.cre_pm1_p2 = in->cre_grid->pm1 * in->cre_grid->pm1;
        cache.cre_pp1_p2 = in->cre_grid->pp1 * in->cre_grid->pp1;
        cache.cre_pm1_exp_cut = std::exp(-pe_cut / in->cre_grid->pm1);
        cache.cre_pp1_exp_cut = std::exp(-pe_cut / in->cre_grid->pp1);
        cuda_check(cudaMemcpy(cache.d_cre_p, in->cre_grid->p, (size_t)npe * sizeof(double), cudaMemcpyHostToDevice), "copy ws cre_p");
        cuda_check(cudaMemcpy(cache.d_cre_dp, in->cre_grid->dp, (size_t)npe * sizeof(double), cudaMemcpyHostToDevice), "copy ws cre_dp");
        cuda_check(cudaMemcpy(cache.d_cre_p2, host.cre_p2.data(), (size_t)npe * sizeof(double), cudaMemcpyHostToDevice), "copy ws cre_p2");
        cuda_check(cudaMemcpy(cache.d_cre_exp_cut, host.cre_exp_cut.data(), (size_t)npe * sizeof(double), cudaMemcpyHostToDevice), "copy ws cre_exp_cut");
        cache.cre_grid_key = in->cre_grid;
        cache.beta_p_key = nullptr;
        cache.theta_key = nullptr;
        cache.ntheta_pitch_key = 0;
    }
    if (cache.beta_p_key != in->beta_p || cache.crp_grid_key != in->crp_grid || cache.cre_grid_key != in->cre_grid) {
        const auto secondary_t0 = std::chrono::steady_clock::now();
        host.fqe_flat.resize((size_t)npe * (size_t)np);
        host.np_min_qe.resize((size_t)npe);
        prepare_secondary_kernel_flat(in->crp_grid, in->cre_grid, in->beta_p,
                                      host.fqe_flat.data(), host.np_min_qe.data());
        ensure_device_buffer(&cache.d_fqe_flat, host.fqe_flat.size(), "alloc ws d_fqe_flat");
        ensure_device_buffer(&cache.d_np_min_qe, host.np_min_qe.size(), "alloc ws d_np_min_qe");
        cuda_check(cudaMemcpy(cache.d_fqe_flat, host.fqe_flat.data(),
                              host.fqe_flat.size() * sizeof(double), cudaMemcpyHostToDevice),
                   "copy ws fqe_flat");
        cuda_check(cudaMemcpy(cache.d_np_min_qe, host.np_min_qe.data(),
                              host.np_min_qe.size() * sizeof(int), cudaMemcpyHostToDevice),
                   "copy ws np_min_qe");
        cache.beta_p_key = in->beta_p;
        host.secondary_setup_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - secondary_t0).count();
    }
    if (synch_in != nullptr) {
        if (cache.ntheta_pitch_key != synch_in->ntheta_pitch) {
            free_device_buffer(&cache.d_pitch_weight);
            free_device_buffer(&cache.d_lognu_crit);
            cache.pitch_weight_key = nullptr;
            cache.theta_key = nullptr;
            cache.ntheta_pitch_key = synch_in->ntheta_pitch;
        }
        if (cache.d_pitch_weight == nullptr || cache.pitch_weight_key != synch_in->pitch_weight) {
            ensure_device_buffer(&cache.d_pitch_weight, (size_t)synch_in->ntheta_pitch, "alloc ws d_pitch_weight");
            cuda_check(cudaMemcpy(cache.d_pitch_weight, synch_in->pitch_weight,
                                  (size_t)synch_in->ntheta_pitch * sizeof(double), cudaMemcpyHostToDevice),
                       "copy ws pitch_weight");
            cache.pitch_weight_key = synch_in->pitch_weight;
        }
        if (cache.fx_tab_key != synch_in->fx_tab) {
            cuda_check(cudaMemcpy(cache.d_fx_tab, synch_in->fx_tab,
                                  (size_t)in->nx_tab * sizeof(double), cudaMemcpyHostToDevice),
                       "copy ws fx_tab");
            cache.fx_tab_key = synch_in->fx_tab;
        }
        if (cache.logx_tab_key != synch_in->logx_tab) {
            cuda_check(cudaMemcpy(cache.d_logx_tab, synch_in->logx_tab,
                                  (size_t)in->nx_tab * sizeof(double), cudaMemcpyHostToDevice),
                       "copy ws logx_tab");
            cache.logx_tab_key = synch_in->logx_tab;
        }
        if (cache.nus_key != synch_in->nus || scratch.capacity_nfreq != in->nfreq) {
            host.lognu_syn.resize((size_t)in->nfreq);
            for (int nf = 0; nf < in->nfreq; nf++) {
                host.lognu_syn[(size_t)nf] = std::log10(synch_in->nus[nf]);
            }
            cuda_check(cudaMemcpy(cache.d_lognu_syn, host.lognu_syn.data(),
                                  (size_t)in->nfreq * sizeof(double), cudaMemcpyHostToDevice),
                       "copy ws lognu_syn");
            cache.nus_key = synch_in->nus;
        }
        if (cache.d_lognu_crit == nullptr || cache.theta_key != synch_in->theta || cache.cre_grid_key != in->cre_grid) {
            host.nu_crit_storage.resize((size_t)npe * (size_t)synch_in->ntheta_pitch);
            host.nu_crit_rows.resize((size_t)npe);
            host.lognu_crit.resize((size_t)npe * (size_t)synch_in->ntheta_pitch);
            for (int je = 0; je < npe; je++) {
                host.nu_crit_rows[(size_t)je] = host.nu_crit_storage.data() + (size_t)je * (size_t)synch_in->ntheta_pitch;
            }
            SYN_nu_crit_subB(host.nu_crit_rows.data(), (double *)synch_in->theta, (double *)in->cre_grid->p);
            for (int je = 0; je < npe; je++) {
                for (int k = 0; k < synch_in->ntheta_pitch; k++) {
                    const size_t off = (size_t)je * (size_t)synch_in->ntheta_pitch + (size_t)k;
                    host.lognu_crit[off] = std::log10(host.nu_crit_storage[off]);
                }
            }
            ensure_device_buffer(&cache.d_lognu_crit, host.lognu_crit.size(), "alloc ws d_lognu_crit");
            cuda_check(cudaMemcpy(cache.d_lognu_crit, host.lognu_crit.data(),
                                  host.lognu_crit.size() * sizeof(double), cudaMemcpyHostToDevice),
                       "copy ws lognu_crit");
            cache.theta_key = synch_in->theta;
        }
        if (synch_in->pitch_kernel_table != nullptr &&
            synch_in->nlogb > 1 &&
            synch_in->inv_dlogb > 0.0) {
            const size_t table_size =
                (size_t)synch_in->nlogb * (size_t)in->nfreq * (size_t)npe;
            if (cache.d_pitch_kernel_table == nullptr ||
                cache.pitch_kernel_table_key != synch_in->pitch_kernel_table ||
                cache.pitch_kernel_nlogb_key != synch_in->nlogb ||
                cache.pitch_kernel_logb_min_key != synch_in->logb_min ||
                cache.pitch_kernel_inv_dlogb_key != synch_in->inv_dlogb) {
                free_device_buffer(&cache.d_pitch_kernel_table);
                ensure_device_buffer(&cache.d_pitch_kernel_table, table_size,
                                               "alloc ws d_pitch_kernel_table");
                cuda_check(cudaMemcpy(cache.d_pitch_kernel_table,
                                      synch_in->pitch_kernel_table,
                                      table_size * sizeof(double),
                                      cudaMemcpyHostToDevice),
                           "copy ws pitch_kernel_table");
                cache.pitch_kernel_table_key = synch_in->pitch_kernel_table;
                cache.pitch_kernel_nlogb_key = synch_in->nlogb;
                cache.pitch_kernel_logb_min_key = synch_in->logb_min;
                cache.pitch_kernel_inv_dlogb_key = synch_in->inv_dlogb;
            }
        } else {
            cache.pitch_kernel_table_key = nullptr;
            cache.pitch_kernel_nlogb_key = 0;
        }
    }
}

extern "C" FpCudaPipelineWorkspace *cuda_pipeline_workspace_create(void)
{
    return new FpCudaPipelineWorkspace();
}

extern "C" int cuda_rank_state_upload(FpCudaPipelineWorkspace *workspace,
                                      int ncell,
                                      const double *qpi,
                                      const double *qepri,
                                      const double *crp,
                                      const double *cre,
                                      double *elapsed_ms)
{
    FpCudaWorkspaceRankState *rank;
    const size_t np_batch = (size_t)ncell * (size_t)np;
    const size_t npe_batch = (size_t)ncell * (size_t)npe;
    const auto t0 = std::chrono::steady_clock::now();

    if (elapsed_ms != nullptr) *elapsed_ms = 0.0;
    if (workspace == nullptr || ncell < 0 ||
        qpi == nullptr || qepri == nullptr || crp == nullptr || cre == nullptr) {
        return -1;
    }

    rank = &workspace->rank;
    if (rank->capacity_ncell < ncell) {
        release_rank_state(rank);
        rank->capacity_ncell = ncell;
    }

    ensure_device_buffer(&rank->d_qpi, (size_t)rank->capacity_ncell * (size_t)np,
                                   "alloc rank d_qpi");
    ensure_device_buffer(&rank->d_qepri, (size_t)rank->capacity_ncell * (size_t)npe,
                                   "alloc rank d_qepri");
    ensure_device_buffer(&rank->d_crp, (size_t)rank->capacity_ncell * (size_t)np,
                                   "alloc rank d_crp");
    ensure_device_buffer(&rank->d_cre, (size_t)rank->capacity_ncell * (size_t)npe,
                                   "alloc rank d_cre");

    cuda_check(cudaMemcpy(rank->d_qpi, qpi, np_batch * sizeof(double), cudaMemcpyHostToDevice),
               "copy rank qpi");
    cuda_check(cudaMemcpy(rank->d_qepri, qepri, npe_batch * sizeof(double), cudaMemcpyHostToDevice),
               "copy rank qepri");
    cuda_check(cudaMemcpy(rank->d_crp, crp, np_batch * sizeof(double), cudaMemcpyHostToDevice),
               "copy rank crp");
    cuda_check(cudaMemcpy(rank->d_cre, cre, npe_batch * sizeof(double), cudaMemcpyHostToDevice),
               "copy rank cre");
    rank->ncell = ncell;
    rank->state_valid = 1;

    if (elapsed_ms != nullptr) {
        *elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }
    return 0;
}

extern "C" int cuda_rank_state_download(FpCudaPipelineWorkspace *workspace,
                                        int ncell,
                                        double *crp,
                                        double *cre,
                                        double *elapsed_ms)
{
    FpCudaWorkspaceRankState *rank;
    const size_t np_batch = (size_t)ncell * (size_t)np;
    const size_t npe_batch = (size_t)ncell * (size_t)npe;
    const auto t0 = std::chrono::steady_clock::now();

    if (elapsed_ms != nullptr) *elapsed_ms = 0.0;
    if (workspace == nullptr || ncell < 0 || crp == nullptr || cre == nullptr) {
        return -1;
    }
    rank = &workspace->rank;
    if (!rank->state_valid || rank->ncell != ncell ||
        rank->d_crp == nullptr || rank->d_cre == nullptr) {
        return -1;
    }

    cuda_check(cudaMemcpy(crp, rank->d_crp, np_batch * sizeof(double), cudaMemcpyDeviceToHost),
               "copy rank crp out");
    cuda_check(cudaMemcpy(cre, rank->d_cre, npe_batch * sizeof(double), cudaMemcpyDeviceToHost),
               "copy rank cre out");
    cuda_check(cudaDeviceSynchronize(), "sync rank download");

    if (elapsed_ms != nullptr) {
        *elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }
    return 0;
}

extern "C" void cuda_pipeline_workspace_destroy(FpCudaPipelineWorkspace *workspace)
{
    if (workspace == nullptr) return;
    release_events(&workspace->events);
    release_cell_scratch(&workspace->scratch);
    release_const_cache(&workspace->cache);
    release_rank_state(&workspace->rank);
    delete workspace;
}
