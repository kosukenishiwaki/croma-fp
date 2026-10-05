#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "fp_cuda_backend.h"
#include "params.h"
#include "fp_cuda_coeff.h"
#include "fp_cuda_emission.h"
#include "fp_cuda_solver.h"
#include "fp_cuda_workspace.h"

namespace {

constexpr double kGyr = 3.1536e16;

static void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(2);
    }
}

static bool should_emit_synch(const FpCudaOutputSchedule *schedule,
                              int step_index,
                              int nstep)
{
    const int step_1based = step_index + 1;

    if (schedule == nullptr || nstep <= 0) return false;
    const int physical_snapshot = nsnp_i + step_index;
    if (schedule->physical_min >= 0 && physical_snapshot < schedule->physical_min) return false;
    if (schedule->physical_max >= 0 && physical_snapshot > schedule->physical_max) return false;
    if (schedule->emit_all_steps) return true;
    if (schedule->emit_final_only) return step_1based == nstep;
    if (schedule->emit_stride > 0) return (step_1based % schedule->emit_stride) == 0;
    for (int i = 0; i < schedule->nselected_steps; i++) {
        if (schedule->selected_steps[i] == step_1based) return true;
    }
    return false;
}

static bool has_intermediate_synch(const FpCudaOutputSchedule *schedule,
                                   int nstep)
{
    for (int istep = 0; istep < nstep - 1; istep++) {
        if (should_emit_synch(schedule, istep, nstep)) return true;
    }
    return false;
}

static bool wants_synch(const FpCudaOutputSchedule *schedule);

static void ensure_timing_events(FpCudaWorkspaceEvents *events)
{
    if (events == nullptr || events->coeff_start != nullptr) return;

    cuda_check(cudaEventCreate(&events->coeff_start), "event ws coeff_start");
    cuda_check(cudaEventCreate(&events->coeff_stop), "event ws coeff_stop");
    cuda_check(cudaEventCreate(&events->interp_start), "event ws interp_start");
    cuda_check(cudaEventCreate(&events->interp_stop), "event ws interp_stop");
    cuda_check(cudaEventCreate(&events->secondary_start), "event ws secondary_start");
    cuda_check(cudaEventCreate(&events->secondary_stop), "event ws secondary_stop");
    cuda_check(cudaEventCreate(&events->solve_start), "event ws solve_start");
    cuda_check(cudaEventCreate(&events->solve_stop), "event ws solve_stop");
    cuda_check(cudaEventCreate(&events->synch_start), "event ws synch_start");
    cuda_check(cudaEventCreate(&events->synch_stop), "event ws synch_stop");
}

__global__ void prepare_secondary_sources_cell_major_kernel(int ncell,
                                                            const double *n_gas,
                                                            const double *crp_state_cell_major,
                                                            const double *crp_dp,
                                                            const int *np_min_qe,
                                                            const double *fqe_flat,
                                                            int fqe_transposed,
                                                            const double *qepri_cell_major,
                                                            double *qe_integral_cell_major,
                                                            double *inje_cell_major)
{
    const int icell = blockIdx.x;
    const int je = threadIdx.x;
    __shared__ double crp_row[np];

    if (icell >= ncell) return;
    (void)crp_dp;

    if (threadIdx.x < np) {
        crp_row[threadIdx.x] = crp_state_cell_major[(size_t)icell * (size_t)np + (size_t)threadIdx.x];
    }
    __syncthreads();

    if (je >= npe) return;

    int jp0 = np_min_qe[je];
    const size_t ecell_off = (size_t)icell * (size_t)npe + (size_t)je;
    const size_t fqe_off = (size_t)je * (size_t)np;
    double qint = 0.0;

    if (jp0 < 0) jp0 = 0;
    if (jp0 > np) jp0 = np;

    for (int jp = jp0; jp < np; jp++) {
        const size_t woff = fqe_transposed
            ? (size_t)jp * (size_t)npe + (size_t)je
            : fqe_off + (size_t)jp;
        qint += crp_row[jp] * fqe_flat[woff];
    }

    qe_integral_cell_major[ecell_off] = qint;
    inje_cell_major[ecell_off] = qint * kGyr * n_gas[icell] + qepri_cell_major[ecell_off];
}

__global__ void gather_rank_rows_kernel(int ncell,
                                        int nrow,
                                        const int *indices,
                                        const double *rank_rows,
                                        double *bucket_rows)
{
    const size_t total = (size_t)ncell * (size_t)nrow;
    const size_t linear = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (linear >= total) return;

    const int icell = (int)(linear / (size_t)nrow);
    const int irow = (int)(linear - (size_t)icell * (size_t)nrow);
    const int src_cell = indices[icell];
    bucket_rows[linear] = rank_rows[(size_t)src_cell * (size_t)nrow + (size_t)irow];
}

__global__ void lerp_array_kernel(size_t count,
                                  const double *curr,
                                  const double *next,
                                  double alpha,
                                  double *out)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= count) return;
    out[i] = curr[i] + alpha * (next[i] - curr[i]);
}

static void ensure_device_double_buffer(double **ptr, size_t count, const char *what)
{
    if (*ptr != nullptr || count == 0) return;
    cuda_check(cudaMalloc(reinterpret_cast<void **>(ptr), count * sizeof(double)), what);
}

static void launch_lerp(size_t count,
                        const double *curr,
                        const double *next,
                        double alpha,
                        double *out,
                        const char *label)
{
    const int threads = 256;
    const int blocks = (int)((count + (size_t)threads - 1) / (size_t)threads);
    if (count == 0) return;
    lerp_array_kernel<<<blocks, threads>>>(count, curr, next, alpha, out);
    cuda_check(cudaGetLastError(), label);
}

static bool coeff_has_device_interp(const FpCoeffBatchInput *in)
{
    return in != nullptr && in->device_interp_background != 0 &&
           in->n_gas_next != nullptr && in->kbt_next != nullptr &&
           in->b_field_next != nullptr && in->divv_gyr_next != nullptr &&
           in->l_turb_mpc_next != nullptr && in->dv_imc_next != nullptr &&
           in->cs_next != nullptr && in->beta_pl_next != nullptr &&
           in->rad_ic_batch_next != nullptr && in->rad_ic_m1_next != nullptr &&
           in->rad_ic_p1_next != nullptr;
}

static bool bg_interp_cache_matches(const FpCudaWorkspaceBgInterpCache &cache,
                                    const FpCoeffBatchInput *in)
{
    return cache.valid && cache.ncell == in->ncell &&
           cache.n_gas == in->n_gas && cache.n_gas_next == in->n_gas_next &&
           cache.kbt == in->kbt && cache.kbt_next == in->kbt_next &&
           cache.b_field == in->b_field && cache.b_field_next == in->b_field_next &&
           cache.divv == in->divv_gyr && cache.divv_next == in->divv_gyr_next &&
           cache.l_turb == in->l_turb_mpc && cache.l_turb_next == in->l_turb_mpc_next &&
           cache.dv == in->dv_imc && cache.dv_next == in->dv_imc_next &&
           cache.cs == in->cs && cache.cs_next == in->cs_next &&
           cache.beta == in->beta_pl && cache.beta_next == in->beta_pl_next &&
           cache.rad_ic == in->rad_ic_batch && cache.rad_ic_next == in->rad_ic_batch_next &&
           cache.rad_ic_m1 == in->rad_ic_m1 && cache.rad_ic_m1_next == in->rad_ic_m1_next &&
           cache.rad_ic_p1 == in->rad_ic_p1 && cache.rad_ic_p1_next == in->rad_ic_p1_next;
}

static void update_bg_interp_cache(FpCudaWorkspaceBgInterpCache *cache,
                                   const FpCoeffBatchInput *in)
{
    cache->valid = 1;
    cache->ncell = in->ncell;
    cache->n_gas = in->n_gas; cache->n_gas_next = in->n_gas_next;
    cache->kbt = in->kbt; cache->kbt_next = in->kbt_next;
    cache->b_field = in->b_field; cache->b_field_next = in->b_field_next;
    cache->divv = in->divv_gyr; cache->divv_next = in->divv_gyr_next;
    cache->l_turb = in->l_turb_mpc; cache->l_turb_next = in->l_turb_mpc_next;
    cache->dv = in->dv_imc; cache->dv_next = in->dv_imc_next;
    cache->cs = in->cs; cache->cs_next = in->cs_next;
    cache->beta = in->beta_pl; cache->beta_next = in->beta_pl_next;
    cache->rad_ic = in->rad_ic_batch; cache->rad_ic_next = in->rad_ic_batch_next;
    cache->rad_ic_m1 = in->rad_ic_m1; cache->rad_ic_m1_next = in->rad_ic_m1_next;
    cache->rad_ic_p1 = in->rad_ic_p1; cache->rad_ic_p1_next = in->rad_ic_p1_next;
}

static void copy_pair_to_device(double *d_curr,
                                double *d_next,
                                const double *h_curr,
                                const double *h_next,
                                size_t count,
                                const char *what_curr,
                                const char *what_next)
{
    cuda_check(cudaMemcpy(d_curr, h_curr, count * sizeof(double), cudaMemcpyHostToDevice), what_curr);
    cuda_check(cudaMemcpy(d_next, h_next, count * sizeof(double), cudaMemcpyHostToDevice), what_next);
}

static bool prepare_device_interpolated_background(FpCudaPipelineWorkspace *ws,
                                                   const FpCoeffBatchInput *in)
{
    FpCudaWorkspaceScratch &scratch = ws->scratch;
    if (!coeff_has_device_interp(in)) return false;

    const size_t ncell = (size_t)in->ncell;
    const size_t ncell_npe = ncell * (size_t)npe;
    const double alpha = in->interp_alpha;

    ensure_device_double_buffer(&scratch.d_bg_n_gas_curr, ncell, "alloc ws d_bg_n_gas_curr");
    ensure_device_double_buffer(&scratch.d_bg_n_gas_next, ncell, "alloc ws d_bg_n_gas_next");
    ensure_device_double_buffer(&scratch.d_bg_kbt_curr, ncell, "alloc ws d_bg_kbt_curr");
    ensure_device_double_buffer(&scratch.d_bg_kbt_next, ncell, "alloc ws d_bg_kbt_next");
    ensure_device_double_buffer(&scratch.d_bg_b_field_curr, ncell, "alloc ws d_bg_b_field_curr");
    ensure_device_double_buffer(&scratch.d_bg_b_field_next, ncell, "alloc ws d_bg_b_field_next");
    ensure_device_double_buffer(&scratch.d_bg_divv_curr, ncell, "alloc ws d_bg_divv_curr");
    ensure_device_double_buffer(&scratch.d_bg_divv_next, ncell, "alloc ws d_bg_divv_next");
    ensure_device_double_buffer(&scratch.d_bg_l_turb_curr, ncell, "alloc ws d_bg_l_turb_curr");
    ensure_device_double_buffer(&scratch.d_bg_l_turb_next, ncell, "alloc ws d_bg_l_turb_next");
    ensure_device_double_buffer(&scratch.d_bg_dv_curr, ncell, "alloc ws d_bg_dv_curr");
    ensure_device_double_buffer(&scratch.d_bg_dv_next, ncell, "alloc ws d_bg_dv_next");
    ensure_device_double_buffer(&scratch.d_bg_cs_curr, ncell, "alloc ws d_bg_cs_curr");
    ensure_device_double_buffer(&scratch.d_bg_cs_next, ncell, "alloc ws d_bg_cs_next");
    ensure_device_double_buffer(&scratch.d_bg_beta_curr, ncell, "alloc ws d_bg_beta_curr");
    ensure_device_double_buffer(&scratch.d_bg_beta_next, ncell, "alloc ws d_bg_beta_next");
    ensure_device_double_buffer(&scratch.d_bg_rad_ic_curr, ncell_npe, "alloc ws d_bg_rad_ic_curr");
    ensure_device_double_buffer(&scratch.d_bg_rad_ic_next, ncell_npe, "alloc ws d_bg_rad_ic_next");
    ensure_device_double_buffer(&scratch.d_bg_rad_ic_m1_curr, ncell, "alloc ws d_bg_rad_ic_m1_curr");
    ensure_device_double_buffer(&scratch.d_bg_rad_ic_m1_next, ncell, "alloc ws d_bg_rad_ic_m1_next");
    ensure_device_double_buffer(&scratch.d_bg_rad_ic_p1_curr, ncell, "alloc ws d_bg_rad_ic_p1_curr");
    ensure_device_double_buffer(&scratch.d_bg_rad_ic_p1_next, ncell, "alloc ws d_bg_rad_ic_p1_next");

    if (!bg_interp_cache_matches(ws->bg_interp, in)) {
        copy_pair_to_device(scratch.d_bg_n_gas_curr, scratch.d_bg_n_gas_next,
                            in->n_gas, in->n_gas_next, ncell,
                            "copy ws bg n_gas curr", "copy ws bg n_gas next");
        copy_pair_to_device(scratch.d_bg_kbt_curr, scratch.d_bg_kbt_next,
                            in->kbt, in->kbt_next, ncell,
                            "copy ws bg kbt curr", "copy ws bg kbt next");
        copy_pair_to_device(scratch.d_bg_b_field_curr, scratch.d_bg_b_field_next,
                            in->b_field, in->b_field_next, ncell,
                            "copy ws bg b_field curr", "copy ws bg b_field next");
        copy_pair_to_device(scratch.d_bg_divv_curr, scratch.d_bg_divv_next,
                            in->divv_gyr, in->divv_gyr_next, ncell,
                            "copy ws bg divv curr", "copy ws bg divv next");
        copy_pair_to_device(scratch.d_bg_l_turb_curr, scratch.d_bg_l_turb_next,
                            in->l_turb_mpc, in->l_turb_mpc_next, ncell,
                            "copy ws bg l_turb curr", "copy ws bg l_turb next");
        copy_pair_to_device(scratch.d_bg_dv_curr, scratch.d_bg_dv_next,
                            in->dv_imc, in->dv_imc_next, ncell,
                            "copy ws bg dv curr", "copy ws bg dv next");
        copy_pair_to_device(scratch.d_bg_cs_curr, scratch.d_bg_cs_next,
                            in->cs, in->cs_next, ncell,
                            "copy ws bg cs curr", "copy ws bg cs next");
        copy_pair_to_device(scratch.d_bg_beta_curr, scratch.d_bg_beta_next,
                            in->beta_pl, in->beta_pl_next, ncell,
                            "copy ws bg beta curr", "copy ws bg beta next");
        copy_pair_to_device(scratch.d_bg_rad_ic_curr, scratch.d_bg_rad_ic_next,
                            in->rad_ic_batch, in->rad_ic_batch_next, ncell_npe,
                            "copy ws bg rad_ic curr", "copy ws bg rad_ic next");
        copy_pair_to_device(scratch.d_bg_rad_ic_m1_curr, scratch.d_bg_rad_ic_m1_next,
                            in->rad_ic_m1, in->rad_ic_m1_next, ncell,
                            "copy ws bg rad_ic_m1 curr", "copy ws bg rad_ic_m1 next");
        copy_pair_to_device(scratch.d_bg_rad_ic_p1_curr, scratch.d_bg_rad_ic_p1_next,
                            in->rad_ic_p1, in->rad_ic_p1_next, ncell,
                            "copy ws bg rad_ic_p1 curr", "copy ws bg rad_ic_p1 next");
        update_bg_interp_cache(&ws->bg_interp, in);
    }

    launch_lerp(ncell, scratch.d_bg_n_gas_curr, scratch.d_bg_n_gas_next, alpha,
                scratch.d_n_gas, "launch ws interp n_gas");
    launch_lerp(ncell, scratch.d_bg_kbt_curr, scratch.d_bg_kbt_next, alpha,
                scratch.d_kbt, "launch ws interp kbt");
    launch_lerp(ncell, scratch.d_bg_b_field_curr, scratch.d_bg_b_field_next, alpha,
                scratch.d_b_field, "launch ws interp b_field");
    launch_lerp(ncell, scratch.d_bg_divv_curr, scratch.d_bg_divv_next, alpha,
                scratch.d_divv, "launch ws interp divv");
    launch_lerp(ncell, scratch.d_bg_l_turb_curr, scratch.d_bg_l_turb_next, alpha,
                scratch.d_l_turb_mpc, "launch ws interp l_turb");
    launch_lerp(ncell, scratch.d_bg_dv_curr, scratch.d_bg_dv_next, alpha,
                scratch.d_dv_imc, "launch ws interp dv");
    launch_lerp(ncell, scratch.d_bg_cs_curr, scratch.d_bg_cs_next, alpha,
                scratch.d_cs, "launch ws interp cs");
    launch_lerp(ncell, scratch.d_bg_beta_curr, scratch.d_bg_beta_next, alpha,
                scratch.d_beta_pl, "launch ws interp beta");
    launch_lerp(ncell_npe, scratch.d_bg_rad_ic_curr, scratch.d_bg_rad_ic_next, alpha,
                scratch.d_rad_ic_batch, "launch ws interp rad_ic");
    launch_lerp(ncell, scratch.d_bg_rad_ic_m1_curr, scratch.d_bg_rad_ic_m1_next, alpha,
                scratch.d_rad_ic_m1, "launch ws interp rad_ic_m1");
    launch_lerp(ncell, scratch.d_bg_rad_ic_p1_curr, scratch.d_bg_rad_ic_p1_next, alpha,
                scratch.d_rad_ic_p1, "launch ws interp rad_ic_p1");

    return true;
}

__global__ void scatter_rank_rows_kernel(int ncell,
                                         int nrow,
                                         const int *indices,
                                         const double *bucket_rows,
                                         double *rank_rows)
{
    const size_t total = (size_t)ncell * (size_t)nrow;
    const size_t linear = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (linear >= total) return;

    const int icell = (int)(linear / (size_t)nrow);
    const int irow = (int)(linear - (size_t)icell * (size_t)nrow);
    const int dst_cell = indices[icell];
    rank_rows[(size_t)dst_cell * (size_t)nrow + (size_t)irow] = bucket_rows[linear];
}

static int run_cached_simple(const FpCudaPipelineInput *in,
                             const FpCudaOutputSchedule *schedule,
                             double *crp_state_out,
                             double *cre_state_out,
                             double *eps_syn_out,
                             FpCudaPipelineTimes *times)
{
    FpCudaPipelineWorkspace *ws = in->workspace;
    const size_t np_batch = (size_t)in->ncell * (size_t)np;
    const size_t npe_batch = (size_t)in->ncell * (size_t)npe;
    const size_t eps_size = (size_t)in->ncell * (size_t)in->nfreq;
    const int fp_cadence_steps = (in->fp_cadence_steps > 1) ? in->fp_cadence_steps : 1;
    const bool use_pcr_proton = (fp_cuda_tridiag_uses_pcr(np) == 1);
    const bool use_pcr_electron = (fp_cuda_tridiag_uses_pcr(npe) == 1);
    const bool want_synch = wants_synch(schedule);
    const bool have_state_inputs =
        (in->qpi_batch != nullptr &&
         in->qepri_batch != nullptr &&
         in->crp_init != nullptr &&
         in->cre_init != nullptr);
    const bool have_any_state_inputs =
        (in->qpi_batch != nullptr ||
         in->qepri_batch != nullptr ||
         in->crp_init != nullptr ||
         in->cre_init != nullptr);
    const bool have_state_outputs =
        (crp_state_out != nullptr && cre_state_out != nullptr);
    const bool have_any_state_outputs =
        (crp_state_out != nullptr || cre_state_out != nullptr);
    const bool gather_rank_state = (in->gather_rank_state != 0);
    const bool scatter_rank_state = (in->scatter_rank_state != 0);
    const bool use_rank_state = gather_rank_state || scatter_rank_state;
    const int cell_threads = 128;
    const int cell_blocks = (in->ncell + cell_threads - 1) / cell_threads;
    const int active_model = resolve_momentumdiff_model(FP_MOMENTUMDIFF_MODEL_AUTO);
    FpCudaCoeffPrepareInput coeff_prepare;
    const auto setup_t0 = std::chrono::steady_clock::now();

    if (ws == nullptr) return -1;
    FpCudaWorkspaceScratch &scratch = ws->scratch;
    FpCudaWorkspaceConstCache &cache = ws->cache;
    FpCudaWorkspaceEvents &events = ws->events;
    FpCudaWorkspaceHostCache &host = ws->host;
    FpCudaWorkspaceResidentState &resident = ws->resident;
    FpCudaWorkspaceRankState &rank = ws->rank;
    *times = {};
    workspace_realloc_simple_buffers(ws, in->ncell, in->nfreq, in->nx_tab);
    workspace_ensure_simple_constants(ws, in);
    times->secondary_ms += host.secondary_setup_ms;

    ensure_timing_events(&events);
    times->setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup_t0).count();

    if (want_synch && (in->synch_in == nullptr || eps_syn_out == nullptr)) return -1;
    if (have_any_state_inputs && !have_state_inputs) return -1;
    if (have_any_state_outputs && !have_state_outputs) return -1;
    if (use_rank_state && in->rank_state_indices == nullptr) return -1;
    if (gather_rank_state && have_any_state_inputs) return -1;
    if (use_rank_state &&
        (!rank.state_valid || rank.ncell <= 0 ||
         rank.d_qpi == nullptr || rank.d_qepri == nullptr ||
         rank.d_crp == nullptr || rank.d_cre == nullptr)) {
        return -1;
    }
    if (!gather_rank_state && !have_state_inputs &&
        (!resident.state_valid || resident.ncell != in->ncell)) {
        return -1;
    }

    const auto total_t0 = std::chrono::steady_clock::now();
    const auto h2d_t0 = std::chrono::steady_clock::now();
    float device_interp_ms = 0.0f;
    bool device_bg_interp = false;
    /* Device background interpolation belongs to coefficient preparation,
     * so keep it separate from the H2D bucket. */
    if (coeff_has_device_interp(in->coeff_in)) {
        cuda_check(cudaEventRecord(events.interp_start), "record ws interp_start");
        device_bg_interp = prepare_device_interpolated_background(ws, in->coeff_in);
        cuda_check(cudaEventRecord(events.interp_stop), "record ws interp_stop");
        cuda_check(cudaEventSynchronize(events.interp_stop), "sync ws interp_stop");
        if (device_bg_interp) {
            cuda_check(cudaEventElapsedTime(&device_interp_ms,
                                            events.interp_start, events.interp_stop),
                       "elapsed ws device interpolation");
        }
    }
    if (!device_bg_interp) {
        cuda_check(cudaMemcpy(scratch.d_n_gas, in->coeff_in->n_gas, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws n_gas");
        cuda_check(cudaMemcpy(scratch.d_kbt, in->coeff_in->kbt, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws kbt");
        cuda_check(cudaMemcpy(scratch.d_b_field, in->coeff_in->b_field, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws b_field");
        cuda_check(cudaMemcpy(scratch.d_divv, in->coeff_in->divv_gyr, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws divv");
        cuda_check(cudaMemcpy(scratch.d_l_turb_mpc, in->coeff_in->l_turb_mpc, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws l_turb_mpc");
        cuda_check(cudaMemcpy(scratch.d_dv_imc, in->coeff_in->dv_imc, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws dv_imc");
        cuda_check(cudaMemcpy(scratch.d_cs, in->coeff_in->cs, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws cs");
        cuda_check(cudaMemcpy(scratch.d_beta_pl, in->coeff_in->beta_pl, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws beta_pl");
    }
    if (in->coeff_in->tracer_mass_msun != nullptr) {
        cuda_check(cudaMemcpy(scratch.d_tracer_mass_msun, in->coeff_in->tracer_mass_msun,
                              (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice),
                   "copy ws tracer_mass_msun");
    }
    if (!device_bg_interp) {
        cuda_check(cudaMemcpy(scratch.d_rad_ic_batch, in->coeff_in->rad_ic_batch, npe_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws rad_ic_batch");
        cuda_check(cudaMemcpy(scratch.d_rad_ic_m1, in->coeff_in->rad_ic_m1, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws rad_ic_m1");
        cuda_check(cudaMemcpy(scratch.d_rad_ic_p1, in->coeff_in->rad_ic_p1, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws rad_ic_p1");
    }
    if (use_rank_state) {
        cuda_check(cudaMemcpy(scratch.d_bucket_indices, in->rank_state_indices,
                              (size_t)in->ncell * sizeof(int), cudaMemcpyHostToDevice),
                   "copy ws rank_state_indices");
    }
    if (gather_rank_state) {
        const int row_threads = 256;
        const int p_blocks = (int)((np_batch + (size_t)row_threads - 1) / (size_t)row_threads);
        const int e_blocks = (int)((npe_batch + (size_t)row_threads - 1) / (size_t)row_threads);

        gather_rank_rows_kernel<<<p_blocks, row_threads>>>(
            in->ncell, np, scratch.d_bucket_indices, rank.d_qpi, scratch.d_qpi_cell);
        cuda_check(cudaGetLastError(), "launch ws gather qpi rank rows");
        gather_rank_rows_kernel<<<e_blocks, row_threads>>>(
            in->ncell, npe, scratch.d_bucket_indices, rank.d_qepri, scratch.d_qepri_cell);
        cuda_check(cudaGetLastError(), "launch ws gather qepri rank rows");
        gather_rank_rows_kernel<<<p_blocks, row_threads>>>(
            in->ncell, np, scratch.d_bucket_indices, rank.d_crp, scratch.d_crp_state);
        cuda_check(cudaGetLastError(), "launch ws gather crp rank rows");
        gather_rank_rows_kernel<<<e_blocks, row_threads>>>(
            in->ncell, npe, scratch.d_bucket_indices, rank.d_cre, scratch.d_cre_state);
        cuda_check(cudaGetLastError(), "launch ws gather cre rank rows");
        resident.state_valid = 1;
        resident.ncell = in->ncell;
    }
    if (have_state_inputs) {
        cuda_check(cudaMemcpy(scratch.d_qpi_cell, in->qpi_batch, np_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws qpi_batch");
        cuda_check(cudaMemcpy(scratch.d_qepri_cell, in->qepri_batch, npe_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws qepri_batch");
        cuda_check(cudaMemcpy(scratch.d_crp_state, in->crp_init, np_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws crp_init");
        cuda_check(cudaMemcpy(scratch.d_cre_state, in->cre_init, npe_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws cre_init");
        resident.state_valid = 1;
        resident.ncell = in->ncell;
    }
    if (want_synch) {
        cuda_check(cudaMemcpy(scratch.d_b_dyn, in->synch_in->b_dyn, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws b_dyn");
        cuda_check(cudaMemcpy(scratch.d_logb, in->synch_in->logb, (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws logb");
    }
    times->h2d_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - h2d_t0).count();
    times->device_interp_ms = (double)device_interp_ms;
    times->h2d_ms -= times->device_interp_ms;
    if (times->h2d_ms < 0.0) times->h2d_ms = 0.0;

    auto build_ws_cc_coeffs = [&](double dt_step, const char *label) {
        FpCudaCcBuildInput coeff_build_on;
        FpCudaCcBuildInput coeff_build_off;

        std::memset(&coeff_build_on, 0, sizeof(coeff_build_on));
        coeff_build_on.ncell = in->ncell;
        coeff_build_on.cell_blocks = cell_blocks;
        coeff_build_on.cell_threads = cell_threads;
        coeff_build_on.dt = dt_step;
        coeff_build_on.d_crp_p = cache.d_crp_p;
        coeff_build_on.d_crp_dp = cache.d_crp_dp;
        coeff_build_on.d_cre_p = cache.d_cre_p;
        coeff_build_on.d_cre_dp = cache.d_cre_dp;
        coeff_build_on.crp_pm1 = in->crp_grid->pm1;
        coeff_build_on.crp_pp1 = in->crp_grid->pp1;
        coeff_build_on.cre_pm1 = in->cre_grid->pm1;
        coeff_build_on.cre_pp1 = in->cre_grid->pp1;
        coeff_build_on.d_crp_radp = scratch.d_crp_radp;
        coeff_build_on.d_crp_tloss = scratch.d_crp_tloss;
        coeff_build_on.d_dpp = scratch.d_dpp;
        coeff_build_on.d_cre_radp = scratch.d_cre_radp;
        coeff_build_on.d_cre_tloss = scratch.d_cre_tloss;
        coeff_build_on.d_dppe = scratch.d_dppe;
        coeff_build_on.d_crp_radpm1 = scratch.d_crp_radpm1;
        coeff_build_on.d_crp_radpp1 = scratch.d_crp_radpp1;
        coeff_build_on.d_cre_radpm1 = scratch.d_cre_radpm1;
        coeff_build_on.d_cre_radpp1 = scratch.d_cre_radpp1;
        coeff_build_on.d_dppm1 = scratch.d_dppm1;
        coeff_build_on.d_dppp1 = scratch.d_dppp1;
        coeff_build_on.d_dppem1 = scratch.d_dppem1;
        coeff_build_on.d_dppep1 = scratch.d_dppep1;
        coeff_build_on.d_ccp_a = scratch.d_ccp_a;
        coeff_build_on.d_ccp_b = scratch.d_ccp_b;
        coeff_build_on.d_ccp_c = scratch.d_ccp_c;
        coeff_build_on.d_cce_a = scratch.d_cce_a;
        coeff_build_on.d_cce_b = scratch.d_cce_b;
        coeff_build_on.d_cce_c = scratch.d_cce_c;
        if (fp_cuda_build_cc_coeffs(&coeff_build_on, label) != 0) return -1;

        coeff_build_off = coeff_build_on;
        coeff_build_off.d_dpp = scratch.d_dpp_off;
        coeff_build_off.d_dppe = scratch.d_dppe_off;
        coeff_build_off.d_dppm1 = scratch.d_dppm1_off;
        coeff_build_off.d_dppp1 = scratch.d_dppp1_off;
        coeff_build_off.d_dppem1 = scratch.d_dppem1_off;
        coeff_build_off.d_dppep1 = scratch.d_dppep1_off;
        coeff_build_off.d_ccp_a = scratch.d_ccp_a_off;
        coeff_build_off.d_ccp_b = scratch.d_ccp_b_off;
        coeff_build_off.d_ccp_c = scratch.d_ccp_c_off;
        coeff_build_off.d_cce_a = scratch.d_cce_a_off;
        coeff_build_off.d_cce_b = scratch.d_cce_b_off;
        coeff_build_off.d_cce_c = scratch.d_cce_c_off;
        return fp_cuda_build_cc_coeffs(&coeff_build_off, "launch ws build_cc_coeffs_kernel_off");
    };

    cuda_check(cudaEventRecord(events.coeff_start), "record ws coeff_start");
    std::memset(&coeff_prepare, 0, sizeof(coeff_prepare));
    coeff_prepare.ncell = in->ncell;
    coeff_prepare.cell_blocks = cell_blocks;
    coeff_prepare.cell_threads = cell_threads;
    coeff_prepare.active_model = active_model;
    coeff_prepare.epmax = in->epmax;
    coeff_prepare.psi_value = in->psi_value;
    coeff_prepare.mach_limit_value = in->mach_limit_value;
    coeff_prepare.ttd_tacc_model = ttd_tacc_model;
    coeff_prepare.t_acc_direct_gyr = t_acc_direct_gyr;
    coeff_prepare.eta_dpp_cap_value = eta_dpp_cap;
    coeff_prepare.d_n_gas = scratch.d_n_gas;
    coeff_prepare.d_kbt = scratch.d_kbt;
    coeff_prepare.d_b_field = scratch.d_b_field;
    coeff_prepare.d_divv = scratch.d_divv;
    coeff_prepare.d_rad_ic_batch = scratch.d_rad_ic_batch;
    coeff_prepare.d_rad_ic_m1 = scratch.d_rad_ic_m1;
    coeff_prepare.d_rad_ic_p1 = scratch.d_rad_ic_p1;
    coeff_prepare.d_l_turb_mpc = scratch.d_l_turb_mpc;
    coeff_prepare.d_dv_imc = scratch.d_dv_imc;
    coeff_prepare.d_cs = scratch.d_cs;
    coeff_prepare.d_beta_pl = scratch.d_beta_pl;
    coeff_prepare.d_crp_p = cache.d_crp_p;
    coeff_prepare.d_crp_e = cache.d_crp_e;
    coeff_prepare.d_crp_dp = cache.d_crp_dp;
    coeff_prepare.d_crp_p2 = cache.d_crp_p2;
    coeff_prepare.d_crp_exp_cut = cache.d_crp_exp_cut;
    coeff_prepare.d_crp_sigma_pp = cache.d_crp_sigma_pp;
    coeff_prepare.d_crp_sigmoid_pp = cache.d_crp_sigmoid_pp;
    coeff_prepare.d_cre_p = cache.d_cre_p;
    coeff_prepare.d_cre_dp = cache.d_cre_dp;
    coeff_prepare.d_cre_p2 = cache.d_cre_p2;
    coeff_prepare.d_cre_exp_cut = cache.d_cre_exp_cut;
    coeff_prepare.d_tracer_mass_msun =
        (in->coeff_in->tracer_mass_msun != nullptr) ? scratch.d_tracer_mass_msun : nullptr;
    coeff_prepare.d_crp_state = scratch.d_crp_state;
    coeff_prepare.d_cre_state = scratch.d_cre_state;
    coeff_prepare.crp_pm1 = in->crp_grid->pm1;
    coeff_prepare.crp_pp1 = in->crp_grid->pp1;
    coeff_prepare.cre_pm1 = in->cre_grid->pm1;
    coeff_prepare.cre_pp1 = in->cre_grid->pp1;
    coeff_prepare.crp_j_pp = cache.crp_j_pp;
    coeff_prepare.crp_pm1_p2 = cache.crp_pm1_p2;
    coeff_prepare.crp_pp1_p2 = cache.crp_pp1_p2;
    coeff_prepare.crp_pm1_exp_cut = cache.crp_pm1_exp_cut;
    coeff_prepare.crp_pp1_exp_cut = cache.crp_pp1_exp_cut;
    coeff_prepare.cre_pm1_p2 = cache.cre_pm1_p2;
    coeff_prepare.cre_pp1_p2 = cache.cre_pp1_p2;
    coeff_prepare.cre_pm1_exp_cut = cache.cre_pm1_exp_cut;
    coeff_prepare.cre_pp1_exp_cut = cache.cre_pp1_exp_cut;
    coeff_prepare.d_crp_radp = scratch.d_crp_radp;
    coeff_prepare.d_crp_tloss = scratch.d_crp_tloss;
    coeff_prepare.d_cre_radp = scratch.d_cre_radp;
    coeff_prepare.d_cre_tloss = scratch.d_cre_tloss;
    coeff_prepare.d_crp_radpm1 = scratch.d_crp_radpm1;
    coeff_prepare.d_crp_radpp1 = scratch.d_crp_radpp1;
    coeff_prepare.d_cre_radpm1 = scratch.d_cre_radpm1;
    coeff_prepare.d_cre_radpp1 = scratch.d_cre_radpp1;
    coeff_prepare.d_dpp = scratch.d_dpp;
    coeff_prepare.d_dppm1 = scratch.d_dppm1;
    coeff_prepare.d_dppp1 = scratch.d_dppp1;
    coeff_prepare.d_dppe = scratch.d_dppe;
    coeff_prepare.d_dppem1 = scratch.d_dppem1;
    coeff_prepare.d_dppep1 = scratch.d_dppep1;
    coeff_prepare.d_dpp_off = scratch.d_dpp_off;
    coeff_prepare.d_dppm1_off = scratch.d_dppm1_off;
    coeff_prepare.d_dppp1_off = scratch.d_dppp1_off;
    coeff_prepare.d_dppe_off = scratch.d_dppe_off;
    coeff_prepare.d_dppem1_off = scratch.d_dppem1_off;
    coeff_prepare.d_dppep1_off = scratch.d_dppep1_off;
    if (fp_cuda_prepare_coeff_terms(&coeff_prepare,
                                    "launch ws prepare_losses_kernel",
                                    "launch ws prepare_momentumdiff_model_kernel",
                                    "launch ws prepare_momentumdiff_off_kernel") != 0) {
        return -1;
    }
    if (build_ws_cc_coeffs(in->dt, "launch ws build_cc_coeffs_kernel") != 0) {
        return -1;
    }
    cuda_check(cudaEventRecord(events.coeff_stop), "record ws coeff_stop");
    cuda_check(cudaEventSynchronize(events.coeff_stop), "sync ws coeff_stop");

    cuda_check(cudaEventRecord(events.secondary_start), "record ws secondary_start");
    prepare_secondary_sources_cell_major_kernel<<<in->ncell, npe>>>(
        in->ncell, scratch.d_n_gas, scratch.d_crp_state, cache.d_crp_dp,
        cache.d_np_min_qe, cache.d_fqe_flat,
        tracer_fp_secondary_transpose_enabled(),
        scratch.d_qepri_cell, scratch.d_qe_integral_cell, scratch.d_inje_cell);
    cuda_check(cudaGetLastError(), "launch ws prepare_secondary_sources");
    cuda_check(cudaEventRecord(events.secondary_stop), "record ws secondary_stop");
    cuda_check(cudaEventSynchronize(events.secondary_stop), "sync ws secondary_stop");
    {
        float secondary_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&secondary_ms, events.secondary_start, events.secondary_stop),
                   "elapsed ws secondary");
        times->secondary_ms += secondary_ms;
    }

    auto maybe_emit_synch = [&](int istep) -> int {
        FpCudaSynchDeviceInput synch_device;
        float synch_ms_step = 0.0f;

        if (!should_emit_synch(schedule, istep, in->nstep)) return 0;

        cuda_check(cudaEventRecord(events.synch_start), "record ws synch_start");
        std::memset(&synch_device, 0, sizeof(synch_device));
        synch_device.ncell = in->ncell;
        synch_device.nfreq = in->nfreq;
        synch_device.nx_tab = in->nx_tab;
        synch_device.xmin = in->xmin;
        synch_device.nlogb = in->synch_in->nlogb;
        synch_device.logb_min = in->synch_in->logb_min;
        synch_device.inv_dlogb = in->synch_in->inv_dlogb;
        synch_device.d_fx_tab = cache.d_fx_tab;
        synch_device.d_logx_tab = cache.d_logx_tab;
        synch_device.d_lognu_syn = cache.d_lognu_syn;
        synch_device.d_lognu_crit = cache.d_lognu_crit;
        synch_device.d_pitch_kernel_table = cache.d_pitch_kernel_table;
        synch_device.d_b_dyn = scratch.d_b_dyn;
        synch_device.d_logb = scratch.d_logb;
        synch_device.d_cre_state = scratch.d_cre_state;
        synch_device.d_cre_dp = cache.d_cre_dp;
        synch_device.d_pitch_weight = cache.d_pitch_weight;
        synch_device.d_eps_out = scratch.d_eps;
        if (fp_cuda_emit_synch_device(&synch_device, 0) != 0) {
            return -1;
        }
        cuda_check(cudaEventRecord(events.synch_stop), "record ws synch_stop");
        cuda_check(cudaEventSynchronize(events.synch_stop), "sync ws synch_stop");
        cuda_check(cudaEventElapsedTime(&synch_ms_step, events.synch_start, events.synch_stop), "elapsed ws synch");
        times->synch_ms += synch_ms_step;
        times->emitted_synch++;
        times->last_synch_step = istep + 1;
        return 0;
    };

    auto run_transport_legacy = [&]() -> int {
        int fp_acc_steps = 0;
        int executed_fp_steps = 0;
        double coeff_dt_built = in->dt;

        for (int istep = 0; istep < in->nstep; istep++) {
            fp_acc_steps++;
            if (fp_acc_steps >= fp_cadence_steps || istep == in->nstep - 1) {
                const double dt_fp_step = (double)fp_acc_steps * in->dt;
                const bool use_on_coeff =
                    !in->use_windowed_reacc ||
                    (executed_fp_steps >= in->on_start_step && executed_fp_steps < in->on_end_step);
                const double *d_ccp_a_step = use_on_coeff ? scratch.d_ccp_a : scratch.d_ccp_a_off;
                const double *d_ccp_b_step = use_on_coeff ? scratch.d_ccp_b : scratch.d_ccp_b_off;
                const double *d_ccp_c_step = use_on_coeff ? scratch.d_ccp_c : scratch.d_ccp_c_off;
                const double *d_cce_a_step = use_on_coeff ? scratch.d_cce_a : scratch.d_cce_a_off;
                const double *d_cce_b_step = use_on_coeff ? scratch.d_cce_b : scratch.d_cce_b_off;
                const double *d_cce_c_step = use_on_coeff ? scratch.d_cce_c : scratch.d_cce_c_off;

                if (std::fabs(dt_fp_step - coeff_dt_built) > 1.0e-15) {
                    if (build_ws_cc_coeffs(dt_fp_step, "launch ws build_cc_coeffs_kernel loop") != 0) {
                        return -1;
                    }
                    coeff_dt_built = dt_fp_step;
                }

                if (fp_cuda_solve_device_batch(in->ncell, np, dt_fp_step,
                                               d_ccp_a_step, d_ccp_b_step, d_ccp_c_step,
                                               scratch.d_qpi_cell, scratch.d_crp_state,
                                               "launch ws solve proton", 0) != 0) {
                    return -1;
                }
                if (fp_cuda_solve_device_batch(in->ncell, npe, dt_fp_step,
                                               d_cce_a_step, d_cce_b_step, d_cce_c_step,
                                               scratch.d_inje_cell, scratch.d_cre_state,
                                               "launch ws solve electron", 0) != 0) {
                    return -1;
                }
                fp_acc_steps = 0;
                executed_fp_steps++;
            }

            if (maybe_emit_synch(istep) != 0) {
                return -1;
            }
        }
        return 0;
    };

    auto run_transport_fused = [&]() -> int {
        FpCudaFusedTransportInput fused_in;

        if (fp_cadence_steps != 1 ||
            has_intermediate_synch(schedule, in->nstep)) {
            return run_transport_legacy();
        }

        std::memset(&fused_in, 0, sizeof(fused_in));
        fused_in.nsys = in->ncell;
        fused_in.nstep = in->nstep;
        fused_in.use_windowed_reacc = in->use_windowed_reacc;
        fused_in.on_start_step = in->on_start_step;
        fused_in.on_end_step = in->on_end_step;
        fused_in.dt = in->dt;
        fused_in.d_ccp_a = scratch.d_ccp_a;
        fused_in.d_ccp_b = scratch.d_ccp_b;
        fused_in.d_ccp_c = scratch.d_ccp_c;
        fused_in.d_ccp_a_off = scratch.d_ccp_a_off;
        fused_in.d_ccp_b_off = scratch.d_ccp_b_off;
        fused_in.d_ccp_c_off = scratch.d_ccp_c_off;
        fused_in.d_cce_a = scratch.d_cce_a;
        fused_in.d_cce_b = scratch.d_cce_b;
        fused_in.d_cce_c = scratch.d_cce_c;
        fused_in.d_cce_a_off = scratch.d_cce_a_off;
        fused_in.d_cce_b_off = scratch.d_cce_b_off;
        fused_in.d_cce_c_off = scratch.d_cce_c_off;
        fused_in.d_qpi_cell = scratch.d_qpi_cell;
        fused_in.d_qepri_cell = scratch.d_qepri_cell;
        fused_in.d_n_gas = scratch.d_n_gas;
        fused_in.d_crp_dp = cache.d_crp_dp;
        fused_in.d_np_min_qe = cache.d_np_min_qe;
        fused_in.d_fqe_flat = cache.d_fqe_flat;
        fused_in.d_qe_integral_cell = scratch.d_qe_integral_cell;
        fused_in.d_inje_cell = scratch.d_inje_cell;
        fused_in.d_crp_state = scratch.d_crp_state;
        fused_in.d_cre_state = scratch.d_cre_state;

        if (fp_cuda_run_fused_transport(&fused_in,
                                        "launch ws fused transport",
                                        &times->used_pcr_proton,
                                        &times->used_pcr_electron) != 0) {
            return -1;
        }

        if (maybe_emit_synch(in->nstep - 1) != 0) {
            return -1;
        }
        return 0;
    };

    cuda_check(cudaEventRecord(events.solve_start), "record ws solve_start");
    if ((in->transport_mode == FP_CUDA_TRANSPORT_FUSED_SUBSTEPS
            ? run_transport_fused()
            : run_transport_legacy()) != 0) {
        return -1;
    }
    cuda_check(cudaEventRecord(events.solve_stop), "record ws solve_stop");
    cuda_check(cudaEventSynchronize(events.solve_stop), "sync ws solve_stop");

    {
        float coeff_ms = 0.0f;
        float solve_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&coeff_ms, events.coeff_start, events.coeff_stop), "elapsed ws coeff");
        cuda_check(cudaEventElapsedTime(&solve_ms, events.solve_start, events.solve_stop), "elapsed ws solve");
        times->coeff_ms = coeff_ms;
        times->coeff_ms += times->device_interp_ms;
        times->solve_ms = solve_ms;
        times->used_pcr_proton = use_pcr_proton ? 1 : 0;
        times->used_pcr_electron = use_pcr_electron ? 1 : 0;
    }

    const auto d2h_t0 = std::chrono::steady_clock::now();
    if (scatter_rank_state) {
        const int row_threads = 256;
        const int p_blocks = (int)((np_batch + (size_t)row_threads - 1) / (size_t)row_threads);
        const int e_blocks = (int)((npe_batch + (size_t)row_threads - 1) / (size_t)row_threads);

        scatter_rank_rows_kernel<<<p_blocks, row_threads>>>(
            in->ncell, np, scratch.d_bucket_indices, scratch.d_crp_state, rank.d_crp);
        cuda_check(cudaGetLastError(), "launch ws scatter crp rank rows");
        scatter_rank_rows_kernel<<<e_blocks, row_threads>>>(
            in->ncell, npe, scratch.d_bucket_indices, scratch.d_cre_state, rank.d_cre);
        cuda_check(cudaGetLastError(), "launch ws scatter cre rank rows");
    }
    if (have_state_outputs) {
        cuda_check(cudaMemcpy(crp_state_out, scratch.d_crp_state, np_batch * sizeof(double), cudaMemcpyDeviceToHost), "copy ws crp_state_out");
        cuda_check(cudaMemcpy(cre_state_out, scratch.d_cre_state, npe_batch * sizeof(double), cudaMemcpyDeviceToHost), "copy ws cre_state_out");
    }
    if (times->emitted_synch > 0 && eps_syn_out != nullptr) {
        cuda_check(cudaMemcpy(eps_syn_out, scratch.d_eps, eps_size * sizeof(double), cudaMemcpyDeviceToHost), "copy ws eps_syn_out");
    }
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize ws");
    times->d2h_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - d2h_t0).count();
    times->total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_t0).count();
    times->other_ms = times->total_ms - (times->setup_ms + times->h2d_ms +
                                         times->coeff_ms + times->secondary_ms +
                                         times->solve_ms + times->synch_ms +
                                         times->gamma_ms + times->neutrino_ms +
                                         times->d2h_ms);
    if (times->other_ms < 0.0) times->other_ms = 0.0;
    return 0;
}

static int run_fused_substeps(const FpCudaPipelineInput *in,
                              const FpCudaOutputSchedule *schedule,
                              double *crp_state_out,
                              double *cre_state_out,
                              double *eps_syn_out,
                              FpCudaPipelineTimes *times)
{
    /* Fused-substep entry point.
     * The cached path now dispatches transport internally based on
     * transport_mode while keeping emission on the existing route. */
    return run_cached_simple(in, schedule,
                             crp_state_out,
                             cre_state_out,
                             eps_syn_out,
                             times);
}

static void accum_pipeline_times(FpCudaPipelineTimes *dst,
                                 const FpCudaPipelineTimes *src)
{
    if (dst == nullptr || src == nullptr) return;
    dst->setup_ms += src->setup_ms;
    dst->h2d_ms += src->h2d_ms;
    dst->device_interp_ms += src->device_interp_ms;
    dst->coeff_ms += src->coeff_ms;
    dst->secondary_ms += src->secondary_ms;
    dst->solve_ms += src->solve_ms;
    dst->diff_ms += src->diff_ms;
    dst->adv_ms += src->adv_ms;
    dst->synch_ms += src->synch_ms;
    dst->gamma_ms += src->gamma_ms;
    dst->neutrino_ms += src->neutrino_ms;
    dst->d2h_ms += src->d2h_ms;
    dst->other_ms += src->other_ms;
    dst->total_ms += src->total_ms;
    dst->emitted_synch += src->emitted_synch;
    dst->emitted_gamma += src->emitted_gamma;
    dst->emitted_neutrino += src->emitted_neutrino;
    dst->last_synch_step = src->last_synch_step;
    dst->used_pcr_proton = src->used_pcr_proton;
    dst->used_pcr_electron = src->used_pcr_electron;
}

static bool same_double(double a, double b)
{
    return std::fabs(a - b) <= 1.0e-15;
}

static bool wants_synch(const FpCudaOutputSchedule *schedule)
{
    return schedule != nullptr &&
           (schedule->emit_final_only != 0 ||
            schedule->emit_all_steps != 0 ||
            schedule->emit_stride > 0 ||
            schedule->nselected_steps > 0);
}

static bool can_merge_group_strict(const FpCudaPipelineInput *inputs,
                                          int ninput,
                                          const FpCudaOutputSchedule *schedule,
                                          int *total_ncell_out)
{
    const FpCudaPipelineInput *base;
    const FpCoeffBatchInput *base_coeff;
    const FpSynchEmissionBatchInput *base_synch;
    const bool need_synch = wants_synch(schedule);
    int total_ncell = 0;
    int cell_offset = 0;

    if (inputs == nullptr || ninput <= 0 || total_ncell_out == nullptr) return false;
    base = &inputs[0];
    base_coeff = base->coeff_in;
    base_synch = base->synch_in;
    if (base->workspace == nullptr || base_coeff == nullptr ||
        (need_synch && base_synch == nullptr)) return false;
    if (base->gamma_in != nullptr || base->eps_gamma_out != nullptr ||
        base->neutrino_in != nullptr || base->eps_nu_out != nullptr) return false;

    for (int i = 0; i < ninput; i++) {
        const FpCudaPipelineInput *in = &inputs[i];
        const FpCoeffBatchInput *coeff = in->coeff_in;
        const FpSynchEmissionBatchInput *synch = in->synch_in;
        const int ncell = in->ncell;

        if (coeff == nullptr || ncell <= 0 || coeff->device_interp_background != 0 ||
            (need_synch && synch == nullptr)) {
            return false;
        }
        if (in->workspace != base->workspace ||
            in->transport_mode != base->transport_mode ||
            in->nstep != base->nstep ||
            in->fp_cadence_steps != base->fp_cadence_steps ||
            in->diff_cadence_steps != base->diff_cadence_steps ||
            in->adv_cadence_steps != base->adv_cadence_steps ||
            in->use_windowed_reacc != base->use_windowed_reacc ||
            in->on_start_step != base->on_start_step ||
            in->on_end_step != base->on_end_step ||
            in->nfreq != base->nfreq ||
            in->nx_tab != base->nx_tab ||
            !same_double(in->dt, base->dt) ||
            !same_double(in->epmax, base->epmax) ||
            !same_double(in->xmin, base->xmin) ||
            !same_double(in->psi_value, base->psi_value) ||
            !same_double(in->mach_limit_value, base->mach_limit_value) ||
            in->crp_grid != base->crp_grid ||
            in->cre_grid != base->cre_grid ||
            in->beta_p != base->beta_p) {
            return false;
        }
        if (coeff->z != base_coeff->z ||
            !same_double(coeff->dt_gyr, base_coeff->dt_gyr) ||
            !same_double(coeff->epmax, base_coeff->epmax)) {
            return false;
        }
        if (need_synch &&
            (synch->z != base_synch->z ||
             synch->nfreq != base_synch->nfreq ||
             synch->nx_tab != base_synch->nx_tab ||
             synch->nlogb != base_synch->nlogb ||
             !same_double(synch->xmin, base_synch->xmin) ||
             !same_double(synch->logb_min, base_synch->logb_min) ||
             !same_double(synch->inv_dlogb, base_synch->inv_dlogb) ||
             synch->fx_tab != base_synch->fx_tab ||
             synch->logfx_tab != base_synch->logfx_tab ||
             synch->logx_tab != base_synch->logx_tab ||
             synch->logy != base_synch->logy ||
             synch->pitch_kernel_table != base_synch->pitch_kernel_table ||
             synch->gamma2e != base_synch->gamma2e ||
             synch->theta != base_synch->theta ||
             synch->dtheta != base_synch->dtheta ||
             synch->pitch_weight != base_synch->pitch_weight ||
             synch->nus != base_synch->nus)) {
            return false;
        }
        if (i > 0) {
            if (coeff->n_gas != base_coeff->n_gas + cell_offset ||
                coeff->kbt != base_coeff->kbt + cell_offset ||
                coeff->b_field != base_coeff->b_field + cell_offset ||
                coeff->divv_gyr != base_coeff->divv_gyr + cell_offset ||
                coeff->l_turb_mpc != base_coeff->l_turb_mpc + cell_offset ||
                coeff->dv_imc != base_coeff->dv_imc + cell_offset ||
                coeff->cs != base_coeff->cs + cell_offset ||
                coeff->beta_pl != base_coeff->beta_pl + cell_offset ||
                coeff->rad_ic_m1 != base_coeff->rad_ic_m1 + cell_offset ||
                coeff->rad_ic_p1 != base_coeff->rad_ic_p1 + cell_offset ||
                in->crp_init != base->crp_init + (size_t)cell_offset * (size_t)np ||
                in->cre_init != base->cre_init + (size_t)cell_offset * (size_t)npe ||
                in->crp_state_out != base->crp_state_out + (size_t)cell_offset * (size_t)np ||
                in->cre_state_out != base->cre_state_out + (size_t)cell_offset * (size_t)npe) {
                return false;
            }
            if (need_synch &&
                (synch->b_dyn != base_synch->b_dyn + cell_offset ||
                 synch->logb != base_synch->logb + cell_offset ||
                 synch->cre_batch != base_synch->cre_batch + (size_t)cell_offset * (size_t)npe)) {
                return false;
            }
            if (in->qpi_batch != base->qpi_batch + (size_t)cell_offset * (size_t)np ||
                in->qepri_batch != base->qepri_batch + (size_t)cell_offset * (size_t)npe) {
                return false;
            }
        }

        total_ncell += ncell;
        cell_offset = total_ncell;
    }

    *total_ncell_out = total_ncell;
    return true;
}

static int run_group_merged_strict(const FpCudaPipelineInput *inputs,
                                   int ninput,
                                   const FpCudaOutputSchedule *schedule,
                                   FpCudaPipelineTimes *times)
{
    FpCudaPipelineWorkspace *ws;
    FpCudaPipelineInput merged_in;
    FpCoeffBatchInput merged_coeff;
    FpSynchEmissionBatchInput merged_synch;
    int total_ncell = 0;
    int cell_offset = 0;

    if (!can_merge_group_strict(inputs, ninput, schedule, &total_ncell)) return -1;
    ws = inputs[0].workspace;
    ws->host.merged_rad_ic_batch.resize((size_t)total_ncell * (size_t)npe);

    for (int i = 0; i < ninput; i++) {
        const FpCudaPipelineInput *in = &inputs[i];
        const double *src = in->coeff_in->rad_ic_batch;
        const int ncell = in->ncell;
        for (int je = 0; je < npe; je++) {
            std::memcpy(ws->host.merged_rad_ic_batch.data() +
                            (size_t)je * (size_t)total_ncell + (size_t)cell_offset,
                        src + (size_t)je * (size_t)ncell,
                        (size_t)ncell * sizeof(double));
        }
        cell_offset += ncell;
    }

    merged_in = inputs[0];
    merged_coeff = *inputs[0].coeff_in;
    merged_in.ncell = total_ncell;
    merged_coeff.ncell = total_ncell;
    merged_coeff.rad_ic_batch = ws->host.merged_rad_ic_batch.data();
    merged_in.coeff_in = &merged_coeff;
    if (inputs[0].synch_in != nullptr) {
        merged_synch = *inputs[0].synch_in;
        merged_synch.ncell = total_ncell;
        merged_in.synch_in = &merged_synch;
    } else {
        merged_in.synch_in = nullptr;
    }

    return run_tracer_pipeline(&merged_in, schedule,
                               inputs[0].crp_state_out,
                               inputs[0].cre_state_out,
                               inputs[0].eps_syn_out,
                               inputs[0].eps_gamma_out,
                               inputs[0].eps_nu_out,
                               times);
}

static bool can_share_group_coeff(const FpCudaPipelineInput *inputs,
                                         int ninput,
                                         const FpCudaOutputSchedule *schedule,
                                         int *total_ncell_out)
{
    const FpCudaPipelineInput *base;
    const FpCoeffBatchInput *base_coeff;
    int total_ncell = 0;
    int cell_offset = 0;

    if (inputs == nullptr || ninput <= 0 || total_ncell_out == nullptr) return false;
    if (wants_synch(schedule)) return false;

    base = &inputs[0];
    base_coeff = base->coeff_in;
    if (base->workspace == nullptr || base_coeff == nullptr) return false;
    if (base->gamma_in != nullptr || base->eps_gamma_out != nullptr ||
        base->neutrino_in != nullptr || base->eps_nu_out != nullptr ||
        base->eps_syn_out != nullptr ||
        base->crp_init == nullptr || base->cre_init == nullptr ||
        base->crp_state_out == nullptr || base->cre_state_out == nullptr) {
        return false;
    }
    if (base->qpi_batch == nullptr || base->qepri_batch == nullptr) {
        return false;
    }

    for (int i = 0; i < ninput; i++) {
        const FpCudaPipelineInput *in = &inputs[i];
        const FpCoeffBatchInput *coeff = in->coeff_in;
        const int ncell = in->ncell;

        if (coeff == nullptr || ncell <= 0 || coeff->device_interp_background != 0) return false;
        if (in->workspace != base->workspace ||
            in->transport_mode != base->transport_mode ||
            in->gamma_in != nullptr ||
            in->eps_gamma_out != nullptr ||
            in->neutrino_in != nullptr ||
            in->eps_nu_out != nullptr ||
            in->eps_syn_out != nullptr ||
            in->crp_init == nullptr ||
            in->cre_init == nullptr ||
            in->crp_state_out == nullptr ||
            in->cre_state_out == nullptr ||
            in->nfreq != base->nfreq ||
            in->nx_tab != base->nx_tab ||
            !same_double(in->epmax, base->epmax) ||
            !same_double(in->xmin, base->xmin) ||
            !same_double(in->psi_value, base->psi_value) ||
            !same_double(in->mach_limit_value, base->mach_limit_value) ||
            in->crp_grid != base->crp_grid ||
            in->cre_grid != base->cre_grid ||
            in->beta_p != base->beta_p ||
            !same_double(coeff->z, base_coeff->z) ||
            !same_double(coeff->epmax, base_coeff->epmax)) {
            return false;
        }
        if (in->qpi_batch == nullptr || in->qepri_batch == nullptr) {
            return false;
        }
        if (i > 0) {
            if (coeff->n_gas != base_coeff->n_gas + cell_offset ||
                coeff->kbt != base_coeff->kbt + cell_offset ||
                coeff->b_field != base_coeff->b_field + cell_offset ||
                coeff->divv_gyr != base_coeff->divv_gyr + cell_offset ||
                coeff->l_turb_mpc != base_coeff->l_turb_mpc + cell_offset ||
                coeff->dv_imc != base_coeff->dv_imc + cell_offset ||
                coeff->cs != base_coeff->cs + cell_offset ||
                coeff->beta_pl != base_coeff->beta_pl + cell_offset ||
                coeff->rad_ic_m1 != base_coeff->rad_ic_m1 + cell_offset ||
                coeff->rad_ic_p1 != base_coeff->rad_ic_p1 + cell_offset ||
                in->crp_init != base->crp_init + (size_t)cell_offset * (size_t)np ||
                in->cre_init != base->cre_init + (size_t)cell_offset * (size_t)npe ||
                in->crp_state_out != base->crp_state_out + (size_t)cell_offset * (size_t)np ||
                in->cre_state_out != base->cre_state_out + (size_t)cell_offset * (size_t)npe) {
                return false;
            }
            if (in->qpi_batch != base->qpi_batch + (size_t)cell_offset * (size_t)np ||
                in->qepri_batch != base->qepri_batch + (size_t)cell_offset * (size_t)npe) {
                return false;
            }
        }

        total_ncell += ncell;
        cell_offset = total_ncell;
    }

    *total_ncell_out = total_ncell;
    return true;
}

static int run_group_shared_coeff(const FpCudaPipelineInput *inputs,
                                  int ninput,
                                  const FpCudaOutputSchedule *schedule,
                                  FpCudaPipelineTimes *times)
{
    const FpCudaPipelineInput *base;
    FpCudaPipelineWorkspace *ws;
    FpCudaWorkspaceScratch *scratch;
    FpCudaWorkspaceConstCache *cache;
    FpCudaWorkspaceEvents *events;
    FpCudaCoeffPrepareInput coeff_prepare;
    const bool use_pcr_proton = (fp_cuda_tridiag_uses_pcr(np) == 1);
    const bool use_pcr_electron = (fp_cuda_tridiag_uses_pcr(npe) == 1);
    const int cell_threads = 128;
    const int active_model = resolve_momentumdiff_model(FP_MOMENTUMDIFF_MODEL_AUTO);
    int total_ncell = 0;
    const auto setup_t0 = std::chrono::steady_clock::now();

    if (!can_share_group_coeff(inputs, ninput, schedule, &total_ncell)) return -1;

    base = &inputs[0];
    ws = base->workspace;
    scratch = &ws->scratch;
    cache = &ws->cache;
    FpCudaWorkspaceHostCache *host = &ws->host;
    events = &ws->events;
    *times = {};

    workspace_realloc_simple_buffers(ws, total_ncell, base->nfreq, base->nx_tab);
    workspace_ensure_simple_constants(ws, base);
    times->secondary_ms += host->secondary_setup_ms;

    ensure_timing_events(events);
    times->setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup_t0).count();

    auto build_ws_cc_coeffs_slice = [&](int ncell_slice,
                                        int cell_offset,
                                        double dt_step,
                                        const char *label) {
        FpCudaCcBuildInput coeff_build_on;
        FpCudaCcBuildInput coeff_build_off;
        const size_t np_off = (size_t)cell_offset * (size_t)np;
        const size_t npe_off = (size_t)cell_offset * (size_t)npe;
        const int cell_blocks = (ncell_slice + cell_threads - 1) / cell_threads;

        std::memset(&coeff_build_on, 0, sizeof(coeff_build_on));
        coeff_build_on.ncell = ncell_slice;
        coeff_build_on.cell_blocks = cell_blocks;
        coeff_build_on.cell_threads = cell_threads;
        coeff_build_on.dt = dt_step;
        coeff_build_on.d_crp_p = cache->d_crp_p;
        coeff_build_on.d_crp_dp = cache->d_crp_dp;
        coeff_build_on.d_cre_p = cache->d_cre_p;
        coeff_build_on.d_cre_dp = cache->d_cre_dp;
        coeff_build_on.crp_pm1 = base->crp_grid->pm1;
        coeff_build_on.crp_pp1 = base->crp_grid->pp1;
        coeff_build_on.cre_pm1 = base->cre_grid->pm1;
        coeff_build_on.cre_pp1 = base->cre_grid->pp1;
        coeff_build_on.d_crp_radp = scratch->d_crp_radp + np_off;
        coeff_build_on.d_crp_tloss = scratch->d_crp_tloss + np_off;
        coeff_build_on.d_dpp = scratch->d_dpp + np_off;
        coeff_build_on.d_cre_radp = scratch->d_cre_radp + npe_off;
        coeff_build_on.d_cre_tloss = scratch->d_cre_tloss + npe_off;
        coeff_build_on.d_dppe = scratch->d_dppe + npe_off;
        coeff_build_on.d_crp_radpm1 = scratch->d_crp_radpm1 + cell_offset;
        coeff_build_on.d_crp_radpp1 = scratch->d_crp_radpp1 + cell_offset;
        coeff_build_on.d_cre_radpm1 = scratch->d_cre_radpm1 + cell_offset;
        coeff_build_on.d_cre_radpp1 = scratch->d_cre_radpp1 + cell_offset;
        coeff_build_on.d_dppm1 = scratch->d_dppm1 + cell_offset;
        coeff_build_on.d_dppp1 = scratch->d_dppp1 + cell_offset;
        coeff_build_on.d_dppem1 = scratch->d_dppem1 + cell_offset;
        coeff_build_on.d_dppep1 = scratch->d_dppep1 + cell_offset;
        coeff_build_on.d_ccp_a = scratch->d_ccp_a + np_off;
        coeff_build_on.d_ccp_b = scratch->d_ccp_b + np_off;
        coeff_build_on.d_ccp_c = scratch->d_ccp_c + np_off;
        coeff_build_on.d_cce_a = scratch->d_cce_a + npe_off;
        coeff_build_on.d_cce_b = scratch->d_cce_b + npe_off;
        coeff_build_on.d_cce_c = scratch->d_cce_c + npe_off;
        if (fp_cuda_build_cc_coeffs(&coeff_build_on, label) != 0) return -1;

        coeff_build_off = coeff_build_on;
        coeff_build_off.d_dpp = scratch->d_dpp_off + np_off;
        coeff_build_off.d_dppe = scratch->d_dppe_off + npe_off;
        coeff_build_off.d_dppm1 = scratch->d_dppm1_off + cell_offset;
        coeff_build_off.d_dppp1 = scratch->d_dppp1_off + cell_offset;
        coeff_build_off.d_dppem1 = scratch->d_dppem1_off + cell_offset;
        coeff_build_off.d_dppep1 = scratch->d_dppep1_off + cell_offset;
        coeff_build_off.d_ccp_a = scratch->d_ccp_a_off + np_off;
        coeff_build_off.d_ccp_b = scratch->d_ccp_b_off + np_off;
        coeff_build_off.d_ccp_c = scratch->d_ccp_c_off + np_off;
        coeff_build_off.d_cce_a = scratch->d_cce_a_off + npe_off;
        coeff_build_off.d_cce_b = scratch->d_cce_b_off + npe_off;
        coeff_build_off.d_cce_c = scratch->d_cce_c_off + npe_off;
        return fp_cuda_build_cc_coeffs(&coeff_build_off, "launch ws build_cc_coeffs_kernel_off");
    };

    {
        const size_t np_batch = (size_t)total_ncell * (size_t)np;
        const size_t npe_batch = (size_t)total_ncell * (size_t)npe;
        const auto total_t0 = std::chrono::steady_clock::now();
        const auto h2d_t0 = std::chrono::steady_clock::now();
        int cell_offset = 0;

        cuda_check(cudaMemcpy(scratch->d_n_gas, base->coeff_in->n_gas, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group n_gas");
        cuda_check(cudaMemcpy(scratch->d_kbt, base->coeff_in->kbt, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group kbt");
        cuda_check(cudaMemcpy(scratch->d_b_field, base->coeff_in->b_field, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group b_field");
        cuda_check(cudaMemcpy(scratch->d_divv, base->coeff_in->divv_gyr, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group divv");
        cuda_check(cudaMemcpy(scratch->d_l_turb_mpc, base->coeff_in->l_turb_mpc, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group l_turb_mpc");
        cuda_check(cudaMemcpy(scratch->d_dv_imc, base->coeff_in->dv_imc, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group dv_imc");
        cuda_check(cudaMemcpy(scratch->d_cs, base->coeff_in->cs, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group cs");
        cuda_check(cudaMemcpy(scratch->d_beta_pl, base->coeff_in->beta_pl, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group beta_pl");
        if (base->coeff_in->tracer_mass_msun != nullptr) {
            cuda_check(cudaMemcpy(scratch->d_tracer_mass_msun, base->coeff_in->tracer_mass_msun,
                                  (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice),
                       "copy ws group tracer_mass_msun");
        }
        ws->host.merged_rad_ic_batch.resize((size_t)total_ncell * (size_t)npe);
        for (int i = 0; i < ninput; i++) {
            const int ncell = inputs[i].ncell;
            const double *src = inputs[i].coeff_in->rad_ic_batch;
            for (int je = 0; je < npe; je++) {
                std::memcpy(ws->host.merged_rad_ic_batch.data() +
                                (size_t)je * (size_t)total_ncell + (size_t)cell_offset,
                            src + (size_t)je * (size_t)ncell,
                            (size_t)ncell * sizeof(double));
            }
            cell_offset += ncell;
        }
        cuda_check(cudaMemcpy(scratch->d_rad_ic_batch, ws->host.merged_rad_ic_batch.data(),
                              npe_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws group rad_ic_batch");
        cuda_check(cudaMemcpy(scratch->d_rad_ic_m1, base->coeff_in->rad_ic_m1, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group rad_ic_m1");
        cuda_check(cudaMemcpy(scratch->d_rad_ic_p1, base->coeff_in->rad_ic_p1, (size_t)total_ncell * sizeof(double), cudaMemcpyHostToDevice), "copy ws group rad_ic_p1");
        cuda_check(cudaMemcpy(scratch->d_qpi_cell, base->qpi_batch, np_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws group qpi_batch");
        cuda_check(cudaMemcpy(scratch->d_qepri_cell, base->qepri_batch, npe_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws group qepri_batch");
        cuda_check(cudaMemcpy(scratch->d_crp_state, base->crp_init, np_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws group crp_init");
        cuda_check(cudaMemcpy(scratch->d_cre_state, base->cre_init, npe_batch * sizeof(double), cudaMemcpyHostToDevice), "copy ws group cre_init");
        times->h2d_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - h2d_t0).count();

        cuda_check(cudaEventRecord(events->coeff_start), "record ws coeff_start");
        std::memset(&coeff_prepare, 0, sizeof(coeff_prepare));
        coeff_prepare.ncell = total_ncell;
        coeff_prepare.cell_blocks = (total_ncell + cell_threads - 1) / cell_threads;
        coeff_prepare.cell_threads = cell_threads;
        coeff_prepare.active_model = active_model;
        coeff_prepare.epmax = base->epmax;
        coeff_prepare.psi_value = base->psi_value;
        coeff_prepare.mach_limit_value = base->mach_limit_value;
        coeff_prepare.ttd_tacc_model = ttd_tacc_model;
        coeff_prepare.t_acc_direct_gyr = t_acc_direct_gyr;
        coeff_prepare.eta_dpp_cap_value = eta_dpp_cap;
        coeff_prepare.d_n_gas = scratch->d_n_gas;
        coeff_prepare.d_kbt = scratch->d_kbt;
        coeff_prepare.d_b_field = scratch->d_b_field;
        coeff_prepare.d_divv = scratch->d_divv;
        coeff_prepare.d_rad_ic_batch = scratch->d_rad_ic_batch;
        coeff_prepare.d_rad_ic_m1 = scratch->d_rad_ic_m1;
        coeff_prepare.d_rad_ic_p1 = scratch->d_rad_ic_p1;
        coeff_prepare.d_l_turb_mpc = scratch->d_l_turb_mpc;
        coeff_prepare.d_dv_imc = scratch->d_dv_imc;
        coeff_prepare.d_cs = scratch->d_cs;
        coeff_prepare.d_beta_pl = scratch->d_beta_pl;
        coeff_prepare.d_crp_p = cache->d_crp_p;
        coeff_prepare.d_crp_e = cache->d_crp_e;
        coeff_prepare.d_crp_dp = cache->d_crp_dp;
        coeff_prepare.d_crp_p2 = cache->d_crp_p2;
        coeff_prepare.d_crp_exp_cut = cache->d_crp_exp_cut;
        coeff_prepare.d_crp_sigma_pp = cache->d_crp_sigma_pp;
        coeff_prepare.d_crp_sigmoid_pp = cache->d_crp_sigmoid_pp;
        coeff_prepare.d_cre_p = cache->d_cre_p;
        coeff_prepare.d_cre_dp = cache->d_cre_dp;
        coeff_prepare.d_cre_p2 = cache->d_cre_p2;
        coeff_prepare.d_cre_exp_cut = cache->d_cre_exp_cut;
        coeff_prepare.d_tracer_mass_msun =
            (base->coeff_in->tracer_mass_msun != nullptr) ? scratch->d_tracer_mass_msun : nullptr;
        coeff_prepare.d_crp_state = scratch->d_crp_state;
        coeff_prepare.d_cre_state = scratch->d_cre_state;
        coeff_prepare.crp_pm1 = base->crp_grid->pm1;
        coeff_prepare.crp_pp1 = base->crp_grid->pp1;
        coeff_prepare.cre_pm1 = base->cre_grid->pm1;
        coeff_prepare.cre_pp1 = base->cre_grid->pp1;
        coeff_prepare.crp_j_pp = cache->crp_j_pp;
        coeff_prepare.crp_pm1_p2 = cache->crp_pm1_p2;
        coeff_prepare.crp_pp1_p2 = cache->crp_pp1_p2;
        coeff_prepare.crp_pm1_exp_cut = cache->crp_pm1_exp_cut;
        coeff_prepare.crp_pp1_exp_cut = cache->crp_pp1_exp_cut;
        coeff_prepare.cre_pm1_p2 = cache->cre_pm1_p2;
        coeff_prepare.cre_pp1_p2 = cache->cre_pp1_p2;
        coeff_prepare.cre_pm1_exp_cut = cache->cre_pm1_exp_cut;
        coeff_prepare.cre_pp1_exp_cut = cache->cre_pp1_exp_cut;
        coeff_prepare.d_crp_radp = scratch->d_crp_radp;
        coeff_prepare.d_crp_tloss = scratch->d_crp_tloss;
        coeff_prepare.d_cre_radp = scratch->d_cre_radp;
        coeff_prepare.d_cre_tloss = scratch->d_cre_tloss;
        coeff_prepare.d_crp_radpm1 = scratch->d_crp_radpm1;
        coeff_prepare.d_crp_radpp1 = scratch->d_crp_radpp1;
        coeff_prepare.d_cre_radpm1 = scratch->d_cre_radpm1;
        coeff_prepare.d_cre_radpp1 = scratch->d_cre_radpp1;
        coeff_prepare.d_dpp = scratch->d_dpp;
        coeff_prepare.d_dppm1 = scratch->d_dppm1;
        coeff_prepare.d_dppp1 = scratch->d_dppp1;
        coeff_prepare.d_dppe = scratch->d_dppe;
        coeff_prepare.d_dppem1 = scratch->d_dppem1;
        coeff_prepare.d_dppep1 = scratch->d_dppep1;
        coeff_prepare.d_dpp_off = scratch->d_dpp_off;
        coeff_prepare.d_dppm1_off = scratch->d_dppm1_off;
        coeff_prepare.d_dppp1_off = scratch->d_dppp1_off;
        coeff_prepare.d_dppe_off = scratch->d_dppe_off;
        coeff_prepare.d_dppem1_off = scratch->d_dppem1_off;
        coeff_prepare.d_dppep1_off = scratch->d_dppep1_off;
        if (fp_cuda_prepare_coeff_terms(&coeff_prepare,
                                        "launch ws prepare_losses_kernel",
                                        "launch ws prepare_momentumdiff_model_kernel",
                                        "launch ws prepare_momentumdiff_off_kernel") != 0) {
            return -1;
        }
        cell_offset = 0;
        for (int i = 0; i < ninput; i++) {
            if (build_ws_cc_coeffs_slice(inputs[i].ncell, cell_offset, inputs[i].dt,
                                         "launch ws build_cc_coeffs_kernel") != 0) {
                return -1;
            }
            cell_offset += inputs[i].ncell;
        }
        cuda_check(cudaEventRecord(events->coeff_stop), "record ws coeff_stop");
        cuda_check(cudaEventSynchronize(events->coeff_stop), "sync ws coeff_stop");

        cuda_check(cudaEventRecord(events->secondary_start), "record ws secondary_start");
        prepare_secondary_sources_cell_major_kernel<<<total_ncell, npe>>>(
            total_ncell, scratch->d_n_gas, scratch->d_crp_state, cache->d_crp_dp,
            cache->d_np_min_qe, cache->d_fqe_flat,
            tracer_fp_secondary_transpose_enabled(),
            scratch->d_qepri_cell, scratch->d_qe_integral_cell, scratch->d_inje_cell);
        cuda_check(cudaGetLastError(), "launch ws group prepare_secondary_sources");
        cuda_check(cudaEventRecord(events->secondary_stop), "record ws secondary_stop");
        cuda_check(cudaEventSynchronize(events->secondary_stop), "sync ws secondary_stop");
        {
            float secondary_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&secondary_ms, events->secondary_start, events->secondary_stop),
                       "elapsed ws secondary");
            times->secondary_ms += secondary_ms;
        }

        cuda_check(cudaEventRecord(events->solve_start), "record ws solve_start");
        cell_offset = 0;
        for (int i = 0; i < ninput; i++) {
            const FpCudaPipelineInput *in = &inputs[i];
            const size_t np_off = (size_t)cell_offset * (size_t)np;
            const size_t npe_off = (size_t)cell_offset * (size_t)npe;
            int fp_acc_steps = 0;
            int executed_fp_steps = 0;
            double coeff_dt_built = in->dt;

            for (int istep = 0; istep < in->nstep; istep++) {
                fp_acc_steps++;
                if (fp_acc_steps >= in->fp_cadence_steps || istep == in->nstep - 1) {
                    const double dt_fp_step = (double)fp_acc_steps * in->dt;
                    const bool use_on_coeff =
                        !in->use_windowed_reacc ||
                        (executed_fp_steps >= in->on_start_step && executed_fp_steps < in->on_end_step);
                    const double *d_ccp_a_step = use_on_coeff ? scratch->d_ccp_a + np_off : scratch->d_ccp_a_off + np_off;
                    const double *d_ccp_b_step = use_on_coeff ? scratch->d_ccp_b + np_off : scratch->d_ccp_b_off + np_off;
                    const double *d_ccp_c_step = use_on_coeff ? scratch->d_ccp_c + np_off : scratch->d_ccp_c_off + np_off;
                    const double *d_cce_a_step = use_on_coeff ? scratch->d_cce_a + npe_off : scratch->d_cce_a_off + npe_off;
                    const double *d_cce_b_step = use_on_coeff ? scratch->d_cce_b + npe_off : scratch->d_cce_b_off + npe_off;
                    const double *d_cce_c_step = use_on_coeff ? scratch->d_cce_c + npe_off : scratch->d_cce_c_off + npe_off;

                    if (std::fabs(dt_fp_step - coeff_dt_built) > 1.0e-15) {
                        if (build_ws_cc_coeffs_slice(in->ncell, cell_offset, dt_fp_step,
                                                     "launch ws build_cc_coeffs_kernel loop") != 0) {
                            return -1;
                        }
                        coeff_dt_built = dt_fp_step;
                    }

                    if (fp_cuda_solve_device_batch(in->ncell, np, dt_fp_step,
                                                   d_ccp_a_step, d_ccp_b_step, d_ccp_c_step,
                                                   scratch->d_qpi_cell + np_off, scratch->d_crp_state + np_off,
                                                   "launch ws group solve proton", 0) != 0) {
                        return -1;
                    }
                    if (fp_cuda_solve_device_batch(in->ncell, npe, dt_fp_step,
                                                   d_cce_a_step, d_cce_b_step, d_cce_c_step,
                                                   scratch->d_inje_cell + npe_off, scratch->d_cre_state + npe_off,
                                                   "launch ws group solve electron", 0) != 0) {
                        return -1;
                    }

                    fp_acc_steps = 0;
                    executed_fp_steps++;
                }
            }

            cell_offset += in->ncell;
        }
        cuda_check(cudaEventRecord(events->solve_stop), "record ws solve_stop");
        cuda_check(cudaEventSynchronize(events->solve_stop), "sync ws solve_stop");

        {
            float coeff_ms = 0.0f;
            float solve_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&coeff_ms, events->coeff_start, events->coeff_stop), "elapsed ws coeff");
            cuda_check(cudaEventElapsedTime(&solve_ms, events->solve_start, events->solve_stop), "elapsed ws solve");
            times->coeff_ms = coeff_ms;
            times->solve_ms = solve_ms;
            times->used_pcr_proton = use_pcr_proton ? 1 : 0;
            times->used_pcr_electron = use_pcr_electron ? 1 : 0;
        }

        {
            const auto d2h_t0 = std::chrono::steady_clock::now();
            cuda_check(cudaMemcpy(base->crp_state_out, scratch->d_crp_state,
                                  np_batch * sizeof(double), cudaMemcpyDeviceToHost), "copy ws group crp_state_out");
            cuda_check(cudaMemcpy(base->cre_state_out, scratch->d_cre_state,
                                  npe_batch * sizeof(double), cudaMemcpyDeviceToHost), "copy ws group cre_state_out");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize ws group");
            times->d2h_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - d2h_t0).count();
        }

        times->total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_t0).count();
        times->other_ms = times->total_ms - (times->setup_ms + times->h2d_ms +
                                             times->coeff_ms + times->secondary_ms +
                                             times->solve_ms + times->synch_ms +
                                             times->gamma_ms + times->neutrino_ms +
                                             times->d2h_ms);
        if (times->other_ms < 0.0) times->other_ms = 0.0;
    }

    return 0;
}

}  // namespace

extern "C" int cuda_pipeline_is_available(void)
{
    return 1;
}

extern "C" int cuda_bind_local_rank(int local_rank, int *device_out)
{
    int device_count = 0;
    int bound_device;

    if (cudaGetDeviceCount(&device_count) != cudaSuccess) return -1;
    if (device_count <= 0) return -1;

    if (local_rank < 0) local_rank = 0;
    bound_device = local_rank % device_count;
    if (cudaSetDevice(bound_device) != cudaSuccess) return -1;

    if (device_out != nullptr) *device_out = bound_device;
    return 0;
}

extern "C" int cuda_get_binding_state(int *device_count_out, int *current_device_out)
{
    int device_count = 0;
    int current_device = -1;

    if (cudaGetDeviceCount(&device_count) != cudaSuccess) return -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) current_device = -1;

    if (device_count_out != nullptr) *device_count_out = device_count;
    if (current_device_out != nullptr) *current_device_out = current_device;
    return 0;
}

extern "C" int cuda_get_current_device_id(char *device_id_out, int device_id_out_len)
{
    int current_device = -1;

    if (device_id_out == nullptr || device_id_out_len <= 0) return -1;
    device_id_out[0] = '\0';
    if (cudaGetDevice(&current_device) != cudaSuccess) return -1;
    if (cudaDeviceGetPCIBusId(device_id_out, device_id_out_len, current_device) != cudaSuccess) {
        device_id_out[0] = '\0';
        return -1;
    }
    return 0;
}

extern "C" int run_tracer_pipeline(const FpCudaPipelineInput *in,
                                   const FpCudaOutputSchedule *schedule,
                                   double *crp_state_out,
                                   double *cre_state_out,
                                   double *eps_syn_out,
                                   double *eps_gamma_out,
                                   double *eps_nu_out,
                                   FpCudaPipelineTimes *times)
{
    FpCudaPipelineInput single_in;

    if (in == nullptr || times == nullptr) return -1;
    if (eps_gamma_out != nullptr || in->gamma_in != nullptr ||
        eps_nu_out != nullptr || in->neutrino_in != nullptr) return -1;
    if (in->workspace == nullptr) return -1;

    single_in = *in;
    single_in.crp_state_out = crp_state_out;
    single_in.cre_state_out = cre_state_out;
    single_in.eps_syn_out = eps_syn_out;
    single_in.eps_gamma_out = eps_gamma_out;
    single_in.eps_nu_out = eps_nu_out;
    switch (single_in.transport_mode) {
        case FP_CUDA_TRANSPORT_FUSED_SUBSTEPS:
            return run_fused_substeps(&single_in, schedule,
                                      single_in.crp_state_out,
                                      single_in.cre_state_out,
                                      single_in.eps_syn_out,
                                      times);
        case FP_CUDA_TRANSPORT_LEGACY:
        default:
            return run_cached_simple(&single_in, schedule,
                                     single_in.crp_state_out,
                                     single_in.cre_state_out,
                                     single_in.eps_syn_out,
                                     times);
    }
}

extern "C" int run_tracer_pipeline_group(const FpCudaPipelineInput *inputs,
                                         int ninput,
                                         const FpCudaOutputSchedule *schedule,
                                         FpCudaPipelineTimes *times)
{
    int i;

    if (inputs == nullptr || ninput <= 0 || times == nullptr) return -1;
    if (inputs[0].transport_mode == FP_CUDA_TRANSPORT_LEGACY) {
        if (run_group_merged_strict(inputs, ninput, schedule, times) == 0) {
            return 0;
        }
        if (run_group_shared_coeff(inputs, ninput, schedule, times) == 0) {
            return 0;
        }
    }
    std::memset(times, 0, sizeof(*times));

    for (i = 0; i < ninput; i++) {
        FpCudaPipelineTimes local_times;
        if (run_tracer_pipeline(&inputs[i], schedule,
                                inputs[i].crp_state_out,
                                inputs[i].cre_state_out,
                                inputs[i].eps_syn_out,
                                inputs[i].eps_gamma_out,
                                inputs[i].eps_nu_out,
                                &local_times) != 0) {
            return -1;
        }
        accum_pipeline_times(times, &local_times);
    }
    return 0;
}
