#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fp_cuda_compat.h"
#include "fp_cuda_emission.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kC = 2.999e10;
constexpr double kLogBnorm = -6.0;

__device__ __forceinline__ double interp_synch_kernel_logk(double k0,
                                                           double k1,
                                                           double t)
{
    if (t <= 0.0) return k0;
    if (t >= 1.0) return k1;
    if (k0 > 0.0 && k1 > 0.0) {
        return exp((1.0 - t) * log(k0) + t * log(k1));
    }
    return (1.0 - t) * k0 + t * k1;
}

struct FpCudaSynchBatchCache {
    int device_ordinal = -1;
    int capacity_ncell = 0;
    int capacity_nfreq = 0;
    int capacity_nx_tab = 0;
    int ntheta_pitch_key = 0;
    size_t pitch_kernel_table_capacity = 0;

    const CRspectrum *cre_grid_key = nullptr;
    const double *fx_tab_key = nullptr;
    const double *logx_tab_key = nullptr;
    const double *pitch_kernel_table_key = nullptr;
    const double *pitch_weight_key = nullptr;
    const double *theta_key = nullptr;
    const double *nus_key = nullptr;
    int nfreq_key = 0;
    int pitch_kernel_nlogb_key = 0;
    double pitch_kernel_logb_min_key = 0.0;
    double pitch_kernel_inv_dlogb_key = 0.0;

    double *d_b_dyn = nullptr;
    double *d_logb = nullptr;
    double *d_cre_state = nullptr;
    double *d_cre_dp = nullptr;
    double *d_fx_tab = nullptr;
    double *d_logx_tab = nullptr;
    double *d_lognu_syn = nullptr;
    double *d_lognu_crit = nullptr;
    double *d_pitch_weight = nullptr;
    double *d_pitch_kernel_table = nullptr;
    double *d_eps = nullptr;

    std::vector<double> lognu_syn;
    std::vector<double> nu_crit_storage;
    std::vector<double *> nu_crit_rows;
    std::vector<double> lognu_crit;
};

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

static void reset_batch_cache(FpCudaSynchBatchCache *cache)
{
    cudaFree(cache->d_b_dyn); cache->d_b_dyn = nullptr;
    cudaFree(cache->d_logb); cache->d_logb = nullptr;
    cudaFree(cache->d_cre_state); cache->d_cre_state = nullptr;
    cudaFree(cache->d_cre_dp); cache->d_cre_dp = nullptr;
    cudaFree(cache->d_fx_tab); cache->d_fx_tab = nullptr;
    cudaFree(cache->d_logx_tab); cache->d_logx_tab = nullptr;
    cudaFree(cache->d_lognu_syn); cache->d_lognu_syn = nullptr;
    cudaFree(cache->d_lognu_crit); cache->d_lognu_crit = nullptr;
    cudaFree(cache->d_pitch_weight); cache->d_pitch_weight = nullptr;
    cudaFree(cache->d_pitch_kernel_table); cache->d_pitch_kernel_table = nullptr;
    cudaFree(cache->d_eps); cache->d_eps = nullptr;

    cache->capacity_ncell = 0;
    cache->capacity_nfreq = 0;
    cache->capacity_nx_tab = 0;
    cache->ntheta_pitch_key = 0;
    cache->pitch_kernel_table_capacity = 0;
    cache->cre_grid_key = nullptr;
    cache->fx_tab_key = nullptr;
    cache->logx_tab_key = nullptr;
    cache->pitch_kernel_table_key = nullptr;
    cache->pitch_weight_key = nullptr;
    cache->theta_key = nullptr;
    cache->nus_key = nullptr;
    cache->nfreq_key = 0;
    cache->pitch_kernel_nlogb_key = 0;
    cache->pitch_kernel_logb_min_key = 0.0;
    cache->pitch_kernel_inv_dlogb_key = 0.0;
    cache->lognu_syn.clear();
    cache->nu_crit_storage.clear();
    cache->nu_crit_rows.clear();
    cache->lognu_crit.clear();
}

static FpCudaSynchBatchCache *synch_batch_cache(void)
{
    static FpCudaSynchBatchCache cache;
    int current_device = -1;

    cuda_check(cudaGetDevice(&current_device), "get current device for synch batch");
    if (cache.device_ordinal != current_device) {
        reset_batch_cache(&cache);
        cache.device_ordinal = current_device;
    }
    return &cache;
}

static void ensure_dynamic_buffers(FpCudaSynchBatchCache *cache,
                                               int ncell,
                                               int nfreq,
                                               int nx_tab)
{
    if (cache->capacity_ncell < ncell) {
        cudaFree(cache->d_b_dyn); cache->d_b_dyn = nullptr;
        cudaFree(cache->d_logb); cache->d_logb = nullptr;
        cudaFree(cache->d_cre_state); cache->d_cre_state = nullptr;
        cudaFree(cache->d_eps); cache->d_eps = nullptr;
        cache->capacity_ncell = ncell;
    }

    if (cache->capacity_nfreq < nfreq) {
        cudaFree(cache->d_eps); cache->d_eps = nullptr;
        cudaFree(cache->d_lognu_syn); cache->d_lognu_syn = nullptr;
        cudaFree(cache->d_pitch_kernel_table); cache->d_pitch_kernel_table = nullptr;
        cache->capacity_nfreq = nfreq;
        cache->pitch_kernel_table_capacity = 0;
        cache->nus_key = nullptr;
        cache->nfreq_key = 0;
        cache->pitch_kernel_table_key = nullptr;
        cache->pitch_kernel_nlogb_key = 0;
    }

    if (cache->capacity_nx_tab < nx_tab) {
        cudaFree(cache->d_fx_tab); cache->d_fx_tab = nullptr;
        cudaFree(cache->d_logx_tab); cache->d_logx_tab = nullptr;
        cache->capacity_nx_tab = nx_tab;
        cache->fx_tab_key = nullptr;
        cache->logx_tab_key = nullptr;
    }

    ensure_device_buffer(&cache->d_b_dyn, (size_t)cache->capacity_ncell, "alloc synch cache d_b_dyn");
    ensure_device_buffer(&cache->d_logb, (size_t)cache->capacity_ncell, "alloc synch cache d_logb");
    ensure_device_buffer(&cache->d_cre_state, (size_t)cache->capacity_ncell * (size_t)npe, "alloc synch cache d_cre_state");
    ensure_device_buffer(&cache->d_eps, (size_t)cache->capacity_ncell * (size_t)cache->capacity_nfreq, "alloc synch cache d_eps");
    ensure_device_buffer(&cache->d_fx_tab, (size_t)cache->capacity_nx_tab, "alloc synch cache d_fx_tab");
    ensure_device_buffer(&cache->d_logx_tab, (size_t)cache->capacity_nx_tab, "alloc synch cache d_logx_tab");
    ensure_device_buffer(&cache->d_lognu_syn, (size_t)cache->capacity_nfreq, "alloc synch cache d_lognu_syn");
}

static void ensure_constants(FpCudaSynchBatchCache *cache,
                                         const FpSynchEmissionBatchInput *in,
                                         const CRspectrum *cre_grid)
{
    if (cache->d_cre_dp == nullptr || cache->cre_grid_key != cre_grid) {
        ensure_device_buffer(&cache->d_cre_dp, (size_t)npe, "alloc synch cache d_cre_dp");
        cuda_check(cudaMemcpy(cache->d_cre_dp, cre_grid->dp,
                              (size_t)npe * sizeof(double), cudaMemcpyHostToDevice),
                   "copy synch cache cre_dp");
        cache->cre_grid_key = cre_grid;
        cache->theta_key = nullptr;
    }

    if (cache->ntheta_pitch_key != in->ntheta_pitch) {
        cudaFree(cache->d_pitch_weight); cache->d_pitch_weight = nullptr;
        cudaFree(cache->d_lognu_crit); cache->d_lognu_crit = nullptr;
        cache->pitch_weight_key = nullptr;
        cache->theta_key = nullptr;
        cache->ntheta_pitch_key = in->ntheta_pitch;
    }

    if (cache->pitch_weight_key != in->pitch_weight) {
        ensure_device_buffer(&cache->d_pitch_weight, (size_t)in->ntheta_pitch, "alloc synch cache d_pitch_weight");
        cuda_check(cudaMemcpy(cache->d_pitch_weight, in->pitch_weight,
                              (size_t)in->ntheta_pitch * sizeof(double), cudaMemcpyHostToDevice),
                   "copy synch cache pitch_weight");
        cache->pitch_weight_key = in->pitch_weight;
    }

    if (cache->fx_tab_key != in->fx_tab) {
        cuda_check(cudaMemcpy(cache->d_fx_tab, in->fx_tab,
                              (size_t)in->nx_tab * sizeof(double), cudaMemcpyHostToDevice),
                   "copy synch cache fx_tab");
        cache->fx_tab_key = in->fx_tab;
    }

    if (cache->logx_tab_key != in->logx_tab) {
        cuda_check(cudaMemcpy(cache->d_logx_tab, in->logx_tab,
                              (size_t)in->nx_tab * sizeof(double), cudaMemcpyHostToDevice),
                   "copy synch cache logx_tab");
        cache->logx_tab_key = in->logx_tab;
    }

    if (cache->nus_key != in->nus || cache->nfreq_key != in->nfreq) {
        cache->lognu_syn.resize((size_t)in->nfreq);
        for (int nf = 0; nf < in->nfreq; nf++) {
            cache->lognu_syn[(size_t)nf] = std::log10(in->nus[nf]);
        }
        cuda_check(cudaMemcpy(cache->d_lognu_syn, cache->lognu_syn.data(),
                              (size_t)in->nfreq * sizeof(double), cudaMemcpyHostToDevice),
                   "copy synch cache lognu_syn");
        cache->nus_key = in->nus;
        cache->nfreq_key = in->nfreq;
    }

    if (cache->d_lognu_crit == nullptr ||
        cache->theta_key != in->theta ||
        cache->cre_grid_key != cre_grid) {
        cache->nu_crit_storage.resize((size_t)npe * (size_t)in->ntheta_pitch);
        cache->nu_crit_rows.resize((size_t)npe);
        cache->lognu_crit.resize((size_t)npe * (size_t)in->ntheta_pitch);
        for (int je = 0; je < npe; je++) {
            cache->nu_crit_rows[(size_t)je] =
                cache->nu_crit_storage.data() + (size_t)je * (size_t)in->ntheta_pitch;
        }
        SYN_nu_crit_subB(cache->nu_crit_rows.data(), (double *)in->theta, (double *)cre_grid->p);
        for (int je = 0; je < npe; je++) {
            for (int k = 0; k < in->ntheta_pitch; k++) {
                const size_t off = (size_t)je * (size_t)in->ntheta_pitch + (size_t)k;
                cache->lognu_crit[off] = std::log10(cache->nu_crit_storage[off]);
            }
        }
        ensure_device_buffer(&cache->d_lognu_crit, cache->lognu_crit.size(), "alloc synch cache d_lognu_crit");
        cuda_check(cudaMemcpy(cache->d_lognu_crit, cache->lognu_crit.data(),
                              cache->lognu_crit.size() * sizeof(double), cudaMemcpyHostToDevice),
                   "copy synch cache lognu_crit");
        cache->theta_key = in->theta;
    }

    if (in->pitch_kernel_table != nullptr &&
        in->nlogb > 1 &&
        in->inv_dlogb > 0.0) {
        const size_t table_size =
            (size_t)in->nlogb * (size_t)in->nfreq * (size_t)npe;
        if (cache->pitch_kernel_table_capacity != table_size) {
            cudaFree(cache->d_pitch_kernel_table);
            cache->d_pitch_kernel_table = nullptr;
            ensure_device_buffer(&cache->d_pitch_kernel_table, table_size,
                                 "alloc synch cache d_pitch_kernel_table");
            cache->pitch_kernel_table_capacity = table_size;
            cache->pitch_kernel_table_key = nullptr;
        }
        if (cache->pitch_kernel_table_key != in->pitch_kernel_table ||
            cache->pitch_kernel_nlogb_key != in->nlogb ||
            cache->pitch_kernel_logb_min_key != in->logb_min ||
            cache->pitch_kernel_inv_dlogb_key != in->inv_dlogb) {
            cuda_check(cudaMemcpy(cache->d_pitch_kernel_table,
                                  in->pitch_kernel_table,
                                  table_size * sizeof(double),
                                  cudaMemcpyHostToDevice),
                       "copy synch cache pitch_kernel_table");
            cache->pitch_kernel_table_key = in->pitch_kernel_table;
            cache->pitch_kernel_nlogb_key = in->nlogb;
            cache->pitch_kernel_logb_min_key = in->logb_min;
            cache->pitch_kernel_inv_dlogb_key = in->inv_dlogb;
        }
    } else {
        cache->pitch_kernel_table_key = nullptr;
        cache->pitch_kernel_nlogb_key = 0;
    }
}

__global__ void prepare_synch_emission_cell_major_kernel(int ncell,
                                                         int nfreq,
                                                         int nx_tab,
                                                         int ntheta_pitch,
                                                         double xmin,
                                                         int nlogb,
                                                         double logb_min,
                                                         double inv_dlogb,
                                                         const double *fx_tab,
                                                         const double *logx_tab,
                                                         const double *lognu_syn,
                                                         const double *lognu_crit,
                                                         const double *pitch_kernel_table,
                                                         const double *b_dyn,
                                                         const double *logb,
                                                         const double *cre_cell_major,
                                                         const double *cre_dp,
                                                         const double *pitch_weight,
                                                         double *eps_syn_batch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ncell * nfreq;
    const double meg = 9.11e-28;
    const double e = 4.8e-10;

    if (idx >= total) return;

    const int icell = idx / nfreq;
    const int ifreq = idx % nfreq;
    const double B = b_dyn[icell];
    const double logB = logb[icell];
    const double A = std::sqrt(3.0) * e * e * e * B /
                     (2.0 * meg * kC * kC) / (4.0 * kPi);
    double integral = 0.0;

    if (pitch_kernel_table != nullptr && nlogb > 1 && inv_dlogb > 0.0) {
        double u = (logB - logb_min) * inv_dlogb;
        int ib = (int)floor(u);
        double t;
        if (ib < 0) ib = 0;
        if (ib > nlogb - 2) ib = nlogb - 2;
        t = u - (double)ib;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;

        for (int je = 2; je < npe - 1; je++) {
            const double next_ne = cre_cell_major[(size_t)icell * (size_t)npe + (size_t)(je + 1)];
            if (next_ne <= 0.0) break;

            const size_t off0 = ((size_t)ib * (size_t)nfreq + (size_t)ifreq) * (size_t)npe + (size_t)je;
            const size_t off1 = ((size_t)(ib + 1) * (size_t)nfreq + (size_t)ifreq) * (size_t)npe + (size_t)je;
            const double K = interp_synch_kernel_logk(
                pitch_kernel_table[off0],
                pitch_kernel_table[off1],
                t);
            integral += cre_cell_major[(size_t)icell * (size_t)npe + (size_t)je] *
                        K * cre_dp[je];
        }

        eps_syn_batch[(size_t)icell * (size_t)nfreq + (size_t)ifreq] = A * integral;
        return;
    }

    for (int je = 2; je < npe - 1; je++) {
        const double next_ne = cre_cell_major[(size_t)icell * (size_t)npe + (size_t)(je + 1)];
        double int_th = 0.0;
        if (next_ne <= 0.0) break;

        for (int k = 0; k < ntheta_pitch; k++) {
            const size_t crit_off = (size_t)je * (size_t)ntheta_pitch + (size_t)k;
            const double logx = lognu_syn[(size_t)ifreq] -
                                lognu_crit[crit_off] -
                                kLogBnorm - logB;
            double Fx = 0.0;
            const double inv_dlogx = (nx_tab > 1) ? 1.0 / (logx_tab[1] - logx_tab[0]) : 0.0;

            if (logx <= logx_tab[nx_tab - 1]) {
                const int ix = (int)floor((logx - xmin) * inv_dlogx);
                if (ix >= 0) {
                    if (ix >= nx_tab - 1) {
                        Fx = fx_tab[nx_tab - 1];
                    } else {
                        const double frac = (logx - logx_tab[ix]) * inv_dlogx;
                        const double logFx0 = log10(fx_tab[ix]);
                        const double logFx1 = log10(fx_tab[ix + 1]);
                        Fx = pow(10.0, logFx0 + frac * (logFx1 - logFx0));
                    }
                }
            }
            int_th += Fx * pitch_weight[k];
        }

        integral += cre_cell_major[(size_t)icell * (size_t)npe + (size_t)je] *
                    int_th * cre_dp[je];
    }

    eps_syn_batch[(size_t)icell * (size_t)nfreq + (size_t)ifreq] = A * integral;
}

__global__ void prepare_gamma_emission_cell_major_kernel(int ncell,
                                                         int nbins,
                                                         const double *n_gas,
                                                         const double *crp_cell_major,
                                                         const double *crp_dp,
                                                         const double *beta_p,
                                                         const double *fga_flat,
                                                         double egamma_min,
                                                         double d_egamma,
                                                         double *eps_gamma_batch)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ncell * nbins;
    const int icell = gid / nbins;
    const int ibin = gid % nbins;
    double egamma_log;
    double egamma;
    double integral = 0.0;
    double prev;

    if (gid >= total) return;

    egamma_log = egamma_min + d_egamma * (double)ibin;
    egamma = pow(10.0, egamma_log);
    prev = beta_p[0] * crp_cell_major[(size_t)icell * (size_t)np] *
           fga_flat[(size_t)ibin * (size_t)np];

    for (int jp = 1; jp < np; jp++) {
        const double curr = beta_p[jp] *
            crp_cell_major[(size_t)icell * (size_t)np + (size_t)jp] *
            fga_flat[(size_t)ibin * (size_t)np + (size_t)jp];
        integral += (prev + curr) * crp_dp[jp];
        prev = curr;
    }

    eps_gamma_batch[(size_t)icell * (size_t)nbins + (size_t)ibin] =
        0.5 * egamma * integral * n_gas[icell];
}

__global__ void prepare_neutrino_emission_cell_major_kernel(int ncell,
                                                            int nbins,
                                                            const double *n_gas,
                                                            const double *crp_cell_major,
                                                            const double *crp_dp,
                                                            const double *beta_p,
                                                            const double *fnu_flat,
                                                            double enu_min,
                                                            double d_enu,
                                                            double *eps_nu_batch)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ncell * nbins;
    const int icell = gid / nbins;
    const int ibin = gid % nbins;
    double enu_log;
    double enu;
    double integral = 0.0;
    double prev;

    if (gid >= total) return;

    enu_log = enu_min + d_enu * (double)ibin;
    enu = pow(10.0, enu_log);
    prev = beta_p[0] * crp_cell_major[(size_t)icell * (size_t)np] *
           fnu_flat[(size_t)ibin * (size_t)np];

    for (int jp = 1; jp < np; jp++) {
        const double curr = beta_p[jp] *
            crp_cell_major[(size_t)icell * (size_t)np + (size_t)jp] *
            fnu_flat[(size_t)ibin * (size_t)np + (size_t)jp];
        integral += (prev + curr) * crp_dp[jp];
        prev = curr;
    }

    eps_nu_batch[(size_t)icell * (size_t)nbins + (size_t)ibin] =
        0.5 * enu * integral * n_gas[icell];
}

}  // namespace

int fp_cuda_emit_synch_device(const FpCudaSynchDeviceInput *in,
                              double *elapsed_ms)
{
    const int em_threads = 128;
    cudaEvent_t start = nullptr, stop = nullptr;

    if (elapsed_ms != nullptr) *elapsed_ms = 0.0;
    if (in == nullptr || in->ncell <= 0 || in->nfreq <= 0 || in->nx_tab <= 0 ||
        in->ntheta_pitch <= 0 ||
        in->d_fx_tab == nullptr || in->d_logx_tab == nullptr ||
        in->d_lognu_syn == nullptr || in->d_lognu_crit == nullptr ||
        in->d_b_dyn == nullptr || in->d_logb == nullptr ||
        in->d_cre_state == nullptr || in->d_cre_dp == nullptr ||
        in->d_pitch_weight == nullptr || in->d_eps_out == nullptr) {
        return -1;
    }

    const int em_blocks = (in->ncell * in->nfreq + em_threads - 1) / em_threads;

    if (elapsed_ms != nullptr) {
        cuda_check(cudaEventCreate(&start), "event synch_start");
        cuda_check(cudaEventCreate(&stop), "event synch_stop");
        cuda_check(cudaEventRecord(start), "record synch_start");
    }
    prepare_synch_emission_cell_major_kernel<<<em_blocks, em_threads>>>(
        in->ncell, in->nfreq, in->nx_tab, in->ntheta_pitch, in->xmin,
        in->nlogb, in->logb_min, in->inv_dlogb,
        in->d_fx_tab, in->d_logx_tab, in->d_lognu_syn, in->d_lognu_crit,
        in->d_pitch_kernel_table,
        in->d_b_dyn, in->d_logb, in->d_cre_state, in->d_cre_dp,
        in->d_pitch_weight, in->d_eps_out);
    cuda_check(cudaGetLastError(), "launch prepare_synch_emission_cell_major_kernel");
    if (elapsed_ms != nullptr) {
        float measured = 0.0f;
        cuda_check(cudaEventRecord(stop), "record synch_stop");
        cuda_check(cudaEventSynchronize(stop), "sync synch_stop");
        cuda_check(cudaEventElapsedTime(&measured, start, stop), "elapsed synch");
        *elapsed_ms = measured;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}

int fp_cuda_emit_gamma_device(const FpCudaGammaDeviceInput *in,
                              double *elapsed_ms)
{
    const int threads = 128;
    cudaEvent_t start = nullptr, stop = nullptr;

    if (elapsed_ms != nullptr) *elapsed_ms = 0.0;
    if (in == nullptr || in->ncell <= 0 || in->nbins <= 0 ||
        in->d_n_gas == nullptr || in->d_crp_state == nullptr ||
        in->d_crp_dp == nullptr || in->d_beta_p == nullptr ||
        in->d_fga_flat == nullptr || in->d_eps_out == nullptr) {
        return -1;
    }

    const int blocks = (in->ncell * in->nbins + threads - 1) / threads;
    const double d_egamma = (in->egamma_max - in->egamma_min) / (double)in->nbins;

    if (elapsed_ms != nullptr) {
        cuda_check(cudaEventCreate(&start), "event gamma_start");
        cuda_check(cudaEventCreate(&stop), "event gamma_stop");
        cuda_check(cudaEventRecord(start), "record gamma_start");
    }
    prepare_gamma_emission_cell_major_kernel<<<blocks, threads>>>(
        in->ncell, in->nbins,
        in->d_n_gas, in->d_crp_state, in->d_crp_dp,
        in->d_beta_p, in->d_fga_flat,
        in->egamma_min, d_egamma, in->d_eps_out);
    cuda_check(cudaGetLastError(), "launch prepare_gamma_emission_cell_major_kernel");
    if (elapsed_ms != nullptr) {
        float measured = 0.0f;
        cuda_check(cudaEventRecord(stop), "record gamma_stop");
        cuda_check(cudaEventSynchronize(stop), "sync gamma_stop");
        cuda_check(cudaEventElapsedTime(&measured, start, stop), "elapsed gamma");
        *elapsed_ms = measured;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}

int fp_cuda_emit_neutrino_device(const FpCudaNeutrinoDeviceInput *in,
                                 double *elapsed_ms)
{
    const int threads = 128;
    cudaEvent_t start = nullptr, stop = nullptr;

    if (elapsed_ms != nullptr) *elapsed_ms = 0.0;
    if (in == nullptr || in->ncell <= 0 || in->nbins <= 0 ||
        in->d_n_gas == nullptr || in->d_crp_state == nullptr ||
        in->d_crp_dp == nullptr || in->d_beta_p == nullptr ||
        in->d_fnu_flat == nullptr || in->d_eps_out == nullptr) {
        return -1;
    }

    const int blocks = (in->ncell * in->nbins + threads - 1) / threads;
    const double d_enu = (in->enu_max - in->enu_min) / (double)in->nbins;

    if (elapsed_ms != nullptr) {
        cuda_check(cudaEventCreate(&start), "event neutrino_start");
        cuda_check(cudaEventCreate(&stop), "event neutrino_stop");
        cuda_check(cudaEventRecord(start), "record neutrino_start");
    }
    prepare_neutrino_emission_cell_major_kernel<<<blocks, threads>>>(
        in->ncell, in->nbins,
        in->d_n_gas, in->d_crp_state, in->d_crp_dp,
        in->d_beta_p, in->d_fnu_flat,
        in->enu_min, d_enu, in->d_eps_out);
    cuda_check(cudaGetLastError(), "launch prepare_neutrino_emission_cell_major_kernel");
    if (elapsed_ms != nullptr) {
        float measured = 0.0f;
        cuda_check(cudaEventRecord(stop), "record neutrino_stop");
        cuda_check(cudaEventSynchronize(stop), "sync neutrino_stop");
        cuda_check(cudaEventElapsedTime(&measured, start, stop), "elapsed neutrino");
        *elapsed_ms = measured;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}

extern "C" int prepare_synch_emission_cuda_batch(const FpSynchEmissionBatchInput *in,
                                                 const CRspectrum *cre_grid,
                                                 double *eps_syn_out,
                                                 double *elapsed_ms)
{
    if (elapsed_ms != nullptr) *elapsed_ms = 0.0;
    if (in == nullptr || cre_grid == nullptr || eps_syn_out == nullptr) return -1;
    if (in->ncell <= 0 || in->nfreq <= 0 || in->nx_tab <= 0 ||
        in->ntheta_pitch <= 0 ||
        in->b_dyn == nullptr || in->logb == nullptr || in->cre_batch == nullptr ||
        in->fx_tab == nullptr || in->logx_tab == nullptr ||
        in->pitch_weight == nullptr || in->theta == nullptr || in->nus == nullptr) {
        return -1;
    }

    const auto total_t0 = std::chrono::steady_clock::now();
    const size_t eps_size = (size_t)in->ncell * (size_t)in->nfreq;
    FpCudaSynchBatchCache *cache = synch_batch_cache();

    ensure_dynamic_buffers(cache, in->ncell, in->nfreq, in->nx_tab);
    ensure_constants(cache, in, cre_grid);
    cuda_check(cudaMemcpy(cache->d_b_dyn, in->b_dyn,
                          (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice),
               "copy synch batch b_dyn");
    cuda_check(cudaMemcpy(cache->d_logb, in->logb,
                          (size_t)in->ncell * sizeof(double), cudaMemcpyHostToDevice),
               "copy synch batch logb");
    cuda_check(cudaMemcpy(cache->d_cre_state, in->cre_batch,
                          (size_t)in->ncell * (size_t)npe * sizeof(double), cudaMemcpyHostToDevice),
               "copy synch batch cre_state");

    {
        FpCudaSynchDeviceInput device_in;

        device_in.ncell = in->ncell;
        device_in.nfreq = in->nfreq;
        device_in.nx_tab = in->nx_tab;
        device_in.ntheta_pitch = in->ntheta_pitch;
        device_in.xmin = in->xmin;
        device_in.nlogb = in->nlogb;
        device_in.logb_min = in->logb_min;
        device_in.inv_dlogb = in->inv_dlogb;
        device_in.d_fx_tab = cache->d_fx_tab;
        device_in.d_logx_tab = cache->d_logx_tab;
        device_in.d_lognu_syn = cache->d_lognu_syn;
        device_in.d_lognu_crit = cache->d_lognu_crit;
        device_in.d_pitch_kernel_table =
            (in->pitch_kernel_table != nullptr && in->nlogb > 1 && in->inv_dlogb > 0.0)
            ? cache->d_pitch_kernel_table : nullptr;
        device_in.d_b_dyn = cache->d_b_dyn;
        device_in.d_logb = cache->d_logb;
        device_in.d_cre_state = cache->d_cre_state;
        device_in.d_cre_dp = cache->d_cre_dp;
        device_in.d_pitch_weight = cache->d_pitch_weight;
        device_in.d_eps_out = cache->d_eps;
        if (fp_cuda_emit_synch_device(&device_in, nullptr) != 0) return -1;
    }

    cuda_check(cudaMemcpy(eps_syn_out, cache->d_eps,
                          eps_size * sizeof(double), cudaMemcpyDeviceToHost),
               "copy synch_only eps");
    if (elapsed_ms != nullptr) {
        *elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_t0).count();
    }
    return 0;
}

extern "C" int prepare_gamma_emission_cuda_batch(const FpGammaEmissionBatchInput *in,
                                                 const CRspectrum *crp_grid,
                                                 double *eps_gamma_out,
                                                 double *elapsed_ms)
{
    double *d_n_gas = nullptr, *d_crp_state = nullptr, *d_crp_dp = nullptr;
    double *d_beta_p = nullptr, *d_fga_flat = nullptr, *d_eps = nullptr;

    auto malloc_copy = [](double **dst, const double *src, size_t nbyte, const char *what) {
        cuda_check(cudaMalloc(dst, nbyte), what);
        cuda_check(cudaMemcpy(*dst, src, nbyte, cudaMemcpyHostToDevice), what);
    };

    if (elapsed_ms != nullptr) *elapsed_ms = 0.0;
    if (in == nullptr || crp_grid == nullptr || eps_gamma_out == nullptr) return -1;
    if (in->ncell <= 0 || in->nbins <= 0 || in->n_gas == nullptr ||
        in->crp_batch == nullptr || in->beta_p == nullptr || in->fga_flat == nullptr) {
        return -1;
    }

    const auto total_t0 = std::chrono::steady_clock::now();
    const size_t eps_size = (size_t)in->ncell * (size_t)in->nbins;

    malloc_copy(&d_n_gas, in->n_gas, (size_t)in->ncell * sizeof(double), "copy gamma n_gas");
    malloc_copy(&d_crp_state, in->crp_batch, (size_t)in->ncell * (size_t)np * sizeof(double), "copy gamma crp_batch");
    malloc_copy(&d_crp_dp, crp_grid->dp, (size_t)np * sizeof(double), "copy gamma crp_dp");
    malloc_copy(&d_beta_p, in->beta_p, (size_t)np * sizeof(double), "copy gamma beta_p");
    malloc_copy(&d_fga_flat, in->fga_flat, (size_t)in->nbins * (size_t)np * sizeof(double), "copy gamma fga_flat");
    cuda_check(cudaMalloc(&d_eps, eps_size * sizeof(double)), "alloc gamma eps");

    {
        FpCudaGammaDeviceInput device_in;

        device_in.ncell = in->ncell;
        device_in.nbins = in->nbins;
        device_in.egamma_min = in->egamma_min;
        device_in.egamma_max = in->egamma_max;
        device_in.d_n_gas = d_n_gas;
        device_in.d_crp_state = d_crp_state;
        device_in.d_crp_dp = d_crp_dp;
        device_in.d_beta_p = d_beta_p;
        device_in.d_fga_flat = d_fga_flat;
        device_in.d_eps_out = d_eps;
        if (fp_cuda_emit_gamma_device(&device_in, nullptr) != 0) return -1;
    }

    cuda_check(cudaMemcpy(eps_gamma_out, d_eps, eps_size * sizeof(double), cudaMemcpyDeviceToHost), "copy gamma eps");
    cudaFree(d_n_gas);
    cudaFree(d_crp_state);
    cudaFree(d_crp_dp);
    cudaFree(d_beta_p);
    cudaFree(d_fga_flat);
    cudaFree(d_eps);
    if (elapsed_ms != nullptr) {
        *elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_t0).count();
    }
    return 0;
}

extern "C" int prepare_neutrino_emission_cuda_batch(const FpNeutrinoEmissionBatchInput *in,
                                                    const CRspectrum *crp_grid,
                                                    double *eps_nu_out,
                                                    double *elapsed_ms)
{
    double *d_n_gas = nullptr, *d_crp_state = nullptr, *d_crp_dp = nullptr;
    double *d_beta_p = nullptr, *d_fnu_flat = nullptr, *d_eps = nullptr;

    auto malloc_copy = [](double **dst, const double *src, size_t nbyte, const char *what) {
        cuda_check(cudaMalloc(dst, nbyte), what);
        cuda_check(cudaMemcpy(*dst, src, nbyte, cudaMemcpyHostToDevice), what);
    };

    if (elapsed_ms != nullptr) *elapsed_ms = 0.0;
    if (in == nullptr || crp_grid == nullptr || eps_nu_out == nullptr) return -1;
    if (in->ncell <= 0 || in->nbins <= 0 || in->n_gas == nullptr ||
        in->crp_batch == nullptr || in->beta_p == nullptr || in->fnu_flat == nullptr) {
        return -1;
    }

    const auto total_t0 = std::chrono::steady_clock::now();
    const size_t eps_size = (size_t)in->ncell * (size_t)in->nbins;

    malloc_copy(&d_n_gas, in->n_gas, (size_t)in->ncell * sizeof(double), "copy neutrino n_gas");
    malloc_copy(&d_crp_state, in->crp_batch, (size_t)in->ncell * (size_t)np * sizeof(double), "copy neutrino crp_batch");
    malloc_copy(&d_crp_dp, crp_grid->dp, (size_t)np * sizeof(double), "copy neutrino crp_dp");
    malloc_copy(&d_beta_p, in->beta_p, (size_t)np * sizeof(double), "copy neutrino beta_p");
    malloc_copy(&d_fnu_flat, in->fnu_flat, (size_t)in->nbins * (size_t)np * sizeof(double), "copy neutrino fnu_flat");
    cuda_check(cudaMalloc(&d_eps, eps_size * sizeof(double)), "alloc neutrino eps");

    {
        FpCudaNeutrinoDeviceInput device_in;

        device_in.ncell = in->ncell;
        device_in.nbins = in->nbins;
        device_in.enu_min = in->enu_min;
        device_in.enu_max = in->enu_max;
        device_in.d_n_gas = d_n_gas;
        device_in.d_crp_state = d_crp_state;
        device_in.d_crp_dp = d_crp_dp;
        device_in.d_beta_p = d_beta_p;
        device_in.d_fnu_flat = d_fnu_flat;
        device_in.d_eps_out = d_eps;
        if (fp_cuda_emit_neutrino_device(&device_in, nullptr) != 0) return -1;
    }

    cuda_check(cudaMemcpy(eps_nu_out, d_eps, eps_size * sizeof(double), cudaMemcpyDeviceToHost), "copy neutrino eps");
    cudaFree(d_n_gas);
    cudaFree(d_crp_state);
    cudaFree(d_crp_dp);
    cudaFree(d_beta_p);
    cudaFree(d_fnu_flat);
    cudaFree(d_eps);
    if (elapsed_ms != nullptr) {
        *elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_t0).count();
    }
    return 0;
}
