#include <cuda_runtime.h>

#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "fp_cuda_solver.h"
#include "fp_shared_core.h"

namespace {

constexpr int kMaxRow = (np > npe) ? np : npe;
constexpr int kPcrRowLimit = 512;
constexpr int kThomasThreadsPerBlock = 32;   //   best  //
//constexpr int kThomasThreadsPerBlock = 64;
//constexpr int kThomasThreadsPerBlock = 128;
enum {
    kCudaTridiagAuto = 0,
    kCudaTridiagThomas = 1,
    kCudaTridiagPcr = 2
};

static void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(2);
    }
}

static bool is_pow2_size(int nrow)
{
    return nrow > 0 && (nrow & (nrow - 1)) == 0;
}

static bool can_use_pcr(int nrow)
{
    return is_pow2_size(nrow) && nrow <= kPcrRowLimit;
}

static int tridiag_solver_mode(void)
{
    static int initialized = 0;
    static int mode = kCudaTridiagAuto;

    if (!initialized) {
        const char *env = std::getenv("CROMA_CUDA_TRIDIAG");

        if (env == nullptr) {
            env = std::getenv("FP_GPU_CUDA_TRIDIAG");
        }

        if (env != nullptr) {
            if (std::strcmp(env, "thomas") == 0) {
                mode = kCudaTridiagThomas;
            } else if (std::strcmp(env, "pcr") == 0) {
                mode = kCudaTridiagPcr;
            }
        }
        initialized = 1;
    }
    return mode;
}

static int select_use_pcr(int nrow)
{
    const bool pcr_ok = can_use_pcr(nrow);
    const int mode = tridiag_solver_mode();

    if (mode == kCudaTridiagThomas) return 0;
    if (mode == kCudaTridiagPcr) return pcr_ok ? 1 : -1;
    return pcr_ok ? 1 : 0;
}

template <int NROW>
__device__ void solve_cc_thomas_inplace(double dt,
                                        const double *a_batch,
                                        const double *b_batch,
                                        const double *c_batch,
                                        const double *source_batch,
                                        double *state_batch,
                                        size_t off)
{
    double cprime[NROW];

    state_batch[off] = dt * source_batch[off] + state_batch[off];
    cprime[0] = -c_batch[off] / b_batch[off];
    state_batch[off] /= b_batch[off];

    for (int i = 1; i < NROW; ++i) {
        const size_t idx = off + (size_t)i;
        const double denom = b_batch[idx] + a_batch[idx] * cprime[i - 1];
        cprime[i] = (i == NROW - 1) ? 0.0 : -c_batch[idx] / denom;
        state_batch[idx] = (dt * source_batch[idx] + state_batch[idx] +
                            a_batch[idx] * state_batch[idx - 1]) / denom;
    }

    for (int i = NROW - 2; i >= 0; --i) {
        const size_t idx = off + (size_t)i;
        state_batch[idx] -= cprime[i] * state_batch[idx + 1];
    }
}

__global__ void solve_cc_batch_kernel(int nsys,
                                      int nrow,
                                      double dt,
                                      const double *a_batch,
                                      const double *b_batch,
                                      const double *c_batch,
                                      const double *source_batch,
                                      double *state_batch)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    double cprime[kMaxRow];

    if (nrow > kMaxRow) return;

    for (int isys = tid; isys < nsys; isys += stride) {
        // Each linear system is stored contiguously in cell-major layout:
        // batch[isys * nrow + irow].
        const size_t off = (size_t)isys * (size_t)nrow;

        state_batch[off] = dt * source_batch[off] + state_batch[off];
        cprime[0] = -c_batch[off] / b_batch[off];
        state_batch[off] /= b_batch[off];

        for (int i = 1; i < nrow; ++i) {
            const size_t idx = off + (size_t)i;
            const double denom = b_batch[idx] + a_batch[idx] * cprime[i - 1];
            cprime[i] = (i == nrow - 1) ? 0.0 : -c_batch[idx] / denom;
            state_batch[idx] = (dt * source_batch[idx] + state_batch[idx] +
                                a_batch[idx] * state_batch[idx - 1]) / denom;
        }

        for (int i = nrow - 2; i >= 0; --i) {
            const size_t idx = off + (size_t)i;
            state_batch[idx] -= cprime[i] * state_batch[idx + 1];
        }
    }
}

/*  This is the heart of PCR solver */
__global__ void solve_cc_pcr_kernel(int nrow,
                                    double dt,
                                    const double *a_batch,
                                    const double *b_batch,
                                    const double *c_batch,
                                    const double *source_batch,
                                    double *state_batch)
{
    const int isys = blockIdx.x; // contiguous system index
    const int i = threadIdx.x;   // row index within that system
    __shared__ double a_buf[2][kPcrRowLimit];   //  double buffer  //
    __shared__ double b_buf[2][kPcrRowLimit];
    __shared__ double c_buf[2][kPcrRowLimit];
    __shared__ double d_buf[2][kPcrRowLimit];
    int src = 0;
    int dst = 1;

    {
        // read coef : NOTE (conventional) sign of a,c //
        const size_t off = (size_t)isys * (size_t)nrow + (size_t)i;   
        a_buf[src][i] = -a_batch[off];
        b_buf[src][i] = b_batch[off];
        c_buf[src][i] = -c_batch[off];
        d_buf[src][i] = dt * source_batch[off] + state_batch[off];
    }
    __syncthreads();

    // stride loop // 1,2,4,8,....
    for (int stride = 1; stride < nrow; stride <<= 1) {
        double next_a = 0.0;
        double next_b = b_buf[src][i];
        double next_c = 0.0;
        double next_d = d_buf[src][i];

        if (i - stride >= 0) {
            const double alpha = -a_buf[src][i] / b_buf[src][i - stride];
            next_a = alpha * a_buf[src][i - stride];
            next_b += alpha * c_buf[src][i - stride];
            next_d += alpha * d_buf[src][i - stride];
        }
        if (i + stride < nrow) {
            const double beta = -c_buf[src][i] / b_buf[src][i + stride];
            next_c = beta * c_buf[src][i + stride];
            next_b += beta * a_buf[src][i + stride];
            next_d += beta * d_buf[src][i + stride];
        }

        a_buf[dst][i] = next_a;
        b_buf[dst][i] = next_b;
        c_buf[dst][i] = next_c;
        d_buf[dst][i] = next_d;
        __syncthreads();

        {
            const int tmp = src;
            src = dst;
            dst = tmp;
        }
        __syncthreads();
    }

    state_batch[(size_t)isys * (size_t)nrow + (size_t)i] = d_buf[src][i] / b_buf[src][i];
}

template <int NROW, int NSCRATCH>
__device__ void solve_cc_pcr_inplace(double dt,
                                     const double *a_batch,
                                     const double *b_batch,
                                     const double *c_batch,
                                     const double *source_batch,
                                     double *state_batch,
                                     size_t off,
                                     double (&a_buf)[2][NSCRATCH],
                                     double (&b_buf)[2][NSCRATCH],
                                     double (&c_buf)[2][NSCRATCH],
                                     double (&d_buf)[2][NSCRATCH])
{
    const int i = threadIdx.x;
    int src = 0;
    int dst = 1;

    if (i < NROW) {
        a_buf[src][i] = -a_batch[off + (size_t)i];
        b_buf[src][i] = b_batch[off + (size_t)i];
        c_buf[src][i] = -c_batch[off + (size_t)i];
        d_buf[src][i] = dt * source_batch[off + (size_t)i] + state_batch[off + (size_t)i];
    }
    __syncthreads();

    for (int stride = 1; stride < NROW; stride <<= 1) {
        if (i < NROW) {
            double next_a = 0.0;
            double next_b = b_buf[src][i];
            double next_c = 0.0;
            double next_d = d_buf[src][i];

            if (i - stride >= 0) {
                const double alpha = -a_buf[src][i] / b_buf[src][i - stride];
                next_a = alpha * a_buf[src][i - stride];
                next_b += alpha * c_buf[src][i - stride];
                next_d += alpha * d_buf[src][i - stride];
            }
            if (i + stride < NROW) {
                const double beta = -c_buf[src][i] / b_buf[src][i + stride];
                next_c = beta * c_buf[src][i + stride];
                next_b += beta * a_buf[src][i + stride];
                next_d += beta * d_buf[src][i + stride];
            }

            a_buf[dst][i] = next_a;
            b_buf[dst][i] = next_b;
            c_buf[dst][i] = next_c;
            d_buf[dst][i] = next_d;
        }
        __syncthreads();

        {
            const int tmp = src;
            src = dst;
            dst = tmp;
        }
        __syncthreads();
    }

    if (i < NROW) {
        state_batch[off + (size_t)i] = d_buf[src][i] / b_buf[src][i];
    }
    __syncthreads();
}

__global__ void run_fused_transport_kernel(const FpCudaFusedTransportInput in)
{
    const int isys = blockIdx.x;
    const int tid = threadIdx.x;
    const int max_row = (np > npe) ? np : npe;
    const size_t poff = (size_t)isys * (size_t)np;
    const size_t eoff = (size_t)isys * (size_t)npe;
    /* Proton and electron PCR solves run sequentially, so one shared scratch
     * bank is enough for both species. */
    __shared__ double scratch_a[2][kMaxRow];
    __shared__ double scratch_b[2][kMaxRow];
    __shared__ double scratch_c[2][kMaxRow];
    __shared__ double scratch_d[2][kMaxRow];

    if (isys >= in.nsys || tid >= max_row) return;

    for (int istep = 0; istep < in.nstep; istep++) {
        const bool use_on_coeff =
            !in.use_windowed_reacc ||
            (istep >= in.on_start_step && istep < in.on_end_step);
        const double *d_ccp_a_step = use_on_coeff ? in.d_ccp_a : in.d_ccp_a_off;
        const double *d_ccp_b_step = use_on_coeff ? in.d_ccp_b : in.d_ccp_b_off;
        const double *d_ccp_c_step = use_on_coeff ? in.d_ccp_c : in.d_ccp_c_off;
        const double *d_cce_a_step = use_on_coeff ? in.d_cce_a : in.d_cce_a_off;
        const double *d_cce_b_step = use_on_coeff ? in.d_cce_b : in.d_cce_b_off;
        const double *d_cce_c_step = use_on_coeff ? in.d_cce_c : in.d_cce_c_off;

        solve_cc_pcr_inplace<np, kMaxRow>(in.dt,
                                          d_ccp_a_step, d_ccp_b_step, d_ccp_c_step,
                                          in.d_qpi_cell, in.d_crp_state,
                                          poff, scratch_a, scratch_b, scratch_c, scratch_d);
        __syncthreads();

        solve_cc_pcr_inplace<npe, kMaxRow>(in.dt,
                                           d_cce_a_step, d_cce_b_step, d_cce_c_step,
                                           in.d_inje_cell, in.d_cre_state,
                                           eoff, scratch_a, scratch_b, scratch_c, scratch_d);
    }
}

__global__ void run_fused_transport_thomas_kernel(const FpCudaFusedTransportInput in)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int isys = tid; isys < in.nsys; isys += stride) {
        const size_t poff = (size_t)isys * (size_t)np;
        const size_t eoff = (size_t)isys * (size_t)npe;

        for (int istep = 0; istep < in.nstep; istep++) {
            const bool use_on_coeff =
                !in.use_windowed_reacc ||
                (istep >= in.on_start_step && istep < in.on_end_step);
            const double *d_ccp_a_step = use_on_coeff ? in.d_ccp_a : in.d_ccp_a_off;
            const double *d_ccp_b_step = use_on_coeff ? in.d_ccp_b : in.d_ccp_b_off;
            const double *d_ccp_c_step = use_on_coeff ? in.d_ccp_c : in.d_ccp_c_off;
            const double *d_cce_a_step = use_on_coeff ? in.d_cce_a : in.d_cce_a_off;
            const double *d_cce_b_step = use_on_coeff ? in.d_cce_b : in.d_cce_b_off;
            const double *d_cce_c_step = use_on_coeff ? in.d_cce_c : in.d_cce_c_off;

            solve_cc_thomas_inplace<np>(in.dt,
                                        d_ccp_a_step, d_ccp_b_step, d_ccp_c_step,
                                        in.d_qpi_cell, in.d_crp_state,
                                        poff);
            solve_cc_thomas_inplace<npe>(in.dt,
                                         d_cce_a_step, d_cce_b_step, d_cce_c_step,
                                         in.d_inje_cell, in.d_cre_state,
                                         eoff);
        }
    }
}

}  // namespace

extern "C" int fp_cuda_tridiag_uses_pcr(int nrow)
{
    return select_use_pcr(nrow);
}

extern "C" const char *fp_cuda_tridiag_solver_name(int nrow)
{
    return select_use_pcr(nrow) == 1 ? "PCR" : "Thomas";
}

int fp_cuda_solve_device_batch(int nsys,
                               int nrow,
                               double dt,
                               const double *d_a,
                               const double *d_b,
                               const double *d_c,
                               const double *d_source,
                               double *d_state,
                               const char *label,
                               int *used_pcr_out)
{
    const int use_pcr = select_use_pcr(nrow);

    if (used_pcr_out != nullptr) *used_pcr_out = (use_pcr == 1) ? 1 : 0;
    if (nsys <= 0 || nrow <= 0 || nrow > kMaxRow ||
        d_a == nullptr || d_b == nullptr || d_c == nullptr ||
        d_source == nullptr || d_state == nullptr) {
        return -1;
    }
    if (use_pcr < 0) {
        std::fprintf(stderr,
                     "fp_cuda_solve_device_batch: CROMA_CUDA_TRIDIAG=pcr is invalid for nrow=%d\n",
                     nrow);
        return -1;
    }

    if (use_pcr == 1) {
        solve_cc_pcr_kernel<<<nsys, nrow>>>(nrow, dt, d_a, d_b, d_c, d_source, d_state);
    } else {
        const int nblock = (nsys + kThomasThreadsPerBlock - 1) / kThomasThreadsPerBlock;
        solve_cc_batch_kernel<<<nblock, kThomasThreadsPerBlock>>>(
            nsys, nrow, dt, d_a, d_b, d_c, d_source, d_state);
    }
    cuda_check(cudaGetLastError(), (label != nullptr) ? label : "fp_cuda_solve_device_batch");
    return 0;
}

int fp_cuda_run_fused_transport(const FpCudaFusedTransportInput *in,
                                const char *label,
                                int *used_pcr_proton_out,
                                int *used_pcr_electron_out)
{
    const int max_row = (np > npe) ? np : npe;
    const int use_pcr_proton = select_use_pcr(np);
    const int use_pcr_electron = select_use_pcr(npe);
    const bool use_pcr = (use_pcr_proton == 1 && use_pcr_electron == 1);

    if (used_pcr_proton_out != nullptr) *used_pcr_proton_out = use_pcr ? 1 : 0;
    if (used_pcr_electron_out != nullptr) *used_pcr_electron_out = use_pcr ? 1 : 0;

    if (in == nullptr || in->nsys <= 0 || in->nstep <= 0 ||
        in->d_ccp_a == nullptr || in->d_ccp_b == nullptr || in->d_ccp_c == nullptr ||
        in->d_ccp_a_off == nullptr || in->d_ccp_b_off == nullptr || in->d_ccp_c_off == nullptr ||
        in->d_cce_a == nullptr || in->d_cce_b == nullptr || in->d_cce_c == nullptr ||
        in->d_cce_a_off == nullptr || in->d_cce_b_off == nullptr || in->d_cce_c_off == nullptr ||
        in->d_qpi_cell == nullptr || in->d_qepri_cell == nullptr || in->d_n_gas == nullptr ||
        in->d_crp_dp == nullptr || in->d_np_min_qe == nullptr || in->d_fqe_flat == nullptr ||
        in->d_inje_cell == nullptr || in->d_crp_state == nullptr || in->d_cre_state == nullptr) {
        return -1;
    }

    if ((use_pcr_proton < 0 || use_pcr_electron < 0) || max_row > 1024) {
        std::fprintf(stderr,
                     "fp_cuda_run_fused_transport: CROMA_CUDA_TRIDIAG=pcr is invalid for (np,npe)=(%d,%d)\n",
                     np, npe);
        return -1;
    }

    if (use_pcr) {
        run_fused_transport_kernel<<<in->nsys, max_row>>>(*in);
    } else {
        const int nblock = (in->nsys + kThomasThreadsPerBlock - 1) / kThomasThreadsPerBlock;
        run_fused_transport_thomas_kernel<<<nblock, kThomasThreadsPerBlock>>>(*in);
    }
    cuda_check(cudaGetLastError(), (label != nullptr) ? label : "fp_cuda_run_fused_transport");
    return 0;
}

extern "C" int solve_cc_cuda_batch(const FpChangCooperSolveBatchInput *in,
                                   const double *state_init,
                                   double *state_out,
                                   double *elapsed_ms,
                                   int *used_pcr_out)
{
    const size_t batch_size =
        (in != nullptr) ? (size_t)in->nsys * (size_t)in->nrow : 0u;
    const size_t nbyte = batch_size * sizeof(double);
    double *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    double *d_src = nullptr, *d_state = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    float kernel_ms = 0.0f;

    if (elapsed_ms != nullptr) *elapsed_ms = 0.0;
    if (used_pcr_out != nullptr) *used_pcr_out = 0;
    if (in == nullptr || state_init == nullptr || state_out == nullptr) return -1;
    if (in->nsys <= 0 || in->nrow <= 0 || in->nrow > kMaxRow ||
        in->a_batch == nullptr || in->b_batch == nullptr ||
        in->c_batch == nullptr || in->source_batch == nullptr) {
        return -1;
    }

    cuda_check(cudaMalloc(&d_a, nbyte), "solve_cc cudaMalloc(d_a)");
    cuda_check(cudaMalloc(&d_b, nbyte), "solve_cc cudaMalloc(d_b)");
    cuda_check(cudaMalloc(&d_c, nbyte), "solve_cc cudaMalloc(d_c)");
    cuda_check(cudaMalloc(&d_src, nbyte), "solve_cc cudaMalloc(d_src)");
    cuda_check(cudaMalloc(&d_state, nbyte), "solve_cc cudaMalloc(d_state)");

    cuda_check(cudaMemcpy(d_a, in->a_batch, nbyte, cudaMemcpyHostToDevice), "solve_cc copy a");
    cuda_check(cudaMemcpy(d_b, in->b_batch, nbyte, cudaMemcpyHostToDevice), "solve_cc copy b");
    cuda_check(cudaMemcpy(d_c, in->c_batch, nbyte, cudaMemcpyHostToDevice), "solve_cc copy c");
    cuda_check(cudaMemcpy(d_src, in->source_batch, nbyte, cudaMemcpyHostToDevice), "solve_cc copy src");
    cuda_check(cudaMemcpy(d_state, state_init, nbyte, cudaMemcpyHostToDevice), "solve_cc copy state");

    cuda_check(cudaEventCreate(&start), "solve_cc event create start");
    cuda_check(cudaEventCreate(&stop), "solve_cc event create stop");
    cuda_check(cudaEventRecord(start), "solve_cc record start");
    if (fp_cuda_solve_device_batch(in->nsys, in->nrow, in->dt,
                                   d_a, d_b, d_c, d_src, d_state,
                                   "solve_cc kernel launch",
                                   used_pcr_out) != 0) {
        return -1;
    }
    cuda_check(cudaEventRecord(stop), "solve_cc record stop");
    cuda_check(cudaEventSynchronize(stop), "solve_cc sync stop");
    cuda_check(cudaEventElapsedTime(&kernel_ms, start, stop), "solve_cc elapsed");
    cuda_check(cudaMemcpy(state_out, d_state, nbyte, cudaMemcpyDeviceToHost), "solve_cc copy out");

    if (elapsed_ms != nullptr) *elapsed_ms = (double)kernel_ms;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_src);
    cudaFree(d_state);
    return 0;
}
