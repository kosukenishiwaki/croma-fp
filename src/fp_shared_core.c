#include "fp_shared_core.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "CONSTANTS.h"
#include "EMISSION.h"
#include "FP_Coef.h"
#include "HADRONIC.h"
#include "Synchrotron.h"
#include "Chang_Cooper.h"

#define FP_CPU_OMP_MIN_CELL 64
#define FP_CPU_OMP_MIN_SOLVE_SYS 512

static int fp_cpu_omp_max_threads(void)
{
#ifdef _OPENMP
    const int nthr = omp_get_max_threads();
    return (nthr > 0) ? nthr : 1;
#else
    return 1;
#endif
}

static int fp_cpu_env_threshold(const char *name, int fallback)
{
    const char *env;
    char *endptr = 0;
    long value;

    if (name == 0) return fallback;
    env = getenv(name);
    if (env == 0 || env[0] == '\0') return fallback;

    value = strtol(env, &endptr, 10);
    if (endptr == env || (endptr != 0 && *endptr != '\0') || value <= 0L) {
        return fallback;
    }
    if (value > 2147483647L) return 2147483647;
    return (int)value;
}

static int fp_cpu_cell_parallel_threshold(void)
{
    static int initialized = 0;
    static int threshold = FP_CPU_OMP_MIN_CELL;

    if (!initialized) {
        const int base = fp_cpu_env_threshold("FP_CPU_OMP_MIN_CELL", FP_CPU_OMP_MIN_CELL);
        const int nthr = fp_cpu_omp_max_threads();
        threshold = base;
        if (nthr > 1) threshold = base * nthr;
        initialized = 1;
    }
    return threshold;
}

static int fp_cpu_solve_parallel_threshold(void)
{
    static int initialized = 0;
    static int threshold = FP_CPU_OMP_MIN_SOLVE_SYS;

    if (!initialized) {
        const int base = fp_cpu_env_threshold("FP_CPU_OMP_MIN_SOLVE_SYS",
                                              FP_CPU_OMP_MIN_SOLVE_SYS);
        const int nthr = fp_cpu_omp_max_threads();
        threshold = base;
        if (nthr > 1) threshold = base * nthr;
        initialized = 1;
    }
    return threshold;
}

static int fp_cpu_should_parallel_cells(int ncell)
{
#ifdef _OPENMP
    if (omp_in_parallel()) return 0;
#endif
    return ncell >= fp_cpu_cell_parallel_threshold();
}

static int fp_cpu_should_parallel_systems(int nsys)
{
#ifdef _OPENMP
    if (omp_in_parallel()) return 0;
#endif
    return nsys >= fp_cpu_solve_parallel_threshold();
}

static double fp_now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double)tv.tv_sec * 1.0e3 + (double)tv.tv_usec * 1.0e-3;
}

int tracer_fp_secondary_transpose_enabled(void)
{
    static int initialized = 0;
    static int enabled = 0;
    const char *env;

    if (initialized) return enabled;

    env = getenv("CROMA_SECONDARY_TRANSPOSE");
    enabled = 1;
    if (env != 0 &&
        (strcmp(env, "0") == 0 ||
         strcmp(env, "false") == 0 ||
         strcmp(env, "FALSE") == 0 ||
         strcmp(env, "no") == 0 ||
         strcmp(env, "NO") == 0 ||
         strcmp(env, "off") == 0 ||
         strcmp(env, "OFF") == 0)) {
        enabled = 0;
    }
    initialized = 1;
    return enabled;
}

static void fp_store_np_batch(double *dst,
                                   const double *src,
                                   int ncell,
                                   int icell)
{
    int jp;

    if (dst == 0 || src == 0) return;
    for (jp = 0; jp < np; jp++)
        dst[(size_t)jp * (size_t)ncell + (size_t)icell] = src[jp];
}

static void fp_store_npe_batch(double *dst,
                                    const double *src,
                                    int ncell,
                                    int icell)
{
    int je;

    if (dst == 0 || src == 0) return;
    for (je = 0; je < npe; je++)
        dst[(size_t)je * (size_t)ncell + (size_t)icell] = src[je];
}

static void fp_store_np_cell_major(double *dst,
                                        const double *src,
                                        int icell)
{
    int jp;
    size_t off;

    if (dst == 0 || src == 0) return;
    off = (size_t)icell * (size_t)np;
    for (jp = 0; jp < np; jp++) dst[off + (size_t)jp] = src[jp];
}

static void fp_store_npe_cell_major(double *dst,
                                         const double *src,
                                         int icell)
{
    int je;
    size_t off;

    if (dst == 0 || src == 0) return;
    off = (size_t)icell * (size_t)npe;
    for (je = 0; je < npe; je++) dst[off + (size_t)je] = src[je];
}

static void fp_transpose_freq_to_cell_major(double *dst_cell_major,
                                                 const double *src_freq_major,
                                                 int ncell,
                                                 int nfreq)
{
    int icell, ifreq;

    if (dst_cell_major == 0 || src_freq_major == 0) return;
    for (icell = 0; icell < ncell; icell++) {
        size_t cell_off = (size_t)icell * (size_t)nfreq;
        for (ifreq = 0; ifreq < nfreq; ifreq++) {
            dst_cell_major[cell_off + (size_t)ifreq] =
                src_freq_major[(size_t)ifreq * (size_t)ncell + (size_t)icell];
        }
    }
}

static void fp_store_cell_major(double *dst,
                                     const double *src,
                                     int ncell,
                                     int nbin,
                                     int icell)
{
    int ibin;
    size_t off;

    if (dst == 0 || src == 0) return;
    off = (size_t)icell * (size_t)nbin;
    for (ibin = 0; ibin < nbin; ibin++) dst[off + (size_t)ibin] = src[ibin];
}

static void fp_gather_momentum_major_to_cell_row(double *dst_row,
                                                      const double *src_momentum_major,
                                                      int ncell,
                                                      int nbin,
                                                      int icell)
{
    int ibin;

    if (dst_row == 0 || src_momentum_major == 0) return;
    for (ibin = 0; ibin < nbin; ibin++) {
        dst_row[ibin] =
            src_momentum_major[(size_t)ibin * (size_t)ncell + (size_t)icell];
    }
}

static int fp_is_power_of_two(int n)
{
    return (n > 0) && ((n & (n - 1)) == 0);
}

static int fp_thomas_solve_ws(int nrow,
                              const double *a,
                              const double *b,
                              const double *c,
                              double *rhs_solution,
                              double *cprime)
{
    int i;

    if (nrow <= 0 || a == 0 || b == 0 || c == 0 || rhs_solution == 0 || cprime == 0) return -1;
    if (fabs(b[0]) <= 1.0e-300) return -2;

    /* Chang-Cooper stores positive coupling magnitudes; the actual matrix
     * rows are b_i x_i - a_i x_{i-1} - c_i x_{i+1} = rhs_i. */
    cprime[0] = -c[0] / b[0];
    rhs_solution[0] /= b[0];

    for (i = 1; i < nrow; i++) {
        double denom = b[i] + a[i] * cprime[i - 1];
        if (fabs(denom) <= 1.0e-300) {
            return -2;
        }
        cprime[i] = (i == nrow - 1) ? 0.0 : -c[i] / denom;
        rhs_solution[i] = (rhs_solution[i] + a[i] * rhs_solution[i - 1]) / denom;
    }

    for (i = nrow - 2; i >= 0; i--) {
        rhs_solution[i] -= cprime[i] * rhs_solution[i + 1];
    }

    return 0;
}

static int fp_cpu_tridiag_solver_mode(void)
{
    static int initialized = 0;
    static int mode = 0; /* 0: auto, 1: thomas, 2: pcr */

    if (!initialized) {
        const char *env = getenv("TRACER_FP_CPU_TRIDIAG");
        if (env != 0 && strcmp(env, "thomas") == 0) {
            mode = 1;
        } else if (env != 0 && strcmp(env, "pcr") == 0) {
            mode = 2;
        }
        initialized = 1;
    }

    return mode;
}

const char *cpu_tridiag_solver_name(void)
{
    const int mode = fp_cpu_tridiag_solver_mode();
    if (mode == 2) return "PCR";
    return "Thomas";
}

int fp_thomas_solve_cpu(int nrow,
                        const double *a,
                        const double *b,
                        const double *c,
                        double *rhs_solution)
{
    double *cprime;
    int ierr;

    if (nrow <= 0 || a == 0 || b == 0 || c == 0 || rhs_solution == 0) return -1;

    cprime = (double *)malloc((size_t)nrow * sizeof(double));
    if (cprime == 0) return -3;

    ierr = fp_thomas_solve_ws(nrow, a, b, c, rhs_solution, cprime);
    free(cprime);
    return ierr;
}

int solve_cc_1d_cpu(int nrow,
                                      double dt,
                                      const double *a,
                                      const double *b,
                                      const double *c,
                                      const double *source,
                                      double *state)
{
    double *rhs;
    int i;
    int ierr;

    if (nrow <= 0 || a == 0 || b == 0 || c == 0 || source == 0 || state == 0) return -1;

    rhs = (double *)malloc((size_t)nrow * sizeof(double));
    if (rhs == 0) return -3;

    for (i = 0; i < nrow; i++) rhs[i] = dt * source[i] + state[i];

    ierr = fp_thomas_solve_cpu(nrow, a, b, c, rhs);
    if (ierr == 0) {
        for (i = 0; i < nrow; i++) state[i] = rhs[i];
    }

    free(rhs);
    return ierr;
}

int fp_thomas_solve_batch_cpu(const FpTridiagBatchInput *in,
                              double *solution_batch)
{
    int isys;
    int ierr_any = 0;

    if (in == 0 || solution_batch == 0) return -1;
    if (in->nsys <= 0 || in->nrow <= 0 || in->a_batch == 0 || in->b_batch == 0 ||
        in->c_batch == 0 || in->rhs_batch == 0) {
        return -1;
    }

    #pragma omp parallel if(fp_cpu_should_parallel_systems(in->nsys))
    {
        double *cprime_ws = (double *)malloc((size_t)in->nrow * sizeof(double));

        if (cprime_ws == 0) {
            #pragma omp critical
            {
                if (ierr_any == 0) ierr_any = -3;
            }
        } else {
            #pragma omp for schedule(static)
            for (isys = 0; isys < in->nsys; isys++) {
                const size_t off = (size_t)isys * (size_t)in->nrow;
                int i;
                int ierr;

                for (i = 0; i < in->nrow; i++) {
                    solution_batch[off + (size_t)i] = in->rhs_batch[off + (size_t)i];
                }
                ierr = fp_thomas_solve_ws(in->nrow,
                                          in->a_batch + off,
                                          in->b_batch + off,
                                          in->c_batch + off,
                                          solution_batch + off,
                                          cprime_ws);
                if (ierr != 0) {
                    #pragma omp critical
                    {
                        if (ierr_any == 0) ierr_any = ierr;
                    }
                }
            }
            free(cprime_ws);
        }
    }

    if (ierr_any != 0) return ierr_any;
    return 0;
}

static int fp_thomas_solve_batch_inplace_cpu(
    const FpTridiagBatchInput *in,
    double *rhs_solution_batch)
{
    int isys;
    int ierr_any = 0;

    if (in == 0 || rhs_solution_batch == 0) return -1;
    if (in->nsys <= 0 || in->nrow <= 0 || in->a_batch == 0 || in->b_batch == 0 ||
        in->c_batch == 0) {
        return -1;
    }

    #pragma omp parallel if(fp_cpu_should_parallel_systems(in->nsys))
    {
        double *cprime_ws = (double *)malloc((size_t)in->nrow * sizeof(double));

        if (cprime_ws == 0) {
            #pragma omp critical
            {
                if (ierr_any == 0) ierr_any = -3;
            }
        } else {
            #pragma omp for schedule(static)
            for (isys = 0; isys < in->nsys; isys++) {
                const size_t off = (size_t)isys * (size_t)in->nrow;
                const int ierr = fp_thomas_solve_ws(in->nrow,
                                                    in->a_batch + off,
                                                    in->b_batch + off,
                                                    in->c_batch + off,
                                                    rhs_solution_batch + off,
                                                    cprime_ws);
                if (ierr != 0) {
                    #pragma omp critical
                    {
                        if (ierr_any == 0) ierr_any = ierr;
                    }
                }
            }
            free(cprime_ws);
        }
    }

    if (ierr_any != 0) return ierr_any;
    return 0;
}

int solve_tridiagonal_batch_pcr_cpu(const FpTridiagBatchInput *in,
                                            double *solution_batch)
{
    double *a_cur, *b_cur, *c_cur, *rhs_cur;
    double *a_next, *b_next, *c_next, *rhs_next;
    size_t nall;
    int stride;
    int isys;

    if (in == 0 || solution_batch == 0) return -1;
    if (in->nsys <= 0 || in->nrow <= 0 || in->a_batch == 0 || in->b_batch == 0 ||
        in->c_batch == 0 || in->rhs_batch == 0) {
        return -1;
    }
    if (!fp_is_power_of_two(in->nrow)) return -4;

    nall = (size_t)in->nsys * (size_t)in->nrow;
    a_cur = (double *)malloc(nall * sizeof(double));
    b_cur = (double *)malloc(nall * sizeof(double));
    c_cur = (double *)malloc(nall * sizeof(double));
    rhs_cur = (double *)malloc(nall * sizeof(double));
    a_next = (double *)malloc(nall * sizeof(double));
    b_next = (double *)malloc(nall * sizeof(double));
    c_next = (double *)malloc(nall * sizeof(double));
    rhs_next = (double *)malloc(nall * sizeof(double));

    if (a_cur == 0 || b_cur == 0 || c_cur == 0 || rhs_cur == 0 ||
        a_next == 0 || b_next == 0 || c_next == 0 || rhs_next == 0) {
        free(a_cur); free(b_cur); free(c_cur); free(rhs_cur);
        free(a_next); free(b_next); free(c_next); free(rhs_next);
        return -3;
    }

    for (isys = 0; isys < in->nsys; isys++) {
        const size_t off = (size_t)isys * (size_t)in->nrow;
        int i;

        for (i = 0; i < in->nrow; i++) {
            a_cur[off + (size_t)i] = -in->a_batch[off + (size_t)i];
            b_cur[off + (size_t)i] = in->b_batch[off + (size_t)i];
            c_cur[off + (size_t)i] = -in->c_batch[off + (size_t)i];
            rhs_cur[off + (size_t)i] = in->rhs_batch[off + (size_t)i];
        }
    }

    for (stride = 1; stride < in->nrow; stride <<= 1) {
        for (isys = 0; isys < in->nsys; isys++) {
            const size_t off = (size_t)isys * (size_t)in->nrow;
            int i;

            for (i = 0; i < in->nrow; i++) {
                const int left = i - stride;
                const int right = i + stride;
                double left_scale = 0.0, right_scale = 0.0;
                double next_a = 0.0, next_b = b_cur[off + (size_t)i];
                double next_c = 0.0, next_rhs = rhs_cur[off + (size_t)i];

                if (left >= 0) {
                    const double denom = b_cur[off + (size_t)left];
                    if (fabs(denom) <= 1.0e-300) {
                        free(a_cur); free(b_cur); free(c_cur); free(rhs_cur);
                        free(a_next); free(b_next); free(c_next); free(rhs_next);
                        return -2;
                    }
                    left_scale = a_cur[off + (size_t)i] / denom;
                    next_a = -a_cur[off + (size_t)left] * left_scale;
                    next_b -= c_cur[off + (size_t)left] * left_scale;
                    next_rhs -= rhs_cur[off + (size_t)left] * left_scale;
                }

                if (right < in->nrow) {
                    const double denom = b_cur[off + (size_t)right];
                    if (fabs(denom) <= 1.0e-300) {
                        free(a_cur); free(b_cur); free(c_cur); free(rhs_cur);
                        free(a_next); free(b_next); free(c_next); free(rhs_next);
                        return -2;
                    }
                    right_scale = c_cur[off + (size_t)i] / denom;
                    next_c = -c_cur[off + (size_t)right] * right_scale;
                    next_b -= a_cur[off + (size_t)right] * right_scale;
                    next_rhs -= rhs_cur[off + (size_t)right] * right_scale;
                }

                a_next[off + (size_t)i] = next_a;
                b_next[off + (size_t)i] = next_b;
                c_next[off + (size_t)i] = next_c;
                rhs_next[off + (size_t)i] = next_rhs;
            }
        }

        {
            double *tmp;
            tmp = a_cur; a_cur = a_next; a_next = tmp;
            tmp = b_cur; b_cur = b_next; b_next = tmp;
            tmp = c_cur; c_cur = c_next; c_next = tmp;
            tmp = rhs_cur; rhs_cur = rhs_next; rhs_next = tmp;
        }
    }

    for (isys = 0; isys < in->nsys; isys++) {
        const size_t off = (size_t)isys * (size_t)in->nrow;
        int i;

        for (i = 0; i < in->nrow; i++) {
            const double denom = b_cur[off + (size_t)i];
            if (fabs(denom) <= 1.0e-300) {
                free(a_cur); free(b_cur); free(c_cur); free(rhs_cur);
                free(a_next); free(b_next); free(c_next); free(rhs_next);
                return -2;
            }
            solution_batch[off + (size_t)i] = rhs_cur[off + (size_t)i] / denom;
        }
    }

    free(a_cur); free(b_cur); free(c_cur); free(rhs_cur);
    free(a_next); free(b_next); free(c_next); free(rhs_next);
    return 0;
}

int solve_tridiagonal_batch_cpu(const FpTridiagBatchInput *in,
                                        double *solution_batch)
{
    const int solver_mode = fp_cpu_tridiag_solver_mode();

    if (in == 0) return -1;
    if (solver_mode == 1) {
        return fp_thomas_solve_batch_cpu(in, solution_batch);
    }
    if (solver_mode == 2) {
        return solve_tridiagonal_batch_pcr_cpu(in, solution_batch);
    }
    return fp_thomas_solve_batch_cpu(in, solution_batch);
}

int solve_cc_batch_cpu_timed(
    const FpChangCooperSolveBatchInput *in,
    double *state_batch,
    FpChangCooperSolveBatchCpuTimes *times)
{
    FpTridiagBatchInput tri_in;
    int isys, i, ierr;
    const int solver_mode = fp_cpu_tridiag_solver_mode();
    double t0;
    double t1;

    if (in == 0 || state_batch == 0) return -1;
    if (in->nsys <= 0 || in->nrow <= 0 || in->a_batch == 0 || in->b_batch == 0 ||
        in->c_batch == 0 || in->source_batch == 0) {
        return -1;
    }

    if (times != 0) {
        times->alloc_ms = 0.0;
        times->rhs_ms = 0.0;
        times->tridiag_ms = 0.0;
    }

    t0 = fp_now_ms();
    #pragma omp parallel for schedule(static) private(i) if(fp_cpu_should_parallel_systems(in->nsys))
    for (isys = 0; isys < in->nsys; isys++) {
        const size_t off = (size_t)isys * (size_t)in->nrow;
        for (i = 0; i < in->nrow; i++) {
            state_batch[off + (size_t)i] =
                in->dt * in->source_batch[off + (size_t)i] + state_batch[off + (size_t)i];
        }
    }
    t1 = fp_now_ms();
    if (times != 0) times->rhs_ms += t1 - t0;

    tri_in.nsys = in->nsys;
    tri_in.nrow = in->nrow;
    tri_in.a_batch = in->a_batch;
    tri_in.b_batch = in->b_batch;
    tri_in.c_batch = in->c_batch;
    tri_in.rhs_batch = state_batch;

    t0 = fp_now_ms();
    if (solver_mode == 2) {
        ierr = solve_tridiagonal_batch_pcr_cpu(&tri_in, state_batch);
    } else {
        ierr = fp_thomas_solve_batch_inplace_cpu(&tri_in, state_batch);
    }
    t1 = fp_now_ms();
    if (times != 0) times->tridiag_ms += t1 - t0;
    return ierr;
}

int solve_cc_batch_cpu(
    const FpChangCooperSolveBatchInput *in,
    double *state_batch)
{
    return solve_cc_batch_cpu_timed(in, state_batch, 0);
}

int solve_local_fp_batch_cpu(const FpLocalFpSolveBatchInput *in,
                                     double *crp_state_batch,
                                     double *cre_state_batch)
{
    FpChangCooperSolveBatchInput cc_in;
    int ierr;

    if (in == 0 || crp_state_batch == 0 || cre_state_batch == 0) return -1;
    if (in->ncell <= 0 || in->ccp_a_batch == 0 || in->ccp_b_batch == 0 ||
        in->ccp_c_batch == 0 || in->qpi_batch == 0 ||
        in->cce_a_batch == 0 || in->cce_b_batch == 0 ||
        in->cce_c_batch == 0 || in->inje_batch == 0) {
        return -1;
    }

    cc_in.nsys = in->ncell;
    cc_in.nrow = np;
    cc_in.dt = in->dt;
    cc_in.a_batch = in->ccp_a_batch;
    cc_in.b_batch = in->ccp_b_batch;
    cc_in.c_batch = in->ccp_c_batch;
    cc_in.source_batch = in->qpi_batch;
    ierr = solve_cc_batch_cpu(&cc_in, crp_state_batch);
    if (ierr != 0) {
        return ierr;
    }

    cc_in.nrow = npe;
    cc_in.a_batch = in->cce_a_batch;
    cc_in.b_batch = in->cce_b_batch;
    cc_in.c_batch = in->cce_c_batch;
    cc_in.source_batch = in->inje_batch;
    ierr = solve_cc_batch_cpu(&cc_in, cre_state_batch);
    return ierr;
}

static void fp_build_cc_coeffs_from_terms(int nm,
                                               double dt,
                                               const double *x,
                                               const double *dx,
                                               const double *cc_a,
                                               const double *cc_b,
                                               const double *cc_c,
                                               const double *cc_t,
                                               double xm1,
                                               double xp1,
                                               double bm1,
                                               double bp1,
                                               double cm1,
                                               double cp1,
                                               double *out_a,
                                               double *out_b,
                                               double *out_c)
{
    int m;
    double delta[nm + 1], w[nm + 1];
    double Wp[nm + 1], Wm[nm + 1];
    double Wpm1 = 0.0, Wmm1 = 0.0;

    for (m = 0; m < nm; m++) {
        cc_prepare_face_weight(cc_b[m], cc_b[m + 1], cc_c[m], cc_c[m + 1],
                               x[m], x[m + 1], &delta[m], &Wp[m], &Wm[m], &w[m]);
    }
    if (fabs(cc_c[0]) > 1.0e-200) {
        cc_prepare_face_weight(bm1, cc_b[0], cm1, cc_c[0], xm1, x[0],
                               &delta[0], &Wpm1, &Wmm1, 0);
    }
    cc_prepare_face_weight(cc_b[nm], bp1, cc_c[nm], cp1, x[nm], xp1,
                           &delta[nm], &Wp[nm], &Wm[nm], &w[nm]);

    for (m = 1; m < nm; m++) {
        if (fabs(cc_c[m]) <= 1.0e-200) delta[m - 1] = 0.0;
        cc_build_interior_coeffs(m, dt, x, dx, cc_a, cc_b, cc_c, cc_t,
                                 delta, Wp, Wm, out_a, out_b, out_c);
    }
    cc_build_left_boundary(nm, dt, x, dx, cc_a, cc_b, cc_c, cc_t,
                           delta, Wp, Wm, xm1, bm1, out_a, out_b, out_c,
                           Wpm1, Wmm1);
    cc_build_right_boundary(nm, dt, x, dx, cc_a, cc_b, cc_c, cc_t,
                            delta, Wp, Wm, bp1, out_a, out_b, out_c, w[nm]);
}

int find_pp_threshold_index(const CRspectrum *crp)
{
    if (crp == 0) return 0;
    return momentumdiff_crp_j_pp_cache(crp);
}

static void prepare_crp_losses_1d_with_threshold(double n_gas,
                                                 double kbt,
                                                 double divv,
                                                 const CRspectrum *crp,
                                                 int j_pp,
                                                 FPloss *crp_loss);

void prepare_crp_losses_1d(double n_gas,
                                   double kbt,
                                   double divv,
                                   const CRspectrum *crp,
                                   FPloss *crp_loss)
{
    prepare_crp_losses_1d_with_threshold(n_gas, kbt, divv, crp,
                                         find_pp_threshold_index(crp), crp_loss);
}

static void prepare_crp_losses_1d_with_threshold(double n_gas,
                                                 double kbt,
                                                 double divv,
                                                 const CRspectrum *crp,
                                                 int j_pp,
                                                 FPloss *crp_loss)
{
    const double kappa_pp = 0.5;
    double tloss_prefactor;
    double adiabatic_prefactor;
    const double *sigma_pp;
    const double *sigmoid_pp;
    int j;

    if (crp == 0 || crp_loss == 0 || crp_loss->radp == 0 ||
        crp_loss->tloss == 0 || crp_loss->invtloss == 0) {
        return;
    }

    sigma_pp = momentumdiff_crp_sigma_pp_cache(crp);
    sigmoid_pp = momentumdiff_crp_sigmoid_pp_cache(crp);
    j_pp = momentumdiff_crp_j_pp_cache(crp);
    tloss_prefactor = 1.0 / (kappa_pp * c * n_gas) / Gyr;
    adiabatic_prefactor = 1.0 / 3.0 * divv;

    for (j = 0; j < np; j++) {
        double sig_pp;
        const double p = crp->p[j];

        crp_loss->radp[j] = b_Coulomb_p(n_gas, p, kbt) + adiabatic_prefactor * p;

        if (j > j_pp) {
            sig_pp = sigma_pp[j];
        } else {
            sig_pp = sigma_pp[j_pp];
        }
        sig_pp *= sigmoid_pp[j];

        crp_loss->tloss[j] = tloss_prefactor / sig_pp;
        crp_loss->invtloss[j] = 1.0 / crp_loss->tloss[j];
    }

    /*
     * Keep the adiabatic term out of the Chang-Cooper ghost-edge loss terms.
     * The interior bins still include adiabatic momentum drift, but adding it
     * at the extrapolated edges can drive artificial pile-up at the boundary.
     */
    crp_loss->radpm1 = b_Coulomb_p(n_gas, crp->pm1, kbt);
    crp_loss->radpp1 = b_Coulomb_p(n_gas, crp->pp1, kbt);
}

void prepare_cre_losses_1d(double n_gas,
                                   double b_field,
                                   double divv,
                                   const double *rad_ic,
                                   double rad_ic_m1,
                                   double rad_ic_p1,
                                   const CRspectrum *cre,
                                   FPloss *cre_loss)
{
    int j;
    const double *cre_p2;
    double cre_pm1_p2;
    double cre_pp1_p2;
    double b_micro;
    double b_ratio;
    double synch_coeff;
    double coulomb_coeff;
    double adiabatic_prefactor;
    const double electron_tloss = 1.0e+100;
    const double electron_invtloss = 1.0e-100;

    if (cre == 0 || cre_loss == 0 || rad_ic == 0 || cre_loss->radp == 0 ||
        cre_loss->tloss == 0 || cre_loss->invtloss == 0) {
        return;
    }

    momentumdiff_prepare_cre_grid_cache(cre);
    cre_p2 = momentumdiff_cre_p2_cache(cre);
    cre_pm1_p2 = momentumdiff_cre_pm1_p2_cache(cre);
    cre_pp1_p2 = momentumdiff_cre_pp1_p2_cache(cre);
    b_micro = b_field * 1.0e+6;
    b_ratio = b_micro / 3.2;
    synch_coeff = 4.8e-4 * b_ratio * b_ratio * 9.11e-28 * 3.0e+10 * 3.1536e+16;
    coulomb_coeff = 3.05e-29 * n_gas * c * Gyr / (me * GeV);
    adiabatic_prefactor = 1.0 / 3.0 * divv;

    for (j = 0; j < npe; j++) {
        const double pe = cre->p[j];
        const double pe2 = cre_p2[j];
        const double beta2 = pe2 / (1.0 + pe2);
        const double synch = synch_coeff * pe2;
        const double coulomb = coulomb_coeff * (1.0 + log(pe / n_gas) / 74.8) / beta2;
        cre_loss->radp[j] = synch + rad_ic[j] + coulomb + adiabatic_prefactor * pe;
        cre_loss->tloss[j] = electron_tloss;
        cre_loss->invtloss[j] = electron_invtloss;
    }

    /*
     * Preserve the current CR_Coef_1D behavior: edge terms are finally stored
     * without the adiabatic contribution because the legacy code overwrites the
     * first assignment.
     */
    {
        const double beta2_m1 = cre_pm1_p2 / (1.0 + cre_pm1_p2);
        const double beta2_p1 = cre_pp1_p2 / (1.0 + cre_pp1_p2);
        const double coulomb_m1 =
            coulomb_coeff * (1.0 + log(cre->pm1 / n_gas) / 74.8) / beta2_m1;
        const double coulomb_p1 =
            coulomb_coeff * (1.0 + log(cre->pp1 / n_gas) / 74.8) / beta2_p1;
        cre_loss->radpm1 = synch_coeff * cre_pm1_p2 + rad_ic_m1 + coulomb_m1;
        cre_loss->radpp1 = synch_coeff * cre_pp1_p2 + rad_ic_p1 + coulomb_p1;
    }
}

int prepare_ic_cooling_row(double z,
                                   const CRspectrum *cre_grid,
                                   double *rad_ic_row,
                                   double *rad_ic_m1_out,
                                   double *rad_ic_p1_out)
{
    CRspectrum cre_local;
    double z_local[1];
    double *rad_ic_planes[1];

    if (cre_grid == 0 || rad_ic_row == 0 || rad_ic_m1_out == 0 || rad_ic_p1_out == 0) {
        return -1;
    }

    cre_local = *cre_grid;
    z_local[0] = z;
    rad_ic_planes[0] = rad_ic_row;

    rad_IC_cool(1, z_local, rad_ic_planes, rad_ic_m1_out, rad_ic_p1_out, &cre_local);
    return 0;
}

int prepare_background_cell(const FpBackgroundCellInput *in,
                                    FpBackgroundCellOutput *out)
{
    double temp;
    double curl_v;
    double l_turb_cm;
    double dv_imc;
    double cs;
    double b_sim;
    double b_dyn;
    double b_eff;
    double n_gas;
    double beta_pl;

    if (in == 0 || out == 0) return -1;
    if (in->density_gcc <= 0.0 || in->target_l_turb_cm <= 0.0) return -1;

    temp = in->temp_K;
    if (in->apply_temp_floor && temp < 1.0e4) temp = 1.0e4;

    curl_v = in->curl_v_s;
    if (in->allow_curl_interp && curl_v == 0.0 &&
        in->z_next != in->z_prev) {
        curl_v = (in->curl_v_next_s - in->curl_v_prev_s) /
                 (in->z_next - in->z_prev) *
                 (in->z_curr - in->z_prev) + in->curl_v_prev_s;
    }

    l_turb_cm = (in->l_turb_cm > 0.0) ? in->l_turb_cm : in->target_l_turb_cm;
    dv_imc = curl_v * l_turb_cm;
    cs = c_sound(temp);
    dv_imc *= pow(in->target_l_turb_cm / l_turb_cm, 0.3333);
    l_turb_cm = in->target_l_turb_cm;
    dv_imc = dv_limit(dv_imc, cs);

    b_sim = sqrt(in->bx_G * in->bx_G +
                 in->by_G * in->by_G +
                 in->bz_G * in->bz_G);
    b_dyn = B_dynamo(in->density_gcc, dv_imc);
    if (bfield_mode == BFIELD_MODE_SIM) {
        b_eff = b_sim;
    } else if (bfield_mode == BFIELD_MODE_DYN) {
        b_eff = b_dyn;
    } else {
        b_eff = fmax(b_dyn, b_sim);
    }
    n_gas = in->density_gcc / (1.6e-24 * mu_mol);
    if (b_eff > 0.0) {
        beta_pl = n_gas * kB * temp * eV / (b_eff * b_eff / (8.0 * M_PI));
    } else {
        beta_pl = 1.0e10;
    }

    out->n_gas = n_gas;
    out->kbt_GeV = kB * temp * 1.0e-9;
    out->b_sim_G = b_sim;
    out->b_dyn_G = b_dyn;
    out->b_eff_G = b_eff;
    out->divv_gyr = in->divv_gyr;
    out->l_turb_mpc = l_turb_cm / Mpc;
    out->dv_imc_cms = dv_imc;
    out->cs_cms = cs;
    out->beta_pl = beta_pl;
    out->curl_v_used_s = curl_v;
    out->temp_used_K = temp;
    return 0;
}

int resolve_momentumdiff_model(int requested_model)
{
    if (requested_model != FP_MOMENTUMDIFF_MODEL_AUTO) return requested_model;

    if (dpp_mode_spec[0] != '\0') {
        if (strcmp(dpp_mode_spec, "asa") == 0) {
            return FP_MOMENTUMDIFF_MODEL_ASA;
        }
        if (strcmp(dpp_mode_spec, "ttd") == 0) {
            return FP_MOMENTUMDIFF_MODEL_TTD;
        }
        if (strcmp(dpp_mode_spec, "direct_tacc") == 0 ||
            strcmp(dpp_mode_spec, "tacc") == 0 ||
            strcmp(dpp_mode_spec, "direct") == 0) {
            return FP_MOMENTUMDIFF_MODEL_DIRECT_TACC;
        }
        if (strcmp(dpp_mode_spec, "off") == 0) {
            return FP_MOMENTUMDIFF_MODEL_OFF;
        }
    }

    return FP_MOMENTUMDIFF_MODEL_OFF;
}

const char *momentumdiff_model_name(int model)
{
    switch (resolve_momentumdiff_model(model)) {
    case FP_MOMENTUMDIFF_MODEL_ASA:
        return "ASA";
    case FP_MOMENTUMDIFF_MODEL_TTD:
        return "TTD";
    case FP_MOMENTUMDIFF_MODEL_DIRECT_TACC:
        return "direct_tacc";
    case FP_MOMENTUMDIFF_MODEL_OFF:
        return "off";
    default:
        return "unknown";
    }
}

static double fp_cr_energy_erg_cell(const CRspectrum *crp_grid,
                                    const CRspectrum *cre_grid,
                                    const double *crp_state,
                                    const double *cre_state)
{
    int j;
    double ecr = 0.0;

    if (crp_grid != 0 && crp_state != 0 && np > 4) {
        for (j = 2; j < np - 2; j++) {
            const double ekin_gev = crp_grid->E[j] - mp;
            if (ekin_gev > 0.0 && isfinite(crp_state[j])) {
                ecr += ekin_gev * GeV * crp_state[j] * crp_grid->dp[j];
            }
        }
    }
    if (cre_grid != 0 && cre_state != 0 && npe > 4) {
        for (j = 2; j < npe - 2; j++) {
            const double ekin_gev = cre_grid->E[j] - me;
            if (ekin_gev > 0.0 && isfinite(cre_state[j])) {
                ecr += ekin_gev * GeV * cre_state[j] * cre_grid->dp[j];
            }
        }
    }
    return ecr;
}

static double fp_momentumdiff_tacc_gyr(int model,
                                       double l_turb_mpc,
                                       double dv_imc,
                                       double cs,
                                       double beta_pl)
{
    const int active_model = resolve_momentumdiff_model(model);
    double mach;

    if (!(dv_imc > 0.0) || !(l_turb_mpc > 0.0) || !(cs > 0.0)) {
        return 0.0;
    }
    mach = dv_imc / cs;
    if (mach < 0.0) mach = 0.0;
    if (mach_limit > 0.0 && mach > mach_limit) mach = mach_limit;
    if (mach <= 0.0) return 0.0;
    if (active_model == FP_MOMENTUMDIFF_MODEL_ASA) {
        if (!(beta_pl > 0.0)) return 0.0;
        return accelerationtime_ASA(l_turb_mpc * Mpc, cs, mach, beta_pl);
    }
    if (active_model == FP_MOMENTUMDIFF_MODEL_TTD) {
        return accelerationtime_TTD(l_turb_mpc * Mpc, cs, mach, dv_imc);
    }
    if (active_model == FP_MOMENTUMDIFF_MODEL_DIRECT_TACC) {
        return (t_acc_direct_gyr > 0.0 && isfinite(t_acc_direct_gyr))
                   ? t_acc_direct_gyr
                   : 0.0;
    }
    return 0.0;
}

static void fp_apply_eta_dpp_cap_cell(int model,
                                      double l_turb_mpc,
                                      double dv_imc,
                                      double cs,
                                      double beta_pl,
                                      double tracer_mass_msun,
                                      const CRspectrum *crp_grid,
                                      const CRspectrum *cre_grid,
                                      const double *crp_state,
                                      const double *cre_state,
                                      double *dpp,
                                      double *dppm1,
                                      double *dppp1,
                                      double *dppe,
                                      double *dppem1,
                                      double *dppep1)
{
    int j;
    double t_acc_gyr;
    double ecr_erg;
    double q_turb_total;
    double eta;
    double scale;

    if (!(eta_dpp_cap > 0.0) || !isfinite(eta_dpp_cap)) return;
    if (!(tracer_mass_msun > 0.0) || !(l_turb_mpc > 0.0) || !(dv_imc > 0.0)) return;
    if (crp_state == 0 && cre_state == 0) return;

    t_acc_gyr = fp_momentumdiff_tacc_gyr(model, l_turb_mpc, dv_imc, cs, beta_pl);
    if (!(t_acc_gyr > 0.0) || !isfinite(t_acc_gyr)) return;

    ecr_erg = fp_cr_energy_erg_cell(crp_grid, cre_grid, crp_state, cre_state);
    q_turb_total = 0.5 * tracer_mass_msun * M_sun * dv_imc * dv_imc * dv_imc /
                   (l_turb_mpc * Mpc);
    if (!(ecr_erg > 0.0) || !(q_turb_total > 0.0) || !isfinite(ecr_erg) ||
        !isfinite(q_turb_total)) {
        return;
    }

    eta = ecr_erg / (q_turb_total * t_acc_gyr * Gyr);
    if (!(eta > eta_dpp_cap) || !isfinite(eta)) return;

    scale = eta_dpp_cap / eta;
    for (j = 0; j < np; j++) dpp[j] *= scale;
    for (j = 0; j < npe; j++) dppe[j] *= scale;
    if (dppm1) *dppm1 *= scale;
    if (dppp1) *dppp1 *= scale;
    if (dppem1) *dppem1 *= scale;
    if (dppep1) *dppep1 *= scale;
}

static int fp_prepare_momentumdiff_batch_impl(
    const FpMomentumDiffBatchInput *in,
    const CRspectrum *crp_grid,
    const CRspectrum *cre_grid,
    FpMomentumDiffBatchOutput *out,
    void (*prepare_cell)(double, double, double, double,
                         CRspectrum *, CRspectrum *,
                         double *, double *, double *,
                         double *, double *, double *, double))
{
    int icell;

    if (in == 0 || crp_grid == 0 || cre_grid == 0 || out == 0) return -1;
    if (in->ncell <= 0 || prepare_cell == 0) {
        return -1;
    }

    momentumdiff_prepare_grid_cache(crp_grid, cre_grid);

    #pragma omp parallel for schedule(static) if(fp_cpu_should_parallel_cells(in->ncell))
    for (icell = 0; icell < in->ncell; icell++) {
        CRspectrum crp = *crp_grid;
        CRspectrum cre = *cre_grid;
        double dpp[np], dppe[npe];
        double dppm1 = 0.0, dppp1 = 0.0, dppem1 = 0.0, dppep1 = 0.0;
        const double dv_imc = in->dv_imc ? in->dv_imc[icell] : 0.0;

        if (!(dv_imc > 0.0) || !isfinite(dv_imc)) {
            momentumdiff_off(&crp, &cre,
                             dpp, &dppm1, &dppp1,
                             dppe, &dppem1, &dppep1);
        } else {
            prepare_cell(in->l_turb_mpc ? in->l_turb_mpc[icell] : 0.0,
                         dv_imc,
                         in->cs ? in->cs[icell] : 0.0,
                         in->beta_pl ? in->beta_pl[icell] : 0.0,
                         &crp, &cre,
                         dpp, &dppm1, &dppp1,
                         dppe, &dppem1, &dppep1, in->epmax);
            fp_apply_eta_dpp_cap_cell(in->model,
                                      in->l_turb_mpc ? in->l_turb_mpc[icell] : 0.0,
                                      dv_imc,
                                      in->cs ? in->cs[icell] : 0.0,
                                      in->beta_pl ? in->beta_pl[icell] : 0.0,
                                      in->tracer_mass_msun ? in->tracer_mass_msun[icell] : 0.0,
                                      crp_grid, cre_grid,
                                      in->crp_state_cell_major
                                          ? in->crp_state_cell_major + (size_t)icell * (size_t)np
                                          : 0,
                                      in->cre_state_cell_major
                                          ? in->cre_state_cell_major + (size_t)icell * (size_t)npe
                                          : 0,
                                      dpp, &dppm1, &dppp1,
                                      dppe, &dppem1, &dppep1);
        }

        fp_store_np_batch(out->dpp_batch, dpp, in->ncell, icell);
        fp_store_npe_batch(out->dppe_batch, dppe, in->ncell, icell);
        if (out->dppm1) out->dppm1[icell] = dppm1;
        if (out->dppp1) out->dppp1[icell] = dppp1;
        if (out->dppem1) out->dppem1[icell] = dppem1;
        if (out->dppep1) out->dppep1[icell] = dppep1;
    }

    return 0;
}

static void fp_prepare_momentumdiff_off_cell(double l_turb_mpc,
                                                  double dv_imc,
                                                  double cs,
                                                  double beta_pl,
                                                  CRspectrum *crp,
                                                  CRspectrum *cre,
                                                  double *dpp,
                                                  double *dppm1,
                                                  double *dppp1,
                                                  double *dppe,
                                                  double *dppem1,
                                                  double *dppep1,
                                                  double epmax)
{
    (void)l_turb_mpc;
    (void)dv_imc;
    (void)cs;
    (void)beta_pl;
    (void)epmax;
    momentumdiff_off(crp, cre, dpp, dppm1, dppp1, dppe, dppem1, dppep1);
}

int prepare_momentumdiff_asa_batch(const FpMomentumDiffBatchInput *in,
                                           const CRspectrum *crp_grid,
                                           const CRspectrum *cre_grid,
                                           FpMomentumDiffBatchOutput *out)
{
    if (in == 0 || in->l_turb_mpc == 0 || in->dv_imc == 0 ||
        in->cs == 0 || in->beta_pl == 0) {
        return -1;
    }
    return fp_prepare_momentumdiff_batch_impl(in, crp_grid, cre_grid, out,
                                                   momentumdiff_asa_1D);
}

int prepare_momentumdiff_ttd_batch(const FpMomentumDiffBatchInput *in,
                                           const CRspectrum *crp_grid,
                                           const CRspectrum *cre_grid,
                                           FpMomentumDiffBatchOutput *out)
{
    if (in == 0 || in->l_turb_mpc == 0 || in->dv_imc == 0 ||
        in->cs == 0 || in->beta_pl == 0) {
        return -1;
    }
    return fp_prepare_momentumdiff_batch_impl(in, crp_grid, cre_grid, out,
                                                   momentumdiff_ttd_1D);
}

int prepare_momentumdiff_direct_tacc_batch(const FpMomentumDiffBatchInput *in,
                                                   const CRspectrum *crp_grid,
                                                   const CRspectrum *cre_grid,
                                                   FpMomentumDiffBatchOutput *out)
{
    return fp_prepare_momentumdiff_batch_impl(in, crp_grid, cre_grid, out,
                                                   momentumdiff_direct_tacc_1D);
}

int prepare_momentumdiff_off_batch(const FpMomentumDiffBatchInput *in,
                                           const CRspectrum *crp_grid,
                                           const CRspectrum *cre_grid,
                                           FpMomentumDiffBatchOutput *out)
{
    return fp_prepare_momentumdiff_batch_impl(in, crp_grid, cre_grid, out,
                                                   fp_prepare_momentumdiff_off_cell);
}

int prepare_momentumdiff_batch(const FpMomentumDiffBatchInput *in,
                                       const CRspectrum *crp_grid,
                                       const CRspectrum *cre_grid,
                                       FpMomentumDiffBatchOutput *out)
{
    int model;

    if (in == 0) return -1;

    model = resolve_momentumdiff_model(in->model);

    switch (model) {
    case FP_MOMENTUMDIFF_MODEL_ASA:
        return prepare_momentumdiff_asa_batch(in, crp_grid, cre_grid, out);
    case FP_MOMENTUMDIFF_MODEL_TTD:
        return prepare_momentumdiff_ttd_batch(in, crp_grid, cre_grid, out);
    case FP_MOMENTUMDIFF_MODEL_DIRECT_TACC:
        return prepare_momentumdiff_direct_tacc_batch(in, crp_grid, cre_grid, out);
    case FP_MOMENTUMDIFF_MODEL_OFF:
        return prepare_momentumdiff_off_batch(in, crp_grid, cre_grid, out);
    default:
        return -1;
    }
}

int prepare_momentumdiff_cell_major(const FpMomentumDiffBatchInput *in,
                                            const CRspectrum *crp_grid,
                                            const CRspectrum *cre_grid,
                                            FpMomentumDiffCellMajorOutput *out)
{
    int icell;
    int model;

    if (in == 0 || crp_grid == 0 || cre_grid == 0 || out == 0) return -1;
    model = resolve_momentumdiff_model(in->model);
    momentumdiff_prepare_grid_cache(crp_grid, cre_grid);

    #pragma omp parallel for schedule(static) if(fp_cpu_should_parallel_cells(in->ncell))
    for (icell = 0; icell < in->ncell; icell++) {
        CRspectrum crp = *crp_grid;
        CRspectrum cre = *cre_grid;
        double dpp[np], dppe[npe];
        double dppm1 = 0.0, dppp1 = 0.0, dppem1 = 0.0, dppep1 = 0.0;
        const double dv_imc = in->dv_imc ? in->dv_imc[icell] : 0.0;

        if (!(dv_imc > 0.0) || !isfinite(dv_imc) || model == FP_MOMENTUMDIFF_MODEL_OFF) {
            momentumdiff_off(&crp, &cre, dpp, &dppm1, &dppp1, dppe, &dppem1, &dppep1);
        } else if (model == FP_MOMENTUMDIFF_MODEL_ASA) {
            momentumdiff_asa_1D(in->l_turb_mpc ? in->l_turb_mpc[icell] : 0.0,
                                dv_imc,
                                in->cs ? in->cs[icell] : 0.0,
                                in->beta_pl ? in->beta_pl[icell] : 0.0,
                                &crp, &cre,
                                dpp, &dppm1, &dppp1,
                                dppe, &dppem1, &dppep1, in->epmax);
        } else if (model == FP_MOMENTUMDIFF_MODEL_TTD) {
            momentumdiff_ttd_1D(in->l_turb_mpc ? in->l_turb_mpc[icell] : 0.0,
                                dv_imc,
                                in->cs ? in->cs[icell] : 0.0,
                                in->beta_pl ? in->beta_pl[icell] : 0.0,
                                &crp, &cre,
                                dpp, &dppm1, &dppp1,
                                dppe, &dppem1, &dppep1, in->epmax);
        } else if (model == FP_MOMENTUMDIFF_MODEL_DIRECT_TACC) {
            momentumdiff_direct_tacc_1D(in->l_turb_mpc ? in->l_turb_mpc[icell] : 0.0,
                                        dv_imc,
                                        in->cs ? in->cs[icell] : 0.0,
                                        in->beta_pl ? in->beta_pl[icell] : 0.0,
                                        &crp, &cre,
                                        dpp, &dppm1, &dppp1,
                                        dppe, &dppem1, &dppep1, in->epmax);
        } else {
            momentumdiff_off(&crp, &cre, dpp, &dppm1, &dppp1, dppe, &dppem1, &dppep1);
        }
        if (model != FP_MOMENTUMDIFF_MODEL_OFF) {
            fp_apply_eta_dpp_cap_cell(model,
                                      in->l_turb_mpc ? in->l_turb_mpc[icell] : 0.0,
                                      dv_imc,
                                      in->cs ? in->cs[icell] : 0.0,
                                      in->beta_pl ? in->beta_pl[icell] : 0.0,
                                      in->tracer_mass_msun ? in->tracer_mass_msun[icell] : 0.0,
                                      crp_grid, cre_grid,
                                      in->crp_state_cell_major
                                          ? in->crp_state_cell_major + (size_t)icell * (size_t)np
                                          : 0,
                                      in->cre_state_cell_major
                                          ? in->cre_state_cell_major + (size_t)icell * (size_t)npe
                                          : 0,
                                      dpp, &dppm1, &dppp1,
                                      dppe, &dppem1, &dppep1);
        }

        fp_store_np_cell_major(out->dpp_batch, dpp, icell);
        fp_store_npe_cell_major(out->dppe_batch, dppe, icell);
        if (out->dppm1) out->dppm1[icell] = dppm1;
        if (out->dppp1) out->dppp1[icell] = dppp1;
        if (out->dppem1) out->dppem1[icell] = dppem1;
        if (out->dppep1) out->dppep1[icell] = dppep1;
    }

    return 0;
}

int prepare_losses_batch(const FpLossBatchInput *in,
                                 const CRspectrum *crp_grid,
                                 const CRspectrum *cre_grid,
                                 FpLossBatchOutput *out)
{
    int icell;
    int j_pp;

    if (in == 0 || crp_grid == 0 || cre_grid == 0 || out == 0) return -1;
    if (in->ncell <= 0 || in->n_gas == 0 || in->kbt == 0 || in->b_field == 0 ||
        in->divv_gyr == 0 || in->rad_ic_batch == 0 ||
        in->rad_ic_m1 == 0 || in->rad_ic_p1 == 0) {
        return -1;
    }

    momentumdiff_prepare_crp_grid_cache(crp_grid);
    j_pp = find_pp_threshold_index(crp_grid);
    momentumdiff_prepare_cre_grid_cache(cre_grid);

    #pragma omp parallel for schedule(static) if(fp_cpu_should_parallel_cells(in->ncell))
    for (icell = 0; icell < in->ncell; icell++) {
        CRspectrum crp = *crp_grid;
        CRspectrum cre = *cre_grid;
        double crp_radp[np], crp_tloss[np], crp_invtloss[np];
        double cre_radp[npe], cre_tloss[npe], cre_invtloss[npe];
        double rad_ic_local[npe];
        FPloss crp_loss, cre_loss;
        int je;

        for (je = 0; je < npe; je++) {
            rad_ic_local[je] = in->rad_ic_batch[(size_t)je * (size_t)in->ncell + (size_t)icell];
        }

        crp_loss.radp = crp_radp;
        crp_loss.tloss = crp_tloss;
        crp_loss.invtloss = crp_invtloss;
        cre_loss.radp = cre_radp;
        cre_loss.tloss = cre_tloss;
        cre_loss.invtloss = cre_invtloss;

        const double divv_loss =
            (in->disable_adiabatic != 0 && in->disable_adiabatic[icell] != 0)
                ? 0.0 : in->divv_gyr[icell];

        prepare_crp_losses_1d_with_threshold(in->n_gas[icell], in->kbt[icell],
                                             divv_loss, &crp, j_pp, &crp_loss);
        prepare_cre_losses_1d(in->n_gas[icell], in->b_field[icell],
                                      divv_loss, rad_ic_local,
                                      in->rad_ic_m1[icell], in->rad_ic_p1[icell],
                                      &cre, &cre_loss);

        fp_store_np_batch(out->crp_radp_batch, crp_radp, in->ncell, icell);
        fp_store_np_batch(out->crp_tloss_batch, crp_tloss, in->ncell, icell);
        fp_store_np_batch(out->crp_invtloss_batch, crp_invtloss, in->ncell, icell);
        fp_store_npe_batch(out->cre_radp_batch, cre_radp, in->ncell, icell);
        fp_store_npe_batch(out->cre_tloss_batch, cre_tloss, in->ncell, icell);
        fp_store_npe_batch(out->cre_invtloss_batch, cre_invtloss, in->ncell, icell);

        if (out->crp_radpm1) out->crp_radpm1[icell] = crp_loss.radpm1;
        if (out->crp_radpp1) out->crp_radpp1[icell] = crp_loss.radpp1;
        if (out->cre_radpm1) out->cre_radpm1[icell] = cre_loss.radpm1;
        if (out->cre_radpp1) out->cre_radpp1[icell] = cre_loss.radpp1;
    }

    return 0;
}

int prepare_losses_cell_major(const FpLossBatchInput *in,
                                      const CRspectrum *crp_grid,
                                      const CRspectrum *cre_grid,
                                      FpLossCellMajorOutput *out)
{
    int icell;
    int j_pp;

    if (in == 0 || crp_grid == 0 || cre_grid == 0 || out == 0) return -1;
    if (in->ncell <= 0 || in->n_gas == 0 || in->kbt == 0 || in->b_field == 0 ||
        in->divv_gyr == 0 || in->rad_ic_batch == 0 ||
        in->rad_ic_m1 == 0 || in->rad_ic_p1 == 0) {
        return -1;
    }

    momentumdiff_prepare_crp_grid_cache(crp_grid);
    j_pp = find_pp_threshold_index(crp_grid);
    momentumdiff_prepare_cre_grid_cache(cre_grid);

    #pragma omp parallel for schedule(static) if(fp_cpu_should_parallel_cells(in->ncell))
    for (icell = 0; icell < in->ncell; icell++) {
        CRspectrum crp = *crp_grid;
        CRspectrum cre = *cre_grid;
        double crp_radp[np], crp_tloss[np], crp_invtloss[np];
        double cre_radp[npe], cre_tloss[npe], cre_invtloss[npe];
        double rad_ic_local[npe];
        FPloss crp_loss, cre_loss;
        int je;

        for (je = 0; je < npe; je++) {
            rad_ic_local[je] = in->rad_ic_batch[(size_t)je * (size_t)in->ncell + (size_t)icell];
        }

        crp_loss.radp = crp_radp;
        crp_loss.tloss = crp_tloss;
        crp_loss.invtloss = crp_invtloss;
        cre_loss.radp = cre_radp;
        cre_loss.tloss = cre_tloss;
        cre_loss.invtloss = cre_invtloss;

        const double divv_loss =
            (in->disable_adiabatic != 0 && in->disable_adiabatic[icell] != 0)
                ? 0.0 : in->divv_gyr[icell];

        prepare_crp_losses_1d_with_threshold(in->n_gas[icell], in->kbt[icell],
                                             divv_loss, &crp, j_pp, &crp_loss);
        prepare_cre_losses_1d(in->n_gas[icell], in->b_field[icell],
                              divv_loss, rad_ic_local,
                              in->rad_ic_m1[icell], in->rad_ic_p1[icell],
                              &cre, &cre_loss);

        fp_store_np_cell_major(out->crp_radp_batch, crp_radp, icell);
        fp_store_np_cell_major(out->crp_tloss_batch, crp_tloss, icell);
        fp_store_np_cell_major(out->crp_invtloss_batch, crp_invtloss, icell);
        fp_store_npe_cell_major(out->cre_radp_batch, cre_radp, icell);
        fp_store_npe_cell_major(out->cre_tloss_batch, cre_tloss, icell);
        fp_store_npe_cell_major(out->cre_invtloss_batch, cre_invtloss, icell);

        if (out->crp_radpm1) out->crp_radpm1[icell] = crp_loss.radpm1;
        if (out->crp_radpp1) out->crp_radpp1[icell] = crp_loss.radpp1;
        if (out->cre_radpm1) out->cre_radpm1[icell] = cre_loss.radpm1;
        if (out->cre_radpp1) out->cre_radpp1[icell] = cre_loss.radpp1;
    }

    return 0;
}

void prepare_secondary_kernel_flat(const CRspectrum *crp,
                                           const CRspectrum *cre,
                                           const double *beta_p,
                                           double *fqe_flat,
                                           int *np_min_qe)
{
    int je, jp;
    const int transpose = tracer_fp_secondary_transpose_enabled();

    if (crp == 0 || cre == 0 || beta_p == 0 || fqe_flat == 0 || np_min_qe == 0) return;

    for (je = 0; je < npe; je++) {
        double values[np];
        int jp0;
        for (jp = 0; jp < np; jp++) {
            double val = QeKernel_BB05(0, cre->E[je], crp->p[jp]) * beta_p[jp];
            if (isnan(val)) val = 0.0;
            values[jp] = val;
        }

        np_min_qe[je] = 1;
        for (jp = 1; jp < np; jp++) {
            if (values[jp] < 1.0e-80)
                np_min_qe[je] = jp;
        }

        jp0 = np_min_qe[je];
        if (jp0 < 1) jp0 = 1;
        if (jp0 > np) jp0 = np;

        for (jp = 0; jp < np; jp++) {
            double weight = 0.0;
            const double val = values[jp];

            if (jp0 < np) {
                if (jp == jp0 - 1) {
                    weight = 0.5 * val * crp->dp[jp0];
                } else if (jp >= jp0 && jp < np - 1) {
                    weight = 0.5 * val * (crp->dp[jp] + crp->dp[jp + 1]);
                } else if (jp == np - 1 && jp >= jp0) {
                    weight = 0.5 * val * crp->dp[jp];
                }
            }

            if (transpose) {
                fqe_flat[(size_t)jp * (size_t)npe + (size_t)je] = weight;
            } else {
                fqe_flat[(size_t)je * (size_t)np + (size_t)jp] = weight;
            }
        }

        np_min_qe[je] = (jp0 < np) ? (jp0 - 1) : np;
    }
}

void prepare_secondary_sources_batch(int ncell,
                                             const double *n_gas,
                                             const double *crp_batch,
                                             const double *crp_dp,
                                             const int *np_min_qe,
                                             const double *fqe_flat,
                                             const double *qepri_batch,
                                             double *qe_integral_batch,
                                             double *inje_batch)
{
    int icell, je, jp;
    const int transpose = tracer_fp_secondary_transpose_enabled();
    int transpose_jp0 = 0;

    if (ncell <= 0 || n_gas == 0 || crp_batch == 0 || crp_dp == 0 ||
        np_min_qe == 0 || fqe_flat == 0 ||
        qe_integral_batch == 0 || inje_batch == 0) {
        return;
    }
    (void)crp_dp;

    if (transpose) {
        transpose_jp0 = np;
        for (je = 0; je < npe; je++) {
            int jp0 = np_min_qe[je];
            if (jp0 < 0) jp0 = 0;
            if (jp0 < transpose_jp0) transpose_jp0 = jp0;
        }
        if (transpose_jp0 > np) transpose_jp0 = np;
    }

    #pragma omp parallel for schedule(static) private(je,jp) if(fp_cpu_should_parallel_cells(ncell))
    for (icell = 0; icell < ncell; icell++) {
        if (transpose) {
            double qint_local[npe];

            #pragma omp simd
            for (je = 0; je < npe; je++) qint_local[je] = 0.0;

            for (jp = transpose_jp0; jp < np; jp++) {
                const double n = crp_batch[(size_t)jp * (size_t)ncell + (size_t)icell];
                const double *wrow = fqe_flat + (size_t)jp * (size_t)npe;

                #pragma omp simd
                for (je = 0; je < npe; je++) {
                    qint_local[je] += n * wrow[je];
                }
            }

            #pragma omp simd
            for (je = 0; je < npe; je++) {
                const size_t je_off = (size_t)je * (size_t)ncell;
                const double qpri = (qepri_batch != 0)
                    ? qepri_batch[je_off + (size_t)icell] : 0.0;
                qe_integral_batch[je_off + (size_t)icell] = qint_local[je];
                inje_batch[je_off + (size_t)icell] =
                    qint_local[je] * Gyr * n_gas[icell] + qpri;
            }
            continue;
        }

        for (je = 0; je < npe; je++) {
            int jp0 = np_min_qe[je];
            size_t je_off = (size_t)je * (size_t)ncell;
            size_t fqe_off = (size_t)je * np;
            double qint = 0.0;
            double qpri = (qepri_batch != 0) ? qepri_batch[je_off + (size_t)icell] : 0.0;

            if (jp0 < 0) jp0 = 0;
            if (jp0 > np) jp0 = np;

            #pragma omp simd reduction(+:qint)
            for (jp = jp0; jp < np; jp++) {
                size_t jp_off = (size_t)jp * (size_t)ncell + (size_t)icell;
                qint += crp_batch[jp_off] * fqe_flat[fqe_off + (size_t)jp];
            }

            qe_integral_batch[je_off + (size_t)icell] = qint;
            inje_batch[je_off + (size_t)icell] = qint * Gyr * n_gas[icell] + qpri;
        }
    }
}

void prepare_secondary_sources_cell_major(int ncell,
                                                  const double *n_gas,
                                                  const double *crp_state_cell_major,
                                                  const double *crp_dp,
                                                  const int *np_min_qe,
                                                  const double *fqe_flat,
                                                  const double *qepri_batch,
                                                  double *qe_integral_batch,
                                                  double *inje_batch)
{
    int icell, je, jp;
    const int transpose = tracer_fp_secondary_transpose_enabled();
    int transpose_jp0 = 0;

    if (ncell <= 0 || n_gas == 0 || crp_state_cell_major == 0 || crp_dp == 0 ||
        np_min_qe == 0 || fqe_flat == 0 ||
        qe_integral_batch == 0 || inje_batch == 0) {
        return;
    }
    (void)crp_dp;

    if (transpose) {
        transpose_jp0 = np;
        for (je = 0; je < npe; je++) {
            int jp0 = np_min_qe[je];
            if (jp0 < 0) jp0 = 0;
            if (jp0 < transpose_jp0) transpose_jp0 = jp0;
        }
        if (transpose_jp0 > np) transpose_jp0 = np;
    }

    #pragma omp parallel for schedule(static) private(je,jp) if(fp_cpu_should_parallel_cells(ncell))
    for (icell = 0; icell < ncell; icell++) {
        const size_t cell_off = (size_t)icell * np;
        const size_t ecell_off = (size_t)icell * npe;
        if (transpose) {
            double qint_local[npe];

            #pragma omp simd
            for (je = 0; je < npe; je++) qint_local[je] = 0.0;

            for (jp = transpose_jp0; jp < np; jp++) {
                const double n = crp_state_cell_major[cell_off + (size_t)jp];
                const double *wrow = fqe_flat + (size_t)jp * (size_t)npe;

                #pragma omp simd
                for (je = 0; je < npe; je++) {
                    qint_local[je] += n * wrow[je];
                }
            }

            #pragma omp simd
            for (je = 0; je < npe; je++) {
                const double qpri = (qepri_batch != 0)
                    ? qepri_batch[ecell_off + (size_t)je] : 0.0;
                qe_integral_batch[ecell_off + (size_t)je] = qint_local[je];
                inje_batch[ecell_off + (size_t)je] =
                    qint_local[je] * Gyr * n_gas[icell] + qpri;
            }
            continue;
        }

        for (je = 0; je < npe; je++) {
            int jp0 = np_min_qe[je];
            size_t fqe_off = (size_t)je * np;
            double qint = 0.0;
            double qpri = (qepri_batch != 0) ? qepri_batch[ecell_off + (size_t)je] : 0.0;

            if (jp0 < 0) jp0 = 0;
            if (jp0 > np) jp0 = np;

            #pragma omp simd reduction(+:qint)
            for (jp = jp0; jp < np; jp++) {
                qint += crp_state_cell_major[cell_off + (size_t)jp] *
                        fqe_flat[fqe_off + (size_t)jp];
            }

            qe_integral_batch[ecell_off + (size_t)je] = qint;
            inje_batch[ecell_off + (size_t)je] = qint * Gyr * n_gas[icell] + qpri;
        }
    }
}

void prepare_secondary_sources_cell_major_gemm_layout(int ncell,
                                                              const double *n_gas,
                                                              const double *crp_state_cell_major,
                                                              const double *fqe_transposed,
                                                              const double *qepri_batch,
                                                              double *qe_integral_batch,
                                                              double *inje_batch)
{
    int icell, je, jp;

    if (ncell <= 0 || n_gas == 0 || crp_state_cell_major == 0 ||
        fqe_transposed == 0 || qe_integral_batch == 0 || inje_batch == 0) {
        return;
    }

    #pragma omp parallel for schedule(static) private(je,jp) if(fp_cpu_should_parallel_cells(ncell))
    for (icell = 0; icell < ncell; icell++) {
        const size_t cell_off = (size_t)icell * (size_t)np;
        const size_t ecell_off = (size_t)icell * (size_t)npe;
        double qint_local[npe];

        #pragma omp simd
        for (je = 0; je < npe; je++) qint_local[je] = 0.0;

        /*
         * GEMM equivalent:
         *   Qe[icell, je] += CRp[icell, jp] * W[jp, je]
         * with W stored as fqe_transposed[jp * npe + je].
         */
        for (jp = 0; jp < np; jp++) {
            const double n = crp_state_cell_major[cell_off + (size_t)jp];
            const double *wrow = fqe_transposed + (size_t)jp * (size_t)npe;

            #pragma omp simd
            for (je = 0; je < npe; je++) {
                qint_local[je] += n * wrow[je];
            }
        }

        #pragma omp simd
        for (je = 0; je < npe; je++) {
            const double qpri = (qepri_batch != 0)
                ? qepri_batch[ecell_off + (size_t)je] : 0.0;
            qe_integral_batch[ecell_off + (size_t)je] = qint_local[je];
            inje_batch[ecell_off + (size_t)je] =
                qint_local[je] * Gyr * n_gas[icell] + qpri;
        }
    }
}

void prepare_gamma_kernel_flat(const CRspectrum *crp,
                                       int nbins,
                                       double egamma_min,
                                       double egamma_max,
                                       double *fga_flat)
{
    int ng, jp;
    double a, d_egamma;

    if (crp == 0 || fga_flat == 0 || nbins <= 0) return;

    d_egamma = (egamma_max - egamma_min) / (double)nbins;
    a = egamma_min - d_egamma;

    for (ng = 0; ng < nbins; ng++) {
        double egamma;

        a += d_egamma;
        egamma = pow(10.0, a);

        for (jp = 0; jp < np; jp++) {
            fga_flat[(size_t)ng * (size_t)np + (size_t)jp] =
                gammaKernel(egamma, crp->p[jp]);
        }
    }
}

void prepare_neutrino_kernel_flat(const CRspectrum *crp,
                                          int nbins,
                                          double enu_min,
                                          double enu_max,
                                          double *fnu_flat)
{
    int nnu, jp;
    double a, d_enu;

    if (crp == 0 || fnu_flat == 0 || nbins <= 0) return;

    d_enu = (enu_max - enu_min) / (double)nbins;
    a = enu_min - d_enu;

    for (nnu = 0; nnu < nbins; nnu++) {
        double enu;

        a += d_enu;
        enu = pow(10.0, a);

        for (jp = 0; jp < np; jp++) {
            fnu_flat[(size_t)nnu * (size_t)np + (size_t)jp] =
                eKernel(HADRON_LEPTON_NUMU, enu, crp->p[jp]) +
                eKernel(HADRON_LEPTON_NUE, enu, crp->p[jp]) +
                numu1Kernel(enu, crp->p[jp]);
        }
    }
}

void prepare_ic_kernel_flat(const CRspectrum *cre,
                                    int nbins,
                                    double nu_min,
                                    double nu_max,
                                    double z,
                                    double *fic_flat)
{
    int nic, je;
    double a, d_nu;

    if (cre == 0 || fic_flat == 0 || nbins <= 0) return;

    d_nu = (nu_max - nu_min) / (double)nbins;
    a = nu_min - d_nu;

    for (nic = 0; nic < nbins; nic++) {
        double nu;

        a += d_nu;
        nu = pow(10.0, a);

        for (je = 0; je < npe; je++) {
            const double gamma = sqrt(1.0 + cre->p[je] * cre->p[je]);
            fic_flat[(size_t)nic * (size_t)npe + (size_t)je] =
                IC_emissivity_kernel(nu, gamma, z);
        }
    }
}

int prepare_fp_coefficients_batch(const FpCoeffBatchInput *in,
                                          const CRspectrum *crp_grid,
                                          const CRspectrum *cre_grid,
                                          FpCoeffBatchOutput *out)
{
    int icell;
    FpCoeffGpuInput gpu_in;
    FpCoeffGpuOutput gpu_out;

    if (in == 0 || crp_grid == 0 || cre_grid == 0 || out == 0) return -1;
    if (in->ncell <= 0 || in->n_gas == 0 || in->kbt == 0 || in->b_field == 0 ||
        in->divv_gyr == 0 || in->l_turb_mpc == 0 || in->dv_imc == 0 ||
        in->cs == 0 || in->beta_pl == 0 ||
        in->rad_ic_batch == 0 || in->rad_ic_m1 == 0 || in->rad_ic_p1 == 0) {
        return -1;
    }

    {
        const int active_model = resolve_momentumdiff_model(
            FP_MOMENTUMDIFF_MODEL_AUTO);

        gpu_in.loss.ncell = in->ncell;
        gpu_in.loss.n_gas = in->n_gas;
        gpu_in.loss.kbt = in->kbt;
        gpu_in.loss.b_field = in->b_field;
        gpu_in.loss.divv_gyr = in->divv_gyr;
        gpu_in.loss.disable_adiabatic = in->disable_adiabatic;
        gpu_in.loss.rad_ic_batch = in->rad_ic_batch;
        gpu_in.loss.rad_ic_m1 = in->rad_ic_m1;
        gpu_in.loss.rad_ic_p1 = in->rad_ic_p1;

        gpu_in.momentumdiff_on.ncell = in->ncell;
        gpu_in.momentumdiff_on.epmax = in->epmax;
        gpu_in.momentumdiff_on.model = active_model;
        gpu_in.momentumdiff_on.l_turb_mpc = in->l_turb_mpc;
        gpu_in.momentumdiff_on.dv_imc = in->dv_imc;
        gpu_in.momentumdiff_on.cs = in->cs;
        gpu_in.momentumdiff_on.beta_pl = in->beta_pl;
        gpu_in.momentumdiff_on.tracer_mass_msun = in->tracer_mass_msun;
        gpu_in.momentumdiff_on.crp_state_cell_major = in->crp_state_cell_major;
        gpu_in.momentumdiff_on.cre_state_cell_major = in->cre_state_cell_major;

        gpu_in.momentumdiff_off.ncell = in->ncell;
        gpu_in.momentumdiff_off.epmax = in->epmax;
        gpu_in.momentumdiff_off.model = FP_MOMENTUMDIFF_MODEL_OFF;
        gpu_in.momentumdiff_off.l_turb_mpc = in->l_turb_mpc;
        gpu_in.momentumdiff_off.dv_imc = in->dv_imc;
        gpu_in.momentumdiff_off.cs = in->cs;
        gpu_in.momentumdiff_off.beta_pl = in->beta_pl;
        gpu_in.momentumdiff_off.tracer_mass_msun = 0;
        gpu_in.momentumdiff_off.crp_state_cell_major = 0;
        gpu_in.momentumdiff_off.cre_state_cell_major = 0;

        gpu_out.loss = out->loss;
        gpu_out.momentumdiff_on.dpp_batch = out->dpp_batch;
        gpu_out.momentumdiff_on.dppe_batch = out->dppe_batch;
        gpu_out.momentumdiff_on.dppm1 = out->dppm1;
        gpu_out.momentumdiff_on.dppp1 = out->dppp1;
        gpu_out.momentumdiff_on.dppem1 = out->dppem1;
        gpu_out.momentumdiff_on.dppep1 = out->dppep1;

        gpu_out.momentumdiff_off.dpp_batch = out->dpp_off_batch;
        gpu_out.momentumdiff_off.dppe_batch = out->dppe_off_batch;
        gpu_out.momentumdiff_off.dppm1 = out->dppm1_off;
        gpu_out.momentumdiff_off.dppp1 = out->dppp1_off;
        gpu_out.momentumdiff_off.dppem1 = out->dppem1_off;
        gpu_out.momentumdiff_off.dppep1 = out->dppep1_off;

        if (prepare_fp_coefficients_gpu_batch(&gpu_in, crp_grid, cre_grid, &gpu_out) != 0) {
            return -1;
        }
    }

    return 0;
}

int prepare_fp_coefficients_cell_major_cpu(const FpCoeffBatchInput *in,
                                                   const CRspectrum *crp_grid,
                                                   const CRspectrum *cre_grid,
                                                   FpCoeffCellMajorOutput *out)
{
    FpMomentumDiffBatchInput diff_in;
    FpMomentumDiffCellMajorOutput diff_out;
    FpLossBatchInput loss_in;

    if (in == 0 || crp_grid == 0 || cre_grid == 0 || out == 0) return -1;

    loss_in.ncell = in->ncell;
    loss_in.n_gas = in->n_gas;
    loss_in.kbt = in->kbt;
    loss_in.b_field = in->b_field;
    loss_in.divv_gyr = in->divv_gyr;
    loss_in.disable_adiabatic = in->disable_adiabatic;
    loss_in.rad_ic_batch = in->rad_ic_batch;
    loss_in.rad_ic_m1 = in->rad_ic_m1;
    loss_in.rad_ic_p1 = in->rad_ic_p1;
    if (prepare_losses_cell_major(&loss_in, crp_grid, cre_grid, &out->loss) != 0) {
        return -1;
    }

    diff_in.ncell = in->ncell;
    diff_in.epmax = in->epmax;
    diff_in.l_turb_mpc = in->l_turb_mpc;
    diff_in.dv_imc = in->dv_imc;
    diff_in.cs = in->cs;
    diff_in.beta_pl = in->beta_pl;
    diff_in.tracer_mass_msun = in->tracer_mass_msun;
    diff_in.crp_state_cell_major = in->crp_state_cell_major;
    diff_in.cre_state_cell_major = in->cre_state_cell_major;

    diff_out.dpp_batch = out->dpp_batch;
    diff_out.dppe_batch = out->dppe_batch;
    diff_out.dppm1 = out->dppm1;
    diff_out.dppp1 = out->dppp1;
    diff_out.dppem1 = out->dppem1;
    diff_out.dppep1 = out->dppep1;
    diff_in.model = resolve_momentumdiff_model(FP_MOMENTUMDIFF_MODEL_AUTO);
    if (prepare_momentumdiff_cell_major(&diff_in, crp_grid, cre_grid, &diff_out) != 0) {
        return -1;
    }

    diff_out.dpp_batch = out->dpp_off_batch;
    diff_out.dppe_batch = out->dppe_off_batch;
    diff_out.dppm1 = out->dppm1_off;
    diff_out.dppp1 = out->dppp1_off;
    diff_out.dppem1 = out->dppem1_off;
    diff_out.dppep1 = out->dppep1_off;
    diff_in.model = FP_MOMENTUMDIFF_MODEL_OFF;
    if (prepare_momentumdiff_cell_major(&diff_in, crp_grid, cre_grid, &diff_out) != 0) {
        return -1;
    }

    return 0;
}

int prepare_fp_coefficients_gpu_batch(const FpCoeffGpuInput *in,
                                              const CRspectrum *crp_grid,
                                              const CRspectrum *cre_grid,
                                              FpCoeffGpuOutput *out)
{
    if (in == 0 || out == 0 || crp_grid == 0 || cre_grid == 0) return -1;

    if (prepare_losses_batch(&in->loss, crp_grid, cre_grid, &out->loss) != 0) {
        return -1;
    }
    if (prepare_momentumdiff_batch(&in->momentumdiff_on, crp_grid, cre_grid,
                                           &out->momentumdiff_on) != 0) {
        return -1;
    }
    if (prepare_momentumdiff_batch(&in->momentumdiff_off, crp_grid, cre_grid,
                                           &out->momentumdiff_off) != 0) {
        return -1;
    }

    return 0;
}

int prepare_synch_emission_batch(const FpSynchEmissionBatchInput *in,
                                         const CRspectrum *cre_grid,
                                         double *eps_syn_batch)
{
    int icell, je, nf;
    double *eps_freq_major;

    if (in == 0 || cre_grid == 0 || eps_syn_batch == 0) return -1;
    if (in->ncell <= 0 || in->nfreq <= 0 || in->nx_tab <= 0 ||
        in->ntheta_pitch <= 0 ||
        in->fx_tab == 0 || in->logfx_tab == 0 || in->logx_tab == 0 || in->logy == 0 ||
        in->b_dyn == 0 || in->logb == 0 || in->cre_batch == 0 ||
        in->gamma2e == 0 || in->theta == 0 || in->dtheta == 0 ||
        in->pitch_weight == 0 || in->nus == 0) {
        return -1;
    }

    eps_freq_major = (double *)malloc((size_t)in->ncell * (size_t)in->nfreq * sizeof(double));
    if (eps_freq_major == 0) return -1;

    for (icell = 0; icell < in->ncell; icell++) {
        double cre_local[npe];

        for (je = 0; je < npe; je++) {
            cre_local[je] = in->cre_batch[(size_t)je * (size_t)in->ncell + (size_t)icell];
        }

        for (nf = 0; nf < in->nfreq; nf++) {
            eps_freq_major[(size_t)nf * (size_t)in->ncell + (size_t)icell] =
                SYN_emissivity_from_arrays(cre_grid->dp, cre_local, in->b_dyn[icell],
                                           in->nx_tab, in->logx_tab, in->xmin,
                                           in->logfx_tab, in->fx_tab, in->logy[nf],
                                           in->logb[icell], in->pitch_weight);
        }
    }

    fp_transpose_freq_to_cell_major(eps_syn_batch, eps_freq_major, in->ncell, in->nfreq);
    free(eps_freq_major);

    return 0;
}

int prepare_synch_pitch_kernel_table(const FpSynchEmissionBatchInput *in,
                                             double *kernel_table)
{
    int ib, nf, je, k;

    if (in == 0 || kernel_table == 0) return -1;
    if (in->nlogb <= 1 || in->nfreq <= 0 || in->logy == 0 ||
        in->ntheta_pitch <= 0 ||
        in->fx_tab == 0 || in->logfx_tab == 0 || in->logx_tab == 0 || in->pitch_weight == 0) {
        return -1;
    }

    for (ib = 0; ib < in->nlogb; ib++) {
        const double logb = in->logb_min + (double)ib / in->inv_dlogb;
        const double inv_dlogx = (in->nx_tab > 1)
            ? 1.0 / (in->logx_tab[1] - in->logx_tab[0])
            : 0.0;
        for (nf = 0; nf < in->nfreq; nf++) {
            for (je = 0; je < npe; je++) {
                double int_th = 0.0;
                for (k = 0; k < in->ntheta_pitch; k++) {
                    double Fx = 0.0;
                    const double logx = in->logy[nf][je][k] - logb;
                    if (logx <= in->logx_tab[in->nx_tab - 1]) {
                        const int ix = (int)floor((logx - in->xmin) * inv_dlogx);
                        if (ix >= 0) {
                            if (ix >= in->nx_tab - 1) {
                                Fx = in->fx_tab[in->nx_tab - 1];
                            } else {
                                const double frac = (logx - in->logx_tab[ix]) * inv_dlogx;
                                Fx = pow(10.0,
                                         in->logfx_tab[ix] +
                                         frac * (in->logfx_tab[ix + 1] - in->logfx_tab[ix]));
                            }
                        }
                    }
                    int_th += Fx * in->pitch_weight[k];
                }
                kernel_table[((size_t)ib * (size_t)in->nfreq + (size_t)nf) * (size_t)npe + (size_t)je] = int_th;
            }
        }
    }

    return 0;
}

static double tracer_fp_interp_synch_kernel_logk(double k0,
                                                 double k1,
                                                 double t)
{
    if (t <= 0.0) return k0;
    if (t >= 1.0) return k1;

    /* The synch kernel varies nearly exponentially in the cutoff tail.
     * Interpolating in log K reduces the tail bias while preserving the
     * old linear path for zero/invalid entries. */
    if (k0 > 0.0 && k1 > 0.0) {
        return exp((1.0 - t) * log(k0) + t * log(k1));
    }
    return (1.0 - t) * k0 + t * k1;
}

int prepare_synch_emission_cell_major(const FpSynchEmissionBatchInput *in,
                                              const CRspectrum *cre_grid,
                                              double *eps_syn_batch)
{
    int icell, je, nf;

    if (in == 0 || cre_grid == 0 || eps_syn_batch == 0) return -1;
    if (in->ncell <= 0 || in->nfreq <= 0 || in->nx_tab <= 0 ||
        in->ntheta_pitch <= 0 ||
        in->fx_tab == 0 || in->logfx_tab == 0 || in->logx_tab == 0 || in->logy == 0 ||
        in->b_dyn == 0 || in->logb == 0 || in->cre_batch == 0 ||
        in->gamma2e == 0 || in->theta == 0 || in->dtheta == 0 ||
        in->pitch_weight == 0 || in->nus == 0) {
        return -1;
    }

    for (icell = 0; icell < in->ncell; icell++) {
        double cre_local[npe];
        const size_t cell_off = (size_t)icell * (size_t)npe;

        for (je = 0; je < npe; je++) {
            cre_local[je] = in->cre_batch[cell_off + (size_t)je];
        }

        for (nf = 0; nf < in->nfreq; nf++) {
            if (in->pitch_kernel_table != 0 && in->nlogb > 1 && in->inv_dlogb > 0.0) {
                const double A =
                    sqrt(3.0) * pow(4.8e-10, 3.0) * in->b_dyn[icell] /
                    (2.0 * 9.11e-28 * c * c) / (4.0 * M_PI);
                double integral = 0.0;
                double u = (in->logb[icell] - in->logb_min) * in->inv_dlogb;
                int ib = (int)floor(u);
                double t;
                if (ib < 0) ib = 0;
                if (ib > in->nlogb - 2) ib = in->nlogb - 2;
                t = u - (double)ib;
                if (t < 0.0) t = 0.0;
                if (t > 1.0) t = 1.0;
                for (je = 2; je < npe - 1 && cre_local[je + 1] > 0.0; je++) {
                    const size_t off0 = ((size_t)ib * (size_t)in->nfreq + (size_t)nf) * (size_t)npe + (size_t)je;
                    const size_t off1 = ((size_t)(ib + 1) * (size_t)in->nfreq + (size_t)nf) * (size_t)npe + (size_t)je;
                    const double K = tracer_fp_interp_synch_kernel_logk(
                        in->pitch_kernel_table[off0],
                        in->pitch_kernel_table[off1],
                        t);
                    integral += cre_local[je] * K * cre_grid->dp[je];
                }
                eps_syn_batch[(size_t)icell * (size_t)in->nfreq + (size_t)nf] = A * integral;
            } else {
                eps_syn_batch[(size_t)icell * (size_t)in->nfreq + (size_t)nf] =
                    SYN_emissivity_from_arrays(cre_grid->dp, cre_local, in->b_dyn[icell],
                                               in->nx_tab, in->logx_tab, in->xmin,
                                               in->logfx_tab, in->fx_tab, in->logy[nf],
                                               in->logb[icell], in->pitch_weight);
            }
        }
    }

    return 0;
}

int prepare_ic_emission_cell_major(const FpIcEmissionBatchInput *in,
                                           const CRspectrum *cre_grid,
                                           double *eps_ic_batch)
{
    int icell, nic, je;

    if (in == 0 || cre_grid == 0 || eps_ic_batch == 0) return -1;
    if (in->ncell <= 0 || in->nbins <= 0 ||
        in->cre_batch == 0 || in->fic_flat == 0) {
        return -1;
    }

    for (icell = 0; icell < in->ncell; icell++) {
        const double *cre_local = in->cre_batch + (size_t)icell * (size_t)npe;
        for (nic = 0; nic < in->nbins; nic++) {
            const double *fic_row = in->fic_flat + (size_t)nic * (size_t)npe;
            double integral = 0.0;

            for (je = 0; je < npe; je++) {
                integral += fic_row[je] * cre_local[je] * cre_grid->dp[je];
            }
            eps_ic_batch[(size_t)icell * (size_t)in->nbins + (size_t)nic] = integral;
        }
    }

    return 0;
}

int prepare_ic_emission_batch(const FpIcEmissionBatchInput *in,
                                      const CRspectrum *cre_grid,
                                      double *eps_ic_batch)
{
    return prepare_ic_emission_cell_major(in, cre_grid, eps_ic_batch);
}

int prepare_gamma_emission_cell_major(const FpGammaEmissionBatchInput *in,
                                              const CRspectrum *crp_grid,
                                              double *eps_gamma_batch)
{
    int icell, ng;
    double d_egamma, a;

    if (in == 0 || crp_grid == 0 || eps_gamma_batch == 0) return -1;
    if (in->ncell <= 0 || in->nbins <= 0 || in->n_gas == 0 ||
        in->crp_batch == 0 || in->beta_p == 0 || in->fga_flat == 0) {
        return -1;
    }

    d_egamma = (in->egamma_max - in->egamma_min) / (double)in->nbins;
    a = in->egamma_min - d_egamma;

    for (ng = 0; ng < in->nbins; ng++) {
        const double *fga_row;
        double eg;

        a += d_egamma;
        eg = pow(10.0, a);
        fga_row = in->fga_flat + (size_t)ng * (size_t)np;

        for (icell = 0; icell < in->ncell; icell++) {
            const double *crp_local = in->crp_batch + (size_t)icell * (size_t)np;
            eps_gamma_batch[(size_t)icell * (size_t)in->nbins + (size_t)ng] =
                eg * Q_one(np, (double *)fga_row, (double *)crp_local,
                           (double *)in->beta_p, (double *)crp_grid->dp) *
                in->n_gas[icell];
        }
    }

    return 0;
}

int prepare_gamma_emission_batch(const FpGammaEmissionBatchInput *in,
                                         const CRspectrum *crp_grid,
                                         double *eps_gamma_batch)
{
    return prepare_gamma_emission_cell_major(in, crp_grid, eps_gamma_batch);
}

int prepare_neutrino_emission_cell_major(const FpNeutrinoEmissionBatchInput *in,
                                                 const CRspectrum *crp_grid,
                                                 double *eps_nu_batch)
{
    int icell, nnu;
    double d_enu, a;

    if (in == 0 || crp_grid == 0 || eps_nu_batch == 0) return -1;
    if (in->ncell <= 0 || in->nbins <= 0 || in->n_gas == 0 ||
        in->crp_batch == 0 || in->beta_p == 0 || in->fnu_flat == 0) {
        return -1;
    }

    d_enu = (in->enu_max - in->enu_min) / (double)in->nbins;
    a = in->enu_min - d_enu;

    for (nnu = 0; nnu < in->nbins; nnu++) {
        const double *fnu_row;
        double enu;

        a += d_enu;
        enu = pow(10.0, a);
        fnu_row = in->fnu_flat + (size_t)nnu * (size_t)np;

        for (icell = 0; icell < in->ncell; icell++) {
            const double *crp_local = in->crp_batch + (size_t)icell * (size_t)np;
            eps_nu_batch[(size_t)icell * (size_t)in->nbins + (size_t)nnu] =
                enu * Q_one(np, (double *)fnu_row, (double *)crp_local,
                            (double *)in->beta_p, (double *)crp_grid->dp) *
                in->n_gas[icell];
        }
    }

    return 0;
}

int prepare_neutrino_emission_batch(const FpNeutrinoEmissionBatchInput *in,
                                            const CRspectrum *crp_grid,
                                            double *eps_nu_batch)
{
    return prepare_neutrino_emission_cell_major(in, crp_grid, eps_nu_batch);
}

int prepare_emission_gpu_batch(const FpEmissionGpuInput *in,
                                       const CRspectrum *crp_grid,
                                       const CRspectrum *cre_grid,
                                       FpEmissionGpuOutput *out)
{
    if (in == 0 || out == 0 || cre_grid == 0) return -1;

    if (prepare_synch_emission_cell_major(&in->synch, cre_grid, out->eps_syn_batch) != 0) {
        return -1;
    }
    if (in->ic != 0 && out->eps_ic_batch != 0) {
        if (prepare_ic_emission_cell_major(in->ic, cre_grid, out->eps_ic_batch) != 0) {
            return -1;
        }
    }
    if (in->gamma != 0 && out->eps_gamma_batch != 0) {
        if (crp_grid == 0) return -1;
        if (prepare_gamma_emission_cell_major(in->gamma, crp_grid, out->eps_gamma_batch) != 0) {
            return -1;
        }
    }
    if (in->neutrino != 0 && out->eps_nu_batch != 0) {
        if (crp_grid == 0) return -1;
        if (prepare_neutrino_emission_cell_major(in->neutrino, crp_grid, out->eps_nu_batch) != 0) {
            return -1;
        }
    }

    return 0;
}

int prepare_cc_coeff_batch(
    const FpChangCooperCoeffBatchInput *in,
    const CRspectrum *crp_grid,
    const CRspectrum *cre_grid,
    FpChangCooperCoeffBatchOutput *out)
{
    int icell, jp;

    if (in == 0 || crp_grid == 0 || cre_grid == 0 || out == 0) return -1;
    if (in->ncell <= 0 || in->crp_radp_batch == 0 || in->crp_tloss_batch == 0 ||
        in->dpp_batch == 0 || in->qpi_batch == 0 ||
        in->cre_radp_batch == 0 || in->cre_tloss_batch == 0 ||
        in->dppe_batch == 0 || in->inje_batch == 0 ||
        in->crp_radpm1 == 0 || in->crp_radpp1 == 0 ||
        in->cre_radpm1 == 0 || in->cre_radpp1 == 0 ||
        in->dppm1 == 0 || in->dppp1 == 0 ||
        in->dppem1 == 0 || in->dppep1 == 0 ||
        out->ccp_a_batch == 0 || out->ccp_b_batch == 0 || out->ccp_c_batch == 0 ||
        out->cce_a_batch == 0 || out->cce_b_batch == 0 || out->cce_c_batch == 0) {
        return -1;
    }

    #pragma omp parallel for schedule(static) private(jp) if(fp_cpu_should_parallel_cells(in->ncell))
    for (icell = 0; icell < in->ncell; icell++) {
        double cc_ap[np], cc_bp[np], cc_cp[np], cc_tp[np], qpi[np];
        double cc_ae[npe], cc_be[npe], cc_ce[npe], cc_te[npe], inje[npe];
        double out_ap[np], out_bp[np], out_cp[np];
        double out_ae[npe], out_be[npe], out_ce[npe];

        for (jp = 0; jp < np; jp++) {
            size_t off = (size_t)jp * (size_t)in->ncell + (size_t)icell;
            size_t src_off = (size_t)icell * (size_t)np + (size_t)jp;
            cc_ap[jp] = 1.0;
            cc_bp[jp] = in->crp_radp_batch[off] - 2.0 / crp_grid->p[jp] * in->dpp_batch[off];
            cc_cp[jp] = in->dpp_batch[off];
            cc_tp[jp] = in->crp_tloss_batch[off];
            qpi[jp] = in->qpi_batch[src_off];
        }
        for (jp = 0; jp < npe; jp++) {
            size_t off = (size_t)jp * (size_t)in->ncell + (size_t)icell;
            size_t src_off = (size_t)icell * (size_t)npe + (size_t)jp;
            cc_ae[jp] = 1.0;
            cc_be[jp] = in->cre_radp_batch[off] - 2.0 / cre_grid->p[jp] * in->dppe_batch[off];
            cc_ce[jp] = in->dppe_batch[off];
            cc_te[jp] = in->cre_tloss_batch[off];
            inje[jp] = in->inje_batch[src_off];
        }

        fp_build_cc_coeffs_from_terms(np - 1, in->dt, crp_grid->p, crp_grid->dp,
                                           cc_ap, cc_bp, cc_cp, cc_tp,
                                           crp_grid->pm1, crp_grid->pp1,
                                           in->crp_radpm1[icell] - 2.0 / crp_grid->pm1 * in->dppm1[icell],
                                           in->crp_radpp1[icell] - 2.0 / crp_grid->pp1 * in->dppp1[icell],
                                           in->dppm1[icell], in->dppp1[icell],
                                           out_ap, out_bp, out_cp);

        fp_build_cc_coeffs_from_terms(npe - 1, in->dt, cre_grid->p, cre_grid->dp,
                                           cc_ae, cc_be, cc_ce, cc_te,
                                           cre_grid->pm1, cre_grid->pp1,
                                           in->cre_radpm1[icell] - 2.0 / cre_grid->pm1 * in->dppem1[icell],
                                           in->cre_radpp1[icell] - 2.0 / cre_grid->pp1 * in->dppep1[icell],
                                           in->dppem1[icell], in->dppep1[icell],
                                           out_ae, out_be, out_ce);
          
        fp_store_cell_major(out->ccp_a_batch, out_ap, in->ncell, np, icell);
        fp_store_cell_major(out->ccp_b_batch, out_bp, in->ncell, np, icell);
        fp_store_cell_major(out->ccp_c_batch, out_cp, in->ncell, np, icell);
        fp_store_cell_major(out->cce_a_batch, out_ae, in->ncell, npe, icell);
        fp_store_cell_major(out->cce_b_batch, out_be, in->ncell, npe, icell);
        fp_store_cell_major(out->cce_c_batch, out_ce, in->ncell, npe, icell);
    }

    return 0;
}

int prepare_cc_coeff_cell_major(
    const FpChangCooperCoeffCellMajorInput *in,
    const CRspectrum *crp_grid,
    const CRspectrum *cre_grid,
    FpChangCooperCoeffBatchOutput *out)
{
    int icell, jp;

    if (in == 0 || crp_grid == 0 || cre_grid == 0 || out == 0) return -1;
    if (in->ncell <= 0 || in->crp_radp_batch == 0 || in->crp_tloss_batch == 0 ||
        in->dpp_batch == 0 || in->qpi_batch == 0 ||
        in->cre_radp_batch == 0 || in->cre_tloss_batch == 0 ||
        in->dppe_batch == 0 || in->inje_batch == 0 ||
        in->crp_radpm1 == 0 || in->crp_radpp1 == 0 ||
        in->cre_radpm1 == 0 || in->cre_radpp1 == 0 ||
        in->dppm1 == 0 || in->dppp1 == 0 ||
        in->dppem1 == 0 || in->dppep1 == 0 ||
        out->ccp_a_batch == 0 || out->ccp_b_batch == 0 || out->ccp_c_batch == 0 ||
        out->cce_a_batch == 0 || out->cce_b_batch == 0 || out->cce_c_batch == 0) {
        return -1;
    }

    #pragma omp parallel for schedule(static) private(jp) if(fp_cpu_should_parallel_cells(in->ncell))
    for (icell = 0; icell < in->ncell; icell++) {
        double cc_ap[np], cc_bp[np], cc_cp[np], cc_tp[np], qpi[np];
        double cc_ae[npe], cc_be[npe], cc_ce[npe], cc_te[npe], inje[npe];
        double out_ap[np], out_bp[np], out_cp[np];
        double out_ae[npe], out_be[npe], out_ce[npe];
        const size_t pcell_off = (size_t)icell * (size_t)np;
        const size_t ecell_off = (size_t)icell * (size_t)npe;

        for (jp = 0; jp < np; jp++) {
            cc_ap[jp] = 1.0;
            cc_bp[jp] = in->crp_radp_batch[pcell_off + (size_t)jp] -
                        2.0 / crp_grid->p[jp] * in->dpp_batch[pcell_off + (size_t)jp];
            cc_cp[jp] = in->dpp_batch[pcell_off + (size_t)jp];
            cc_tp[jp] = in->crp_tloss_batch[pcell_off + (size_t)jp];
            qpi[jp] = in->qpi_batch[pcell_off + (size_t)jp];
        }
        for (jp = 0; jp < npe; jp++) {
            cc_ae[jp] = 1.0;
            cc_be[jp] = in->cre_radp_batch[ecell_off + (size_t)jp] -
                        2.0 / cre_grid->p[jp] * in->dppe_batch[ecell_off + (size_t)jp];
            cc_ce[jp] = in->dppe_batch[ecell_off + (size_t)jp];
            cc_te[jp] = in->cre_tloss_batch[ecell_off + (size_t)jp];
            inje[jp] = in->inje_batch[ecell_off + (size_t)jp];
        }

        fp_build_cc_coeffs_from_terms(np - 1, in->dt, crp_grid->p, crp_grid->dp,
                                      cc_ap, cc_bp, cc_cp, cc_tp,
                                      crp_grid->pm1, crp_grid->pp1,
                                      in->crp_radpm1[icell] - 2.0 / crp_grid->pm1 * in->dppm1[icell],
                                      in->crp_radpp1[icell] - 2.0 / crp_grid->pp1 * in->dppp1[icell],
                                      in->dppm1[icell], in->dppp1[icell],
                                      out_ap, out_bp, out_cp);

        fp_build_cc_coeffs_from_terms(npe - 1, in->dt, cre_grid->p, cre_grid->dp,
                                      cc_ae, cc_be, cc_ce, cc_te,
                                      cre_grid->pm1, cre_grid->pp1,
                                      in->cre_radpm1[icell] - 2.0 / cre_grid->pm1 * in->dppem1[icell],
                                      in->cre_radpp1[icell] - 2.0 / cre_grid->pp1 * in->dppep1[icell],
                                      in->dppem1[icell], in->dppep1[icell],
                                      out_ae, out_be, out_ce);

        fp_store_cell_major(out->ccp_a_batch, out_ap, in->ncell, np, icell);
        fp_store_cell_major(out->ccp_b_batch, out_bp, in->ncell, np, icell);
        fp_store_cell_major(out->ccp_c_batch, out_cp, in->ncell, np, icell);
        fp_store_cell_major(out->cce_a_batch, out_ae, in->ncell, npe, icell);
        fp_store_cell_major(out->cce_b_batch, out_be, in->ncell, npe, icell);
        fp_store_cell_major(out->cce_c_batch, out_ce, in->ncell, npe, icell);
    }

    return 0;
}
