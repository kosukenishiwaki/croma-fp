#ifndef INCLUDED_fp_cuda_compat_h_
#define INCLUDED_fp_cuda_compat_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "params.h"
#include "fp_shared_core.h"
#include "Synchrotron.h"

/*
 * CUDA-side tests should include only declarations that are valid in C++
 * compilation under nvcc. Do not include legacy headers here if they expose
 * VLA-based prototypes.
 */
void momentum_bin(CRspectrum *CRp, CRspectrum *CRe, double *bp, double *gamma2e);

void cc_eval_face_weights(double w,
                          double *delta,
                          double *wp,
                          double *wm);
void cc_prepare_face_weight(double b_left,
                            double b_right,
                            double c_left,
                            double c_right,
                            double x_left,
                            double x_right,
                            double *delta,
                            double *wp,
                            double *wm,
                            double *w_out);
void cc_build_interior_coeffs(int m,
                              double dt,
                              const double *x,
                              const double *dx,
                              const double *A,
                              const double *B,
                              const double *C,
                              const double *T,
                              const double *delta,
                              const double *Wp,
                              const double *Wm,
                              double *a,
                              double *b,
                              double *c);
void cc_build_left_boundary(int Nm,
                            double dt,
                            const double *x,
                            const double *dx,
                            const double *A,
                            const double *B,
                            const double *C,
                            const double *T,
                            const double *delta,
                            const double *Wp,
                            const double *Wm,
                            double xm1,
                            double Bm1,
                            double *a,
                            double *b,
                            double *c,
                            double Wpm1,
                            double Wmm1);
void cc_build_right_boundary(int Nm,
                             double dt,
                             const double *x,
                             const double *dx,
                             const double *A,
                             const double *B,
                             const double *C,
                             const double *T,
                             const double *delta,
                             const double *Wp,
                             const double *Wm,
                             double Bp1,
                             double *a,
                             double *b,
                             double *c,
                             double w_nm);

#ifdef __cplusplus
}
#endif

#endif
