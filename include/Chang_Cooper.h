#ifndef _CC_h
#define _CC_h
#include"params.h"

#ifdef __cplusplus
extern "C" {
#endif

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

void Coef_CC (int Nm,double dt,double *x,double *dx,double *A,double *B,double *C,double *T,double *Q,double *a,double *b,double *c,double xm1,double xp1,double Bm1,double Bp1,double Cm1,double Cp1);
void Chang_Cooper (int Narray,double dt,double *a,double *b,double *c,double *Q,double *u);
double PP95_analytic(double x,double x0,double a,double b,double D,double theta,double r,double lambda);
void CC_1D(int sign,int N_energy, double dt, ChangCooper *CC,double *Q_p,CRspectrum *CR);
double log_linear_interpolate(double x,double x1,double y1,double x2,double y2);
double PL_exp_interpolate(double x,double x1,double y1,double x2,double y2,double x3,double y3);
double PL_exp_interpolate_debug(double x,double x1,double y1,double x2,double y2,double x3,double y3);
void CC1D_Coef(
    double z, double dt, ChangCooper *CCp, 
    CRspectrum *CRP, FPloss *CRPloss,
    double *Dpp, double Dppm1, double Dppp1, double *Qpi,
    ChangCooper *CCe,
    CRspectrum *CRE, FPloss *CREloss,
    double *Dppe,double Dppem1,double Dppep1,double *Inje);


#ifdef __cplusplus
}
#endif

#endif
