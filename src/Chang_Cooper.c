#include <stdio.h>
#include<math.h>
#include<stdlib.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_sf.h>
#include"Chang_Cooper.h"
#include"fp_shared_core.h"
#include"params.h"

/*

Fokker-Planck equation in form of

du/ = 1/   d/[ C(x) du/ + B(x)u ] - u(x)/ + Q(x) 
dt    A(x) dx[      dx          ]   T(x)


ref:  Park & Petrosian 1995, 1996

*/

void cc_eval_face_weights(double w,
                          double *delta,
                          double *wp,
                          double *wm)
{
    if (delta == 0 || wp == 0 || wm == 0) return;

    *delta = 1.0 / w - 1.0 / (exp(w) - 1.0);
    if (fabs(w) < 0.1) {
        double denom = 1.0 + pow(w, 2.0) / 24.0 + pow(w, 4.0) / 1920.0;
        *wp = exp(0.5 * w) / denom;
        *wm = exp(-0.5 * w) / denom;
    } else {
        double aw = fabs(w);
        double denom = 1.0 - exp(-aw);
        *wp = aw * exp(-0.5 * aw + 0.5 * w) / denom;
        *wm = aw * exp(-0.5 * aw - 0.5 * w) / denom;
    }
}

void cc_prepare_face_weight(double b_left,
                            double b_right,
                            double c_left,
                            double c_right,
                            double x_left,
                            double x_right,
                            double *delta,
                            double *wp,
                            double *wm,
                            double *w_out)
{
    double w;

    if (delta == 0 || wp == 0 || wm == 0) return;
    *delta = 0.0;
    *wp = 0.0;
    *wm = 0.0;
    if (w_out) *w_out = 0.0;

    if (fabs(c_left) <= 1.0e-200) return;

    w = 0.5 * (b_left + b_right) / (0.5 * (c_left + c_right)) * (x_right - x_left);
    cc_eval_face_weights(w, delta, wp, wm);
    if (w_out) *w_out = w;
}

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
                              double *c)
{
    if (fabs(C[m]) > 1.0e-200) {
        double dxm_12 = x[m] - x[m - 1];
        double dxm12 = x[m + 1] - x[m];
        double cm_12 = 0.5 * (C[m] + C[m - 1]);
        double cm12 = 0.5 * (C[m] + C[m + 1]);
        a[m] = dt / (A[m] * dx[m]) * cm_12 / dxm_12 * Wm[m - 1];
        b[m] = 1.0 + dt / (A[m] * dx[m]) *
                     (cm_12 / dxm_12 * Wp[m - 1] + cm12 / dxm12 * Wm[m]) +
               dt / T[m];
        c[m] = dt / (A[m] * dx[m]) * cm12 / dxm12 * Wp[m];
    } else {
        double deltam_12 = delta[m - 1];
        double deltam12 = delta[m];
        double bm_12 = 0.5 * (B[m] + B[m - 1]);
        double bm12 = 0.5 * (B[m] + B[m + 1]);
        a[m] = -deltam_12 * bm_12 * dt / (A[m] * dx[m]);
        b[m] = 1.0 + dt / (A[m] * dx[m]) *
                     ((1.0 - deltam_12) * bm_12 - deltam12 * bm12) +
               dt / T[m];
        c[m] = dt / (A[m] * dx[m]) * (1.0 - delta[m]) * bm12;
    }
}

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
                            double Wmm1)
{
    (void)Nm;
    if (fabs(C[0]) > 1.0e-200) {
        double c_12 = 0.0;
        a[0] = dt / (A[0] * dx[0]) * c_12 / (x[0] - xm1) * Wmm1;
        b[0] = 1.0 + dt / (A[0] * dx[0]) *
                     (c_12 / (x[0] - xm1) * Wpm1 +
                      0.5 * (C[0] + C[1]) / (x[1] - x[0]) * Wm[0]) +
               dt / T[0];
        c[0] = dt / (A[0] * dx[0]) * 0.5 * (C[0] + C[1]) / (x[1] - x[0]) * Wp[0];
    } else {
        double b_12 = 0.5 * (B[0] + Bm1);
        b_12 = 0.0;
        a[0] = -delta[0] * b_12 * dt / (A[0] * dx[0]);
        b[0] = 1.0 + dt / (A[0] * dx[0]) * b_12 + dt / T[0];
        c[0] = dt / (A[0] * dx[0]) * (1.0 - delta[0]) * 0.5 * (B[0] + B[1]);
    }
}

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
                             double w_nm)
{
    if (fabs(C[Nm]) > 1.0e-200) {
        a[Nm] = dt / (A[Nm] * dx[Nm]) *
                (0.5 * (C[Nm] + C[Nm - 1])) / (x[Nm] - x[Nm - 1]) * Wm[Nm - 1];
        b[Nm] = 1.0 + dt / (A[Nm] * dx[Nm]) *
                      ((0.5 * (C[Nm] + C[Nm - 1])) / (x[Nm] - x[Nm - 1]) * Wp[Nm - 1]) +
                dt / T[Nm];
        if (isnan(a[Nm])) {
            printf("dt/(A[Nm]*dx[Nm] = %e, %e, %e\n",
                   dt / (A[Nm] * dx[Nm]),
                   (0.5 * (C[Nm] + C[Nm - 1])) / (x[Nm] - x[Nm - 1]), dt);
            printf("ERROR a[Nm] NAN / %d %e %e %e %e %e %e %e\n",
                   Nm, x[Nm + 1], A[Nm], dx[Nm], C[Nm - 1], delta[Nm - 1], Wm[Nm], w_nm);
            exit(1);
        }
        c[Nm] = 0.0;
    } else {
        a[Nm] = -delta[Nm - 1] * 0.5 * (B[Nm] + B[Nm - 1]) * dt / (A[Nm] * dx[Nm]);
        b[Nm] = 1.0 + dt / (A[Nm] * dx[Nm]) * (0.5 * (B[Nm - 1] + B[Nm])) + dt / T[Nm];
        c[Nm] = 0.0;
    }
    if (isnan(c[Nm])) {
        printf("ERROR c[Nm] %e %e %e %e\n", A[Nm], dx[Nm], B[Nm], Bp1);
        exit(1);
    }
}

void Coef_CC (int Nm,double dt,double *x,double *dx,double *A,double *B,double *C,double *T,double *Q,double *a,double *b,double *c,double xm1,double xp1,double Bm1,double Bp1,double Cm1,double Cp1){
    int m;
    double delta[Nm+1],w[Nm+1];
    double Wp[Nm+1],Wm[Nm+1];
    double Wpm1 = 0.0, Wmm1 = 0.0;

    (void)Q;

    for (m = 0; m < Nm; m++) {
        cc_prepare_face_weight(B[m], B[m + 1], C[m], C[m + 1], x[m], x[m + 1],
                               &delta[m], &Wp[m], &Wm[m], &w[m]);
    }

    if (fabs(C[0]) > 1.0e-200) {
        cc_prepare_face_weight(Bm1, B[0], Cm1, C[0], xm1, x[0],
                               &delta[0], &Wpm1, &Wmm1, 0);
    }

    cc_prepare_face_weight(B[Nm], Bp1, C[Nm], Cp1, x[Nm], xp1,
                           &delta[Nm], &Wp[Nm], &Wm[Nm], &w[Nm]);
    if (isnan(w[Nm])) {
        printf("ERROR w[Nm] / %e %e %e %e\n", B[Nm], Bp1, Cp1, C[Nm]);
        exit(1);
    }
    if (fabs(C[Nm]) > 1.0e-200 && isnan(Wp[Nm])) {
        printf("ERROR Wp[Nm] /%e %e %e %e %e\n", w[Nm], B[Nm], Bp1, C[Nm], Cp1);
        exit(1);
    }

    for (m = 1; m < Nm; m++) {
        if (fabs(C[m]) <= 1.0e-200) delta[m - 1] = 0.0;
        cc_build_interior_coeffs(m, dt, x, dx, A, B, C, T, delta, Wp, Wm, a, b, c);
    }

    cc_build_left_boundary(Nm, dt, x, dx, A, B, C, T, delta, Wp, Wm,
                           xm1, Bm1, a, b, c, Wpm1, Wmm1);
    cc_build_right_boundary(Nm, dt, x, dx, A, B, C, T, delta, Wp, Wm,
                            Bp1, a, b, c, w[Nm]);
}


void Chang_Cooper(int Narray,double dt,double *a,double *b,double *c,double *Q,double *u){
    int ierr;
    ierr = solve_cc_1d_cpu(Narray + 1, dt, a, b, c, Q, u);
    if (ierr != 0) {
        printf("ERROR solve_cc_1d_cpu ierr=%d Narray=%d\n",
               ierr, Narray);
        exit(1);
    }
}

double PP95_analytic(double x,double x0,double a,double b,double D,double theta,double r,double lambda){  // analytical solution alpha <0.0, beta > 0.0//
    double alpha = r-1.0,beta,g,y,y0;
    beta = -b/alpha;
    double delta,delbar,del_p,del_m;

    double lambda0 = pow(0.5*(a+1.0),2.0)+theta,mu = sqrt(lambda0-lambda);

    del_p = 0.5*(a+1.0)+mu;del_m =0.5*(a+1.0)-mu;

    if(alpha>0.0){delta = del_p;delbar = del_m;}
    else{delta = del_m;delbar = del_p;}

    g = (1.0-delta)/alpha; 
    double tila = 1.0+delta/alpha,tilb = 1.0+(delta-delbar)/alpha;
    y = fabs(beta)*pow(x,alpha);y0 = fabs(beta)*pow(x0,alpha);

    double G = 0.0;
    double G0;

    if (beta>0.0){
    G = 1.0/(D*fabs(b))*gsl_sf_gamma(tila)/gsl_sf_gamma(tilb)*pow(x0,-r)*pow(y/y0,-g)*pow(y0,tilb);
    if(y<y0){
        G0 = exp(-y0)*gsl_sf_hyperg_1F1(tila,tilb,y)*gsl_sf_hyperg_U(tila,tilb,y0);
    }else{
        G0 = exp(-y0)*gsl_sf_hyperg_1F1(tila,tilb,y0)*gsl_sf_hyperg_U(tila,tilb,y);
    }}

    else{
    G = 1.0/(D*fabs(b))*gsl_sf_gamma(tilb-tila)/gsl_sf_gamma(tilb)*pow(x0,-r)*pow(y/y0,-g)*pow(y0,tilb);
    if(y<y0){
        G0 = exp(-y)*gsl_sf_hyperg_1F1(tilb-tila,tilb,y)*gsl_sf_hyperg_U(tilb-tila,tilb,y0);}
    else{
        G0 = exp(-y)*gsl_sf_hyperg_1F1(tilb-tila,tilb,y0)*gsl_sf_hyperg_U(tilb-tila,tilb,y);
    }
    }

    G *= G0;
    return G;
}

void CC1D_Coef(
        double z, double dt, ChangCooper *CCp, 
        CRspectrum *CRP, FPloss *CRPloss,
        double *Dpp, double Dppm1, double Dppp1, double *Qpi,
        ChangCooper *CCe,
        CRspectrum *CRE, FPloss *CREloss,
    double *Dppe,double Dppem1,double Dppep1,double *Inje)
    {
    int j;
    double CC_Ap[np],CC_Bp[np],CC_Cp[np],CC_Tp[np],CC_Qp[np];
    double CC_Ape[npe],CC_Bpe[npe],CC_Cpe[npe],CC_Tpe[npe],CC_Qpe[npe];

    (void)z;

       for (j = 0; j < np; j++){
        CC_Ap[j] = 1.0;
        CC_Bp[j] = CRPloss->radp[j]-2.0/CRP->p[j]*Dpp[j]; 
        CC_Cp[j] = Dpp[j];
        CC_Tp[j] = CRPloss->tloss[j];
        CC_Qp[j] = Qpi[j];
       }
       for (j = 0; j < npe; j++){
        CC_Ape[j] = 1.0;
        CC_Bpe[j] = CREloss->radp[j]-2.0/CRE->p[j]*Dppe[j];
        if(isnan(CC_Bpe[j])){printf("ERROR Bpe[i][j] / %d %e %e %e\n",j,CC_Bpe[j],CREloss->radp[j],Dppe[j]);exit(1);}
        CC_Cpe[j] = Dppe[j];
        CC_Tpe[j] = CREloss->tloss[j];
        CC_Qpe[j] = Inje[j];
       }

    double Bpm1,Bpp1,Cpm1,Cpp1;
    double Bpem1,Bpep1,Cpem1,Cpep1;

        Bpm1 = CRPloss->radpm1-2.0/CRP->pm1*Dppm1;
        Bpp1 = CRPloss->radpp1-2.0/CRP->pp1*Dppp1;  
        if(isnan(Bpp1)){printf("ERROR Bpp1[i] %e %e %e %e\n",CRP->pp1,CRPloss->radp[np-1],CRPloss->radp[np-2],Dppp1);}
        Cpm1 = Dppm1;
        Cpp1 = Dppp1;

        Bpem1 = CREloss->radpm1-2.0/CRE->pm1*Dppem1;
        Bpep1 = CREloss->radpp1-2.0/CRE->pp1*Dppep1;
        if(isnan(Bpep1)){printf("ERROR Bpep1[i] %e %e %e %e\n",CRE->pp1,CREloss->radp[npe-1],CREloss->radp[npe-2],Dppep1);}
        Cpem1 = Dppem1;
        Cpep1 = Dppep1;

    Coef_CC(np-1,dt,CRP->p,CRP->dp,CC_Ap,CC_Bp,CC_Cp,CC_Tp,Qpi,CCp->A,CCp->B,CCp->C,CRP->pm1,CRP->pp1,Bpm1,Bpp1,Cpm1,Cpp1);
    Coef_CC(npe-1,dt,CRE->p,CRE->dp,CC_Ape,CC_Bpe,CC_Cpe,CC_Tpe,Inje,CCe->A,CCe->B,CCe->C,CRE->pm1,CRE->pp1,Bpem1,Bpep1,Cpem1,Cpep1);
}


void CC_1D(int sign,int N_energy, double dt, ChangCooper *CC,double *Q_p,CRspectrum *CR){
        int ierr;
        ierr = solve_cc_1d_cpu(N_energy, dt,
                               CC->A, CC->B, CC->C,
                               Q_p, CR->N);
        if (ierr != 0) {
            printf("ERROR solve_cc_1d_cpu ierr=%d sign=%d N_energy=%d\n",
                   ierr, sign, N_energy);
            exit(1);
        }
}


double log_linear_interpolate(double x,double x1,double y1,double x2,double y2){
    double a,y;
    if(y1*y2 != 0.0){a = (log10(y2)-log10(y1))/(log10(x2)-log10(x1))*(log10(x)-log10(x1))+log10(y1);
    y = pow(10.0,a);}
    else{y = 0.0;}
    return y;
}

double PL_exp_interpolate(double x,double x1,double y1,double x2,double y2,double x3,double y3){
    double A12 = x2*log(y1)-x1*log(y2),A23= x3*log(y2)-x2*log(y3),
    B12 = x2*log(x1)-x1*log(x2),B23= x3*log(x2)-x2*log(x3);
    double alpha = -((x3-x2)*A12-(x2-x1)*A23)/((x3-x2)*B12-(x2-x1)*B23);
    double logC = (A23+alpha*B23)/(x3-x2);
    double xcut = -x1/(log(y1)-logC+alpha*log(x1));
    return exp(logC-alpha*log(x)-x/xcut);
}

double PL_exp_interpolate_debug(double x,double x1,double y1,double x2,double y2,double x3,double y3){
    double A12 = x2*log(y1)-x1*log(y2),A23= x3*log(y2)-x2*log(y3),
    B12 = x2*log(x1)-x1*log(x2),B23= x3*log(x2)-x2*log(x3);
    double alpha = -((x3-x2)*A12-(x2-x1)*A23)/((x3-x2)*B12-(x2-x1)*B23);
    double logC = (A23+alpha*B23)/(x3-x2);
    double xcut = -x1/(log(y1)-logC+alpha*log(x1));
    printf("alpha = %e, logC = %e, xcut = %e, A23 = %e, %e %e %e %e\n",alpha,logC,xcut,A23,exp(logC),pow(x,-alpha),exp(-x/xcut),exp(logC-alpha*log(x)-x/xcut));
    return exp(logC-alpha*log(x)-x/xcut);
}
