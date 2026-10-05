#ifndef _FP_Coef_h
#define _FP_Coef_h
#include"params.h"

#ifdef __cplusplus
extern "C" {
#define FP_DECL_2D(name, n2) double *name
#else
#define FP_DECL_2D(name, n2) double name[][n2]
#endif

void CR_Coef_1D(double z,double dt,double n_ISM,double n_nxt,double kBT,double B, double divv, double *rad_IC, double rad_IC_m1, double rad_IC_p1, CRspectrum *CRP, CRspectrum *CRE, FPloss *CRPloss, FPloss *CREloss);

void momentum_bin(CRspectrum *CRp, CRspectrum *CRe,double *bp,double *gamma2e);

void rad_IC_cool(int N_z, double *z, double **radIC, double *radIC_m1, double *radIC_p1, CRspectrum *CRe);
double b_Coulomb_p(double nicm,double p,double kBT);
double b_Coulomb_e(double nicm,double pe);
double b_Brems_BG70(double n_p,double n_He,double pe);
double b_synch(double p,double B);
double IC_emissivity_kernel(double nu,double gamma,double z);
void fill_IC_kernel(double **fic,double z,double *pe);
int Hillas(int N_Emax,int N_energy,double *E,double Z,double B,double R);

double accelerationtime_ASA(double L,double cs,double Ms,double beta_pl);
double accelerationtime_TTD(double L,double cs, double Ms, double v_turb_com);
double treacc_merger(double z, double M, double xi);
double Etu_Eth(int merger_flag,double x);
double F_turb(double v_turb,double rho,double L);

void momentumdiff_asa_1D(double L, double dv_imc, double cs, double beta_pl,
    CRspectrum *CRp, CRspectrum *CRe,
    double *Dpp, double *Dppm1, double *Dppp1,
    double *Dppe, double *Dppem1, double *Dppep1, double Epmax);
void momentumdiff_ttd_1D(double L, double dv_imc, double cs, double beta_pl,
    CRspectrum *CRp, CRspectrum *CRe,
    double *Dpp, double *Dppm1, double *Dppp1,
    double *Dppe, double *Dppem1, double *Dppep1, double Epmax);
void momentumdiff_direct_tacc_1D(double L, double dv_imc, double cs, double beta_pl,
    CRspectrum *CRp, CRspectrum *CRe,
    double *Dpp, double *Dppm1, double *Dppp1,
    double *Dppe, double *Dppem1, double *Dppep1, double Epmax);
void momentumdiff_prepare_grid_cache(const CRspectrum *CRp, const CRspectrum *CRe);
void momentumdiff_prepare_crp_grid_cache(const CRspectrum *CRp);
void momentumdiff_prepare_cre_grid_cache(const CRspectrum *CRe);
const double *momentumdiff_crp_sigma_pp_cache(const CRspectrum *CRp);
const double *momentumdiff_crp_sigmoid_pp_cache(const CRspectrum *CRp);
int momentumdiff_crp_j_pp_cache(const CRspectrum *CRp);
const double *momentumdiff_cre_p2_cache(const CRspectrum *CRe);
double momentumdiff_cre_pm1_p2_cache(const CRspectrum *CRe);
double momentumdiff_cre_pp1_p2_cache(const CRspectrum *CRe);
void momentumdiff_1D(double L, double dv_imc, double cs, double beta_pl,CRspectrum *CRp, CRspectrum *CRe,double *Dpp,double *Dppm1,double *Dppp1,double *Dppe,double *Dppem1,double *Dppep1,double Epmax);
void momentumdiff_off(CRspectrum *CRp, CRspectrum *CRe,double *Dpp,double *Dppm1,double *Dppp1,double *Dppe,double *Dppem1,double *Dppep1);

double Drr_pitch(double p,double B,double lc,double m,double Z);
double Drr_test(double r,double p);
double Drr_lA(double lA);
double adiabatic_V(double r,double dr,double r_pre,double dr_pre,double dt,double p);
double adiabatic_n(double n,double n_pre,double dt,double p);
double adiabatic_divv(double divv,double p);

void Secondary_electron_1D(double n_ISM,double *Inje,double *Npave,double *dp,int *np_min_Qe,double **fQe,double **fQe_knock,double *Qepri);
void Secondary_electrons(double *Qe_buff,double n_ISM,double *Inje,CRspectrum *CRp,int *np_min_Qe,double **fQe,double *Qepri);

double c_sound(double T);
double l_Alfven(double beta_pl,double L,double M_s);
double v_Alfven(double B,double rho);
double B_dynamo(double rho, double dv);
double dv_limit(double dv, double cs);




#ifdef __cplusplus
}
#endif

#undef FP_DECL_2D

#endif
