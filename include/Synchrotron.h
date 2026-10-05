#ifndef _SYN_h
#define _SYN_h
#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

void SYN_nu_crit_subB(double **nu_crit_table, double *theta, double *p);
void SYN_logy_table(double ***logy, double **nu_crit_table, int Nnu_s, double *nu_syn, double B_norm);
double SYN_crit_freq(double gamma2,double B,double theta);
void SYN_pitch_weight_table(const double *theta, const double *dtheta, double *pitch_weight);
double SYN_emissivity_from_arrays(const double *dp, const double *Ne, double B,
                                  int Nx, const double *logx_tab, double xmin,
                                  const double *logFx_tab, const double *Fx_tab, double **logy,
                                  double logB, const double *pitch_weight);
double SYN_emissivity(CRspectrum *CRe,double *theta,double *dtheta,double *Ne,double B,int Nx,double *logx_tab,double xmin,double *logFx_tab,double *Fx_tab,double **logy,double logB);
void SYN_logFx_table(int Nx, double *logx, double *logFx);

#ifdef __cplusplus
}
#endif

#endif
