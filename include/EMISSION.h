#include "params.h"
void outputSyn_Kernel(int Nx, double *Fx_tab, double *logFx_tab, double *logx,double xmin, double ***logy,double z,int Nnu,double numin,double numax,CRspectrum *CRe,double *N,double *gamma2e,double B,double logB,double *epsSyn, double *theta, double *dtheta, double *nus);
void outputIC_one(char *strpass,double z,double numin,double numax,double *pe,double *dpe,double *gamma2e,double R,double nicm,double **fic,double *N,double *L_IC);
double jIC(int k,int Np,double *dp,double *Ne,double **fic);
double j_brems(double E_gamma,double n_proton,double *Ne,double *E_e,double *dpe);
double P_Brems(double n_proton,double E_gamma,double E_e);
double sigma_Brems(double E_gamma,double E_e);
double sigma_Brems_Haung97(double E_gamma,double E_e);
double Q_one(int N_energy,double *f,double *Np,double *bp,double *dp);
void read_emissivity_core(char *strpass, char *stremiss,int N_trc_core, int mpi_rank, double ***epsilon);
void output_emissivity_core(char *strpass, char *stremiss,int N_trc_core, int size_per_tracer,int mpi_rank, double **epsilon);
void outputgamma(double Eg_min,double Eg_max,double *betap,CRspectrum *CRp,double **f,double nism,double *N,double z,double *epsgamma);
void outputnu_one(char *strpass,double Enumin,double Enumax,double *bp,double *dp,double **f,double n_ISM,double R,double *N,double z,double *L_nu);
