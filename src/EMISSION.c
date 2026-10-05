#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Synchrotron.h"
#include "EMISSION.h"
#include "CONSTANTS.h"
#include "COSFUNC.h"
#include "params.h"

void outputSyn_Kernel(int Nx,double *Fx_tab,double *logFx_tab,double *logx,double xmin,double ***logy,double z,int Nnu,double numin,double numax,
    CRspectrum *CRe,double *N,double *gamma2e,double B,double logB,double *epsSyn, double *theta, double *dtheta, double *nus){
        int k;
        double pitch_weight[N_theta_pitch];

        (void)z;
        (void)numin;
        (void)numax;
        (void)gamma2e;
        (void)nus;

        SYN_pitch_weight_table(theta, dtheta, pitch_weight);
        for (k = 0; k < Nnu; k++) {
            epsSyn[k] = SYN_emissivity_from_arrays(CRe->dp, N, B, Nx, logx, xmin,
                                                   logFx_tab, Fx_tab, logy[k], logB, pitch_weight);
        }
}


void read_emissivity_core(char *strpass, char *stremiss,int N_trc_core, int mpi_rank, double ***epsilon)
{
    FILE *fp; int i;
    char str[MAX_LINE_LENGTH];
    sprintf(str,"%s/%s_core%02d.bin",strpass,stremiss,mpi_rank);
    fp = fopen(str,"rb"); if(fp == NULL){printf("FILE OPEN ERROR !!! %s \n", str); exit(1);}

    *epsilon = (double**)malloc(N_trc_core * sizeof(double*));
    if (*epsilon == NULL) {
        printf("Memory allocation error for epsilon\n");
        exit(1);
    }

    size_t total_size = (size_t)N_trc_core * nsnp_f * MAX_SYNCH_FREQ_BINS;
    double *dumm = (double*)malloc(total_size * sizeof(double));

    for ( i = 0; i < N_trc_core; i++) {
        (*epsilon)[i] = dumm + i * nsnp_f * MAX_SYNCH_FREQ_BINS;
    }

    fread(dumm, sizeof(double), total_size, fp);

    fclose(fp);
}

void output_emissivity_core(char *strpass, char *stremiss,int N_trc_core, int size_per_tracer,int mpi_rank, double **epsilon){
    FILE *fp;
    char str[MAX_LINE_LENGTH];
    size_t total_size;
    double *packed;
    sprintf(str,"%s/%s_core%02d.bin",strpass,stremiss,mpi_rank);

    fp = fopen(str,"wb"); if (fp == NULL){printf("Cannot open file %s\n", str); exit (1);}

    total_size = (size_t)N_trc_core * (size_t)size_per_tracer;
    packed = (double *)malloc(total_size * sizeof(double));
    if (packed == NULL) {
        fprintf(stderr, "rank %d: packed emissivity allocation failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (size_t i = 0; i < (size_t)N_trc_core; ++i) {
        if (!epsilon[i]) {
            fprintf(stderr,"rank %d: epsilon[%zu] is NULL\n",
                    mpi_rank, i);
            free(packed);
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        memcpy(packed + i * (size_t)size_per_tracer, epsilon[i],
               (size_t)size_per_tracer * sizeof(double));
    }

    if (fwrite(packed, sizeof(double), total_size, fp) != total_size) {
        fprintf(stderr, "rank %d: fwrite failed for %s\n", mpi_rank, str);
        free(packed);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    free(packed);
    fclose(fp);
}

void outputgamma(double Eg_min,double Eg_max,double *betap,CRspectrum *CRp,double **f,double nism,double *N,double z,double *epsgamma)
{
    int j,k;
    double a;
    double dEgamma = (Eg_max-Eg_min)/(double)bins_gamma;
    double Np[np];

    (void)z;

    if (epsgamma == NULL) {
        return;
    }

    for ( j = 0; j < np;j++){
        Np[j] = N[j];
    }

    a = Eg_min-dEgamma;
    for ( k = 0; k < bins_gamma; k++){
        double Eg;

        a += dEgamma;
        Eg = pow(10.0,a);
        epsgamma[k] = Eg*Q_one(np,f[k],Np,betap,CRp->dp)*nism;
    }
}

void outputIC_one(char *strpass,double z,double numin,double numax,double *pe,double *dpe,double *gamma2e,double R,double nicm,double **fic,double *N,double *L_IC)
{
    FILE *fp;
    char strIC[MAX_LINE_LENGTH];
    int j,k;
    double a,nu_ic[bins_IC],dnu=(double)(numax-numin)/(double)bins_IC;
    double Ne[npe];
    double emiss[bins_IC],Intensity[bins_IC],L[bins_IC],F[bins_IC];
    double E_ic[bins_IC];

    (void)pe;
    (void)gamma2e;
    (void)nicm;

    sprintf(strIC,"%s/FIC_z%.2f.txt",strpass,z);
    fp=fopen(strIC,"w");

    for ( j = 0; j < npe;j++){
        Ne[j] = N[j]/(4.0*M_PI*pow(R*Mpc,3.0)/3.0);
    }

    a = (double)numin-dnu;
    for ( k = 0; k < bins_IC; k++){
        a = a+dnu;
        nu_ic[k] = pow(10.0,a);
        E_ic[k] = planck_const_erg_s*nu_ic[k]/GeV;
    }

    for ( k = 0; k < bins_IC; k++){
        emiss[k] = jIC(k,npe,dpe,Ne,fic);
        Intensity[k] = emiss[k]*R*Mpc;
        L[k] = 4.0*M_PI*(4.0*M_PI*pow(R*Mpc,3.0)/3.0)*emiss[k];
        F[k] = L[k]/(4.0*M_PI*luminosity_distance_mpc(z)*luminosity_distance_mpc(z)*Mpc*Mpc);

        fprintf(fp,"%e %e %e %e %e %e\n",nu_ic[k],nu_ic[k]/(1.0+z),E_ic[k],F[k],nu_ic[k]*F[k],Intensity[k]);
    }

    fclose(fp);

    double L_dum = 0.0;
    for ( k = 0; k < bins_IC-1; k++){
        L_dum += L[k]*(nu_ic[k+1] - nu_ic[k]);
    }

    *L_IC = L_dum;
}

double jIC(int k,int Np,double *dp,double *Ne,double **fic)
{
    int j;
    double Integral=0.0;

    for ( j = 0; j < Np; j++){
        Integral += fic[k][j]*Ne[j]*dp[j];
    }
    return Integral;
}

double j_brems(double E_gamma,double n_proton,double *Ne,double *E_e,double *dpe)
{
    int j;
    double Integral = 0.0;

    for ( j = 0; j < npe; j++){
        Integral += P_Brems(n_proton,E_gamma,E_e[j])*Ne[j]*dpe[j];
    }
    return 1.0/(4.0*M_PI)*Integral;
}

double P_Brems(double n_proton,double E_gamma,double E_e)
{
    double p_over_mc;
    double beta_e;

    if (n_proton <= 0.0 || E_gamma <= 0.0 || E_e <= me) {
        return 0.0;
    }

    p_over_mc = sqrt(E_e/me*E_e/me-1.0);
    beta_e = sqrt(1.0-1.0/pow(E_e/me,2.0));

    if (p_over_mc > 1.0){
        return beta_e*c*n_proton*E_gamma*sigma_Brems(E_gamma,E_e);
    }
    return beta_e*c*n_proton*E_gamma*sigma_Brems_Haung97(E_gamma,E_e);
}

double sigma_Brems(double E_gamma,double E_e)
{
    double sigmaB = 0.0;
    if (E_gamma > 0.0 && E_e - E_gamma >= me){
        double phi0 = 4.0*(log(2.0*(E_e/me)*((E_e-E_gamma)/E_gamma))-0.5);
        if (phi0<0.0){phi0 = 0.0;}
        double phi1 = phi0, phi2 = phi0;
        double Phi = (1.0+pow(1.0-E_gamma/E_e,2.0))*phi1 - 2.0/3.0*(1.0-E_gamma/E_e)*phi2;
        sigmaB = 1.0/E_gamma*(3.0/(8.0*M_PI))*(1.0/137.0)*sigma_Thomson*Phi;
    }
    return sigmaB;
}

double sigma_Brems_Haung97(double E_gamma,double E_e)
{
    double sigmaB = 0.0;
    double eps1 = E_e/me,eps2 = (E_e - E_gamma)/me;
    double k = E_gamma/me;

    if (eps1 > 1.0 && eps2 >= 1.0 && k > 0.0 && log(2.0*eps1*eps2/k)>0.5){
        double p1 = sqrt(eps1*eps1-1.0), p2 = sqrt(eps2*eps2-1.0);
        double A = 4.0/3.0*eps1*eps2+k*k-7.0/15.0*k*k/(eps1*eps2)-11.0/70.0*k*k*(p1*p1+p2*p2)/(pow(eps1*eps2,4.0));
        double B = 2.0*log((eps1*eps2+p1*p2-1.0)/k) - p1*p1/(eps1*eps2)*(1.0+1.0/(eps1*eps2)+7.0/20.0*(p1*p1+p2*p2)/(pow(eps1*eps2,3.0))+(9.0/28.0*k*k+263.0/210.0*pow(p1*p2,2.0))*1.0/pow(eps1*eps2,3.0));
        sigmaB  = 3.0*sigma_Thomson/(4.0*M_PI)*(1.0/137.0)/(k*p1*p1)*A*B;
    }
    return sigmaB/me;
}




void outputnu_one(char *strpass,double Enumin,double Enumax,double *bp,double *dp,double **f,double n_ISM,double R,double *N,double z,double *L_nu){
        char strnu[MAX_LINE_LENGTH];
        sprintf(strnu,"%s/Fnu_z%.2f.txt",strpass,z);

        FILE *fp;
        int j,k;
        double a,nu=0.0,dE = (double)(Enumax-Enumin)/bins_nu;
        double emiss[bins_nu],Intensity[bins_nu];
        double Np[np];
        double F[bins_nu],Fo[bins_nu],E=0.0,L[bins_nu];
        a=(double)Enumin-dE;

        double DL = luminosity_distance_mpc(z);

        fp=fopen(strnu,"w");


        for ( j = 0; j < np;j++){
            Np[j] = N[j]/(4.0*M_PI*pow(R*Mpc,3.0)/3.0);
        }

        a=(double)Enumin-dE;
        for ( k = 0; k < bins_nu; k++){
            a = a+dE;
            E = pow(10.0,a);
            nu = E/planck_const_erg_s*GeV;

            emiss[k] = E*Q_one(np,f[k],Np,bp,dp)*n_ISM;

            Intensity[k] = emiss[k]*R*Mpc;

            L[k] = 4.0*M_PI*(4.0*M_PI*pow(R*Mpc,3.0)/3.0)*emiss[k];

            F[k] = L[k]/(4.0*M_PI*DL*DL*Mpc*Mpc);
            Fo[k] = L[k]/(4.0*M_PI*DL*DL*Mpc*Mpc)*(1.0+z);

            F[k] = F[k]*GeV;
            Fo[k] = Fo[k]*GeV;

            fprintf(fp,"%e %e %e %e %e %e %e %e\n",nu,E,E/(1.0+z),F[k],E*F[k],Fo[k],L[k],Intensity[k]);

        }
        fclose(fp);

        double L_dum = 0.0; a=(double)Enumin-dE; E = pow(10.0,a);nu = E*GeV/planck_const_erg_s;
        double E_pre = E;
        for ( k = 0; k < bins_gamma-1; k++){
            a = a+dE;
            E = pow(10.0,a);
            nu = E/planck_const_erg_s*GeV;
            if(E > 1.0e+3){
                L_dum += L[k]*GeV*(E-E_pre);
            }
            E_pre = E;
        }
        *L_nu = L_dum;
}


double Q_one(int N_energy,double *f,double *Np,double *bp,double *dp){
    double Integral=0.0,fn,fpre;
    int i;
    fpre=bp[0]*Np[0]*f[0];
    for ( i = 1; i < N_energy; i++)
    {
        fn=bp[i]*Np[i]*f[i];
        Integral+=(fpre+fn)*dp[i];
        fpre=fn;
    }
    return Integral*0.5;
}
