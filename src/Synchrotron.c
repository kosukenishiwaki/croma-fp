#include <stdio.h>
#include <gsl/gsl_sf.h>
#include "params.h"
#include "CONSTANTS.h"
#include "math.h"
#include "Synchrotron.h"


void SYN_nu_crit_subB(double **nu_crit_table, double *theta, double *p){
    int j,k;
    /// critical freq. (without B) ////////
            for( j = 0; j <npe; j++){
                for( k = 0; k <tracer_synch_ntheta_pitch; k++){  // pitch angle integration //
                // B = 1 Norm //
                nu_crit_table[j][k] = SYN_crit_freq(1.0+pow(p[j],2.0), 1.0, theta[k]);                
                }
            }
    // printf("%e\n", gsl_sf_synchrotron_1(1.0e-30)); //  1.0e-10 //
}


double SYN_crit_freq(double gamma2,double B,double theta){  // Syncrotron critical frequency [Hz] //
    double m_e_g = 9.11e-28; // [g]  //
    double e = 4.8e-10;    //  [esu]  //
    // c0 = 3.0*e/(4.0*M_PI*m_e_g*c);
    double c0 = 4.194284e+06;
    return B*gamma2*c0*sin(theta);
}


void SYN_logFx_table(int Nx, double *logx, double *logFx){
    int i;
    for ( i = 0; i < Nx; i++){
        double x = pow(10.0,logx[i]);
        double F = 0.0;
        if(x > 300.0){F = sqrt(0.5*M_PI)*exp(-x)*pow(x,0.5);}
        else{F = gsl_sf_synchrotron_1(x);}
        logFx[i] = log10(F);
       // printf("%d logx = %e, logFx = %e\n",i, logx[i], logFx[i]);
    }   
}


void SYN_logy_table(double ***logy, double **nu_crit_table, int Nnu_s, double *nu_syn, double B_norm){
    int i,j,k;
    for( i = 0; i < Nnu_s; i++){
            for( j = 0; j <npe; j++){
                for( k = 0; k <tracer_synch_ntheta_pitch; k++){  // pitch angle integration //
                logy[i][j][k] =  log10(nu_syn[i]/(B_norm*nu_crit_table[j][k]));
    }}}
}

void SYN_pitch_weight_table(const double *theta, const double *dtheta, double *pitch_weight)
{
    int k;

    for (k = 0; k < tracer_synch_ntheta_pitch; k++) {
        pitch_weight[k] = sin(theta[k]) * sin(theta[k]) * dtheta[k];
    }
}

double SYN_emissivity_from_arrays(const double *dp, const double *Ne, double B,
                                  int Nx, const double *logx_tab, double xmin,
                                  const double *logFx_tab, const double *Fx_tab, double **logy,
                                  double logB, const double *pitch_weight)
{
    int j, k;
    double integral = 0.0;
    double A;
    double meg = 9.11e-28;
    double e = 4.8e-10;
    double inv_dlogx = 0.0;

    if (Nx > 1) inv_dlogx = 1.0 / (logx_tab[1] - logx_tab[0]);

    A = sqrt(3.0) * pow(e, 3.0) * B / (2.0 * meg * c * c) / (4.0 * M_PI);
    j = 2;
    while (j < npe - 1 && Ne[j + 1] > 0.0) {
        double int_th = 0.0;

        for (k = 0; k < tracer_synch_ntheta_pitch; k++) {
            double Fx = 0.0;
            double logx = logy[j][k] - logB;

            if (logx <= logx_tab[Nx - 1]) {
                int ix = (int)floor((logx - xmin) * inv_dlogx);

                if (ix < 0) {
                    Fx = 0.0;
                } else if (ix >= Nx - 1) {
                    Fx = Fx_tab[Nx - 1];
                } else {
                    const double frac = (logx - logx_tab[ix]) * inv_dlogx;
                    Fx = pow(10.0, logFx_tab[ix] + frac * (logFx_tab[ix + 1] - logFx_tab[ix]));
                }
            }
            int_th += Fx * pitch_weight[k];
        }
        integral += Ne[j] * int_th * dp[j];
        j += 1;
    }
    return A * integral;
}

double SYN_emissivity(CRspectrum *CRe,double *theta,double *dtheta,double *Ne,double B,int Nx,double *logx_tab,double xmin,double *logFx_tab,double *Fx_tab,double **logy,double logB){
    double *pitch_weight;

    pitch_weight = (double *)malloc((size_t)tracer_synch_ntheta_pitch * sizeof(double));
    if (pitch_weight == NULL) return 0.0;

    SYN_pitch_weight_table(theta, dtheta, pitch_weight);
    {
        const double emiss = SYN_emissivity_from_arrays(CRe->dp, Ne, B, Nx, logx_tab, xmin,
                                                        logFx_tab, Fx_tab, logy, logB, pitch_weight);
        free(pitch_weight);
        return emiss;
    }
}
