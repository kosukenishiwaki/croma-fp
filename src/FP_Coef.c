
/*
  FP_Coef.c
  - physical terms in Fokker-Planck
  - cooling, diffusion, reacceleration, injection, etc.

*/


#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include"params.h"
#include"CONSTANTS.h"
#include"FP_Coef.h"
#include"HADRONIC.h"
#include"fp_shared_core.h"

static double IC_kernel(double eps0,double gamma,double eps);
static double cmb_photon_density(double eps,double z);
static double d_cmb_photon_density_deps(double eps,double z);
static void cmb_photon_energy_bounds(double z,double *epsmin,double *epsmax);
static double b_Coulomb_kernel(double x);
static double b_IC(double gamma,double e0min,double e0max,double z);
static double b_IC_integrand(double eps0,double gamma,double emin,double emax,double z);
static double I_theta(double x);

typedef struct {
    int initialized;
    int crp_initialized;
    int cre_initialized;
    int crp_j_pp;
    double p_cut;
    double pe_cut;
    double crp_p0;
    double crp_pm;
    double crp_pn;
    double crp_pm1;
    double crp_pp1;
    double cre_p0;
    double cre_pm;
    double cre_pn;
    double cre_pm1;
    double cre_pp1;
    double crp_p2[np];
    double crp_exp_cut[np];
    double crp_sigma_pp[np];
    double crp_sigmoid_pp[np];
    double cre_p2[npe];
    double cre_exp_cut[npe];
    double crp_pm1_p2;
    double crp_pp1_p2;
    double crp_pm1_exp_cut;
    double crp_pp1_exp_cut;
    double cre_pm1_p2;
    double cre_pp1_p2;
    double cre_pm1_exp_cut;
    double cre_pp1_exp_cut;
} MomentumdiffGridCache;

static MomentumdiffGridCache momentumdiff_grid_cache;

static int momentumdiff_crp_grid_cache_matches(const MomentumdiffGridCache *cache,
                                               const CRspectrum *CRp,
                                               double p_cut)
{
    return cache->crp_initialized &&
           cache->p_cut == p_cut &&
           cache->crp_p0 == CRp->p[0] &&
           cache->crp_pm == CRp->p[np / 2] &&
           cache->crp_pn == CRp->p[np - 1] &&
           cache->crp_pm1 == CRp->pm1 &&
           cache->crp_pp1 == CRp->pp1;
}

static void momentumdiff_fill_crp_grid_cache(MomentumdiffGridCache *cache,
                                             const CRspectrum *CRp,
                                             double p_cut)
{
    int j;
    int j_pp = 0;
    const double k_sigmoid = 5.0;
    double p_pp;

    cache->crp_initialized = 0;
    cache->p_cut = p_cut;
    cache->crp_p0 = CRp->p[0];
    cache->crp_pm = CRp->p[np / 2];
    cache->crp_pn = CRp->p[np - 1];
    cache->crp_pm1 = CRp->pm1;
    cache->crp_pp1 = CRp->pp1;

    for (j = 0; j < np; j++) {
        cache->crp_p2[j] = CRp->p[j] * CRp->p[j];
        cache->crp_exp_cut[j] = exp(-p_cut / CRp->p[j]);
        cache->crp_sigma_pp[j] = pp_pion_cross_section(2, mp * CRp->p[j]);
    }
    for (j = 0; j < np - 100; j++) {
        if (j_pp == 0 && cache->crp_sigma_pp[j] > 0.0) j_pp = j;
    }
    j_pp += 2;
    if (j_pp < 0) j_pp = 0;
    if (j_pp >= np) j_pp = np - 1;
    cache->crp_j_pp = j_pp;
    p_pp = CRp->p[j_pp];
    for (j = 0; j < np; j++) {
        cache->crp_sigmoid_pp[j] =
            1.0 / (1.0 + exp(-k_sigmoid * (CRp->p[j] - p_pp)));
    }
    cache->crp_pm1_p2 = CRp->pm1 * CRp->pm1;
    cache->crp_pp1_p2 = CRp->pp1 * CRp->pp1;
    cache->crp_pm1_exp_cut = exp(-p_cut / CRp->pm1);
    cache->crp_pp1_exp_cut = exp(-p_cut / CRp->pp1);
    cache->crp_initialized = 1;
}

static int momentumdiff_cre_grid_cache_matches(const MomentumdiffGridCache *cache,
                                               const CRspectrum *CRe,
                                               double pe_cut)
{
    return cache->cre_initialized &&
           cache->pe_cut == pe_cut &&
           cache->cre_p0 == CRe->p[0] &&
           cache->cre_pm == CRe->p[npe / 2] &&
           cache->cre_pn == CRe->p[npe - 1] &&
           cache->cre_pm1 == CRe->pm1 &&
           cache->cre_pp1 == CRe->pp1;
}

static void momentumdiff_fill_cre_grid_cache(MomentumdiffGridCache *cache,
                                             const CRspectrum *CRe,
                                             double pe_cut)
{
    int j;

    cache->cre_initialized = 0;
    cache->pe_cut = pe_cut;
    cache->cre_p0 = CRe->p[0];
    cache->cre_pm = CRe->p[npe / 2];
    cache->cre_pn = CRe->p[npe - 1];
    cache->cre_pm1 = CRe->pm1;
    cache->cre_pp1 = CRe->pp1;

    for (j = 0; j < npe; j++) {
        cache->cre_p2[j] = CRe->p[j] * CRe->p[j];
        cache->cre_exp_cut[j] = exp(-pe_cut / CRe->p[j]);
    }
    cache->cre_pm1_p2 = CRe->pm1 * CRe->pm1;
    cache->cre_pp1_p2 = CRe->pp1 * CRe->pp1;
    cache->cre_pm1_exp_cut = exp(-pe_cut / CRe->pm1);
    cache->cre_pp1_exp_cut = exp(-pe_cut / CRe->pp1);
    cache->cre_initialized = 1;
}

static int momentumdiff_grid_cache_matches(const MomentumdiffGridCache *cache,
                                           const CRspectrum *CRp,
                                           const CRspectrum *CRe,
                                           double p_cut,
                                           double pe_cut)
{
    return cache->initialized &&
           momentumdiff_crp_grid_cache_matches(cache, CRp, p_cut) &&
           momentumdiff_cre_grid_cache_matches(cache, CRe, pe_cut);
}

static void momentumdiff_fill_grid_cache(MomentumdiffGridCache *cache,
                                         const CRspectrum *CRp,
                                         const CRspectrum *CRe,
                                         double p_cut,
                                         double pe_cut)
{
    cache->initialized = 0;
    momentumdiff_fill_crp_grid_cache(cache, CRp, p_cut);
    momentumdiff_fill_cre_grid_cache(cache, CRe, pe_cut);
    cache->initialized = 1;
}

void momentumdiff_prepare_grid_cache(const CRspectrum *CRp, const CRspectrum *CRe)
{
    const double p_cut = 1.0;
    const double pe_cut = 1.0;

    if (CRp == 0 || CRe == 0) return;
    if (momentumdiff_grid_cache_matches(&momentumdiff_grid_cache, CRp, CRe,
                                        p_cut, pe_cut)) {
        return;
    }

    #pragma omp critical(croma_momentumdiff_grid_cache)
    {
        if (!momentumdiff_grid_cache_matches(&momentumdiff_grid_cache, CRp, CRe,
                                             p_cut, pe_cut)) {
            momentumdiff_fill_grid_cache(&momentumdiff_grid_cache, CRp, CRe,
                                         p_cut, pe_cut);
        }
    }
}

void momentumdiff_prepare_crp_grid_cache(const CRspectrum *CRp)
{
    const double p_cut = 1.0;

    if (CRp == 0) return;
    if (momentumdiff_crp_grid_cache_matches(&momentumdiff_grid_cache, CRp,
                                            p_cut)) {
        return;
    }

    #pragma omp critical(croma_momentumdiff_grid_cache)
    {
        if (!momentumdiff_crp_grid_cache_matches(&momentumdiff_grid_cache, CRp,
                                                 p_cut)) {
            momentumdiff_fill_crp_grid_cache(&momentumdiff_grid_cache, CRp,
                                             p_cut);
            momentumdiff_grid_cache.initialized = 0;
        }
    }
}

void momentumdiff_prepare_cre_grid_cache(const CRspectrum *CRe)
{
    const double pe_cut = 1.0;

    if (CRe == 0) return;
    if (momentumdiff_cre_grid_cache_matches(&momentumdiff_grid_cache, CRe,
                                            pe_cut)) {
        return;
    }

    #pragma omp critical(croma_momentumdiff_grid_cache)
    {
        if (!momentumdiff_cre_grid_cache_matches(&momentumdiff_grid_cache, CRe,
                                                 pe_cut)) {
            momentumdiff_fill_cre_grid_cache(&momentumdiff_grid_cache, CRe,
                                             pe_cut);
            momentumdiff_grid_cache.initialized = 0;
        }
    }
}

const double *momentumdiff_crp_sigma_pp_cache(const CRspectrum *CRp)
{
    momentumdiff_prepare_crp_grid_cache(CRp);
    return momentumdiff_grid_cache.crp_sigma_pp;
}

const double *momentumdiff_crp_sigmoid_pp_cache(const CRspectrum *CRp)
{
    momentumdiff_prepare_crp_grid_cache(CRp);
    return momentumdiff_grid_cache.crp_sigmoid_pp;
}

int momentumdiff_crp_j_pp_cache(const CRspectrum *CRp)
{
    momentumdiff_prepare_crp_grid_cache(CRp);
    return momentumdiff_grid_cache.crp_j_pp;
}

const double *momentumdiff_cre_p2_cache(const CRspectrum *CRe)
{
    momentumdiff_prepare_cre_grid_cache(CRe);
    return momentumdiff_grid_cache.cre_p2;
}

double momentumdiff_cre_pm1_p2_cache(const CRspectrum *CRe)
{
    momentumdiff_prepare_cre_grid_cache(CRe);
    return momentumdiff_grid_cache.cre_pm1_p2;
}

double momentumdiff_cre_pp1_p2_cache(const CRspectrum *CRe)
{
    momentumdiff_prepare_cre_grid_cache(CRe);
    return momentumdiff_grid_cache.cre_pp1_p2;
}


void CR_Coef_1D(double z,double dt,double n_ISM,double n_nxt,double kBT,double B, double divv, double *rad_IC, double rad_IC_m1, double rad_IC_p1, CRspectrum *CRP, CRspectrum *CRE, FPloss *CRPloss, FPloss *CREloss){
    (void)z;
    (void)dt;
    (void)n_nxt;

    prepare_crp_losses_1d(n_ISM, kBT, divv, CRP, CRPloss);
    prepare_cre_losses_1d(n_ISM, B, divv, rad_IC, rad_IC_m1, rad_IC_p1,
                                  CRE, CREloss);
}



void momentum_bin(CRspectrum *CRp, CRspectrum *CRe,double *bp,double *gamma2e){

    double a;int i,j;
    double da=(pmax-pmin)/(double)np;
    a=pmin-da;
    CRp->pm1 = pow(10.0,a);
        for ( j = 0; j < np; j++){
            a = a+da;
            CRp->p[j]=pow(10.0,a);
            CRp->E[j]=mp*sqrt(1.0+CRp->p[j]*CRp->p[j]); ////GeV///
            bp[j]=CRp->p[j]/sqrt(1.0+CRp->p[j]*CRp->p[j]);
        }
            a = a+da;
            CRp->pp1 =pow(10.0,a);
        for ( j = 1; j < np-1; j++){
           CRp->dp[j] = 0.5*(CRp->p[j+1]-CRp->p[j-1]);
        }
        CRp->dp[0] = 0.5*(CRp->p[1]-CRp->pm1);
        CRp->dp[np-1] = 0.5*(pow(10.0,a+da)-CRp->p[np-2]);


        da=(pemax-pemin)/(double)npe;
        a=pemin-da;CRe->pm1 = pow(10.0,a);

        for ( i = 0; i <npe; i++){
            a += da;
            CRe->p[i]=pow(10.0,a);
            CRe->dp[i]=(CRe->p[i]-pow(10.0,a-da));
            gamma2e[i]=1.0+CRe->p[i]*CRe->p[i];//printf("%e %e\n",pe[i],gamma2e[i]);
            CRe->E[i]=me*sqrt(gamma2e[i]);

        }
            a += da;
            CRe->pp1 =pow(10.0,a);
        for ( j = 1; j <npe-1; j++){
           CRe->dp[j] = 0.5*(CRe->p[j+1]-CRe->p[j-1]);
        }
        CRe->dp[0] = 0.5*(CRe->p[1]-CRe->pm1);//printf("dpe[0] = %e\n",dpe[0]);
        CRe->dp[npe-1] =0.5*(pow(10.0,a+da)-CRe->p[npe-2]);
}



void Secondary_electrons(double *Qe_integral_buff,double n_ISM,double *Inje,CRspectrum *CRp,int *np_min_Qe,double **fQe,double *Qepri){
    int j,jp;
    for ( j = 0; j < npe; j++){
        Qe_integral_buff[j]=0.0;
        for (jp = np_min_Qe[j]; jp < np ; jp++){
            Qe_integral_buff[j] += 0.5*(CRp->N_ave[jp-1]*fQe[j][jp-1] + CRp->N_ave[jp]*fQe[j][jp])*CRp->dp[jp];
        }
        Inje[j] = Qe_integral_buff[j]*Gyr*n_ISM+Qepri[j];
    }
}


void Secondary_electron_1D(double n_ISM,double *Inje,double *Npave,double *dp,int *np_min_Qe,double **fQe,double **fQe_knock,double *Qepri){
    int j,jp;
    double QeIntegral[npe];
    for ( j = 0; j < npe; j++){
        QeIntegral[j]=0.0;
        for (jp = np_min_Qe[j]; jp < np ; jp++){
            QeIntegral[j] += 0.5*(Npave[jp-1]*fQe[j][jp-1]+Npave[jp]*fQe[j][jp])*dp[jp];
            QeIntegral[j] += 0.5*(Npave[jp-1]*fQe_knock[j][jp-1]+Npave[jp]*fQe_knock[j][jp])*dp[jp];
        }
        Inje[j] = QeIntegral[j]*Gyr*n_ISM+Qepri[j];
    }
}



void rad_IC_cool(int N_z, double *z, double **radIC, double *radIC_m1, double *radIC_p1, CRspectrum *CRe){

    int i,j;

    for ( i = 0; i < N_z; i++){
        double e0min,e0max;
        cmb_photon_energy_bounds(z[i],&e0min,&e0max);  // gamma2 = 1.0+p*p
        for(j = 0; j < npe; j++){
        radIC[i][j] = b_IC(sqrt(1.0+CRe->p[j]*CRe->p[j]),e0min,e0max,z[i]);
        }
        radIC_m1[i] =  b_IC(sqrt(1.0+CRe->pm1*CRe->pm1),e0min,e0max,z[i]); //if(isnan(radICm1)){printf("radIC ERROR\n");exit(1);}
        radIC_p1[i] =  b_IC(sqrt(1.0+CRe->pp1*CRe->pp1),e0min,e0max,z[i]);
    }

}


static double IC_kernel(double eps0,double gamma,double eps){
    double re=2.82e-13;
    double kappa,C=0.0;

    kappa=eps/(4.0*eps0*gamma*(gamma-eps));
    if(kappa<=0.0){C=0.0;}
    else{C=2.0*kappa*log(kappa)+(1.0+2.0*kappa)*(1.0-kappa)+pow(4.0*eps0*kappa*gamma,2.0)/(2.0*(1.0+4.0*eps0*gamma*kappa))*(1.0-kappa);
    C=2.0*M_PI*re*re*c/(gamma*gamma*eps0)*C;
    }
    return C;
}

static double cmb_photon_density(double eps,double z){
        double Tcmb,T0 = 2.757;
        double meg=9.11e-28,k=1.38e-16;
        double epsilon=1.0e-40,x;
        Tcmb=T0*(1.0+z);
        if(eps*meg*c*c/(k*Tcmb)>90.0*log(10.0)){return 0;}
        if(exp(eps*meg*c*c/(k*Tcmb))-1.0<epsilon){
            x=eps*meg*c*c/(k*Tcmb);
            return 8.0*M_PI*c*c*c*pow(meg/planck_const_erg_s,3.0)*eps/(meg*c*c*(1.0+0.5*x+x/6.0)/(k*Tcmb));

        }else
        {
            return 8.0*M_PI*c*c*c*pow(meg/planck_const_erg_s,3.0)*pow(eps,2.0)/(exp(eps*meg*c*c/(k*Tcmb))-1.0);
        }

}
static double d_cmb_photon_density_deps(double eps,double z){
        double h_PL = 6.63e-27;
        double Tcmb,T0=2.757;
        double meg=9.11e-28,k=1.38*pow(10.0,-16.0);
        Tcmb=T0*(1.0+z);
        return 8.0*M_PI*c*c*c*pow(meg/h_PL,3.0)*eps/(exp(eps*meg*c*c/(k*Tcmb))-1.0)*(2.0-eps*meg*c*c*exp(eps*meg*c*c/(k*Tcmb))/(k*Tcmb*(exp(eps*meg*c*c/(k*Tcmb))-1.0)));
}
static void cmb_photon_energy_bounds(double z,double *epsmin,double *epsmax){
        double eps,epsnew,delta=pow(10.0,-5.5),peak,epsp;
        double meg=9.11*pow(10.0,-28.0),c=3.0e+10;
        double T0=2.725,k=1.38*pow(10.0,-16.0);
        epsp=k/(meg*c*c)*T0*(1.0+z);
        peak=cmb_photon_density(epsp,z);
        epsnew=0.01*epsp;
        eps=1.0;
        int loop_count = 0;
        while(fabs(epsnew-eps)>delta*eps){
            eps=epsnew;
            epsnew=eps-(cmb_photon_density(eps,z)-peak*delta)/d_cmb_photon_density_deps(eps,z);
                loop_count+=1;
        }
        *epsmin=epsnew;
        epsnew=10.0*epsp;
        eps=1.0;loop_count = 0;
        while(fabs(epsnew-eps)>delta*eps){
            eps=epsnew;
            epsnew=eps-(cmb_photon_density(eps,z)-peak*delta)/d_cmb_photon_density_deps(eps,z);
             loop_count+=1;
        }
        *epsmax=epsnew;
}

double b_Coulomb_p(double nicm,double p,double kBT){  //  kBT in GeV  //
        const double p2 = p * p;
        const double gamma = sqrt(1.0 + p2);
        const double betap2 = p2 / (1.0 + p2);
        const double kinetic = gamma - 1.0;
        const double Ep = gamma * mp - mp;
        double xp=Ep/(kBT),xe=xp*(me/mp);  //kinetic energy//
        double Sum=b_Coulomb_kernel(xe)+b_Coulomb_kernel(xp)*(me/mp);
        double thermal = 1.0 - kBT / (2.0 * mp * kinetic);
        if(thermal<0.0){return 0.0;}
        return Gyr*c/(mp*GeV)*Sum*nicm*(1.0/betap2)*thermal*3.5e-29;
}

static double b_Coulomb_kernel(double x){
        return 0.5*sqrt(M_PI)*erf(sqrt(x))-sqrt(x)*exp(-x);
}
double b_synch(double p,double B)///////Synchrotron Cooling////////
 {
        const double meg_local = 9.11e-28;
        const double c_local = 3.0e+10;
        const double gyr_local = 3.1536e+16;
        const double b_micro = B * 1.0e+6;//////G -> mu G///////
        const double b_ratio = b_micro / 3.2;
        return 4.8e-4 * p * p * b_ratio * b_ratio * meg_local * c_local * gyr_local;
}

double b_Coulomb_e(double nicm,double pe){
        const double pe2 = pe * pe;
        double beta2=pe2/(1.0+pe2);
        double A=3.05e-29;
        return A*nicm*(1.0+log(pe/nicm)/74.8)/(me*GeV)*c*Gyr/beta2;
    }

double b_Brems_BG70(double n_p,double n_He,double pe){
        const double alpha_fine = 7.2973525693e-3;
        const double r0 = 2.8179403262e-13;
        const double n_eff = n_p + 3.0*n_He;

        if(pe <= 0.0 || n_eff <= 0.0){return 0.0;}

        double gamma2 = 1.0+pe*pe;
        double gamma = sqrt(gamma2);
        double loss_rate = 8.0*alpha_fine*c*r0*r0*n_eff*(log(2.0*gamma)+1.0/3.0);

        return loss_rate*gamma2/pe*Gyr;
    }

static double b_IC(double gamma,double e0min,double e0max,double z){
        int i,N = 128;
        double emin,emax,Integral=0.0;
        double deps0,e0,e0new,a,da,n0new,IC2,IC2new,n;

        da = 0.50e-1;
        N = round((log10(e0max)-log10(e0min))/da);

        a = log10(e0min);
        e0new = e0min;
        emin = e0new;
        emax = gamma*4.0*e0new*gamma/(1.0+4.0*e0new*gamma);
        IC2new = b_IC_integrand(e0new,gamma,emin,emax,z);
        n0new = cmb_photon_density(e0new,z);

        for ( i = 0; i < N; i++){  // CMB integral //
            a = a+da;
            e0 = e0new;
            n = n0new;
            IC2 = IC2new;
            e0new = pow(10.0,a);
            n0new = cmb_photon_density(e0new,z);
            deps0 = e0new-e0;
            emin = e0new;
            emax = gamma*4.0*e0new*gamma/(1.0+4.0*e0new*gamma);
            IC2new = b_IC_integrand(e0new,gamma,emin,emax,z);

            Integral = Integral+0.5*(n*IC2+n0new*IC2new)*deps0;
        }

        return Integral*Gyr;
}
static double b_IC_integrand(double eps0,double gamma,double emin,double emax,double z){
        int i,N = 256;
        double eps,epsnew,deps,Integral=0.0;
        double a,da,C,newC;

        da = 0.50e-1;
        N = round((log10(emax)-log10(emin))/da);
        epsnew=emin;
        a = log10(emin);
        newC = IC_kernel(eps0,gamma,epsnew);

        for ( i = 0; i < N; i++){
            a = a+da;
            eps = epsnew;
            epsnew = pow(10.0,a);
            deps = epsnew-eps;
            C = newC;
            newC = IC_kernel(eps0,gamma,epsnew);
            Integral = Integral+0.5*(eps*C+epsnew*newC)*deps;
        }

    return Integral;
}
void fill_IC_kernel(double **fic,double z,double *pe){
    double a,nuic,dnuic=(nu_max_ic-nu_min_ic)/(double)bins_IC;
    a=nu_min_ic-dnuic; int j,k;
        double gamma;
        for ( k = 0; k < bins_IC; k++){
            a=a+dnuic;
            nuic=pow(10.0,a);
            for ( j = 0; j < npe; j++){
            gamma=sqrt(1.0+pe[j]*pe[j]);
            fic[k][j]=IC_emissivity_kernel(nuic,gamma,z);
            }
        }
}
double IC_emissivity_kernel(double nu,double gamma,double z){
        int i,N=400;
        double Integral=0.0,eps;
        double meg=9.11e-28,c=3.0e+10;
        double a,da,C,Cnew,eps0,eps0new,deps0,n,n0new,e0min,e0max;

        eps = planck_const_erg_s*nu/(meg*c*c);  ///  photon energy normalized by mec^2 ////
        e0max=eps;
        e0min=eps/(4.0*gamma*(gamma-eps));
        if(e0min<0){e0min=eps/(gamma*gamma);}
        da=(log10(e0max)-log10(e0min))/(double)N;
        a=log10(e0min);
        eps0new=pow(10.0,a);
        Cnew=IC_kernel(eps0new,gamma,eps);
        n0new=cmb_photon_density(eps0new,z);
        for ( i = 0; i <N; i++)
        {
            a=a+da;
            eps0=eps0new;
            eps0new=pow(10.0,a);
            deps0=eps0new-eps0;
            C=Cnew;
            n=n0new;
            n0new=cmb_photon_density(eps0new,z);
            Cnew=IC_kernel(eps0new,gamma,eps);
            Integral=Integral+0.5*(C*n+Cnew*n0new)*deps0;
        }

        return planck_const_erg_s*eps*Integral/(4.0*M_PI);
}


double treacc_merger(double z, double M, double xi){
    double t_reacc_Norm = 0.25; // (paper II) 0.25 [Gyr] //
    double xi0 = 0.4;  // (paper II) 0.2 //
    double M0 = 1.0e+15; // (paper II) 10^15 //
    return t_reacc_Norm*pow(xi/xi0,-2.0/3.0)*(1.0+xi)/(1.0+xi0)*pow(M/M0,-1.0/3.0);
}




double F_turb(double v_turb,double rho,double L){ // [erg/s/Mpc^3] //
    return 0.5*pow(v_turb*Mpc,3.0)*rho/L;
}



double Etu_Eth(int merger_flag,double x){// fit to Vazza 2011 //  for a fixed scale //
    double A = 3.62e-2, B = 3.71e-2, C = 2.62e-1;
    if(merger_flag == 1){




        A = 0.14, B = 0.16, C = 0.24;}   // FINAL pmin 300, 1000// or "same" model //NEW !! FLAT MODEL !! // P0E41_off03, pmin1000, galinj  // TEST  R500 //   280 Myr  ??





    return A*pow(1.0+pow(x/B,2.0),C);
}






double dv_limit(double dv, double cs){
    double dv_new = dv;
    if (mach_limit > 0.0 && cs > 0.0 && dv/cs > mach_limit){
        dv_new = mach_limit*cs;
    }
    return dv_new;
}


static double I_theta(double x){
    return pow(x,4.0)/4.0+pow(x,2.0)-(1.0+2.0*x*x)*log(x)-5.0/4.0;
}


double accelerationtime_TTD(double L,double cs, double Ms, double v_turb_com){
    double x = cs/c;

    if (ttd_tacc_model == TTD_TACC_MODEL_BRUNETTI16) {
        const double t_norm_gyr = 2.5e-3;
        const double l_norm_cm = 300.0*kpc;
        const double cs_norm = 1500.0e5;
        double theta = I_theta(x);

        (void)v_turb_com;
        if (!(x > 0.0) || !(theta > 0.0) || !(Ms > 0.0) ||
            !(L > 0.0) || !(cs > 0.0)) {
            return 0.0;
        }
        return t_norm_gyr/(x*theta)*pow(Ms/0.5,-4.0)*(L/l_norm_cm)*
               pow(cs/cs_norm,-1.0);
    }

    double kL = 2.0*M_PI/L;  // [cgs] //
    double k_cut_th = 1.04e+4*pow(Ms,4.0)*kL;  //collisionless TTD [cm^-1] //
    double k_cut = k_cut_th;
    double kWk_L = 0.5*pow(v_turb_com,2.0);
    double kWk_cut = kWk_L*pow(k_cut/kL,-0.5);  //  IK spectrum [cm^2/s^2]   //
    double tacc = c/M_PI/I_theta(x)/k_cut/kWk_cut/Gyr;
    return tacc;  // TTD (collisioinless) // [Gyr]
}

double accelerationtime_ASA(double L,double cs,double Ms,double beta_pl){
    double tacc = sqrt(6.0/5.0)/12.0*(c/cs/cs)*L/sqrt(beta_pl)/pow(Ms,3.0)*pow(psi,3.0)/Gyr;
    return tacc; // solenidal turbulence // [Gyr]
}

static double momentumdiff_effective_mach(double dv_imc, double cs)
{
    double mach;

    if (cs <= 0.0) return 0.0;

    mach = dv_imc / cs;
    if (mach < 0.0) mach = 0.0;
    if (mach_limit > 0.0 && mach > mach_limit) mach = mach_limit;
    return mach;
}

static void momentumdiff_fill_from_tacc(double t_acc,
                                        CRspectrum *CRp,
                                        CRspectrum *CRe,
                                        double *Dpp,
                                        double *Dppm1,
                                        double *Dppp1,
                                        double *Dppe,
                                        double *Dppem1,
                                        double *Dppep1,
                                        double Epmax)
{
    const double q = 2.0;
    const double inv_q_tacc = 1.0 / ((q + 2.0) * t_acc);
    const MomentumdiffGridCache *cache;
    int j;

    momentumdiff_prepare_grid_cache(CRp, CRe);
    cache = &momentumdiff_grid_cache;

    for (j = 0; j < np; j++) {
        Dpp[j] = cache->crp_p2[j] * inv_q_tacc *
                 exp(-CRp->E[j] / Epmax) * cache->crp_exp_cut[j];
    }
    *Dppp1 = cache->crp_pp1_p2 * inv_q_tacc *
             exp(-mp * sqrt(1.0 + CRp->pp1 * CRp->pp1) / Epmax) *
             cache->crp_pp1_exp_cut;
    *Dppm1 = cache->crp_pm1_p2 * inv_q_tacc *
             exp(-mp * sqrt(1.0 + CRp->pm1 * CRp->pm1) / Epmax) *
             cache->crp_pm1_exp_cut;

    for (j = 0; j < npe; j++) {
        Dppe[j] = cache->cre_p2[j] * inv_q_tacc * cache->cre_exp_cut[j];
        if (isnan(Dppe[j])) {
            fprintf(stderr, "ERROR Dppe NAN: %e %e\n", CRe->p[j], t_acc);
        }
    }
    *Dppep1 = cache->cre_pp1_p2 * inv_q_tacc * cache->cre_pp1_exp_cut;
    *Dppem1 = cache->cre_pm1_p2 * inv_q_tacc * cache->cre_pm1_exp_cut;
}

static void momentumdiff_fill_direct_tacc(double t_acc,
                                          CRspectrum *CRp,
                                          CRspectrum *CRe,
                                          double *Dpp,
                                          double *Dppm1,
                                          double *Dppp1,
                                          double *Dppe,
                                          double *Dppem1,
                                          double *Dppep1,
                                          double Epmax)
{
    const double inv_4_tacc = 1.0 / (4.0 * t_acc);
    const MomentumdiffGridCache *cache;
    int j;

    momentumdiff_prepare_grid_cache(CRp, CRe);
    cache = &momentumdiff_grid_cache;

    for (j = 0; j < np; j++) {
        Dpp[j] = cache->crp_p2[j] * inv_4_tacc *
                 exp(-CRp->E[j] / Epmax) * cache->crp_exp_cut[j];
    }
    *Dppp1 = cache->crp_pp1_p2 * inv_4_tacc *
             exp(-mp * sqrt(1.0 + CRp->pp1 * CRp->pp1) / Epmax) *
             cache->crp_pp1_exp_cut;
    *Dppm1 = cache->crp_pm1_p2 * inv_4_tacc *
             exp(-mp * sqrt(1.0 + CRp->pm1 * CRp->pm1) / Epmax) *
             cache->crp_pm1_exp_cut;

    for (j = 0; j < npe; j++) {
        Dppe[j] = cache->cre_p2[j] * inv_4_tacc * cache->cre_exp_cut[j];
        if (isnan(Dppe[j])) {
            fprintf(stderr, "ERROR Dppe NAN: %e %e\n", CRe->p[j], t_acc);
        }
    }
    *Dppep1 = cache->cre_pp1_p2 * inv_4_tacc * cache->cre_pp1_exp_cut;
    *Dppem1 = cache->cre_pm1_p2 * inv_4_tacc * cache->cre_pm1_exp_cut;
}

void momentumdiff_asa_1D(double L,double dv_imc,double cs,double beta_pl,CRspectrum *CRp, CRspectrum *CRe,double *Dpp,double *Dppm1,double *Dppp1,double *Dppe,double *Dppem1,double *Dppep1,double Epmax){
    const double mach = momentumdiff_effective_mach(dv_imc, cs);
    double t_acc_ASA;

    if (mach <= 0.0 || beta_pl <= 0.0) {
        momentumdiff_off(CRp, CRe, Dpp, Dppm1, Dppp1, Dppe, Dppem1, Dppep1);
        return;
    }

    t_acc_ASA = accelerationtime_ASA(L * Mpc, cs, mach, beta_pl);
    momentumdiff_fill_from_tacc(t_acc_ASA, CRp, CRe,
                                Dpp, Dppm1, Dppp1,
                                Dppe, Dppem1, Dppep1, Epmax);
}

void momentumdiff_ttd_1D(double L,double dv_imc,double cs,double beta_pl,CRspectrum *CRp, CRspectrum *CRe,double *Dpp,double *Dppm1,double *Dppp1,double *Dppe,double *Dppem1,double *Dppep1,double Epmax){
    const double mach = momentumdiff_effective_mach(dv_imc, cs);
    double t_acc_TTD;

    (void)beta_pl;

    if (mach <= 0.0 || dv_imc <= 0.0) {
        momentumdiff_off(CRp, CRe, Dpp, Dppm1, Dppp1, Dppe, Dppem1, Dppep1);
        return;
    }

    t_acc_TTD = accelerationtime_TTD(L * Mpc, cs, mach, dv_imc);
    momentumdiff_fill_from_tacc(t_acc_TTD, CRp, CRe,
                                Dpp, Dppm1, Dppp1,
                                Dppe, Dppem1, Dppep1, Epmax);
}

void momentumdiff_direct_tacc_1D(double L,double dv_imc,double cs,double beta_pl,CRspectrum *CRp, CRspectrum *CRe,double *Dpp,double *Dppm1,double *Dppp1,double *Dppe,double *Dppem1,double *Dppep1,double Epmax){
    (void)L;
    (void)dv_imc;
    (void)cs;
    (void)beta_pl;

    if (!(t_acc_direct_gyr > 0.0) || !isfinite(t_acc_direct_gyr)) {
        momentumdiff_off(CRp, CRe, Dpp, Dppm1, Dppp1, Dppe, Dppem1, Dppep1);
        return;
    }

    momentumdiff_fill_direct_tacc(t_acc_direct_gyr, CRp, CRe,
                                  Dpp, Dppm1, Dppp1,
                                  Dppe, Dppem1, Dppep1, Epmax);
}

void momentumdiff_1D(double L,double dv_imc,double cs,double beta_pl,CRspectrum *CRp, CRspectrum *CRe,double *Dpp,double *Dppm1,double *Dppp1,double *Dppe,double *Dppem1,double *Dppep1,double Epmax){
    switch (resolve_momentumdiff_model(FP_MOMENTUMDIFF_MODEL_AUTO)) {
    case FP_MOMENTUMDIFF_MODEL_ASA:
        momentumdiff_asa_1D(L, dv_imc, cs, beta_pl, CRp, CRe,
                            Dpp, Dppm1, Dppp1, Dppe, Dppem1, Dppep1, Epmax);
        break;
    case FP_MOMENTUMDIFF_MODEL_DIRECT_TACC:
        momentumdiff_direct_tacc_1D(L, dv_imc, cs, beta_pl, CRp, CRe,
                                    Dpp, Dppm1, Dppp1, Dppe, Dppem1, Dppep1, Epmax);
        break;
    case FP_MOMENTUMDIFF_MODEL_TTD:
        momentumdiff_ttd_1D(L, dv_imc, cs, beta_pl, CRp, CRe,
                            Dpp, Dppm1, Dppp1, Dppe, Dppem1, Dppep1, Epmax);
        break;
    case FP_MOMENTUMDIFF_MODEL_OFF:
    default:
        momentumdiff_off(CRp, CRe, Dpp, Dppm1, Dppp1, Dppe, Dppem1, Dppep1);
        break;
    }
}

void momentumdiff_off(CRspectrum *CRp, CRspectrum *CRe,double *Dpp,double *Dppm1,double *Dppp1,double *Dppe,double *Dppem1,double *Dppep1){
    int j;
    const MomentumdiffGridCache *cache;
    double t_acc = 1.0e+5; // [Gyr]
    const double inv_4_tacc = 1.0 / (4.0 * t_acc);

    momentumdiff_prepare_grid_cache(CRp, CRe);
    cache = &momentumdiff_grid_cache;

    for (j = 0; j < np; j++) {
        Dpp[j] = cache->crp_p2[j] * inv_4_tacc;
    }
    *Dppp1 = cache->crp_pp1_p2 * inv_4_tacc;
    *Dppm1 = cache->crp_pm1_p2 * inv_4_tacc;

    for (j = 0; j < npe; j++) {
        Dppe[j] = cache->cre_p2[j] * inv_4_tacc;
    }
    *Dppep1 = cache->cre_pp1_p2 * inv_4_tacc;
    *Dppem1 = cache->cre_pm1_p2 * inv_4_tacc;

}

double Drr_pitch(double p,double B,double lc,double m,double Z){///[Mpc^2/Gyr]///
    double A = 6.8e+29;
    double beta=sqrt(p*p/(1.0+p*p));
    B = B*1.0e+6;  /////mu G////////
    double rL = 3.3e+12*p*m/B/Z/Mpc; /// Larmor radius [Mpc] ///
    double D;
    if(rL < lc){D = A*pow(lc/0.100,2.0/3.0)*pow(m*p,1.0/3.0)*beta*pow(B*Z,-1.0/3.0)/Mpc/Mpc*Gyr;}
    else{D = 1.0/3.0*c*rL*rL/lc/Mpc*Gyr;}// semi-diffusive //
    return D;
}


double Drr_lA(double lA){///[Mpc^2/Gyr]///
    return 1.0/3.0*c*psi*lA*Gyr/Mpc;
}


double Drr_test(double r,double p){
    (void)r;
    (void)p;
    return 1.0e-5;
}


double adiabatic_V(double r,double dr,double r_pre,double dr_pre,double dt,double p){
    double V  = 4.0*M_PI*pow(r_pre,2.0)*dr_pre;
    double dVdt = 4.0*M_PI*(pow(r,2.0)*dr-pow(r_pre,2.0)*dr_pre)/dt;
    return 1.0/3.0*dVdt/V*p;
}

double adiabatic_n(double n,double n_pre,double dt,double p){
    double dndt = (n-n_pre)/dt;  // double C = 0.0;
    return -1.0/3.0*dndt/(0.5*(n_pre+n))*p;
}


double adiabatic_divv(double divv,double p){
    return 1.0/3.0*divv*p;
}














int Hillas(int N_Emax,int N_energy,double *E,double Z,double B,double R){
    double e = 4.8e-10;  // [esu] //
    int i = N_Emax;double Emax = e*B*R*Z/GeV; // [GeV] //
    while(E[i]<Emax && i<N_energy-1){i+=1;}
    return i;
}






double c_sound(double T){  // [cm/s] //
    double mproton = 0.938; /// [GeV]  ///  kB in [eV/K] //
    return sqrt(5.0/3.0*kB*T/(mu_mol*mproton*1.0e+9))*c;
}

double l_Alfven(double beta_pl,double L,double M_s){   // Alfven scale [Mpc], Kolmogorov,  l_A = L_0 * M_A^-3  //
    return pow(6.0/5.0,3.0/2.0)*L/pow(beta_pl,3.0/2.0)/pow(M_s,3.0);
}
double v_Alfven(double B,double rho){  //  [cm/s]  //
    return B/sqrt(4.0*M_PI*rho);
}


double B_dynamo(double rho, double dv){ //  c.g.s  //
    return sqrt(4.0*M_PI*eta_B*rho*dv*dv);
}
