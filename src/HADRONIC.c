/*
   HADRONIC.c
   - tables and kernels for the hadronic pp chain
   - refs: Kelner+06, Kamae+06, Blattnig 2000, Brunetti & Blasi 05
*/


#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<complex.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_sf_synchrotron.h>
#include<gsl/gsl_integration.h>
#include<gsl/gsl_sf.h>
#include"HADRONIC.h"
#include"CONSTANTS.h"
#include"params.h"
#include"FP_Coef.h"

void Kernels(int nsnp_i, int nsnp_f, int bins,double **fga,double **fnu,double ***fic,double *z,double *p,double *pe){
    int i,j,k;

    double dEgamma=(E_gamma_max-E_gamma_min)/(double)bins_gamma;
    double a,nuic,dnuic = (nu_max_ic-nu_min_ic)/(double)bins_IC;
    //////////////////gamma////////////////////////////////
        double Egamma;
        a = E_gamma_min-dEgamma;
        for ( k = 0; k < bins_gamma ; k++){
            a += dEgamma;
            Egamma = pow(10.0,a);

            if (!fga || !fga[k]) {
                    fprintf(stderr,"NULL row: fga[%d]\n", k);
            }

            for ( j = 0; j < bins; j++){
                if (k >= bins_gamma || j >= bins) {
                    fprintf(stderr,"OOB k=%d/%d  j=%d/%d\n",
                            k, bins_gamma-1, j, bins-1);
                }

                //printf("%e \n",Egamma);
            fga[k][j]= gammaKernel(Egamma,p[j]); ///[/GeV*sec/str]//
            }
        }
    ////////////////////////////////////////////////////

    ///////////////////neutrino///////////////////////////////////////
    double dEnu=(E_nu_max-E_nu_min)/(double)bins_nu;
    double Enu;
        a = E_nu_min-dEnu;
        for ( k = 0; k < bins_nu; k++){
            a += dEnu;
            Enu=pow(10.0,a);
            for ( j = 0; j < bins; j++){
                    /// unit  [/cm3/sec/str/GeV]///
                    /////////// elecrton neutrinos////// muon neutrinos ///// muon neutrinos  ////////
                    fnu[k][j]= eKernel(HADRON_LEPTON_NUMU,Enu,p[j])+eKernel(HADRON_LEPTON_NUE,Enu,p[j])+numu1Kernel(Enu,p[j]);
                    // only mu neutrinos ///
                    //fnumu[i][j]=QeKernel(HADRON_LEPTON_NUMU,Enu,p[j])+Qnumu1Kernel(Enu,p[j]);
            }
        }
    ////////////////////////////////////////////////

    for ( i = 0; i < nsnp_f; i++){
        if(i>= nsnp_i){
                //////////////// CMB IC  fIC=nCMB*Kernel///////////////////////////
                    a = nu_min_ic-dnuic;
                    double gamma;
                    for ( k = 0; k < bins_IC; k++){
                        a = a+dnuic;
                        nuic=pow(10.0,a);
                        for ( j = 0; j < npe; j++){
                        gamma=sqrt(1.0+pe[j]*pe[j]);
                        fic[i][k][j]=IC_emissivity_kernel(nuic,gamma,z[i]);
                        //printf("%e\n",fic[i][j]);
                        }
                    }
            ////////////////////////////////////////////////////////////////
        }
        else{
            for ( k = 0; k < bins_IC; k++){
                for ( j = 0; j < npe; j++){fic[i][k][j]=0.0;}
            }

        }
    }
}
void Kernels_z(int bins,double **fga,double **fnu,double **fic,double z,double *p,double *pe){
    int i,j,k;
    double dEgamma=(E_gamma_max-E_gamma_min)/(double)bins_gamma;
    double a;
    //////////////////gamma////////////////////////////////
        double Egamma;
        a = E_gamma_min-dEgamma;
        for ( k = 0; k < bins_gamma ; k++){
            a += dEgamma;
            Egamma = pow(10.0,a);
            for ( j = 0; j < bins; j++){
                if (k >= bins_gamma || j >= bins) {
                    fprintf(stderr,"OOB k=%d/%d  j=%d/%d\n",
                            k, bins_gamma-1, j, bins-1);
                }
                if (!fga || !fga[k]) {
                    fprintf(stderr,"NULL row: fga[%d]\n", k);
                }
                //printf("%e \n",Egamma);
            fga[k][j]= gammaKernel(Egamma,p[j]); ///[/GeV*sec/str]//
            }
        }
    ////////////////////////////////////////////////////

    ///////////////////neutrino///////////////////////////////////////
        double dEnu=(E_nu_max-E_nu_min)/(double)bins_nu;
       double Enu;
        a = E_nu_min-dEnu;
        for ( i = 0; i < bins_nu; i++){
            a += dEnu;
            Enu=pow(10.0,a);
            for ( j = 0; j < bins; j++){
                    /// unit  [/cm3/sec/str/GeV]///
                    /////////// elecrton neutrinos////// muon neutrinos ///// muon neutrinos  ////////
                    fnu[i][j]= eKernel(HADRON_LEPTON_NUMU,Enu,p[j])+eKernel(HADRON_LEPTON_NUE,Enu,p[j])+numu1Kernel(Enu,p[j]);
                    // only mu neutrinos ///
                    //fnumu[i][j]=QeKernel(HADRON_LEPTON_NUMU,Enu,p[j])+Qnumu1Kernel(Enu,p[j]);
            }
        }
    ////////////////////////////////////////////////
     //printf("nu\n");

    fill_IC_kernel(fic, z, pe);
    //printf("Ic\n");
}


////////  GAMMA-RAY emission Kernel  ////////////////////////////
//////// Kelner Eq.(14) ////
double gammaKernel(double Egamma,double p){
    int i;////////////////GeV//////////////////
    ///double h2=4.14e-24;//////Planck const in [GeVsec] ////////////
    double gp=sqrt(p*p+1.0),Ep=gp*mp;
    double Emax = Ep,Emin=Egamma+pion0_mass_gev*pion0_mass_gev/(4.0*Egamma);  /// minimum pion energy for E_gamma   ///

        if (Emax<Emin){return 0;}
        if (Emin<pion0_mass_gev){Emin=pion0_mass_gev;} /// minimum energy for pi^0 production ///
        if((Ep-mp)<HADRON_GAMMA_EMISSION_THRESHOLD_GEV){return 0;} /// minimum energy for gamma emission///


    double dEpi,Epi,Integral = 0.0;
    int Na = HADRON_KERNEL_EPI_STEPS;
    double a,amin = log10(Emin),amax = log10(Emax),da=(amax-amin)/(double)Na;

    double f,fpre,Epipre;

    a = amin; /////Epi=Emin///
    Epi = pow(10.0,a);////////////////////////
    f = dsigma(0,Epi,Ep)/sqrt(Epi*Epi-pion0_mass_gev*pion0_mass_gev);
    //printf("%e %e %e %e\n",dsigma(0,Epi,Ep),sqrt(Epi*Epi-mpi*mpi),Ep,Epi);

    for ( i = 1; i < Na; i++){    ////  Integrate over pion energy //////
        Epipre=Epi;
        fpre=f;

        a += da;
        Epi = pow(10.0,a);
        dEpi = Epi-Epipre;

        f = dsigma(0,Epi,Ep)/sqrt(Epi*Epi-pion0_mass_gev*pion0_mass_gev);
        Integral += 0.5*(f+fpre)*dEpi;
    }

    return 2.0*c*Integral/(4.0*M_PI); /////////[/GeV/sec/cm3/str]/////////
}

////////  Electron & Neutrino Kernel /////////////
///////HADRON_LEPTON_ELECTRON electron///HADRON_LEPTON_NUE nue /////HADRON_LEPTON_NUMU numu/////////
///////   Kelner eq.(52), (53) /////
double eKernel(HadronLeptonKernel lepton_kind,double E,double p){
    if(lepton_kind != HADRON_LEPTON_ELECTRON &&
       lepton_kind != HADRON_LEPTON_NUE &&
       lepton_kind != HADRON_LEPTON_NUMU){
        fprintf(stderr,"eKernel: invalid lepton kind %d\n", (int)lepton_kind);
        return 0.0;
    }

    double ge=E/me,be=sqrt(1.0-1.0/(ge*ge)),gp=sqrt(p*p+1.0),Ep=gp*mp;
    double Emax=gp*mp,Emin=2.0*E*pion_charged_mass_gev*pion_charged_mass_gev/(pion_charged_mass_gev*pion_charged_mass_gev+muon_mass_gev*muon_mass_gev);   ////  minimum enrgy for pion  ////

    double Epi,Integral=0.0;
    double x,dx=1.0/(double)HADRON_ELECTRON_X_STEPS;

        if (Emax<Emin){return 0;}
        if (Emin<pion_charged_mass_gev){Emin=pion_charged_mass_gev;}
        if((Ep-mp)<HADRON_ELECTRON_EMISSION_THRESHOLD_GEV){return 0;}

    double f,fpre;
    int i;

    x=0.0;
    //Epi=E/x;
    f=0.0;

    for ( i = 1; i < HADRON_ELECTRON_X_STEPS; i++){
        x += dx;
        Epi = E/x;
        fpre = f;
        if(lepton_kind==HADRON_LEPTON_NUE){f=dsigma(1,Epi,Ep)*fnue(x)/x;}  // electron neutrino ///
        else{f=dsigma(1,Epi,Ep)*fe(x)/x;}  // electron & muon neutrino //

        if (i==1){
            Integral += f*dx;
        }
        else{
            Integral += 0.5*(f+fpre)*dx;
        }
    }

    double Integral_2 = 0.0;
    x=0.0; //Epi=E/x;
    f=0.0;
    for ( i = 1; i < HADRON_ELECTRON_X_STEPS; i++){
        x += dx;
        Epi = E/x;
        fpre = f;
        if(lepton_kind==HADRON_LEPTON_NUE){f=dsigma(-1,Epi,Ep)*fnue(x)/x;}  // electron neutrino ///
        else{f=dsigma(-1,Epi,Ep)*fe(x)/x;}  // electron & muon neutrino //

        if (i==1){
            Integral_2 += f*dx;
        }
        else{
            Integral_2 += 0.5*(f+fpre)*dx;
        }
    }

       if(lepton_kind==HADRON_LEPTON_ELECTRON){return 2.0*c*me*be*Integral;} ////(nth*c)/(GeV/c)//
    else{return c*(Integral+Integral_2)/(4.0*M_PI);}    ///////2 denotes pi+ and pi-///////////
}

///////  muon Neutrino Kernel only from pion decay ///
////////   Kelner Eq.(27)  ///////
double numu1Kernel(double Enu,double p){
    double lambda=1.0-muon_mass_gev*muon_mass_gev/(pion_charged_mass_gev*pion_charged_mass_gev);
    double gp=sqrt(p*p+1.0),Ep=gp*mp,Emax=gp*mp,Emin=Enu/lambda;
    double Epi,Integral=0.0;
    int i;

    double f,fpre;
    double x,dx=lambda/(double)HADRON_ELECTRON_X_STEPS;

    if (Emax<Emin){return 0;}
    if (Emin<pion_charged_mass_gev){Emin=pion_charged_mass_gev;}
    if((Ep-mp)<HADRON_NUMU_EMISSION_THRESHOLD_GEV){return 0;}

    x=0.0;
    //Epi=Enu/x;
    f=0.0;
    for ( i = 1; i < HADRON_ELECTRON_X_STEPS; i++){
        x += dx;
        Epi = Enu/x;
        fpre = f;
        f = dsigma(1,Epi,Ep)/x;
        if (i==1){
        Integral += f*dx;
        }
        else{
        Integral += 0.5*(f+fpre)*dx;
        }
    }
    double Integral_2 = 0.0;
    x=0.0;
    //Epi=Enu/x;
    f=0.0;
    for ( i = 1; i < HADRON_ELECTRON_X_STEPS; i++){
        x += dx;
        Epi = Enu/x;
        fpre = f;
        f = dsigma(-1,Epi,Ep)/x;
        if (i==1){
        Integral_2 += f*dx;
        }
        else{
        Integral_2 += 0.5*(f+fpre)*dx;
        }
    }
        return c*(Integral+Integral_2)/lambda/(4.0*M_PI);
}

///////   F_pi*(sigma_pp)/E_pi  ///////////////
double dsigma(int i,double Epi,double Ep){
    double gp=Ep/mp,gbp=sqrt(gp*gp-1.0),pp=gbp*mp,Fpi=0.0,s;
    /////////Kelner//////////////
        //Fpi = FpiQ(i,Epi,Ep);
        Fpi = FpiQ(0,Epi,Ep);
        s = pp_pion_cross_section(i,pp);
        //printf("pp = %e, Fpi = %e, s = %e \n",pp,Fpi,s);
    return s*Fpi/sqrt(1.0+1.0/(gbp*gbp));
}

//////// formulae in Kelner(2006) ///////////////////////////
double fe(double x){
    // Kelner 2006, Eq.(36) CAUTION : typos in the paper ////
     double r=muon_mass_gev*muon_mass_gev/(pion_charged_mass_gev*pion_charged_mass_gev),x2=x*x,r2=r*r;
     double g=(3.0-2.0*r)*(9.0*x2-6.0*log(x)-4.0*x2*x-5.0)/(9.0*pow(1.0-r,2.0));
     double h1=(3.0-2.0*r)*(9.0*r2-6.0*log(r)-4.0*r2*r-5.0)/(9.0*pow(1.0-r,2.0));
     double h2=(1.0+2.0*r)*(r-x)*(9.0*(r+x)-4.0*(r2+r*x+x2))/(9.0*r2);
     if (x>r){return g;}
     else{return h1+h2;}
}
double fnue(double x){
    // Kelner 2006, Eq.(40) CAUTION : typos in the paper ////
     double r=0.573,x2=x*x,r2=r*r;
     double g=2.0*(1.0-x)*(6.0*pow(1.0-x,2.0)+r*(5.0+5.0*x-4.0*x2)+6.0*r*log(x)/(1.0-x))/(3.0*pow(1.0-r,2.0));
     double h1=2.0*((1.0-r)*(6.0-7.0*r+11.0*r2-4.0*r2*r)+6.0*r*log(r))/(3.0*pow(1.0-r,2.0));
     //typo//double h1=2.0*((1.0-r)*(6.0-7.0*r+11.0*r-2.0-4.0*r2*r)+6.0*r*log(r))/(3.0*pow(1.0-r,2.0));
     double h2=2.0*(r-x)*(7.0*r2-4.0*r2*r+7.0*x*r-4.0*x*r2-2.0*x2-4.0*x2*r)/(3.0*r2);
     if (x>r){return g;}
     else{return h1+h2;}
}

///////////// pion enery spectrum //////////
////////////Kelner QGSJET Eq.(6)////////////////////
double FpiQ(int m,double Epi,double Ep){
        double L=log(0.001*Ep),x=Epi/Ep,mpi=1.35e-1;
        if(m==1){mpi=1.40e-1;}
        if(m==0){mpi=1.35e-1;}
        double xmin=mpi/Ep;
        if(x<=xmin){return 0.0;}
        if(x>1.0){return 0.0;}
        double Bpi=5.58+0.78*L+0.10*L*L,alpha=0.89/(sqrt(Bpi)*(1.0-exp(-0.33*Bpi))),r=3.1/(pow(Bpi,1.5));
        double xa=pow(x,alpha),A=pow((1.0-xa)/pow(1.0+r*xa,3.0),4.0);
                A=A*(1.0/(1.0-xa)+3.0*r/(1.0+r*xa))*sqrt(1.0-mpi/(x*Ep));
        //printf("%e %e %e,%e\n",x,xmin,1.0-mpi/(x*Ep),sqrt(1.0-mpi/(x*Ep)));
        return 4.0*alpha*Bpi*xa*A/x/Ep;
}
//////////////////////////////////////////////////

//////////////// pp cross section, Kamae(2006)//////////////////////
//// See also  Blattnig  (2000)    //////
// i = 0 (pi0), i = 1, (pi_plus), i = -1 (pi_minus), i = 2 (pi_plus+pi_minus)///
double pp_pion_cross_section(int i,double p){
        if(abs(i)>2){
            fprintf(stderr,"pp_pion_cross_section: invalid pion kind %d\n", i);
            return 0.0;
        }
        double Ep=sqrt(mp*mp+p*p),Tp=Ep-mp;
        double x=log10(p),x2=x*x,x3=x2*x,nondiff= 0.0,diff=0.0,Delta=0.0,res=0.0,tot=0.0,pi0=0.0,piplus=0.0,piminus=0.0;
        /////  parameters on table 1 (CAUTION : ERRATA Kamae 2007) ///
        double a[8],b[3],c[3],d[7],e[2],f[5],g[5];
            a[0]=0.1176;a[1]=0.3829;a[2]=23.10;a[3]=6.454;a[4]=-5.764;a[5]=-23.63;a[6]=94.75;a[7]=0.02667;
            b[0]=11.34;b[1]=23.72;
            c[0]=28.5;c[1]=-6.133;c[2]=1.464;
            d[0]=0.3522;d[1]=0.1530;d[2]=1.498;d[3]=2.0;d[4]=30.0;d[5]=3.155;d[6]=1.042;
            e[0]=5.922;e[1]=1.632;
            f[0]=0.0834;f[1]=9.5;f[2]=5.5;f[3]=1.68;f[4]=3134.0;
            g[0]=0.0004257;g[1]=4.5;g[2]=7.0;g[3]=2.1;g[4]=503.5;

        //// nondiffractive /////
        if(p<1.0){nondiff=0.0;}
        else if(p>=1.0 && p<1.3){
            nondiff=0.57*pow(x/a[0],1.2)*(a[2]+a[3]*x2+a[4]*x3+a[5]*exp(-a[6]*pow(x+a[7],2.0)));
        }
        else if(p>=1.3 && p<2.4){
            nondiff=(b[0]*fabs(a[1]-x)+b[1]*fabs(a[0]-x))/(a[1]-a[0]);
        }else if(p>=2.4 && p<10.0){
            nondiff=a[2]+a[3]*x2+a[4]*x3+a[5]*exp(-a[6]*pow(x+a[7],2.0));
        }else{
            nondiff=c[0]+c[1]*x+c[2]*x2;
        }

        //// diffractive /////
        if(p<2.25){diff=0.0;}
        else if(p>=2.25 && p<3.2){
            //printf("read 1 %e %e %e %e\n",sqrt((x-d[0])/d[1]),(x-d[0]),x,d[0]);
            diff=sqrt(fabs(x-d[0])/d[1])*(d[2]+d[3]*log10(d[4]*(x-0.25))+d[5]*x2-d[6]*x3);
        }else if(p>=3.2 && p<100.0){
            diff=d[2]+d[3]*log10(d[4]*(x-0.25))+d[5]*x2-d[6]*x3;
        }else{
            diff=e[0]+e[1]*x;
        }

        ////// Delta resonance///////
        if(Ep<1.4){Delta=0.0;}
        else if(Ep>=1.4 && Ep<1.6){
            Delta=f[0]*pow(Ep,10.0);
        }else if(Ep>=1.6 && Ep<1.8){
            Delta=f[1]*exp(-f[2]*pow(Ep-f[3],2.0));
        }else if(Ep>=1.8 && Ep<10.0){
            Delta=f[4]*pow(Ep,-10.0);
        }else{
            Delta=0.0;
        }

        ////// resonant /////////
        if(Ep<1.6){res=0.0;}
        else if(Ep>=1.6 && Ep<1.9){
            res=g[0]*pow(Ep,14.0);
        }else if(Ep>=1.9 && Ep<2.3){
            res=g[1]*exp(-g[2]*pow(Ep-g[3],2.0));
        }else if(Ep>=2.3 && Ep<20.0){
            res=g[4]*pow(Ep,-6.0);
        }else{
            res=0.0;
        }

        /// sum of the four components = pi^0 total cross section ///
        /////   not the total inelastic cross section //////////////
        tot = (nondiff+diff+Delta+res)*millibarn_cm2;
        //printf("nondiff = %e, diff = %e, Delta =%e, res = %e\n",nondiff,diff,Delta,res);
        //diff=sqrt((x-d[0])/d[1])*(d[2]+d[3]*log(d[4]*(x-0.25))+d[5]*x2-d[6]*x3);

        ////   Blattnig  (2000)  scaling    //////
        pi0 = 1.0/(0.007+0.1*log(Tp)/Tp+0.3/Tp/Tp);
        piplus = 1.0/(0.00717+0.0652*log(Tp)/Tp+0.162/Tp/Tp);
        piminus = 1.0/(0.00456+0.0846/pow(Tp,0.5)+0.577/pow(Tp,1.5));

        double sig=0.0;

        if(i==0){sig=tot;}
        else if(i==1){sig=tot*piplus/pi0;}
        else if(i==-1){sig= tot*piminus/pi0;}
        else if(i==2){sig=tot*(1.0+piplus/pi0+piminus/pi0);}
        return sig;
    }
//////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////   SECONDARY ELECTRONs  /////////////////////////
double lorentz_beta(double gamma){
    double beta_squared=gamma*gamma-1.0;
    if(beta_squared<0.0){return 0.0;}
    return sqrt(beta_squared)/gamma;
}
double QeKernel_BB05(int m,double Ee,double p){
        int i;
        double bmu=0.2714;

        double gp=sqrt(p*p+1.0),Ep=gp*mp,Emu,Emax=gp*mp,betamu;
        double Ethre;
        double dEpi,Epi,Integral=0.0,r=0.5*(pion_charged_mass_gev*pion_charged_mass_gev-muon_mass_gev*muon_mass_gev)/(bmu*pion_charged_mass_gev*pion_charged_mass_gev),Emin=Ee/r;
        Ethre = HADRON_SECONDARY_ELECTRON_THRESHOLD_GEV;
            if(Emax<Emin || (Ep-mp)<Ethre){ return 0;}
            //if (Emin<pion_charged_mass_gev){ Emin = pion_charged_mass_gev;}
            //if(r*Emin<muon_mass_local_gev){ //Emumin<muon_mass_gev///
            //    Emin = muon_mass_gev/r;   // 1.302e-1 < pion_charged_mass_gev //
            //}
            Emin = fmax(Emin,pion_charged_mass_gev);


        int Na = HADRON_SECONDARY_ELECTRON_MAX_STEPS;
        double a,amin = log10(Emin),amax = log10(Emax),da = HADRON_SECONDARY_ELECTRON_LOGE_STEP;
        Na = fmax(HADRON_SECONDARY_ELECTRON_MIN_STEPS,round((amax-amin)/da));
        double f,fpre,Epipre;

        a=amin; /////Epi=Emin=Ee/r///
        Epi=pow(10.0,a);////////// !!! diverge at i=Na !!!//////////////
        //Emu=0.5*Epi*(pion_mass_local_gev*pion_mass_local_gev-muon_mass_local_gev*muon_mass_local_gev)/(bmu*pion_mass_local_gev*pion_mass_local_gev);
        Emu=Epi*r;
        betamu=lorentz_beta(Emu/muon_mass_gev); //////////Emu=Ee//////////////
        if(betamu==0.0){f=0.0;}
        else{f=(dsigma(1,Epi,Ep)+dsigma(-1,Epi,Ep))*Fe(Ee,Epi)/(betamu*Epi);}
        //else{f=(dsigma(1,Epi,Ep))*Fe(Ee,Epi)/(betamu*Epi);}
        //printf("%e %e %e %e\n",dsigma(-1,Epi,Ep),Fe(Ee,Epi),Epi,Ep);

        for ( i = 1; i < Na; i++)    ////   pion energy loop ////
        {
            Epipre=Epi;
            a=a+da;
            Epi=pow(10.0,a);
            dEpi=Epi-Epipre;
            fpre=f;
            //Emu=0.5*Epi*(pion_mass_local_gev*pion_mass_local_gev-muon_mass_local_gev*muon_mass_local_gev)/(bmu*pion_mass_local_gev*pion_mass_local_gev);
            Emu=Epi*r;
            betamu = lorentz_beta(Emu/muon_mass_gev);
            f=(dsigma(1,Epi,Ep)+dsigma(-1,Epi,Ep))*Fe(Ee,Epi)/(betamu*Epi);
            //f=(dsigma(1,Epi,Ep))*Fe(Ee,Epi)/(betamu*Epi);
            Integral=Integral+0.5*(f+fpre)*dEpi;
            //printf("%e \n",(dsigma(1,Epi,Ep)+dsigma(-1,Epi,Ep))*Fe(Ee,Epi)/(betamu*Epi));
            //if(dsigma(1,Epi,Ep)<0.0){
            //printf("%e %e %e %e %e %e\n",Integral,f,Fe(Ee,Epi),dsigma(1,Epi,Ep),Ep,Epi);
           // }

            if(Integral <0.0 || isnan(Integral)){
                fprintf(stderr,"QeKernel_BB05 ERROR %d %d %e %e %e %e %e %e %e %e %e\n",i,Na,Integral,f,fpre,Fe(Ee,Epi),betamu,Ee,Epi,Ep,Emu);
                return 0.0;
            }
        }
        //printf("%e \n",Integral);


            //  (m_ec) factor is important!  //
            //  defined for p_e=p/(m_ec)  normalized momentum  //
            return 8.0*bmu*pion_charged_mass_gev*pion_charged_mass_gev*(c*me)*Integral/(pion_charged_mass_gev*pion_charged_mass_gev-muon_mass_gev*muon_mass_gev);

    }
////////////////Brunetti///Lazarian///2010///////////////////////////////
double Fe(double Ee,double Epi){
    double pion_mass_local_gev=1.3957e-1, muon_mass_local_gev=1.05658e-1, bmu=0.2714;
    double gpi=Epi/pion_mass_local_gev;
    double xi=muon_mass_local_gev*muon_mass_local_gev/(pion_mass_local_gev*pion_mass_local_gev); /////typo???///
    //double xi=pion_mass_local_gev*pion_mass_local_gev/(muon_mass_local_gev*muon_mass_local_gev); /////typo???///
    double Emu=(pion_mass_local_gev*pion_mass_local_gev-muon_mass_local_gev*muon_mass_local_gev)/(pion_mass_local_gev*pion_mass_local_gev)*Epi/(2.0*bmu),betamu=sqrt(1.0-muon_mass_local_gev*muon_mass_local_gev/(Emu*Emu));
    double lambda=2.0*pion_mass_local_gev*pion_mass_local_gev*Ee*bmu/((pion_mass_local_gev*pion_mass_local_gev-muon_mass_local_gev*muon_mass_local_gev)*Epi),P=-xi*xi/(betamu*pow((xi-1.0),2.0))*(4.0*bmu-1.0+pow(muon_mass_local_gev/pion_mass_local_gev,4.0));
    double X,lambda2=lambda*lambda,lambda3=lambda2*lambda;
    //xi=pion_mass_local_gev*pion_mass_local_gev/(muon_mass_local_gev*muon_mass_local_gev);
    //P=xi*xi/(bpi*(1.0-xi*xi))*(4.0-(1.0+1.0/xi)*(1.0+1.0/xi));
    ///Blasi 1999/////
    P=1.0/betamu*(2.0*Epi*xi/(Emu*(1.0-xi))-(1.0+xi)/(1.0-xi));
    //printf("%e %e %e\n",xi,P,(1.0-xi)/(1.0+xi));
    /////////////////
    if(lambda>1.0){return 0.0;}
    if(gpi==1.0){return 0.0;}
    if(lambda<0.0){return 0.0;}
    //if (gpi*pow(1.0+bpi,2.0)>(pion_mass_local_gev*pion_mass_local_gev-muon_mass_local_gev*muon_mass_local_gev)/(2.0*mpi*Ee*bmu))
    if ( (1.0-betamu)/(1.0+betamu) < lambda)
    {
        //X=5.0/12.0-3.0/4.0*lambda2+lambda3/3.0-P/(2.0*bpi)*(1.0/6.0-(bpi+0.5)*lambda2+(bpi+1.0/3.0)*lambda3);
        X=5.0-9.0*lambda2+4.0*lambda3-P/betamu*(1.0-(6.0*betamu+3.0)*lambda2+(6.0*betamu+2.0)*lambda3);
        return X/12.0;

    //}else if(gpi*pow(1.0+bpi,2.0)<=(pion_mass_local_gev*pion_mass_local_gev-muon_mass_local_gev*muon_mass_local_gev)/(2.0*mpi*Ee*bmu))
    }else if(lambda < (1.0-betamu)/(1.0+betamu))
    {
        X=lambda2*betamu/pow(1.0-betamu,2.0)*(3.0-2.0/3.0*lambda*(3.0+betamu*betamu)/(1.0-betamu))-P/(1.0-betamu)*(lambda2*(1.0+betamu)-2.0*lambda2/(1.0-betamu)*(0.5+lambda*(1.0+betamu))+2.0*lambda3*(3.0+betamu*betamu)/(3.0*pow(1.0-betamu,2.0)));
        //printf("%e %e %e\n",X,P,-P/(1.0-bpi)*(lambda2*(1.0+bpi)-2.0*lambda2/(1.0-bpi)*(0.5+lambda*(1.0+bpi))+2.0*lambda3*(3.0+bpi*bpi)/(3.0*pow(1.0-bpi,2.0))));
        //if(X<0){X=0;}
        return X;
    }
    return 0.0;

}
