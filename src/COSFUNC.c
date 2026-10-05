/*
   COSFUNC.c
   - cosmology utility functions
*/

#include"COSFUNC.h"
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<gsl/gsl_math.h>
#include<gsl/gsl_integration.h>
#include "CONSTANTS.h"
#include "params.h"

struct dl_params { double omegam; double omegaL; };
static double dl_integrand(double x, void * O);

////////////////Default cosmological parameters////////////////////////////
// Planck 2018 //
double OmBh2,OmCh2;
double TCMB = COS_DEFAULT_TCMB_K; //[K]//

////////////////Cosmological Parameters////////////////////////////
double H0 = COS_DEFAULT_H0_KM_S_MPC;
double h70 = COS_DEFAULT_H70,h100 = COS_DEFAULT_H100,ns = COS_DEFAULT_NS,sig8 = COS_DEFAULT_SIG8,deltac = COS_DEFAULT_DELTAC,OmB = COS_DEFAULT_OMB,OmC = COS_DEFAULT_OMC,OmL = COS_DEFAULT_OML;


double Hubble(double z){
    // [km/s/Mpc] //
    return H0*sqrt((OmC+OmB)*pow(1.0+z,3.0)+(1.0-(OmC+OmB)));
}
double Costime(double z){
    //Cosmological time [Gyr] since BigBang//
    double Integral=0.0;
    double zmax = COS_COSTIME_ZMAX;

    int i,Nint = COS_COSTIME_LOG_NINT;double x;double dz;
    double az,daz;
    if(z<COS_COSTIME_LINEAR_Z_THRESHOLD){
      dz=(COS_COSTIME_LINEAR_Z_TARGET-z)/(double)COS_COSTIME_LINEAR_STEPS; daz=(log10(zmax)-log10(COS_COSTIME_LINEAR_Z_TARGET))/(double)Nint;
      x=z;
      for ( i = 0; i <COS_COSTIME_LINEAR_STEPS; i++)
      {
        Integral+=0.5*(1.0/((1.0+x)*Hubble(x))+1.0/((1.0+x+dz)*Hubble(x+dz)))*dz;
        x+=dz;
      }
    }
    else{daz=(log10(zmax)-log10(z))/(double)Nint;x=z;}

    double xpre=0.0;

    az=log10(x);
    for ( i = 0; i <Nint; i++)
    {
      xpre=x;
      az+=daz;
      x=pow(10.0,az);
      Integral+=0.5*(1.0/((1.0+xpre)*Hubble(xpre))+1.0/((1.0+x)*Hubble(x)))*(x-xpre);
      //printf("%e %e %e %e\n",az,x,xpre,Integral);
    }

    return Integral*Mpc*1.0e-5/Gyr;
}
double dz_to_dt(double dz,double z){
    // dt in [Gyr]  > 0 //
    return dz/Hubble(z)/(1.0+z)/Gyr*Mpc*1.0e-5;
}
static double dl_integrand(double x, void * O) {
    //parameters
    struct dl_params * params = (struct dl_params *)O;
    double omegam = (params->omegam);
    double omegaL = (params->omegaL);
    //integrand
    double f = 1.0/ sqrt(omegam*pow(1.0+x,3.0)+omegaL);
    return f;
}

double E_cos(double z){
  return Hubble(z)/H0;
}

double angular_diameter_distance_mpc(double z){
    //Angular diameter distance//Mpc///
    return pow(1.0+z,-2.0)*luminosity_distance_mpc(z);
}
double luminosity_distance_mpc(double z){
    //Luminosity distance//Mpc///
    //workspace
    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(COS_GSL_CQUAD_WORKSPACE_SMALL);
    double Integral, error;
    size_t nevals;
    //parameter
    struct dl_params params = {OmC + OmB, OmL};
    //GSL integrand
    gsl_function F = { &dl_integrand, &params };
    //CQUAD
    gsl_integration_cquad(
        &F,      //integrand
        0, z,  //range(a,b)
        0, 1e-5, //error
        w,       //workspace
        &Integral, //result
        &error,
        &nevals  //trial
    );
    gsl_integration_cquad_workspace_free(w);
    return (1.0+z)*Integral*c/1.0e+5/H0;
}

double meandMdt(double z,double M){ //[M_sun/dz]// originally [M_sun/dt] //
    // M_sun / sec  //
  double A=46.1*pow(M*1.0e-12,1.1)*(1.0+1.11*z)*sqrt((OmC+OmB)*pow(1.0+z,3.0)+OmL);
  //double A=50.0*pow(M*1.0e-12,1.1)*(1.0+1.11*z)*sqrt((OmC+OmB)*pow(1.0+z,3.0)+OmL);
  return -A/(1.0+z)/Hubble(z)*Mpc/1.0e+5/(Gyr*1.0e-9); // [dM/dz]  //
}
