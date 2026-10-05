#ifndef _COSFUNC_h
#define _COSFUNC_h

#ifdef __cplusplus
extern "C" {
#endif

// Planck 2018 //
#define COS_DEFAULT_TCMB_K 2.7255
#define COS_DEFAULT_H0_KM_S_MPC 67.36
#define COS_DEFAULT_H70 0.9622857142857143
#define COS_DEFAULT_H100 0.6736
#define COS_DEFAULT_NS 0.9649
#define COS_DEFAULT_SIG8 0.8111
#define COS_DEFAULT_DELTAC 1.686
#define COS_DEFAULT_OMB 0.049301692328524445
#define COS_DEFAULT_OMC 0.26447041034523616
#define COS_DEFAULT_OML 0.6862278973262395

#define COS_COSTIME_ZMAX 2200.0
#define COS_COSTIME_LOG_NINT 1000
#define COS_COSTIME_LINEAR_Z_THRESHOLD 1.0e-3
#define COS_COSTIME_LINEAR_Z_TARGET 0.001
#define COS_COSTIME_LINEAR_STEPS 100
#define COS_GSL_CQUAD_WORKSPACE_SMALL 128


////////////////Cosmological Parameters WMAP9////////////////////////////
extern double H0; //[km/s/Mpc]//
extern double h70,h100,ns,sig8,deltac,OmB,OmC,OmL;
extern double OmBh2,OmCh2;
extern double TCMB;

double luminosity_distance_mpc(double z);
double angular_diameter_distance_mpc(double z);
double E_cos(double z);
double dz_to_dt(double dz,double z);
double Costime(double z);
double Hubble(double z);
double meandMdt(double z,double M);

#ifdef __cplusplus
}
#endif

#endif
