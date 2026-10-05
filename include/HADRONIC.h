#ifndef _HADRONIC_h
#define _HADRONIC_h

#include"params.h"

#define HADRON_KERNEL_EPI_STEPS 200
#define HADRON_ELECTRON_X_STEPS 200
#define HADRON_PHOTON_FIELD_STEPS 128

#define HADRON_GAMMA_EMISSION_THRESHOLD_GEV 0.285
#define HADRON_ELECTRON_EMISSION_THRESHOLD_GEV 0.289
#define HADRON_NUMU_EMISSION_THRESHOLD_GEV 0.29

#define HADRON_SECONDARY_ELECTRON_THRESHOLD_GEV 0.289
#define HADRON_SECONDARY_ELECTRON_LOGE_STEP 0.05
#define HADRON_SECONDARY_ELECTRON_MIN_STEPS 50
#define HADRON_SECONDARY_ELECTRON_MAX_STEPS 512

typedef enum {
    HADRON_LEPTON_ELECTRON = 0,
    HADRON_LEPTON_NUE = 1,
    HADRON_LEPTON_NUMU = 2
} HadronLeptonKernel;

void Kernels(int snap_i,int snap_fin,int bins,double **fga,double **fnu,double ***fic,double *z,double *p,double *pe);
void Kernels_z(int bins,double **fga,double **fnu,double **fic,double z,double *p,double *pe);
double gammaKernel(double Egamma,double p);
double eKernel(HadronLeptonKernel lepton_kind,double E,double p);
double numu1Kernel(double Enu,double p);
double dsigma(int i,double Epi,double Ep);
double fe(double x);
double fnue(double x);
double FpiQ(int m,double Epi,double Ep);
double pp_pion_cross_section(int i,double p);

double lorentz_beta(double gamma);
double QeKernel_BB05(int m,double Ee,double p);
double Fe(double Ee,double Epi);

#endif
