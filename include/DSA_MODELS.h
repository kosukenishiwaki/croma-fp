#ifndef DSA_MODELS_H
#define DSA_MODELS_H

#include"params.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DSA_SPECIES_ELECTRON = 0,
    DSA_SPECIES_PROTON   = 1
} DSASpecies;

typedef enum {
    DSA_FAMILY_KR07  = 0,
    DSA_FAMILY_KR13  = 1,
    DSA_FAMILY_RYU19 = 2
} DSAFamily;

typedef enum {
    DSA_MODE_INITIAL = 0,
    DSA_MODE_REACC   = 1
} DSAMode;

typedef struct {
    const char *name;
    DSAFamily family;
    DSAFamily base_family;
    DSAMode mode;
    double x_cr0;
    double b[5];
    double mach_crit;
    double scale;
} DSAModel;

typedef enum {
    DSA_MODEL_KR07 = 0,
    DSA_MODEL_KR07R,
    DSA_MODEL_KR13,
    DSA_MODEL_KR13R,
    DSA_MODEL_CS14,
    DSA_MODEL_CS14R,
    DSA_MODEL_RYU19,
    DSA_MODEL_RYU19R,
    DSA_MODEL_COUNT
} DSAModelID;

extern DSAModelID DSAEtaModelInitial;
extern DSAModelID DSAEtaModelReacc;

typedef struct {
    double p_dsainj_e_mc;
    double p_dsainj_p_mc;
    double p_dsamax_e_mc;
    double p_dsamax_p_mc;
} DSAMomentumBounds;

/* model name helpers */
DSAModelID dsa_model_id_from_name(const char *name);
const char *dsa_model_name_from_id(DSAModelID id);

/* built-in models */
const DSAModel *dsa_get_model(DSAModelID id);
double dsa_eta_model(double mach, const DSAModel *model);
double dsa_interpolate_acceleration(double eta_initial,
                                    double eta_reacc,
                                    double x_cr,
                                    double x_cr0);
double dsa_eta_from_id(double mach, DSAModelID id);
double dsa_eta_interpolated(double mach,
                            DSAModelID initial_id,
                            DSAModelID reacc_id,
                            double x_cr);

/* slopes */
double dsa_compression_ratio(double mach, double gamma_gas);
double dsa_momentum_slope_d3p(double mach, double gamma_gas);
double dsa_momentum_slope_d3p_gam53(double mach);
double dsa_momentum_slope_dp(double mach, double gamma_gas);
double dsa_momentum_slope_dp_gam53(double mach);

/* thermal / injection momenta */
double dsa_thermal_momentum_mc(double temperature_K, DSASpecies species);
double dsa_injection_momentum_mc(double temperature_K,
                                 double chi,
                                 DSASpecies species);
double dsa_electron_thermal_momentum_mc(double temperature_K);
double dsa_proton_thermal_momentum_mc(double temperature_K);
double dsa_electron_injection_momentum_mc(double temperature_K, double chi_e);
double dsa_proton_injection_momentum_mc(double temperature_K, double chi_p);
void dsa_injection_momenta_mc(double temperature_K,
                              double chi_e,
                              double chi_p,
                              double *p_dsainj_e_mc,
                              double *p_dsainj_p_mc);
double dsa_electron_inj_from_proton_inj(double p_dsainj_p_mc);

/* spectrum */
double dsa_spectrum_cutoff(double p_mc,
                           double A,
                           double p_dsainj_mc,
                           double q,
                           double p_dsamax_mc);

/* proton energy density / normalization */
double dsa_proton_edens_unit(double p_dsainj_p_mc,
                             double q_p,
                             double p_dsamax_p_mc,
                             int nstep);
double dsa_proton_edens(double A_p,
                        double p_dsainj_p_mc,
                        double q_p,
                        double p_dsamax_p_mc,
                        int nstep);
double dsa_proton_norm_from_eta(double eta,
                                double rho1,
                                double u1,
                                double u2,
                                double p_dsainj_p_mc,
                                double q_p,
                                double p_dsamax_p_mc,
                                int nstep);

/* K_ep */
double dsa_mass_ratio_mp_me(void);
double dsa_electron_norm_from_kep_left(double K_ep,
                                       double A_p,
                                       double p_dsainj_p_mc,
                                       double p_dsamax_p_mc,
                                       double p_dsainj_e_mc,
                                       double p_dsamax_e_mc);
double dsa_kep_left_edge(double A_e,
                         double p_dsainj_e_mc,
                         double p_dsamax_e_mc,
                         double A_p,
                         double p_dsainj_p_mc,
                         double p_dsamax_p_mc);

double dsa_electron_ndens_unit(double p_dsainj_e_mc,
                               double q_e,
                               double p_dsamax_e_mc,
                               int nstep);

double dsa_electron_ndens(double A_e,
                          double p_dsainj_e_mc,
                          double q_e,
                          double p_dsamax_e_mc,
                          int nstep);

double dsa_xi_e_downstream(double A_e,
                           double p_dsainj_e_mc,
                           double q_e,
                           double p_dsamax_e_mc,
                           double rho1,
                           double mach,
                           double gamma_gas,
                           double mu_e,
                           int nstep);

double dsa_chi_e_from_proton_inj(double p_dsainj_e_mc,
                                 double temperature_K);

double dsa_analytic_xi_e_from_chi(double chi_e, double q);

double dsa_xi_e_from_kep_model(double A_e,
                               double p_dsainj_e_mc,
                               double q_e,
                               double p_dsamax_e_mc,
                               double rho1,
                               double mach,
                               double gamma_gas,
                               double mu_e,
                               int nstep);

double dsa_electron_norm_from_kep_at_pinj_p(double K_ep,
                                            double A_p,
                                            double p_dsainj_p_mc,
                                            double q_p,
                                            double p_dsamax_p_mc,
                                            double p_dsainj_e_mc,
                                            double q_e,
                                            double p_dsamax_e_mc);

double dsa_electron_ndens_from_pmin_unit(double pmin_e_mc,
                                         double p_dsainj_e_mc,
                                         double q_e,
                                         double p_dsamax_e_mc,
                                         int nstep);

double dsa_electron_ndens_from_pmin(double A_e,
                                    double pmin_e_mc,
                                    double p_dsainj_e_mc,
                                    double q_e,
                                    double p_dsamax_e_mc,
                                    int nstep);

double dsa_xi_e_downstream_from_pmin(double A_e,
                                     double pmin_e_mc,
                                     double p_dsainj_e_mc,
                                     double q_e,
                                     double p_dsamax_e_mc,
                                     double rho1,
                                     double mach,
                                     double gamma_gas,
                                     double mu_e,
                                     int nstep);

double dsa_electron_p_from_proton_p(double p_p_mc);


/* ------------------------------------------------------------------------- */
/* DSA -> FP coupling                                                         */
/* ------------------------------------------------------------------------- */

typedef enum {
    DSA_SPEC_PROTON = 0,
    DSA_SPEC_ELECTRON = 1
} DSASpeciesID;

typedef enum {
    DSA_ELECTRON_INJ_SAME_PHYSICAL_PINJ = 0,
    DSA_ELECTRON_INJ_SUPRATHERMAL = 1
} DSAElectronInjectionMode;

/* Shock / upstream state used to build an injection spectrum */
typedef struct {
    double mach;
    double gamma_gas;
    double rho1;
    double u1;
    double u2;
    double temperature;
} DSAShockState;

/* Injection model parameters.
   For now, reacceleration is ignored. */
typedef struct {
    DSAModelID eta_model_initial;
    double chi_p;
    double chi_e;
    double K_ep;
    double p_dsamax_p_mc;
    double p_dsamax_e_mc;
    DSAElectronInjectionMode electron_mode;
} DSAInjectionModel;

/* Internal DSA spectrum in f(p)=dN/d^3p form */
typedef struct {
    DSASpeciesID species;
    double q;
    double A_f;
    double p_inj_mc;
    double p_max_mc;
} DSASpectrum;

typedef struct {
    int nbin;
    double dlogp;
    double pmin_log10;
    double pmax_log10;
    int offset_to_fp;

    double *p;
    double *dp;
    double *E;

    /* DSA source stored on this grid as dN/dp */
    double *source_dp;
} DSAGrid1D;


typedef struct {
    DSAGrid1D proton;
    DSAGrid1D electron;
    int initialized;
} DSAGrid;

/* ------------------------------------------------------------------------- */
/* DSA grid initialization                                                    */
/* ------------------------------------------------------------------------- */
int dsa_init_grid(DSAGrid *grid,
                const CRspectrum *CRp,
                const CRspectrum *CRe,
                double pmin_dsa_p_log10,
                double pmin_dsa_e_log10);


void dsa_free_grid(DSAGrid *grid);

/* ------------------------------------------------------------------------- */
/* Build DSA spectra from shock parameters                                    */
/* ------------------------------------------------------------------------- */

/* Build proton injection spectrum from the shock state and model parameters */
int dsa_build_proton_spectrum(const DSAShockState *shock,
                              const DSAInjectionModel *model,
                              DSASpectrum *spec_p);

/* Build electron injection spectrum from the shock state and proton spectrum */
int dsa_build_electron_spectrum(const DSAShockState *shock,
                                const DSAInjectionModel *model,
                                const DSASpectrum *spec_p,
                                DSASpectrum *spec_e);

/* ------------------------------------------------------------------------- */
/* Spectrum evaluation                                                        */
/* ------------------------------------------------------------------------- */

/* Evaluate internal f(p)=dN/d^3p spectrum */
double dsa_spectrum_f(const DSASpectrum *spec, double p_mc);

/* Evaluate dN/dp = 4 pi p^2 f(p) */
double dsa_spectrum_dp(const DSASpectrum *spec, double p_mc);

/* ------------------------------------------------------------------------- */
/* Fill source terms on DSA / FP grids                                        */
/* ------------------------------------------------------------------------- */

/* Fill a DSA working grid with dN/dp source values from a given spectrum */
int dsa_fill_source_dp_on_grid(const DSASpectrum *spec,
                               DSAGrid1D *grid);

/* Copy the high-p part of the DSA source array onto the existing FP grid.
   The mapping is done through the precomputed offset_to_fp. */
int dsa_copy_source_to_fp_grid(const DSAGrid1D *grid_dsa,
                               int nbin_fp,
                               double *source_fp);

/* Convenience wrappers: build proton/electron source directly on FP grids */
int dsa_build_proton_source_fp(const DSAShockState *shock,
                               const DSAInjectionModel *model,
                               DSAGrid *grid,
                               const CRspectrum *CRp,
                               double *source_p_fp,
                               DSASpectrum *spec_p_out);

int dsa_build_electron_source_fp(const DSAShockState *shock,
                                 const DSAInjectionModel *model,
                                 DSAGrid *grid,
                                 const CRspectrum *CRe,
                                 const DSASpectrum *spec_p,
                                 double *source_e_fp,
                                 DSASpectrum *spec_e_out);

/* ------------------------------------------------------------------------- */
/* Diagnostics                                                                */
/* ------------------------------------------------------------------------- */

/* Integrate dN/dp source over one grid */
double dsa_integrate_source_number(const DSAGrid1D *grid);

/* Integrate E(p) * dN/dp source over one grid */
double dsa_integrate_source_energy(const DSAGrid1D *grid);

#ifdef __cplusplus
}
#endif

#endif
