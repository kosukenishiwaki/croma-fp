#include "DSA_MODELS.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KR07_LOW_MACH_PREFACTOR 1.96e-3
#define KR13_INITIAL_ETA_MAX    0.21152
#define KR13_REACC_ETA_MAX      0.2055
#define RYU19_ETA_MAX           0.0348

#define DSA_M_ELECTRON  9.1093837015e-28   /* g */
#define DSA_M_PROTON    1.67262192369e-24  /* g */
#define DSA_K_BOLTZ     1.380649e-16       /* erg K^-1 */
#define DSA_C_LIGHT     2.99792458e10      /* cm s^-1 */
#define DSA_FOUR_PI     12.566370614359172


/* ------------------------------------------------------------------------- */
/* Internal helpers                                                           */
/* ------------------------------------------------------------------------- */

static double polynomial_branch(double mach, const double b[5])
{
    double x;

    if (mach <= 0.0) {
        return 0.0;
    }

    x = mach - 1.0;
    return (b[0]
          + b[1] * x
          + b[2] * x * x
          + b[3] * x * x * x
          + b[4] * x * x * x * x) / (mach * mach * mach * mach);
}

static double kr07_reacc_low_mach_branch(double mach)
{
    const double gamma = 5.0 / 3.0;
    double m2, xs, delta0;

    if (mach <= 0.0) {
        return 0.0;
    }

    m2 = mach * mach;
    xs = (gamma + 1.0) / (gamma - 1.0 + 2.0 / m2);

    delta0 = 2.0
           * (((2.0 * gamma * m2 - gamma + 1.0) / (gamma + 1.0))
              - pow(xs, gamma))
           / (gamma * (gamma - 1.0) * m2 * xs);

    return 1.025 * delta0;
}

static double kr13_initial_low_mach_branch(double mach)
{
    return -0.0005950569221922047
         + 1.880258286365841e-5 * pow(mach, 5.334076006529829);
}

static double dsa_species_mass_cgs(DSASpecies species)
{
    if (species == DSA_SPECIES_ELECTRON) {
        return DSA_M_ELECTRON;
    }
    return DSA_M_PROTON;
}

static double dsa_safe_exp_argument(double x)
{
    if (x < -700.0) {
        return -700.0;
    }
    if (x > 700.0) {
        return 700.0;
    }
    return x;
}

static double dsa_spectrum_shape(double p_mc,
                                 double p_dsainj_mc,
                                 double q,
                                 double p_dsamax_mc)
{
    double ratio_inj, ratio_max, expo;

    if (p_mc <= 0.0 || p_dsainj_mc <= 0.0 || p_dsamax_mc <= 0.0) {
        return 0.0;
    }
    if (p_mc < p_dsainj_mc) {
        return 0.0;
    }

    ratio_inj = p_mc / p_dsainj_mc;
    ratio_max = p_mc / p_dsamax_mc;
    expo = -ratio_max * ratio_max;

    return pow(ratio_inj, -q) * exp(dsa_safe_exp_argument(expo));
}

static double dsa_proton_energy_integrand_unit(double p_mc,
                                               double p_dsainj_p_mc,
                                               double q_p,
                                               double p_dsamax_p_mc)
{
    double kinetic_mc2;

    if (p_mc <= 0.0 || p_dsainj_p_mc <= 0.0 || p_dsamax_p_mc <= 0.0) {
        return 0.0;
    }
    if (p_mc < p_dsainj_p_mc) {
        return 0.0;
    }

    /* kinetic energy per particle in units of m_p c^2 */
    kinetic_mc2 = sqrt(1.0 + p_mc * p_mc) - 1.0;

    return kinetic_mc2
         * p_mc * p_mc
         * dsa_spectrum_shape(p_mc, p_dsainj_p_mc, q_p, p_dsamax_p_mc);
}

static int dsa_name_equal_casefold(const char *lhs, const char *rhs)
{
    unsigned char a, b;

    if (lhs == NULL || rhs == NULL) return 0;

    while (*lhs != '\0' && *rhs != '\0') {
        a = (unsigned char)*lhs;
        b = (unsigned char)*rhs;
        if (tolower(a) != tolower(b)) return 0;
        lhs++;
        rhs++;
    }

    return (*lhs == '\0' && *rhs == '\0') ? 1 : 0;
}


static void dsa_zero_grid1d(DSAGrid1D *grid)
{
    if (grid == NULL) {
        return;
    }

    grid->nbin = 0;
    grid->dlogp = 0.0;
    grid->pmin_log10 = 0.0;
    grid->pmax_log10 = 0.0;
    grid->offset_to_fp = 0;

    grid->p = NULL;
    grid->dp = NULL;
    grid->E = NULL;
    grid->source_dp = NULL;
}

static int dsa_allocate_grid1d(DSAGrid1D *grid, int nbin)
{
    if (grid == NULL || nbin <= 0) {
        return 0;
    }

    dsa_zero_grid1d(grid);
    grid->nbin = nbin;

    grid->p = (double *)malloc((size_t)nbin * sizeof(double));
    grid->dp = (double *)malloc((size_t)nbin * sizeof(double));
    grid->E = (double *)malloc((size_t)nbin * sizeof(double));
    grid->source_dp = (double *)malloc((size_t)nbin * sizeof(double));

    if (grid->p == NULL || grid->dp == NULL || grid->E == NULL || grid->source_dp == NULL) {
        free(grid->p);
        free(grid->dp);
        free(grid->E);
        free(grid->source_dp);
        dsa_zero_grid1d(grid);
        return 0;
    }

    memset(grid->source_dp, 0, (size_t)nbin * sizeof(double));
    return 1;
}



static void dsa_free_grid1d(DSAGrid1D *grid)
{
    if (grid == NULL) {
        return;
    }

    free(grid->p);
    free(grid->dp);
    free(grid->E);
    free(grid->source_dp);

    dsa_zero_grid1d(grid);
}

static double dsa_grid_dlogp_from_cr(const CRspectrum *CR)
{
    if (CR == NULL) {
        return 0.0;
    }

    if (CR->p[0] <= 0.0 || CR->p[1] <= 0.0) {
        return 0.0;
    }

    return log10(CR->p[1]) - log10(CR->p[0]);
}

static int dsa_extra_bins_needed(double pmin_fp_log10,
                                 double pmin_dsa_log10,
                                 double dlogp)
{
    double delta;
    int n_extra;

    if (dlogp <= 0.0) {
        return 0;
    }

    if (pmin_dsa_log10 >= pmin_fp_log10) {
        return 0;
    }

    delta = (pmin_fp_log10 - pmin_dsa_log10) / dlogp;
    n_extra = (int)ceil(delta - 1.0e-12);

    if (n_extra < 0) {
        n_extra = 0;
    }

    return n_extra;
}

static int dsa_fill_grid1d_from_spacing(DSAGrid1D *grid,
                                        double pmin_log10,
                                        double dlogp,
                                        double mass_cgs)
{
    int j;
    double logp, pL, pR;

    if (grid == NULL || grid->nbin <= 0) {
        return 0;
    }
    if (dlogp <= 0.0 || mass_cgs <= 0.0) {
        return 0;
    }

    grid->dlogp = dlogp;
    grid->pmin_log10 = pmin_log10;
    grid->pmax_log10 = pmin_log10 + dlogp * (double)(grid->nbin - 1);

    for (j = 0; j < grid->nbin; j++) {
        logp = pmin_log10 + dlogp * (double)j;
        grid->p[j] = pow(10.0, logp);

        pL = pow(10.0, logp - 0.5 * dlogp);
        pR = pow(10.0, logp + 0.5 * dlogp);
        grid->dp[j] = pR - pL;

        grid->E[j] = mass_cgs * sqrt(1.0 + grid->p[j] * grid->p[j]);

        grid->source_dp[j] = 0.0;
    }

    return 1;
}



/* ------------------------------------------------------------------------- */
/* Built-in models                                                            */
/* ------------------------------------------------------------------------- */

static const DSAModel g_dsa_models[DSA_MODEL_COUNT] = {
    {
        "KR07",
        DSA_FAMILY_KR07, DSA_FAMILY_KR07, DSA_MODE_INITIAL,
        0.0,
        {5.46, -9.78, 4.17, -0.33, 0.57},
        1.0,
        1.0
    },
    {
        "KR07r",
        DSA_FAMILY_KR07, DSA_FAMILY_KR07, DSA_MODE_REACC,
        0.3,
        {0.24, -1.56, 2.8, 0.512, 0.557},
        1.0,
        1.0
    },
    {
        "KR13",
        DSA_FAMILY_KR13, DSA_FAMILY_KR13, DSA_MODE_INITIAL,
        0.0,
        {-2.87, 9.67, -8.88, 1.94, 0.18},
        2.0,
        1.0
    },
    {
        "KR13r",
        DSA_FAMILY_KR13, DSA_FAMILY_KR13, DSA_MODE_REACC,
        0.05,
        {-0.72, 2.73, -3.29, 1.34, 0.19},
        2.0,
        1.0
    },
    {
        "CS14",
        DSA_FAMILY_KR13, DSA_FAMILY_KR13, DSA_MODE_INITIAL,
        0.0,
        {-2.87, 9.67, -8.88, 1.94, 0.18},
        2.0,
        0.5
    },
    {
        "CS14r",
        DSA_FAMILY_KR13, DSA_FAMILY_KR13, DSA_MODE_REACC,
        0.05,
        {-0.72, 2.73, -3.29, 1.34, 0.19},
        2.0,
        0.5
    },
    {
        "Ryu19",
        DSA_FAMILY_RYU19, DSA_FAMILY_RYU19, DSA_MODE_INITIAL,
        0.0,
        {-1.5255, 2.4026, -1.2534, 0.2215, 0.0336},
        2.25,
        1.0
    },
    {
        "Ryu19r",
        DSA_FAMILY_RYU19, DSA_FAMILY_RYU19, DSA_MODE_REACC,
        0.05,
        {0.3965, -0.21898, -0.2074, 0.1319, 0.0351},
        2.25,
        1.0
    }
};

DSAModelID dsa_model_id_from_name(const char *name)
{
    int i;

    if (name == NULL) {
        return DSA_MODEL_COUNT;
    }

    for (i = 0; i < DSA_MODEL_COUNT; i++) {
        if (dsa_name_equal_casefold(name, g_dsa_models[i].name)) {
            return (DSAModelID)i;
        }
    }

    return DSA_MODEL_COUNT;
}

const char *dsa_model_name_from_id(DSAModelID id)
{
    const DSAModel *model = dsa_get_model(id);

    if (model == NULL) {
        return "INVALID";
    }

    return model->name;
}

/* ------------------------------------------------------------------------- */
/* Public API                                                                 */
/* ------------------------------------------------------------------------- */

const DSAModel *dsa_get_model(DSAModelID id)
{
    if (id < 0 || id >= DSA_MODEL_COUNT) {
        return NULL;
    }
    return &g_dsa_models[id];
}

double dsa_eta_model(double mach, const DSAModel *model)
{
    double eta = 0.0;
    double poly;
    DSAFamily base_family;

    if (model == NULL) {
        return 0.0;
    }

    if (mach < model->mach_crit) {
        return 0.0;
    }

    poly = polynomial_branch(mach, model->b);
    base_family = model->base_family;

    if (base_family == DSA_FAMILY_KR07 && model->mode == DSA_MODE_INITIAL) {

        if (mach <= 2.0) {
            eta = KR07_LOW_MACH_PREFACTOR * (mach * mach - 1.0);
        } else {
            eta = poly;
        }

    } else if (base_family == DSA_FAMILY_KR07 && model->mode == DSA_MODE_REACC) {

        if (mach <= 1.5) {
            eta = kr07_reacc_low_mach_branch(mach);
        } else {
            eta = poly;
        }

    } else if (base_family == DSA_FAMILY_KR13 && model->mode == DSA_MODE_INITIAL) {

        if (mach <= 5.0) {
            eta = kr13_initial_low_mach_branch(mach);
        } else if (mach <= 15.0) {
            eta = poly;
        } else {
            eta = KR13_INITIAL_ETA_MAX;
        }

    } else if (base_family == DSA_FAMILY_KR13 && model->mode == DSA_MODE_REACC) {

        if (mach <= 17.7) {
            eta = poly;
        } else {
            eta = KR13_REACC_ETA_MAX;
        }

    } else if (base_family == DSA_FAMILY_RYU19 && model->mode == DSA_MODE_INITIAL) {

        if (mach <= 34.0) {
            eta = poly;
        } else {
            eta = RYU19_ETA_MAX;
        }

    } else if (base_family == DSA_FAMILY_RYU19 && model->mode == DSA_MODE_REACC) {

        if (mach <= 34.0) {
            eta = poly;
        } else {
            eta = RYU19_ETA_MAX;
        }

    } else {
        return 0.0;
    }

    eta *= model->scale;

    if (eta < 0.0) {
        eta = 0.0;
    }

    return eta;
}

double dsa_interpolate_acceleration(double eta_initial,
                                    double eta_reacc,
                                    double x_cr,
                                    double x_cr0)
{
    double weight;

    if (x_cr0 <= 0.0) {
        return eta_initial;
    }

    weight = x_cr / x_cr0;
    if (weight < 0.0) {
        weight = 0.0;
    }
    if (weight > 1.0) {
        weight = 1.0;
    }

    return (1.0 - weight) * eta_initial + weight * eta_reacc;
}

double dsa_eta_from_id(double mach, DSAModelID id)
{
    const DSAModel *model = dsa_get_model(id);
    return dsa_eta_model(mach, model);
}

double dsa_eta_interpolated(double mach,
                            DSAModelID initial_id,
                            DSAModelID reacc_id,
                            double x_cr)
{
    const DSAModel *init_model = dsa_get_model(initial_id);
    const DSAModel *reacc_model = dsa_get_model(reacc_id);
    double eta_init, eta_reacc;

    if (init_model == NULL || reacc_model == NULL) {
        return 0.0;
    }

    eta_init  = dsa_eta_model(mach, init_model);
    eta_reacc = dsa_eta_model(mach, reacc_model);

    return dsa_interpolate_acceleration(eta_init,
                                        eta_reacc,
                                        x_cr,
                                        reacc_model->x_cr0);
}

double dsa_compression_ratio(double mach, double gamma_gas)
{
    double m2;
    double numerator, denominator;

    if (mach <= 1.0) {
        return 1.0;
    }
    if (gamma_gas <= 1.0) {
        return 1.0;
    }

    m2 = mach * mach;
    numerator   = (gamma_gas + 1.0) * m2;
    denominator = (gamma_gas - 1.0) * m2 + 2.0;

    if (denominator <= 0.0) {
        return 1.0;
    }

    return numerator / denominator;
}

double dsa_momentum_slope_d3p(double mach, double gamma_gas)
{
    double r;

    if (mach <= 1.0) {
        return 0.0;
    }

    r = dsa_compression_ratio(mach, gamma_gas);
    if (r <= 1.0) {
        return 0.0;
    }

    return 3.0 * r / (r - 1.0);
}

double dsa_momentum_slope_d3p_gam53(double mach)
{
    double m2;

    if (mach <= 1.0) {
        return 0.0;
    }

    m2 = mach * mach;
    return 4.0 * m2 / (m2 - 1.0);
}

double dsa_momentum_slope_dp(double mach, double gamma_gas)
{
    double q;

    q = dsa_momentum_slope_d3p(mach, gamma_gas);
    if (q <= 0.0) {
        return 0.0;
    }

    return q - 2.0;
}

double dsa_momentum_slope_dp_gam53(double mach)
{
    double q;

    q = dsa_momentum_slope_d3p_gam53(mach);
    if (q <= 0.0) {
        return 0.0;
    }

    return q - 2.0;
}

double dsa_thermal_momentum_mc(double temperature_K, DSASpecies species)
{
    double mass;
    double theta;

    if (temperature_K <= 0.0) {
        return 0.0;
    }

    mass = dsa_species_mass_cgs(species);

    /* p_th/(m c) = sqrt(2 k_B T / (m c^2)) */
    theta = 2.0 * DSA_K_BOLTZ * temperature_K
          / (mass * DSA_C_LIGHT * DSA_C_LIGHT);

    if (theta <= 0.0) {
        return 0.0;
    }

    return sqrt(theta);
}

double dsa_injection_momentum_mc(double temperature_K,
                                 double chi,
                                 DSASpecies species)
{
    if (chi <= 0.0) {
        return 0.0;
    }

    return chi * dsa_thermal_momentum_mc(temperature_K, species);
}

double dsa_electron_thermal_momentum_mc(double temperature_K)
{
    return dsa_thermal_momentum_mc(temperature_K, DSA_SPECIES_ELECTRON);
}

double dsa_proton_thermal_momentum_mc(double temperature_K)
{
    return dsa_thermal_momentum_mc(temperature_K, DSA_SPECIES_PROTON);
}

double dsa_electron_injection_momentum_mc(double temperature_K, double chi_e)
{
    return dsa_injection_momentum_mc(temperature_K, chi_e, DSA_SPECIES_ELECTRON);
}

double dsa_proton_injection_momentum_mc(double temperature_K, double chi_p)
{
    return dsa_injection_momentum_mc(temperature_K, chi_p, DSA_SPECIES_PROTON);
}

void dsa_injection_momenta_mc(double temperature_K,
                              double chi_e,
                              double chi_p,
                              double *p_dsainj_e_mc,
                              double *p_dsainj_p_mc)
{
    if (p_dsainj_e_mc != NULL) {
        *p_dsainj_e_mc = dsa_electron_injection_momentum_mc(temperature_K, chi_e);
    }

    if (p_dsainj_p_mc != NULL) {
        *p_dsainj_p_mc = dsa_proton_injection_momentum_mc(temperature_K, chi_p);
    }
}

double dsa_electron_inj_from_proton_inj(double p_dsainj_p_mc)
{
    if (p_dsainj_p_mc <= 0.0) {
        return 0.0;
    }

    /* p_inj,e * m_e c = p_inj,p * m_p c */
    return dsa_mass_ratio_mp_me() * p_dsainj_p_mc;
}

double dsa_spectrum_cutoff(double p_mc,
                           double A,
                           double p_dsainj_mc,
                           double q,
                           double p_dsamax_mc)
{
    if (A == 0.0) {
        return 0.0;
    }

    return A * dsa_spectrum_shape(p_mc, p_dsainj_mc, q, p_dsamax_mc);
}

double dsa_proton_edens_unit(double p_dsainj_p_mc,
                             double q_p,
                             double p_dsamax_p_mc,
                             int nstep)
{
    int i;
    double h, p, sum, integral_dimless;

    if (p_dsainj_p_mc <= 0.0 || p_dsamax_p_mc <= 0.0) {
        return 0.0;
    }
    if (p_dsamax_p_mc <= p_dsainj_p_mc) {
        return 0.0;
    }

    if (nstep < 2) {
        nstep = 2000;
    }
    if (nstep % 2 != 0) {
        nstep += 1;
    }

    h = (p_dsamax_p_mc - p_dsainj_p_mc) / (double)nstep;

    sum = dsa_proton_energy_integrand_unit(p_dsainj_p_mc, p_dsainj_p_mc, q_p, p_dsamax_p_mc)
        + dsa_proton_energy_integrand_unit(p_dsamax_p_mc, p_dsainj_p_mc, q_p, p_dsamax_p_mc);

    for (i = 1; i < nstep; i++) {
        p = p_dsainj_p_mc + h * (double)i;
        if (i % 2 == 0) {
            sum += 2.0 * dsa_proton_energy_integrand_unit(p, p_dsainj_p_mc, q_p, p_dsamax_p_mc);
        } else {
            sum += 4.0 * dsa_proton_energy_integrand_unit(p, p_dsainj_p_mc, q_p, p_dsamax_p_mc);
        }
    }

    integral_dimless = sum * h / 3.0;

    return DSA_FOUR_PI * DSA_M_PROTON * DSA_C_LIGHT * DSA_C_LIGHT * integral_dimless;
}

double dsa_proton_edens(double A_p,
                        double p_dsainj_p_mc,
                        double q_p,
                        double p_dsamax_p_mc,
                        int nstep)
{
    return A_p * dsa_proton_edens_unit(p_dsainj_p_mc,
                                       q_p,
                                       p_dsamax_p_mc,
                                       nstep);
}

double dsa_proton_norm_from_eta(double eta,
                                double rho1,
                                double u1,
                                double u2,
                                double p_dsainj_p_mc,
                                double q_p,
                                double p_dsamax_p_mc,
                                int nstep)
{
    double ecrp_unit;
    double target_ecrp;

    if (eta <= 0.0) {
        return 0.0;
    }
    if (rho1 <= 0.0 || u1 <= 0.0 || u2 <= 0.0) {
        return 0.0;
    }
    if (p_dsainj_p_mc <= 0.0 || p_dsamax_p_mc <= p_dsainj_p_mc) {
        return 0.0;
    }

    ecrp_unit = dsa_proton_edens_unit(p_dsainj_p_mc,
                                      q_p,
                                      p_dsamax_p_mc,
                                      nstep);
    if (ecrp_unit <= 0.0) {
        return 0.0;
    }

    target_ecrp = eta * (0.5 * rho1 * u1 * u1 * u1) / u2;

    return target_ecrp / ecrp_unit;
}

double dsa_mass_ratio_mp_me(void)
{
    return DSA_M_PROTON / DSA_M_ELECTRON;
}

double dsa_electron_norm_from_kep_left(double K_ep,
                                       double A_p,
                                       double p_dsainj_p_mc,
                                       double p_dsamax_p_mc,
                                       double p_dsainj_e_mc,
                                       double p_dsamax_e_mc)
{
    double x_e, x_p;
    double mass_factor;

    if (K_ep <= 0.0 || A_p <= 0.0) {
        return 0.0;
    }
    if (p_dsainj_p_mc <= 0.0 || p_dsamax_p_mc <= 0.0) {
        return 0.0;
    }
    if (p_dsainj_e_mc <= 0.0 || p_dsamax_e_mc <= 0.0) {
        return 0.0;
    }

    x_p = p_dsainj_p_mc / p_dsamax_p_mc;
    x_e = p_dsainj_e_mc / p_dsamax_e_mc;

    /* K_ep is defined at the same physical momentum P.
       Since f_s here is per d^3p_s with p_s = P/(m_s c),
       we must include the Jacobian factor (m_e/m_p)^3. */
    mass_factor = 1.0 / (dsa_mass_ratio_mp_me()
                       * dsa_mass_ratio_mp_me()
                       * dsa_mass_ratio_mp_me());

    return K_ep * mass_factor * A_p * exp(x_e * x_e - x_p * x_p);
}


double dsa_kep_left_edge(double A_e,
                         double p_dsainj_e_mc,
                         double p_dsamax_e_mc,
                         double A_p,
                         double p_dsainj_p_mc,
                         double p_dsamax_p_mc)
{
    double x_e, x_p;
    double f_e_left, f_p_left;
    double mass_ratio_cubed;

    if (A_e <= 0.0 || A_p <= 0.0) {
        return 0.0;
    }
    if (p_dsainj_e_mc <= 0.0 || p_dsamax_e_mc <= 0.0 ||
        p_dsainj_p_mc <= 0.0 || p_dsamax_p_mc <= 0.0) {
        return 0.0;
    }

    x_e = p_dsainj_e_mc / p_dsamax_e_mc;
    x_p = p_dsainj_p_mc / p_dsamax_p_mc;

    f_e_left = A_e * exp(-(x_e * x_e));
    f_p_left = A_p * exp(-(x_p * x_p));

    if (f_p_left <= 0.0) {
        return 0.0;
    }

    mass_ratio_cubed = dsa_mass_ratio_mp_me()
                     * dsa_mass_ratio_mp_me()
                     * dsa_mass_ratio_mp_me();

    /* K_ep = F_e(P)/F_p(P) = (m_p/m_e)^3 * f_e/f_p */
    return mass_ratio_cubed * (f_e_left / f_p_left);
}


static double dsa_electron_number_integrand_unit(double p_mc,
                                                 double p_dsainj_e_mc,
                                                 double q_e,
                                                 double p_dsamax_e_mc)
{
    if (p_mc <= 0.0 || p_dsainj_e_mc <= 0.0 || p_dsamax_e_mc <= 0.0) {
        return 0.0;
    }
    if (p_mc < p_dsainj_e_mc) {
        return 0.0;
    }

    return p_mc * p_mc
         * dsa_spectrum_shape(p_mc, p_dsainj_e_mc, q_e, p_dsamax_e_mc);
}


double dsa_electron_ndens(double A_e,
                          double p_dsainj_e_mc,
                          double q_e,
                          double p_dsamax_e_mc,
                          int nstep)
{
    return A_e * dsa_electron_ndens_unit(p_dsainj_e_mc,
                                         q_e,
                                         p_dsamax_e_mc,
                                         nstep);
}

double dsa_xi_e_downstream(double A_e,
                           double p_dsainj_e_mc,
                           double q_e,
                           double p_dsamax_e_mc,
                           double rho1,
                           double mach,
                           double gamma_gas,
                           double mu_e,
                           int nstep)
{
    double r, rho2, n_e2, n_cre;

    if (A_e <= 0.0 || rho1 <= 0.0 || mu_e <= 0.0) {
        return 0.0;
    }

    r = dsa_compression_ratio(mach, gamma_gas);
    rho2 = r * rho1;
    n_e2 = rho2 / (mu_e * DSA_M_PROTON);

    if (n_e2 <= 0.0) {
        return 0.0;
    }

    n_cre = dsa_electron_ndens(A_e,
                               p_dsainj_e_mc,
                               q_e,
                               p_dsamax_e_mc,
                               nstep);

    return n_cre / n_e2;
}

double dsa_chi_e_from_proton_inj(double p_dsainj_e_mc,
                                 double temperature_K)
{
    double p_the_mc;

    if (p_dsainj_e_mc <= 0.0 || temperature_K <= 0.0) {
        return 0.0;
    }

    p_the_mc = dsa_electron_thermal_momentum_mc(temperature_K);
    if (p_the_mc <= 0.0) {
        return 0.0;
    }

    return p_dsainj_e_mc / p_the_mc;
}


//   See Kang (2024)  //
double dsa_analytic_xi_e_from_chi(double chi_e, double q)
{
    if (chi_e <= 0.0 || q <= 3.0) {
        return 0.0;
    }

    return (4.0 / sqrt(DSA_FOUR_PI))
         * chi_e * chi_e * chi_e
         * exp(-chi_e * chi_e)
         / (q - 3.0);
}

double dsa_xi_e_from_kep_model(double A_e,
                               double p_dsainj_e_mc,
                               double q_e,
                               double p_dsamax_e_mc,
                               double rho1,
                               double mach,
                               double gamma_gas,
                               double mu_e,
                               int nstep)
{
    double r, rho2, n_e2, n_cre;

    if (A_e <= 0.0 || rho1 <= 0.0 || mu_e <= 0.0) {
        return 0.0;
    }

    r = dsa_compression_ratio(mach, gamma_gas);
    rho2 = r * rho1;
    n_e2 = rho2 / (mu_e * DSA_M_PROTON);

    if (n_e2 <= 0.0) {
        return 0.0;
    }

    n_cre = dsa_electron_ndens(A_e,
                               p_dsainj_e_mc,
                               q_e,
                               p_dsamax_e_mc,
                               nstep);

    return n_cre / n_e2;
}



double dsa_electron_ndens_unit(double p_dsainj_e_mc,
                               double q_e,
                               double p_dsamax_e_mc,
                               int nstep)
{
    int i;
    double h, p, sum;

    if (p_dsainj_e_mc <= 0.0 || p_dsamax_e_mc <= 0.0) {
        return 0.0;
    }
    if (p_dsamax_e_mc <= p_dsainj_e_mc) {
        return 0.0;
    }

    if (nstep < 2) {
        nstep = 2000;
    }
    if (nstep % 2 != 0) {
        nstep += 1;
    }

    h = (p_dsamax_e_mc - p_dsainj_e_mc) / (double)nstep;

    sum = dsa_electron_number_integrand_unit(p_dsainj_e_mc, p_dsainj_e_mc, q_e, p_dsamax_e_mc)
        + dsa_electron_number_integrand_unit(p_dsamax_e_mc, p_dsainj_e_mc, q_e, p_dsamax_e_mc);

    for (i = 1; i < nstep; i++) {
        p = p_dsainj_e_mc + h * (double)i;
        if (i % 2 == 0) {
            sum += 2.0 * dsa_electron_number_integrand_unit(p, p_dsainj_e_mc, q_e, p_dsamax_e_mc);
        } else {
            sum += 4.0 * dsa_electron_number_integrand_unit(p, p_dsainj_e_mc, q_e, p_dsamax_e_mc);
        }
    }

    return DSA_FOUR_PI * sum * h / 3.0;
}

double dsa_electron_norm_from_kep_at_pinj_p(double K_ep,
                                            double A_p,
                                            double p_dsainj_p_mc,
                                            double q_p,
                                            double p_dsamax_p_mc,
                                            double p_dsainj_e_mc,
                                            double q_e,
                                            double p_dsamax_e_mc)
{
    double p_pref_p_mc, p_pref_e_mc;
    double shape_p_ref, shape_e_ref;
    double mass_factor;

    if (K_ep <= 0.0 || A_p <= 0.0) {
        return 0.0;
    }
    if (p_dsainj_p_mc <= 0.0 || p_dsamax_p_mc <= 0.0) {
        return 0.0;
    }
    if (p_dsainj_e_mc <= 0.0 || p_dsamax_e_mc <= 0.0) {
        return 0.0;
    }

    /* Reference physical momentum: P_ref = p_inj,p * m_p c */
    p_pref_p_mc = p_dsainj_p_mc;
    p_pref_e_mc = dsa_mass_ratio_mp_me() * p_dsainj_p_mc;

    shape_p_ref = dsa_spectrum_shape(p_pref_p_mc,
                                     p_dsainj_p_mc,
                                     q_p,
                                     p_dsamax_p_mc);

    shape_e_ref = dsa_spectrum_shape(p_pref_e_mc,
                                     p_dsainj_e_mc,
                                     q_e,
                                     p_dsamax_e_mc);

    if (shape_p_ref <= 0.0 || shape_e_ref <= 0.0) {
        return 0.0;
    }

    /* K_ep = F_e(P_ref)/F_p(P_ref)
       with F(P) = f(p)/(m c)^3 */
    mass_factor = 1.0 / (dsa_mass_ratio_mp_me()
                       * dsa_mass_ratio_mp_me()
                       * dsa_mass_ratio_mp_me());

    return K_ep * mass_factor * A_p * shape_p_ref / shape_e_ref;
}

double dsa_electron_p_from_proton_p(double p_p_mc)
{
    if (p_p_mc <= 0.0) {
        return 0.0;
    }

    return dsa_mass_ratio_mp_me() * p_p_mc;
}

double dsa_electron_ndens_from_pmin_unit(double pmin_e_mc,
                                         double p_dsainj_e_mc,
                                         double q_e,
                                         double p_dsamax_e_mc,
                                         int nstep)
{
    int i;
    double h, x, p, sum;
    double p_lo;
    double x_lo, x_hi;

    if (p_dsainj_e_mc <= 0.0 || p_dsamax_e_mc <= 0.0) {
        return 0.0;
    }

    p_lo = pmin_e_mc;
    if (p_lo < p_dsainj_e_mc) {
        p_lo = p_dsainj_e_mc;
    }

    if (p_lo >= p_dsamax_e_mc) {
        return 0.0;
    }

    if (nstep < 2) {
        nstep = 2000;
    }
    if (nstep % 2 != 0) {
        nstep += 1;
    }

    x_lo = log(p_lo);
    x_hi = log(p_dsamax_e_mc);
    if (!(x_hi > x_lo)) {
        return 0.0;
    }

    h = (x_hi - x_lo) / (double)nstep;

    sum = p_lo * dsa_electron_number_integrand_unit(p_lo, p_dsainj_e_mc, q_e, p_dsamax_e_mc)
        + p_dsamax_e_mc * dsa_electron_number_integrand_unit(p_dsamax_e_mc,
                                                             p_dsainj_e_mc,
                                                             q_e,
                                                             p_dsamax_e_mc);

    for (i = 1; i < nstep; i++) {
        x = x_lo + h * (double)i;
        p = exp(x);
        if (i % 2 == 0) {
            sum += 2.0 * p
                 * dsa_electron_number_integrand_unit(p, p_dsainj_e_mc, q_e, p_dsamax_e_mc);
        } else {
            sum += 4.0 * p
                 * dsa_electron_number_integrand_unit(p, p_dsainj_e_mc, q_e, p_dsamax_e_mc);
        }
    }

    return DSA_FOUR_PI * sum * h / 3.0;
}

double dsa_electron_ndens_from_pmin(double A_e,
                                    double pmin_e_mc,
                                    double p_dsainj_e_mc,
                                    double q_e,
                                    double p_dsamax_e_mc,
                                    int nstep)
{
    return A_e * dsa_electron_ndens_from_pmin_unit(pmin_e_mc,
                                                   p_dsainj_e_mc,
                                                   q_e,
                                                   p_dsamax_e_mc,
                                                   nstep);
}

double dsa_xi_e_downstream_from_pmin(double A_e,
                                     double pmin_e_mc,
                                     double p_dsainj_e_mc,
                                     double q_e,
                                     double p_dsamax_e_mc,
                                     double rho1,
                                     double mach,
                                     double gamma_gas,
                                     double mu_e,
                                     int nstep)
{
    double r, rho2, n_e2, n_cre;

    if (A_e <= 0.0 || rho1 <= 0.0 || mu_e <= 0.0) {
        return 0.0;
    }

    r = dsa_compression_ratio(mach, gamma_gas);
    rho2 = r * rho1;
    n_e2 = rho2 / (mu_e * DSA_M_PROTON);

    if (n_e2 <= 0.0) {
        return 0.0;
    }

    n_cre = dsa_electron_ndens_from_pmin(A_e,
                                         pmin_e_mc,
                                         p_dsainj_e_mc,
                                         q_e,
                                         p_dsamax_e_mc,
                                         nstep);

    return n_cre / n_e2;
}


int dsa_init_grid(DSAGrid *grid,
                const CRspectrum *CRp,
                const CRspectrum *CRe,
                double pmin_dsa_p_log10,
                double pmin_dsa_e_log10)
{
    double dlogp_p, dlogp_e;
    double pmin_fp_p_log10, pmin_fp_e_log10;
    int n_extra_p, n_extra_e;
    int nbin_p, nbin_e;

    if (grid == NULL || CRp == NULL || CRe == NULL) {
        return 0;
    }

    dsa_free_grid(grid);

    dlogp_p = dsa_grid_dlogp_from_cr(CRp);
    dlogp_e = dsa_grid_dlogp_from_cr(CRe);

    if (dlogp_p <= 0.0 || dlogp_e <= 0.0) {
        return 0;
    }

    if (CRp->p[0] <= 0.0 || CRe->p[0] <= 0.0) {
        return 0;
    }

    pmin_fp_p_log10 = log10(CRp->p[0]);
    pmin_fp_e_log10 = log10(CRe->p[0]);

    n_extra_p = dsa_extra_bins_needed(pmin_fp_p_log10, pmin_dsa_p_log10, dlogp_p);
    n_extra_e = dsa_extra_bins_needed(pmin_fp_e_log10, pmin_dsa_e_log10, dlogp_e);

    nbin_p = np + n_extra_p;
    nbin_e = npe + n_extra_e;

    if (!dsa_allocate_grid1d(&grid->proton, nbin_p)) {
        dsa_free_grid(grid);
        return 0;
    }

    if (!dsa_allocate_grid1d(&grid->electron, nbin_e)) {
        dsa_free_grid(grid);
        return 0;
    }

    if (!dsa_fill_grid1d_from_spacing(&grid->proton,
                                      pmin_fp_p_log10 - dlogp_p * (double)n_extra_p,
                                      dlogp_p,
                                      DSA_M_PROTON)) {
        dsa_free_grid(grid);
        return 0;
    }

    if (!dsa_fill_grid1d_from_spacing(&grid->electron,
                                      pmin_fp_e_log10 - dlogp_e * (double)n_extra_e,
                                      dlogp_e,
                                      DSA_M_ELECTRON)) {
        dsa_free_grid(grid);
        return 0;
    }

    grid->proton.offset_to_fp = n_extra_p;
    grid->electron.offset_to_fp = n_extra_e;
    grid->initialized = 1;

    return 1;
}



int dsa_build_proton_spectrum(const DSAShockState *shock,
                              const DSAInjectionModel *model,
                              DSASpectrum *spec_p)
{
    double eta_p;
    double q_p;
    double p_inj_p_mc;
    double A_p;

    if (shock == NULL || model == NULL || spec_p == NULL) {
        return 0;
    }

    if (shock->mach <= 1.0 || shock->gamma_gas <= 1.0) {
        return 0;
    }
    if (shock->rho1 <= 0.0 || shock->u1 <= 0.0 || shock->u2 <= 0.0 || shock->temperature <= 0.0) {
        return 0;
    }
    if (model->chi_p <= 0.0 || model->p_dsamax_p_mc <= 0.0) {
        return 0;
    }

    /* initial-acceleration efficiency only; reacceleration ignored for now */
    eta_p = dsa_eta_from_id(shock->mach, model->eta_model_initial);
    q_p = dsa_momentum_slope_d3p(shock->mach, shock->gamma_gas);
    p_inj_p_mc = dsa_proton_injection_momentum_mc(shock->temperature, model->chi_p);

    if (eta_p <= 0.0 || q_p <= 0.0 || p_inj_p_mc <= 0.0) {
        return 0;
    }
    if (model->p_dsamax_p_mc <= p_inj_p_mc) {
        return 0;
    }

    A_p = dsa_proton_norm_from_eta(eta_p,
                                            shock->rho1,
                                            shock->u1,
                                            shock->u2,
                                            p_inj_p_mc,
                                            q_p,
                                            model->p_dsamax_p_mc,
                                            4000);

    if (A_p <= 0.0) {
        return 0;
    }

    spec_p->species  = DSA_SPEC_PROTON;
    spec_p->q        = q_p;
    spec_p->A_f      = A_p;
    spec_p->p_inj_mc = p_inj_p_mc;
    spec_p->p_max_mc = model->p_dsamax_p_mc;

    return 1;
}

int dsa_build_electron_spectrum(const DSAShockState *shock,
                                const DSAInjectionModel *model,
                                const DSASpectrum *spec_p,
                                DSASpectrum *spec_e)
{
    double q_e;
    double p_inj_e_mc;
    double A_e;

    if (shock == NULL || model == NULL || spec_p == NULL || spec_e == NULL) {
        return 0;
    }

    if (spec_p->species != DSA_SPEC_PROTON) {
        return 0;
    }

    if (shock->temperature <= 0.0 || model->p_dsamax_e_mc <= 0.0) {
        return 0;
    }

    q_e = spec_p->q;

    if (model->electron_mode == DSA_ELECTRON_INJ_SAME_PHYSICAL_PINJ) {

        p_inj_e_mc = dsa_electron_inj_from_proton_inj(spec_p->p_inj_mc);

        if (p_inj_e_mc <= 0.0 || model->p_dsamax_e_mc <= p_inj_e_mc) {
            return 0;
        }

        A_e = dsa_electron_norm_from_kep_left(model->K_ep,
                                                            spec_p->A_f,
                                                            spec_p->p_inj_mc,
                                                            spec_p->p_max_mc,
                                                            p_inj_e_mc,
                                                            model->p_dsamax_e_mc);

    } else if (model->electron_mode == DSA_ELECTRON_INJ_SUPRATHERMAL) {

        if (model->chi_e <= 0.0) {
            return 0;
        }

        p_inj_e_mc = dsa_electron_injection_momentum_mc(shock->temperature, model->chi_e);

        if (p_inj_e_mc <= 0.0 || model->p_dsamax_e_mc <= p_inj_e_mc) {
            return 0;
        }

        A_e = dsa_electron_norm_from_kep_at_pinj_p(model->K_ep,
                                                                      spec_p->A_f,
                                                                      spec_p->p_inj_mc,
                                                                      spec_p->q,
                                                                      spec_p->p_max_mc,
                                                                      p_inj_e_mc,
                                                                      q_e,
                                                                      model->p_dsamax_e_mc);

    } else {
        return 0;
    }

    if (A_e <= 0.0) {
        return 0;
    }

    spec_e->species  = DSA_SPEC_ELECTRON;
    spec_e->q        = q_e;
    spec_e->A_f      = A_e;
    spec_e->p_inj_mc = p_inj_e_mc;
    spec_e->p_max_mc = model->p_dsamax_e_mc;

    return 1;
}

double dsa_spectrum_f(const DSASpectrum *spec, double p_mc)
{
    if (spec == NULL) {
        return 0.0;
    }
    if (p_mc <= 0.0) {
        return 0.0;
    }

    /* IMPORTANT:
       The spectrum is defined only for p_inj <= p <= p_max.
       Do NOT extend it below p_inj, because the normalization A_f
       was determined using integrals starting at p_inj. */
    if (p_mc < spec->p_inj_mc) {
        return 0.0;
    }
    if (p_mc > spec->p_max_mc) {
        return 0.0;
    }

    return spec->A_f
         * pow(p_mc / spec->p_inj_mc, -spec->q)
         * exp(-(p_mc / spec->p_max_mc) * (p_mc / spec->p_max_mc));
}

double dsa_spectrum_dp(const DSASpectrum *spec, double p_mc)
{
    double f;

    if (spec == NULL) {
        return 0.0;
    }
    if (p_mc <= 0.0) {
        return 0.0;
    }

    f = dsa_spectrum_f(spec, p_mc);
    if (f <= 0.0) {
        return 0.0;
    }

    return DSA_FOUR_PI * p_mc * p_mc * f;
}

int dsa_fill_source_dp_on_grid(const DSASpectrum *spec,
                               DSAGrid1D *grid)
{
    int j;

    if (spec == NULL || grid == NULL) {
        return 0;
    }
    if (grid->nbin <= 0 || grid->p == NULL || grid->source_dp == NULL) {
        return 0;
    }

    for (j = 0; j < grid->nbin; j++) {
        grid->source_dp[j] = dsa_spectrum_dp(spec, grid->p[j]);
    }

    return 1;
}

int dsa_copy_source_to_fp_grid(const DSAGrid1D *grid_dsa,
                               int nbin_fp,
                               double *source_fp)
{
    int j;
    int j0;

    if (grid_dsa == NULL || source_fp == NULL) {
        return 0;
    }
    if (nbin_fp <= 0 || grid_dsa->nbin <= 0) {
        return 0;
    }

    j0 = grid_dsa->offset_to_fp;

    if (j0 < 0) {
        return 0;
    }
    if (j0 + nbin_fp > grid_dsa->nbin) {
        return 0;
    }

    for (j = 0; j < nbin_fp; j++) {
        source_fp[j] = grid_dsa->source_dp[j + j0];
    }

    return 1;
}


int dsa_build_proton_source_fp(const DSAShockState *shock,
                               const DSAInjectionModel *model,
                               DSAGrid *grid,
                               const CRspectrum *CRp,
                               double *source_p_fp,
                               DSASpectrum *spec_p_out)
{
    DSASpectrum spec_p_local;
    DSASpectrum *spec_p_use;

    if (shock == NULL || model == NULL || grid == NULL || CRp == NULL || source_p_fp == NULL) {
        return 0;
    }
    if (!grid->initialized) {
        return 0;
    }

    spec_p_use = (spec_p_out != NULL) ? spec_p_out : &spec_p_local;

    if (!dsa_build_proton_spectrum(shock, model, spec_p_use)) {
        return 0;
    }

    if (!dsa_fill_source_dp_on_grid(spec_p_use, &grid->proton)) {
        return 0;
    }

    if (!dsa_copy_source_to_fp_grid(&grid->proton, np, source_p_fp)) {
        return 0;
    }

    return 1;
}

int dsa_build_electron_source_fp(const DSAShockState *shock,
                                 const DSAInjectionModel *model,
                                 DSAGrid *grid,
                                 const CRspectrum *CRe,
                                 const DSASpectrum *spec_p,
                                 double *source_e_fp,
                                 DSASpectrum *spec_e_out)
{
    DSASpectrum spec_e_local;
    DSASpectrum *spec_e_use;

    if (shock == NULL || model == NULL || grid == NULL || CRe == NULL ||
        spec_p == NULL || source_e_fp == NULL) {
        return 0;
    }
    if (!grid->initialized) {
        return 0;
    }

    spec_e_use = (spec_e_out != NULL) ? spec_e_out : &spec_e_local;

    if (!dsa_build_electron_spectrum(shock, model, spec_p, spec_e_use)) {
        return 0;
    }

    if (!dsa_fill_source_dp_on_grid(spec_e_use, &grid->electron)) {
        return 0;
    }

    if (!dsa_copy_source_to_fp_grid(&grid->electron, npe, source_e_fp)) {
        return 0;
    }

    return 1;
}
void dsa_free_grid(DSAGrid *grid)
{
    if (grid == NULL) {
        return;
    }

    dsa_free_grid1d(&grid->proton);
    dsa_free_grid1d(&grid->electron);
    grid->initialized = 0;
}
