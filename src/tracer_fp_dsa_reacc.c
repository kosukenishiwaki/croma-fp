#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <math.h>

#include "CONSTANTS.h"
#include "tracer_fp_dsa_reacc.h"

static const double kTracerProtonMassCgs = 1.67262192369e-24;
static const double kTracerElectronMassCgs = 9.1093837015e-28;

static int tracer_dsa_reacc_energy_bounds(int nbin, int *jfirst, int *jlast)
{
    if (jfirst == 0 || jlast == 0) return 0;
    *jfirst = 0;
    *jlast = -1;
    if (nbin <= 4) return 0;
    *jfirst = 2;
    *jlast = nbin - 3;
    return (*jfirst <= *jlast);
}

static double tracer_dsa_reacc_interp_spectrum(const CRspectrum *grid,
                                               const double *state_density,
                                               int nbin,
                                               double p_mc)
{
    int lo;
    int hi;

    if (grid == 0 || state_density == 0 || nbin <= 2 || !(p_mc > 0.0)) return 0.0;
    lo = 1;
    hi = nbin - 2;
    if (p_mc < grid->p[lo] || p_mc > grid->p[hi]) return 0.0;

    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (grid->p[mid] <= p_mc) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    if (grid->p[hi] <= grid->p[lo]) return state_density[lo];
    return state_density[lo] +
           (state_density[hi] - state_density[lo]) *
               (p_mc - grid->p[lo]) / (grid->p[hi] - grid->p[lo]);
}

static double tracer_dsa_reacc_particle_ekin_erg(double p_mc, DSASpecies species)
{
    const double mass = (species == DSA_SPECIES_ELECTRON)
                      ? kTracerElectronMassCgs
                      : kTracerProtonMassCgs;
    return mass * c * c * (sqrt(1.0 + p_mc * p_mc) - 1.0);
}

double tracer_dsa_reacc_positive_delta_1d(const CRspectrum *grid,
                                          const double *state_tracer,
                                          int nbin,
                                          double volume_downstream,
                                          double q,
                                          double pmin_mc,
                                          double *delta_density,
                                          DSASpecies species)
{
    int j;
    /* The lowest FP bin is not robust enough to use as a pre-shock anchor. */
    const int jmin_valid = (nbin > 1) ? 1 : 0;
    double integral = 0.0;
    double delta_energy_density = 0.0;

    if (grid == 0 || state_tracer == 0 || delta_density == 0 || nbin <= 0 ||
        !(volume_downstream > 0.0) || !(q > 0.0)) {
        return 0.0;
    }

    for (j = 0; j < nbin; j++) {
        const double p = grid->p[j];
        double current_density = 0.0;
        double f_pre = 0.0;
        double reacc_density = 0.0;

        delta_density[j] = 0.0;
        if (j < jmin_valid) continue;
        if (!(p > 0.0) || p < pmin_mc) continue;

        current_density = state_tracer[j] / volume_downstream;
        f_pre = current_density / (4.0 * M_PI * p * p);
        integral += pow(p, q - 1.0) * f_pre * grid->dp[j];
        reacc_density = 4.0 * M_PI * p * p * q * pow(p, -q) * integral;

        if (reacc_density > current_density) {
            delta_density[j] = reacc_density - current_density;
            delta_energy_density += tracer_dsa_reacc_particle_ekin_erg(p, species)
                                  * delta_density[j] * grid->dp[j];
        }
    }

    return delta_energy_density;
}

double tracer_dsa_reacc_convolution_delta_1d(const CRspectrum *grid,
                                             const double *state_tracer,
                                             int nbin,
                                             double volume_upstream,
                                             double volume_downstream,
                                             double q,
                                             double pmin_mc,
                                             double *delta_density,
                                             DSASpecies species)
{
    int j;
    /* The lowest FP bin is not robust enough to use as a pre-shock anchor. */
    const int jmin_valid = (nbin > 1) ? 1 : 0;
    double alpha;
    double integral = 0.0;
    double delta_energy_density = 0.0;

    if (grid == 0 || state_tracer == 0 || delta_density == 0 || nbin <= 0 ||
        !(volume_upstream > 0.0) || !(volume_downstream > 0.0) || !(q > 0.0)) {
        return 0.0;
    }
    alpha = q - 2.0;

    for (j = 0; j < nbin; j++) {
        const double p = grid->p[j];
        double seed_spectrum = 0.0;
        double current_downstream_spectrum = 0.0;
        double reacc_spectrum = 0.0;
        double delta = 0.0;

        delta_density[j] = 0.0;
        if (j < jmin_valid) continue;
        if (!(p > 0.0)) continue;

        seed_spectrum = state_tracer[j] / volume_upstream;
        current_downstream_spectrum = state_tracer[j] / volume_downstream;
        if (p < pmin_mc) continue;

        integral += pow(p, alpha - 1.0) * seed_spectrum * grid->dp[j];
        reacc_spectrum = (alpha + 2.0) * pow(p, -alpha) * integral;
        delta = reacc_spectrum - current_downstream_spectrum;

        if (!isfinite(delta)) {
            delta_density[j] = 0.0;
            continue;
        }
        if (delta < -current_downstream_spectrum) {
            delta = -current_downstream_spectrum;
        }

        delta_density[j] = delta;
        delta_energy_density += tracer_dsa_reacc_particle_ekin_erg(p, species)
                              * delta * grid->dp[j];
    }

    return delta_energy_density;
}

double tracer_dsa_reacc_delta_energy_density_1d(const CRspectrum *grid,
                                                const double *delta_density,
                                                int nbin,
                                                double pmin_mc,
                                                DSASpecies species)
{
    int j;
    int jfirst, jlast;
    double delta_energy_density = 0.0;

    if (grid == 0 || delta_density == 0) return 0.0;
    if (!tracer_dsa_reacc_energy_bounds(nbin, &jfirst, &jlast)) return 0.0;

    for (j = jfirst; j <= jlast; j++) {
        const double p = grid->p[j];
        if (!(p > 0.0) || p < pmin_mc) continue;
        if (!isfinite(delta_density[j])) continue;
        delta_energy_density += tracer_dsa_reacc_particle_ekin_erg(p, species)
                              * delta_density[j] * grid->dp[j];
    }

    return delta_energy_density;
}

double tracer_dsa_reacc_adiabatic_delta_1d(const CRspectrum *grid,
                                           const double *state_tracer,
                                           int nbin,
                                           double volume_upstream,
                                           double volume_downstream,
                                           double compression_ratio,
                                           double pmin_mc,
                                           double *delta_density,
                                           DSASpecies species)
{
    int j;
    double p_shift;
    double norm_shift;
    double state_density[np > npe ? np : npe];

    (void)species;

    if (grid == 0 || state_tracer == 0 || delta_density == 0 || nbin <= 0 ||
        !(volume_upstream > 0.0) || !(volume_downstream > 0.0) ||
        !(compression_ratio > 0.0) || !isfinite(compression_ratio)) {
        return 0.0;
    }

    p_shift = cbrt(compression_ratio);
    /* Input state_density is upstream-volume normalized.  Pure adiabatic
     * compression shifts p -> r^(1/3) p and maps the result to downstream
     * density, giving r^(2/3) = p_shift^2. */
    norm_shift = p_shift * p_shift;
    for (j = 0; j < nbin; j++) {
        state_density[j] = state_tracer[j] / volume_upstream;
        delta_density[j] = 0.0;
    }

    for (j = 0; j < nbin; j++) {
        const double p = grid->p[j];
        const double p_up = p / p_shift;
        double current_spectrum;
        double adiabatic_spectrum;

        if (!(p > 0.0) || p < pmin_mc) continue;
        current_spectrum = state_tracer[j] / volume_downstream;
        adiabatic_spectrum = norm_shift *
                             tracer_dsa_reacc_interp_spectrum(grid, state_density, nbin, p_up);
        if (!isfinite(current_spectrum) || !isfinite(adiabatic_spectrum)) continue;
        delta_density[j] = adiabatic_spectrum - current_spectrum;
    }

    return tracer_dsa_reacc_delta_energy_density_1d(grid, delta_density, nbin,
                                                    pmin_mc, species);
}
