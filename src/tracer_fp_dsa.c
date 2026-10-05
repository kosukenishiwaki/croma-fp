/*
    K. Nishiwaki, 2026-06-18
    - DSA(Fermi I) utilities
    - add fresh/reacc CRs in the source terms
    - specific DSA models can be found in DSA_MODELS
*/

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "CONSTANTS.h"
#include "DSA_MODELS.h"
#include "tracer_fp_dsa.h"
#include "tracer_fp_dsa_reacc.h"
#include "tracer_fp_setup.h"

static const double kTracerBoltzmannCgs = 1.380649e-16;
static const double kTracerProtonMassCgs = 1.67262192369e-24;
static const double kTracerElectronMassCgs = 9.1093837015e-28;

static int tracer_dsa_resolve_tracer_input(const TracerFpRawBackgroundSlot *raw,
                                           int itr,
                                           double density_unit_cgs,
                                           double *mach_out,
                                           double *temp1_out,
                                           double *rho1_out,
                                           double *u1_out,
                                           double *flux_out,
                                           int *side_code_out,
                                           int *used_dsa_out)
{
    double mach;
    double temp1;
    double rho1;
    double u1;
    int side_code = 99;
    int use_merged_dsa = 0;
    double flux = 0.0;

    if (mach_out == 0 || temp1_out == 0 || rho1_out == 0 || u1_out == 0) return 0;
    if (flux_out != 0) *flux_out = 0.0;
    if (side_code_out != 0) *side_code_out = side_code;
    if (used_dsa_out != 0) *used_dsa_out = 0;
    if (raw == 0 || itr < 0) return 0;

    if (raw->has_dsa) {
        if (raw->dsa_trigger == 0 || raw->dsa_mach == 0 ||
            raw->dsa_pre_density == 0 || raw->dsa_kinetic_energy_flux_cgs == 0) {
            return 0;
        }
        if (!(raw->dsa_trigger[itr] > 0.5)) return 0;
        flux = raw->dsa_kinetic_energy_flux_cgs[itr];
        use_merged_dsa = 1;
        mach = raw->dsa_mach[itr];
        rho1 = raw->dsa_pre_density[itr] * density_unit_cgs;
        if (!(mach >= DSAMinMach) || !isfinite(mach) || !(rho1 > 0.0) || !isfinite(rho1)) {
            return 0;
        }
        if (!(flux > 0.0) || !isfinite(flux)) return 0;
        u1 = cbrt(2.0 * flux / rho1);
        if (!(u1 > 0.0) || !isfinite(u1)) return 0;
        temp1 = (mu_mol * kTracerProtonMassCgs / (DSAGammaGas * kTracerBoltzmannCgs)) *
                (u1 * u1) / (mach * mach);
        if (!(temp1 > 0.0) || !isfinite(temp1)) return 0;
        side_code = 1;
        if (raw->shock_side_code != 0 && isfinite(raw->shock_side_code[itr])) {
            side_code = (int)floor(raw->shock_side_code[itr] + 0.5);
        }
    } else {
        if (raw->mach == 0 || raw->prestemp == 0 || raw->presden == 0) return 0;
        mach = raw->mach[itr];
        temp1 = raw->prestemp[itr];
        if (!(mach >= DSAMinMach) || !(temp1 > 0.0) || !isfinite(mach) || !isfinite(temp1)) {
            return 0;
        }

        if (raw->shock_side_code != 0) {
            side_code = (int)floor(raw->shock_side_code[itr] + 0.5);
            if (side_code == -1 || side_code == 99) return 0;
        }

        rho1 = (raw->pre_density_cgs != 0 && raw->pre_density_cgs[itr] > 0.0)
             ? raw->pre_density_cgs[itr]
             : raw->presden[itr] * density_unit_cgs;
        if (!(rho1 > 0.0) || !isfinite(rho1)) return 0;

        u1 = (raw->upstream_speed_cgs != 0 && raw->upstream_speed_cgs[itr] > 0.0)
           ? raw->upstream_speed_cgs[itr]
           : mach * sqrt(DSAGammaGas * kTracerBoltzmannCgs * temp1 /
                         (mu_mol * kTracerProtonMassCgs));
        if (!(u1 > 0.0) || !isfinite(u1)) return 0;
        flux = 0.5 * rho1 * u1 * u1 * u1;
    }

    *mach_out = mach;
    *temp1_out = temp1;
    *rho1_out = rho1;
    *u1_out = u1;
    if (flux_out != 0) *flux_out = flux;
    if (side_code_out != 0) *side_code_out = side_code;
    if (used_dsa_out != 0) *used_dsa_out = use_merged_dsa;
    return 1;
}

static double tracer_dsa_proton_ekin_erg(double p_mc)
{
    return kTracerProtonMassCgs * c * c * (sqrt(1.0 + p_mc * p_mc) - 1.0);
}

static double tracer_dsa_electron_ekin_erg(double p_mc)
{
    return kTracerElectronMassCgs * c * c * (sqrt(1.0 + p_mc * p_mc) - 1.0);
}

static double tracer_dsa_energy_density_from_state(const CRspectrum *grid,
                                                   const double *state,
                                                   int nbin,
                                                   double volume_cgs,
                                                   double pmin_mc,
                                                   int is_electron)
{
    int j;
    double e = 0.0;

    if (grid == 0 || state == 0 || nbin <= 0 || !(volume_cgs > 0.0)) return 0.0;
    for (j = 0; j < nbin; j++) {
        const double p = grid->p[j];
        const double ekin = is_electron ? tracer_dsa_electron_ekin_erg(p)
                                        : tracer_dsa_proton_ekin_erg(p);
        if (p < pmin_mc) continue;
        e += ekin * state[j] * grid->dp[j];
    }
    return e / volume_cgs;
}

static double tracer_dsa_energy_density_from_source(const CRspectrum *grid,
                                                    const double *source_dp,
                                                    int nbin,
                                                    int is_electron)
{
    int j;
    double e = 0.0;

    if (grid == 0 || source_dp == 0 || nbin <= 0) return 0.0;
    for (j = 0; j < nbin; j++) {
        const double p = grid->p[j];
        const double ekin = is_electron ? tracer_dsa_electron_ekin_erg(p)
                                        : tracer_dsa_proton_ekin_erg(p);
        e += ekin * source_dp[j] * grid->dp[j];
    }
    return e;
}

static double tracer_dsa_xcr_proton(const CRspectrum *crp_grid,
                                    const double *crp_state_tracer,
                                    double volume_downstream,
                                    double rho1,
                                    double temp1)
{
    double ecr;
    double pcr;
    double pth;

    if (crp_grid == 0 || crp_state_tracer == 0 || !(volume_downstream > 0.0) ||
        !(rho1 > 0.0) || !(temp1 > 0.0)) {
        return 0.0;
    }

    ecr = tracer_dsa_energy_density_from_state(crp_grid, crp_state_tracer, np,
                                               volume_downstream,
                                               DSAXcrPminPmc, 0);
    pcr = ecr / 3.0;
    pth = rho1 * kTracerBoltzmannCgs * temp1 / (mu_mol * kTracerProtonMassCgs);
    if (!(pth > 0.0) || !isfinite(pcr) || !isfinite(pth)) return 0.0;
    if (pcr <= 0.0) return 0.0;
    return pcr / pth;
}

int tracer_fp_apply_dsa_tracer_injection_snapshot(const TracerFpRawBackgroundSlot *raw,
                                                  int ntracer,
                                                  const long int *tracer_ids,
                                                  const double *tracer_mass,
                                                  double density_unit_cgs,
                                                  double dt_gyr,
                                                  TracerDsaInjectionMode mode,
                                                  TracerDsaReaccMode reacc_mode,
                                                  const CRspectrum *crp_grid,
                                                  const CRspectrum *cre_grid,
                                                  DSAGrid *dsa_grid,
                                                  double *crp_state,
                                                  double *cre_state,
                                                  double *qpi_batch,
                                                  double *qepri_batch,
                                                  unsigned char *disable_adiabatic,
                                                  int snapshot_1based,
                                                  int mpi_rank,
                                                  FILE *reacc_debug_fp,
                                                  int *ninjected_out)
{
    DSAInjectionModel model;
    int itr;
    int ninjected = 0;

    if (ninjected_out != 0) *ninjected_out = 0;
    if (raw == 0 || !raw->has_shock || ntracer <= 0 || tracer_mass == 0 ||
        crp_grid == 0 || cre_grid == 0 || dsa_grid == 0 || !dsa_grid->initialized ||
        mode == TRACER_DSA_INJECTION_OFF) {
        return 0;
    }
    if (mode == TRACER_DSA_INJECTION_TRACER_STATE &&
        (crp_state == 0 || cre_state == 0)) {
        return -1;
    }
    if ((mode == TRACER_DSA_INJECTION_TRACER_SOURCE || reacc_mode != TRACER_DSA_REACC_OFF) &&
        (crp_state == 0 || cre_state == 0)) {
        return -1;
    }
    if (mode == TRACER_DSA_INJECTION_TRACER_SOURCE &&
        (qpi_batch == 0 || qepri_batch == 0 || !(dt_gyr > 0.0) || !isfinite(dt_gyr))) {
        return -1;
    }
    if (mode != TRACER_DSA_INJECTION_TRACER_STATE &&
        mode != TRACER_DSA_INJECTION_TRACER_SOURCE) {
        return 0;
    }

    memset(&model, 0, sizeof(model));
    model.eta_model_initial = DSAEtaModelInitial;
    model.chi_p = DSAChiP;
    model.chi_e = (DSAChiE > 0.0) ? DSAChiE : DSAChiP;
    model.K_ep = DSAKep;
    model.p_dsamax_p_mc = DSAPmaxPmc;
    model.p_dsamax_e_mc = DSAPmaxEmc;
    model.electron_mode = DSA_ELECTRON_INJ_SUPRATHERMAL;

    for (itr = 0; itr < ntracer; itr++) {
        DSAShockState shock;
        DSASpectrum spec_p, spec_e;
        double source_p[np];
        double source_e[npe];
        double reacc_delta_p[np];
        double reacc_delta_e[npe];
        double reacc_ad_p[np];
        double reacc_ad_e[npe];
        double mach, temp1, rho1, ratio, rho2, u1, u2, q_shock;
        double kinetic_flux_cgs = 0.0;
        double mass_cgs, volume_upstream, volume_downstream;
        double p_reacc_min_p_mc, p_reacc_min_e_mc;
        double eta_p, target_energy, integrated_energy, energy_ratio;
        double fp_added_energy, fp_added_ratio;
        double x_cr = 0.0;
        double eta_interp = 0.0;
        double eta_extra = 0.0;
        double reacc_budget = 0.0;
        double reacc_energy = 0.0;
        double reacc_nonadiabatic_energy = 0.0;
        double reacc_scale = 0.0;
        double reacc_conv_p_edens = 0.0;
        double reacc_conv_e_edens = 0.0;
        double reacc_ad_p_edens = 0.0;
        double reacc_ad_e_edens = 0.0;
        double reacc_nonadiabatic_p_energy = 0.0;
        double reacc_nonadiabatic_e_energy = 0.0;
        double seed_p_energy = 0.0;
        double seed_e_energy = 0.0;
        int reacc_cap_hit = 0;
        double rho_local_ratio = 0.0;
        int side_code = 0;
        int used_dsa = 0;
        int applied = 0;
        const int inject_proton = (DSAInjectSpecies != DSA_INJECT_SPECIES_ELECTRON);
        const int inject_electron = (DSAInjectSpecies != DSA_INJECT_SPECIES_PROTON);
        int have_fresh_p = 0;
        int have_fresh_e = 0;
        int j;
        const double *crp_state_pre = crp_state + (size_t)itr * (size_t)np;
        const double *cre_state_pre = cre_state + (size_t)itr * (size_t)npe;
        const int needs_drury83 = (reacc_mode == TRACER_DSA_REACC_DRURY83);

        if (!tracer_dsa_resolve_tracer_input(raw, itr, density_unit_cgs,
                                             &mach, &temp1, &rho1, &u1,
                                             &kinetic_flux_cgs,
                                             &side_code, &used_dsa)) {
            continue;
        }

        ratio = dsa_compression_ratio(mach, DSAGammaGas);
        if (!(ratio > 0.0) || !isfinite(ratio)) continue;
        rho2 = ratio * rho1;
        u2 = u1 / ratio;

        memset(&shock, 0, sizeof(shock));
        memset(source_p, 0, sizeof(source_p));
        memset(source_e, 0, sizeof(source_e));
        memset(reacc_delta_p, 0, sizeof(reacc_delta_p));
        memset(reacc_delta_e, 0, sizeof(reacc_delta_e));
        memset(reacc_ad_p, 0, sizeof(reacc_ad_p));
        memset(reacc_ad_e, 0, sizeof(reacc_ad_e));
        shock.mach = mach;
        shock.gamma_gas = DSAGammaGas;
        shock.rho1 = rho1;
        shock.u1 = u1;
        shock.u2 = u2;
        shock.temperature = temp1;
        q_shock = dsa_momentum_slope_d3p(mach, DSAGammaGas);
        if (!(q_shock > 0.0) || !isfinite(q_shock)) continue;
        p_reacc_min_p_mc = dsa_proton_injection_momentum_mc(temp1, model.chi_p);
        p_reacc_min_e_mc = dsa_electron_injection_momentum_mc(temp1, model.chi_e);
        if (!(p_reacc_min_p_mc > 0.0) || !isfinite(p_reacc_min_p_mc)) continue;
        if (!(p_reacc_min_e_mc > 0.0) || !isfinite(p_reacc_min_e_mc)) continue;

        have_fresh_p = dsa_build_proton_source_fp(&shock, &model, dsa_grid, crp_grid, source_p, &spec_p);
        if (have_fresh_p && inject_electron) {
            have_fresh_e = dsa_build_electron_source_fp(&shock, &model, dsa_grid, cre_grid,
                                                        &spec_p, source_e, &spec_e);
        }

        mass_cgs = tracer_mass[itr] * M_sun;
        volume_upstream = mass_cgs / rho1;
        volume_downstream = mass_cgs / rho2;
        if (!(volume_upstream > 0.0) || !isfinite(volume_upstream)) continue;
        if (!(volume_downstream > 0.0) || !isfinite(volume_downstream)) continue;
        seed_p_energy = tracer_dsa_energy_density_from_state(crp_grid, crp_state_pre, np,
                                                             volume_upstream,
                                                             p_reacc_min_p_mc, 0)
                      * volume_upstream;
        seed_e_energy = tracer_dsa_energy_density_from_state(cre_grid, cre_state_pre, npe,
                                                             volume_upstream,
                                                             p_reacc_min_e_mc, 1)
                      * volume_upstream;

        if (reacc_mode == TRACER_DSA_REACC_POSITIVE_DELTA) {
            x_cr = tracer_dsa_xcr_proton(crp_grid, crp_state_pre,
                                         volume_downstream, rho1, temp1);
            eta_p = dsa_eta_from_id(mach, model.eta_model_initial);
            eta_interp = dsa_eta_interpolated(mach,
                                              DSAEtaModelInitial,
                                              DSAEtaModelReacc,
                                              x_cr);
            eta_extra = eta_interp - eta_p;
            if (eta_extra < 0.0 || !isfinite(eta_extra)) eta_extra = 0.0;
            reacc_budget = eta_extra * 0.5 * mass_cgs * u1 * u1;

            if (reacc_budget > 0.0) {
                double e_reacc_p, e_reacc_e;
                e_reacc_p = tracer_dsa_reacc_positive_delta_1d(
                    crp_grid, crp_state_pre, np,
                    volume_downstream, q_shock, p_reacc_min_p_mc, reacc_delta_p, DSA_SPECIES_PROTON);
                e_reacc_e = tracer_dsa_reacc_positive_delta_1d(
                    cre_grid, cre_state_pre, npe,
                    volume_downstream, q_shock, p_reacc_min_e_mc, reacc_delta_e, DSA_SPECIES_ELECTRON);
                reacc_energy = (e_reacc_p + e_reacc_e) * volume_downstream;
                if (reacc_energy > 0.0 && isfinite(reacc_energy)) {
                    reacc_scale = (reacc_energy > reacc_budget) ? reacc_budget / reacc_energy : 1.0;
                } else {
                    memset(reacc_delta_p, 0, sizeof(reacc_delta_p));
                    memset(reacc_delta_e, 0, sizeof(reacc_delta_e));
                }
            }
        } else if (needs_drury83) {
            tracer_dsa_reacc_convolution_delta_1d(crp_grid, crp_state_pre, np,
                                                  volume_upstream, volume_downstream, q_shock,
                                                  p_reacc_min_p_mc, reacc_delta_p, DSA_SPECIES_PROTON);
            tracer_dsa_reacc_convolution_delta_1d(cre_grid, cre_state_pre, npe,
                                                  volume_upstream, volume_downstream, q_shock,
                                                  p_reacc_min_e_mc, reacc_delta_e, DSA_SPECIES_ELECTRON);
            reacc_conv_p_edens = tracer_dsa_reacc_delta_energy_density_1d(
                crp_grid, reacc_delta_p, np, p_reacc_min_p_mc, DSA_SPECIES_PROTON);
            reacc_conv_e_edens = tracer_dsa_reacc_delta_energy_density_1d(
                cre_grid, reacc_delta_e, npe, p_reacc_min_e_mc, DSA_SPECIES_ELECTRON);
            reacc_energy = (reacc_conv_p_edens + reacc_conv_e_edens) * volume_downstream;
            reacc_scale = 1.0;

            if (DSAReaccEtaCap > 0.0 && isfinite(DSAReaccEtaCap)) {
                reacc_budget = DSAReaccEtaCap * 0.5 * mass_cgs * u1 * u1;
                reacc_ad_p_edens = tracer_dsa_reacc_adiabatic_delta_1d(
                    crp_grid, crp_state_pre, np, volume_upstream, volume_downstream, ratio,
                    p_reacc_min_p_mc, reacc_ad_p, DSA_SPECIES_PROTON);
                reacc_ad_e_edens = tracer_dsa_reacc_adiabatic_delta_1d(
                    cre_grid, cre_state_pre, npe, volume_upstream, volume_downstream, ratio,
                    p_reacc_min_e_mc, reacc_ad_e, DSA_SPECIES_ELECTRON);
                reacc_nonadiabatic_energy =
                    (reacc_conv_p_edens + reacc_conv_e_edens
                     - reacc_ad_p_edens - reacc_ad_e_edens) * volume_downstream;
                reacc_nonadiabatic_p_energy =
                    (reacc_conv_p_edens - reacc_ad_p_edens) * volume_downstream;
                reacc_nonadiabatic_e_energy =
                    (reacc_conv_e_edens - reacc_ad_e_edens) * volume_downstream;

                if (reacc_nonadiabatic_energy > 0.0 &&
                    isfinite(reacc_nonadiabatic_energy) &&
                    reacc_budget > 0.0 && isfinite(reacc_budget)) {
                    if (reacc_nonadiabatic_energy > reacc_budget) {
                        reacc_scale = reacc_budget / reacc_nonadiabatic_energy;
                        reacc_cap_hit = 1;
                    }
                    for (j = 0; j < np; j++) {
                        reacc_delta_p[j] = reacc_ad_p[j] +
                                           reacc_scale * (reacc_delta_p[j] - reacc_ad_p[j]);
                    }
                    for (j = 0; j < npe; j++) {
                        reacc_delta_e[j] = reacc_ad_e[j] +
                                           reacc_scale * (reacc_delta_e[j] - reacc_ad_e[j]);
                    }
                    reacc_energy =
                        (reacc_ad_p_edens + reacc_ad_e_edens +
                         reacc_scale * (reacc_conv_p_edens + reacc_conv_e_edens
                                        - reacc_ad_p_edens - reacc_ad_e_edens))
                        * volume_downstream;
                }
            }
            if (disable_adiabatic != 0) {
                disable_adiabatic[itr] = 1;
            }
        }

        if (reacc_debug_fp != 0 &&
            (reacc_mode == TRACER_DSA_REACC_POSITIVE_DELTA || needs_drury83)) {
            const double e_conv_p = reacc_conv_p_edens * volume_downstream;
            const double e_conv_e = reacc_conv_e_edens * volume_downstream;
            const double e_ad_p = reacc_ad_p_edens * volume_downstream;
            const double e_ad_e = reacc_ad_e_edens * volume_downstream;
            const double shock_area_dt =
                (rho1 > 0.0 && u1 > 0.0) ? mass_cgs / (rho1 * u1) : 0.0;
            const double flux_budget =
                DSAReaccEtaCap * kinetic_flux_cgs * shock_area_dt;
            fprintf(reacc_debug_fp,
                    "%d\t%d\t%ld\t%s\t%s\t%s\t%d\t%d\t"
                    "%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t"
                    "%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t"
                    "%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%.8e\t%d\t%d\t%d\n",
                    mpi_rank, snapshot_1based,
                    tracer_ids != 0 ? tracer_ids[itr] : (long int)itr,
                    tracer_fp_dsa_injection_mode_name(mode),
                    tracer_fp_dsa_reacc_mode_name(reacc_mode),
                    used_dsa ? "merge_dsa" : "shock_fields",
                    side_code, DSAInjectSpecies,
                    mach, ratio, rho1, rho2, u1, kinetic_flux_cgs,
                    mass_cgs, volume_upstream, volume_downstream,
                    shock_area_dt, flux_budget,
                    DSAReaccEtaCap, reacc_budget, reacc_energy,
                    reacc_nonadiabatic_energy,
                    (reacc_budget > 0.0) ? reacc_nonadiabatic_energy / reacc_budget : 0.0,
                    reacc_scale,
                    seed_p_energy, seed_e_energy,
                    e_conv_p, e_conv_e, e_ad_p, e_ad_e,
                    reacc_nonadiabatic_p_energy, reacc_nonadiabatic_e_energy,
                    reacc_cap_hit, inject_proton, inject_electron);
        }

        if (mode == TRACER_DSA_INJECTION_TRACER_STATE) {
            if (inject_proton && have_fresh_p) {
                for (j = 0; j < np; j++) {
                    crp_state[(size_t)itr * (size_t)np + (size_t)j] += source_p[j] * volume_downstream;
                }
                applied = 1;
            }
            if (inject_electron && have_fresh_p && have_fresh_e) {
                for (j = 0; j < npe; j++) {
                    cre_state[(size_t)itr * (size_t)npe + (size_t)j] += source_e[j] * volume_downstream;
                }
                applied = 1;
            }
        } else {
            const double inv_dt = 1.0 / dt_gyr;
            if (inject_proton && have_fresh_p) {
                for (j = 0; j < np; j++) {
                    qpi_batch[(size_t)itr * (size_t)np + (size_t)j] +=
                        source_p[j] * volume_downstream * inv_dt;
                }
                applied = 1;
            }
            if (inject_electron && have_fresh_p && have_fresh_e) {
                for (j = 0; j < npe; j++) {
                    qepri_batch[(size_t)itr * (size_t)npe + (size_t)j] +=
                        source_e[j] * volume_downstream * inv_dt;
                }
                applied = 1;
            }
        }

        if (reacc_mode == TRACER_DSA_REACC_POSITIVE_DELTA || needs_drury83) {
            const double reacc_apply_scale = needs_drury83 ? 1.0 : reacc_scale;
            if (mode == TRACER_DSA_INJECTION_TRACER_STATE) {
                for (j = 0; j < np; j++) {
                    crp_state[(size_t)itr * (size_t)np + (size_t)j] +=
                        reacc_delta_p[j] * volume_downstream * reacc_apply_scale;
                }
                for (j = 0; j < npe; j++) {
                    cre_state[(size_t)itr * (size_t)npe + (size_t)j] +=
                        reacc_delta_e[j] * volume_downstream * reacc_apply_scale;
                }
            } else {
                const double inv_dt = 1.0 / dt_gyr;
                for (j = 0; j < np; j++) {
                    qpi_batch[(size_t)itr * (size_t)np + (size_t)j] +=
                        reacc_delta_p[j] * volume_downstream * reacc_apply_scale * inv_dt;
                }
                for (j = 0; j < npe; j++) {
                    qepri_batch[(size_t)itr * (size_t)npe + (size_t)j] +=
                        reacc_delta_e[j] * volume_downstream * reacc_apply_scale * inv_dt;
                }
            }
            if (reacc_scale > 0.0 && isfinite(reacc_energy) && fabs(reacc_energy) > 0.0) {
                applied = 1;
            }
        }

        if (applied) ninjected++;

        if (DSADebug) {
            eta_p = dsa_eta_from_id(mach, model.eta_model_initial);
            target_energy = eta_p * 0.5 * mass_cgs * u1 * u1;
            integrated_energy = have_fresh_p
                              ? dsa_proton_edens(spec_p.A_f,
                                                                    spec_p.p_inj_mc,
                                                                    spec_p.q,
                                                                    spec_p.p_max_mc,
                                                                    4000) * volume_downstream
                              : 0.0;
            energy_ratio = (target_energy > 0.0) ? integrated_energy / target_energy : 0.0;
            fp_added_energy = (have_fresh_p && have_fresh_e)
                            ? tracer_dsa_energy_density_from_source(crp_grid, source_p, np, 0)
                            * volume_downstream
                            : 0.0;
            fp_added_ratio = (target_energy > 0.0) ? fp_added_energy / target_energy : 0.0;
            if (raw->rho != 0 && rho2 > 0.0) rho_local_ratio = raw->rho[itr] / rho2;
            fprintf(stderr,
                    "[DSA inject] snapshot=%d tracer=%ld mach=%e side=%d source=%s "
                    "mode=%s reacc=%s rho1=%e rho2=%e rho_local/rho2=%e u1=%e "
                    "volume_up=%e volume_down=%e dt=%e eta=%e Xcr=%e eta_interp=%e "
                    "reacc_eta_cap=%e reacc_budget=%e reacc_energy=%e "
                    "reacc_nonadiabatic_energy=%e reacc_budget_ratio=%e "
                    "reacc_cap_hit=%d reacc_scale=%e "
                    "Econv_p=%e Econv_e=%e Ead_p=%e Ead_e=%e "
                    "Enonad_p=%e Enonad_e=%e "
                    "Etotal/Etarget=%e Efp/Etarget=%e\n",
                    snapshot_1based, tracer_ids != 0 ? tracer_ids[itr] : (long int)itr, mach, side_code,
                    used_dsa ? "merge_dsa" : "shock_fields",
                    tracer_fp_dsa_injection_mode_name(mode),
                    tracer_fp_dsa_reacc_mode_name(reacc_mode),
                    rho1, rho2, rho_local_ratio, u1, volume_upstream, volume_downstream, dt_gyr,
                    eta_p, x_cr, eta_interp,
                    DSAReaccEtaCap, reacc_budget, reacc_energy,
                    reacc_nonadiabatic_energy,
                    (reacc_budget > 0.0) ? reacc_nonadiabatic_energy / reacc_budget : 0.0,
                    reacc_cap_hit, reacc_scale,
                    reacc_conv_p_edens * volume_downstream,
                    reacc_conv_e_edens * volume_downstream,
                    reacc_ad_p_edens * volume_downstream,
                    reacc_ad_e_edens * volume_downstream,
                    reacc_nonadiabatic_p_energy,
                    reacc_nonadiabatic_e_energy,
                    energy_ratio, fp_added_ratio);
        }
    }

    if (ninjected_out != 0) *ninjected_out = ninjected;
    return 0;
}
