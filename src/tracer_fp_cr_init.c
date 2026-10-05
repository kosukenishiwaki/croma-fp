/*
    tracer_fp_cr_init.c

    K. Nishiwaki, 2026-06-18
    - initialize CR spectra for each tracer at the start of the simulation
    - spectral index, min momentum, normalization are determined using parameters from param file
*/

#include <stdio.h>
#include <stdlib.h>

#include "FP_Coef.h"
#include "tracer_fp_cr_init.h"
#include "tracer_fp_cr_source.h"
#include "tracer_fp_setup.h"

void tracer_fp_init_state(double *crp_state,
                          double *cre_state,
                          double *qpi_batch,
                          double *qepri_batch,
                          const CRspectrum *crp_grid,
                          const CRspectrum *cre_grid,
                          const double *tracer_mass,
                          double z_init,
                          int ntracer,
                          long int tracer_start,
                          int ntracer_global,
                          long int debug_target_global)
{
    int itr, jp;
    double *crp_template = 0, *cre_template = 0;
    double temp_num_p = 0.0, temp_e_p = 0.0, temp_num_e = 0.0, temp_e_e = 0.0;
    const int debug_enabled = tracer_fp_should_log_root_only() && debug_target_global >= 0;

    (void)ntracer_global;

    if (crp_state == 0 || cre_state == 0 || qpi_batch == 0 || qepri_batch == 0 ||
        crp_grid == 0 || cre_grid == 0 || tracer_mass == 0 || ntracer <= 0) {
        return;
    }

    crp_template = (double *)calloc((size_t)np, sizeof(double));
    cre_template = (double *)calloc((size_t)npe, sizeof(double));
    if (crp_template == 0 || cre_template == 0) {
        free(crp_template);
        free(cre_template);
        return;
    }

    spectrum_temprate(delta_CR_inj, (CRspectrum *)crp_grid, (CRspectrum *)cre_grid,
                      crp_template, cre_template,
                      &temp_num_p, &temp_e_p, &temp_num_e, &temp_e_e);

    for (itr = 0; itr < ntracer; itr++) {
        const double mass = tracer_mass[itr];
        const double dt_fix = 1.0;
        const double steady_duration_gyr = (t_fp_total > 0.0) ? t_fp_total : 1.0;
        double norm_crp = 0.0;
        double norm_cre = 0.0;
        const int use_initial_crp = (seed_cr_species == SEED_CR_SPECIES_PROTON_ONLY ||
                                     seed_cr_species == SEED_CR_SPECIES_ELECTRON_PROTON);
        const int use_initial_primary_cre = (seed_cr_species == SEED_CR_SPECIES_ELECTRON_ONLY ||
                                             seed_cr_species == SEED_CR_SPECIES_ELECTRON_PROTON);
        const int use_steady_primary_e =
            (steady_primary_electron_injection != 0 && use_initial_primary_cre);
        const size_t p_off = (size_t)itr * (size_t)np;
        const size_t e_off = (size_t)itr * (size_t)npe;

        if (InjectionModel == 1) {
            if (use_initial_primary_cre) {
                norm_cre = CRe_Norm_AGN(delta_CR_inj, (CRspectrum *)cre_grid, (CRspectrum *)crp_grid,
                                        mass, z_init, dt_fix);
            }
            if (use_initial_crp) {
                norm_crp = CRp_Norm_AGN(delta_CR_inj, (CRspectrum *)cre_grid, (CRspectrum *)crp_grid,
                                        mass, z_init, dt_fix);
            }
        } else {
            if (use_initial_primary_cre) {
                norm_cre = CRe_Norm(delta_CR_inj, (CRspectrum *)cre_grid, (CRspectrum *)crp_grid, mass);
            }
            if (use_initial_crp) {
                norm_crp = CRp_Norm(delta_CR_inj, (CRspectrum *)cre_grid, (CRspectrum *)crp_grid, mass);
            }
            if (initial_cr_norm_mode == INITIAL_CR_NORM_ENERGY_RATIO &&
                use_initial_primary_cre && use_initial_crp &&
                temp_e_e > 0.0 && temp_e_p > 0.0) {
                if (initial_cr_energy_anchor == INITIAL_CR_ENERGY_ANCHOR_ELECTRON) {
                    norm_crp = initial_cr_energy_ratio_p_to_e * norm_cre *
                               temp_e_e / temp_e_p;
                } else {
                    norm_cre = norm_crp * temp_e_p /
                               (initial_cr_energy_ratio_p_to_e * temp_e_e);
                }
            }
        }

        for (jp = 0; jp < np; jp++) {
            crp_state[p_off + (size_t)jp] = norm_crp * crp_template[jp];
            qpi_batch[p_off + (size_t)jp] = 0.0;
        }

        for (jp = 0; jp < npe; jp++) {
            if (use_steady_primary_e) {
                cre_state[e_off + (size_t)jp] = 0.0;
                qepri_batch[e_off + (size_t)jp] =
                    (norm_cre * cre_template[jp]) / steady_duration_gyr;
            } else {
                cre_state[e_off + (size_t)jp] = norm_cre * cre_template[jp];
                qepri_batch[e_off + (size_t)jp] = 0.0;
            }
        }

        if (debug_enabled && tracer_start + (long int)itr == debug_target_global) {
            fprintf(stderr,
                    "[debug init params] delta_CR_inj=%e norm_cre=%e mass=%e "
                    "template0=%e template1=%e template2=%e template10=%e template50=%e "
                    "CRe0=%e CRe1=%e CRe2=%e CRe10=%e CRe50=%e qpri50=%e\n",
                    delta_CR_inj, norm_cre, mass,
                    cre_template[0], cre_template[1], cre_template[2],
                    cre_template[10], cre_template[50],
                    norm_cre * cre_template[0],
                    norm_cre * cre_template[1],
                    norm_cre * cre_template[2],
                    norm_cre * cre_template[10],
                    norm_cre * cre_template[50],
                    qepri_batch[e_off + 50]);
            fprintf(stderr,
                    "[debug inj] pemin=%e peinjmin=%e template50=%e template80=%e template120=%e templateN=%e\n",
                    pemin,
                    peinjmin,
                    cre_template[50],
                    cre_template[80],
                    cre_template[120],
                    cre_template[npe - 1]);
        }
    }

    if (debug_enabled && debug_target_global >= tracer_start &&
        debug_target_global < tracer_start + (long int)ntracer) {
        const size_t debug_e_off =
            (size_t)(debug_target_global - tracer_start) * (size_t)npe;
        fprintf(stderr,
                "[debug init CRe] %e %e %e %e %e\n",
                cre_state[debug_e_off + 0], cre_state[debug_e_off + 1],
                cre_state[debug_e_off + 2], cre_state[debug_e_off + 10],
                cre_state[debug_e_off + 50]);
    }

    free(crp_template);
    free(cre_template);
}
