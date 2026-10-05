#include <math.h>

#include "tracer_fp_cr_source.h"
#include "CONSTANTS.h"

static double initial_cr_electron_template(double p, double delta)
{
    return pow(p, -delta) * exp(-peinjmin / p);
}

static double initial_cr_proton_template(double p, double delta)
{
    return pow(p, -delta) * exp(-pinjmin / p) * exp(-p / pinjmax);
}

static void fill_initial_cr_templates(double delta,
                                      CRspectrum *cre,
                                      CRspectrum *crp)
{
    for (int i = 0; i < np; i++) {
        cre->N[i] = initial_cr_electron_template(cre->p[i], delta);
        cre->N_pre[i] = 0.0;
        cre->N_ave[i] = 0.0;

        crp->N[i] = initial_cr_proton_template(crp->p[i], delta);
        crp->N_pre[i] = 0.0;
        crp->N_ave[i] = 0.0;
    }
}

static double agn_density_scale(double tracer_mass, double z)
{
    const double psi_0 = 1.0 / (1.0 + pow(1.0 / 2.9, 5.6));
    const double sfrd_md14 =
        pow(1.0 + z, 2.7) / (1.0 + pow((1.0 + z) / 2.9, 5.6)) / psi_0;

    return sfrd_md14 * (tracer_mass / 1.0e15);
}

double CRe_Norm(double delta, CRspectrum *cre, CRspectrum *crp, double tracer_mass)
{
    double num_cre = 0.0;
    const double proton_mass_g = 1.6e-24;

    fill_initial_cr_templates(delta, cre, crp);

    for (int j = 0; j < npe; j++) {
        if (cre->p[j] > peinjmin) {
            num_cre += cre->N[j] * cre->dp[j];
        }
    }

    const double n_th_e = 0.52 * tracer_mass * M_sun / (mu_mol * proton_mass_g);
    return phi_CRe / (num_cre / n_th_e);
}

double CRp_Norm(double delta, CRspectrum *cre, CRspectrum *crp, double tracer_mass)
{
    double num_crp = 0.0;
    const double proton_mass_g = 1.6e-24;

    fill_initial_cr_templates(delta, cre, crp);

    for (int j = 0; j < np; j++) {
        if (crp->p[j] > pinjmin) {
            num_crp += crp->N[j] * crp->dp[j];
        }
    }

    const double n_th_p = 0.52 * tracer_mass * M_sun / (mu_mol * proton_mass_g);
    return phi_CRp / (num_crp / n_th_p);
}

double CRe_Norm_AGN(double delta,
                    CRspectrum *cre,
                    CRspectrum *crp,
                    double tracer_mass,
                    double z,
                    double dt_gyr)
{
    (void)crp;

    double eps_cre = 0.0;

    for (int i = 0; i < np; i++) {
        cre->N[i] = initial_cr_electron_template(cre->p[i], delta);
        cre->N_pre[i] = 0.0;
        cre->N_ave[i] = 0.0;
    }

    for (int j = 0; j < npe; j++) {
        if (cre->p[j] > peinjmin) {
            eps_cre += cre->E[j] * cre->N[j] * cre->dp[j] * GeV;
        }
    }

    return L_CR_e * agn_density_scale(tracer_mass, z) * dt_gyr * Gyr / eps_cre;
}

double CRp_Norm_AGN(double delta,
                    CRspectrum *cre,
                    CRspectrum *crp,
                    double tracer_mass,
                    double z,
                    double dt_gyr)
{
    (void)cre;

    double eps_crp = 0.0;

    for (int i = 0; i < np; i++) {
        crp->N[i] = initial_cr_proton_template(crp->p[i], delta);
        crp->N_pre[i] = 0.0;
        crp->N_ave[i] = 0.0;
    }

    for (int j = 0; j < np; j++) {
        if (crp->p[j] > pinjmin) {
            eps_crp += crp->E[j] * crp->N[j] * crp->dp[j] * GeV;
        }
    }

    return L_CR_p * agn_density_scale(tracer_mass, z) * dt_gyr * Gyr / eps_crp;
}

double CRe_Norm_Inj_AGN(double temp_e_cre,
                        double tracer_mass,
                        double z,
                        double dt_gyr)
{
    const double l_agn_e = L_CR_e * agn_density_scale(tracer_mass, z);
    const double template_l_agn_e = temp_e_cre / (dt_gyr * Gyr);

    return l_agn_e / template_l_agn_e;
}

void spectrum_temprate(double delta,
                       CRspectrum *crp,
                       CRspectrum *cre,
                       double *np_template,
                       double *ne_template,
                       double *num_p_template,
                       double *e_p_template,
                       double *num_e_template,
                       double *e_e_template)
{
    double eps_crp = 0.0;
    double num_crp = 0.0;
    double eps_cre = 0.0;
    double num_cre = 0.0;

    for (int i = 0; i < np; i++) {
        ne_template[i] = initial_cr_electron_template(cre->p[i], delta);
        np_template[i] = initial_cr_proton_template(crp->p[i], delta);
    }

    for (int j = 0; j < np; j++) {
        if (crp->p[j] > pinjmin) {
            num_crp += np_template[j] * crp->dp[j];
            eps_crp += crp->E[j] * np_template[j] * crp->dp[j] * GeV;
        }
    }

    for (int j = 0; j < npe; j++) {
        if (cre->p[j] > peinjmin) {
            num_cre += ne_template[j] * cre->dp[j];
            eps_cre += cre->E[j] * ne_template[j] * cre->dp[j] * GeV;
        }
    }

    *num_p_template = num_crp;
    *e_p_template = eps_crp;
    *num_e_template = num_cre;
    *e_e_template = eps_cre;
}
