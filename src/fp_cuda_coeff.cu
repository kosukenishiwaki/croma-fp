#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "fp_cuda_coeff.h"

namespace {

constexpr int kMaxRow = (np > npe) ? np : npe;
constexpr double kPi = 3.14159265358979323846;
constexpr double kC = 2.999e10;
constexpr double kGeV = 1.6e-3;
constexpr double kGyr = 3.1536e16;
constexpr double kMp = 0.938;
constexpr double kMe = 0.511e-3;
constexpr double kMillibarnCm2 = 1.0e-27;
constexpr double kMpc = 3.0857e24;
constexpr double kKpc = 3.0857e21;
constexpr double kMsun = 1.99e33;

static void cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(2);
    }
}

__device__ double device_b_Coulomb_kernel(double x)
{
    return 0.5 * std::sqrt(kPi) * erf(std::sqrt(x)) - std::sqrt(x) * std::exp(-x);
}

__device__ bool device_isfinite(double x)
{
    return x == x && std::fabs(x) < 1.0e300;
}

__device__ double device_b_Coulomb_p(double nicm, double p, double kBT)
{
    const double p2 = p * p;
    const double gamma = std::sqrt(1.0 + p2);
    const double betap2 = p2 / (1.0 + p2);
    const double kinetic = gamma - 1.0;
    const double Ep = gamma * kMp - kMp;
    const double xp = Ep / kBT;
    const double xe = xp * (kMe / kMp);
    const double sum = device_b_Coulomb_kernel(xe) + device_b_Coulomb_kernel(xp) * (kMe / kMp);
    const double factor = 1.0 - kBT / (2.0 * kMp * kinetic);

    if (factor < 0.0) return 0.0;

    return kGyr * kC / (kMp * kGeV) * sum * nicm * (1.0 / betap2) * factor * 3.5e-29;
}

__device__ double device_b_synch(double p, double b_field)
{
    const double b_mu = b_field * 1.0e6;
    const double meg = 9.11e-28;
    const double c_local = 3.0e10;
    return 4.8e-4 * p * p * (b_mu / 3.2) * (b_mu / 3.2) * meg * c_local * kGyr;
}

__device__ double device_b_Coulomb_e(double nicm, double pe)
{
    const double pe2 = pe * pe;
    const double beta2 = pe2 / (1.0 + pe2);
    const double A = 3.05e-29;
    return A * nicm * (1.0 + std::log(pe / nicm) / 74.8) / (kMe * kGeV) * kC * kGyr / beta2;
}

__device__ double device_adiabatic_divv(double divv, double p)
{
    return (1.0 / 3.0) * divv * p;
}

__device__ double device_effective_mach(double dv_imc, double cs, double mach_limit_value)
{
    if (cs <= 0.0) return 0.0;

    double mach = dv_imc / cs;
    if (mach < 0.0) mach = 0.0;
    if (mach_limit_value > 0.0 && mach > mach_limit_value) mach = mach_limit_value;
    return mach;
}

__device__ double device_accelerationtime_asa(double l_turb_mpc,
                                              double cs,
                                              double mach,
                                              double beta_pl,
                                              double psi_value)
{
    return std::sqrt(6.0 / 5.0) / 12.0 *
           (kC / (cs * cs)) *
           (l_turb_mpc * kMpc) /
           std::sqrt(beta_pl) /
           std::pow(mach, 3.0) *
           std::pow(psi_value, 3.0) / kGyr;
}

__device__ double device_i_theta(double x)
{
    return std::pow(x, 4.0) / 4.0 + x * x -
           (1.0 + 2.0 * x * x) * std::log(x) - 5.0 / 4.0;
}

__device__ double device_accelerationtime_ttd(double l_turb_mpc,
                                              double cs,
                                              double mach,
                                              double dv_imc,
                                              int ttd_model)
{
    const double L = l_turb_mpc * kMpc;
    const double x = cs / kC;

    if (ttd_model == TTD_TACC_MODEL_BRUNETTI16) {
        const double theta = device_i_theta(x);
        if (!(x > 0.0) || !(theta > 0.0) || !(mach > 0.0) ||
            !(L > 0.0) || !(cs > 0.0)) {
            return 0.0;
        }
        return 2.5e-3 / (x * theta) *
               std::pow(mach / 0.5, -4.0) *
               (L / (300.0 * kKpc)) *
               std::pow(cs / 1.5e8, -1.0);
    }

    const double kL = 2.0 * kPi / L;
    const double k_cut = 1.04e4 * std::pow(mach, 4.0) * kL;
    const double kWk_L = 0.5 * dv_imc * dv_imc;
    const double kWk_cut = kWk_L * std::pow(k_cut / kL, -0.5);
    return kC / kPi / device_i_theta(x) / k_cut / kWk_cut / kGyr;
}

__device__ double device_eta_dpp_scale(int icell,
                                       double eta_cap,
                                       double t_acc_gyr,
                                       const double *tracer_mass_msun,
                                       const double *crp_state,
                                       const double *cre_state,
                                       const double *l_turb_mpc,
                                       const double *dv_imc,
                                       const double *crp_e,
                                       const double *crp_dp,
                                       const double *cre_p,
                                       const double *cre_dp)
{
    if (!(eta_cap > 0.0) || !device_isfinite(eta_cap)) return 1.0;
    if (tracer_mass_msun == nullptr || crp_state == nullptr || cre_state == nullptr) return 1.0;
    if (!(t_acc_gyr > 0.0) || !device_isfinite(t_acc_gyr)) return 1.0;
    if (!(l_turb_mpc[icell] > 0.0) || !(dv_imc[icell] > 0.0) ||
        !(tracer_mass_msun[icell] > 0.0)) {
        return 1.0;
    }

    double ecr_erg = 0.0;
    for (int jp = 2; jp < np - 2; jp++) {
        const double e_kin_gev = crp_e[jp] - kMp;
        const double n_bin = crp_state[(size_t)icell * (size_t)np + (size_t)jp];
        if (e_kin_gev > 0.0 && device_isfinite(n_bin)) {
            ecr_erg += e_kin_gev * kGeV * n_bin * crp_dp[jp];
        }
    }
    for (int je = 2; je < npe - 2; je++) {
        const double e_kin_gev = kMe * (std::sqrt(1.0 + cre_p[je] * cre_p[je]) - 1.0);
        const double n_bin = cre_state[(size_t)icell * (size_t)npe + (size_t)je];
        if (e_kin_gev > 0.0 && device_isfinite(n_bin)) {
            ecr_erg += e_kin_gev * kGeV * n_bin * cre_dp[je];
        }
    }

    const double q_turb_total =
        0.5 * tracer_mass_msun[icell] * kMsun *
        dv_imc[icell] * dv_imc[icell] * dv_imc[icell] /
        (l_turb_mpc[icell] * kMpc);
    const double denom = q_turb_total * t_acc_gyr * kGyr;
    if (!(denom > 0.0) || !(ecr_erg > 0.0)) return 1.0;

    const double eta = ecr_erg / denom;
    if (!(eta > eta_cap) || !device_isfinite(eta)) return 1.0;
    return eta_cap / eta;
}

__device__ double device_sigma(int i, double p)
{
    const double Ep = std::sqrt(kMp * kMp + p * p);
    const double Tp = Ep - kMp;
    const double x = std::log10(p);
    const double x2 = x * x;
    const double x3 = x2 * x;
    double nondiff = 0.0, diff = 0.0, Delta = 0.0, res = 0.0;
    double a[8], b[2], c[3], d[7], e[2], f[5], g[5];

    a[0] = 0.1176; a[1] = 0.3829; a[2] = 23.10; a[3] = 6.454;
    a[4] = -5.764; a[5] = -23.63; a[6] = 94.75; a[7] = 0.02667;
    b[0] = 11.34; b[1] = 23.72;
    c[0] = 28.5; c[1] = -6.133; c[2] = 1.464;
    d[0] = 0.3522; d[1] = 0.1530; d[2] = 1.498; d[3] = 2.0;
    d[4] = 30.0; d[5] = 3.155; d[6] = 1.042;
    e[0] = 5.922; e[1] = 1.632;
    f[0] = 0.0834; f[1] = 9.5; f[2] = 5.5; f[3] = 1.68; f[4] = 3134.0;
    g[0] = 0.0004257; g[1] = 4.5; g[2] = 7.0; g[3] = 2.1; g[4] = 503.5;

    if (p >= 1.0 && p < 1.3) {
        nondiff = 0.57 * std::pow(x / a[0], 1.2) *
                  (a[2] + a[3] * x2 + a[4] * x3 + a[5] * std::exp(-a[6] * std::pow(x + a[7], 2.0)));
    } else if (p >= 1.3 && p < 2.4) {
        nondiff = (b[0] * std::fabs(a[1] - x) + b[1] * std::fabs(a[0] - x)) / (a[1] - a[0]);
    } else if (p >= 2.4 && p < 10.0) {
        nondiff = a[2] + a[3] * x2 + a[4] * x3 + a[5] * std::exp(-a[6] * std::pow(x + a[7], 2.0));
    } else if (p >= 10.0) {
        nondiff = c[0] + c[1] * x + c[2] * x2;
    }

    if (p >= 2.25 && p < 3.2) {
        diff = std::sqrt(std::fabs(x - d[0]) / d[1]) *
               (d[2] + d[3] * std::log10(d[4] * (x - 0.25)) + d[5] * x2 - d[6] * x3);
    } else if (p >= 3.2 && p < 100.0) {
        diff = d[2] + d[3] * std::log10(d[4] * (x - 0.25)) + d[5] * x2 - d[6] * x3;
    } else if (p >= 100.0) {
        diff = e[0] + e[1] * x;
    }

    if (Ep >= 1.4 && Ep < 1.6) {
        Delta = f[0] * std::pow(Ep, 10.0);
    } else if (Ep >= 1.6 && Ep < 1.8) {
        Delta = f[1] * std::exp(-f[2] * std::pow(Ep - f[3], 2.0));
    } else if (Ep >= 1.8 && Ep < 10.0) {
        Delta = f[4] * std::pow(Ep, -10.0);
    }

    if (Ep >= 1.6 && Ep < 1.9) {
        res = g[0] * std::pow(Ep, 14.0);
    } else if (Ep >= 1.9 && Ep < 2.3) {
        res = g[1] * std::exp(-g[2] * std::pow(Ep - g[3], 2.0));
    } else if (Ep >= 2.3 && Ep < 20.0) {
        res = g[4] * std::pow(Ep, -6.0);
    }

    const double tot = (nondiff + diff + Delta + res) * kMillibarnCm2;
    const double pi0 = 1.0 / (0.007 + 0.1 * std::log(Tp) / Tp + 0.3 / (Tp * Tp));
    const double piplus = 1.0 / (0.00717 + 0.0652 * std::log(Tp) / Tp + 0.162 / (Tp * Tp));
    const double piminus = 1.0 / (0.00456 + 0.0846 / std::pow(Tp, 0.5) + 0.577 / std::pow(Tp, 1.5));

    if (i == 0) return tot;
    if (i == 1) return tot * piplus / pi0;
    if (i == -1) return tot * piminus / pi0;
    if (i == 2) return tot * (1.0 + piplus / pi0 + piminus / pi0);
    return 0.0;
}

__device__ void device_cc_eval_face_weights(double w,
                                            double *delta,
                                            double *wp,
                                            double *wm)
{
    *delta = 1.0 / w - 1.0 / (std::exp(w) - 1.0);
    if (std::fabs(w) < 0.1) {
        const double denom = 1.0 + w * w / 24.0 + std::pow(w, 4.0) / 1920.0;
        *wp = std::exp(0.5 * w) / denom;
        *wm = std::exp(-0.5 * w) / denom;
    } else {
        const double aw = std::fabs(w);
        const double denom = 1.0 - std::exp(-aw);
        *wp = aw * std::exp(-0.5 * aw + 0.5 * w) / denom;
        *wm = aw * std::exp(-0.5 * aw - 0.5 * w) / denom;
    }
}

__device__ void device_cc_prepare_face_weight(double b_left,
                                              double b_right,
                                              double c_left,
                                              double c_right,
                                              double x_left,
                                              double x_right,
                                              double *delta,
                                              double *wp,
                                              double *wm,
                                              double *w_out)
{
    *delta = 0.0;
    *wp = 0.0;
    *wm = 0.0;
    if (w_out != nullptr) *w_out = 0.0;

    if (std::fabs(c_left) <= 1.0e-200) return;

    const double w = (x_right - x_left) * (b_left + b_right) / (c_left + c_right);
    device_cc_eval_face_weights(w, delta, wp, wm);
    if (w_out != nullptr) *w_out = w;
}

__device__ void device_cc_build_coeffs(int nm,
                                       double dt,
                                       const double *x,
                                       const double *dx,
                                       const double *A,
                                       const double *B,
                                       const double *C,
                                       const double *T,
                                       double xm1,
                                       double xp1,
                                       double bm1,
                                       double bp1,
                                       double cm1,
                                       double cp1,
                                       double *out_a,
                                       double *out_b,
                                       double *out_c)
{
    double delta[kMaxRow], w[kMaxRow];
    double Wp[kMaxRow], Wm[kMaxRow];
    double Wpm1 = 0.0, Wmm1 = 0.0;

    for (int m = 0; m < nm; m++) {
        device_cc_prepare_face_weight(B[m], B[m + 1], C[m], C[m + 1],
                                      x[m], x[m + 1], &delta[m], &Wp[m], &Wm[m], &w[m]);
    }
    if (std::fabs(C[0]) > 1.0e-200) {
        device_cc_prepare_face_weight(bm1, B[0], cm1, C[0], xm1, x[0],
                                      &delta[0], &Wpm1, &Wmm1, nullptr);
    }
    device_cc_prepare_face_weight(B[nm], bp1, C[nm], cp1, x[nm], xp1,
                                  &delta[nm], &Wp[nm], &Wm[nm], &w[nm]);

    for (int m = 1; m < nm; m++) {
        if (std::fabs(C[m]) <= 1.0e-200) {
            const double bm_12 = 0.5 * (B[m] + B[m - 1]);
            const double bm12 = 0.5 * (B[m] + B[m + 1]);
            const double deltam_12 = delta[m - 1];
            const double deltam12 = delta[m];

            out_a[m] = -deltam_12 * bm_12 * dt / (A[m] * dx[m]);
            out_b[m] = 1.0 + dt / (A[m] * dx[m]) *
                             ((1.0 - deltam_12) * bm_12 - deltam12 * bm12) +
                       dt / T[m];
            out_c[m] = dt / (A[m] * dx[m]) * (1.0 - deltam12) * bm12;
        } else {
            out_a[m] = dt / (A[m] * dx[m]) *
                       (0.5 * (C[m] + C[m - 1])) / (x[m] - x[m - 1]) * Wm[m - 1];
            out_b[m] = 1.0 + dt / (A[m] * dx[m]) *
                             ((0.5 * (C[m] + C[m + 1])) / (x[m + 1] - x[m]) * Wm[m] +
                              (0.5 * (C[m] + C[m - 1])) / (x[m] - x[m - 1]) * Wp[m - 1]) +
                       dt / T[m];
            out_c[m] = dt / (A[m] * dx[m]) *
                       (0.5 * (C[m] + C[m + 1])) / (x[m + 1] - x[m]) * Wp[m];
        }
    }

    if (std::fabs(C[0]) > 1.0e-200) {
        const double c_12 = 0.0;
        out_a[0] = dt / (A[0] * dx[0]) * c_12 / (x[0] - xm1) * Wmm1;
        out_b[0] = 1.0 + dt / (A[0] * dx[0]) *
                         (c_12 / (x[0] - xm1) * Wpm1 +
                          0.5 * (C[0] + C[1]) / (x[1] - x[0]) * Wm[0]) +
                   dt / T[0];
        out_c[0] = dt / (A[0] * dx[0]) * 0.5 * (C[0] + C[1]) / (x[1] - x[0]) * Wp[0];
    } else {
        out_a[0] = 0.0;
        out_b[0] = 1.0 + dt / T[0];
        out_c[0] = dt / (A[0] * dx[0]) * (1.0 - delta[0]) * 0.5 * (B[0] + B[1]);
    }

    if (std::fabs(C[nm]) > 1.0e-200) {
        out_a[nm] = dt / (A[nm] * dx[nm]) *
                    (0.5 * (C[nm] + C[nm - 1])) / (x[nm] - x[nm - 1]) * Wm[nm - 1];
        out_b[nm] = 1.0 + dt / (A[nm] * dx[nm]) *
                          ((0.5 * (C[nm] + C[nm - 1])) / (x[nm] - x[nm - 1]) * Wp[nm - 1]) +
                    dt / T[nm];
    } else {
        out_a[nm] = -delta[nm - 1] * 0.5 * (B[nm] + B[nm - 1]) * dt / (A[nm] * dx[nm]);
        out_b[nm] = 1.0 + dt / (A[nm] * dx[nm]) * (0.5 * (B[nm - 1] + B[nm])) + dt / T[nm];
    }
    out_c[nm] = 0.0;
}

__global__ void prepare_losses_kernel(int ncell,
                                      const double *n_gas,
                                      const double *kbt,
                                      const double *b_field,
                                      const double *divv_gyr,
                                      const double *rad_ic_batch,
                                      const double *rad_ic_m1,
                                      const double *rad_ic_p1,
                                      const double *crp_p,
                                      const double *crp_e,
                                      const double *crp_sigma_pp,
                                      const double *crp_sigmoid_pp,
                                      int crp_j_pp,
                                      const double *cre_p,
                                      const double *cre_p2,
                                      double crp_pm1,
                                      double crp_pp1,
                                      double cre_pm1,
                                      double cre_pp1,
                                      double cre_pm1_p2,
                                      double cre_pp1_p2,
                                      double *crp_radp_batch,
                                      double *crp_tloss_batch,
                                      double *cre_radp_batch,
                                      double *cre_tloss_batch,
                                      double *crp_radpm1,
                                      double *crp_radpp1,
                                      double *cre_radpm1,
                                      double *cre_radpp1)
{
    const int icell = blockIdx.x * blockDim.x + threadIdx.x;
    const double kappa_pp = 0.5;
    const int j_pp = crp_j_pp;

    if (icell >= ncell) return;
    (void)crp_e;

    const double tloss_prefactor = 1.0 / (kappa_pp * kC * n_gas[icell]) / kGyr;
    const double adiabatic_prefactor = (1.0 / 3.0) * divv_gyr[icell];
    const double b_mu = b_field[icell] * 1.0e6;
    const double b_ratio = b_mu / 3.2;
    const double synch_coeff = 4.8e-4 * b_ratio * b_ratio * 9.11e-28 * 3.0e10 * kGyr;
    const double coulomb_e_coeff = 3.05e-29 * n_gas[icell] * kC * kGyr / (kMe * kGeV);

    for (int j = 0; j < np; j++) {
        const size_t off = (size_t)j * (size_t)ncell + (size_t)icell;
        double sig_pp = (j > j_pp) ? crp_sigma_pp[j] : crp_sigma_pp[j_pp];
        sig_pp *= crp_sigmoid_pp[j];

        crp_radp_batch[off] = device_b_Coulomb_p(n_gas[icell], crp_p[j], kbt[icell]) +
                              adiabatic_prefactor * crp_p[j];
        crp_tloss_batch[off] = tloss_prefactor / sig_pp;
    }

    for (int j = 0; j < npe; j++) {
        const size_t off = (size_t)j * (size_t)ncell + (size_t)icell;
        const double beta2 = cre_p2[j] / (1.0 + cre_p2[j]);
        const double coulomb =
            coulomb_e_coeff * (1.0 + std::log(cre_p[j] / n_gas[icell]) / 74.8) / beta2;
        cre_radp_batch[off] = synch_coeff * cre_p2[j] + rad_ic_batch[off] +
                              coulomb + adiabatic_prefactor * cre_p[j];
        cre_tloss_batch[off] = 1.0e100;
    }

    crp_radpm1[icell] = device_b_Coulomb_p(n_gas[icell], crp_pm1, kbt[icell]);
    crp_radpp1[icell] = device_b_Coulomb_p(n_gas[icell], crp_pp1, kbt[icell]);
    {
        const double beta2_m1 = cre_pm1_p2 / (1.0 + cre_pm1_p2);
        const double beta2_p1 = cre_pp1_p2 / (1.0 + cre_pp1_p2);
        const double coulomb_m1 =
            coulomb_e_coeff * (1.0 + std::log(cre_pm1 / n_gas[icell]) / 74.8) / beta2_m1;
        const double coulomb_p1 =
            coulomb_e_coeff * (1.0 + std::log(cre_pp1 / n_gas[icell]) / 74.8) / beta2_p1;
        cre_radpm1[icell] = synch_coeff * cre_pm1_p2 + rad_ic_m1[icell] + coulomb_m1;
        cre_radpp1[icell] = synch_coeff * cre_pp1_p2 + rad_ic_p1[icell] + coulomb_p1;
    }
}

__global__ void prepare_momentumdiff_model_kernel(int ncell,
                                                  int model,
                                                  double epmax,
                                                  double psi_value,
                                                  double mach_limit_value,
                                                  int ttd_tacc_model,
                                                  double t_acc_direct_gyr,
                                                  double eta_dpp_cap_value,
                                                  const double *l_turb_mpc,
                                                  const double *dv_imc,
                                                  const double *cs,
                                                  const double *beta_pl,
                                                  const double *crp_p,
                                                  const double *crp_e,
                                                  const double *crp_dp,
                                                  const double *crp_p2,
                                                  const double *crp_exp_cut,
                                                  double crp_pm1,
                                                  double crp_pp1,
                                                  double crp_pm1_p2,
                                                  double crp_pp1_p2,
                                                  double crp_pm1_exp_cut,
                                                  double crp_pp1_exp_cut,
                                                  const double *cre_p,
                                                  const double *cre_dp,
                                                  const double *cre_p2,
                                                  const double *cre_exp_cut,
                                                  const double *tracer_mass_msun,
                                                  const double *crp_state,
                                                  const double *cre_state,
                                                  double cre_pm1,
                                                  double cre_pp1,
                                                  double cre_pm1_p2,
                                                  double cre_pp1_p2,
                                                  double cre_pm1_exp_cut,
                                                  double cre_pp1_exp_cut,
                                                  double *dpp_batch,
                                                  double *dppm1,
                                                  double *dppp1,
                                                  double *dppe_batch,
                                                  double *dppem1,
                                                  double *dppep1)
{
    const int icell = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr double q = 2.0;
    double t_acc = 1.0e5;

    if (icell >= ncell) return;

    if (model == FP_MOMENTUMDIFF_MODEL_ASA ||
        model == FP_MOMENTUMDIFF_MODEL_TTD) {
        const double mach = device_effective_mach(dv_imc[icell], cs[icell], mach_limit_value);
        if (mach <= 0.0) {
            model = FP_MOMENTUMDIFF_MODEL_OFF;
        } else if (model == FP_MOMENTUMDIFF_MODEL_ASA) {
            if (beta_pl[icell] <= 0.0) {
                model = FP_MOMENTUMDIFF_MODEL_OFF;
            } else {
                t_acc = device_accelerationtime_asa(l_turb_mpc[icell], cs[icell],
                                                    mach, beta_pl[icell], psi_value);
            }
        } else {
            if (dv_imc[icell] <= 0.0) {
                model = FP_MOMENTUMDIFF_MODEL_OFF;
            } else {
                t_acc = device_accelerationtime_ttd(l_turb_mpc[icell], cs[icell],
                                                    mach, dv_imc[icell],
                                                    ttd_tacc_model);
            }
        }
    } else if (model == FP_MOMENTUMDIFF_MODEL_DIRECT_TACC) {
        if (t_acc_direct_gyr > 0.0) {
            t_acc = t_acc_direct_gyr;
        } else {
            model = FP_MOMENTUMDIFF_MODEL_OFF;
        }
    } else {
        model = FP_MOMENTUMDIFF_MODEL_OFF;
    }

    if (model == FP_MOMENTUMDIFF_MODEL_OFF) {
        for (int jp = 0; jp < np; jp++) {
            dpp_batch[(size_t)jp * (size_t)ncell + (size_t)icell] =
                crp_p2[jp] / (4.0 * 1.0e5);
        }
        dppp1[icell] = crp_pp1_p2 / (4.0 * 1.0e5);
        dppm1[icell] = crp_pm1_p2 / (4.0 * 1.0e5);
        for (int je = 0; je < npe; je++) {
            dppe_batch[(size_t)je * (size_t)ncell + (size_t)icell] =
                cre_p2[je] / (4.0 * 1.0e5);
        }
        dppep1[icell] = cre_pp1_p2 / (4.0 * 1.0e5);
        dppem1[icell] = cre_pm1_p2 / (4.0 * 1.0e5);
        return;
    }

    const double eta_scale =
        device_eta_dpp_scale(icell, eta_dpp_cap_value, t_acc,
                             tracer_mass_msun, crp_state, cre_state,
                             l_turb_mpc, dv_imc,
                             crp_e, crp_dp, cre_p, cre_dp);

    for (int jp = 0; jp < np; jp++) {
        dpp_batch[(size_t)jp * (size_t)ncell + (size_t)icell] =
            eta_scale *
            crp_p2[jp] / ((q + 2.0) * t_acc) *
            std::exp(-crp_e[jp] / epmax) *
            crp_exp_cut[jp];
    }
    dppp1[icell] = eta_scale * crp_pp1_p2 / ((q + 2.0) * t_acc) *
                   std::exp(-kMp * std::sqrt(1.0 + crp_pp1 * crp_pp1) / epmax) *
                   crp_pp1_exp_cut;
    dppm1[icell] = eta_scale * crp_pm1_p2 / ((q + 2.0) * t_acc) *
                   std::exp(-kMp * std::sqrt(1.0 + crp_pm1 * crp_pm1) / epmax) *
                   crp_pm1_exp_cut;

    for (int je = 0; je < npe; je++) {
        dppe_batch[(size_t)je * (size_t)ncell + (size_t)icell] =
            eta_scale *
            cre_p2[je] / ((q + 2.0) * t_acc) *
            cre_exp_cut[je];
    }
    dppep1[icell] = eta_scale * cre_pp1_p2 / ((q + 2.0) * t_acc) *
                    cre_pp1_exp_cut;
    dppem1[icell] = eta_scale * cre_pm1_p2 / ((q + 2.0) * t_acc) *
                    cre_pm1_exp_cut;
}

__global__ void prepare_momentumdiff_off_kernel(int ncell,
                                                const double *crp_p,
                                                const double *crp_p2,
                                                double crp_pm1,
                                                double crp_pp1,
                                                double crp_pm1_p2,
                                                double crp_pp1_p2,
                                                const double *cre_p,
                                                const double *cre_p2,
                                                double cre_pm1,
                                                double cre_pp1,
                                                double cre_pm1_p2,
                                                double cre_pp1_p2,
                                                double *dpp_batch,
                                                double *dppm1,
                                                double *dppp1,
                                                double *dppe_batch,
                                                double *dppem1,
                                                double *dppep1)
{
    const int icell = blockIdx.x * blockDim.x + threadIdx.x;

    if (icell >= ncell) return;
    (void)crp_p;
    (void)crp_pm1;
    (void)crp_pp1;
    (void)cre_p;
    (void)cre_pm1;
    (void)cre_pp1;

    for (int jp = 0; jp < np; jp++) {
        dpp_batch[(size_t)jp * (size_t)ncell + (size_t)icell] =
            crp_p2[jp] / (4.0 * 1.0e5);
    }
    dppp1[icell] = crp_pp1_p2 / (4.0 * 1.0e5);
    dppm1[icell] = crp_pm1_p2 / (4.0 * 1.0e5);

    for (int je = 0; je < npe; je++) {
        dppe_batch[(size_t)je * (size_t)ncell + (size_t)icell] =
            cre_p2[je] / (4.0 * 1.0e5);
    }
    dppep1[icell] = cre_pp1_p2 / (4.0 * 1.0e5);
    dppem1[icell] = cre_pm1_p2 / (4.0 * 1.0e5);
}

__global__ void build_cc_coeffs_kernel(int ncell,
                                       double dt,
                                       const double *crp_p,
                                       const double *crp_dp,
                                       double crp_pm1,
                                       double crp_pp1,
                                       const double *cre_p,
                                       const double *cre_dp,
                                       double cre_pm1,
                                       double cre_pp1,
                                       const double *crp_radp_batch,
                                       const double *crp_tloss_batch,
                                       const double *dpp_batch,
                                       const double *cre_radp_batch,
                                       const double *cre_tloss_batch,
                                       const double *dppe_batch,
                                       const double *crp_radpm1,
                                       const double *crp_radpp1,
                                       const double *cre_radpm1,
                                       const double *cre_radpp1,
                                       const double *dppm1,
                                       const double *dppp1,
                                       const double *dppem1,
                                       const double *dppep1,
                                       double *ccp_a_batch,
                                       double *ccp_b_batch,
                                       double *ccp_c_batch,
                                       double *cce_a_batch,
                                       double *cce_b_batch,
                                       double *cce_c_batch)
{
    const int icell = blockIdx.x * blockDim.x + threadIdx.x;
    double cc_ap[np], cc_bp[np], cc_cp[np], cc_tp[np];
    double cc_ae[npe], cc_be[npe], cc_ce[npe], cc_te[npe];
    double out_ap[np], out_bp[np], out_cp[np];
    double out_ae[npe], out_be[npe], out_ce[npe];

    if (icell >= ncell) return;

    for (int jp = 0; jp < np; jp++) {
        const size_t off = (size_t)jp * (size_t)ncell + (size_t)icell;
        cc_ap[jp] = 1.0;
        cc_bp[jp] = crp_radp_batch[off] - 2.0 / crp_p[jp] * dpp_batch[off];
        cc_cp[jp] = dpp_batch[off];
        cc_tp[jp] = crp_tloss_batch[off];
    }
    for (int je = 0; je < npe; je++) {
        const size_t off = (size_t)je * (size_t)ncell + (size_t)icell;
        cc_ae[je] = 1.0;
        cc_be[je] = cre_radp_batch[off] - 2.0 / cre_p[je] * dppe_batch[off];
        cc_ce[je] = dppe_batch[off];
        cc_te[je] = cre_tloss_batch[off];
    }

    device_cc_build_coeffs(np - 1, dt, crp_p, crp_dp,
                           cc_ap, cc_bp, cc_cp, cc_tp,
                           crp_pm1, crp_pp1,
                           crp_radpm1[icell] - 2.0 / crp_pm1 * dppm1[icell],
                           crp_radpp1[icell] - 2.0 / crp_pp1 * dppp1[icell],
                           dppm1[icell], dppp1[icell],
                           out_ap, out_bp, out_cp);

    device_cc_build_coeffs(npe - 1, dt, cre_p, cre_dp,
                           cc_ae, cc_be, cc_ce, cc_te,
                           cre_pm1, cre_pp1,
                           cre_radpm1[icell] - 2.0 / cre_pm1 * dppem1[icell],
                           cre_radpp1[icell] - 2.0 / cre_pp1 * dppep1[icell],
                           dppem1[icell], dppep1[icell],
                           out_ae, out_be, out_ce);

    for (int jp = 0; jp < np; jp++) {
        const size_t off = (size_t)icell * (size_t)np + (size_t)jp;
        ccp_a_batch[off] = out_ap[jp];
        ccp_b_batch[off] = out_bp[jp];
        ccp_c_batch[off] = out_cp[jp];
    }
    for (int je = 0; je < npe; je++) {
        const size_t off = (size_t)icell * (size_t)npe + (size_t)je;
        cce_a_batch[off] = out_ae[je];
        cce_b_batch[off] = out_be[je];
        cce_c_batch[off] = out_ce[je];
    }
}

}  // namespace

int fp_cuda_prepare_coeff_terms(const FpCudaCoeffPrepareInput *in,
                                const char *losses_label,
                                const char *model_label,
                                const char *off_label)
{
    if (in == nullptr) return -1;
    if (in->d_crp_p == nullptr || in->d_crp_e == nullptr ||
        in->d_crp_dp == nullptr || in->d_crp_p2 == nullptr ||
        in->d_crp_exp_cut == nullptr || in->d_crp_sigma_pp == nullptr ||
        in->d_crp_sigmoid_pp == nullptr || in->d_cre_p == nullptr ||
        in->d_cre_dp == nullptr || in->d_cre_p2 == nullptr ||
        in->d_cre_exp_cut == nullptr) {
        return -1;
    }

    prepare_losses_kernel<<<in->cell_blocks, in->cell_threads>>>(
        in->ncell,
        in->d_n_gas, in->d_kbt, in->d_b_field, in->d_divv,
        in->d_rad_ic_batch, in->d_rad_ic_m1, in->d_rad_ic_p1,
        in->d_crp_p, in->d_crp_e,
        in->d_crp_sigma_pp, in->d_crp_sigmoid_pp, in->crp_j_pp,
        in->d_cre_p, in->d_cre_p2,
        in->crp_pm1, in->crp_pp1, in->cre_pm1, in->cre_pp1,
        in->cre_pm1_p2, in->cre_pp1_p2,
        in->d_crp_radp, in->d_crp_tloss,
        in->d_cre_radp, in->d_cre_tloss,
        in->d_crp_radpm1, in->d_crp_radpp1,
        in->d_cre_radpm1, in->d_cre_radpp1);
    cuda_check(cudaGetLastError(),
               (losses_label != nullptr) ? losses_label : "launch prepare_losses_kernel");

    prepare_momentumdiff_model_kernel<<<in->cell_blocks, in->cell_threads>>>(
        in->ncell, in->active_model,
        in->epmax, in->psi_value, in->mach_limit_value, in->ttd_tacc_model,
        in->t_acc_direct_gyr,
        in->eta_dpp_cap_value,
        in->d_l_turb_mpc, in->d_dv_imc, in->d_cs, in->d_beta_pl,
        in->d_crp_p, in->d_crp_e, in->d_crp_dp,
        in->d_crp_p2, in->d_crp_exp_cut,
        in->crp_pm1, in->crp_pp1,
        in->crp_pm1_p2, in->crp_pp1_p2,
        in->crp_pm1_exp_cut, in->crp_pp1_exp_cut,
        in->d_cre_p, in->d_cre_dp,
        in->d_cre_p2, in->d_cre_exp_cut,
        in->d_tracer_mass_msun, in->d_crp_state, in->d_cre_state,
        in->cre_pm1, in->cre_pp1,
        in->cre_pm1_p2, in->cre_pp1_p2,
        in->cre_pm1_exp_cut, in->cre_pp1_exp_cut,
        in->d_dpp, in->d_dppm1, in->d_dppp1,
        in->d_dppe, in->d_dppem1, in->d_dppep1);
    cuda_check(cudaGetLastError(),
               (model_label != nullptr) ? model_label : "launch prepare_momentumdiff_model_kernel");

    prepare_momentumdiff_off_kernel<<<in->cell_blocks, in->cell_threads>>>(
        in->ncell,
        in->d_crp_p, in->d_crp_p2,
        in->crp_pm1, in->crp_pp1,
        in->crp_pm1_p2, in->crp_pp1_p2,
        in->d_cre_p, in->d_cre_p2,
        in->cre_pm1, in->cre_pp1,
        in->cre_pm1_p2, in->cre_pp1_p2,
        in->d_dpp_off, in->d_dppm1_off, in->d_dppp1_off,
        in->d_dppe_off, in->d_dppem1_off, in->d_dppep1_off);
    cuda_check(cudaGetLastError(),
               (off_label != nullptr) ? off_label : "launch prepare_momentumdiff_off_kernel");
    return 0;
}

int fp_cuda_build_cc_coeffs(const FpCudaCcBuildInput *in,
                            const char *label)
{
    if (in == nullptr) return -1;

    build_cc_coeffs_kernel<<<in->cell_blocks, in->cell_threads>>>(
        in->ncell, in->dt,
        in->d_crp_p, in->d_crp_dp, in->crp_pm1, in->crp_pp1,
        in->d_cre_p, in->d_cre_dp, in->cre_pm1, in->cre_pp1,
        in->d_crp_radp, in->d_crp_tloss, in->d_dpp,
        in->d_cre_radp, in->d_cre_tloss, in->d_dppe,
        in->d_crp_radpm1, in->d_crp_radpp1,
        in->d_cre_radpm1, in->d_cre_radpp1,
        in->d_dppm1, in->d_dppp1, in->d_dppem1, in->d_dppep1,
        in->d_ccp_a, in->d_ccp_b, in->d_ccp_c,
        in->d_cce_a, in->d_cce_b, in->d_cce_c);
    cuda_check(cudaGetLastError(),
               (label != nullptr) ? label : "launch build_cc_coeffs_kernel");
    return 0;
}
