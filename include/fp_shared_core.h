#ifndef INCLUDED_fp_shared_core_h_
#define INCLUDED_fp_shared_core_h_

#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Build the hadronic secondary-electron quadrature weights in a flat
 * row-major layout:
 *
 *   fqe_flat[je * np + jp]
 *
 * where:
 *   je = electron momentum-bin index [0, npe)
 *   jp = proton   momentum-bin index [0, np)
 *
 * np_min_qe[je] stores the first proton-bin index with a nonzero weight.
 * The weights include the trapezoidal proton-bin widths, so source assembly is
 * a dot product over CRp[jp] * fqe_flat[je, jp].
 */
void prepare_secondary_kernel_flat(const CRspectrum *crp,
                                           const CRspectrum *cre,
                                           const double *beta_p,
                                           double *fqe_flat,
                                           int *np_min_qe);

/*
 * Runtime switch for the secondary-electron weight layout.
 * Transposed layout is enabled by default:
 *
 *   fqe_flat[jp * npe + je]
 *
 * Set CROMA_SECONDARY_TRANSPOSE=0 to use the legacy row-major layout:
 *
 *   fqe_flat[je * np + jp]
 */
int tracer_fp_secondary_transpose_enabled(void);

/*
 * Assemble secondary-electron source terms from a momentum-major proton batch.
 *
 * Input layout:
 *   crp_batch[jp * ncell + icell]
 *   qepri_batch[je * ncell + icell]
 *   n_gas[icell]
 *
 * Output layout:
 *   qe_integral_batch[je * ncell + icell]
 *   inje_batch[je * ncell + icell]
 *
 * This layout is chosen to match a GPU-oriented execution model where each
 * (electron-bin, cell) pair performs a reduction over proton bins.
 */
void prepare_secondary_sources_batch(int ncell,
                                             const double *n_gas,
                                             const double *crp_batch,
                                             const double *crp_dp,
                                             const int *np_min_qe,
                                             const double *fqe_flat,
                                             const double *qepri_batch,
                                             double *qe_integral_batch,
                                             double *inje_batch);

/*
 * Assemble secondary-electron source terms from a cell-major proton state:
 *
 *   crp_state_cell_major[icell * np + jp]
 *
 * Primary and output source terms are cell-major:
 *
 *   qepri_batch[icell * npe + je]
 *   qe_integral_batch[icell * npe + je]
 *   inje_batch[icell * npe + je]
 *
 * This path avoids an intermediate transpose when the local FP solve keeps
 * proton/electron state in cell-major layout.
 */
void prepare_secondary_sources_cell_major(int ncell,
                                                  const double *n_gas,
                                                  const double *crp_state_cell_major,
                                                  const double *crp_dp,
                                                  const int *np_min_qe,
                                                  const double *fqe_flat,
                                                  const double *qepri_batch,
                                                  double *qe_integral_batch,
                                                  double *inje_batch);

/*
 * GEMM-shaped experimental helper for secondary-electron source assembly.
 *
 * Required input layout:
 *   crp_state_cell_major[icell * np + jp]
 *   fqe_transposed[jp * npe + je]
 *   qepri_batch[icell * npe + je]
 *
 * Output layout:
 *   qe_integral_batch[icell * npe + je]
 *   inje_batch[icell * npe + je]
 *
 * This computes the same matrix product shape as:
 *   qe_integral[ncell, npe] = crp_state[ncell, np] * fqe_transposed[np, npe]
 *
 * It is intentionally not wired into the production path yet. The current
 * implementation is a scalar/SIMD fallback with the same data contract as a
 * future CBLAS/cuBLAS GEMM call.
 */
void prepare_secondary_sources_cell_major_gemm_layout(int ncell,
                                                              const double *n_gas,
                                                              const double *crp_state_cell_major,
                                                              const double *fqe_transposed,
                                                              const double *qepri_batch,
                                                              double *qe_integral_batch,
                                                              double *inje_batch);

/*
 * Build the hadronic gamma-emission kernel in a flat row-major layout:
 *
 *   fga_flat[ng * np + jp]
 *
 * where:
 *   ng = gamma-energy bin index [0, nbins)
 *   jp = proton momentum-bin index [0, np)
 *
 * The energy grid follows the same log-spacing convention as outputgamma():
 *   log10(E_gamma/GeV) in [egamma_min, egamma_max]
 */
void prepare_gamma_kernel_flat(const CRspectrum *crp,
                                       int nbins,
                                       double egamma_min,
                                       double egamma_max,
                                       double *fga_flat);

/*
 * Build the hadronic neutrino-emission kernel in a flat row-major layout:
 *
 *   fnu_flat[nnu * np + jp]
 *
 * where:
 *   nnu = neutrino-energy bin index [0, nbins)
 *   jp  = proton momentum-bin index [0, np)
 *
 * The energy grid is log-spaced in GeV:
 *   log10(E_nu/GeV) in [enu_min, enu_max]
 *
 * The default kernel sums nu_e + nu_mu from muon decay + nu_mu from pion
 * decay, matching Kernels_z().
 */
void prepare_neutrino_kernel_flat(const CRspectrum *crp,
                                          int nbins,
                                          double enu_min,
                                          double enu_max,
                                          double *fnu_flat);

/*
 * Build the inverse-Compton emission kernel in a flat row-major layout:
 *
 *   fic_flat[nic * npe + je]
 *
 * The frequency grid is log-spaced in Hz:
 *   log10(nu/Hz) in [nu_min, nu_max]
 */
void prepare_ic_kernel_flat(const CRspectrum *cre,
                                    int nbins,
                                    double nu_min,
                                    double nu_max,
                                    double z,
                                    double *fic_flat);

/*
 * GPU-oriented decomposition of CR_Coef_1D:
 * build proton and electron loss terms separately from read-only cell inputs.
 */
int find_pp_threshold_index(const CRspectrum *crp);

void prepare_crp_losses_1d(double n_gas,
                                   double kbt,
                                   double divv,
                                   const CRspectrum *crp,
                                   FPloss *crp_loss);

void prepare_cre_losses_1d(double n_gas,
                                   double b_field,
                                   double divv,
                                   const double *rad_ic,
                                   double rad_ic_m1,
                                   double rad_ic_p1,
                                   const CRspectrum *cre,
                                   FPloss *cre_loss);

/*
 * Build one snapshot-wide IC-cooling row for the electron momentum grid.
 *
 * The returned row is common to every cell/tracer at the same snapshot:
 *   rad_ic_row[je]     for je in [0, npe)
 *   *rad_ic_m1_out     edge value at cre->pm1
 *   *rad_ic_p1_out     edge value at cre->pp1
 */
int prepare_ic_cooling_row(double z,
                                   const CRspectrum *cre_grid,
                                   double *rad_ic_row,
                                   double *rad_ic_m1_out,
                                   double *rad_ic_p1_out);

typedef struct {
    double density_gcc;
    double temp_K;
    double bx_G;
    double by_G;
    double bz_G;
    double divv_gyr;
    double curl_v_s;
    double curl_v_prev_s;
    double curl_v_next_s;
    double z_prev;
    double z_curr;
    double z_next;
    double l_turb_cm;
    double target_l_turb_cm;
    int apply_temp_floor;
    int allow_curl_interp;
} FpBackgroundCellInput;

typedef struct {
    double n_gas;
    double kbt_GeV;
    double b_sim_G;
    double b_dyn_G;
    double b_eff_G;
    double divv_gyr;
    double l_turb_mpc;
    double dv_imc_cms;
    double cs_cms;
    double beta_pl;
    double curl_v_used_s;
    double temp_used_K;
} FpBackgroundCellOutput;

int prepare_background_cell(const FpBackgroundCellInput *in,
                                    FpBackgroundCellOutput *out);

typedef struct {
    int ncell;
    double epmax;
    int model;

    const double *l_turb_mpc;     /* [ncell] */
    const double *dv_imc;         /* [ncell] */
    const double *cs;             /* [ncell] */
    const double *beta_pl;        /* [ncell] */
    const double *tracer_mass_msun;       /* [ncell], optional for eta_dpp_cap */
    const double *crp_state_cell_major;   /* [ncell * np], optional for eta_dpp_cap */
    const double *cre_state_cell_major;   /* [ncell * npe], optional for eta_dpp_cap */
} FpMomentumDiffBatchInput;

enum {
    FP_MOMENTUMDIFF_MODEL_AUTO = 0,
    FP_MOMENTUMDIFF_MODEL_ASA = 1,
    FP_MOMENTUMDIFF_MODEL_TTD = 2,
    FP_MOMENTUMDIFF_MODEL_OFF = 3,
    FP_MOMENTUMDIFF_MODEL_DIRECT_TACC = 4
};

typedef struct {
    double *dpp_batch;            /* [np  * ncell] */
    double *dppe_batch;           /* [npe * ncell] */

    double *dppm1;                /* [ncell] */
    double *dppp1;                /* [ncell] */
    double *dppem1;               /* [ncell] */
    double *dppep1;               /* [ncell] */
} FpMomentumDiffBatchOutput;

int prepare_momentumdiff_asa_batch(const FpMomentumDiffBatchInput *in,
                                           const CRspectrum *crp_grid,
                                           const CRspectrum *cre_grid,
                                           FpMomentumDiffBatchOutput *out);
int prepare_momentumdiff_ttd_batch(const FpMomentumDiffBatchInput *in,
                                           const CRspectrum *crp_grid,
                                           const CRspectrum *cre_grid,
                                           FpMomentumDiffBatchOutput *out);
int prepare_momentumdiff_direct_tacc_batch(const FpMomentumDiffBatchInput *in,
                                                   const CRspectrum *crp_grid,
                                                   const CRspectrum *cre_grid,
                                                   FpMomentumDiffBatchOutput *out);
int prepare_momentumdiff_off_batch(const FpMomentumDiffBatchInput *in,
                                           const CRspectrum *crp_grid,
                                           const CRspectrum *cre_grid,
                                           FpMomentumDiffBatchOutput *out);
int prepare_momentumdiff_batch(const FpMomentumDiffBatchInput *in,
                                       const CRspectrum *crp_grid,
                                       const CRspectrum *cre_grid,
                                       FpMomentumDiffBatchOutput *out);
typedef struct {
    double *dpp_batch;            /* [ncell * np ] cell-major */
    double *dppe_batch;           /* [ncell * npe] cell-major */

    double *dppm1;                /* [ncell] */
    double *dppp1;                /* [ncell] */
    double *dppem1;               /* [ncell] */
    double *dppep1;               /* [ncell] */
} FpMomentumDiffCellMajorOutput;

int prepare_momentumdiff_cell_major(const FpMomentumDiffBatchInput *in,
                                            const CRspectrum *crp_grid,
                                            const CRspectrum *cre_grid,
                                            FpMomentumDiffCellMajorOutput *out);
int resolve_momentumdiff_model(int requested_model);
const char *momentumdiff_model_name(int model);

typedef struct {
    int ncell;

    const double *n_gas;          /* [ncell] */
    const double *kbt;            /* [ncell] */
    const double *b_field;        /* [ncell] */
    const double *divv_gyr;       /* [ncell] */
    const unsigned char *disable_adiabatic; /* [ncell], optional */

    /* electron-loss IC tables, momentum-major by electron bin */
    const double *rad_ic_batch;   /* [npe * ncell] */
    const double *rad_ic_m1;      /* [ncell] */
    const double *rad_ic_p1;      /* [ncell] */
} FpLossBatchInput;

typedef struct {
    double *crp_radp_batch;       /* [np  * ncell] */
    double *crp_tloss_batch;      /* [np  * ncell] */
    double *crp_invtloss_batch;   /* [np  * ncell] */
    double *cre_radp_batch;       /* [npe * ncell] */
    double *cre_tloss_batch;      /* [npe * ncell] */
    double *cre_invtloss_batch;   /* [npe * ncell] */

    double *crp_radpm1;           /* [ncell] */
    double *crp_radpp1;           /* [ncell] */
    double *cre_radpm1;           /* [ncell] */
    double *cre_radpp1;           /* [ncell] */
} FpLossBatchOutput;

int prepare_losses_batch(const FpLossBatchInput *in,
                                 const CRspectrum *crp_grid,
                                 const CRspectrum *cre_grid,
                                 FpLossBatchOutput *out);

typedef struct {
    double *crp_radp_batch;       /* [ncell * np ] cell-major */
    double *crp_tloss_batch;      /* [ncell * np ] cell-major */
    double *crp_invtloss_batch;   /* [ncell * np ] cell-major */
    double *cre_radp_batch;       /* [ncell * npe] cell-major */
    double *cre_tloss_batch;      /* [ncell * npe] cell-major */
    double *cre_invtloss_batch;   /* [ncell * npe] cell-major */

    double *crp_radpm1;           /* [ncell] */
    double *crp_radpp1;           /* [ncell] */
    double *cre_radpm1;           /* [ncell] */
    double *cre_radpp1;           /* [ncell] */
} FpLossCellMajorOutput;

int prepare_losses_cell_major(const FpLossBatchInput *in,
                                      const CRspectrum *crp_grid,
                                      const CRspectrum *cre_grid,
                                      FpLossCellMajorOutput *out);

typedef struct {
    int ncell;
    double z;
    double dt_gyr;
    double epmax;

    const double *n_gas;
    const double *kbt;
    const double *b_field;
    const double *divv_gyr;
    const unsigned char *disable_adiabatic;
    const double *l_turb_mpc;
    const double *dv_imc;
    const double *cs;
    const double *beta_pl;
    const double *tracer_mass_msun;
    const double *crp_state_cell_major;
    const double *cre_state_cell_major;

    /* electron-loss IC tables, momentum-major by electron bin */
    const double *rad_ic_batch;   /* [npe * ncell] */
    const double *rad_ic_m1;      /* [ncell] */
    const double *rad_ic_p1;      /* [ncell] */

    /* Optional CUDA-only path: interpolate background inputs on device.
     * When enabled, the primary pointers above are the current-snapshot
     * values, the *_next pointers below are the next-snapshot values, and
     * interp_alpha is used to form the coefficient inputs on the GPU. */
    int device_interp_background;
    double interp_alpha;
    const double *n_gas_next;
    const double *kbt_next;
    const double *b_field_next;
    const double *divv_gyr_next;
    const double *l_turb_mpc_next;
    const double *dv_imc_next;
    const double *cs_next;
    const double *beta_pl_next;
    const double *rad_ic_batch_next;   /* [npe * ncell] */
    const double *rad_ic_m1_next;      /* [ncell] */
    const double *rad_ic_p1_next;      /* [ncell] */
} FpCoeffBatchInput;

typedef struct {
    FpLossBatchOutput loss;

    /* momentum diffusion terms, momentum-major by bin */
    double *dpp_batch;            /* [np  * ncell] */
    double *dppe_batch;           /* [npe * ncell] */
    double *dpp_off_batch;        /* [np  * ncell] */
    double *dppe_off_batch;       /* [npe * ncell] */

    /* edge terms, cell-major */
    double *dppm1;                /* [ncell] */
    double *dppp1;                /* [ncell] */
    double *dppem1;               /* [ncell] */
    double *dppep1;               /* [ncell] */

    double *dppm1_off;            /* [ncell] */
    double *dppp1_off;            /* [ncell] */
    double *dppem1_off;           /* [ncell] */
    double *dppep1_off;           /* [ncell] */
} FpCoeffBatchOutput;

typedef struct {
    FpLossCellMajorOutput loss;

    double *dpp_batch;            /* [ncell * np ] cell-major */
    double *dppe_batch;           /* [ncell * npe] cell-major */
    double *dpp_off_batch;        /* [ncell * np ] cell-major */
    double *dppe_off_batch;       /* [ncell * npe] cell-major */

    double *dppm1;                /* [ncell] */
    double *dppp1;                /* [ncell] */
    double *dppem1;               /* [ncell] */
    double *dppep1;               /* [ncell] */

    double *dppm1_off;            /* [ncell] */
    double *dppp1_off;            /* [ncell] */
    double *dppem1_off;           /* [ncell] */
    double *dppep1_off;           /* [ncell] */
} FpCoeffCellMajorOutput;

typedef struct {
    FpLossBatchInput loss;
    FpMomentumDiffBatchInput momentumdiff_on;
    FpMomentumDiffBatchInput momentumdiff_off;
} FpCoeffGpuInput;

typedef struct {
    FpLossBatchOutput loss;
    FpMomentumDiffBatchOutput momentumdiff_on;
    FpMomentumDiffBatchOutput momentumdiff_off;
} FpCoeffGpuOutput;

int prepare_fp_coefficients_gpu_batch(const FpCoeffGpuInput *in,
                                              const CRspectrum *crp_grid,
                                              const CRspectrum *cre_grid,
                                              FpCoeffGpuOutput *out);

typedef struct {
    int ncell;
    int nfreq;
    int nx_tab;
    int ntheta_pitch;
    int nlogb;
    double z;
    double xmin;
    double logb_min;
    double inv_dlogb;

    const double *fx_tab;
    const double *logfx_tab;
    const double *logx_tab;
    double ***logy;
    const double *pitch_kernel_table; /* [nlogb * nfreq * npe] or NULL */

    const double *b_dyn;          /* [ncell] */
    const double *logb;           /* [ncell] */
    const double *cre_batch;      /* [ncell * npe] cell-major */

    const double *gamma2e;
    const double *theta;
    const double *dtheta;
    const double *pitch_weight;
    const double *nus;
} FpSynchEmissionBatchInput;

typedef struct {
    int ncell;
    int nbins;
    double z;
    double nu_min;                /* log10(nu/Hz) */
    double nu_max;                /* log10(nu/Hz) */

    const double *cre_batch;      /* [ncell * npe] cell-major */
    const double *fic_flat;       /* [nbins * npe] row-major */
} FpIcEmissionBatchInput;

typedef struct {
    int ncell;
    int nbins;
    double z;
    double egamma_min;
    double egamma_max;

    const double *n_gas;          /* [ncell] */
    const double *crp_batch;      /* [ncell * np] cell-major */
    const double *beta_p;
    const double *fga_flat;       /* [nbins * np] row-major */
} FpGammaEmissionBatchInput;

typedef struct {
    int ncell;
    int nbins;
    double z;
    double enu_min;               /* log10(E_nu/GeV) */
    double enu_max;               /* log10(E_nu/GeV) */

    const double *n_gas;          /* [ncell] */
    const double *crp_batch;      /* [ncell * np] cell-major */
    const double *beta_p;
    const double *fnu_flat;       /* [nbins * np] row-major */
} FpNeutrinoEmissionBatchInput;

typedef struct {
    FpSynchEmissionBatchInput synch;
    const FpIcEmissionBatchInput *ic;
    const FpGammaEmissionBatchInput *gamma;
    const FpNeutrinoEmissionBatchInput *neutrino;
} FpEmissionGpuInput;

typedef struct {
    double *eps_syn_batch;        /* [ncell * nfreq]  cell-major */
    double *eps_ic_batch;         /* [ncell * nbins]  cell-major, optional */
    double *eps_gamma_batch;      /* [ncell * nbins]  cell-major, optional */
    double *eps_nu_batch;         /* [ncell * nbins]  cell-major, optional */
} FpEmissionGpuOutput;

int prepare_synch_emission_batch(const FpSynchEmissionBatchInput *in,
                                         const CRspectrum *cre_grid,
                                         double *eps_syn_batch);

int prepare_synch_pitch_kernel_table(const FpSynchEmissionBatchInput *in,
                                             double *kernel_table);

int prepare_synch_emission_cell_major(const FpSynchEmissionBatchInput *in,
                                              const CRspectrum *cre_grid,
                                              double *eps_syn_batch);

int prepare_ic_emission_batch(const FpIcEmissionBatchInput *in,
                                      const CRspectrum *cre_grid,
                                      double *eps_ic_batch);

int prepare_ic_emission_cell_major(const FpIcEmissionBatchInput *in,
                                           const CRspectrum *cre_grid,
                                           double *eps_ic_batch);

int prepare_gamma_emission_batch(const FpGammaEmissionBatchInput *in,
                                         const CRspectrum *crp_grid,
                                         double *eps_gamma_batch);

int prepare_gamma_emission_cell_major(const FpGammaEmissionBatchInput *in,
                                              const CRspectrum *crp_grid,
                                              double *eps_gamma_batch);

int prepare_neutrino_emission_batch(const FpNeutrinoEmissionBatchInput *in,
                                            const CRspectrum *crp_grid,
                                            double *eps_nu_batch);

int prepare_neutrino_emission_cell_major(const FpNeutrinoEmissionBatchInput *in,
                                                 const CRspectrum *crp_grid,
                                                 double *eps_nu_batch);

int prepare_emission_gpu_batch(const FpEmissionGpuInput *in,
                                       const CRspectrum *crp_grid,
                                       const CRspectrum *cre_grid,
                                       FpEmissionGpuOutput *out);

typedef struct {
    int ncell;
    double dt;

    const double *crp_radp_batch;       /* [np  * ncell] momentum-major */
    const double *crp_tloss_batch;      /* [np  * ncell] momentum-major */
    const double *dpp_batch;            /* [np  * ncell] momentum-major */
    const double *qpi_batch;            /* [ncell * np ] cell-major */

    const double *cre_radp_batch;       /* [npe * ncell] momentum-major */
    const double *cre_tloss_batch;      /* [npe * ncell] momentum-major */
    const double *dppe_batch;           /* [npe * ncell] momentum-major */
    const double *inje_batch;           /* [ncell * npe] cell-major */

    const double *crp_radpm1;           /* [ncell] */
    const double *crp_radpp1;           /* [ncell] */
    const double *cre_radpm1;           /* [ncell] */
    const double *cre_radpp1;           /* [ncell] */
    const double *dppm1;                /* [ncell] */
    const double *dppp1;                /* [ncell] */
    const double *dppem1;               /* [ncell] */
    const double *dppep1;               /* [ncell] */
} FpChangCooperCoeffBatchInput;

typedef struct {
    double *ccp_a_batch;                /* [ncell * np ] cell-major */
    double *ccp_b_batch;                /* [ncell * np ] cell-major */
    double *ccp_c_batch;                /* [ncell * np ] cell-major */
    double *cce_a_batch;                /* [ncell * npe] cell-major */
    double *cce_b_batch;                /* [ncell * npe] cell-major */
    double *cce_c_batch;                /* [ncell * npe] cell-major */
} FpChangCooperCoeffBatchOutput;

int prepare_cc_coeff_batch(
    const FpChangCooperCoeffBatchInput *in,
    const CRspectrum *crp_grid,
    const CRspectrum *cre_grid,
    FpChangCooperCoeffBatchOutput *out);

typedef struct {
    int ncell;
    double dt;

    const double *crp_radp_batch;       /* [ncell * np ] cell-major */
    const double *crp_tloss_batch;      /* [ncell * np ] cell-major */
    const double *dpp_batch;            /* [ncell * np ] cell-major */
    const double *qpi_batch;            /* [ncell * np ] cell-major */

    const double *cre_radp_batch;       /* [ncell * npe] cell-major */
    const double *cre_tloss_batch;      /* [ncell * npe] cell-major */
    const double *dppe_batch;           /* [ncell * npe] cell-major */
    const double *inje_batch;           /* [ncell * npe] cell-major */

    const double *crp_radpm1;           /* [ncell] */
    const double *crp_radpp1;           /* [ncell] */
    const double *cre_radpm1;           /* [ncell] */
    const double *cre_radpp1;           /* [ncell] */
    const double *dppm1;                /* [ncell] */
    const double *dppp1;                /* [ncell] */
    const double *dppem1;               /* [ncell] */
    const double *dppep1;               /* [ncell] */
} FpChangCooperCoeffCellMajorInput;

int prepare_cc_coeff_cell_major(
    const FpChangCooperCoeffCellMajorInput *in,
    const CRspectrum *crp_grid,
    const CRspectrum *cre_grid,
    FpChangCooperCoeffBatchOutput *out);

typedef struct {
    int nsys;
    int nrow;

    const double *a_batch;        /* [nsys * nrow] cell-major, positive CC lower-coupling magnitudes */
    const double *b_batch;        /* [nsys * nrow] cell-major */
    const double *c_batch;        /* [nsys * nrow] cell-major, positive CC upper-coupling magnitudes */
    const double *rhs_batch;      /* [nsys * nrow] cell-major */
} FpTridiagBatchInput;

/*
 * Thomas-batch is the scalar reference path.
 * PCR-batch mirrors the data flow we ultimately want on GPU.
 */
int fp_thomas_solve_batch_cpu(const FpTridiagBatchInput *in,
                              double *solution_batch);
int solve_tridiagonal_batch_pcr_cpu(const FpTridiagBatchInput *in,
                                            double *solution_batch);
int solve_tridiagonal_batch_cpu(const FpTridiagBatchInput *in,
                                        double *solution_batch);
int fp_thomas_solve_cpu(int nrow,
                        const double *a,
                        const double *b,
                        const double *c,
                        double *rhs_solution);
const char *cpu_tridiag_solver_name(void);

typedef struct {
    int nsys;
    int nrow;
    double dt;

    const double *a_batch;        /* [nsys * nrow] cell-major */
    const double *b_batch;        /* [nsys * nrow] cell-major */
    const double *c_batch;        /* [nsys * nrow] cell-major */
    const double *source_batch;   /* [nsys * nrow] cell-major */
} FpChangCooperSolveBatchInput;

typedef struct {
    double alloc_ms;
    double rhs_ms;
    double tridiag_ms;
} FpChangCooperSolveBatchCpuTimes;

int solve_cc_1d_cpu(int nrow,
                                      double dt,
                                      const double *a,
                                      const double *b,
                                      const double *c,
                                      const double *source,
                                      double *state);
int solve_cc_batch_cpu(
    const FpChangCooperSolveBatchInput *in,
    double *state_batch);
int solve_cc_batch_cpu_timed(
    const FpChangCooperSolveBatchInput *in,
    double *state_batch,
    FpChangCooperSolveBatchCpuTimes *times);

typedef struct {
    int ncell;
    double dt;

    const double *ccp_a_batch;    /* [ncell * np ] cell-major */
    const double *ccp_b_batch;    /* [ncell * np ] cell-major */
    const double *ccp_c_batch;    /* [ncell * np ] cell-major */
    const double *qpi_batch;      /* [ncell * np ] cell-major */

    const double *cce_a_batch;    /* [ncell * npe] cell-major */
    const double *cce_b_batch;    /* [ncell * npe] cell-major */
    const double *cce_c_batch;    /* [ncell * npe] cell-major */
    const double *inje_batch;     /* [ncell * npe] cell-major */
} FpLocalFpSolveBatchInput;

int solve_local_fp_batch_cpu(const FpLocalFpSolveBatchInput *in,
                                     double *crp_state_batch,
                                     double *cre_state_batch);

/*
 * GPU-oriented batch API for cell-local FP coefficient preparation.
 *
 * The API is intentionally contiguous and batch-oriented even though the
 * current implementation is a CPU reference wrapper around the existing
 * scalar routines.
 *
 * Returns 0 on success, non-zero on invalid input.
 */
int prepare_fp_coefficients_batch(const FpCoeffBatchInput *in,
                                          const CRspectrum *crp_grid,
                                          const CRspectrum *cre_grid,
                                          FpCoeffBatchOutput *out);
int prepare_fp_coefficients_cell_major_cpu(const FpCoeffBatchInput *in,
                                                   const CRspectrum *crp_grid,
                                                   const CRspectrum *cre_grid,
                                                   FpCoeffCellMajorOutput *out);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_fp_shared_core_h_ */
