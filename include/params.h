#ifndef INCLUDED_params_h_
#define INCLUDED_params_h_

#define SUCCESS 0
#define FAIL    1

#define bins_gamma 128
#define bins_IC  128
#define bins_nu 128
#define bins_brem 128
#define MAX_SYNCH_FREQ_BINS 128
#define N_theta_pitch 512
#define MAX_LINE_LENGTH 512

////////   simulation set-ups  ////
#define SIM_DIMENSION 3

#define NUM_FIELD 17
#include "tracer_input_fields.h"

//extern int np,npe;
#ifndef npe
#define npe 128
#endif
#ifndef np
#define np 128
#endif

typedef struct {
    double *M;
    double *R_vir;
    double *V_vir;
    double *T_vir;
    double *t_dyn;
    //double *t_dyn_ga;
}DM_halo;


//  np  = 128 //
//typedef struct {
//    double *p;
//    double *dp;
//    double pp1;
//    double pm1;
//    double *E;
//    double *N;
//    double *N_pre;
//    double *N_ave;
//}CRspectrum;


typedef struct {
    double p[np];
    double dp[np];
    double pp1;
    double pm1;
    double E[np];
    double N[np];
    double N_pre[np];
    double N_ave[np];
}CRspectrum;


//  np  = 128 //
typedef struct {
    double *radp;
    double radpp1;
    double radpm1;
    double *tloss;
    double *invtloss;
}FPloss;


typedef struct { //  onezone CC coef. //
    double *A;
    double *B;
    double *C;
}ChangCooper;


typedef struct {
    double m_min;
    double m_max;
    double *m;
    double *dm;
    double mm1;
    double mp1;
    double *r_m;
    double *dr_m;
    double *rho_DM;
    double *rho_gas;
    double rm_rm1;
    double rm_rp1;
}masscorr;

/* Tracer run control */
extern char output_dir[MAX_LINE_LENGTH];
extern long int N_TRACERS;
extern char tracer_backend_spec[MAX_LINE_LENGTH];
extern char tracer_input_mode_spec[MAX_LINE_LENGTH];
extern char tracer_file_output_spec[MAX_LINE_LENGTH];
extern char tracer_write_buffer_mode_spec[MAX_LINE_LENGTH];
extern char tracer_integration_mode_spec[MAX_LINE_LENGTH];
extern int frozen_background;
extern int one_rank_one_gpu;
extern int tracer_debug_max_snapshots;
extern int nsnp_i , nsnp_f, nsnp_start;
extern int synch_nfreq;

/* Tracer input files */
extern char tracer_file_dir[MAX_LINE_LENGTH], tracer_file_extension[MAX_LINE_LENGTH];
extern char tracer_filename_base1[MAX_LINE_LENGTH], tracer_filename_base2[MAX_LINE_LENGTH];

/* Tracer output and restart */
extern char tracer_synch_output_spec[MAX_LINE_LENGTH];
extern char tracer_checkpoint_dir[MAX_LINE_LENGTH];
extern char tracer_restart_dir[MAX_LINE_LENGTH];
extern int tracer_checkpoint_interval;
extern int tracer_write_buffer_chunk_snapshots;
extern int tracer_write_crp_output;
extern int tracer_write_ic_output;
extern int tracer_write_gamma_output;
extern int tracer_write_neutrino_output;
extern int tracer_output_per_cc;
extern int tracer_output_nsnp_min;
extern int tracer_output_nsnp_max;
extern double tracer_bucketstats_top_frac;

/* Heterogeneous execution and load balancing */
extern char hetero_heavy_id_file[MAX_LINE_LENGTH];
extern int hetero_skip_heavy;
extern int load_balancing;
extern double load_balance_top_frac;
extern int load_balance_chunk_size;
extern int load_balance_diagnostics;
extern int load_estimate_virtual_ranks;

/* CR momentum grids and injection */
extern int variable_tracer_mass;
extern double M_trc_fix;
extern double pmax,pmin,pinjmin,pinjmax;
extern double pemax,pemin,peinjmin,peinjmax;
extern double phi_CRe, phi_CRp, delta_CR_inj;
extern char initial_cr_norm_mode_spec[MAX_LINE_LENGTH];
extern char initial_cr_energy_anchor_spec[MAX_LINE_LENGTH];
enum {
    INITIAL_CR_NORM_NUMBER = 0,
    INITIAL_CR_NORM_ENERGY_RATIO = 1
};
enum {
    INITIAL_CR_ENERGY_ANCHOR_ELECTRON = 0,
    INITIAL_CR_ENERGY_ANCHOR_PROTON = 1
};
extern int initial_cr_norm_mode;
extern int initial_cr_energy_anchor;
extern double initial_cr_energy_ratio_p_to_e;
extern double L_CR_e, L_CR_p;
extern int InjectionModel;
extern int steady_primary_electron_injection;
enum {
    SEED_CR_SPECIES_ELECTRON_ONLY = 0,
    SEED_CR_SPECIES_PROTON_ONLY = 1,
    SEED_CR_SPECIES_ELECTRON_PROTON = 2
};
extern char seed_cr_species_spec[MAX_LINE_LENGTH];
extern int seed_cr_species;

/* Fokker-Planck coefficients and reacceleration */
extern char dpp_mode_spec[MAX_LINE_LENGTH];
enum {
    TTD_TACC_MODEL_SPECT = 0,
    TTD_TACC_MODEL_BRUNETTI16 = 1
};
extern char ttd_tacc_model_spec[MAX_LINE_LENGTH];
extern int ttd_tacc_model;
extern double psi;
extern double mach_limit;
extern double eta_dpp_cap;
extern double t_acc_direct_gyr;
extern double t_off_gyr;
extern double t_acc_off_gyr;
extern double f_eddy;
extern double tracer_nsub_safety;
extern int tracer_nsub_max;
extern int coeff_interp_min_steps;
extern int coeff_interp_max_segments;

#define REACC_WINDOW_ALWAYS_ON 0
#define REACC_WINDOW_CENTERED  1
extern int reacc_window_mode;
extern double q_sol, q_comp, L_turb;
extern double L_turb_target_kpc;

/* Magnetic field */
enum {
    BFIELD_MODE_MAX = 0,
    BFIELD_MODE_SIM = 1,
    BFIELD_MODE_DYN = 2
};
extern char bfield_mode_spec[MAX_LINE_LENGTH];
extern int bfield_mode;
extern double eta_B;

/* DSA parameters */
extern char DSAInjectionModeSpec[MAX_LINE_LENGTH];
extern char DSAReaccModeSpec[MAX_LINE_LENGTH];
extern char DSAInjectSpeciesSpec[MAX_LINE_LENGTH];
extern double DSAChiP;
extern double DSAChiE;
extern double DSAKep;
extern double DSAPmaxPmc;
extern double DSAPmaxEmc;
extern double DSAMinMach;
extern double DSAGammaGas;
extern double DSAXcrPminPmc;
extern double DSAReaccEtaCap;
extern int DSAShockRequired;
extern int DSADebug;
enum {
    DSA_INJECT_SPECIES_BOTH = 0,
    DSA_INJECT_SPECIES_ELECTRON = 1,
    DSA_INJECT_SPECIES_PROTON = 2
};
extern int DSAInjectSpecies;

/* Emission spectra */
extern double nu_min_s,nu_max_s,nu_min_ic,nu_max_ic;
extern double tracer_synch_logb_min, tracer_synch_logb_max;
extern int adaptive_synch_logb;
extern int tracer_synch_ntheta_pitch;
extern double E_gamma_max,E_gamma_min, E_nu_max,E_nu_min;
extern double E_gamma_max_pg,E_gamma_min_pg, E_nu_max_pg,E_nu_min_pg;

/* Cosmology and gas */
extern double mu_mol;

/* Parallel run parameters */
extern int job_count, job_index;
extern int job_parallel_enabled;
extern int openmp_enabled, openmp_threads;

/* -------------------------------------------------------------------------
 * Grid mode parameters
 * GRID_MODE_TRACER (0) : original tracer-particle path
 * GRID_MODE_YT     (1) : yt-exported uniform HDF5  (preprocess_grid.py)
 * GRID_MODE_ENZO_RAW(2): raw Enzo DD* (reserved, not yet implemented)
 * -------------------------------------------------------------------------*/
#define GRID_MODE_TRACER   0
#define GRID_MODE_YT       1
#define GRID_MODE_ENZO_RAW 2

#define DECOMP_LAYOUT_PARAM_AUTO      0
#define DECOMP_LAYOUT_PARAM_SLAB_X    1
#define DECOMP_LAYOUT_PARAM_BLOCK_XYZ 2

extern int    grid_mode;
extern int    decomp_layout;          /* 0:auto, 1:slab_x, 2:block_xyz      */
extern int    Nx, Ny, Nz;              /* grid dimensions                    */
extern char   grid_file_dir[MAX_LINE_LENGTH];    /* dir for main grid HDF5   */
extern char   grid_filename_base[MAX_LINE_LENGTH]; /* e.g. "grid_"           */
extern char   aux_file_dir[MAX_LINE_LENGTH];     /* dir for aux turbulence   */
extern char   aux_filename_base[MAX_LINE_LENGTH];  /* e.g. "grid_turb_"      */
extern int    grid_snap_pad;           /* zero-padding width for snap numbers */

/* snapshot field interpolation */
extern int    interp_snap;            /* 0: use current snap; 1: midpoint interp to next */

/* single-snapshot FP mode */
extern double t_fp_total;             /* total evolution time [Gyr] */
extern double dt_fp;                  /* FP timestep [Gyr] */
extern int    n_fp_out;               /* number of output dumps after t=0 */

/* spatial diffusion */
extern int    diffuse_mode;           /* 0: off, 1: Lie splitting, 2: ADI */
extern int    diffuse_halo;           /* MPI transport: -1=auto, 0=allgather, 1=halo */
extern double kappa_diff_0;          /* κ₀ [cm²/s]  — CRPropa default 6.1e28 */
extern double delta_diff;            /* power-law index δ — CRPropa default 1/3 */
extern double R0_diff;               /* reference rigidity [GV] — default 4.0 */
extern double kappa_perp_ratio;      /* κ_perp / κ_parallel for anisotropic diffusion */

/* bulk advection */
extern int    advect_mode;            /* 0: off, 1: upwind (allgather path only) */

/* Legacy compatibility declarations. Prefer canonical parameters for new code. */
extern int ASA, TTD;
extern int direct_tacc_enabled;
extern int variable_turb_scale;
extern int Secondary, Mix;
extern int ParallelHDFIO;
extern int Use_MPI;
extern int Use_OMP, NUM_THREADS;
extern int NUM_RUN, RUN;
extern int hetero_mode, hetero_gpu_ranks;
extern int np_10GeV;
extern int nsnp_out_i, nsnp_out_f;
extern double cellwidth_Mpc, z_ini, f_bary;
extern double f_ep, phi_CR_mix;
extern double E_brem_min, E_brem_max;
extern double q_reacc;
extern double fep_sec,fep_pri,fep_mix,alpha_AGN,alpha_sh,eta_CR_sec,eta_CR_pri,eta_CR_mix;
extern double delta_tu,rcut_tu,beta_tu,f_comp_merge,f_comp_relax;
extern double f_diff,B0;

#endif
