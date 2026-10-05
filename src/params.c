#include "params.h"
#include "DSA_MODELS.h"

/* Tracer run control */
char output_dir[MAX_LINE_LENGTH];
long int N_TRACERS = 0;
char tracer_backend_spec[MAX_LINE_LENGTH] = "auto";   //  cpu or cuda, or auto //
char tracer_input_mode_spec[MAX_LINE_LENGTH] = "hdf5";  // only hdf5 is supported for now //
char tracer_file_output_spec[MAX_LINE_LENGTH] = "write";
char tracer_write_buffer_mode_spec[MAX_LINE_LENGTH] = "tile";
char tracer_integration_mode_spec[MAX_LINE_LENGTH] = "quantized";  //  no other options... //
int frozen_background = 0;    //  for debug  //
int one_rank_one_gpu = 0;   // (optional) just to make sure  //
int tracer_debug_max_snapshots = 0;
int nsnp_i = 0, nsnp_f = 0;
int nsnp_start = 0;
int synch_nfreq = MAX_SYNCH_FREQ_BINS;

/* Tracer input files */
char tracer_file_dir[MAX_LINE_LENGTH];
char tracer_file_extension[MAX_LINE_LENGTH];
char tracer_filename_base1[MAX_LINE_LENGTH];
char tracer_filename_base2[MAX_LINE_LENGTH];

/* Tracer output and restart */
char tracer_synch_output_spec[MAX_LINE_LENGTH] = "all";
char tracer_checkpoint_dir[MAX_LINE_LENGTH] = "";
char tracer_restart_dir[MAX_LINE_LENGTH] = "";
int tracer_checkpoint_interval = 0;
int tracer_write_buffer_chunk_snapshots = 1;
int tracer_write_crp_output = 1;
int tracer_write_ic_output = 0;
int tracer_write_gamma_output = 0;
int tracer_write_neutrino_output = 0;
int tracer_output_per_cc = 0;            // 0: per tracer, 1: per cm^3 //
int tracer_output_nsnp_min = -1;
int tracer_output_nsnp_max = -1;
double tracer_bucketstats_top_frac = 0.001;

/* Heterogeneous execution and load balancing */
char hetero_heavy_id_file[MAX_LINE_LENGTH] = "";    //  (prelim)  //
int hetero_skip_heavy = 0;    //   (prelim)  //
int load_balancing = 0;
double load_balance_top_frac = 0.01;
int load_balance_chunk_size = 0;
int load_balance_diagnostics = 0;
int load_estimate_virtual_ranks = 10;

/* CR momentum grids and injection */
int variable_tracer_mass = 1;
double M_trc_fix = 1.0e+7;
double pmin = -1.00, pmax = 8.0;
double pemin = -0.5000, pemax = 6.00;   //  p_min = 0.3  //
double pinjmin = 1.00, pinjmax = 1.0e+6;
double peinjmin = 10.0, peinjmax = 1.0e+9;  // "Injection" momentum range, not the grid range  ///
double phi_CRe = 1.0e-6;  // initial CRe number fraction //
double phi_CRp = 0.0 ;    // initial CRp number fraction  //
double delta_CR_inj = 2.2;
char initial_cr_norm_mode_spec[MAX_LINE_LENGTH] = "number";
char initial_cr_energy_anchor_spec[MAX_LINE_LENGTH] = "electron";
int initial_cr_norm_mode = INITIAL_CR_NORM_NUMBER;
int initial_cr_energy_anchor = INITIAL_CR_ENERGY_ANCHOR_ELECTRON;
double initial_cr_energy_ratio_p_to_e = 1.0;
double L_CR_e = 1.0e+41;  // (prelim) CR injection luminosity  //
double L_CR_p = 0.0;  // (prelim) CR injection luminosity  //
int InjectionModel = 0;  // 0: one-shot,  1: continuous (AGN model)
int steady_primary_electron_injection = 0;  // 0: one-shot init, 1: constant qepri source
char seed_cr_species_spec[MAX_LINE_LENGTH] = "";
int seed_cr_species = SEED_CR_SPECIES_ELECTRON_ONLY;

/* Fokker-Planck coefficients and reacceleration */
char dpp_mode_spec[MAX_LINE_LENGTH] = "";
char ttd_tacc_model_spec[MAX_LINE_LENGTH] = "brunetti16";
int ttd_tacc_model = TTD_TACC_MODEL_BRUNETTI16;
double psi = 0.5;       //  MFP parameter //
double mach_limit = 0.5;       // cap for turb Mach number //
double eta_dpp_cap = 0.0;      // cap for epsilon_CR/(Q_turb*t_acc); <=0 disables //
double t_acc_direct_gyr = 0.0;
double t_off_gyr = 0.0;
double t_acc_off_gyr = 3.0;
double f_eddy = 2.0;             // (optional) 0: no limits,  >0: limiting to f_eddy*t_eddy  //
double tracer_nsub_safety = 0.25;  // CFL-like safety factor  //
int tracer_nsub_max = 0;
int coeff_interp_min_steps = 4;
int coeff_interp_max_segments = 16;   // if increased, the run may slow down //
int reacc_window_mode = REACC_WINDOW_ALWAYS_ON;
double q_sol = 5.0/3.0, q_comp = 3.0/2.0;
double L_turb = 2.0*0.0158;      // (deprecated) if variable_turb_scale = 0, you need to specify the scale //
double L_turb_target_kpc = 150.0; // target scale used to rescale tracer turbulence [kpc]

/* Magnetic field */
char bfield_mode_spec[MAX_LINE_LENGTH] = "max";
int bfield_mode = BFIELD_MODE_MAX;     //  0: max, 1: simulation original, 2: dynamo //
double eta_B = 0.05;

/* DSA parameters */
DSAModelID DSAEtaModelInitial = DSA_MODEL_RYU19;
DSAModelID DSAEtaModelReacc   = DSA_MODEL_RYU19R;
char DSAInjectionModeSpec[MAX_LINE_LENGTH] = "off";
char DSAReaccModeSpec[MAX_LINE_LENGTH] = "off";
char DSAInjectSpeciesSpec[MAX_LINE_LENGTH] = "both";
double DSAChiP    = 3.5;         //  p_inj/p_th,p  for CRp  //
double DSAChiE    = 3.5;         //  p_inj/p_th,e  for CRe  //
double DSAKep     = 1.0e-2;      //  electron/proton ratio  //
double DSAPmaxPmc = 1.0e6;       //  DSA p_max (CRp)        //
double DSAPmaxEmc = 1.0e5;       //  DSA p_max (CRe)        //
double DSAMinMach = 1.5;
double DSAGammaGas = 5.0 / 3.0;  // adiab index //
double DSAXcrPminPmc = 1.0;
double DSAReaccEtaCap = 0.0;  // <=0 disables //
int DSAShockRequired = 0;
int DSADebug = 0;
int DSAInjectSpecies = DSA_INJECT_SPECIES_BOTH;

/* Emission spectra */
double nu_min_s = 5.5,nu_max_s = 10.5;       //  synchrotron frequency in log_10 Hz  //
double tracer_synch_logb_min = -6.0, tracer_synch_logb_max = 2.0;   //  synchrotron precomputed log B table  //
int adaptive_synch_logb = 0;         //  adaptive log B table, depending on B during run  //
int tracer_synch_ntheta_pitch = N_theta_pitch;
double nu_min_ic = 13.0,nu_max_ic = 28.0;      //  inverse-Compton  //
double E_gamma_min = -2.0,E_gamma_max = 6.0;
double E_nu_min = -2.0,E_nu_max = 8.0;
double E_gamma_min_pg = 15.0, E_gamma_max_pg = 20.0;
double E_nu_min_pg = 15.0, E_nu_max_pg = 20.0;

/* Cosmology and gas */
double mu_mol = 0.59;    // mean molecular weight ///

/* Parallel run parameters */
int job_count = 1, job_index = 0;
int job_parallel_enabled = 0;
int openmp_enabled = 0, openmp_threads = 0;

/* Grid and mesh-FP parameters */
int  grid_mode = GRID_MODE_TRACER;
int  decomp_layout = DECOMP_LAYOUT_PARAM_AUTO;
int  Nx = 0, Ny = 0, Nz = 0;
char grid_file_dir[MAX_LINE_LENGTH]     = "";
char grid_filename_base[MAX_LINE_LENGTH] = "grid_";
char aux_file_dir[MAX_LINE_LENGTH]      = "";
char aux_filename_base[MAX_LINE_LENGTH]  = "grid_turb_";
int  grid_snap_pad = 4;
int    interp_snap  = 0; // (deprecated) turn on FP coeff interpolation //
double t_fp_total   = 1.0;
double dt_fp        = 0.1;
int    n_fp_out     = 10;
int    diffuse_mode = 0;
int    diffuse_halo = -1;
double kappa_diff_0 = 6.1e28;
double delta_diff   = 1.0/3.0;
double R0_diff      = 4.0;
double kappa_perp_ratio = 1.0;
int    advect_mode  = 0;

/* Legacy compatibility parameters. Prefer canonical parameters for new code. */
int ASA = 0, TTD = 0, direct_tacc_enabled = 0;  //  Fermi-II models  //
int variable_turb_scale = 0;     // 0: fixed, 1: variable //
int Secondary = 0, Mix = 0;      // CR species legacy flags //
int ParallelHDFIO = 0;
int Use_MPI = 0;
int Use_OMP = 0, NUM_THREADS = 0;
int NUM_RUN = 1, RUN = 0;
int hetero_mode = 0, hetero_gpu_ranks = 0;
int np_10GeV = 0;
int nsnp_out_i = 0, nsnp_out_f = 500;
double cellwidth_Mpc = 0.0158, z_ini = 30.0, f_bary = 0.13;
double f_ep = 1.0e-3, phi_CR_mix = 1.0e-12;
double E_brem_min = -6.0, E_brem_max = 10.0;
