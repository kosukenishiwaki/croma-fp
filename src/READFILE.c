/*
    READFILE.c

    K. Nishiwaki, 2026-06-18
    - recognize param file designated at command line at the execution
    
*/


#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<string.h>
#include"COSFUNC.h"
#include"params.h"
#include"DSA_MODELS.h"

static int normalize_dpp_mode_config(void)
{
    if (dpp_mode_spec[0] == '\0') return SUCCESS;

    if (strcmp(dpp_mode_spec, "asa") == 0) {
        snprintf(dpp_mode_spec, MAX_LINE_LENGTH, "%s", "asa");
        return SUCCESS;
    }
    if (strcmp(dpp_mode_spec, "ttd") == 0) {
        snprintf(dpp_mode_spec, MAX_LINE_LENGTH, "%s", "ttd");
        return SUCCESS;
    }
    if (strcmp(dpp_mode_spec, "direct_tacc") == 0 ||
        strcmp(dpp_mode_spec, "tacc") == 0 ||
        strcmp(dpp_mode_spec, "direct") == 0) {
        snprintf(dpp_mode_spec, MAX_LINE_LENGTH, "%s", "direct_tacc");
        return SUCCESS;
    }
    if (strcmp(dpp_mode_spec, "off") == 0) {
        snprintf(dpp_mode_spec, MAX_LINE_LENGTH, "%s", "off");
        return SUCCESS;
    }

    fprintf(stderr,
            "Dpp_mode must be asa, ttd, direct_tacc, tacc, direct, or off\n");
    return FAIL;
}

static int normalize_ttd_tacc_model_config(void)
{
    if (ttd_tacc_model_spec[0] == '\0' ||
        strcmp(ttd_tacc_model_spec, "default") == 0) {
        snprintf(ttd_tacc_model_spec, MAX_LINE_LENGTH, "%s", "brunetti16");
        ttd_tacc_model = TTD_TACC_MODEL_BRUNETTI16;
        return SUCCESS;
    }
    if (strcmp(ttd_tacc_model_spec, "spect") == 0 ||
        strcmp(ttd_tacc_model_spec, "legacy") == 0) {
        snprintf(ttd_tacc_model_spec, MAX_LINE_LENGTH, "%s", "spect");
        ttd_tacc_model = TTD_TACC_MODEL_SPECT;
        return SUCCESS;
    }
    if (strcmp(ttd_tacc_model_spec, "brunetti16") == 0 ||
        strcmp(ttd_tacc_model_spec, "brunetti2016") == 0 ||
        strcmp(ttd_tacc_model_spec, "brunetti_2016") == 0) {
        snprintf(ttd_tacc_model_spec, MAX_LINE_LENGTH, "%s", "brunetti16");
        ttd_tacc_model = TTD_TACC_MODEL_BRUNETTI16;
        return SUCCESS;
    }

    fprintf(stderr,
            "ttd_tacc_model must be spect or brunetti16\n");
    return FAIL;
}

static int normalize_bfield_mode_config(void)
{
    if (strcmp(bfield_mode_spec, "max") == 0 ||
        strcmp(bfield_mode_spec, "hybrid") == 0) {
        snprintf(bfield_mode_spec, MAX_LINE_LENGTH, "%s", "max");
        bfield_mode = BFIELD_MODE_MAX;
        return SUCCESS;
    }
    if (strcmp(bfield_mode_spec, "sim") == 0 ||
        strcmp(bfield_mode_spec, "simulation") == 0) {
        snprintf(bfield_mode_spec, MAX_LINE_LENGTH, "%s", "sim");
        bfield_mode = BFIELD_MODE_SIM;
        return SUCCESS;
    }
    if (strcmp(bfield_mode_spec, "dyn") == 0 ||
        strcmp(bfield_mode_spec, "dynamo") == 0) {
        snprintf(bfield_mode_spec, MAX_LINE_LENGTH, "%s", "dyn");
        bfield_mode = BFIELD_MODE_DYN;
        return SUCCESS;
    }

    fprintf(stderr, "bfield_mode must be max, sim, or dyn\n");
    return FAIL;
}

static int normalize_seed_cr_species_config(void)
{
    if (strcmp(seed_cr_species_spec, "electron_only") == 0 ||
        strcmp(seed_cr_species_spec, "electron") == 0 ||
        strcmp(seed_cr_species_spec, "electrons") == 0 ||
        strcmp(seed_cr_species_spec, "eonly") == 0) {
        snprintf(seed_cr_species_spec, MAX_LINE_LENGTH, "%s", "electron_only");
        seed_cr_species = SEED_CR_SPECIES_ELECTRON_ONLY;
        return SUCCESS;
    }
    if (strcmp(seed_cr_species_spec, "proton_only") == 0 ||
        strcmp(seed_cr_species_spec, "proton") == 0 ||
        strcmp(seed_cr_species_spec, "protons") == 0 ||
        strcmp(seed_cr_species_spec, "ponly") == 0 ||
        strcmp(seed_cr_species_spec, "secondary") == 0) {
        snprintf(seed_cr_species_spec, MAX_LINE_LENGTH, "%s", "proton_only");
        seed_cr_species = SEED_CR_SPECIES_PROTON_ONLY;
        return SUCCESS;
    }
    if (strcmp(seed_cr_species_spec, "electron_proton") == 0 ||
        strcmp(seed_cr_species_spec, "electron_protons") == 0 ||
        strcmp(seed_cr_species_spec, "electron+proton") == 0 ||
        strcmp(seed_cr_species_spec, "mixed") == 0 ||
        strcmp(seed_cr_species_spec, "mix") == 0 ||
        strcmp(seed_cr_species_spec, "both") == 0) {
        snprintf(seed_cr_species_spec, MAX_LINE_LENGTH, "%s", "electron_proton");
        seed_cr_species = SEED_CR_SPECIES_ELECTRON_PROTON;
        return SUCCESS;
    }

    fprintf(stderr,
            "seed_cr_species must be electron_only, proton_only, or electron_proton\n");
    return FAIL;
}

static int normalize_initial_cr_norm_config(void)
{
    if (initial_cr_norm_mode_spec[0] == '\0' ||
        strcmp(initial_cr_norm_mode_spec, "number") == 0 ||
        strcmp(initial_cr_norm_mode_spec, "number_fraction") == 0) {
        snprintf(initial_cr_norm_mode_spec, MAX_LINE_LENGTH, "%s", "number");
        initial_cr_norm_mode = INITIAL_CR_NORM_NUMBER;
    } else if (strcmp(initial_cr_norm_mode_spec, "energy_ratio") == 0 ||
               strcmp(initial_cr_norm_mode_spec, "integrated_energy") == 0 ||
               strcmp(initial_cr_norm_mode_spec, "energy") == 0) {
        snprintf(initial_cr_norm_mode_spec, MAX_LINE_LENGTH, "%s", "energy_ratio");
        initial_cr_norm_mode = INITIAL_CR_NORM_ENERGY_RATIO;
    } else {
        fprintf(stderr,
                "initial_cr_norm_mode must be number or energy_ratio\n");
        return FAIL;
    }

    if (initial_cr_energy_anchor_spec[0] == '\0' ||
        strcmp(initial_cr_energy_anchor_spec, "electron") == 0 ||
        strcmp(initial_cr_energy_anchor_spec, "cre") == 0 ||
        strcmp(initial_cr_energy_anchor_spec, "e") == 0) {
        snprintf(initial_cr_energy_anchor_spec, MAX_LINE_LENGTH, "%s", "electron");
        initial_cr_energy_anchor = INITIAL_CR_ENERGY_ANCHOR_ELECTRON;
    } else if (strcmp(initial_cr_energy_anchor_spec, "proton") == 0 ||
               strcmp(initial_cr_energy_anchor_spec, "crp") == 0 ||
               strcmp(initial_cr_energy_anchor_spec, "p") == 0) {
        snprintf(initial_cr_energy_anchor_spec, MAX_LINE_LENGTH, "%s", "proton");
        initial_cr_energy_anchor = INITIAL_CR_ENERGY_ANCHOR_PROTON;
    } else {
        fprintf(stderr,
                "initial_cr_energy_anchor must be electron or proton\n");
        return FAIL;
    }

    if (!(initial_cr_energy_ratio_p_to_e > 0.0)) {
        fprintf(stderr,
                "initial_cr_energy_ratio_p_to_e must be > 0\n");
        return FAIL;
    }

    return SUCCESS;
}

static int normalize_dsa_inject_species_config(void)
{
    if (strcmp(DSAInjectSpeciesSpec, "both") == 0 ||
        strcmp(DSAInjectSpeciesSpec, "all") == 0) {
        snprintf(DSAInjectSpeciesSpec, MAX_LINE_LENGTH, "%s", "both");
        DSAInjectSpecies = DSA_INJECT_SPECIES_BOTH;
        return SUCCESS;
    }
    if (strcmp(DSAInjectSpeciesSpec, "electron") == 0 ||
        strcmp(DSAInjectSpeciesSpec, "electrons") == 0 ||
        strcmp(DSAInjectSpeciesSpec, "electron_only") == 0 ||
        strcmp(DSAInjectSpeciesSpec, "eonly") == 0) {
        snprintf(DSAInjectSpeciesSpec, MAX_LINE_LENGTH, "%s", "electron");
        DSAInjectSpecies = DSA_INJECT_SPECIES_ELECTRON;
        return SUCCESS;
    }
    if (strcmp(DSAInjectSpeciesSpec, "proton") == 0 ||
        strcmp(DSAInjectSpeciesSpec, "protons") == 0 ||
        strcmp(DSAInjectSpeciesSpec, "proton_only") == 0 ||
        strcmp(DSAInjectSpeciesSpec, "ponly") == 0) {
        snprintf(DSAInjectSpeciesSpec, MAX_LINE_LENGTH, "%s", "proton");
        DSAInjectSpecies = DSA_INJECT_SPECIES_PROTON;
        return SUCCESS;
    }

    fprintf(stderr, "DSAInjectSpecies must be both, electron, or proton\n");
    return FAIL;
}

static int readfile_should_log_root_only(void)
{
    int initialized = 0;
    int rank = 0;

    MPI_Initialized(&initialized);
    if (!initialized) return 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return (rank == 0);
}

static const char *readfile_legacy_dpp_mode_name(void)
{
    if (ASA == 1) return "asa";
    if (direct_tacc_enabled == 1) return "direct_tacc";
    if (TTD == 1) return "ttd";
    return "off";
}

static const char *readfile_legacy_seed_cr_species_name(void)
{
    if (Mix != 0) return "electron_proton";
    if (Secondary != 0) return "proton_only";
    return "electron_only";
}

int finalize_dpp_mode_config(int legacy_param_seen)
{
    if (dpp_mode_spec[0] == '\0') {
        const int legacy_mode_active =
            (ASA == 1 || TTD == 1 || direct_tacc_enabled == 1);
        const char *mode_name = legacy_mode_active ?
            readfile_legacy_dpp_mode_name() : "asa";

        if (legacy_param_seen || legacy_mode_active) {
            snprintf(dpp_mode_spec, MAX_LINE_LENGTH, "%s", mode_name);
        } else {
            snprintf(dpp_mode_spec, MAX_LINE_LENGTH, "%s", "asa");
        }

        if (legacy_param_seen && readfile_should_log_root_only()) {
            fprintf(stderr,
                    "Deprecated reacceleration parameters "
                    "(ASA/TTD/direct_tacc_enabled) were mapped to "
                    "Dpp_mode = %s\n",
                    dpp_mode_spec);
        }
    } else if (legacy_param_seen && readfile_should_log_root_only()) {
        fprintf(stderr,
                "Deprecated reacceleration parameters "
                "(ASA/TTD/direct_tacc_enabled) were ignored because "
                "Dpp_mode = %s is set\n",
                dpp_mode_spec);
    }

    return normalize_dpp_mode_config();
}

static int finalize_ttd_tacc_model_config(void)
{
    return normalize_ttd_tacc_model_config();
}

static int finalize_seed_cr_species_config(int legacy_param_seen)
{
    if (seed_cr_species_spec[0] == '\0') {
        snprintf(seed_cr_species_spec, MAX_LINE_LENGTH, "%s",
                 readfile_legacy_seed_cr_species_name());
        if (legacy_param_seen && readfile_should_log_root_only()) {
            fprintf(stderr,
                    "Deprecated CR species parameters (Secondary/Mix) "
                    "were mapped to seed_cr_species = %s\n",
                    seed_cr_species_spec);
        }
    } else if (legacy_param_seen && readfile_should_log_root_only()) {
        fprintf(stderr,
                "Deprecated CR species parameters (Secondary/Mix) were "
                "ignored because seed_cr_species = %s is set\n",
                seed_cr_species_spec);
    }

    return normalize_seed_cr_species_config();
}

static void finalize_parallel_param_aliases(int mpi_enabled)
{
    Use_MPI = mpi_enabled;
    if (job_count <= 0) job_count = 1;
    if (job_parallel_enabled != 0) job_parallel_enabled = 1;
    NUM_RUN = job_count;
    RUN = job_index;
    if (openmp_enabled != 0) openmp_enabled = 1;
    if (openmp_threads < 0) openmp_threads = 0;
    Use_OMP = openmp_enabled;
    NUM_THREADS = openmp_threads;
}

static int finalize_cosmology_config(int log_root)
{
    if (!(H0 > 0.0)) {
        if (log_root) fprintf(stderr, "H0 must be > 0\n");
        return FAIL;
    }
    if (!(ns > 0.0)) {
        if (log_root) fprintf(stderr, "ns must be > 0\n");
        return FAIL;
    }
    if (!(sig8 > 0.0)) {
        if (log_root) fprintf(stderr, "sig8 must be > 0\n");
        return FAIL;
    }
    if (!(deltac > 0.0)) {
        if (log_root) fprintf(stderr, "deltac must be > 0\n");
        return FAIL;
    }
    if (!(OmB >= 0.0)) {
        if (log_root) fprintf(stderr, "OmB must be >= 0\n");
        return FAIL;
    }
    if (!(OmC >= 0.0)) {
        if (log_root) fprintf(stderr, "OmC must be >= 0\n");
        return FAIL;
    }
    if (!(OmL >= 0.0)) {
        if (log_root) fprintf(stderr, "OmL must be >= 0\n");
        return FAIL;
    }
    if (!((OmB + OmC) > 0.0)) {
        if (log_root) fprintf(stderr, "OmB + OmC must be > 0\n");
        return FAIL;
    }
    if (!(TCMB > 0.0)) {
        if (log_root) fprintf(stderr, "TCMB must be > 0\n");
        return FAIL;
    }

    h100 = H0 / 100.0;
    h70 = H0 / 70.0;
    OmBh2 = OmB * h100 * h100;
    OmCh2 = OmC * h100 * h100;
    return SUCCESS;
}

int read_param_file(int rank, const char *filename) {
    
    if (rank == 0) {

    char DSAEtaModelInitialName[MAX_LINE_LENGTH];
    char DSAEtaModelReaccName[MAX_LINE_LENGTH];
    char deprecated_window_mode_spec[MAX_LINE_LENGTH];
    int legacy_dpp_param_seen = 0;
    int legacy_seed_cr_species_param_seen = 0;
   
    
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open parameter file");
        //MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return -1;
    }
    rewind(file);
    char line[MAX_LINE_LENGTH];
    //char *dummy = new char[MAX_LINE_LENGTH];

    int ret;
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL){
        ret = 0;
        ret += sscanf(line, "output_dir = %s", output_dir);
        ret += sscanf(line, "strmodel = %s", output_dir);
        ret += sscanf(line, "tracer_file_dir = %s", tracer_file_dir);
        ret += sscanf(line, "tracer_file_extension = %s", tracer_file_extension);
        ret += sscanf(line, "tracer_filename_base1 = %s", tracer_filename_base1);
        ret += sscanf(line, "tracer_filename_base2 = %s", tracer_filename_base2);
        ret += sscanf(line, "backend = %s", tracer_backend_spec);
        ret += sscanf(line, "input_mode = %s", tracer_input_mode_spec);
        ret += sscanf(line, "synch_output_spec = %s", tracer_synch_output_spec);
        ret += sscanf(line, "file_output_mode = %s", tracer_file_output_spec);
        ret += sscanf(line, "output_mode = %s", tracer_file_output_spec);
        ret += sscanf(line, "write_buffer_mode = %s", tracer_write_buffer_mode_spec);
        ret += sscanf(line, "write_buffer_chunk_snapshots = %d", &tracer_write_buffer_chunk_snapshots);
        ret += sscanf(line, "write_crp_output = %d", &tracer_write_crp_output);
        ret += sscanf(line, "write_proton_output = %d", &tracer_write_crp_output);
        ret += sscanf(line, "write_ic_output = %d", &tracer_write_ic_output);
        ret += sscanf(line, "write_gamma_output = %d", &tracer_write_gamma_output);
        ret += sscanf(line, "write_neutrino_output = %d", &tracer_write_neutrino_output);
        ret += sscanf(line, "output_nsnp_min = %d", &tracer_output_nsnp_min);
        ret += sscanf(line, "output_nsnp_max = %d", &tracer_output_nsnp_max);
        ret += sscanf(line, "tracer_debug_max_snapshots = %d", &tracer_debug_max_snapshots);
        ret += sscanf(line, "tracer_bucketstats_top_frac = %lf", &tracer_bucketstats_top_frac);
        ret += sscanf(line, "output_per_cc = %d", &tracer_output_per_cc);
        ret += sscanf(line, "frozen_background = %d", &frozen_background);
        ret += sscanf(line, "checkpoint_interval = %d", &tracer_checkpoint_interval);
        ret += sscanf(line, "checkpoint_dir = %s", tracer_checkpoint_dir);
        ret += sscanf(line, "restart_dir = %s", tracer_restart_dir);
        ret += sscanf(line, "window_mode = %s", deprecated_window_mode_spec);
        ret += sscanf(line, "integration_mode = %s", tracer_integration_mode_spec);
        ret += sscanf(line, "hetero_heavy_id_file = %s", hetero_heavy_id_file);
        ret += sscanf(line, "Dpp_mode = %s", dpp_mode_spec);
        ret += sscanf(line, "dpp_mode = %s", dpp_mode_spec);
        ret += sscanf(line, "fermi2_mode = %s", dpp_mode_spec);
        ret += sscanf(line, "accel_mode = %s", dpp_mode_spec);
        ret += sscanf(line, "ttd_tacc_model = %s", ttd_tacc_model_spec);
        ret += sscanf(line, "TTD_tacc_model = %s", ttd_tacc_model_spec);
        ret += sscanf(line, "seed_cr_species = %s", seed_cr_species_spec);
        ret += sscanf(line, "cr_species = %s", seed_cr_species_spec);
        ret += sscanf(line, "CR_species = %s", seed_cr_species_spec);
        ret += sscanf(line, "seed_cr_species_mode = %s", seed_cr_species_spec);
        ret += sscanf(line, "cr_species_mode = %s", seed_cr_species_spec);
        ret += sscanf(line, "bfield_mode = %s", bfield_mode_spec);
        ret += sscanf(line, "Bfield_mode = %s", bfield_mode_spec);
        ret += sscanf(line, "magnetic_field_mode = %s", bfield_mode_spec);
        
        ret += sscanf(line, "N_TRACERS = %ld", &N_TRACERS);
        ret += sscanf(line, "synch_nfreq = %d", &synch_nfreq);
        ret += sscanf(line, "nfreq = %d", &synch_nfreq);
        ret += sscanf(line, "N_freq = %d", &synch_nfreq);
        ret += sscanf(line, "variable_tracer_mass = %d", &variable_tracer_mass);
        ret += sscanf(line, "M_trc_fix = %lf", &M_trc_fix);
        
        ret += sscanf(line, "ParallelHDFIO = %d", &ParallelHDFIO);
        {
            int parsed = 0;
            parsed = sscanf(line, "Use_MPI = %d", &Use_MPI);
            ret += parsed;
            parsed = sscanf(line, "MPI = %d", &Use_MPI);
            ret += parsed;
            parsed = sscanf(line, "openmp_enabled = %d", &openmp_enabled);
            ret += parsed;
            if (parsed == 1) Use_OMP = openmp_enabled;
            parsed = sscanf(line, "Use_OMP = %d", &Use_OMP);
            ret += parsed;
            if (parsed == 1) openmp_enabled = Use_OMP;
            parsed = sscanf(line, "OMP = %d", &Use_OMP);
            ret += parsed;
            if (parsed == 1) openmp_enabled = Use_OMP;
            parsed = sscanf(line, "openmp_threads = %d", &openmp_threads);
            ret += parsed;
            if (parsed == 1) NUM_THREADS = openmp_threads;
            parsed = sscanf(line, "NUM_THREADS = %d", &NUM_THREADS);
            ret += parsed;
            if (parsed == 1) openmp_threads = NUM_THREADS;
        }

        {
            int parsed = 0;
            parsed = sscanf(line, "job_count = %d", &job_count);
            ret += parsed;
            if (parsed == 1) NUM_RUN = job_count;
            parsed = sscanf(line, "NUM_RUN = %d", &NUM_RUN);
            ret += parsed;
            if (parsed == 1) job_count = NUM_RUN;
            parsed = sscanf(line, "job_index = %d", &job_index);
            ret += parsed;
            if (parsed == 1) RUN = job_index;
            parsed = sscanf(line, "RUN = %d", &RUN);
            ret += parsed;
            if (parsed == 1) job_index = RUN;
            ret += sscanf(line, "job_parallel_enabled = %d", &job_parallel_enabled);
        }

        ret += sscanf(line, "z_ini = %lf", &z_ini);
        ret += sscanf(line, "H0 = %lf", &H0);
        ret += sscanf(line, "ns = %lf", &ns);
        ret += sscanf(line, "sig8 = %lf", &sig8);
        ret += sscanf(line, "deltac = %lf", &deltac);
        ret += sscanf(line, "OmB = %lf", &OmB);
        ret += sscanf(line, "OmC = %lf", &OmC);
        ret += sscanf(line, "OmL = %lf", &OmL);
        ret += sscanf(line, "TCMB = %lf", &TCMB);

        {
            int parsed = 0;
            parsed = sscanf(line, "ASA = %d", &ASA);
            ret += parsed;
            legacy_dpp_param_seen |= (parsed == 1);
            parsed = sscanf(line, "TTD = %d", &TTD);
            ret += parsed;
            legacy_dpp_param_seen |= (parsed == 1);
            parsed = sscanf(line, "direct_tacc_enabled = %d", &direct_tacc_enabled);
            ret += parsed;
            legacy_dpp_param_seen |= (parsed == 1);
        }
        ret += sscanf(line, "variable_turb_scale = %d", &variable_turb_scale);
        ret += sscanf(line, "L_turb = %lf", &L_turb);
        ret += sscanf(line, "L_turb_target_kpc = %lf", &L_turb_target_kpc);

        {
            int parsed = 0;
            parsed = sscanf(line, "Secondary = %d", &Secondary);
            ret += parsed;
            legacy_seed_cr_species_param_seen |= (parsed == 1);
            parsed = sscanf(line, "Mix = %d", &Mix);
            ret += parsed;
            legacy_seed_cr_species_param_seen |= (parsed == 1);
        }
        
        ret += sscanf(line, "InjectionModel = %d", &InjectionModel);
        ret += sscanf(line, "steady_primary_electron_injection = %d", &steady_primary_electron_injection);

        ret += sscanf(line, "psi = %lf", &psi);
        ret += sscanf(line, "mach_limit = %lf", &mach_limit);
        ret += sscanf(line, "eta_dpp_cap = %lf", &eta_dpp_cap);
        ret += sscanf(line, "eta_Dpp_cap = %lf", &eta_dpp_cap);
        ret += sscanf(line, "eta_cap = %lf", &eta_dpp_cap);
        ret += sscanf(line, "t_acc_direct_gyr = %lf", &t_acc_direct_gyr);
        ret += sscanf(line, "t_off_gyr = %lf", &t_off_gyr);
        ret += sscanf(line, "t_acc_off_gyr = %lf", &t_acc_off_gyr);
        ret += sscanf(line, "tracer_nsub_safety = %lf", &tracer_nsub_safety);
        ret += sscanf(line, "tracer_nsub_max = %d", &tracer_nsub_max);
        ret += sscanf(line, "coeff_interp_min_steps = %d", &coeff_interp_min_steps);
        ret += sscanf(line, "coeff_interp_max_segments = %d", &coeff_interp_max_segments);
        ret += sscanf(line, "hetero_mode = %d", &hetero_mode);
        ret += sscanf(line, "hetero_gpu_ranks = %d", &hetero_gpu_ranks);
        ret += sscanf(line, "hetero_skip_heavy = %d", &hetero_skip_heavy);
        ret += sscanf(line, "load_balancing = %d", &load_balancing);
        ret += sscanf(line, "load_balance_top_frac = %lf", &load_balance_top_frac);
        ret += sscanf(line, "load_balance_chunk_size = %d", &load_balance_chunk_size);
        ret += sscanf(line, "load_balance_diagnostics = %d", &load_balance_diagnostics);
        ret += sscanf(line, "load_estimate_virtual_ranks = %d", &load_estimate_virtual_ranks);
        ret += sscanf(line, "one_rank_one_gpu = %d", &one_rank_one_gpu);
        ret += sscanf(line, "reacc_window_mode = %d", &reacc_window_mode);

        ret += sscanf(line, "pmin = %lf", &pmin);
        ret += sscanf(line, "pmax = %lf", &pmax);
        ret += sscanf(line, "pemin = %lf", &pemin);
        ret += sscanf(line, "pemax = %lf", &pemax);

        ret += sscanf(line, "nsnp_i = %d", &nsnp_i);
        ret += sscanf(line, "nsnp_f = %d", &nsnp_f);
        ret += sscanf(line, "nsnp_start = %d", &nsnp_start);

        ret += sscanf(line, "nsnp_output_i = %d", &nsnp_out_i);
        ret += sscanf(line, "nsnp_output_f = %d", &nsnp_out_f);

  
        ret += sscanf(line, "delta_CR_inj = %lf", &delta_CR_inj);
        ret += sscanf(line, "pinjmin = %lf", &pinjmin);
        ret += sscanf(line, "pinjmax = %lf", &pinjmax);
        ret += sscanf(line, "peinjmin = %lf", &peinjmin);
        ret += sscanf(line, "peinjmax = %lf", &peinjmax);
        ret += sscanf(line, "f_ep = %lf", &f_ep);
        ret += sscanf(line, "phi_CRe = %lf", &phi_CRe);
        ret += sscanf(line, "phi_CRp = %lf", &phi_CRp);
        ret += sscanf(line, "phi_CR_mix  = %lf", &phi_CR_mix );
        ret += sscanf(line, "initial_cr_norm_mode = %s", initial_cr_norm_mode_spec);
        ret += sscanf(line, "initial_cr_energy_anchor = %s", initial_cr_energy_anchor_spec);
        ret += sscanf(line, "initial_cr_energy_ratio_p_to_e = %lf",
                      &initial_cr_energy_ratio_p_to_e);

        ret += sscanf(line, "L_CR_p  = %lf", &L_CR_p );
        ret += sscanf(line, "L_CR_e  = %lf", &L_CR_e);

        ret += sscanf(line, "eta_B = %lf", &eta_B);
        

        ret += sscanf(line, "nu_min_s = %lf", &nu_min_s);
        ret += sscanf(line, "nu_max_s = %lf", &nu_max_s);
        ret += sscanf(line, "synch_logb_min = %lf", &tracer_synch_logb_min);
        ret += sscanf(line, "synch_logb_max = %lf", &tracer_synch_logb_max);
        ret += sscanf(line, "adaptive_synch_logb = %d", &adaptive_synch_logb);
        ret += sscanf(line, "synch_N_theta = %d", &tracer_synch_ntheta_pitch);
        ret += sscanf(line, "nu_min_ic = %lf", &nu_min_ic);
        ret += sscanf(line, "nu_max_ic = %lf", &nu_max_ic);
        ret += sscanf(line, "E_gamma_min = %lf", &E_gamma_min);
        ret += sscanf(line, "E_gamma_max = %lf", &E_gamma_max);
        ret += sscanf(line, "E_nu_min = %lf", &E_nu_min);
        ret += sscanf(line, "E_nu_max = %lf", &E_nu_max);
        ret += sscanf(line, "E_gamma_min_pg = %lf", &E_gamma_min_pg);
        ret += sscanf(line, "E_gamma_max_pg = %lf", &E_gamma_max_pg);
        ret += sscanf(line, "E_nu_min_pg = %lf", &E_nu_min_pg);
        ret += sscanf(line, "E_nu_max_pg = %lf", &E_nu_max_pg);
        ret += sscanf(line, "E_brem_min = %lf", &E_brem_min);
        ret += sscanf(line, "E_brem_max = %lf", &E_brem_max);

        if (sscanf(line, "DSAEtaModelInitial = %s", DSAEtaModelInitialName) == 1) {
            ret += 1;
            DSAEtaModelInitial = dsa_model_id_from_name(DSAEtaModelInitialName);
        }
        if (sscanf(line, "DSAEtaModelReacc = %s", DSAEtaModelReaccName) == 1) {
            ret += 1;
            DSAEtaModelReacc = dsa_model_id_from_name(DSAEtaModelReaccName);
        }
        ret += sscanf(line, "DSAInjectionMode = %s", DSAInjectionModeSpec);
        ret += sscanf(line, "DSAReaccMode = %s", DSAReaccModeSpec);
        ret += sscanf(line, "DSAInjectSpecies = %s", DSAInjectSpeciesSpec);
        ret += sscanf(line, "DSASpecies = %s", DSAInjectSpeciesSpec);
        ret += sscanf(line, "DSAChiP = %lf", &DSAChiP);
        ret += sscanf(line, "DSAChiE = %lf", &DSAChiE);
        ret += sscanf(line, "DSAKep = %lf", &DSAKep);
        ret += sscanf(line, "DSAPmaxPmc = %lf", &DSAPmaxPmc);
        ret += sscanf(line, "DSAPmaxEmc = %lf", &DSAPmaxEmc);
        ret += sscanf(line, "DSAMinMach = %lf", &DSAMinMach);
        ret += sscanf(line, "DSAGammaGas = %lf", &DSAGammaGas);
        ret += sscanf(line, "DSAXcrPminPmc = %lf", &DSAXcrPminPmc);
        ret += sscanf(line, "DSAReaccEtaCap = %lf", &DSAReaccEtaCap);
        ret += sscanf(line, "DSAReaccCap = %lf", &DSAReaccEtaCap);
        ret += sscanf(line, "DSAShockRequired = %d", &DSAShockRequired);
        ret += sscanf(line, "DSADebug = %d", &DSADebug);

        /* --- Grid mode --- */
        ret += sscanf(line, "grid_mode = %d", &grid_mode);
        ret += sscanf(line, "decomp_layout = %d", &decomp_layout);
        ret += sscanf(line, "Nx = %d", &Nx);
        ret += sscanf(line, "Ny = %d", &Ny);
        ret += sscanf(line, "Nz = %d", &Nz);
        ret += sscanf(line, "grid_file_dir = %s",      grid_file_dir);
        ret += sscanf(line, "grid_filename_base = %s", grid_filename_base);
        ret += sscanf(line, "aux_file_dir = %s",       aux_file_dir);
        ret += sscanf(line, "aux_filename_base = %s",  aux_filename_base);
        ret += sscanf(line, "grid_snap_pad = %d",      &grid_snap_pad);

        ret += sscanf(line, "interp_snap = %d",  &interp_snap);
        ret += sscanf(line, "t_fp_total = %lf",  &t_fp_total);
        ret += sscanf(line, "dt_fp = %lf",        &dt_fp);
        ret += sscanf(line, "n_fp_out = %d",      &n_fp_out);

        ret += sscanf(line, "diffuse_mode = %d",   &diffuse_mode);
        ret += sscanf(line, "diffuse_halo = %d",   &diffuse_halo);
        ret += sscanf(line, "kappa_diff_0 = %lf",  &kappa_diff_0);
        ret += sscanf(line, "delta_diff = %lf",    &delta_diff);
        ret += sscanf(line, "R0_diff = %lf",       &R0_diff);
        ret += sscanf(line, "kappa_perp_ratio = %lf", &kappa_perp_ratio);
        ret += sscanf(line, "advect_mode = %d",    &advect_mode);

        if(ret == 0 && strstr(line, "=") != NULL && line[0] != '#')
            printf("Unused parameter? %s\n", line);
    }


        fclose(file);

        if (finalize_dpp_mode_config(legacy_dpp_param_seen) != SUCCESS) {
            return FAIL;
        }
        if (finalize_ttd_tacc_model_config() != SUCCESS) {
            return FAIL;
        }
        if (finalize_seed_cr_species_config(legacy_seed_cr_species_param_seen) != SUCCESS) {
            return FAIL;
        }
        if (normalize_initial_cr_norm_config() != SUCCESS) {
            return FAIL;
        }
        if (normalize_bfield_mode_config() != SUCCESS) {
            return FAIL;
        }
        if (normalize_dsa_inject_species_config() != SUCCESS) {
            return FAIL;
        }

        if (DSAEtaModelInitial == DSA_MODEL_COUNT) {
            fprintf(stderr, "Unknown DSAEtaModelInitial\n");
            return FAIL;
        }

        if (DSAEtaModelReacc == DSA_MODEL_COUNT) {
            fprintf(stderr, "Unknown DSAEtaModelReacc\n");
            return FAIL;
        }
        if (kappa_perp_ratio < 0.0) {
            fprintf(stderr, "kappa_perp_ratio must be >= 0\n");
            return FAIL;
        }
        if (synch_nfreq <= 0 || synch_nfreq > MAX_SYNCH_FREQ_BINS) {
            fprintf(stderr,
                    "synch_nfreq/nfreq must be in [1, %d]\n",
                    MAX_SYNCH_FREQ_BINS);
            return FAIL;
        }
        if (!(tracer_synch_logb_max > tracer_synch_logb_min)) {
            fprintf(stderr, "tracer_synch_logb_max must be > tracer_synch_logb_min\n");
            return FAIL;
        }
        if (adaptive_synch_logb != 0) adaptive_synch_logb = 1;
        if (tracer_synch_ntheta_pitch <= 0 || tracer_synch_ntheta_pitch > N_theta_pitch) {
            fprintf(stderr, "synch_N_theta must be in [1, %d]\n", N_theta_pitch);
            return FAIL;
        }
        if (tracer_output_per_cc != 0) tracer_output_per_cc = 1;
        if (reacc_window_mode != REACC_WINDOW_ALWAYS_ON &&
            reacc_window_mode != REACC_WINDOW_CENTERED) {
            fprintf(stderr, "reacc_window_mode must be 0(always-on) or 1(centered)\n");
            return FAIL;
        }
        if (!(tracer_nsub_safety > 0.0)) {
            fprintf(stderr, "tracer_nsub_safety must be > 0\n");
            return FAIL;
        }
        if (tracer_nsub_max < 0) {
            fprintf(stderr, "tracer_nsub_max must be >= 0\n");
            return FAIL;
        }
        if (coeff_interp_min_steps <= 0) {
            fprintf(stderr, "coeff_interp_min_steps must be > 0\n");
            return FAIL;
        }
        if (coeff_interp_max_segments <= 0) {
            fprintf(stderr, "coeff_interp_max_segments must be > 0\n");
            return FAIL;
        }
        if (hetero_mode < 0) {
            fprintf(stderr, "hetero_mode must be >= 0\n");
            return FAIL;
        }
        if (hetero_gpu_ranks < 0) {
            fprintf(stderr, "hetero_gpu_ranks must be >= 0\n");
            return FAIL;
        }
        if (hetero_skip_heavy < 0) {
            fprintf(stderr, "hetero_skip_heavy must be >= 0\n");
            return FAIL;
        }
        if (load_balancing < 0) {
            fprintf(stderr, "load_balancing must be >= 0\n");
            return FAIL;
        }
        if (!(load_balance_top_frac > 0.0 && load_balance_top_frac <= 1.0)) {
            fprintf(stderr, "load_balance_top_frac must be in (0, 1]\n");
            return FAIL;
        }
        if (load_balance_chunk_size < 0) {
            fprintf(stderr, "load_balance_chunk_size must be >= 0\n");
            return FAIL;
        }
        if (load_balance_diagnostics < 0) {
            fprintf(stderr, "load_balance_diagnostics must be >= 0\n");
            return FAIL;
        }
        if (load_estimate_virtual_ranks <= 0) {
            fprintf(stderr, "load_estimate_virtual_ranks must be > 0\n");
            return FAIL;
        }
        if (tracer_debug_max_snapshots < 0) {
            fprintf(stderr, "tracer_debug_max_snapshots must be >= 0\n");
            return FAIL;
        }
        if (!(tracer_bucketstats_top_frac > 0.0 && tracer_bucketstats_top_frac <= 1.0)) {
            fprintf(stderr, "tracer_bucketstats_top_frac must be in (0, 1]\n");
            return FAIL;
        }
        if (one_rank_one_gpu < 0) {
            fprintf(stderr, "one_rank_one_gpu must be >= 0\n");
            return FAIL;
        }
        finalize_parallel_param_aliases(1);
        
    
    }

    {
    MPI_Bcast(&output_dir, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    // printf("dir = %s\n", output_dir);
    MPI_Bcast(&tracer_file_dir, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_file_extension, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_filename_base1, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_filename_base2, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_backend_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_input_mode_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_synch_output_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_file_output_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_write_buffer_mode_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_write_buffer_chunk_snapshots, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_write_crp_output, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_write_ic_output, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_write_gamma_output, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_write_neutrino_output, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_output_nsnp_min, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_output_nsnp_max, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_debug_max_snapshots, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_bucketstats_top_frac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_output_per_cc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&frozen_background, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_checkpoint_interval, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_checkpoint_dir, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_restart_dir, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_integration_mode_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&hetero_heavy_id_file, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dpp_mode_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ttd_tacc_model_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed_cr_species_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&initial_cr_norm_mode_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&initial_cr_energy_anchor_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bfield_mode_spec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAInjectionModeSpec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAReaccModeSpec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAInjectSpeciesSpec, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&N_TRACERS, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&synch_nfreq, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Use_MPI, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ParallelHDFIO, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Use_OMP, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_THREADS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&openmp_enabled, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&openmp_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // printf("NUM_THREADS = %d\n", NUM_THREADS);
    MPI_Bcast(&NUM_RUN, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&RUN, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&job_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&job_index, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&job_parallel_enabled, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ns, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sig8, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&deltac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&OmB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&OmC, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&OmL, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&TCMB, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&InjectionModel, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&steady_primary_electron_injection, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed_cr_species, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&initial_cr_norm_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&initial_cr_energy_anchor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L_turb, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L_turb_target_kpc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    MPI_Bcast(&nsnp_f, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nsnp_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nsnp_start, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&pmin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pemin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pemax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(&delta_CR_inj, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&f_ep, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pinjmin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pinjmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&peinjmin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&peinjmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&phi_CRe, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&phi_CRp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&phi_CR_mix, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&initial_cr_energy_ratio_p_to_e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // printf("phi_CR = %le\n", phi_CR);

    MPI_Bcast(&L_CR_e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L_CR_p, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&eta_B, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&psi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mach_limit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ttd_tacc_model, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eta_dpp_cap, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t_acc_direct_gyr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t_off_gyr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t_acc_off_gyr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bfield_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_nsub_safety, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_nsub_max, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coeff_interp_min_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coeff_interp_max_segments, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&hetero_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&hetero_gpu_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&hetero_skip_heavy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&load_balancing, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&load_balance_top_frac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&load_balance_chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&load_balance_diagnostics, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&load_estimate_virtual_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&one_rank_one_gpu, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reacc_window_mode, 1, MPI_INT, 0, MPI_COMM_WORLD);


    MPI_Bcast(&nu_min_s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nu_max_s, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_synch_logb_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_synch_logb_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&adaptive_synch_logb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tracer_synch_ntheta_pitch, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nu_min_ic, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nu_max_ic, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_gamma_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_gamma_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_nu_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_nu_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_gamma_min_pg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_gamma_max_pg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_nu_min_pg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_nu_max_pg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_brem_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E_brem_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAEtaModelInitial, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAEtaModelReacc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAChiP, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAChiE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAKep, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAPmaxPmc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAPmaxEmc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAMinMach, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAGammaGas, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAXcrPminPmc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAReaccEtaCap, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAShockRequired, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSADebug, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DSAInjectSpecies, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Grid mode */
    MPI_Bcast(&grid_mode,     1,               MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&decomp_layout, 1,               MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&Nx,            1,               MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&Ny,            1,               MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&Nz,            1,               MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&grid_snap_pad, 1,               MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(grid_file_dir,      MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(grid_filename_base, MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(aux_file_dir,       MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(aux_filename_base,  MAX_LINE_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Bcast(&interp_snap,  1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&t_fp_total,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt_fp,        1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_fp_out,     1, MPI_INT,    0, MPI_COMM_WORLD);

    MPI_Bcast(&diffuse_mode, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&diffuse_halo, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&kappa_diff_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&delta_diff,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&R0_diff,      1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kappa_perp_ratio, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&advect_mode,  1, MPI_INT,    0, MPI_COMM_WORLD);

    }

    if (finalize_cosmology_config(rank == 0) != SUCCESS) {
        return FAIL;
    }

    return 0;
}



int read_param_file_noMPI(const char *filename) {
    const int log_root = readfile_should_log_root_only();
   
    char DSAEtaModelInitialName[MAX_LINE_LENGTH];
    char DSAEtaModelReaccName[MAX_LINE_LENGTH];
    char deprecated_window_mode_spec[MAX_LINE_LENGTH];
    int legacy_dpp_param_seen = 0;
    int legacy_seed_cr_species_param_seen = 0;
   
    
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open parameter file");
        //MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return -1;
    }
    rewind(file);
    char line[MAX_LINE_LENGTH];
    //char *dummy = new char[MAX_LINE_LENGTH];

    int ret;
    while (fgets(line, MAX_LINE_LENGTH, file) != NULL){
        ret = 0;
        ret += sscanf(line, "output_dir = %s", output_dir);
        ret += sscanf(line, "strmodel = %s", output_dir);
        ret += sscanf(line, "tracer_file_dir = %s", tracer_file_dir);
        ret += sscanf(line, "tracer_file_extension = %s", tracer_file_extension);
        ret += sscanf(line, "tracer_filename_base1 = %s", tracer_filename_base1);
        ret += sscanf(line, "tracer_filename_base2 = %s", tracer_filename_base2);
        ret += sscanf(line, "backend = %s", tracer_backend_spec);
        ret += sscanf(line, "input_mode = %s", tracer_input_mode_spec);
        ret += sscanf(line, "synch_output_spec = %s", tracer_synch_output_spec);
        ret += sscanf(line, "file_output_mode = %s", tracer_file_output_spec);
        ret += sscanf(line, "output_mode = %s", tracer_file_output_spec);
        ret += sscanf(line, "write_buffer_mode = %s", tracer_write_buffer_mode_spec);
        ret += sscanf(line, "write_buffer_chunk_snapshots = %d", &tracer_write_buffer_chunk_snapshots);
        ret += sscanf(line, "write_crp_output = %d", &tracer_write_crp_output);
        ret += sscanf(line, "write_proton_output = %d", &tracer_write_crp_output);
        ret += sscanf(line, "write_ic_output = %d", &tracer_write_ic_output);
        ret += sscanf(line, "write_gamma_output = %d", &tracer_write_gamma_output);
        ret += sscanf(line, "write_neutrino_output = %d", &tracer_write_neutrino_output);
        ret += sscanf(line, "output_nsnp_min = %d", &tracer_output_nsnp_min);
        ret += sscanf(line, "output_nsnp_max = %d", &tracer_output_nsnp_max);
        ret += sscanf(line, "tracer_debug_max_snapshots = %d", &tracer_debug_max_snapshots);
        ret += sscanf(line, "tracer_bucketstats_top_frac = %lf", &tracer_bucketstats_top_frac);
        ret += sscanf(line, "output_per_cc = %d", &tracer_output_per_cc);
        ret += sscanf(line, "frozen_background = %d", &frozen_background);
        ret += sscanf(line, "checkpoint_interval = %d", &tracer_checkpoint_interval);
        ret += sscanf(line, "checkpoint_dir = %s", tracer_checkpoint_dir);
        ret += sscanf(line, "restart_dir = %s", tracer_restart_dir);
        ret += sscanf(line, "window_mode = %s", deprecated_window_mode_spec);
        ret += sscanf(line, "integration_mode = %s", tracer_integration_mode_spec);
        ret += sscanf(line, "hetero_heavy_id_file = %s", hetero_heavy_id_file);
        ret += sscanf(line, "Dpp_mode = %s", dpp_mode_spec);
        ret += sscanf(line, "dpp_mode = %s", dpp_mode_spec);
        ret += sscanf(line, "fermi2_mode = %s", dpp_mode_spec);
        ret += sscanf(line, "accel_mode = %s", dpp_mode_spec);
        ret += sscanf(line, "ttd_tacc_model = %s", ttd_tacc_model_spec);
        ret += sscanf(line, "TTD_tacc_model = %s", ttd_tacc_model_spec);
        ret += sscanf(line, "seed_cr_species = %s", seed_cr_species_spec);
        ret += sscanf(line, "cr_species = %s", seed_cr_species_spec);
        ret += sscanf(line, "CR_species = %s", seed_cr_species_spec);
        ret += sscanf(line, "seed_cr_species_mode = %s", seed_cr_species_spec);
        ret += sscanf(line, "cr_species_mode = %s", seed_cr_species_spec);
        ret += sscanf(line, "bfield_mode = %s", bfield_mode_spec);
        ret += sscanf(line, "Bfield_mode = %s", bfield_mode_spec);
        ret += sscanf(line, "magnetic_field_mode = %s", bfield_mode_spec);
        
        ret += sscanf(line, "N_TRACERS = %ld", &N_TRACERS);
        ret += sscanf(line, "synch_nfreq = %d", &synch_nfreq);
        ret += sscanf(line, "nfreq = %d", &synch_nfreq);
        ret += sscanf(line, "N_freq = %d", &synch_nfreq);
        ret += sscanf(line, "variable_tracer_mass = %d", &variable_tracer_mass);
        ret += sscanf(line, "M_trc_fix = %lf", &M_trc_fix);
       
        
        ret += sscanf(line, "ParallelHDFIO = %d", &ParallelHDFIO);
        {
            int parsed = 0;
            parsed = sscanf(line, "Use_MPI = %d", &Use_MPI);
            ret += parsed;
            parsed = sscanf(line, "MPI = %d", &Use_MPI);
            ret += parsed;
            parsed = sscanf(line, "openmp_enabled = %d", &openmp_enabled);
            ret += parsed;
            if (parsed == 1) Use_OMP = openmp_enabled;
            parsed = sscanf(line, "Use_OMP = %d", &Use_OMP);
            ret += parsed;
            if (parsed == 1) openmp_enabled = Use_OMP;
            parsed = sscanf(line, "OMP = %d", &Use_OMP);
            ret += parsed;
            if (parsed == 1) openmp_enabled = Use_OMP;
            parsed = sscanf(line, "openmp_threads = %d", &openmp_threads);
            ret += parsed;
            if (parsed == 1) NUM_THREADS = openmp_threads;
            parsed = sscanf(line, "NUM_THREADS = %d", &NUM_THREADS);
            ret += parsed;
            if (parsed == 1) openmp_threads = NUM_THREADS;
        }

        {
            int parsed = 0;
            parsed = sscanf(line, "job_count = %d", &job_count);
            ret += parsed;
            if (parsed == 1) NUM_RUN = job_count;
            parsed = sscanf(line, "NUM_RUN = %d", &NUM_RUN);
            ret += parsed;
            if (parsed == 1) job_count = NUM_RUN;
            parsed = sscanf(line, "job_index = %d", &job_index);
            ret += parsed;
            if (parsed == 1) RUN = job_index;
            parsed = sscanf(line, "RUN = %d", &RUN);
            ret += parsed;
            if (parsed == 1) job_index = RUN;
            ret += sscanf(line, "job_parallel_enabled = %d", &job_parallel_enabled);
        }
        ret += sscanf(line, "H0 = %lf", &H0);
        ret += sscanf(line, "ns = %lf", &ns);
        ret += sscanf(line, "sig8 = %lf", &sig8);
        ret += sscanf(line, "deltac = %lf", &deltac);
        ret += sscanf(line, "OmB = %lf", &OmB);
        ret += sscanf(line, "OmC = %lf", &OmC);
        ret += sscanf(line, "OmL = %lf", &OmL);
        ret += sscanf(line, "TCMB = %lf", &TCMB);

        {
            int parsed = 0;
            parsed = sscanf(line, "ASA = %d", &ASA);
            ret += parsed;
            legacy_dpp_param_seen |= (parsed == 1);
            parsed = sscanf(line, "TTD = %d", &TTD);
            ret += parsed;
            legacy_dpp_param_seen |= (parsed == 1);
            parsed = sscanf(line, "direct_tacc_enabled = %d", &direct_tacc_enabled);
            ret += parsed;
            legacy_dpp_param_seen |= (parsed == 1);
        }
        ret += sscanf(line, "L_turb = %lf", &L_turb);
        ret += sscanf(line, "L_turb_target_kpc = %lf", &L_turb_target_kpc);

        {
            int parsed = 0;
            parsed = sscanf(line, "Secondary = %d", &Secondary);
            ret += parsed;
            legacy_seed_cr_species_param_seen |= (parsed == 1);
            parsed = sscanf(line, "Mix = %d", &Mix);
            ret += parsed;
            legacy_seed_cr_species_param_seen |= (parsed == 1);
        }

        ret += sscanf(line, "InjectionModel = %d", &InjectionModel);
        ret += sscanf(line, "steady_primary_electron_injection = %d", &steady_primary_electron_injection);


        ret += sscanf(line, "psi = %lf", &psi);
        ret += sscanf(line, "mach_limit = %lf", &mach_limit);
        ret += sscanf(line, "eta_dpp_cap = %lf", &eta_dpp_cap);
        ret += sscanf(line, "eta_Dpp_cap = %lf", &eta_dpp_cap);
        ret += sscanf(line, "eta_cap = %lf", &eta_dpp_cap);
        ret += sscanf(line, "t_acc_direct_gyr = %lf", &t_acc_direct_gyr);
        ret += sscanf(line, "t_off_gyr = %lf", &t_off_gyr);
        ret += sscanf(line, "t_acc_off_gyr = %lf", &t_acc_off_gyr);
        ret += sscanf(line, "tracer_nsub_safety = %lf", &tracer_nsub_safety);
        ret += sscanf(line, "tracer_nsub_max = %d", &tracer_nsub_max);
        ret += sscanf(line, "coeff_interp_min_steps = %d", &coeff_interp_min_steps);
        ret += sscanf(line, "coeff_interp_max_segments = %d", &coeff_interp_max_segments);
        ret += sscanf(line, "hetero_mode = %d", &hetero_mode);
        ret += sscanf(line, "hetero_gpu_ranks = %d", &hetero_gpu_ranks);
        ret += sscanf(line, "hetero_skip_heavy = %d", &hetero_skip_heavy);
        ret += sscanf(line, "load_balancing = %d", &load_balancing);
        ret += sscanf(line, "load_balance_top_frac = %lf", &load_balance_top_frac);
        ret += sscanf(line, "load_balance_chunk_size = %d", &load_balance_chunk_size);
        ret += sscanf(line, "load_balance_diagnostics = %d", &load_balance_diagnostics);
        ret += sscanf(line, "load_estimate_virtual_ranks = %d", &load_estimate_virtual_ranks);
        ret += sscanf(line, "one_rank_one_gpu = %d", &one_rank_one_gpu);
        ret += sscanf(line, "reacc_window_mode = %d", &reacc_window_mode);
        //mach_limit = 1.000000

        ret += sscanf(line, "pmin = %lf", &pmin);
        ret += sscanf(line, "pmax = %lf", &pmax);
        ret += sscanf(line, "pemin = %lf", &pemin);
        ret += sscanf(line, "pemax = %lf", &pemax);

        ret += sscanf(line, "nsnp_i = %d", &nsnp_i);
        ret += sscanf(line, "nsnp_f = %d", &nsnp_f);
        ret += sscanf(line, "nsnp_start = %d", &nsnp_start);

  
        ret += sscanf(line, "delta_CR_inj = %lf", &delta_CR_inj);
        ret += sscanf(line, "pinjmin = %lf", &pinjmin);
        ret += sscanf(line, "pinjmax = %lf", &pinjmax);
        ret += sscanf(line, "peinjmin = %lf", &peinjmin);
        ret += sscanf(line, "peinjmax = %lf", &peinjmax);
        ret += sscanf(line, "f_ep = %lf", &f_ep);
        ret += sscanf(line, "phi_CRe = %lf", &phi_CRe);
        ret += sscanf(line, "phi_CRp = %lf", &phi_CRp);
        ret += sscanf(line, "phi_CR_mix  = %lf", &phi_CR_mix );
        ret += sscanf(line, "initial_cr_norm_mode = %s", initial_cr_norm_mode_spec);
        ret += sscanf(line, "initial_cr_energy_anchor = %s", initial_cr_energy_anchor_spec);
        ret += sscanf(line, "initial_cr_energy_ratio_p_to_e = %lf",
                      &initial_cr_energy_ratio_p_to_e);

        ret += sscanf(line, "L_CR_p  = %lf", &L_CR_p );
        ret += sscanf(line, "L_CR_e  = %lf", &L_CR_e);

        ret += sscanf(line, "eta_B = %lf", &eta_B);
        

        ret += sscanf(line, "nu_min_s = %lf", &nu_min_s);
        ret += sscanf(line, "nu_max_s = %lf", &nu_max_s);
        ret += sscanf(line, "synch_logb_min = %lf", &tracer_synch_logb_min);
        ret += sscanf(line, "synch_logb_max = %lf", &tracer_synch_logb_max);
        ret += sscanf(line, "adaptive_synch_logb = %d", &adaptive_synch_logb);
        ret += sscanf(line, "synch_N_theta = %d", &tracer_synch_ntheta_pitch);
        ret += sscanf(line, "nu_min_ic = %lf", &nu_min_ic);
        ret += sscanf(line, "nu_max_ic = %lf", &nu_max_ic);
        ret += sscanf(line, "E_gamma_min = %lf", &E_gamma_min);
        ret += sscanf(line, "E_gamma_max = %lf", &E_gamma_max);
        ret += sscanf(line, "E_nu_min = %lf", &E_nu_min);
        ret += sscanf(line, "E_nu_max = %lf", &E_nu_max);
        ret += sscanf(line, "E_gamma_min_pg = %lf", &E_gamma_min_pg);
        ret += sscanf(line, "E_gamma_max_pg = %lf", &E_gamma_max_pg);
        ret += sscanf(line, "E_nu_min_pg = %lf", &E_nu_min_pg);
        ret += sscanf(line, "E_nu_max_pg = %lf", &E_nu_max_pg);
        ret += sscanf(line, "E_brem_min = %lf", &E_brem_min);
        ret += sscanf(line, "E_brem_max = %lf", &E_brem_max);


        if (sscanf(line, "DSAEtaModelInitial = %s", DSAEtaModelInitialName) == 1) {
            ret += 1;
            DSAEtaModelInitial = dsa_model_id_from_name(DSAEtaModelInitialName);
        }
        if (sscanf(line, "DSAEtaModelReacc = %s", DSAEtaModelReaccName) == 1) {
            ret += 1;
            DSAEtaModelReacc = dsa_model_id_from_name(DSAEtaModelReaccName);
        }
        ret += sscanf(line, "DSAInjectionMode = %s", DSAInjectionModeSpec);
        ret += sscanf(line, "DSAReaccMode = %s", DSAReaccModeSpec);
        ret += sscanf(line, "DSAInjectSpecies = %s", DSAInjectSpeciesSpec);
        ret += sscanf(line, "DSASpecies = %s", DSAInjectSpeciesSpec);
        ret += sscanf(line, "DSAChiP = %lf", &DSAChiP);
        ret += sscanf(line, "DSAChiE = %lf", &DSAChiE);
        ret += sscanf(line, "DSAKep = %lf", &DSAKep);
        ret += sscanf(line, "DSAPmaxPmc = %lf", &DSAPmaxPmc);
        ret += sscanf(line, "DSAPmaxEmc = %lf", &DSAPmaxEmc);
        ret += sscanf(line, "DSAMinMach = %lf", &DSAMinMach);
        ret += sscanf(line, "DSAGammaGas = %lf", &DSAGammaGas);
        ret += sscanf(line, "DSAXcrPminPmc = %lf", &DSAXcrPminPmc);
        ret += sscanf(line, "DSAReaccEtaCap = %lf", &DSAReaccEtaCap);
        ret += sscanf(line, "DSAReaccCap = %lf", &DSAReaccEtaCap);
        ret += sscanf(line, "DSAShockRequired = %d", &DSAShockRequired);
        ret += sscanf(line, "DSADebug = %d", &DSADebug);

        ret += sscanf(line, "grid_mode = %d", &grid_mode);
        ret += sscanf(line, "decomp_layout = %d", &decomp_layout);
        ret += sscanf(line, "Nx = %d", &Nx);
        ret += sscanf(line, "Ny = %d", &Ny);
        ret += sscanf(line, "Nz = %d", &Nz);
        ret += sscanf(line, "grid_file_dir = %s",      grid_file_dir);
        ret += sscanf(line, "grid_filename_base = %s", grid_filename_base);
        ret += sscanf(line, "aux_file_dir = %s",       aux_file_dir);
        ret += sscanf(line, "aux_filename_base = %s",  aux_filename_base);
        ret += sscanf(line, "grid_snap_pad = %d",      &grid_snap_pad);
        ret += sscanf(line, "interp_snap = %d",        &interp_snap);
        ret += sscanf(line, "t_fp_total = %lf",        &t_fp_total);
        ret += sscanf(line, "dt_fp = %lf",             &dt_fp);
        ret += sscanf(line, "n_fp_out = %d",           &n_fp_out);
        ret += sscanf(line, "diffuse_mode = %d",       &diffuse_mode);
        ret += sscanf(line, "diffuse_halo = %d",       &diffuse_halo);
        ret += sscanf(line, "kappa_diff_0 = %lf",      &kappa_diff_0);
        ret += sscanf(line, "delta_diff = %lf",        &delta_diff);
        ret += sscanf(line, "R0_diff = %lf",           &R0_diff);
        ret += sscanf(line, "kappa_perp_ratio = %lf", &kappa_perp_ratio);
        ret += sscanf(line, "advect_mode = %d",        &advect_mode);


        if(log_root && ret == 0 && strstr(line, "=") != NULL && line[0] != '#')
            printf("Unused parameter? %s\n", line);
    }


    fclose(file);

    if (finalize_dpp_mode_config(legacy_dpp_param_seen) != SUCCESS) {
        return FAIL;
    }
    if (finalize_ttd_tacc_model_config() != SUCCESS) {
        return FAIL;
    }
    if (finalize_seed_cr_species_config(legacy_seed_cr_species_param_seen) != SUCCESS) {
        return FAIL;
    }
    if (normalize_initial_cr_norm_config() != SUCCESS) {
        return FAIL;
    }
    if (normalize_bfield_mode_config() != SUCCESS) {
        return FAIL;
    }
    if (normalize_dsa_inject_species_config() != SUCCESS) {
        return FAIL;
    }


    if (DSAEtaModelInitial == DSA_MODEL_COUNT) {
        if (log_root) fprintf(stderr, "Unknown DSAEtaModelInitial\n");
        return FAIL;
    }

    if (DSAEtaModelReacc == DSA_MODEL_COUNT) {
        if (log_root) fprintf(stderr, "Unknown DSAEtaModelReacc\n");
        return FAIL;
    }
    if (kappa_perp_ratio < 0.0) {
        if (log_root) fprintf(stderr, "kappa_perp_ratio must be >= 0\n");
        return FAIL;
    }
    if (synch_nfreq <= 0 || synch_nfreq > MAX_SYNCH_FREQ_BINS) {
        if (log_root) {
            fprintf(stderr,
                    "synch_nfreq/nfreq must be in [1, %d]\n",
                    MAX_SYNCH_FREQ_BINS);
        }
        return FAIL;
    }
    if (!(tracer_synch_logb_max > tracer_synch_logb_min)) {
        if (log_root) fprintf(stderr, "tracer_synch_logb_max must be > tracer_synch_logb_min\n");
        return FAIL;
    }
    if (adaptive_synch_logb != 0) adaptive_synch_logb = 1;
    if (tracer_synch_ntheta_pitch <= 0 || tracer_synch_ntheta_pitch > N_theta_pitch) {
        if (log_root) fprintf(stderr, "synch_N_theta must be in [1, %d]\n", N_theta_pitch);
        return FAIL;
    }
    if (tracer_output_per_cc != 0) tracer_output_per_cc = 1;
    if (reacc_window_mode != REACC_WINDOW_ALWAYS_ON &&
        reacc_window_mode != REACC_WINDOW_CENTERED) {
        if (log_root) fprintf(stderr, "reacc_window_mode must be 0(always-on) or 1(centered)\n");
        return FAIL;
    }
    if (!(tracer_nsub_safety > 0.0)) {
        if (log_root) fprintf(stderr, "tracer_nsub_safety must be > 0\n");
        return FAIL;
    }
    if (tracer_nsub_max < 0) {
        if (log_root) fprintf(stderr, "tracer_nsub_max must be >= 0\n");
        return FAIL;
    }
    if (coeff_interp_min_steps <= 0) {
        if (log_root) fprintf(stderr, "coeff_interp_min_steps must be > 0\n");
        return FAIL;
    }
    if (coeff_interp_max_segments <= 0) {
        if (log_root) fprintf(stderr, "coeff_interp_max_segments must be > 0\n");
        return FAIL;
    }
    if (hetero_mode < 0) {
        if (log_root) fprintf(stderr, "hetero_mode must be >= 0\n");
        return FAIL;
    }
    if (hetero_gpu_ranks < 0) {
        if (log_root) fprintf(stderr, "hetero_gpu_ranks must be >= 0\n");
        return FAIL;
    }
    if (hetero_skip_heavy < 0) {
        if (log_root) fprintf(stderr, "hetero_skip_heavy must be >= 0\n");
        return FAIL;
    }
    if (load_balancing < 0) {
        if (log_root) fprintf(stderr, "load_balancing must be >= 0\n");
        return FAIL;
    }
    if (!(load_balance_top_frac > 0.0 && load_balance_top_frac <= 1.0)) {
        if (log_root) fprintf(stderr, "load_balance_top_frac must be in (0, 1]\n");
        return FAIL;
    }
    if (load_balance_chunk_size < 0) {
        if (log_root) fprintf(stderr, "load_balance_chunk_size must be >= 0\n");
        return FAIL;
    }
    if (load_balance_diagnostics < 0) {
        if (log_root) fprintf(stderr, "load_balance_diagnostics must be >= 0\n");
        return FAIL;
    }
    if (load_estimate_virtual_ranks <= 0) {
        if (log_root) fprintf(stderr, "load_estimate_virtual_ranks must be > 0\n");
        return FAIL;
    }
    if (tracer_debug_max_snapshots < 0) {
        if (log_root) fprintf(stderr, "tracer_debug_max_snapshots must be >= 0\n");
        return FAIL;
    }
    if (!(tracer_bucketstats_top_frac > 0.0 && tracer_bucketstats_top_frac <= 1.0)) {
        if (log_root) fprintf(stderr, "tracer_bucketstats_top_frac must be in (0, 1]\n");
        return FAIL;
    }
    if (one_rank_one_gpu < 0) {
        if (log_root) fprintf(stderr, "one_rank_one_gpu must be >= 0\n");
        return FAIL;
    }
    finalize_parallel_param_aliases(0);
    

    if (finalize_cosmology_config(log_root) != SUCCESS) {
        return FAIL;
    }

    return 0;
}
