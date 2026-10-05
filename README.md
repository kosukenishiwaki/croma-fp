# CROMA: Fokker-Planck cosmic ray spectral solver in post-processing astrophysical simulations

Welcome!

`croma` provides parallel calculations of the spectral evolution of cosmic-ray (CR) protons and electrons by solving the Fokker-Planck equation. It includes first- and second-order Fermi (re)acceleration and the production of secondary particles through hadronic interactions.

CR evolution is modeled as a sub-grid physics under MHD backgrounds. Currently, `croma` uses MHD data stored in tracer-particle outputs from other simulations (e.g., Enzo).
This code was originally designed to simulate non-thermal, multi-messenger emission in the large-scale structure (clusters and filaments).

GPU-offloading is implemented using CUDA, and has been tested on Leonardo supercomputer at CINECA.

## Overview

FP physics:
- Coulomb cooling
- Synchrotron, inverse-Compton, bremsstrahlung losses
- Adiabatic compression/expansion
- pp hadronic losses and secondaries
- Stochastic reacceleration (Fermi II) via `ASA` or `TTD`
- DSA (Fermi I) models

Outputs:
- CR spectra
- Synchrotron emissivity
- Hadronic gamma-ray/neutrino emissivity

## Tracer Mode

### Executables

- `croma.out`: MPI tracer pipeline on the shared-core CPU backend
- `croma_fp.out`: alias build of the CPU tracer pipeline
- `croma_cuda`: CUDA + MPI tracer pipeline with bucketed batched solves
- `tracer_fp_cpu.out` and `tracer_fp_cuda.out`: compatibility/legacy

### Input Data

Tracer mode supports two input modes:
- `input_mode = hdf5`: production tracer snapshots
- `input_mode = synthetic`: generated test background without tracer HDF5 input (for debug)

The HDF5 path consumes per-snapshot files with one row per tracer:

```text
tracer_dump_NNNN.h5
├── density, temperature, B_x/y/z   [N_tracers]
├── div_v, curl_v_mag               [N_tracers]
├── dx_phys, M_tracer               [N_tracers]
└── Redshift                        scalar
```



### Background Modes

- `frozen_background = 0`: normal evolving tracer history 
- `frozen_background = 1`: reuse snapshot-0 background fields across the whole run (mostly for a debug purpose)

CLI override:

```bash
croma.out [params_file]
or
croma.out [params_file] [background_mode]
```

`background_mode` accepts `evolving` or `frozen`.

For `input_mode = hdf5` with `frozen_background = 1`, the code reads only the initial snapshot and derives the runtime from `t_fp_total` and `n_fp_out`.

## Launching Runs

### Direct Tracer Binaries

```bash
mpirun -np 4 ./croma_cuda params.txt
mpirun -np 4 ./croma.out params.txt
```

Backend selection comes from the parameter file:

```text
backend = auto | cpu | cuda
```

Reacceleration mode is also parameter-driven:

```text
Dpp_mode = asa | ttd | direct_tacc | off
t_acc_direct_gyr = 0.3
```

Initial CR species are selected with:

```text
seed_cr_species = electron_only | proton_only | electron_proton
```
Note that secondary electrons emerge even when the initial population is proton_only.

## Job Splitting and MPI Layout

For memory-limited production runs, the safest pattern is one `job_index` chunk per launch with a dedicated output directory for each chunk.

The CUDA tracer path also supports multi-job execution above MPI. Set `job_parallel_enabled = 1` with `job_count > 1` to split `MPI_COMM_WORLD` into independent jobs, or configure `CROMA_NUM_JOBS` / `CROMA_JOB_SIZE` in the environment. 

`job_parallel_enabled=1` is currently incompatible with `load_balancing=1`. Load balancing still uses global MPI collectives internally, so the executable rejects that combination instead of mixing independent jobs.

Example:

```bash
# params.txt contains job_count = 4 and job_parallel_enabled = 1
mpirun -np 8 ./croma_cuda params.txt
```


## Run-mode and Output

The tracer path reads `file_output_mode` from the parameter file:

```text
file_output_mode = write | nowrite | bucketstats | load_estimate
```

Supported modes:
- `write`: normal CR and emission outputs
- `nowrite`: run the solve without writing science outputs

for debug and scaling tests
- `bucketstats`: write bucket diagnostics instead of CR/emission products
- `load_estimate`: estimate tracer cost, write load-balance reports, and exit before the FP solve, this must be run with exactly one MPI rank.

Typical output files in `write` mode:
- `CRE_coreNN.bin`
- `CRP_coreNN.bin` when `write_crp_output = 1`
- `eSyn_coreNN.bin`
- `eIC_coreNN.bin` when `write_ic_output = 1`
- `eGamma_coreNN.bin` when `write_gamma_output = 1` and CR protons are enabled
- `eNu_coreNN.bin` when `write_neutrino_output = 1` and CR protons are enabled
- `timing_coreNNN.tsv`
- `run_summary.tsv`




## Build

Source files live under `src/`; headers live under `include/`. The top-level
Makefile still writes object files and executables to the repository root.


### Dependencies

```makefile
CC     = gcc
LDLIBS = -lm -lgsl -lmpi -lhdf5 -fopenmp

NVCC = nvcc
CUDA_ARCH = sm_75
```

## Parameters

### Run and Input

- `output_dir`: directory for CR spectra, emission products, timings, and reports.
- `N_TRACERS`: number of tracer rows to process from each snapshot. 
- `input_mode`: `hdf5` for tracer snapshot files, or `synthetic` for generated debug backgrounds.
- `tracer_file_dir`, `tracer_filename_base1`, `tracer_filename_base2`,
  `tracer_file_extension`: HDF5 input names.
- `nsnp_i`, `nsnp_f`, `nsnp_start`: first snapshot, final snapshot, and start snapshot index. 


### Backend and Parallel Layout

- `backend`: `auto`, `cpu`, or `cuda`. `auto` uses CUDA only when the executable was built with CUDA.
- `job_count`, `job_index`: split a large tracer set into independent chunks.
  `job_index` is zero-based.
- `job_parallel_enabled`: split `MPI_COMM_WORLD` into independent jobs in one launch. This is currently incompatible with `load_balancing = 1`.

### Output Control

- `file_output_mode`: `write`, `nowrite`, `bucketstats`, or `load_estimate`.
- `synch_output_spec`: synchrotron output cadence. Accepted values are `all`, `none`, `final`, `every:N`, or `steps:i,j,k` where step numbers are one-based.
- `write_buffer_mode`: `tile`, `mapped`, or `buffered`. `tile` is the preferred production mode for bounded memory use.
- `write_buffer_chunk_snapshots`: number of snapshots per tiled output flush.
- `write_crp_output`: write CR proton spectra when set to `1`.
- `write_ic_output`: write inverse-Compton spectra from CR electrons when set to `1`.
- `write_gamma_output`, `write_neutrino_output`: write hadronic gamma-ray and neutrino spectra when set to `1`; CR protons must also be enabled.

### CR Population and FP Coefficients

- `seed_cr_species`: `electron_only`, `proton_only`, or `electron_proton`.
  Secondary electrons can still be generated in proton-seeded runs.
- `InjectionModel`, `steady_primary_electron_injection`, `delta_CR_inj`,
  `phi_CRe`, `phi_CRp`, `pinjmin`, `pinjmax`, `peinjmin`, `peinjmax`: initial and steady injection model controls.
- `Dpp_mode`: `asa`, `ttd`, `direct_tacc`, or `off` for stochastic
  reacceleration. Legacy aliases `tacc` and `direct` are also accepted for the direct acceleration-time path.
- `mach_limit`, `psi`: turbulence/reacceleration model parameters.
- `L_turb_target_kpc`: target turbulence scale used to rescale tracer
  turbulence for ASA/TTD and dynamo-field calculations; defaults to `150` kpc.
- `tracer_nsub_safety`: safety factor for adaptive subcycling.
- `coeff_interp_min_steps`, `coeff_interp_max_segments`: controls for
  interpolation of time-dependent coefficients between tracer snapshots.

### Magnetic Field, Emission, and DSA

- `bfield_mode`: `max`, `sim`, or `dyn`. `max` uses the maximum-field recipe, `sim` uses the simulation magnetic field, and `dyn` enables the dynamo model.
- `eta_B`: magnetic-field model normalization used by the `max`/dynamo paths.
- `nu_min_s`, `nu_max_s`, `synch_logb_min`, `synch_logb_max`,
  `adaptive_synch_logb`, `synch_N_theta`: synchrotron frequency and lookup-table controls.
- `nu_min_ic`, `nu_max_ic`, `E_gamma_min`, `E_gamma_max`, `E_nu_min`,
  `E_nu_max`: inverse-Compton, gamma-ray, and neutrino ranges. Inverse-Compton uses a `log10(nu/Hz)` frequency grid; gamma-ray and neutrino spectral energy grids are `log10(E/GeV)`.
- `DSAInjectionMode`: `off`, `tracer_state`, or `tracer_source`.
- `DSAReaccMode`: `off`, `positive_delta`, or `convolution`. DSA reacceleration requires DSA injection to be enabled.
- `DSAInjectSpecies`, `DSAEtaModelInitial`, `DSAEtaModelReacc`, `DSAChiP`,
  `DSAChiE`, `DSAKep`, `DSAPmaxPmc`, `DSAPmaxEmc`, `DSAMinMach`,
  `DSAGammaGas`, `DSAXcrPminPmc`, `DSAReaccEtaCap`: DSA efficiency, injection, cutoff, and gas-model parameters.

### Load Balancing and Diagnostics

- `load_balancing`: enable cost-based tracer redistribution across MPI ranks.
- `load_balance_top_frac`: fraction of the heaviest tracers used to build the load-balance plan; must be in `(0, 1]`.
