# Analysis Utilities

This directory contains optional post-processing utilities for tracer runs.
Benchmark execution harnesses live under `benchmarks/`.

## Run Analysis

- `analyze_tracer_load_balance.py`: summarize rank timing and load-balance output
- `analyze_tracer_bucketstats.py`: inspect bucket statistics from `file_output_mode=bucketstats`
- `aggregate_heavy_tracers.py`: derive heavy-tracer ID lists from bucketstats output
- `compare_tracer_outputs.py`: compare CPU and CUDA binary outputs by tracer ID
- `plot_tracer_cpu_cuda_spectra.py`: plot selected CPU/CUDA tracer spectra
- `plot_tracer_scaling.py`: aggregate and plot scaling runs
- `plot_virtual_load_balance_example.py`: create a load-balance example figure
- `estimate_eta_cap.py`: estimate inputs for the `eta_dpp_cap` model
