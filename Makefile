# =========================
# default settings
# =========================
CC       ?= cc
CFLAGS   ?= -O2
CPPFLAGS ?=
LDFLAGS  ?=
LDLIBS   ?= -lm
SRC_DIR := src
INC_DIR := include
CPPFLAGS += -I$(INC_DIR)
vpath %.c $(SRC_DIR)
vpath %.cu $(SRC_DIR)
vpath %.h $(INC_DIR)
vpath %.inc $(INC_DIR)

.DEFAULT_GOAL := croma.out

## OpenMP
OPENMP_CFLAGS  ?= -fopenmp
OPENMP_LDLIBS  ?= -fopenmp

CFLAGS += $(OPENMP_CFLAGS)

## GSL / MPI / HDF5
LDLIBS   += -lgsl -lgslcblas -lmpi -lhdf5

# environment settings
CONFIG ?= config_local.mk
-include $(CONFIG)

COMMON_FP_OBJS = params.o FP_Coef.o fp_shared_core.o DSA_MODELS.o CONSTANTS.o READFILE.o HADRONIC.o Chang_Cooper.o Synchrotron.o tracer_fp_alloc.o tracer_fp_cr_source.o COSFUNC.o EMISSION.o read_grid_hdf5.o
TRACER_MODE_OBJS = decomposition_metadata.o

TRACER_FP_COMMON_OBJS = tracer_fp_solve.o tracer_fp_restart.o tracer_fp_cpu.o tracer_fp_coef.o tracer_fp_batch.o tracer_fp_setup.o tracer_fp_background.o tracer_fp_bucket.o tracer_fp_cr_init.o tracer_fp_data.o tracer_fp_debug.o tracer_fp_dsa.o tracer_fp_dsa_reacc.o tracer_fp_entry_helpers.o tracer_fp_nsub.o tracer_fp_output.o tracer_fp_multirate.o tracer_fp_loadbalance.o tracer_fp_selection.o tracer_fp_step.o tracer_fp_synch.o
TRACER_FP_CPU_OBJS = tracer_fp.o $(TRACER_FP_COMMON_OBJS)
TRACER_FP_CUDA_AUX_OBJS = tracer_fp_background_cuda.o tracer_fp_entry_helpers_cuda.o tracer_fp_output_cuda.o tracer_fp_loadbalance_cuda.o
TRACER_FP_CUDA_OBJS = tracer_fp_gpu_cuda.o tracer_fp_solve_cuda.o tracer_fp_restart_cuda.o tracer_fp_cpu.o tracer_fp_coef.o tracer_fp_batch.o tracer_fp_setup.o $(TRACER_FP_CUDA_AUX_OBJS) tracer_fp_bucket.o tracer_fp_cr_init.o tracer_fp_data.o tracer_fp_debug.o tracer_fp_dsa.o tracer_fp_dsa_reacc.o tracer_fp_nsub.o tracer_fp_multirate.o tracer_fp_selection.o tracer_fp_step.o tracer_fp_synch.o
TRACER_CUDA_BACKEND_OBJS = fp_cuda_tracer_backend.o fp_cuda_workspace.o fp_cuda_coeff.o fp_cuda_solver.o fp_cuda_emission.o
CPU_DRIVER_OBJS = $(TRACER_FP_CPU_OBJS) $(COMMON_FP_OBJS) $(TRACER_MODE_OBJS)
CUDA_DRIVER_OBJS = $(TRACER_FP_CUDA_OBJS) $(TRACER_CUDA_BACKEND_OBJS) $(COMMON_FP_OBJS) $(TRACER_MODE_OBJS)

NVCC ?= nvcc
CUDA_ARCH ?= sm_75
CUDA_CXXFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Wno-deprecated-gpu-targets
CUDA_CPPFLAGS ?=
CUDA_LDFLAGS ?= $(filter -L%,$(LDFLAGS))
CUDA_LDLIBS  ?= -Xcompiler -fopenmp -lgomp
CUDA_OPENMP_LDLIBS ?= -Xcompiler -fopenmp -lgomp

croma.out croma_fp.out tracer_fp_cpu.out: $(CPU_DRIVER_OBJS)
	$(CC) $(CFLAGS) -o $@ $(CPU_DRIVER_OBJS) $(LDFLAGS) $(LDLIBS) $(OPENMP_LDLIBS)
	@echo "Success: built $@"

croma_cuda tracer_fp_cuda.out: $(CUDA_DRIVER_OBJS)
	$(NVCC) $(CUDA_CXXFLAGS) -o $@ $(CUDA_DRIVER_OBJS) $(CUDA_LDFLAGS) $(LDLIBS) $(CUDA_OPENMP_LDLIBS)
	@echo "Success: built $@"

tracer_fp.o: tracer_fp.c tracer_fp.h tracer_fp_setup.h tracer_fp_background.h tracer_fp_bucket.h tracer_fp_cr_init.h tracer_fp_data.h tracer_fp_debug.h tracer_fp_dsa.h tracer_fp_entry_helpers.h tracer_fp_nsub.h tracer_fp_output.h tracer_fp_restart.h tracer_fp_loadbalance.h tracer_fp_selection.h tracer_fp_solve.h tracer_fp_step.h tracer_fp_synch.h tracer_fp_alloc.h params.h FP_Coef.h Synchrotron.h read_grid_hdf5.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_solve.o: tracer_fp_solve.c tracer_fp_solve.h tracer_fp.h tracer_fp_background.h tracer_fp_bucket.h tracer_fp_cr_init.h tracer_fp_coef.h tracer_fp_debug.h tracer_fp_dsa.h tracer_fp_loadbalance.h tracer_fp_nsub.h tracer_fp_output.h tracer_fp_restart.h tracer_fp_setup.h tracer_fp_step.h tracer_fp_synch.h params.h FP_Coef.h Synchrotron.h read_grid_hdf5.h fp_cuda_backend.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_restart.o: tracer_fp_restart.c tracer_fp_restart.h tracer_fp.h tracer_fp_background.h tracer_fp_debug.h tracer_fp_output.h params.h FP_Coef.h READFILE.h read_grid_hdf5.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_entry_helpers.o: tracer_fp_entry_helpers.c tracer_fp_entry_helpers.h tracer_fp_background.h tracer_fp_debug.h tracer_fp_loadbalance.h tracer_fp_output.h tracer_fp_selection.h tracer_fp.h params.h FP_Coef.h read_grid_hdf5.h READFILE.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_entry_helpers_cuda.o: tracer_fp_entry_helpers.c tracer_fp_entry_helpers.h tracer_fp_background.h tracer_fp_debug.h tracer_fp_loadbalance.h tracer_fp_output.h tracer_fp_selection.h tracer_fp.h params.h FP_Coef.h read_grid_hdf5.h READFILE.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DFP_USE_CUDA_BACKEND -c -o $@ $<

tracer_fp_cpu.o: tracer_fp_cpu.c tracer_fp_coef.h tracer_fp.h fp_shared_core.h params.h FP_Coef.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_coef.o: tracer_fp_coef.c tracer_fp_coef.h tracer_fp.h fp_shared_core.h params.h FP_Coef.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_batch.o: tracer_fp_batch.c tracer_fp_batch.h tracer_fp_coef.h tracer_fp.h params.h FP_Coef.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_setup.o: tracer_fp_setup.c tracer_fp_setup.h tracer_fp.h fp_shared_core.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_background.o: tracer_fp_background.c tracer_fp_background.h tracer_fp_data.h tracer_fp.h params.h FP_Coef.h read_grid_hdf5.h tracer_input_fields.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_background_cuda.o: tracer_fp_background.c tracer_fp_background.h tracer_fp_data.h tracer_fp.h params.h FP_Coef.h read_grid_hdf5.h tracer_input_fields.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DFP_USE_CUDA_BACKEND -c -o $@ $<

tracer_fp_bucket.o: tracer_fp_bucket.c tracer_fp_bucket.h tracer_fp.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_data.o: tracer_fp_data.c tracer_fp_data.h tracer_fp.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_debug.o: tracer_fp_debug.c tracer_fp_debug.h tracer_fp_background.h tracer_fp.h params.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_cr_init.o: tracer_fp_cr_init.c tracer_fp_cr_init.h tracer_fp_setup.h tracer_fp.h params.h FP_Coef.h tracer_fp_cr_source.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_nsub.o: tracer_fp_nsub.c tracer_fp_nsub.h tracer_fp.h params.h FP_Coef.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_dsa.o: tracer_fp_dsa.c tracer_fp_dsa.h tracer_fp_dsa_reacc.h tracer_fp_setup.h tracer_fp.h DSA_MODELS.h CONSTANTS.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_dsa_reacc.o: tracer_fp_dsa_reacc.c tracer_fp_dsa_reacc.h DSA_MODELS.h CONSTANTS.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_output.o: tracer_fp_output.c tracer_fp_output.h tracer_fp_setup.h tracer_fp.h params.h tracer_fp_alloc.h EMISSION.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_output_cuda.o: tracer_fp_output.c tracer_fp_output.h tracer_fp_setup.h tracer_fp.h params.h tracer_fp_alloc.h EMISSION.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DFP_USE_CUDA_BACKEND -c -o $@ $<

tracer_fp_multirate.o: tracer_fp_multirate.c tracer_fp_multirate.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_loadbalance.o: tracer_fp_loadbalance.c tracer_fp_loadbalance.h tracer_fp_background.h tracer_fp_cr_init.h tracer_fp_debug.h tracer_fp_nsub.h tracer_fp_output.h tracer_fp.h read_grid_hdf5.h READFILE.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_loadbalance_cuda.o: tracer_fp_loadbalance.c tracer_fp_loadbalance.h tracer_fp_background.h tracer_fp_cr_init.h tracer_fp_debug.h tracer_fp_nsub.h tracer_fp_output.h tracer_fp.h read_grid_hdf5.h READFILE.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DFP_USE_CUDA_BACKEND -c -o $@ $<

tracer_fp_selection.o: tracer_fp_selection.c tracer_fp_selection.h tracer_fp.h read_grid_hdf5.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_step.o: tracer_fp_step.c tracer_fp_step.h tracer_fp.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_synch.o: tracer_fp_synch.c tracer_fp_synch.h tracer_fp.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_gpu_cuda.o: tracer_fp.c tracer_fp.h tracer_fp_setup.h tracer_fp_background.h tracer_fp_bucket.h tracer_fp_cr_init.h tracer_fp_data.h tracer_fp_debug.h tracer_fp_dsa.h tracer_fp_entry_helpers.h tracer_fp_nsub.h tracer_fp_output.h tracer_fp_restart.h tracer_fp_loadbalance.h tracer_fp_selection.h tracer_fp_solve.h tracer_fp_step.h tracer_fp_synch.h tracer_fp_alloc.h fp_cuda_backend.h params.h FP_Coef.h Synchrotron.h read_grid_hdf5.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DFP_USE_CUDA_BACKEND -c -o $@ $<

tracer_fp_solve_cuda.o: tracer_fp_solve.c tracer_fp_solve.h tracer_fp.h tracer_fp_background.h tracer_fp_bucket.h tracer_fp_cr_init.h tracer_fp_coef.h tracer_fp_debug.h tracer_fp_dsa.h tracer_fp_loadbalance.h tracer_fp_nsub.h tracer_fp_output.h tracer_fp_restart.h tracer_fp_setup.h tracer_fp_step.h tracer_fp_synch.h fp_cuda_backend.h params.h FP_Coef.h Synchrotron.h read_grid_hdf5.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DFP_USE_CUDA_BACKEND -c -o $@ $<

tracer_fp_restart_cuda.o: tracer_fp_restart.c tracer_fp_restart.h tracer_fp.h tracer_fp_background.h tracer_fp_debug.h tracer_fp_output.h fp_cuda_backend.h params.h FP_Coef.h READFILE.h read_grid_hdf5.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -DFP_USE_CUDA_BACKEND -c -o $@ $<

fp_cuda_tracer_backend.o: fp_cuda_tracer_backend.cu fp_cuda_backend.h fp_cuda_coeff.h fp_cuda_emission.h fp_cuda_solver.h fp_cuda_workspace.h fp_shared_core.h params.h
	$(NVCC) $(CUDA_CXXFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

fp_cuda_workspace.o: fp_cuda_workspace.cu fp_cuda_workspace.h fp_cuda_backend.h fp_shared_core.h params.h Synchrotron.h
	$(NVCC) $(CUDA_CXXFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

fp_cuda_coeff.o: fp_cuda_coeff.cu fp_cuda_coeff.h fp_shared_core.h params.h
	$(NVCC) $(CUDA_CXXFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

fp_cuda_solver.o: fp_cuda_solver.cu fp_cuda_solver.h fp_cuda_backend.h fp_shared_core.h
	$(NVCC) $(CUDA_CXXFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

fp_cuda_emission.o: fp_cuda_emission.cu fp_cuda_emission.h fp_cuda_backend.h fp_cuda_compat.h
	$(NVCC) $(CUDA_CXXFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<

fp_shared_core.o: fp_shared_core.c fp_shared_core.h params.h CONSTANTS.h HADRONIC.h FP_Coef.h EMISSION.h Synchrotron.h Chang_Cooper.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

params.o: params.c params.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

READFILE.o: READFILE.c READFILE.h params.h DSA_MODELS.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

CONSTANTS.o: CONSTANTS.c CONSTANTS.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

FP_Coef.o: FP_Coef.c FP_Coef.h HADRONIC.h params.h CONSTANTS.h fp_shared_core.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_alloc.o: tracer_fp_alloc.c tracer_fp_alloc.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

tracer_fp_cr_source.o: tracer_fp_cr_source.c tracer_fp_cr_source.h params.h CONSTANTS.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

Synchrotron.o: Synchrotron.c Synchrotron.h CONSTANTS.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

EMISSION.o: EMISSION.c EMISSION.h params.h CONSTANTS.h COSFUNC.h Synchrotron.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

HADRONIC.o: HADRONIC.c HADRONIC.h params.h CONSTANTS.h FP_Coef.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

COSFUNC.o: COSFUNC.c COSFUNC.h params.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

Chang_Cooper.o: Chang_Cooper.c Chang_Cooper.h fp_shared_core.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

DSA_MODELS.o: DSA_MODELS.c DSA_MODELS.h params.h CONSTANTS.h FP_Coef.h HADRONIC.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

read_grid_hdf5.o: read_grid_hdf5.c read_grid_hdf5.h params.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

decomposition_metadata.o: decomposition_metadata.c decomposition_metadata.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $<

.PHONY: clean all

all: croma.out
	@echo "Success: completed $@"

clean:
	rm -f *.o croma.out croma_fp.out croma_cuda tracer_fp_cpu.out tracer_fp_cuda.out
