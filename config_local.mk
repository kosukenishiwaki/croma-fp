# default build configuration
#
# Adjust the commented paths below only when dependencies are installed in
# non-standard locations. The Makefile reads this file automatically.

# C compiler / MPI wrapper
CC = mpicc

# OpenMP 
OPENMP_CFLAGS = -fopenmp
OPENMP_LDLIBS = -fopenmp
CFLAGS = -O3 $(OPENMP_CFLAGS)

# GSL, MPI, and parallel HDF5
#
# CPPFLAGS += -I/path/to/hdf5/include -I/path/to/gsl/include
# LDFLAGS  += -L/path/to/hdf5/lib -L/path/to/gsl/lib
#
# Add an rpath only when it is required on your system:
# LDFLAGS  += -Wl,-rpath,/path/to/hdf5/lib

# CUDA(optional)
#
# adjust these settings to build `croma_cuda`. CUDA_ARCH must
# match the compute capability supported by the target NVIDIA GPU.
#
# NVCC = nvcc
# CUDA_ARCH = sm_80
# CUDA_OPENMP_LDLIBS = -Xcompiler -fopenmp -lgomp
