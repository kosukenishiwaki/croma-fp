/*
    tracer_fp_output.c

    K. Nishiwaki, 2026-06-18
    - file output utils

*/

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "CONSTANTS.h"
#include "EMISSION.h"
#include "tracer_fp_alloc.h"
#include "DSA_MODELS.h"
#include "params.h"
#include "tracer_fp_output.h"
#include "tracer_fp_setup.h"

#ifdef FP_USE_CUDA_BACKEND
#define TRACER_FP_PROGNAME "tracer_fp_cuda.out"
#else
#define TRACER_FP_PROGNAME "tracer_fp_cpu.out"
#endif

void output_CRspectrum_buffer_node(char *strout,
                                   int n_tracer_node,
                                   int size_per_tracer,
                                   int sign,
                                   int mpi_rank,
                                   double **n_buffer)
{
    char path[MAX_LINE_LENGTH];
    FILE *fp;

    if (sign < 0) {
        snprintf(path, sizeof(path), "%s/CRE_core%02d.bin", strout, mpi_rank);
    } else {
        snprintf(path, sizeof(path), "%s/CRP_core%02d.bin", strout, mpi_rank);
    }

    fprintf(stderr, "writing ... %s\n", path);

    fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "rank %d: Cannot open file %s\n", mpi_rank, path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int itr = 0; itr < n_tracer_node; itr++) {
        if (n_buffer[itr] == NULL) {
            fprintf(stderr, "rank %d: n_buffer[%d] is NULL\n", mpi_rank, itr);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fwrite(n_buffer[itr], sizeof(double), (size_t)size_per_tracer, fp);
    }

    fclose(fp);
}

void output_tracerid_node(char *strout,
                          int n_tracer_node,
                          int mpi_rank,
                          int *id_thread)
{
    char path[MAX_LINE_LENGTH];
    FILE *fp;

    snprintf(path, sizeof(path), "%s/tracerid_core%02d.txt", strout, mpi_rank);

    fp = fopen(path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "rank %d: Cannot open file %s\n", mpi_rank, path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int itr = 0; itr < n_tracer_node; itr++) {
        fprintf(fp, "%09d\n", id_thread[itr]);
    }

    fclose(fp);
}

int ensure_output_dir(const char *path)
{
    struct stat st;

    if (path == 0 || *path == '\0') return -1;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }
    if (mkdir(path, 0755) == 0) return 0;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return 0;
    return -1;
}

void tracer_fp_map_reset(TracerFpMappedOutput *out)
{
    if (out == 0) return;
    out->rows = 0;
    out->base = 0;
    out->nrow = 0;
    out->ncol = 0;
    out->nbyte = 0;
    out->fd = -1;
    out->enabled = 0;
    out->path[0] = '\0';
}

void tracer_fp_tile_reset(TracerFpTileOutput *out)
{
    if (out == 0) return;
    out->nslab = 0;
    out->nrow = 0;
    out->ncol = 0;
    out->nbyte = 0;
    out->fd = -1;
    out->enabled = 0;
    out->path[0] = '\0';
}

static int reserve_mapped_file(int fd, off_t nbyte)
{
#if defined(__linux__)
    return posix_fallocate(fd, 0, nbyte);
#else
    (void)fd;
    (void)nbyte;
    return 0;
#endif
}

static int pwrite_all(int fd, const void *buf, size_t nbyte, off_t offset)
{
    const char *ptr = (const char *)buf;
    size_t done = 0;

    while (done < nbyte) {
        ssize_t nwritten = pwrite(fd, ptr + done, nbyte - done, offset + (off_t)done);
        if (nwritten < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (nwritten == 0) {
            errno = EIO;
            return -1;
        }
        done += (size_t)nwritten;
    }
    return 0;
}

int tracer_fp_map_sync(const TracerFpMappedOutput *out)
{
    if (out == 0 || !out->enabled || out->fd < 0) return 0;
    return fsync(out->fd);
}

int tracer_fp_tile_sync(const TracerFpTileOutput *out)
{
    if (out == 0 || !out->enabled || out->fd < 0) return 0;
    return fsync(out->fd);
}

static double output_volume_factor(const double *rho_gcc,
                                             const double *tracer_mass,
                                             int itr,
                                             int output_per_cc)
{
    /* The tracer HDF5 path converts rho to physical cgs via DATASET_NAME_DCONV,
     * and M_tracer is stored as physical mass in M_sun. Therefore output_per_cc
     * means physical per-cm^3, with no extra cosmological (1+z)^3 factor here. */
    if (!output_per_cc) return 1.0;
    if (rho_gcc == 0 || tracer_mass == 0 || !(M_sun > 0.0)) return 1.0;
    if (!(rho_gcc[itr] > 0.0) || !(tracer_mass[itr] > 0.0)) return 0.0;
    return rho_gcc[itr] / (tracer_mass[itr] * M_sun);
}

static int mapped_output_write_row(TracerFpMappedOutput *out,
                                             size_t row,
                                             size_t col,
                                             const double *values,
                                             size_t nvalue)
{
    size_t first_value;
    size_t nbyte;

    if (out == 0 || !out->enabled || values == 0) return -1;
    if (row >= out->nrow || col > out->ncol || nvalue > out->ncol - col) {
        errno = EINVAL;
        return -1;
    }
    first_value = row * out->ncol + col;
    if (first_value > ((size_t)-1) / sizeof(double) ||
        nvalue > ((size_t)-1) / sizeof(double)) {
        errno = EOVERFLOW;
        return -1;
    }
    nbyte = nvalue * sizeof(double);
    return pwrite_all(out->fd, values, nbyte, (off_t)(first_value * sizeof(double)));
}

static int tile_output_write_row(TracerFpTileOutput *out,
                                           size_t slab,
                                           size_t row,
                                           size_t col,
                                           const double *values,
                                           size_t nvalue)
{
    size_t slab_values;
    size_t first_value;
    size_t nbyte;

    if (out == 0 || !out->enabled || values == 0) return -1;
    if (slab >= out->nslab || row >= out->nrow || col > out->ncol ||
        nvalue > out->ncol - col) {
        errno = EINVAL;
        return -1;
    }
    if (out->nrow > ((size_t)-1) / out->ncol) {
        errno = EOVERFLOW;
        return -1;
    }
    slab_values = out->nrow * out->ncol;
    if (slab > ((size_t)-1) / slab_values) {
        errno = EOVERFLOW;
        return -1;
    }
    first_value = slab * slab_values + row * out->ncol + col;
    if (first_value > ((size_t)-1) / sizeof(double) ||
        nvalue > ((size_t)-1) / sizeof(double)) {
        errno = EOVERFLOW;
        return -1;
    }
    nbyte = nvalue * sizeof(double);
    return pwrite_all(out->fd, values, nbyte, (off_t)(first_value * sizeof(double)));
}

int tracer_fp_write_cr_map(TracerFpMappedOutput *ne_out,
                           TracerFpMappedOutput *np_out,
                           int nlocal,
                           int snapshot_slot,
                           const double *cre_state,
                           const double *crp_state,
                           const double *rho_gcc,
                           const double *tracer_mass,
                           int output_per_cc)
{
    int itr;
    const size_t dst_p_col = (size_t)snapshot_slot * (size_t)np;
    const size_t dst_e_col = (size_t)snapshot_slot * (size_t)npe;

    if (ne_out == 0 || cre_state == 0) return -1;
    for (itr = 0; itr < nlocal; itr++) {
        const size_t src_p_off = (size_t)itr * (size_t)np;
        const size_t src_e_off = (size_t)itr * (size_t)npe;
        if (output_per_cc) {
            double factor = output_volume_factor(rho_gcc, tracer_mass, itr, output_per_cc);
            double tmp[np > npe ? np : npe];
            int j;

            if (np_out != 0 && np_out->enabled && crp_state != 0) {
                for (j = 0; j < np; j++) tmp[j] = crp_state[src_p_off + (size_t)j] * factor;
                if (mapped_output_write_row(np_out, (size_t)itr, dst_p_col,
                                                      tmp, (size_t)np) != 0) {
                    return -1;
                }
            }
            for (j = 0; j < npe; j++) tmp[j] = cre_state[src_e_off + (size_t)j] * factor;
            if (mapped_output_write_row(ne_out, (size_t)itr, dst_e_col,
                                                  tmp, (size_t)npe) != 0) {
                return -1;
            }
        } else {
            if (np_out != 0 && np_out->enabled && crp_state != 0 &&
                mapped_output_write_row(np_out, (size_t)itr, dst_p_col,
                                                  crp_state + src_p_off, (size_t)np) != 0) {
                return -1;
            }
            if (mapped_output_write_row(ne_out, (size_t)itr, dst_e_col,
                                                  cre_state + src_e_off, (size_t)npe) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

int tracer_fp_write_emit_map(TracerFpMappedOutput *out,
                             int nlocal,
                             int snapshot_index,
                             int nbin,
                             const double *emission_cell_major,
                             const double *rho_gcc,
                             const double *tracer_mass,
                             int output_per_cc)
{
    int itr;
    const size_t dst_col = (size_t)snapshot_index * (size_t)nbin;

    if (out == 0 || emission_cell_major == 0) return -1;
    for (itr = 0; itr < nlocal; itr++) {
        const size_t src_off = (size_t)itr * (size_t)nbin;
        if (output_per_cc) {
            double factor = output_volume_factor(rho_gcc, tracer_mass, itr, output_per_cc);
            double tmp[nbin];
            int j;
            for (j = 0; j < nbin; j++) tmp[j] = emission_cell_major[src_off + (size_t)j] * factor;
            if (mapped_output_write_row(out, (size_t)itr, dst_col,
                                                  tmp, (size_t)nbin) != 0) {
                return -1;
            }
        } else if (mapped_output_write_row(out, (size_t)itr, dst_col,
                                                     emission_cell_major + src_off,
                                                     (size_t)nbin) != 0) {
            return -1;
        }
    }
    return 0;
}

void tracer_fp_zero_rows(double **buffer, int nrow, size_t ncol)
{
    int itr;

    if (buffer == 0) return;
    for (itr = 0; itr < nrow; itr++) {
        if (buffer[itr] != 0) memset(buffer[itr], 0, ncol * sizeof(double));
    }
}

static int write_buffer_rows_mapped(TracerFpMappedOutput *out,
                                              int nlocal,
                                              size_t dst_col,
                                              double **rows,
                                              size_t ncol)
{
    int itr;

    if (out == 0 || rows == 0) return -1;
    for (itr = 0; itr < nlocal; itr++) {
        if (mapped_output_write_row(out, (size_t)itr, dst_col,
                                              rows[itr], ncol) != 0) {
            return -1;
        }
    }
    return 0;
}

int tracer_fp_tile_open(TracerFpTileOutput *out,
                               const char *output_dir,
                               const char *stem,
                               int mpi_rank,
                               size_t nslab,
                               size_t nrow,
                               size_t ncol,
                               int resume_existing)
{
    size_t total_values;
    int fallocate_err = 0;
    struct stat st;

    if (out == 0 || output_dir == 0 || *output_dir == '\0' || stem == 0 ||
        nslab == 0 || nrow == 0 || ncol == 0) return -1;
    tracer_fp_tile_reset(out);
    out->nslab = nslab;
    out->nrow = nrow;
    out->ncol = ncol;
    if (nslab > ((size_t)-1) / nrow || nslab * nrow > ((size_t)-1) / ncol) {
        errno = EOVERFLOW;
        goto fail;
    }
    total_values = nslab * nrow * ncol;
    if (total_values > ((size_t)-1) / sizeof(double)) {
        errno = EOVERFLOW;
        goto fail;
    }
    out->nbyte = total_values * sizeof(double);
    snprintf(out->path, sizeof(out->path), "%s/%s_tile_core%02d.bin", output_dir, stem, mpi_rank);

    out->fd = resume_existing ? open(out->path, O_RDWR) : open(out->path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out->fd < 0) goto fail;
    if (resume_existing) {
        if (fstat(out->fd, &st) != 0) goto fail;
        if ((size_t)st.st_size != out->nbyte) {
            errno = EINVAL;
            fprintf(stderr,
                    TRACER_FP_PROGNAME ": existing tile output '%s' has size=%lld, expected=%zu\n",
                    out->path, (long long)st.st_size, out->nbyte);
            goto fail;
        }
    } else {
        if (ftruncate(out->fd, (off_t)out->nbyte) != 0) goto fail;
        fallocate_err = reserve_mapped_file(out->fd, (off_t)out->nbyte);
        if (fallocate_err != 0) {
            errno = fallocate_err;
            goto fail;
        }
    }

    out->enabled = 1;
    return 0;

fail:
    fprintf(stderr, TRACER_FP_PROGNAME ": tile output allocation failed for '%s' (%s)\n",
            out->path[0] ? out->path : stem, strerror(errno));
    if (out->fd >= 0) close(out->fd);
    tracer_fp_tile_reset(out);
    return -1;
}

int tracer_fp_tile_write(TracerFpTileOutput *out,
                         size_t slab,
                         const double *values)
{
    size_t slab_values;
    size_t byte_offset;

    if (out == 0 || !out->enabled || values == 0) return -1;
    if (slab >= out->nslab) {
        errno = EINVAL;
        return -1;
    }
    if (out->nrow > ((size_t)-1) / out->ncol) {
        errno = EOVERFLOW;
        return -1;
    }
    slab_values = out->nrow * out->ncol;
    if (slab_values > ((size_t)-1) / sizeof(double) ||
        slab > ((size_t)-1) / slab_values) {
        errno = EOVERFLOW;
        return -1;
    }
    byte_offset = slab * slab_values * sizeof(double);
    return pwrite_all(out->fd, values, slab_values * sizeof(double), (off_t)byte_offset);
}

int tracer_fp_tile_write_cr(TracerFpTileOutput *ne_out,
                            TracerFpTileOutput *np_out,
                            int nlocal,
                            size_t slab,
                            const double *cre_state,
                            const double *crp_state,
                            const double *rho_gcc,
                            const double *tracer_mass,
                            int output_per_cc,
                            int write_crp_output)
{
    int itr;

    if (ne_out == 0 || cre_state == 0) return -1;
    if (!output_per_cc) {
        if (tracer_fp_tile_write(ne_out, slab, cre_state) != 0) return -1;
        if (write_crp_output && np_out != 0 && crp_state != 0 &&
            tracer_fp_tile_write(np_out, slab, crp_state) != 0) {
            return -1;
        }
        return 0;
    }

    for (itr = 0; itr < nlocal; itr++) {
        const size_t src_p_off = (size_t)itr * (size_t)np;
        const size_t src_e_off = (size_t)itr * (size_t)npe;
        double factor = output_volume_factor(rho_gcc, tracer_mass, itr, output_per_cc);
        double tmp[np > npe ? np : npe];
        int j;

        if (write_crp_output && np_out != 0 && crp_state != 0) {
            for (j = 0; j < np; j++) tmp[j] = crp_state[src_p_off + (size_t)j] * factor;
            if (tile_output_write_row(np_out, slab, (size_t)itr, 0, tmp, (size_t)np) != 0) {
                return -1;
            }
        }
        for (j = 0; j < npe; j++) tmp[j] = cre_state[src_e_off + (size_t)j] * factor;
        if (tile_output_write_row(ne_out, slab, (size_t)itr, 0, tmp, (size_t)npe) != 0) {
            return -1;
        }
    }
    return 0;
}

int tracer_fp_tile_write_emit(TracerFpTileOutput *out,
                              int nlocal,
                              size_t slab,
                              int nbin,
                              const double *emission_cell_major,
                              const double *rho_gcc,
                              const double *tracer_mass,
                              int output_per_cc)
{
    int itr;

    if (out == 0 || emission_cell_major == 0) return -1;
    if (!output_per_cc) return tracer_fp_tile_write(out, slab, emission_cell_major);

    for (itr = 0; itr < nlocal; itr++) {
        const size_t src_off = (size_t)itr * (size_t)nbin;
        double factor = output_volume_factor(rho_gcc, tracer_mass, itr, output_per_cc);
        double tmp[nbin];
        int j;
        for (j = 0; j < nbin; j++) tmp[j] = emission_cell_major[src_off + (size_t)j] * factor;
        if (tile_output_write_row(out, slab, (size_t)itr, 0, tmp, (size_t)nbin) != 0) {
            return -1;
        }
    }
    return 0;
}

int tracer_fp_tile_close(TracerFpTileOutput *out)
{
    int ierr = 0;

    if (out == 0 || !out->enabled) return 0;
    if (out->fd >= 0) {
        if (fsync(out->fd) != 0) ierr = -1;
        if (close(out->fd) != 0) ierr = -1;
    }
    tracer_fp_tile_reset(out);
    return ierr;
}

int tracer_fp_flush_map_chunk(TracerFpMappedOutput *ne_out,
                                        TracerFpMappedOutput *np_out,
                                        TracerFpMappedOutput *epssyn_out,
                                        TracerFpMappedOutput *epsic_out,
                                        TracerFpMappedOutput *epsgamma_out,
                                        TracerFpMappedOutput *epsnu_out,
                                        int ic_enabled,
                                        int gamma_enabled,
                                        int neutrino_enabled,
                                        int nlocal,
                                        int nfreq,
                                        int chunk_start_snapshot,
                                        int chunk_count,
                                        int chunk_cr_base_slot,
                                        int chunk_cr_slots,
                                        double **ne_chunk,
                                        double **np_chunk,
                                        double **epssyn_chunk,
                                        double **epsic_chunk,
                                        double **epsgamma_chunk,
                                        double **epsnu_chunk)
{
    if (chunk_count <= 0 && chunk_cr_slots <= 0) return 0;
    if (chunk_cr_slots > 0) {
        if (write_buffer_rows_mapped(ne_out, nlocal,
                                               (size_t)chunk_cr_base_slot * (size_t)npe,
                                               ne_chunk,
                                               (size_t)chunk_cr_slots * (size_t)npe) != 0) {
            return -1;
        }
        if (np_out != 0 && np_out->enabled && np_chunk != 0 &&
            write_buffer_rows_mapped(np_out, nlocal,
                                               (size_t)chunk_cr_base_slot * (size_t)np,
                                               np_chunk,
                                               (size_t)chunk_cr_slots * (size_t)np) != 0) {
            return -1;
        }
    }
    if (chunk_count > 0) {
        if (write_buffer_rows_mapped(epssyn_out, nlocal,
                                               (size_t)chunk_start_snapshot * (size_t)nfreq,
                                               epssyn_chunk,
                                               (size_t)chunk_count * (size_t)nfreq) != 0) {
            return -1;
        }
        if (ic_enabled && epsic_chunk != 0) {
            if (write_buffer_rows_mapped(epsic_out, nlocal,
                                                   (size_t)chunk_start_snapshot * (size_t)bins_IC,
                                                   epsic_chunk,
                                                   (size_t)chunk_count * (size_t)bins_IC) != 0) {
                return -1;
            }
        }
        if (gamma_enabled && epsgamma_chunk != 0) {
            if (write_buffer_rows_mapped(epsgamma_out, nlocal,
                                                   (size_t)chunk_start_snapshot * (size_t)bins_gamma,
                                                   epsgamma_chunk,
                                                   (size_t)chunk_count * (size_t)bins_gamma) != 0) {
                return -1;
            }
        }
        if (neutrino_enabled && epsnu_chunk != 0) {
            if (write_buffer_rows_mapped(epsnu_out, nlocal,
                                                   (size_t)chunk_start_snapshot * (size_t)bins_nu,
                                                   epsnu_chunk,
                                                   (size_t)chunk_count * (size_t)bins_nu) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

int tracer_fp_map_open(TracerFpMappedOutput *out,
                                 const char *output_dir,
                                 const char *stem,
                                 int mpi_rank,
                                 int nrow,
                                 size_t ncol,
                                 int resume_existing)
{
    size_t total_values;
    struct stat st;
    int fallocate_err = 0;

    if (out == 0 || output_dir == 0 || *output_dir == '\0' || stem == 0 || nrow < 0 || ncol == 0) return -1;
    tracer_fp_map_reset(out);
    out->nrow = (size_t)nrow;
    out->ncol = ncol;
    if (out->nrow != 0 && out->ncol > ((size_t)-1) / out->nrow) {
        errno = EOVERFLOW;
        goto fail;
    }
    total_values = out->nrow * out->ncol;
    if (total_values > ((size_t)-1) / sizeof(double)) {
        errno = EOVERFLOW;
        goto fail;
    }
    out->nbyte = total_values * sizeof(double);
    snprintf(out->path, sizeof(out->path), "%s/%s_core%02d.bin", output_dir, stem, mpi_rank);

    out->fd = resume_existing ? open(out->path, O_RDWR) : open(out->path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out->fd < 0) goto fail;
    if (resume_existing) {
        if (fstat(out->fd, &st) != 0 || (size_t)st.st_size != out->nbyte) goto fail;
    } else {
        if (ftruncate(out->fd, (off_t)out->nbyte) != 0) goto fail;
        fallocate_err = reserve_mapped_file(out->fd, (off_t)out->nbyte);
        if (fallocate_err != 0) {
            errno = fallocate_err;
            goto fail;
        }
    }

    if (mpi_rank == 0) {
        printf("  mapped output      : %s rows=%zu cols=%zu size=%.3f GiB mode=direct-pwrite\n",
               stem, out->nrow, out->ncol,
               (double)out->nbyte / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);
    }
    out->enabled = 1;
    return 0;

fail:
    fprintf(stderr, TRACER_FP_PROGNAME ": mapped output allocation failed for '%s' (%s)\n",
            out->path[0] ? out->path : stem, strerror(errno));
    if (out->fd >= 0) close(out->fd);
    tracer_fp_map_reset(out);
    return -1;
}

int tracer_fp_map_close(TracerFpMappedOutput *out)
{
    int ierr = 0;

    if (out == 0 || !out->enabled) return 0;
    if (out->base != 0 && out->nbyte > 0) {
        if (msync(out->base, out->nbyte, MS_SYNC) != 0) ierr = -1;
        if (munmap(out->base, out->nbyte) != 0) ierr = -1;
    }
    if (out->base == 0 && out->fd >= 0) {
        if (fsync(out->fd) != 0) ierr = -1;
    }
    if (out->fd >= 0 && close(out->fd) != 0) ierr = -1;
    free(out->rows);
    tracer_fp_map_reset(out);
    return ierr;
}

int tracer_fp_outputs_init(TracerFpOutputs *files,
                           const TracerFpOutputCfg *cfg)
{
    int itr;
    char *output_dir;
    size_t output_dir_size;
    char *checkpoint_dir_effective;
    size_t checkpoint_dir_size;

    if (files == 0 || cfg == 0) return 1;
    output_dir = cfg->output_dir;
    output_dir_size = cfg->output_dir_size;
    checkpoint_dir_effective = cfg->checkpoint_dir;
    checkpoint_dir_size = cfg->checkpoint_dir_size;

    if (cfg->configured_output_dir[0] != '\0') {
        snprintf(output_dir, output_dir_size, "%s", cfg->configured_output_dir);
    } else if (cfg->write_output_files || cfg->bucket_stats_only) {
        snprintf(output_dir, output_dir_size, "tracer_fp_gpu_output");
    }
    if (cfg->checkpoint_enabled) {
        if (tracer_checkpoint_dir[0] != '\0') {
            snprintf(checkpoint_dir_effective, checkpoint_dir_size,
                     "%s", tracer_checkpoint_dir);
        } else {
            snprintf(checkpoint_dir_effective, checkpoint_dir_size,
                     "%s/checkpoints", output_dir);
        }
    }

    if (output_dir[0] != '\0') {
        if (ensure_output_dir(output_dir) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to create output directory '%s'\n",
                    output_dir);
            return 1;
        }
    }
    if (cfg->checkpoint_enabled && ensure_output_dir(checkpoint_dir_effective) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to create checkpoint directory '%s'\n",
                checkpoint_dir_effective);
        return 1;
    }

    if (cfg->use_tile_output) {
        if (tracer_fp_tile_open(files->ne_tile_output, output_dir, "CRE", cfg->mpi_rank,
                                       (size_t)(cfg->nsnap + 1), (size_t)cfg->ntracer, (size_t)npe,
                                       cfg->restart_enabled) != 0 ||
            (cfg->write_crp_output &&
             tracer_fp_tile_open(files->np_tile_output, output_dir, "CRP", cfg->mpi_rank,
                                        (size_t)(cfg->nsnap + 1), (size_t)cfg->ntracer, (size_t)np,
                                        cfg->restart_enabled) != 0) ||
            tracer_fp_tile_open(files->epssyn_tile_output, output_dir, "eSyn", cfg->mpi_rank,
                                       (size_t)cfg->nsnap, (size_t)cfg->ntracer, (size_t)cfg->nfreq,
                                       cfg->restart_enabled) != 0 ||
            (cfg->ic_enabled &&
             tracer_fp_tile_open(files->epsic_tile_output, output_dir, "eIC", cfg->mpi_rank,
                                        (size_t)cfg->nsnap, (size_t)cfg->ntracer, (size_t)bins_IC,
                                        cfg->restart_enabled) != 0) ||
            (cfg->gamma_enabled &&
             tracer_fp_tile_open(files->epsgamma_tile_output, output_dir, "eGamma", cfg->mpi_rank,
                                        (size_t)cfg->nsnap, (size_t)cfg->ntracer, (size_t)bins_gamma,
                                        cfg->restart_enabled) != 0) ||
            (cfg->neutrino_enabled &&
             tracer_fp_tile_open(files->epsnu_tile_output, output_dir, "eNu", cfg->mpi_rank,
                                        (size_t)cfg->nsnap, (size_t)cfg->ntracer, (size_t)bins_nu,
                                        cfg->restart_enabled) != 0)) {
            return 1;
        }
    } else if (cfg->use_mapped_output) {
        if (tracer_fp_map_open(files->ne_output, output_dir, "CRE", cfg->mpi_rank,
                                         cfg->ntracer, (size_t)(cfg->nsnap + 1) * (size_t)npe,
                                         cfg->restart_enabled) != 0 ||
            (cfg->write_crp_output &&
             tracer_fp_map_open(files->np_output, output_dir, "CRP", cfg->mpi_rank,
                                          cfg->ntracer, (size_t)(cfg->nsnap + 1) * (size_t)np,
                                          cfg->restart_enabled) != 0) ||
            tracer_fp_map_open(files->epssyn_output, output_dir, "eSyn", cfg->mpi_rank,
                                         cfg->ntracer, (size_t)cfg->nsnap * (size_t)cfg->nfreq,
                                         cfg->restart_enabled) != 0 ||
            (cfg->ic_enabled &&
             tracer_fp_map_open(files->epsic_output, output_dir, "eIC", cfg->mpi_rank,
                                          cfg->ntracer, (size_t)cfg->nsnap * (size_t)bins_IC,
                                          cfg->restart_enabled) != 0) ||
            (cfg->gamma_enabled &&
             tracer_fp_map_open(files->epsgamma_output, output_dir, "eGamma", cfg->mpi_rank,
                                          cfg->ntracer, (size_t)cfg->nsnap * (size_t)bins_gamma,
                                          cfg->restart_enabled) != 0) ||
            (cfg->neutrino_enabled &&
             tracer_fp_map_open(files->epsnu_output, output_dir, "eNu", cfg->mpi_rank,
                                          cfg->ntracer, (size_t)cfg->nsnap * (size_t)bins_nu,
                                          cfg->restart_enabled) != 0)) {
            return 1;
        }
        if (cfg->use_mapped_chunk) {
            *files->ne_chunk_core = allocate2DArray(cfg->ntracer, (cfg->mapped_chunk_snapshots + 1) * npe);
            *files->np_chunk_core = cfg->write_crp_output
                ? allocate2DArray(cfg->ntracer, (cfg->mapped_chunk_snapshots + 1) * np) : 0;
            *files->epssyn_chunk_core = allocate2DArray(cfg->ntracer, cfg->mapped_chunk_snapshots * cfg->nfreq);
            *files->epsic_chunk_core = cfg->ic_enabled
                ? allocate2DArray(cfg->ntracer, cfg->mapped_chunk_snapshots * bins_IC) : 0;
            *files->epsgamma_chunk_core = cfg->gamma_enabled
                ? allocate2DArray(cfg->ntracer, cfg->mapped_chunk_snapshots * bins_gamma) : 0;
            *files->epsnu_chunk_core = cfg->neutrino_enabled
                ? allocate2DArray(cfg->ntracer, cfg->mapped_chunk_snapshots * bins_nu) : 0;
            if (*files->ne_chunk_core == 0 || (cfg->write_crp_output && *files->np_chunk_core == 0) ||
                *files->epssyn_chunk_core == 0 ||
                (cfg->ic_enabled && *files->epsic_chunk_core == 0) ||
                (cfg->gamma_enabled && *files->epsgamma_chunk_core == 0) ||
                (cfg->neutrino_enabled && *files->epsnu_chunk_core == 0)) {
                fprintf(stderr, TRACER_FP_PROGNAME ": failed to allocate mapped output chunk buffers\n");
                return 1;
            }
            if (cfg->log_root) {
                const double chunk_gib =
                    (double)cfg->ntracer *
                    ((double)(cfg->mapped_chunk_snapshots + 1) *
                     (double)(npe + (cfg->write_crp_output ? np : 0)) +
                     (double)cfg->mapped_chunk_snapshots *
                     (double)(cfg->nfreq +
                              (cfg->ic_enabled ? bins_IC : 0) +
                              (cfg->gamma_enabled ? bins_gamma : 0) +
                              (cfg->neutrino_enabled ? bins_nu : 0))) *
                    sizeof(double) / (1024.0 * 1024.0 * 1024.0);
                printf("  mapped chunk buffer: snapshots=%d approx=%.3f GiB/rank\n",
                       cfg->mapped_chunk_snapshots, chunk_gib);
                fflush(stdout);
            }
        }
    }

    if (cfg->write_output_files || cfg->bucket_stats_only) {
        if (cfg->bucket_stats_only && cfg->mpi_rank == 0) {
            char bucketstats_path[MAX_LINE_LENGTH];
            snprintf(bucketstats_path, sizeof(bucketstats_path), "%s/%s", output_dir,
                     "bucketstats_top_tracers.tsv");
            *files->bucketstats_top_fp = fopen(bucketstats_path, "w");
            if (*files->bucketstats_top_fp == 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": failed to open bucketstats output '%s'\n",
                        bucketstats_path);
                return 1;
            }
            fprintf(*files->bucketstats_top_fp,
                    "snapshot_index\tphysical_snapshot\tz\ttracer_id\tnsub\ttarget_nsub\tinflate\tn_gas\tkbt_GeV\tb_field_G\tdivv_Gyr\tlturb_Mpc\tdv_cms\tbeta_pl\n");
            fflush(*files->bucketstats_top_fp);
        }
        if (cfg->bucket_stats_only) {
            char bucketstats_rank_path[MAX_LINE_LENGTH];
            snprintf(bucketstats_rank_path, sizeof(bucketstats_rank_path), "%s/bucketstats_rank%03d.tsv",
                     output_dir, cfg->mpi_rank);
            *files->bucketstats_rank_fp = fopen(bucketstats_rank_path, "w");
            if (*files->bucketstats_rank_fp == 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": failed to open bucketstats rank output '%s'\n",
                        bucketstats_rank_path);
                return 1;
            }
            fprintf(*files->bucketstats_rank_fp,
                    "rank\tsnapshot_index\tphysical_snapshot\tz\tbucket_index\tnbuckets\tbucket_size\tnsub\tn_on\tcoeff_segments\ttarget_min\ttarget_max\ttarget_avg\tinflate_avg\tinflate_max\n");
            fflush(*files->bucketstats_rank_fp);
        }
        for (itr = 0; itr < cfg->ntracer; itr++) {
            cfg->tracer_id_core[itr] = (int)cfg->tracer_ids[itr];
        }
    }

    return 0;
}

void tracer_fp_outputs_free(TracerFpOutputs *files,
                               int ntracer)
{
    if (files == 0) return;
    tracer_fp_tile_close(files->ne_tile_output);
    tracer_fp_tile_close(files->np_tile_output);
    tracer_fp_tile_close(files->epssyn_tile_output);
    tracer_fp_tile_close(files->epsic_tile_output);
    tracer_fp_tile_close(files->epsgamma_tile_output);
    tracer_fp_tile_close(files->epsnu_tile_output);

    if (files->ne_output->enabled) *files->ne_buffer_core = 0;
    if (files->np_output->enabled) *files->np_buffer_core = 0;
    if (files->epssyn_output->enabled) *files->epssyn_buffer_core = 0;
    if (files->epsic_output->enabled) *files->epsic_buffer_core = 0;
    if (files->epsgamma_output->enabled) *files->epsgamma_buffer_core = 0;
    if (files->epsnu_output->enabled) *files->epsnu_buffer_core = 0;

    tracer_fp_map_close(files->ne_output);
    tracer_fp_map_close(files->np_output);
    tracer_fp_map_close(files->epssyn_output);
    tracer_fp_map_close(files->epsic_output);
    tracer_fp_map_close(files->epsgamma_output);
    tracer_fp_map_close(files->epsnu_output);

    if (*files->ne_buffer_core) free2Darray(*files->ne_buffer_core, ntracer);
    if (*files->np_buffer_core) free2Darray(*files->np_buffer_core, ntracer);
    if (*files->epssyn_buffer_core) free2Darray(*files->epssyn_buffer_core, ntracer);
    if (*files->epsic_buffer_core) free2Darray(*files->epsic_buffer_core, ntracer);
    if (*files->epsgamma_buffer_core) free2Darray(*files->epsgamma_buffer_core, ntracer);
    if (*files->epsnu_buffer_core) free2Darray(*files->epsnu_buffer_core, ntracer);
    if (*files->ne_chunk_core) free2Darray(*files->ne_chunk_core, ntracer);
    if (*files->np_chunk_core) free2Darray(*files->np_chunk_core, ntracer);
    if (*files->epssyn_chunk_core) free2Darray(*files->epssyn_chunk_core, ntracer);
    if (*files->epsic_chunk_core) free2Darray(*files->epsic_chunk_core, ntracer);
    if (*files->epsgamma_chunk_core) free2Darray(*files->epsgamma_chunk_core, ntracer);
    if (*files->epsnu_chunk_core) free2Darray(*files->epsnu_chunk_core, ntracer);
    if (*files->bucketstats_top_fp) fclose(*files->bucketstats_top_fp);
    if (*files->bucketstats_rank_fp) fclose(*files->bucketstats_rank_fp);

    *files->ne_buffer_core = 0;
    *files->np_buffer_core = 0;
    *files->epssyn_buffer_core = 0;
    *files->epsic_buffer_core = 0;
    *files->epsgamma_buffer_core = 0;
    *files->epsnu_buffer_core = 0;
    *files->ne_chunk_core = 0;
    *files->np_chunk_core = 0;
    *files->epssyn_chunk_core = 0;
    *files->epsic_chunk_core = 0;
    *files->epsgamma_chunk_core = 0;
    *files->epsnu_chunk_core = 0;
    *files->bucketstats_top_fp = 0;
    *files->bucketstats_rank_fp = 0;
}

int tracer_fp_outputs_done(TracerFpOutputs *files,
                               const TracerFpOutputDone *cfg)
{
    if (files == 0 || cfg == 0) return 1;
    if (tracer_nsub_max > 0 && cfg->total_capped_tracer_snapshots > 0) {
        if (output_flagged_tracer_ids(cfg->output_dir, cfg->ntracer, cfg->mpi_rank,
                                      cfg->tracer_id_core, cfg->capped_flags) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to write capped tracer IDs\n");
            return 1;
        }
    }

    if (cfg->use_tile_output) {
        if (tracer_fp_tile_close(files->ne_tile_output) != 0 ||
            tracer_fp_tile_close(files->np_tile_output) != 0 ||
            tracer_fp_tile_close(files->epssyn_tile_output) != 0 ||
            (cfg->ic_enabled && tracer_fp_tile_close(files->epsic_tile_output) != 0) ||
            (cfg->gamma_enabled && tracer_fp_tile_close(files->epsgamma_tile_output) != 0) ||
            (cfg->neutrino_enabled && tracer_fp_tile_close(files->epsnu_tile_output) != 0)) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to finalize tile output files\n");
            return 1;
        }
    } else if (cfg->use_mapped_output) {
        if (tracer_fp_map_close(files->ne_output) != 0 ||
            tracer_fp_map_close(files->np_output) != 0 ||
            tracer_fp_map_close(files->epssyn_output) != 0 ||
            (cfg->ic_enabled && tracer_fp_map_close(files->epsic_output) != 0) ||
            (cfg->gamma_enabled && tracer_fp_map_close(files->epsgamma_output) != 0) ||
            (cfg->neutrino_enabled && tracer_fp_map_close(files->epsnu_output) != 0)) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to finalize mapped output files\n");
            return 1;
        }
        *files->ne_buffer_core = 0;
        *files->np_buffer_core = 0;
        *files->epssyn_buffer_core = 0;
        *files->epsic_buffer_core = 0;
        *files->epsgamma_buffer_core = 0;
        *files->epsnu_buffer_core = 0;
    } else {
        output_CRspectrum_buffer_node((char *)cfg->output_dir, cfg->ntracer, (cfg->nsnap + 1) * npe,
                                      -1, cfg->mpi_rank, *files->ne_buffer_core);
        if (cfg->write_crp_output) {
            output_CRspectrum_buffer_node((char *)cfg->output_dir, cfg->ntracer, (cfg->nsnap + 1) * np,
                                          1, cfg->mpi_rank, *files->np_buffer_core);
        }
        output_emissivity_core((char *)cfg->output_dir, (char *)"eSyn", cfg->ntracer,
                               cfg->nsnap * cfg->nfreq, cfg->mpi_rank, *files->epssyn_buffer_core);
        if (cfg->ic_enabled && *files->epsic_buffer_core != 0) {
            output_emissivity_core((char *)cfg->output_dir, (char *)"eIC", cfg->ntracer,
                                   cfg->nsnap * bins_IC, cfg->mpi_rank, *files->epsic_buffer_core);
        }
        if (cfg->gamma_enabled && *files->epsgamma_buffer_core != 0) {
            output_emissivity_core((char *)cfg->output_dir, (char *)"eGamma", cfg->ntracer,
                                   cfg->nsnap * bins_gamma, cfg->mpi_rank, *files->epsgamma_buffer_core);
        }
        if (cfg->neutrino_enabled && *files->epsnu_buffer_core != 0) {
            output_emissivity_core((char *)cfg->output_dir, (char *)"eNu", cfg->ntracer,
                                   cfg->nsnap * bins_nu, cfg->mpi_rank, *files->epsnu_buffer_core);
        }
    }

    return 0;
}

int tracer_fp_write_rank_log(const char *output_dir,
                                    int mpi_rank,
                                    int world_size,
                                    int mpi_size,
                                    long int global_ntracer,
                                    int local_ntracer,
                                    int nsnap,
                                    TracerFpInputMode input_mode,
                                    TracerFpFileOutputMode file_output_mode,
                                    TracerFpIntegrationMode integration_mode,
                                    TracerFpBackgroundMode background_mode,
                                    const TracerFpOpenmpInfo *omp_info,
                                    const TracerFpGpuTimes *times,
                                    double load_balance_ms,
                                    long long runtime_sum_nsub_local,
                                    long long runtime_target_nsub_local,
                                    double wall_ms,
                                    double max_wall_ms,
                                    double mean_wall_ms,
                                    double min_wall_ms)
{
    char path[MAX_LINE_LENGTH];
    FILE *fp;
    TracerFpStageTimes stage_times;
    const double wall_s = wall_ms * 1.0e-3;
    const double max_wall_s = max_wall_ms * 1.0e-3;
    const double local_throughput = (wall_s > 0.0) ? (double)local_ntracer / wall_s : 0.0;
    const double global_throughput = (max_wall_s > 0.0)
        ? (double)global_ntracer / max_wall_s : 0.0;

    if (output_dir == 0 || *output_dir == '\0' || times == 0) return -1;
    tracer_fp_stage_times_from_raw(times, load_balance_ms, &stage_times);

    snprintf(path, sizeof(path), "%s/timing_core%03d.tsv", output_dir, mpi_rank);
    fp = fopen(path, "w");
    if (fp == 0) return -1;

    fprintf(fp,
            "rank\tworld_size\tmpi_size\tglobal_ntracer\tlocal_ntracer\truntime_steps\tinput_mode\tfile_output_mode\tintegration_mode\tbackground_mode\tomp_compiled\tomp_use_param\tomp_param_threads\tomp_env_threads\tomp_requested_threads\tomp_max_threads\tomp_num_procs\tomp_dynamic_enabled\twall_ms\twall_ms_max\twall_ms_mean\twall_ms_min\tlocal_throughput_tracers_per_s\tglobal_throughput_tracers_per_s\tload_balance_ms\truntime_sum_nsub_local\truntime_target_nsub_local\tload_balancing_enabled\tstage_io_ms\tstage_scheduling_ms\tstage_host_ms\tstage_coeffprep_ms\tstage_secondary_ms\tstage_solve_ms\tstage_emission_ms\tstage_backend_ms\tstage_total_plus_lb_ms\tloss_prepass_ms\tnsub_estimate_ms\tbucket_build_ms\tbucketstats_top_ms\tgpu_group_build_ms\tpack_ms\tbucket_host_ms\tinterp_ms\tsnapshot_prep_ms\tcoeff_ms\tsecondary_ms\tsolve_ms\tsolve_alloc_ms\tsolve_rhs_ms\tsolve_tridiag_ms\tsynch_table_ms\tsynch_ms\tic_ms\tgamma_ms\tneutrino_ms\tcuda_setup_ms\tcuda_h2d_ms\tcuda_d2h_ms\tcuda_other_ms\tcuda_total_ms\tinput_read_ms\tbg_prepare_ms\ttracer_mass_ms\toutput_write_ms\toutput_sync_ms\tcheckpoint_ms\trestart_ms\tinput_read_calls\tinput_selected_runs\tinput_offset_runs\tinput_offset_min_run_len\tinput_offset_max_run_len\tinput_offset_max_gap\toutput_write_calls\toutput_sync_calls\tcheckpoint_calls\tgpu_pipeline_calls\tgpu_pipeline_cells\tgpu_pipeline_fp_steps\tgpu_pipeline_cell_steps\ttotal_staged_ms\n");
    fprintf(fp,
            "%d\t%d\t%d\t%ld\t%d\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t"
            "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%lld\t%lld\t%d\t"
            "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
            "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
            "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
            "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
            "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t"
            "%d\t%d\t%lld\t%d\t%d\t%d\t%d\t%d\t%d\t%lld\t%lld\t%lld\t%lld\t%.6f\n",
            mpi_rank, world_size, mpi_size, global_ntracer, local_ntracer, nsnap,
            tracer_fp_input_mode_name(input_mode),
            tracer_fp_file_output_mode_name(file_output_mode),
            tracer_fp_integration_mode_name(integration_mode),
            tracer_fp_background_mode_name(background_mode),
            (omp_info != 0) ? omp_info->compiled : 0,
            (omp_info != 0) ? omp_info->use_omp_param : 0,
            (omp_info != 0) ? omp_info->param_threads : 0,
            (omp_info != 0) ? omp_info->env_threads : 0,
            (omp_info != 0) ? omp_info->requested_threads : 1,
            (omp_info != 0) ? omp_info->max_threads : 1,
            (omp_info != 0) ? omp_info->num_procs : 1,
            (omp_info != 0) ? omp_info->dynamic_enabled : 0,
            wall_ms, max_wall_ms, mean_wall_ms, min_wall_ms,
            local_throughput, global_throughput,
            load_balance_ms,
            runtime_sum_nsub_local,
            runtime_target_nsub_local,
            load_balancing,
            stage_times.io_ms,
            stage_times.scheduling_ms,
            stage_times.host_ms,
            stage_times.coeffprep_ms,
            stage_times.secondary_ms,
            stage_times.solve_ms,
            stage_times.emission_ms,
            stage_times.backend_ms,
            stage_times.total_ms,
            times->loss_prepass_ms,
            times->nsub_estimate_ms,
            times->bucket_build_ms,
            times->bucketstats_top_ms,
            times->gpu_group_build_ms,
            times->pack_ms, times->bucket_host_ms, times->interp_ms,
            times->snapshot_prep_ms,
            times->coeff_ms, times->secondary_ms, times->solve_ms,
            times->solve_alloc_ms, times->solve_rhs_ms, times->solve_tridiag_ms,
            times->synch_table_ms, times->synch_ms, times->ic_ms, times->gamma_ms,
            times->neutrino_ms,
            times->cuda_setup_ms, times->cuda_h2d_ms, times->cuda_d2h_ms,
            times->cuda_other_ms,
            times->cuda_total_ms,
            times->input_read_ms, times->bg_prepare_ms, times->tracer_mass_ms,
            times->output_write_ms, times->output_sync_ms,
            times->checkpoint_ms, times->restart_ms,
            times->input_read_calls, times->input_selected_runs,
            times->input_offset_runs,
            times->input_offset_min_run_len,
            times->input_offset_max_run_len,
            times->input_offset_max_gap,
            times->output_write_calls, times->output_sync_calls,
            times->checkpoint_calls,
            times->gpu_pipeline_calls, times->gpu_pipeline_cells,
            times->gpu_pipeline_fp_steps, times->gpu_pipeline_cell_steps,
            times->total_ms);
    fclose(fp);
    return 0;
}

int tracer_fp_write_summary(const char *output_dir,
                                int world_size,
                                int mpi_size,
                                long int global_ntracer,
                                int runtime_steps,
                                TracerFpInputMode input_mode,
                                TracerFpFileOutputMode file_output_mode,
                                TracerFpIntegrationMode integration_mode,
                                TracerFpBackgroundMode background_mode,
                                const TracerFpOpenmpInfo *omp_info,
                                double load_balance_ms,
                                double wall_ms_min,
                                double wall_ms_mean,
                                double wall_ms_max,
                                double global_throughput)
{
    char path[MAX_LINE_LENGTH];
    FILE *fp;

    if (output_dir == 0 || *output_dir == '\0') return -1;
    snprintf(path, sizeof(path), "%s/run_summary.tsv", output_dir);
    fp = fopen(path, "w");
    if (fp == 0) return -1;

    fprintf(fp,
            "world_size\tmpi_size\tglobal_ntracer\truntime_steps\tinput_mode\tfile_output_mode\tintegration_mode\tbackground_mode\tomp_compiled\tomp_use_param\tomp_param_threads\tomp_env_threads\tomp_requested_threads\tomp_max_threads\tomp_num_procs\tomp_dynamic_enabled\tload_balance_ms\twall_ms_min\twall_ms_mean\twall_ms_max\tglobal_throughput_tracers_per_s\n");
    fprintf(fp,
            "%d\t%d\t%ld\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
            world_size, mpi_size, global_ntracer, runtime_steps,
            tracer_fp_input_mode_name(input_mode),
            tracer_fp_file_output_mode_name(file_output_mode),
            tracer_fp_integration_mode_name(integration_mode),
            tracer_fp_background_mode_name(background_mode),
            (omp_info != 0) ? omp_info->compiled : 0,
            (omp_info != 0) ? omp_info->use_omp_param : 0,
            (omp_info != 0) ? omp_info->param_threads : 0,
            (omp_info != 0) ? omp_info->env_threads : 0,
            (omp_info != 0) ? omp_info->requested_threads : 1,
            (omp_info != 0) ? omp_info->max_threads : 1,
            (omp_info != 0) ? omp_info->num_procs : 1,
            (omp_info != 0) ? omp_info->dynamic_enabled : 0,
            load_balance_ms,
            wall_ms_min, wall_ms_mean, wall_ms_max, global_throughput);
    fclose(fp);
    return 0;
}

static int write_dsa_params(const char *output_dir)
{
    char path[MAX_LINE_LENGTH];
    FILE *fp;
    TracerDsaInjectionMode injection_mode = TRACER_DSA_INJECTION_OFF;
    TracerDsaReaccMode reacc_mode = TRACER_DSA_REACC_OFF;
    const char *injection_name = DSAInjectionModeSpec;
    const char *reacc_name = DSAReaccModeSpec;

    if (output_dir == 0 || *output_dir == '\0') return -1;

    if (tracer_fp_parse_dsa_injection_mode(DSAInjectionModeSpec, &injection_mode) == 0) {
        injection_name = tracer_fp_dsa_injection_mode_name(injection_mode);
    }
    if (tracer_fp_parse_dsa_reacc_mode(DSAReaccModeSpec, &reacc_mode) == 0) {
        reacc_name = tracer_fp_dsa_reacc_mode_name(reacc_mode);
    }

    snprintf(path, sizeof(path), "%s/DSA_params.txt", output_dir);
    fp = fopen(path, "w");
    if (fp == 0) return -1;

    fprintf(fp, "# DSA parameters used by this run\n");
    fprintf(fp, "DSAInjectionMode = %s\n", injection_name);
    fprintf(fp, "DSAReaccMode = %s\n", reacc_name);
    fprintf(fp, "DSAEtaModelInitial = %s\n", dsa_model_name_from_id(DSAEtaModelInitial));
    fprintf(fp, "DSAEtaModelReacc = %s\n", dsa_model_name_from_id(DSAEtaModelReacc));
    fprintf(fp, "DSAChiP = %.17g\n", DSAChiP);
    fprintf(fp, "DSAChiE = %.17g\n", DSAChiE);
    fprintf(fp, "DSAKep = %.17g\n", DSAKep);
    fprintf(fp, "DSAPmaxPmc = %.17g\n", DSAPmaxPmc);
    fprintf(fp, "DSAPmaxEmc = %.17g\n", DSAPmaxEmc);
    fprintf(fp, "DSAMinMach = %.17g\n", DSAMinMach);
    fprintf(fp, "DSAGammaGas = %.17g\n", DSAGammaGas);
    fprintf(fp, "DSAXcrPminPmc = %.17g\n", DSAXcrPminPmc);
    fprintf(fp, "DSAReaccEtaCap = %.17g\n", DSAReaccEtaCap);
    fprintf(fp, "DSAShockRequired = %d\n", DSAShockRequired);
    fprintf(fp, "DSADebug = %d\n", DSADebug);

    fclose(fp);
    return 0;
}

int tracer_fp_write_nsub_log(const char *output_dir,
                                         int nsnap,
                                         const long long *estimate_nsub,
                                         const long long *runtime_nsub,
                                         const long long *estimate_target_nsub,
                                         const long long *runtime_target_nsub)
{
    char path[MAX_LINE_LENGTH];
    FILE *fp;
    int isnap;

    if (output_dir == 0 || *output_dir == '\0' || nsnap <= 0 ||
        estimate_nsub == 0 || runtime_nsub == 0) {
        return -1;
    }

    snprintf(path, sizeof(path), "%s/nsub_consistency.tsv", output_dir);
    fp = fopen(path, "w");
    if (fp == 0) return -1;

    fprintf(fp,
            "snapshot_index\testimate_sum_nsub\truntime_sum_nsub\tratio_runtime_over_estimate"
            "\testimate_target_sum_nsub\truntime_target_sum_nsub\tratio_runtime_target_over_estimate\n");
    for (isnap = 0; isnap < nsnap; isnap++) {
        const double nsub_ratio = (estimate_nsub[isnap] != 0)
            ? (double)runtime_nsub[isnap] / (double)estimate_nsub[isnap] : 0.0;
        const long long est_target =
            (estimate_target_nsub != 0) ? estimate_target_nsub[isnap] : 0;
        const long long run_target =
            (runtime_target_nsub != 0) ? runtime_target_nsub[isnap] : 0;
        const double target_ratio = (est_target != 0)
            ? (double)run_target / (double)est_target : 0.0;
        fprintf(fp, "%d\t%lld\t%lld\t%.12f\t%lld\t%lld\t%.12f\n",
                isnap + 1,
                estimate_nsub[isnap], runtime_nsub[isnap], nsub_ratio,
                est_target, run_target, target_ratio);
    }

    fclose(fp);
    return 0;
}

void tracer_fp_write_ids(const char *output_dir,
                                int ntracer,
                                int mpi_rank,
                                const int *tracer_id_core)
{
    if (output_dir == 0 || *output_dir == '\0' || tracer_id_core == 0) return;
    output_tracerid_node((char *)output_dir, ntracer, mpi_rank, (int *)tracer_id_core);
}

void tracer_fp_write_artifacts(const TracerFpRunInfo *cfg,
                                   const long long *estimate_nsub,
                                   const long long *runtime_nsub,
                                   const long long *estimate_target_nsub,
                                   const long long *runtime_target_nsub)
{
    const char *output_dir;
    const double global_throughput =
        (cfg->wall_ms_max > 0.0) ? (1.0e3 * (double)cfg->global_ntracer / cfg->wall_ms_max) : 0.0;

    if (cfg == 0) return;
    output_dir = cfg->output_dir;
    if (output_dir == 0 || *output_dir == '\0') return;

    if (cfg->log_root) {
        printf("  output              : writing run summary/timing logs to %s\n", output_dir);
        fflush(stdout);
    }
    if (cfg->mpi_rank == 0) {
        if (tracer_fp_write_summary(output_dir, cfg->world_size, cfg->mpi_size,
                                        cfg->global_ntracer, cfg->nsnap,
                                        cfg->input_mode, cfg->file_output_mode,
                                        cfg->integration_mode, cfg->background_mode, cfg->omp_info,
                                        cfg->load_balance_ms,
                                        cfg->wall_ms_min, cfg->wall_ms_mean, cfg->wall_ms_max,
                                        global_throughput) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to write run summary log\n");
        }
        if (estimate_nsub != 0 &&
            tracer_fp_write_nsub_log(output_dir, cfg->nsnap,
                                                 estimate_nsub,
                                                 runtime_nsub,
                                                 estimate_target_nsub,
                                                 runtime_target_nsub) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to write nsub consistency log\n");
        }
        if (write_dsa_params(output_dir) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to write DSA parameter log\n");
        }
    }
    if (tracer_fp_write_rank_log(output_dir, cfg->mpi_rank, cfg->world_size, cfg->mpi_size,
                                        cfg->global_ntracer,
                                        cfg->local_ntracer, cfg->nsnap,
                                        cfg->input_mode, cfg->file_output_mode,
                                        cfg->integration_mode, cfg->background_mode,
                                        cfg->omp_info, cfg->times, cfg->load_balance_ms,
                                        cfg->runtime_sum_nsub_local,
                                        cfg->runtime_target_nsub_local,
                                        cfg->wall_ms, cfg->wall_ms_max, cfg->wall_ms_mean,
                                        cfg->wall_ms_min) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to write timing log for rank %d\n",
                cfg->mpi_rank);
    }
}

void populate_output_buffers(double **ne_buffer,
                             double **np_buffer,
                             int nlocal,
                             int snapshot_slot,
                             const double *cre_state,
                             const double *crp_state,
                             const double *rho_gcc,
                             const double *tracer_mass,
                             int output_per_cc)
{
    int itr, j;

    if (ne_buffer == 0 || cre_state == 0) return;
    for (itr = 0; itr < nlocal; itr++) {
        const size_t src_p_off = (size_t)itr * (size_t)np;
        const size_t src_e_off = (size_t)itr * (size_t)npe;
        const size_t dst_p_off = (size_t)snapshot_slot * (size_t)np;
        const size_t dst_e_off = (size_t)snapshot_slot * (size_t)npe;
        double factor = output_volume_factor(rho_gcc, tracer_mass, itr, output_per_cc);
        if (np_buffer != 0 && crp_state != 0) {
            for (j = 0; j < np; j++) {
                np_buffer[itr][dst_p_off + (size_t)j] =
                    crp_state[src_p_off + (size_t)j] * factor;
            }
        }
        for (j = 0; j < npe; j++) {
            ne_buffer[itr][dst_e_off + (size_t)j] =
                cre_state[src_e_off + (size_t)j] * factor;
        }
    }
}

void populate_emission_buffer(double **buffer,
                              int nlocal,
                              int snapshot_index,
                              int nbin,
                              const double *emission_cell_major,
                              const double *rho_gcc,
                              const double *tracer_mass,
                              int output_per_cc)
{
    int itr, j;

    if (buffer == 0 || emission_cell_major == 0) return;
    for (itr = 0; itr < nlocal; itr++) {
        const size_t src_off = (size_t)itr * (size_t)nbin;
        const size_t dst_off = (size_t)snapshot_index * (size_t)nbin;
        double factor = output_volume_factor(rho_gcc, tracer_mass, itr, output_per_cc);
        for (j = 0; j < nbin; j++) {
            buffer[itr][dst_off + (size_t)j] =
                emission_cell_major[src_off + (size_t)j] * factor;
        }
    }
}

int output_flagged_tracer_ids(const char *strout,
                              int nlocal,
                              int mpi_rank,
                              const int *tracer_ids,
                              const unsigned char *flags)
{
    char str[MAX_LINE_LENGTH];
    FILE *fp;
    int itr;

    if (strout == 0 || tracer_ids == 0 || flags == 0) return -1;

    snprintf(str, sizeof(str), "%s/capped_tracerid_core%02d.txt", strout, mpi_rank);
    fp = fopen(str, "w");
    if (fp == 0) return -1;
    for (itr = 0; itr < nlocal; itr++) {
        if (flags[itr]) {
            fprintf(fp, "%d\n", tracer_ids[itr]);
        }
    }
    fclose(fp);
    return 0;
}
