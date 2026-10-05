#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <hdf5.h>
#include <mpi.h>

#include "params.h"
#include "read_grid_hdf5.h"

/* -------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------*/

/* Read a scalar double attribute from the root group of an open HDF5 file. */
static int read_attr_double(hid_t file_id, const char *name, double *out)
{
    hid_t root = H5Gopen(file_id, "/", H5P_DEFAULT);
    hid_t attr = H5Aopen(root, name, H5P_DEFAULT);
    if (attr < 0) {
        fprintf(stderr, "read_attr_double: attribute '%s' not found\n", name);
        H5Gclose(root);
        return FAIL;
    }
    H5Aread(attr, H5T_NATIVE_DOUBLE, out);
    H5Aclose(attr);
    H5Gclose(root);
    return SUCCESS;
}

static int read_attr_int(hid_t file_id, const char *name, int *out)
{
    hid_t root = H5Gopen(file_id, "/", H5P_DEFAULT);
    hid_t attr = H5Aopen(root, name, H5P_DEFAULT);
    if (attr < 0) {
        fprintf(stderr, "read_attr_int: attribute '%s' not found\n", name);
        H5Gclose(root);
        return FAIL;
    }
    H5Aread(attr, H5T_NATIVE_INT, out);
    H5Aclose(attr);
    H5Gclose(root);
    return SUCCESS;
}

/* Read a flat 3D dataset (all elements) into a pre-allocated double buffer. */
static int read_dataset(hid_t file_id, const char *name,
                        long int N_expected, double *buf)
{
    hid_t dset = H5Dopen(file_id, name, H5P_DEFAULT);
    if (dset < 0) {
        fprintf(stderr, "read_dataset: cannot open '%s'\n", name);
        return FAIL;
    }
    hid_t  space   = H5Dget_space(dset);
    hsize_t N_file = (hsize_t)H5Sget_simple_extent_npoints(space);
    H5Sclose(space);
    if ((long int)N_file != N_expected) {
        fprintf(stderr, "read_dataset: '%s' has %lld elements, expected %ld\n",
                name, (long long)N_file, N_expected);
        H5Dclose(dset);
        return FAIL;
    }
    herr_t st = H5Dread(dset, H5T_NATIVE_DOUBLE,
                         H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset);
    if (st < 0) {
        fprintf(stderr, "read_dataset: read error for '%s'\n", name);
        return FAIL;
    }
    return SUCCESS;
}

/* Read the scalar (0-d) L_turb dataset from the auxiliary file. */
static int read_scalar_dataset(hid_t file_id, const char *name, double *out)
{
    hid_t dset = H5Dopen(file_id, name, H5P_DEFAULT);
    if (dset < 0) {
        fprintf(stderr, "read_scalar_dataset: cannot open '%s'\n", name);
        return FAIL;
    }
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out);
    H5Dclose(dset);
    return SUCCESS;
}

static int init_rank_decomp_like(const DecompositionMetadata *tmpl,
                                 int rank, int size,
                                 DecompositionMetadata *out)
{
    if (tmpl == NULL || out == NULL) return FAIL;

    switch (tmpl->layout) {
        case DECOMP_LAYOUT_SLAB_X:
            return (decomp_init_slab_x(tmpl->global_nx, tmpl->global_ny, tmpl->global_nz,
                                       rank, size, out) == 0) ? SUCCESS : FAIL;

        case DECOMP_LAYOUT_BLOCK_XYZ:
            return (decomp_init_block_xyz(tmpl->global_nx, tmpl->global_ny, tmpl->global_nz,
                                          tmpl->proc_dims[0],
                                          tmpl->proc_dims[1],
                                          tmpl->proc_dims[2],
                                          rank, size, out) == 0) ? SUCCESS : FAIL;

        default:
            return FAIL;
    }
}

static void pack_decomp_box_fields(const DecompositionMetadata *decomp,
                                   const double *send_buf,
                                   double *packed_buf)
{
    for (int ix = decomp->start_x; ix < decomp->end_x; ix++)
    for (int iy = decomp->start_y; iy < decomp->end_y; iy++)
    for (int iz = decomp->start_z; iz < decomp->end_z; iz++) {
        int ix_local = ix - decomp->start_x;
        int iy_local = iy - decomp->start_y;
        int iz_local = iz - decomp->start_z;
        long int cl = decomp_local_index(decomp, ix_local, iy_local, iz_local);
        long int cg = decomp_global_index(decomp, ix, iy, iz);
        memcpy(packed_buf + cl * NUM_GRID_FIELD,
               send_buf    + cg * NUM_GRID_FIELD,
               NUM_GRID_FIELD * sizeof(double));
    }
}

/* -------------------------------------------------------------------------
 * grid_cell_range
 * -------------------------------------------------------------------------*/
void grid_cell_range(long int N_total, int mpi_rank, int mpi_size,
                     long int *cell_start_out, long int *N_cells_out)
{
    long int base = N_total / mpi_size;
    long int rem  = N_total % mpi_size;
    *N_cells_out    = base + (mpi_rank < (int)rem ? 1 : 0);
    *cell_start_out = (long int)mpi_rank * base
                    + (mpi_rank < (int)rem ? mpi_rank : rem);
}

/* -------------------------------------------------------------------------
 * read_grid_metadata
 * -------------------------------------------------------------------------*/
int read_grid_metadata(int mpi_rank, MPI_Comm comm,
                       const char *filename, GridMeta *meta)
{
    if (mpi_rank == 0) {
        hid_t fid = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fid < 0) {
            fprintf(stderr, "read_grid_metadata: cannot open %s\n", filename);
            return FAIL;
        }
        if (read_attr_int(fid, "Nx", &meta->Nx)        != SUCCESS ||
            read_attr_int(fid, "Ny", &meta->Ny)        != SUCCESS ||
            read_attr_int(fid, "Nz", &meta->Nz)        != SUCCESS ||
            read_attr_double(fid, "dx_cm",      &meta->dx_cm)   != SUCCESS ||
            read_attr_double(fid, "dy_cm",      &meta->dy_cm)   != SUCCESS ||
            read_attr_double(fid, "dz_cm",      &meta->dz_cm)   != SUCCESS ||
            read_attr_double(fid, "left_x_Mpc", &meta->left_x)  != SUCCESS ||
            read_attr_double(fid, "left_y_Mpc", &meta->left_y)  != SUCCESS ||
            read_attr_double(fid, "left_z_Mpc", &meta->left_z)  != SUCCESS) {
            H5Fclose(fid);
            return FAIL;
        }
        H5Fclose(fid);
        printf("Grid metadata: Nx=%d Ny=%d Nz=%d  dx=%.3e cm\n",
               meta->Nx, meta->Ny, meta->Nz, meta->dx_cm);
    }

    /* Broadcast to all ranks */
    MPI_Bcast(&meta->Nx,     1, MPI_INT,    0, comm);
    MPI_Bcast(&meta->Ny,     1, MPI_INT,    0, comm);
    MPI_Bcast(&meta->Nz,     1, MPI_INT,    0, comm);
    MPI_Bcast(&meta->dx_cm,  1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&meta->dy_cm,  1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&meta->dz_cm,  1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&meta->left_x, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&meta->left_y, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&meta->left_z, 1, MPI_DOUBLE, 0, comm);

    return SUCCESS;
}

/* -------------------------------------------------------------------------
 * read_grid_snapshot
 * -------------------------------------------------------------------------*/
int read_grid_snapshot_decomp(int mpi_rank, MPI_Comm comm,
                              const GridMeta *meta,
                              const DecompositionMetadata *decomp,
                              const char *grid_file, const char *aux_file,
                              double *data_node,
                              double *z_out)
{
    const double MPC_CM = 3.0857e24;
    int mpi_size;

    long int N_total = (long int)meta->Nx * meta->Ny * meta->Nz;
    int      Ny      = meta->Ny;
    int      Nz      = meta->Nz;
    double   dx_Mpc  = meta->dx_cm / MPC_CM;
    double   dy_Mpc  = meta->dy_cm / MPC_CM;
    double   dz_Mpc  = meta->dz_cm / MPC_CM;

    double *send_buf   = NULL;
    double *pack_buf   = NULL;
    double  redshift   = 0.0;
    double  L_turb_Mpc = 0.0;

    /* Buffers for raw field arrays (rank 0 only) */
    double *buf_density = NULL, *buf_temp    = NULL;
    double *buf_Bx      = NULL, *buf_By      = NULL, *buf_Bz = NULL;
    double *buf_vx      = NULL, *buf_vy      = NULL, *buf_vz = NULL;
    double *buf_divv    = NULL, *buf_curlv   = NULL;

    if (meta == NULL || decomp == NULL || data_node == NULL) return FAIL;

    MPI_Comm_size(comm, &mpi_size);

    if (mpi_rank == 0) {

        buf_density = (double *)malloc(N_total * sizeof(double));
        buf_temp    = (double *)malloc(N_total * sizeof(double));
        buf_Bx      = (double *)malloc(N_total * sizeof(double));
        buf_By      = (double *)malloc(N_total * sizeof(double));
        buf_Bz      = (double *)malloc(N_total * sizeof(double));
        buf_vx      = (double *)malloc(N_total * sizeof(double));
        buf_vy      = (double *)malloc(N_total * sizeof(double));
        buf_vz      = (double *)malloc(N_total * sizeof(double));
        buf_divv    = (double *)malloc(N_total * sizeof(double));
        buf_curlv   = (double *)malloc(N_total * sizeof(double));

        if (!buf_density || !buf_temp  || !buf_Bx || !buf_By || !buf_Bz ||
            !buf_vx      || !buf_vy    || !buf_vz ||
            !buf_divv    || !buf_curlv) {
            fprintf(stderr, "read_grid_snapshot: malloc failed for raw buffers\n");
            return FAIL;
        }

        /* --- Read main fields --- */
        hid_t fmain = H5Fopen(grid_file, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (fmain < 0) {
            fprintf(stderr, "read_grid_snapshot: cannot open %s\n", grid_file);
            return FAIL;
        }
        read_attr_double(fmain, "Redshift", &redshift);

        if (read_dataset(fmain, "density",          N_total, buf_density) != SUCCESS ||
            read_dataset(fmain, "temperature",      N_total, buf_temp)    != SUCCESS ||
            read_dataset(fmain, "magnetic_field_x", N_total, buf_Bx)      != SUCCESS ||
            read_dataset(fmain, "magnetic_field_y", N_total, buf_By)      != SUCCESS ||
            read_dataset(fmain, "magnetic_field_z", N_total, buf_Bz)      != SUCCESS ||
            read_dataset(fmain, "velocity_x",       N_total, buf_vx)      != SUCCESS ||
            read_dataset(fmain, "velocity_y",       N_total, buf_vy)      != SUCCESS ||
            read_dataset(fmain, "velocity_z",       N_total, buf_vz)      != SUCCESS) {
            H5Fclose(fmain);
            return FAIL;
        }
        H5Fclose(fmain);

        /* --- Read auxiliary turbulence fields --- */
        hid_t faux = H5Fopen(aux_file, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (faux < 0) {
            fprintf(stderr, "read_grid_snapshot: cannot open %s\n", aux_file);
            return FAIL;
        }
        if (read_dataset(faux, "div_v",      N_total, buf_divv)  != SUCCESS ||
            read_dataset(faux, "curl_v_mag", N_total, buf_curlv) != SUCCESS ||
            read_scalar_dataset(faux, "L_turb", &L_turb_Mpc)    != SUCCESS) {
            H5Fclose(faux);
            return FAIL;
        }
        H5Fclose(faux);

        /* --- Build send_buf [N_total × NUM_GRID_FIELD], cell-major --- */
        send_buf = (double *)malloc(N_total * NUM_GRID_FIELD * sizeof(double));
        if (!send_buf) {
            fprintf(stderr, "read_grid_snapshot: malloc failed for send_buf\n");
            return FAIL;
        }

        for (long int c = 0; c < N_total; c++) {
            int ix = (int)(c / ((long int)Ny * Nz));
            int iy = (int)((c / Nz) % Ny);
            int iz = (int)(c % Nz);
            double *f = send_buf + c * NUM_GRID_FIELD;

            f[GRID_FIELD_X]       = meta->left_x + (ix + 0.5) * dx_Mpc;
            f[GRID_FIELD_Y]       = meta->left_y + (iy + 0.5) * dy_Mpc;
            f[GRID_FIELD_Z]       = meta->left_z + (iz + 0.5) * dz_Mpc;
            f[GRID_FIELD_VX]      = buf_vx[c];
            f[GRID_FIELD_VY]      = buf_vy[c];
            f[GRID_FIELD_VZ]      = buf_vz[c];
            f[GRID_FIELD_DENSITY] = buf_density[c];
            f[GRID_FIELD_TEMP]    = buf_temp[c];
            f[GRID_FIELD_BX]      = buf_Bx[c];
            f[GRID_FIELD_BY]      = buf_By[c];
            f[GRID_FIELD_BZ]      = buf_Bz[c];
            f[GRID_FIELD_BMAG]    = sqrt(buf_Bx[c]*buf_Bx[c]
                                       + buf_By[c]*buf_By[c]
                                       + buf_Bz[c]*buf_Bz[c]);
            f[GRID_FIELD_DIVV]    = buf_divv[c];
            f[GRID_FIELD_CURLV]   = buf_curlv[c];
            f[GRID_FIELD_LTURB]   = L_turb_Mpc;
        }

        free(buf_density); free(buf_temp);
        free(buf_Bx); free(buf_By); free(buf_Bz);
        free(buf_vx); free(buf_vy); free(buf_vz);
        free(buf_divv); free(buf_curlv);

        for (int r = 0; r < mpi_size; r++) {
            DecompositionMetadata rank_decomp;
            long int pack_count;

            if (init_rank_decomp_like(decomp, r, mpi_size, &rank_decomp) != SUCCESS) {
                fprintf(stderr,
                        "read_grid_snapshot_decomp: cannot reconstruct metadata for rank %d\n",
                        r);
                free(send_buf);
                return FAIL;
            }

            pack_count = rank_decomp.local_cell_count * NUM_GRID_FIELD;
            if (r == 0) {
                pack_decomp_box_fields(&rank_decomp, send_buf, data_node);
            } else {
                pack_buf = (double *)malloc((size_t)pack_count * sizeof(double));
                if (!pack_buf) {
                    fprintf(stderr,
                            "read_grid_snapshot_decomp: malloc failed for rank pack buffer\n");
                    free(send_buf);
                    return FAIL;
                }
                pack_decomp_box_fields(&rank_decomp, send_buf, pack_buf);
                MPI_Send(pack_buf, (int)pack_count, MPI_DOUBLE, r, 0, comm);
                free(pack_buf);
                pack_buf = NULL;
            }
        }
    }

    /* Broadcast scalars */
    MPI_Bcast(&redshift,   1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&L_turb_Mpc, 1, MPI_DOUBLE, 0, comm);
    if (z_out) *z_out = redshift;

    if (mpi_rank != 0) {
        int recv_count = (int)(decomp->local_cell_count * NUM_GRID_FIELD);
        MPI_Recv(data_node, recv_count, MPI_DOUBLE, 0, 0, comm, MPI_STATUS_IGNORE);
    }

    if (mpi_rank == 0) {
        free(send_buf);
    }

    return SUCCESS;
}

int read_grid_snapshot(int mpi_rank, int mpi_size, MPI_Comm comm,
                       const GridMeta *meta,
                       const char *grid_file, const char *aux_file,
                       double *data_node,
                       double *z_out)
{
    DecompositionMetadata decomp_slab;

    if (decomp_init_slab_x(meta->Nx, meta->Ny, meta->Nz,
                           mpi_rank, mpi_size, &decomp_slab) != 0) {
        return FAIL;
    }

    return read_grid_snapshot_decomp(mpi_rank, comm, meta, &decomp_slab,
                                     grid_file, aux_file, data_node, z_out);
}
