#ifndef INCLUDED_read_grid_hdf5_h_
#define INCLUDED_read_grid_hdf5_h_

#include <mpi.h>

#include "decomposition_metadata.h"

/* -------------------------------------------------------------------------
 * Grid input mode flags  (used by params.h: grid_mode)
 * -------------------------------------------------------------------------*/
#define GRID_MODE_TRACER   0   /* original tracer-particle mode (no grid reader) */
#define GRID_MODE_YT       1   /* yt-exported uniform HDF5 (preprocess_grid.py) */
#define GRID_MODE_ENZO_RAW 2   /* raw Enzo DD* (not yet implemented)            */

/* -------------------------------------------------------------------------
 * Per-cell field indices
 *
 * data_node layout: data_node[cell_local * NUM_GRID_FIELD + field_idx]
 * where cell_local is in [0, N_cells_node).
 *
 * Matches the tracer data_slc layout so the same FP coefficients code works
 * in both modes (indices 6-12 are identical to the tracer convention).
 * -------------------------------------------------------------------------*/
#define GRID_FIELD_X        0   /* cell centre x           [Mpc]      */
#define GRID_FIELD_Y        1   /* cell centre y           [Mpc]      */
#define GRID_FIELD_Z        2   /* cell centre z           [Mpc]      */
#define GRID_FIELD_VX       3   /* velocity_x              [cm/s]     */
#define GRID_FIELD_VY       4   /* velocity_y              [cm/s]     */
#define GRID_FIELD_VZ       5   /* velocity_z              [cm/s]     */
#define GRID_FIELD_DENSITY  6   /* density                 [g/cm^3]   */
#define GRID_FIELD_TEMP     7   /* temperature             [K]        */
#define GRID_FIELD_BX       8   /* magnetic_field_x        [G]        */
#define GRID_FIELD_BY       9   /* magnetic_field_y        [G]        */
#define GRID_FIELD_BZ      10   /* magnetic_field_z        [G]        */
#define GRID_FIELD_BMAG    11   /* |B| = sqrt(Bx^2+By^2+Bz^2) [G]   */
#define GRID_FIELD_DIVV    12   /* div(v)   = ∂vx/∂x+…    [1/s]     */
#define GRID_FIELD_CURLV   13   /* |curl(v)|               [1/s]     */
#define GRID_FIELD_LTURB   14   /* turbulent scale L_turb  [Mpc]     */
#define NUM_GRID_FIELD     15

/* -------------------------------------------------------------------------
 * Grid metadata
 * -------------------------------------------------------------------------*/
typedef struct {
    int    Nx, Ny, Nz;            /* grid dimensions                */
    double dx_cm, dy_cm, dz_cm;   /* cell widths [cm]               */
    double left_x, left_y, left_z;/* domain left edge [Mpc]         */
} GridMeta;

/* -------------------------------------------------------------------------
 * Public interface
 * -------------------------------------------------------------------------*/

/*
 * Read grid dimensions and domain geometry from the first snapshot.
 * Must be called on ALL ranks (rank 0 reads; result is broadcast).
 *
 * filename : path to a main grid HDF5 file (e.g. grid_0001.h5)
 */
int read_grid_metadata(int mpi_rank, MPI_Comm comm,
                       const char *filename, GridMeta *meta);

/*
 * Compute the cell range assigned to a given MPI rank.
 * Cells are distributed round-robin: ranks < (N_total % mpi_size)
 * receive one extra cell.
 *
 * cell_start_out  : first linear cell index for this rank
 * N_cells_out     : number of cells for this rank
 */
void grid_cell_range(long int N_total, int mpi_rank, int mpi_size,
                     long int *cell_start_out, long int *N_cells_out);

/*
 * Read one snapshot and scatter per-cell data across MPI ranks.
 *
 * data_node must be pre-allocated with N_cells_node * NUM_GRID_FIELD doubles,
 * where N_cells_node comes from grid_cell_range().
 *
 * grid_file : path to the main HDF5 for this snapshot
 * aux_file  : path to the auxiliary turbulence HDF5 for this snapshot
 * z_out     : (optional, may be NULL) receives the snapshot redshift
 */
int read_grid_snapshot(int mpi_rank, int mpi_size, MPI_Comm comm,
                       const GridMeta *meta,
                       const char *grid_file, const char *aux_file,
                       double *data_node,
                       double *z_out);

/*
 * Read one snapshot and distribute per-cell data according to the given
 * decomposition metadata.
 *
 * data_node must be pre-allocated with
 *   decomp->local_cell_count * NUM_GRID_FIELD
 * doubles, laid out in local box order:
 *   ix_local outermost, then iy_local, then iz_local.
 *
 * For slab-x layouts this is compatible with the existing linear storage.
 * For block layouts this packs the owned 3-D sub-box explicitly.
 */
int read_grid_snapshot_decomp(int mpi_rank, MPI_Comm comm,
                              const GridMeta *meta,
                              const DecompositionMetadata *decomp,
                              const char *grid_file, const char *aux_file,
                              double *data_node,
                              double *z_out);

#endif /* INCLUDED_read_grid_hdf5_h_ */
