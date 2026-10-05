#ifndef INCLUDED_decomposition_metadata_h_
#define INCLUDED_decomposition_metadata_h_

#include <mpi.h>

/*
 * Structured decomposition metadata for meshFP.
 *
 * Current status:
 *   - 1-D x-slab metadata is preserved for the current runtime path
 *   - 3-D box metadata is available for future block decomposition work
 *   - existing linear cell-range logic remains unchanged elsewhere
 *
 * Purpose:
 *   - centralise decomposition-related bookkeeping in one place
 *   - provide a stable API for future pencil/block implementations
 */

typedef enum {
    DECOMP_LAYOUT_SLAB_X = 0,
    DECOMP_LAYOUT_PENCIL_XY = 1,
    DECOMP_LAYOUT_BLOCK_XYZ = 2
} DecompLayout;

typedef enum {
    DECOMP_FACE_XM = 0,
    DECOMP_FACE_XP = 1,
    DECOMP_FACE_YM = 2,
    DECOMP_FACE_YP = 3,
    DECOMP_FACE_ZM = 4,
    DECOMP_FACE_ZP = 5
} DecompFace;

typedef struct {
    DecompFace face;
    int rank;
    int exists;

    int proc_coords[3];

    /* Neighbor-owned global box [start, end). */
    int start_x;
    int end_x;
    int start_y;
    int end_y;
    int start_z;
    int end_z;

    int local_nx;
    int local_ny;
    int local_nz;
} DecompNeighborMetadata;

typedef struct {
    DecompLayout layout;

    int rank;
    int size;

    int global_nx;
    int global_ny;
    int global_nz;

    /* Process-grid metadata.
     * For the current slab-x implementation:
     *   proc_dims   = {size, 1, 1}
     *   proc_coords = {rank, 0, 0}
     */
    int proc_dims[3];
    int proc_coords[3];

    /* Local domain [start, end) in global cell indices. */
    int start_x;
    int end_x;
    int start_y;
    int end_y;
    int start_z;
    int end_z;

    int local_nx;
    int local_ny;
    int local_nz;

    /* Useful for compatibility with the current x-slab storage model. */
    long int local_cell_start;
    long int local_cell_count;
    int is_contiguous_linear;

    /* 6-face neighbor ranks. MPI_PROC_NULL means physical boundary. */
    int nbr_xm;
    int nbr_xp;
    int nbr_ym;
    int nbr_yp;
    int nbr_zm;
    int nbr_zp;

    DecompNeighborMetadata neighbors[6];
} DecompositionMetadata;

/*
 * Structured view of a contiguous linear cell-ID block in C-order storage:
 *   linear = ix*(Ny*Nz) + iy*Nz + iz
 *
 * A balanced ID partition does not, in general, map to a single axis-aligned
 * box. This descriptor makes that explicit and records when the block is an
 * exact x-slab box that can be promoted to DecompositionMetadata.
 */
typedef struct {
    int global_nx;
    int global_ny;
    int global_nz;

    long int linear_start;
    long int linear_count;
    long int linear_end;

    int start_x;
    int start_y;
    int start_z;

    /* Coordinate of the first cell after the block in linear scan order.
     * When linear_end == global_nx*global_ny*global_nz, end_x == global_nx
     * and end_y == end_z == 0.
     */
    int end_x;
    int end_y;
    int end_z;

    int yz_plane_size;
    int start_plane_aligned;
    int end_plane_aligned;

    /* True only when this local block is one exact x-slab box. */
    int is_exact_x_slab_box;

    /* True only when every rank in the balanced contiguous-ID partition is
     * also an exact x-slab box.
     */
    int all_ranks_exact_x_slab;

    int slab_start_x;
    int slab_end_x;
    int slab_nx;
} DecompLinearBlockMetadata;

/*
 * 1-D balanced range partition of n_global items over MPI ranks.
 *
 * Ranks [0, remainder) receive one extra item.
 */
void decomp_partition_1d(int n_global, int rank, int size,
                         int *start_out, int *count_out);

/*
 * Choose a 3-D process grid. Entries already > 0 are treated as fixed, while
 * zero entries are filled by MPI_Dims_create().
 *
 * Returns 0 on success, non-zero on invalid input.
 */
int decomp_choose_proc_dims_3d(int size, int proc_dims[3]);

/*
 * Choose a 3-D process grid for a given global mesh.
 *
 * The selected (px, py, pz) satisfies px*py*pz = size and tries to keep local
 * boxes as close to isotropic as possible. Entries already > 0 are treated as
 * fixed, while zero entries are searched automatically.
 *
 * Returns 0 on success, non-zero on invalid input or if no valid factorisation
 * exists.
 */
int decomp_choose_proc_dims_3d_for_box(int global_nx, int global_ny, int global_nz,
                                       int size, int proc_dims[3]);

/*
 * Initialise metadata for a 1-D x-slab decomposition.
 *
 * Returns 0 on success, non-zero on invalid input.
 */
int decomp_init_slab_x(int global_nx, int global_ny, int global_nz,
                       int rank, int size,
                       DecompositionMetadata *meta);

/*
 * Initialise metadata for a 3-D axis-aligned box decomposition.
 *
 * Ranks are mapped as:
 *   rank = px + proc_dims[0] * (py + proc_dims[1] * pz)
 *
 * Returns 0 on success, non-zero on invalid input.
 */
int decomp_init_block_xyz(int global_nx, int global_ny, int global_nz,
                          int proc_dims_x, int proc_dims_y, int proc_dims_z,
                          int rank, int size,
                          DecompositionMetadata *meta);

/*
 * Convenience wrapper that chooses a process grid automatically.
 */
int decomp_init_block_xyz_auto(int global_nx, int global_ny, int global_nz,
                               int rank, int size,
                               DecompositionMetadata *meta);

/*
 * Describe one balanced contiguous linear block.
 */
int decomp_describe_linear_block(int global_nx, int global_ny, int global_nz,
                                 long int linear_start, long int linear_count,
                                 int rank, int size,
                                 DecompLinearBlockMetadata *desc);

/*
 * Promote a balanced contiguous linear block to exact x-slab metadata.
 *
 * Returns 0 only when this rank's block, and every rank's block in the same
 * balanced contiguous-ID partition, is an exact x-slab box.
 */
int decomp_init_slab_x_from_linear_block(int global_nx, int global_ny, int global_nz,
                                         long int linear_start, long int linear_count,
                                         int rank, int size,
                                         DecompositionMetadata *meta);

/*
 * Return the rank owning the given process-grid coordinate, or MPI_PROC_NULL
 * if the coordinate lies outside the process grid.
 */
int decomp_rank_from_proc_coords(const DecompositionMetadata *meta,
                                 int px, int py, int pz);

/*
 * Face-name helper for logging and debugging.
 */
const char *decomp_face_name(DecompFace face);

/*
 * Return the global linear index for C-order storage:
 *   idx = ix*(Ny*Nz) + iy*Nz + iz
 */
long int decomp_global_index(const DecompositionMetadata *meta,
                             int ix, int iy, int iz);

/*
 * Return the local linear index inside the local subdomain using local
 * coordinates:
 *   idx_local = ix_local*(local_ny*local_nz) + iy_local*local_nz + iz_local
 */
long int decomp_local_index(const DecompositionMetadata *meta,
                            int ix_local, int iy_local, int iz_local);

/*
 * True if the given global cell belongs to this rank's local subdomain.
 */
int decomp_contains_global(const DecompositionMetadata *meta,
                           int ix, int iy, int iz);

/*
 * Convert global coordinates to local coordinates.
 *
 * Returns 1 on success, 0 if the cell is outside this rank's local subdomain.
 */
int decomp_global_to_local(const DecompositionMetadata *meta,
                           int ix, int iy, int iz,
                           int *ix_local, int *iy_local, int *iz_local);

#endif /* INCLUDED_decomposition_metadata_h_ */
