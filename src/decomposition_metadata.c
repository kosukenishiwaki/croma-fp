#include "decomposition_metadata.h"

#include <math.h>
#include <string.h>

static void decomp_linear_to_coords(int global_ny, int global_nz,
                                    long int linear,
                                    int *ix, int *iy, int *iz)
{
    long int yz_plane = (long int)global_ny * (long int)global_nz;
    long int rem;

    if (yz_plane <= 0) {
        if (ix != 0) *ix = 0;
        if (iy != 0) *iy = 0;
        if (iz != 0) *iz = 0;
        return;
    }

    if (ix != 0) *ix = (int)(linear / yz_plane);
    rem = linear % yz_plane;
    if (iy != 0) *iy = (int)(rem / (long int)global_nz);
    if (iz != 0) *iz = (int)(rem % (long int)global_nz);
}

static int decomp_rank_from_dims_coords(int proc_dims_x, int proc_dims_y, int proc_dims_z,
                                        int px, int py, int pz)
{
    if (px < 0 || px >= proc_dims_x) return MPI_PROC_NULL;
    if (py < 0 || py >= proc_dims_y) return MPI_PROC_NULL;
    if (pz < 0 || pz >= proc_dims_z) return MPI_PROC_NULL;

    return px + proc_dims_x * (py + proc_dims_y * pz);
}

static void decomp_fill_neighbor(DecompNeighborMetadata *nbr,
                                 DecompFace face,
                                 int rank,
                                 int proc_dims_x, int proc_dims_y, int proc_dims_z,
                                 int px, int py, int pz,
                                 int global_nx, int global_ny, int global_nz)
{
    int start_x, local_nx;
    int start_y, local_ny;
    int start_z, local_nz;

    nbr->face = face;
    nbr->rank = rank;
    nbr->exists = (rank != MPI_PROC_NULL);
    nbr->proc_coords[0] = px;
    nbr->proc_coords[1] = py;
    nbr->proc_coords[2] = pz;

    if (rank == MPI_PROC_NULL) {
        nbr->start_x = 0;
        nbr->end_x = 0;
        nbr->start_y = 0;
        nbr->end_y = 0;
        nbr->start_z = 0;
        nbr->end_z = 0;
        nbr->local_nx = 0;
        nbr->local_ny = 0;
        nbr->local_nz = 0;
        return;
    }

    decomp_partition_1d(global_nx, px, proc_dims_x, &start_x, &local_nx);
    decomp_partition_1d(global_ny, py, proc_dims_y, &start_y, &local_ny);
    decomp_partition_1d(global_nz, pz, proc_dims_z, &start_z, &local_nz);

    nbr->start_x = start_x;
    nbr->end_x = start_x + local_nx;
    nbr->start_y = start_y;
    nbr->end_y = start_y + local_ny;
    nbr->start_z = start_z;
    nbr->end_z = start_z + local_nz;
    nbr->local_nx = local_nx;
    nbr->local_ny = local_ny;
    nbr->local_nz = local_nz;
}

static void decomp_fill_face_neighbors(DecompositionMetadata *meta)
{
    int px = meta->proc_coords[0];
    int py = meta->proc_coords[1];
    int pz = meta->proc_coords[2];
    int pdx = meta->proc_dims[0];
    int pdy = meta->proc_dims[1];
    int pdz = meta->proc_dims[2];

    meta->nbr_xm = decomp_rank_from_dims_coords(pdx, pdy, pdz, px - 1, py, pz);
    meta->nbr_xp = decomp_rank_from_dims_coords(pdx, pdy, pdz, px + 1, py, pz);
    meta->nbr_ym = decomp_rank_from_dims_coords(pdx, pdy, pdz, px, py - 1, pz);
    meta->nbr_yp = decomp_rank_from_dims_coords(pdx, pdy, pdz, px, py + 1, pz);
    meta->nbr_zm = decomp_rank_from_dims_coords(pdx, pdy, pdz, px, py, pz - 1);
    meta->nbr_zp = decomp_rank_from_dims_coords(pdx, pdy, pdz, px, py, pz + 1);

    decomp_fill_neighbor(&meta->neighbors[DECOMP_FACE_XM], DECOMP_FACE_XM,
                         meta->nbr_xm, pdx, pdy, pdz, px - 1, py, pz,
                         meta->global_nx, meta->global_ny, meta->global_nz);
    decomp_fill_neighbor(&meta->neighbors[DECOMP_FACE_XP], DECOMP_FACE_XP,
                         meta->nbr_xp, pdx, pdy, pdz, px + 1, py, pz,
                         meta->global_nx, meta->global_ny, meta->global_nz);
    decomp_fill_neighbor(&meta->neighbors[DECOMP_FACE_YM], DECOMP_FACE_YM,
                         meta->nbr_ym, pdx, pdy, pdz, px, py - 1, pz,
                         meta->global_nx, meta->global_ny, meta->global_nz);
    decomp_fill_neighbor(&meta->neighbors[DECOMP_FACE_YP], DECOMP_FACE_YP,
                         meta->nbr_yp, pdx, pdy, pdz, px, py + 1, pz,
                         meta->global_nx, meta->global_ny, meta->global_nz);
    decomp_fill_neighbor(&meta->neighbors[DECOMP_FACE_ZM], DECOMP_FACE_ZM,
                         meta->nbr_zm, pdx, pdy, pdz, px, py, pz - 1,
                         meta->global_nx, meta->global_ny, meta->global_nz);
    decomp_fill_neighbor(&meta->neighbors[DECOMP_FACE_ZP], DECOMP_FACE_ZP,
                         meta->nbr_zp, pdx, pdy, pdz, px, py, pz + 1,
                         meta->global_nx, meta->global_ny, meta->global_nz);
}

static int decomp_dim_is_fixed_valid(int global_n, int proc_dim)
{
    if (proc_dim < 0) return 0;
    if (proc_dim == 0) return 1;
    if (global_n <= 0) return 0;
    return proc_dim <= global_n;
}

static double decomp_local_box_score(int global_nx, int global_ny, int global_nz,
                                     int px, int py, int pz)
{
    double lx = ceil((double)global_nx / (double)px);
    double ly = ceil((double)global_ny / (double)py);
    double lz = ceil((double)global_nz / (double)pz);
    double llx, lly, llz;
    double aspect_term;
    double surface_term;

    llx = log(lx);
    lly = log(ly);
    llz = log(lz);

    aspect_term =
        (llx - lly) * (llx - lly) +
        (lly - llz) * (lly - llz) +
        (llz - llx) * (llz - llx);

    surface_term = (lx * ly + ly * lz + lz * lx) / (lx * ly * lz);

    return aspect_term + 1.0e-6 * surface_term;
}

int decomp_choose_proc_dims_3d(int size, int proc_dims[3])
{
    int dims[3];
    int ierr;

    if (proc_dims == 0) return -1;
    if (size <= 0) return -1;

    dims[0] = proc_dims[0];
    dims[1] = proc_dims[1];
    dims[2] = proc_dims[2];

    if (dims[0] < 0 || dims[1] < 0 || dims[2] < 0) return -1;

    ierr = MPI_Dims_create(size, 3, dims);
    if (ierr != MPI_SUCCESS) return -1;

    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) return -1;
    if (dims[0] * dims[1] * dims[2] != size) return -1;

    proc_dims[0] = dims[0];
    proc_dims[1] = dims[1];
    proc_dims[2] = dims[2];

    return 0;
}

int decomp_choose_proc_dims_3d_for_box(int global_nx, int global_ny, int global_nz,
                                       int size, int proc_dims[3])
{
    int fixed_x, fixed_y, fixed_z;
    int best_dims[3] = {0, 0, 0};
    double best_score = 0.0;
    int have_best = 0;

    if (proc_dims == 0) return -1;
    if (global_nx <= 0 || global_ny <= 0 || global_nz <= 0) return -1;
    if (size <= 0) return -1;
    if ((long int)global_nx * global_ny * global_nz < (long int)size) return -1;

    fixed_x = proc_dims[0];
    fixed_y = proc_dims[1];
    fixed_z = proc_dims[2];

    if (!decomp_dim_is_fixed_valid(global_nx, fixed_x)) return -1;
    if (!decomp_dim_is_fixed_valid(global_ny, fixed_y)) return -1;
    if (!decomp_dim_is_fixed_valid(global_nz, fixed_z)) return -1;

    for (int px = 1; px <= size; px++) {
        int rem_xy;

        if (size % px != 0) continue;
        if (fixed_x > 0 && px != fixed_x) continue;
        if (px > global_nx) continue;

        rem_xy = size / px;
        for (int py = 1; py <= rem_xy; py++) {
            int pz;
            double score;

            if (rem_xy % py != 0) continue;
            if (fixed_y > 0 && py != fixed_y) continue;
            if (py > global_ny) continue;

            pz = rem_xy / py;
            if (fixed_z > 0 && pz != fixed_z) continue;
            if (pz > global_nz) continue;

            score = decomp_local_box_score(global_nx, global_ny, global_nz,
                                           px, py, pz);
            if (!have_best || score < best_score) {
                best_dims[0] = px;
                best_dims[1] = py;
                best_dims[2] = pz;
                best_score = score;
                have_best = 1;
            }
        }
    }

    if (!have_best) return -1;

    proc_dims[0] = best_dims[0];
    proc_dims[1] = best_dims[1];
    proc_dims[2] = best_dims[2];

    return 0;
}

void decomp_partition_1d(int n_global, int rank, int size,
                         int *start_out, int *count_out)
{
    int base = n_global / size;
    int rem  = n_global % size;
    int cnt  = base + (rank < rem ? 1 : 0);
    int off  = rank * base + (rank < rem ? rank : rem);

    if (start_out != 0) *start_out = off;
    if (count_out != 0) *count_out = cnt;
}

int decomp_init_slab_x(int global_nx, int global_ny, int global_nz,
                       int rank, int size,
                       DecompositionMetadata *meta)
{
    int start_x, local_nx;

    if (meta == 0) return -1;
    if (global_nx <= 0 || global_ny <= 0 || global_nz <= 0) return -1;
    if (size <= 0 || rank < 0 || rank >= size) return -1;

    decomp_partition_1d(global_nx, rank, size, &start_x, &local_nx);

    meta->layout = DECOMP_LAYOUT_SLAB_X;

    meta->rank = rank;
    meta->size = size;

    meta->global_nx = global_nx;
    meta->global_ny = global_ny;
    meta->global_nz = global_nz;

    meta->proc_dims[0] = size;
    meta->proc_dims[1] = 1;
    meta->proc_dims[2] = 1;

    meta->proc_coords[0] = rank;
    meta->proc_coords[1] = 0;
    meta->proc_coords[2] = 0;

    meta->start_x = start_x;
    meta->end_x   = start_x + local_nx;
    meta->start_y = 0;
    meta->end_y   = global_ny;
    meta->start_z = 0;
    meta->end_z   = global_nz;

    meta->local_nx = local_nx;
    meta->local_ny = global_ny;
    meta->local_nz = global_nz;

    meta->local_cell_start =
        (long int)meta->start_x * global_ny * global_nz;
    meta->local_cell_count =
        (long int)meta->local_nx * meta->local_ny * meta->local_nz;
    meta->is_contiguous_linear = 1;

    decomp_fill_face_neighbors(meta);

    return 0;
}

int decomp_describe_linear_block(int global_nx, int global_ny, int global_nz,
                                 long int linear_start, long int linear_count,
                                 int rank, int size,
                                 DecompLinearBlockMetadata *desc)
{
    long int total_cells;
    long int yz_plane;
    long int linear_end;
    int all_exact = 1;
    int r;

    if (desc == 0) return -1;
    if (global_nx <= 0 || global_ny <= 0 || global_nz <= 0) return -1;
    if (size <= 0 || rank < 0 || rank >= size) return -1;

    total_cells = (long int)global_nx * (long int)global_ny * (long int)global_nz;
    if (total_cells > 2147483647L) return -1;
    yz_plane = (long int)global_ny * (long int)global_nz;
    linear_end = linear_start + linear_count;

    if (linear_start < 0 || linear_count < 0 || linear_end > total_cells) return -1;

    memset(desc, 0, sizeof(*desc));
    desc->global_nx = global_nx;
    desc->global_ny = global_ny;
    desc->global_nz = global_nz;
    desc->linear_start = linear_start;
    desc->linear_count = linear_count;
    desc->linear_end = linear_end;
    desc->yz_plane_size = (int)yz_plane;

    decomp_linear_to_coords(global_ny, global_nz, linear_start,
                            &desc->start_x, &desc->start_y, &desc->start_z);

    if (linear_end == total_cells) {
        desc->end_x = global_nx;
        desc->end_y = 0;
        desc->end_z = 0;
    } else {
        decomp_linear_to_coords(global_ny, global_nz, linear_end,
                                &desc->end_x, &desc->end_y, &desc->end_z);
    }

    desc->start_plane_aligned = (yz_plane > 0 && (linear_start % yz_plane) == 0) ? 1 : 0;
    desc->end_plane_aligned = (yz_plane > 0 && (linear_end % yz_plane) == 0) ? 1 : 0;
    desc->is_exact_x_slab_box =
        (linear_count == 0) ? 1 : (desc->start_plane_aligned && desc->end_plane_aligned);

    if (desc->is_exact_x_slab_box) {
        desc->slab_start_x = desc->start_x;
        desc->slab_end_x = (linear_end == total_cells) ? global_nx : desc->end_x;
        desc->slab_nx = desc->slab_end_x - desc->slab_start_x;
    }

    for (r = 0; r < size; r++) {
        int start_r, count_r;
        long int start_l, count_l, end_l;

        decomp_partition_1d((int)total_cells, r, size, &start_r, &count_r);
        start_l = (long int)start_r;
        count_l = (long int)count_r;
        end_l = start_l + count_l;
        if ((count_l > 0) &&
            (((start_l % yz_plane) != 0) || ((end_l % yz_plane) != 0))) {
            all_exact = 0;
            break;
        }
    }
    desc->all_ranks_exact_x_slab = all_exact;

    return 0;
}

int decomp_init_slab_x_from_linear_block(int global_nx, int global_ny, int global_nz,
                                         long int linear_start, long int linear_count,
                                         int rank, int size,
                                         DecompositionMetadata *meta)
{
    DecompLinearBlockMetadata desc;
    int r;

    if (meta == 0) return -1;
    if (decomp_describe_linear_block(global_nx, global_ny, global_nz,
                                     linear_start, linear_count,
                                     rank, size, &desc) != 0) {
        return -1;
    }
    if (!desc.is_exact_x_slab_box || !desc.all_ranks_exact_x_slab) return -1;

    memset(meta, 0, sizeof(*meta));
    meta->layout = DECOMP_LAYOUT_SLAB_X;
    meta->rank = rank;
    meta->size = size;

    meta->global_nx = global_nx;
    meta->global_ny = global_ny;
    meta->global_nz = global_nz;

    meta->proc_dims[0] = size;
    meta->proc_dims[1] = 1;
    meta->proc_dims[2] = 1;
    meta->proc_coords[0] = rank;
    meta->proc_coords[1] = 0;
    meta->proc_coords[2] = 0;

    meta->start_x = desc.slab_start_x;
    meta->end_x = desc.slab_end_x;
    meta->start_y = 0;
    meta->end_y = global_ny;
    meta->start_z = 0;
    meta->end_z = global_nz;

    meta->local_nx = desc.slab_nx;
    meta->local_ny = global_ny;
    meta->local_nz = global_nz;

    meta->local_cell_start = linear_start;
    meta->local_cell_count = linear_count;
    meta->is_contiguous_linear = 1;

    meta->nbr_xm = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    meta->nbr_xp = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;
    meta->nbr_ym = MPI_PROC_NULL;
    meta->nbr_yp = MPI_PROC_NULL;
    meta->nbr_zm = MPI_PROC_NULL;
    meta->nbr_zp = MPI_PROC_NULL;

    for (r = 0; r < 6; r++) {
        meta->neighbors[r].face = (DecompFace)r;
        meta->neighbors[r].rank = MPI_PROC_NULL;
        meta->neighbors[r].exists = 0;
    }

    if (meta->nbr_xm != MPI_PROC_NULL) {
        int start_r, count_r;
        decomp_partition_1d((int)((long int)global_nx * global_ny * global_nz),
                           meta->nbr_xm, size, &start_r, &count_r);
        meta->neighbors[DECOMP_FACE_XM].face = DECOMP_FACE_XM;
        meta->neighbors[DECOMP_FACE_XM].rank = meta->nbr_xm;
        meta->neighbors[DECOMP_FACE_XM].exists = 1;
        meta->neighbors[DECOMP_FACE_XM].proc_coords[0] = rank - 1;
        meta->neighbors[DECOMP_FACE_XM].proc_coords[1] = 0;
        meta->neighbors[DECOMP_FACE_XM].proc_coords[2] = 0;
        meta->neighbors[DECOMP_FACE_XM].start_x = (int)((long int)start_r / ((long int)global_ny * global_nz));
        meta->neighbors[DECOMP_FACE_XM].end_x = meta->neighbors[DECOMP_FACE_XM].start_x +
                                                (int)((long int)count_r / ((long int)global_ny * global_nz));
        meta->neighbors[DECOMP_FACE_XM].start_y = 0;
        meta->neighbors[DECOMP_FACE_XM].end_y = global_ny;
        meta->neighbors[DECOMP_FACE_XM].start_z = 0;
        meta->neighbors[DECOMP_FACE_XM].end_z = global_nz;
        meta->neighbors[DECOMP_FACE_XM].local_nx =
            meta->neighbors[DECOMP_FACE_XM].end_x - meta->neighbors[DECOMP_FACE_XM].start_x;
        meta->neighbors[DECOMP_FACE_XM].local_ny = global_ny;
        meta->neighbors[DECOMP_FACE_XM].local_nz = global_nz;
    }

    if (meta->nbr_xp != MPI_PROC_NULL) {
        int start_r, count_r;
        decomp_partition_1d((int)((long int)global_nx * global_ny * global_nz),
                           meta->nbr_xp, size, &start_r, &count_r);
        meta->neighbors[DECOMP_FACE_XP].face = DECOMP_FACE_XP;
        meta->neighbors[DECOMP_FACE_XP].rank = meta->nbr_xp;
        meta->neighbors[DECOMP_FACE_XP].exists = 1;
        meta->neighbors[DECOMP_FACE_XP].proc_coords[0] = rank + 1;
        meta->neighbors[DECOMP_FACE_XP].proc_coords[1] = 0;
        meta->neighbors[DECOMP_FACE_XP].proc_coords[2] = 0;
        meta->neighbors[DECOMP_FACE_XP].start_x = (int)((long int)start_r / ((long int)global_ny * global_nz));
        meta->neighbors[DECOMP_FACE_XP].end_x = meta->neighbors[DECOMP_FACE_XP].start_x +
                                                (int)((long int)count_r / ((long int)global_ny * global_nz));
        meta->neighbors[DECOMP_FACE_XP].start_y = 0;
        meta->neighbors[DECOMP_FACE_XP].end_y = global_ny;
        meta->neighbors[DECOMP_FACE_XP].start_z = 0;
        meta->neighbors[DECOMP_FACE_XP].end_z = global_nz;
        meta->neighbors[DECOMP_FACE_XP].local_nx =
            meta->neighbors[DECOMP_FACE_XP].end_x - meta->neighbors[DECOMP_FACE_XP].start_x;
        meta->neighbors[DECOMP_FACE_XP].local_ny = global_ny;
        meta->neighbors[DECOMP_FACE_XP].local_nz = global_nz;
    }

    return 0;
}

int decomp_init_block_xyz(int global_nx, int global_ny, int global_nz,
                          int proc_dims_x, int proc_dims_y, int proc_dims_z,
                          int rank, int size,
                          DecompositionMetadata *meta)
{
    int start_x, local_nx;
    int start_y, local_ny;
    int start_z, local_nz;
    int cells_per_plane;

    if (meta == 0) return -1;
    if (global_nx <= 0 || global_ny <= 0 || global_nz <= 0) return -1;
    if (size <= 0 || rank < 0 || rank >= size) return -1;
    if (proc_dims_x <= 0 || proc_dims_y <= 0 || proc_dims_z <= 0) return -1;
    if (proc_dims_x * proc_dims_y * proc_dims_z != size) return -1;

    meta->layout = DECOMP_LAYOUT_BLOCK_XYZ;

    meta->rank = rank;
    meta->size = size;

    meta->global_nx = global_nx;
    meta->global_ny = global_ny;
    meta->global_nz = global_nz;

    meta->proc_dims[0] = proc_dims_x;
    meta->proc_dims[1] = proc_dims_y;
    meta->proc_dims[2] = proc_dims_z;

    meta->proc_coords[0] = rank % proc_dims_x;
    meta->proc_coords[1] = (rank / proc_dims_x) % proc_dims_y;
    meta->proc_coords[2] = rank / (proc_dims_x * proc_dims_y);

    decomp_partition_1d(global_nx, meta->proc_coords[0], proc_dims_x,
                        &start_x, &local_nx);
    decomp_partition_1d(global_ny, meta->proc_coords[1], proc_dims_y,
                        &start_y, &local_ny);
    decomp_partition_1d(global_nz, meta->proc_coords[2], proc_dims_z,
                        &start_z, &local_nz);

    meta->start_x = start_x;
    meta->end_x   = start_x + local_nx;
    meta->start_y = start_y;
    meta->end_y   = start_y + local_ny;
    meta->start_z = start_z;
    meta->end_z   = start_z + local_nz;

    meta->local_nx = local_nx;
    meta->local_ny = local_ny;
    meta->local_nz = local_nz;

    cells_per_plane = global_ny * global_nz;
    meta->local_cell_start =
        (long int)meta->start_x * cells_per_plane +
        (long int)meta->start_y * global_nz +
        (long int)meta->start_z;
    meta->local_cell_count =
        (long int)local_nx * local_ny * local_nz;
    meta->is_contiguous_linear =
        (meta->start_y == 0 && meta->end_y == global_ny &&
         meta->start_z == 0 && meta->end_z == global_nz);

    decomp_fill_face_neighbors(meta);

    return 0;
}

int decomp_init_block_xyz_auto(int global_nx, int global_ny, int global_nz,
                               int rank, int size,
                               DecompositionMetadata *meta)
{
    int proc_dims[3] = {0, 0, 0};

    if (decomp_choose_proc_dims_3d_for_box(global_nx, global_ny, global_nz,
                                           size, proc_dims) != 0) {
        return -1;
    }

    return decomp_init_block_xyz(global_nx, global_ny, global_nz,
                                 proc_dims[0], proc_dims[1], proc_dims[2],
                                 rank, size, meta);
}

int decomp_rank_from_proc_coords(const DecompositionMetadata *meta,
                                 int px, int py, int pz)
{
    if (meta == 0) return MPI_PROC_NULL;

    return decomp_rank_from_dims_coords(meta->proc_dims[0], meta->proc_dims[1],
                                        meta->proc_dims[2], px, py, pz);
}

const char *decomp_face_name(DecompFace face)
{
    switch (face) {
        case DECOMP_FACE_XM: return "xm";
        case DECOMP_FACE_XP: return "xp";
        case DECOMP_FACE_YM: return "ym";
        case DECOMP_FACE_YP: return "yp";
        case DECOMP_FACE_ZM: return "zm";
        case DECOMP_FACE_ZP: return "zp";
        default:             return "unknown";
    }

    return 0;
}

long int decomp_global_index(const DecompositionMetadata *meta,
                             int ix, int iy, int iz)
{
    return (long int)ix * meta->global_ny * meta->global_nz
         + (long int)iy * meta->global_nz
         + (long int)iz;
}

long int decomp_local_index(const DecompositionMetadata *meta,
                            int ix_local, int iy_local, int iz_local)
{
    return (long int)ix_local * meta->local_ny * meta->local_nz
         + (long int)iy_local * meta->local_nz
         + (long int)iz_local;
}

int decomp_contains_global(const DecompositionMetadata *meta,
                           int ix, int iy, int iz)
{
    return (ix >= meta->start_x && ix < meta->end_x &&
            iy >= meta->start_y && iy < meta->end_y &&
            iz >= meta->start_z && iz < meta->end_z);
}

int decomp_global_to_local(const DecompositionMetadata *meta,
                           int ix, int iy, int iz,
                           int *ix_local, int *iy_local, int *iz_local)
{
    if (!decomp_contains_global(meta, ix, iy, iz)) return 0;

    if (ix_local != 0) *ix_local = ix - meta->start_x;
    if (iy_local != 0) *iy_local = iy - meta->start_y;
    if (iz_local != 0) *iz_local = iz - meta->start_z;

    return 1;
}
