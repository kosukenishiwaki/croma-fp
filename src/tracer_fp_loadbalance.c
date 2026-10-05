/*
    tracer_fp_loadbalance.c

    K. Nishiwaki, 2026-06-18
    - load-balancing utils. the load is calclated with number of substeps.
    - load-balancing works before the FP main loop. skipped when MPI rank = 1.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "FP_Coef.h"
#include "READFILE.h"
#include "tracer_fp_background.h"
#include "tracer_fp_cr_init.h"
#include "tracer_fp_debug.h"
#include "read_grid_hdf5.h"
#include "tracer_fp_loadbalance.h"
#include "tracer_fp_nsub.h"
#include "tracer_fp_output.h"

#ifdef FP_USE_CUDA_BACKEND
#define TRACER_FP_PROGNAME "tracer_fp_cuda.out"
#else
#define TRACER_FP_PROGNAME "tracer_fp_cpu.out"
#endif

static double bucketstats_top_fraction(void)
{
    static int initialized = 0;
    static double fraction = -1.0;

    if (!initialized) {
        fraction = tracer_bucketstats_top_frac;
        const char *env = getenv("TRACER_BUCKETSTATS_TOP_FRAC");
        if (env != 0 && *env != '\0') {
            const double parsed = atof(env);
            if (parsed > 0.0 && parsed <= 1.0) fraction = parsed;
        }
        if (!(fraction > 0.0 && fraction <= 1.0)) fraction = 0.001;
        initialized = 1;
    }
    return fraction;
}

static void init_test_params(void)
{
    pmin = -1.0;
    pmax = 8.0;
    pemin = -0.5;
    pemax = 6.0;
}

static double lerp_linear(double x0, double x1, double t)
{
    return x0 + (x1 - x0) * t;
}

static double lerp_log10(double x0, double x1, double t)
{
    return pow(10.0, lerp_linear(log10(x0), log10(x1), t));
}

static int bucketstats_record_compare_desc(const void *lhs,
                                                     const void *rhs)
{
    const TracerBucketStatsRecord *a = (const TracerBucketStatsRecord *)lhs;
    const TracerBucketStatsRecord *b = (const TracerBucketStatsRecord *)rhs;

    if (a->nsub != b->nsub) return (b->nsub - a->nsub);
    if (a->target_nsub != b->target_nsub) return (b->target_nsub - a->target_nsub);
    if (a->tracer_id < b->tracer_id) return -1;
    if (a->tracer_id > b->tracer_id) return 1;
    return 0;
}

int tracer_fp_bucket_top_write(FILE *fp,
                                             int mpi_rank,
                                             int mpi_size,
                                             int ntracer_global,
                                             int snapshot_1based,
                                             int physical_snapshot,
                                             double z_snapshot,
                                             int nlocal,
                                             const long int *tracer_ids,
                                             const int *nsubsteps,
                                             const int *target_nsubsteps,
                                             const double *n_gas,
                                             const double *kbt,
                                             const double *b_field,
                                             const double *divv,
                                             const double *lturb,
                                             const double *dv,
                                             const double *beta)
{
    const int fields_per_tracer = 10;
    const double top_fraction = bucketstats_top_fraction();
    double *sendbuf = 0;
    int *recvcounts = 0;
    int *displs = 0;
    double *recvbuf = 0;
    TracerBucketStatsRecord *records = 0;
    int local_count = nlocal * fields_per_tracer;
    int total_count = 0;
    int top_count = 0;
    int itr;
    int ierr = -1;

    if ((mpi_rank == 0 && fp == 0) || tracer_ids == 0 ||
        nsubsteps == 0 || target_nsubsteps == 0 ||
        n_gas == 0 || kbt == 0 || b_field == 0 || divv == 0 ||
        lturb == 0 || dv == 0 || beta == 0) return -1;

    sendbuf = (double *)calloc((size_t)local_count, sizeof(double));
    if (sendbuf == 0) goto cleanup;
    for (itr = 0; itr < nlocal; itr++) {
        const size_t off = (size_t)itr * (size_t)fields_per_tracer;
        sendbuf[off + 0] = (double)tracer_ids[itr];
        sendbuf[off + 1] = (double)nsubsteps[itr];
        sendbuf[off + 2] = (double)target_nsubsteps[itr];
        sendbuf[off + 3] = n_gas[itr];
        sendbuf[off + 4] = kbt[itr];
        sendbuf[off + 5] = b_field[itr];
        sendbuf[off + 6] = divv[itr];
        sendbuf[off + 7] = lturb[itr];
        sendbuf[off + 8] = dv[itr];
        sendbuf[off + 9] = beta[itr];
    }

    if (mpi_rank == 0) {
        recvcounts = (int *)calloc((size_t)mpi_size, sizeof(int));
        displs = (int *)calloc((size_t)mpi_size, sizeof(int));
        if (recvcounts == 0 || displs == 0) goto cleanup;
    }
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        int irank;
        for (irank = 0; irank < mpi_size; irank++) {
            displs[irank] = total_count;
            total_count += recvcounts[irank];
        }
        recvbuf = (double *)calloc((size_t)total_count, sizeof(double));
        if (recvbuf == 0) goto cleanup;
    }
    MPI_Gatherv(sendbuf, local_count, MPI_DOUBLE,
                recvbuf, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        const int nrecord = (fields_per_tracer > 0) ? (total_count / fields_per_tracer) : 0;
        top_count = (int)ceil((double)ntracer_global * top_fraction);
        if (top_count < 1) top_count = 1;
        if (top_count > nrecord) top_count = nrecord;
        records = (TracerBucketStatsRecord *)calloc((size_t)nrecord, sizeof(TracerBucketStatsRecord));
        if (records == 0) goto cleanup;
        for (itr = 0; itr < nrecord; itr++) {
            const size_t off = (size_t)itr * (size_t)fields_per_tracer;
            records[itr].tracer_id = (long int)llround(recvbuf[off + 0]);
            records[itr].nsub = (int)llround(recvbuf[off + 1]);
            records[itr].target_nsub = (int)llround(recvbuf[off + 2]);
            records[itr].n_gas = recvbuf[off + 3];
            records[itr].kbt = recvbuf[off + 4];
            records[itr].b_field = recvbuf[off + 5];
            records[itr].divv = recvbuf[off + 6];
            records[itr].lturb = recvbuf[off + 7];
            records[itr].dv = recvbuf[off + 8];
            records[itr].beta = recvbuf[off + 9];
        }
        qsort(records, (size_t)nrecord, sizeof(*records),
              bucketstats_record_compare_desc);
        for (itr = 0; itr < top_count; itr++) {
            const double inflate = (records[itr].target_nsub > 0)
                ? (double)records[itr].nsub / (double)records[itr].target_nsub : 1.0;
            fprintf(fp,
                    "%d\t%d\t%.6f\t%ld\t%d\t%d\t%.6f\t%.9e\t%.9e\t%.9e\t%.9e\t%.9e\t%.9e\t%.9e\n",
                    snapshot_1based, physical_snapshot, z_snapshot,
                    records[itr].tracer_id, records[itr].nsub, records[itr].target_nsub,
                    inflate, records[itr].n_gas, records[itr].kbt, records[itr].b_field,
                    records[itr].divv, records[itr].lturb, records[itr].dv, records[itr].beta);
        }
        fflush(fp);
    }

    ierr = 0;

cleanup:
    free(sendbuf);
    free(recvcounts);
    free(displs);
    free(recvbuf);
    free(records);
    return ierr;
}

static int tracer_load_balance_record_compare_desc(const void *a, const void *b)
{
    const TracerLoadBalanceRecord *ra = (const TracerLoadBalanceRecord *)a;
    const TracerLoadBalanceRecord *rb = (const TracerLoadBalanceRecord *)b;

    if (ra->sum_nsub < rb->sum_nsub) return 1;
    if (ra->sum_nsub > rb->sum_nsub) return -1;
    if (ra->tracer_id < rb->tracer_id) return -1;
    if (ra->tracer_id > rb->tracer_id) return 1;
    return 0;
}

static int tracer_load_balance_record_compare_id_asc(const void *a, const void *b)
{
    const TracerLoadBalanceRecord *ra = (const TracerLoadBalanceRecord *)a;
    const TracerLoadBalanceRecord *rb = (const TracerLoadBalanceRecord *)b;

    if (ra->tracer_id < rb->tracer_id) return -1;
    if (ra->tracer_id > rb->tracer_id) return 1;
    return 0;
}

static int tracer_load_balance_record_compare_start_asc(const void *a, const void *b)
{
    const TracerLoadBalanceRecord *ra = (const TracerLoadBalanceRecord *)a;
    const TracerLoadBalanceRecord *rb = (const TracerLoadBalanceRecord *)b;

    if (ra->tracer_id < rb->tracer_id) return -1;
    if (ra->tracer_id > rb->tracer_id) return 1;
    return 0;
}

typedef struct {
    int source_offset;
    long int global_id;
} TracerLoadBalanceSelectionPair;

static int tracer_load_balance_selection_pair_compare(const void *a, const void *b)
{
    const TracerLoadBalanceSelectionPair *pa = (const TracerLoadBalanceSelectionPair *)a;
    const TracerLoadBalanceSelectionPair *pb = (const TracerLoadBalanceSelectionPair *)b;

    if (pa->source_offset < pb->source_offset) return -1;
    if (pa->source_offset > pb->source_offset) return 1;
    return 0;
}

static int write_load_balance_files(const char *output_dir,
                                              long int run_tracer_start,
                                              int ntracer_global,
                                              int mpi_size,
                                              double top_frac,
                                              const TracerLoadBalancePlan *plan,
                                              const long int *all_ids,
                                              const long long *all_sums,
                                              int total_count)
{
    char path[MAX_LINE_LENGTH];
    FILE *fp = 0;
    long long *rank_sum_nsub = 0;
    long long *sum_by_offset = 0;
    int *rank_counts = 0;
    int iheavy;
    int irank;
    int ierr = -1;

    if (output_dir == 0 || *output_dir == '\0' || plan == 0 || !plan->enabled) return 0;
    if (ensure_output_dir(output_dir) != 0) return -1;

    rank_sum_nsub = (long long *)calloc((size_t)((mpi_size > 0) ? mpi_size : 1), sizeof(long long));
    rank_counts = (int *)calloc((size_t)((mpi_size > 0) ? mpi_size : 1), sizeof(int));
    if (rank_sum_nsub == 0 || rank_counts == 0) goto cleanup;

    for (iheavy = 0; iheavy < plan->nheavy; iheavy++) {
        const int rank = plan->records[iheavy].target_rank;
        if (rank >= 0 && rank < mpi_size) {
            rank_sum_nsub[rank] += plan->records[iheavy].sum_nsub;
            rank_counts[rank] += plan->records[iheavy].count;
        }
    }

    snprintf(path, sizeof(path), "%s/heavy_assignments.txt", output_dir);
    fp = fopen(path, "w");
    if (fp == 0) goto cleanup;
    fprintf(fp, "# tracer_id source_offset target_rank count sum_nsub\n");
    for (iheavy = 0; iheavy < plan->nheavy; iheavy++) {
        fprintf(fp, "%ld %ld %d %d %lld\n",
                plan->records[iheavy].tracer_id,
                plan->records[iheavy].tracer_id - run_tracer_start,
                plan->records[iheavy].target_rank,
                plan->records[iheavy].count,
                plan->records[iheavy].sum_nsub);
    }
    fclose(fp);
    fp = 0;

    snprintf(path, sizeof(path), "%s/heavy_summary.txt", output_dir);
    fp = fopen(path, "w");
    if (fp == 0) goto cleanup;
    fprintf(fp, "# global_ntracer %d\n", ntracer_global);
    fprintf(fp, "# top_frac %.6f\n", top_frac);
    fprintf(fp, "# heavy_count %d\n", plan->nheavy);
    fprintf(fp, "# rank heavy_count heavy_sum_nsub\n");
    for (iheavy = 0; iheavy < mpi_size; iheavy++) {
        fprintf(fp, "%d %d %lld\n", iheavy, rank_counts[iheavy], rank_sum_nsub[iheavy]);
    }
    fclose(fp);
    fp = 0;

    if (all_ids != 0 && all_sums != 0 && total_count == ntracer_global) {
        sum_by_offset = (long long *)calloc((size_t)ntracer_global, sizeof(long long));
        if (sum_by_offset == 0) goto cleanup;
        for (iheavy = 0; iheavy < total_count; iheavy++) {
            const long int off = all_ids[iheavy] - run_tracer_start;
            if (off >= 0 && off < (long int)ntracer_global) {
                sum_by_offset[off] = all_sums[iheavy];
            }
        }

        snprintf(path, sizeof(path), "%s/load_balance_rank_totals.txt", output_dir);
        fp = fopen(path, "w");
        if (fp == 0) goto cleanup;
        fprintf(fp, "# rank base_count heavy_removed_count heavy_assigned_count final_count ");
        fprintf(fp, "base_sum_nsub heavy_removed_sum_nsub heavy_assigned_sum_nsub final_sum_nsub\n");
        for (irank = 0; irank < mpi_size; irank++) {
            long int start = 0;
            long int count = 0;
            long long base_sum = 0;
            long long heavy_removed_sum = 0;
            long long heavy_assigned_sum = 0;
            int heavy_removed_count = 0;
            int heavy_assigned_count = 0;
            int final_count;
            long long final_sum;
            long int off;

            grid_cell_range((long int)ntracer_global, irank, mpi_size, &start, &count);
            for (off = start; off < start + count; off++) {
                base_sum += sum_by_offset[off];
            }
            for (iheavy = 0; iheavy < plan->nheavy; iheavy++) {
                const long int gid = plan->records[iheavy].tracer_id;
                const long int gid_off = gid - run_tracer_start;
                const long int rec_start = gid_off;
                const long int rec_end = rec_start + (long int)plan->records[iheavy].count;
                const long int rank_end = start + count;
                const long int overlap_start = (rec_start > start) ? rec_start : start;
                const long int overlap_end = (rec_end < rank_end) ? rec_end : rank_end;
                long int off;
                if (overlap_end > overlap_start) {
                    heavy_removed_count += (int)(overlap_end - overlap_start);
                    for (off = overlap_start; off < overlap_end; off++) {
                        heavy_removed_sum += sum_by_offset[off];
                    }
                }
                if (plan->records[iheavy].target_rank == irank) {
                    heavy_assigned_sum += plan->records[iheavy].sum_nsub;
                    heavy_assigned_count += plan->records[iheavy].count;
                }
            }
            final_count = (int)count - heavy_removed_count + heavy_assigned_count;
            final_sum = base_sum - heavy_removed_sum + heavy_assigned_sum;
            fprintf(fp, "%d %ld %d %d %d %lld %lld %lld %lld\n",
                    irank, count, heavy_removed_count, heavy_assigned_count, final_count,
                    base_sum, heavy_removed_sum, heavy_assigned_sum, final_sum);
        }
        fclose(fp);
        fp = 0;
    }

    ierr = 0;

cleanup:
    if (fp != 0) fclose(fp);
    free(rank_sum_nsub);
    free(rank_counts);
    free(sum_by_offset);
    return ierr;
}

static int write_chunk_diagnostics(const char *output_dir,
                                             int ntracer_global,
                                             int mpi_size,
                                             const long long *sum_by_offset);

static int write_planning_rank_totals(const char *output_dir,
                                                long int run_tracer_start,
                                                int ntracer_global,
                                                int mpi_size,
                                                double top_frac,
                                                const long int *all_ids,
                                                const long long *all_sums,
                                                int total_count,
                                                const TracerLoadBalancePlan *plan)
{
    char path[MAX_LINE_LENGTH];
    FILE *fp = 0;
    long long *sum_by_offset = 0;
    int irank;
    int iheavy;
    int ierr = -1;

    if (output_dir == 0 || *output_dir == '\0' || ntracer_global <= 0 || mpi_size <= 0 ||
        all_ids == 0 || all_sums == 0 || total_count != ntracer_global) {
        return -1;
    }
    if (ensure_output_dir(output_dir) != 0) return -1;

    sum_by_offset = (long long *)calloc((size_t)ntracer_global, sizeof(long long));
    if (sum_by_offset == 0) goto cleanup;
    for (iheavy = 0; iheavy < total_count; iheavy++) {
        const long int off = all_ids[iheavy] - run_tracer_start;
        if (off >= 0 && off < (long int)ntracer_global) {
            sum_by_offset[off] = all_sums[iheavy];
        }
    }

    snprintf(path, sizeof(path), "%s/planning_rank_totals.tsv", output_dir);
    fp = fopen(path, "w");
    if (fp == 0) goto cleanup;

    fprintf(fp,
            "rank\tbase_count\tpredicted_count\tbase_sum_nsub\tpredicted_sum_nsub\t"
            "heavy_removed_count\theavy_assigned_count\theavy_removed_sum_nsub\t"
            "heavy_assigned_sum_nsub\tbase_over_mean\tpredicted_over_mean\t"
            "load_balancing_enabled\theavy_top_frac\n");

    {
        long long total_load = 0;
        double mean_load = 0.0;
        int k;
        for (k = 0; k < mpi_size; k++) {
            long int start = 0;
            long int count = 0;
            long int off;
            grid_cell_range((long int)ntracer_global, k, mpi_size, &start, &count);
            for (off = start; off < start + count; off++) total_load += sum_by_offset[off];
        }
        if (mpi_size > 0) mean_load = (double)total_load / (double)mpi_size;

        for (irank = 0; irank < mpi_size; irank++) {
            long int start = 0;
            long int count = 0;
            long int off;
            long long base_sum = 0;
            long long heavy_removed_sum = 0;
            long long heavy_assigned_sum = 0;
            int heavy_removed_count = 0;
            int heavy_assigned_count = 0;
            int predicted_count;
            long long predicted_sum;

            grid_cell_range((long int)ntracer_global, irank, mpi_size, &start, &count);
            for (off = start; off < start + count; off++) {
                base_sum += sum_by_offset[off];
            }
            if (plan != 0 && plan->enabled) {
                for (iheavy = 0; iheavy < plan->nheavy; iheavy++) {
                    const long int gid = plan->records[iheavy].tracer_id;
                    const long int gid_off = gid - run_tracer_start;
                    const long int rec_start = gid_off;
                    const long int rec_end = rec_start + (long int)plan->records[iheavy].count;
                    const long int rank_end = start + count;
                    const long int overlap_start = (rec_start > start) ? rec_start : start;
                    const long int overlap_end = (rec_end < rank_end) ? rec_end : rank_end;
                    long int off;
                    if (overlap_end > overlap_start) {
                        heavy_removed_count += (int)(overlap_end - overlap_start);
                        for (off = overlap_start; off < overlap_end; off++) {
                            heavy_removed_sum += sum_by_offset[off];
                        }
                    }
                    if (plan->records[iheavy].target_rank == irank) {
                        heavy_assigned_sum += plan->records[iheavy].sum_nsub;
                        heavy_assigned_count += plan->records[iheavy].count;
                    }
                }
            }
            predicted_count = (int)count - heavy_removed_count + heavy_assigned_count;
            predicted_sum = base_sum - heavy_removed_sum + heavy_assigned_sum;
            fprintf(fp, "%d\t%ld\t%d\t%lld\t%lld\t%d\t%d\t%lld\t%lld\t%.12f\t%.12f\t%d\t%.6f\n",
                    irank, count, predicted_count, base_sum, predicted_sum,
                    heavy_removed_count, heavy_assigned_count,
                    heavy_removed_sum, heavy_assigned_sum,
                    (mean_load > 0.0) ? (double)base_sum / mean_load : 0.0,
                    (mean_load > 0.0) ? (double)predicted_sum / mean_load : 0.0,
                    (plan != 0 && plan->enabled) ? 1 : 0,
                    top_frac);
        }
    }
    fclose(fp);
    fp = 0;

    if (load_balance_diagnostics) {
        if (write_chunk_diagnostics(output_dir, ntracer_global,
                                              mpi_size, sum_by_offset) != 0) {
            goto cleanup;
        }
    }

    ierr = 0;

cleanup:
    if (fp != 0) fclose(fp);
    free(sum_by_offset);
    return ierr;
}

typedef struct {
    long int start;
    int count;
    long long sum_nsub;
    int target_rank;
} TracerLoadBalanceChunkDiag;

static int chunk_diag_compare_desc(const void *a, const void *b)
{
    const TracerLoadBalanceChunkDiag *ca = (const TracerLoadBalanceChunkDiag *)a;
    const TracerLoadBalanceChunkDiag *cb = (const TracerLoadBalanceChunkDiag *)b;

    if (ca->sum_nsub < cb->sum_nsub) return 1;
    if (ca->sum_nsub > cb->sum_nsub) return -1;
    if (ca->start < cb->start) return -1;
    if (ca->start > cb->start) return 1;
    return 0;
}

static int chunk_diag_compare_start_asc(const void *a, const void *b)
{
    const TracerLoadBalanceChunkDiag *ca = (const TracerLoadBalanceChunkDiag *)a;
    const TracerLoadBalanceChunkDiag *cb = (const TracerLoadBalanceChunkDiag *)b;

    if (ca->start < cb->start) return -1;
    if (ca->start > cb->start) return 1;
    return 0;
}

static int write_chunk_diagnostics(const char *output_dir,
                                             int ntracer_global,
                                             int mpi_size,
                                             const long long *sum_by_offset)
{
    static const int chunk_sizes[] = {
        512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
    };
    char path[MAX_LINE_LENGTH];
    FILE *fp = 0;
    int ichunk_size;
    int ierr = -1;

    if (output_dir == 0 || *output_dir == '\0' || ntracer_global <= 0 ||
        mpi_size <= 0 || sum_by_offset == 0) {
        return 0;
    }
    if (ensure_output_dir(output_dir) != 0) return -1;

    snprintf(path, sizeof(path), "%s/load_balance_chunk_diagnostics.tsv", output_dir);
    fp = fopen(path, "w");
    if (fp == 0) return -1;

    fprintf(fp,
            "chunk_size\trank\tpredicted_count\tpredicted_sum_nsub\t"
            "predicted_over_mean\tassigned_chunks\toffset_runs\t"
            "max_contiguous_chunks\tmax_contiguous_tracers\n");

    for (ichunk_size = 0;
         ichunk_size < (int)(sizeof(chunk_sizes) / sizeof(chunk_sizes[0]));
         ichunk_size++) {
        const int chunk_size = chunk_sizes[ichunk_size];
        const int nchunk = (ntracer_global + chunk_size - 1) / chunk_size;
        TracerLoadBalanceChunkDiag *chunks = 0;
        long long *rank_loads = 0;
        int *rank_counts = 0;
        int *rank_chunk_counts = 0;
        int *rank_offset_runs = 0;
        int *rank_max_contig_chunks = 0;
        int *current_contig_chunks = 0;
        int *prev_chunk_index = 0;
        long long total_load = 0;
        double mean_load = 0.0;
        int ichunk;
        int irank;

        chunks = (TracerLoadBalanceChunkDiag *)calloc((size_t)nchunk, sizeof(*chunks));
        rank_loads = (long long *)calloc((size_t)mpi_size, sizeof(*rank_loads));
        rank_counts = (int *)calloc((size_t)mpi_size, sizeof(*rank_counts));
        rank_chunk_counts = (int *)calloc((size_t)mpi_size, sizeof(*rank_chunk_counts));
        rank_offset_runs = (int *)calloc((size_t)mpi_size, sizeof(*rank_offset_runs));
        rank_max_contig_chunks = (int *)calloc((size_t)mpi_size, sizeof(*rank_max_contig_chunks));
        current_contig_chunks = (int *)calloc((size_t)mpi_size, sizeof(*current_contig_chunks));
        prev_chunk_index = (int *)calloc((size_t)mpi_size, sizeof(*prev_chunk_index));
        if (chunks == 0 || rank_loads == 0 || rank_counts == 0 ||
            rank_chunk_counts == 0 || rank_offset_runs == 0 ||
            rank_max_contig_chunks == 0 || current_contig_chunks == 0 ||
            prev_chunk_index == 0) {
            free(chunks);
            free(rank_loads);
            free(rank_counts);
            free(rank_chunk_counts);
            free(rank_offset_runs);
            free(rank_max_contig_chunks);
            free(current_contig_chunks);
            free(prev_chunk_index);
            goto cleanup;
        }

        for (irank = 0; irank < mpi_size; irank++) prev_chunk_index[irank] = -2;

        for (ichunk = 0; ichunk < nchunk; ichunk++) {
            const long int start = (long int)ichunk * (long int)chunk_size;
            long int end = start + (long int)chunk_size;
            long int off;
            chunks[ichunk].start = start;
            if (end > (long int)ntracer_global) end = (long int)ntracer_global;
            chunks[ichunk].count = (int)(end - start);
            for (off = start; off < end; off++) {
                chunks[ichunk].sum_nsub += sum_by_offset[off];
            }
            total_load += chunks[ichunk].sum_nsub;
            chunks[ichunk].target_rank = -1;
        }
        mean_load = (mpi_size > 0) ? (double)total_load / (double)mpi_size : 0.0;

        qsort(chunks, (size_t)nchunk, sizeof(*chunks),
              chunk_diag_compare_desc);

        for (ichunk = 0; ichunk < nchunk; ichunk++) {
            int target_rank = 0;
            for (irank = 1; irank < mpi_size; irank++) {
                if (rank_loads[irank] < rank_loads[target_rank] ||
                    (rank_loads[irank] == rank_loads[target_rank] &&
                     rank_counts[irank] < rank_counts[target_rank])) {
                    target_rank = irank;
                }
            }
            chunks[ichunk].target_rank = target_rank;
            rank_loads[target_rank] += chunks[ichunk].sum_nsub;
            rank_counts[target_rank] += chunks[ichunk].count;
            rank_chunk_counts[target_rank]++;
        }

        /* Recover locality after sorting chunks by offset. */
        qsort(chunks, (size_t)nchunk, sizeof(*chunks),
              chunk_diag_compare_start_asc);
        for (ichunk = 0; ichunk < nchunk; ichunk++) {
            const int rank = chunks[ichunk].target_rank;
            const int chunk_index = (int)(chunks[ichunk].start / (long int)chunk_size);
            if (rank < 0 || rank >= mpi_size) continue;
            if (prev_chunk_index[rank] + 1 == chunk_index) {
                current_contig_chunks[rank]++;
            } else {
                rank_offset_runs[rank]++;
                current_contig_chunks[rank] = 1;
            }
            if (current_contig_chunks[rank] > rank_max_contig_chunks[rank]) {
                rank_max_contig_chunks[rank] = current_contig_chunks[rank];
            }
            prev_chunk_index[rank] = chunk_index;
        }

        for (irank = 0; irank < mpi_size; irank++) {
            fprintf(fp, "%d\t%d\t%d\t%lld\t%.12f\t%d\t%d\t%d\t%d\n",
                    chunk_size, irank, rank_counts[irank], rank_loads[irank],
                    (mean_load > 0.0) ? (double)rank_loads[irank] / mean_load : 0.0,
                    rank_chunk_counts[irank],
                    rank_offset_runs[irank],
                    rank_max_contig_chunks[irank],
                    rank_max_contig_chunks[irank] * chunk_size);
        }

        free(chunks);
        free(rank_loads);
        free(rank_counts);
        free(rank_chunk_counts);
        free(rank_offset_runs);
        free(rank_max_contig_chunks);
        free(current_contig_chunks);
        free(prev_chunk_index);
    }

    ierr = 0;

cleanup:
    if (fp != 0) fclose(fp);
    return ierr;
}

void tracer_fp_lb_plan_free(TracerLoadBalancePlan *plan)
{
    if (plan == 0) return;
    free(plan->records);
    plan->records = 0;
    plan->nheavy = 0;
    plan->chunk_size = 0;
    plan->enabled = 0;
}

static void load_balance_refine_assignments(TracerLoadBalanceRecord *records,
                                                      int nheavy,
                                                      int mpi_size,
                                                      long long *predicted_loads,
                                                      int *predicted_counts)
{
    int pass;
    const int max_passes = (mpi_size > 0) ? (8 * mpi_size) : 0;

    if (records == 0 || predicted_loads == 0 || predicted_counts == 0 ||
        nheavy <= 0 || mpi_size <= 1) {
        return;
    }

    for (pass = 0; pass < max_passes; pass++) {
        int max_rank = 0;
        long long current_max = predicted_loads[0];
        long long current_min = predicted_loads[0];
        long long max_other = -1;
        int iheavy;
        int irank;
        int best_record = -1;
        int best_target = -1;
        long long best_new_peak = current_max;
        long long best_new_spread = current_max - current_min;

        for (irank = 1; irank < mpi_size; irank++) {
            if (predicted_loads[irank] > current_max) {
                current_max = predicted_loads[irank];
                max_rank = irank;
            }
            if (predicted_loads[irank] < current_min) {
                current_min = predicted_loads[irank];
            }
        }
        for (irank = 0; irank < mpi_size; irank++) {
            if (irank == max_rank) continue;
            if (predicted_loads[irank] > max_other) max_other = predicted_loads[irank];
        }

        for (iheavy = 0; iheavy < nheavy; iheavy++) {
            long long new_src_load;
            long long tracer_load;

            if (records[iheavy].target_rank != max_rank) continue;
            tracer_load = records[iheavy].sum_nsub;
            new_src_load = predicted_loads[max_rank] - tracer_load;

            for (irank = 0; irank < mpi_size; irank++) {
                long long new_target_load;
                long long new_peak;
                long long new_min = new_src_load;
                long long new_max = new_src_load;
                long long new_spread;
                int jrank;

                if (irank == max_rank) continue;
                new_target_load = predicted_loads[irank] + tracer_load;
                new_peak = max_other;
                if (new_src_load > new_peak) new_peak = new_src_load;
                if (new_target_load > new_peak) new_peak = new_target_load;
                if (new_peak > best_new_peak) continue;

                for (jrank = 0; jrank < mpi_size; jrank++) {
                    long long load = predicted_loads[jrank];
                    if (jrank == max_rank) load = new_src_load;
                    else if (jrank == irank) load = new_target_load;
                    if (load < new_min) new_min = load;
                    if (load > new_max) new_max = load;
                }
                new_spread = new_max - new_min;
                if (new_peak < best_new_peak ||
                    (new_peak == best_new_peak && new_spread < best_new_spread) ||
                    (new_peak == best_new_peak && new_spread == best_new_spread &&
                     predicted_counts[irank] < predicted_counts[best_target < 0 ? irank : best_target])) {
                    best_record = iheavy;
                    best_target = irank;
                    best_new_peak = new_peak;
                    best_new_spread = new_spread;
                }
            }
        }

        if (best_record < 0 || best_target < 0 || best_new_peak >= current_max) break;

        predicted_loads[max_rank] -= records[best_record].sum_nsub;
        predicted_counts[max_rank] -= records[best_record].count;
        predicted_loads[best_target] += records[best_record].sum_nsub;
        predicted_counts[best_target] += records[best_record].count;
        records[best_record].target_rank = best_target;
    }
}

int tracer_fp_lb_plan_build_global(long int run_tracer_start,
                                                  int ntracer_global,
                                                  int mpi_size,
                                                  double top_frac,
                                                  const char *output_dir,
                                                  const long int *all_ids,
                                                  const long long *all_sums,
                                                  int total_count,
                                                  TracerLoadBalancePlan *plan)
{
    long long *sum_by_offset = 0;
    long long *predicted_loads = 0;
    int *predicted_counts = 0;
    int *owner_by_offset = 0;
    int nheavy = 0;
    int itr;

    if (plan == 0) return -1;
    plan->enabled = 0;
    plan->nheavy = 0;
    plan->chunk_size = 0;
    plan->records = 0;
    if (ntracer_global <= 0 || mpi_size <= 0 || all_ids == 0 || all_sums == 0 || total_count <= 0) return 0;
    if (total_count != ntracer_global) return -1;

    sum_by_offset = (long long *)calloc((size_t)ntracer_global, sizeof(long long));
    predicted_loads = (long long *)calloc((size_t)mpi_size, sizeof(long long));
    predicted_counts = (int *)calloc((size_t)mpi_size, sizeof(int));
    owner_by_offset = (int *)calloc((size_t)ntracer_global, sizeof(int));
    if (sum_by_offset == 0 || predicted_loads == 0 || predicted_counts == 0 || owner_by_offset == 0) {
        free(sum_by_offset);
        free(predicted_loads);
        free(predicted_counts);
        free(owner_by_offset);
        return -1;
    }

    for (itr = 0; itr < total_count; itr++) {
        const long int off = all_ids[itr] - run_tracer_start;
        if (off >= 0 && off < (long int)ntracer_global) {
            sum_by_offset[off] = all_sums[itr];
        }
    }
    for (itr = 0; itr < mpi_size; itr++) {
        long int start = 0;
        long int count = 0;
        long int off;
        grid_cell_range((long int)ntracer_global, itr, mpi_size, &start, &count);
        predicted_counts[itr] = (int)count;
        for (off = start; off < start + count; off++) {
            owner_by_offset[off] = itr;
            predicted_loads[itr] += sum_by_offset[off];
        }
    }

    if (load_balance_chunk_size > 0) {
        const int chunk_size = load_balance_chunk_size;
        const int nchunk = (ntracer_global + chunk_size - 1) / chunk_size;
        TracerLoadBalanceRecord *chunks = 0;
        int ichunk;
        int irank;

        if (chunk_size <= 0) goto cleanup;
        chunks = (TracerLoadBalanceRecord *)calloc((size_t)nchunk, sizeof(*chunks));
        if (chunks == 0) goto cleanup;
        memset(predicted_loads, 0, (size_t)mpi_size * sizeof(*predicted_loads));
        memset(predicted_counts, 0, (size_t)mpi_size * sizeof(*predicted_counts));

        for (ichunk = 0; ichunk < nchunk; ichunk++) {
            const long int start = (long int)ichunk * (long int)chunk_size;
            long int end = start + (long int)chunk_size;
            long int off;
            if (end > (long int)ntracer_global) end = (long int)ntracer_global;
            chunks[ichunk].tracer_id = run_tracer_start + start;
            chunks[ichunk].count = (int)(end - start);
            chunks[ichunk].target_rank = -1;
            for (off = start; off < end; off++) {
                chunks[ichunk].sum_nsub += sum_by_offset[off];
            }
        }
        qsort(chunks, (size_t)nchunk, sizeof(*chunks),
              tracer_load_balance_record_compare_desc);
        for (ichunk = 0; ichunk < nchunk; ichunk++) {
            int target_rank = 0;
            for (irank = 1; irank < mpi_size; irank++) {
                if (predicted_loads[irank] < predicted_loads[target_rank] ||
                    (predicted_loads[irank] == predicted_loads[target_rank] &&
                     predicted_counts[irank] < predicted_counts[target_rank])) {
                    target_rank = irank;
                }
            }
            chunks[ichunk].target_rank = target_rank;
            predicted_loads[target_rank] += chunks[ichunk].sum_nsub;
            predicted_counts[target_rank] += chunks[ichunk].count;
        }
        qsort(chunks, (size_t)nchunk, sizeof(*chunks),
              tracer_load_balance_record_compare_start_asc);
        plan->records = chunks;
        plan->nheavy = nchunk;
        plan->chunk_size = chunk_size;
        plan->enabled = 1;
        if (output_dir != 0 && *output_dir != '\0') {
            if (write_load_balance_files(output_dir, run_tracer_start,
                                                   ntracer_global, mpi_size, top_frac,
                                                   plan, all_ids, all_sums,
                                                   total_count) != 0) {
                goto cleanup;
            }
            if (write_planning_rank_totals(output_dir, run_tracer_start,
                                                     ntracer_global, mpi_size, top_frac,
                                                     all_ids, all_sums, total_count,
                                                     plan) != 0) {
                goto cleanup;
            }
        }
        free(sum_by_offset);
        free(predicted_loads);
        free(predicted_counts);
        free(owner_by_offset);
        return 0;
    }

    {
        TracerLoadBalanceRecord *sorted =
            (TracerLoadBalanceRecord *)calloc((size_t)total_count, sizeof(TracerLoadBalanceRecord));
        int max_candidates;
        int selected = 0;
        long long *overloads = 0;
        char *chosen = 0;
        if (sorted == 0) goto cleanup;
        for (itr = 0; itr < total_count; itr++) {
            sorted[itr].tracer_id = all_ids[itr];
            sorted[itr].sum_nsub = all_sums[itr];
            sorted[itr].target_rank = -1;
            sorted[itr].count = 1;
        }
        qsort(sorted, (size_t)total_count, sizeof(*sorted),
              tracer_load_balance_record_compare_desc);

        max_candidates = (int)ceil((double)ntracer_global * top_frac);
        if (max_candidates < 1) max_candidates = 1;
        if (max_candidates > total_count) max_candidates = total_count;
        overloads = (long long *)calloc((size_t)mpi_size, sizeof(long long));
        chosen = (char *)calloc((size_t)total_count, sizeof(char));
        if (overloads == 0 || chosen == 0) {
            free(sorted);
            free(overloads);
            free(chosen);
            goto cleanup;
        }
        {
            long long total_load = 0;
            int irank;
            for (irank = 0; irank < mpi_size; irank++) total_load += predicted_loads[irank];
            for (irank = 0; irank < mpi_size; irank++) {
                overloads[irank] = predicted_loads[irank] - total_load / (long long)mpi_size;
                if (overloads[irank] < 0) overloads[irank] = 0;
            }
        }
        for (itr = 0; itr < total_count && selected < max_candidates; itr++) {
            const long int off = sorted[itr].tracer_id - run_tracer_start;
            const int source_rank =
                (off >= 0 && off < (long int)ntracer_global) ? owner_by_offset[off] : 0;
            if (overloads[source_rank] <= 0) continue;
            chosen[itr] = 1;
            overloads[source_rank] -= sorted[itr].sum_nsub;
            selected++;
        }
        if (selected == 0) {
            chosen[0] = 1;
            selected = 1;
        }
        nheavy = selected;
        plan->records = (TracerLoadBalanceRecord *)calloc((size_t)nheavy, sizeof(TracerLoadBalanceRecord));
        if (plan->records == 0) {
            free(sorted);
            free(overloads);
            free(chosen);
            goto cleanup;
        }
        selected = 0;
        for (itr = 0; itr < total_count; itr++) {
            if (!chosen[itr]) continue;
            plan->records[selected++] = sorted[itr];
            if (selected >= nheavy) break;
        }
        free(sorted);
        free(overloads);
        free(chosen);
    }

    for (itr = 0; itr < nheavy; itr++) {
        const long int off = plan->records[itr].tracer_id - run_tracer_start;
        const int source_rank =
            (off >= 0 && off < (long int)ntracer_global) ? owner_by_offset[off] : 0;
        int target_rank = 0;
        int irank;
        predicted_loads[source_rank] -= plan->records[itr].sum_nsub;
        predicted_counts[source_rank] -= plan->records[itr].count;
        for (irank = 1; irank < mpi_size; irank++) {
            if (predicted_loads[irank] < predicted_loads[target_rank] ||
                (predicted_loads[irank] == predicted_loads[target_rank] &&
                 predicted_counts[irank] < predicted_counts[target_rank])) {
                target_rank = irank;
            }
        }
        plan->records[itr].target_rank = target_rank;
        predicted_loads[target_rank] += plan->records[itr].sum_nsub;
        predicted_counts[target_rank] += plan->records[itr].count;
    }
    load_balance_refine_assignments(plan->records, nheavy, mpi_size,
                                              predicted_loads, predicted_counts);
    qsort(plan->records, (size_t)nheavy, sizeof(*plan->records),
          tracer_load_balance_record_compare_id_asc);
    plan->nheavy = nheavy;
    plan->enabled = 1;

    if (output_dir != 0 && *output_dir != '\0') {
        if (write_load_balance_files(output_dir, run_tracer_start, ntracer_global,
                                               mpi_size, top_frac, plan,
                                               all_ids, all_sums, total_count) != 0) {
            goto cleanup;
        }
        if (write_planning_rank_totals(output_dir, run_tracer_start, ntracer_global,
                                                 mpi_size, top_frac, all_ids, all_sums,
                                                 total_count, plan) != 0) {
            goto cleanup;
        }
    }

    free(sum_by_offset);
    free(predicted_loads);
    free(predicted_counts);
    free(owner_by_offset);
    return 0;

cleanup:
    tracer_fp_lb_plan_free(plan);
    free(sum_by_offset);
    free(predicted_loads);
    free(predicted_counts);
    free(owner_by_offset);
    return -1;
}

int tracer_fp_lb_plan_build(long int run_tracer_start,
                                      long int tracer_start,
                                      int ntracer_global,
                                      int mpi_rank,
                                      int mpi_size,
                                      int ntracer_local,
                                      const long long *sum_nsub_local,
                                      double top_frac,
                                      const char *output_dir,
                                      TracerLoadBalancePlan *plan)
{
    int *recvcounts = 0;
    int *displs = 0;
    long int *all_ids = 0;
    long long *all_sums = 0;
    long int *send_ids = 0;
    long long *send_sums = 0;
    int local_count = ntracer_local;
    int total_count = 0;
    int itr;
    int ierr = -1;

    if (plan == 0) return -1;
    plan->enabled = 0;
    plan->nheavy = 0;
    plan->chunk_size = 0;
    plan->records = 0;
    if (ntracer_global <= 0 || mpi_size <= 0) return 0;

    if (mpi_rank == 0) {
        printf("  load balance        : gathering estimated per-tracer costs from %d ranks\n",
               mpi_size);
        fflush(stdout);
    }

    send_ids = (long int *)calloc((size_t)((local_count > 0) ? local_count : 1), sizeof(long int));
    send_sums = (long long *)calloc((size_t)((local_count > 0) ? local_count : 1), sizeof(long long));
    if (send_ids == 0 || send_sums == 0) goto cleanup;
    for (itr = 0; itr < local_count; itr++) {
        send_ids[itr] = tracer_start + (long int)itr;
        send_sums[itr] = (sum_nsub_local != 0) ? sum_nsub_local[itr] : 0;
    }

    if (mpi_rank == 0) {
        recvcounts = (int *)calloc((size_t)mpi_size, sizeof(int));
        displs = (int *)calloc((size_t)mpi_size, sizeof(int));
        if (recvcounts == 0 || displs == 0) goto cleanup;
    }
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        for (itr = 0; itr < mpi_size; itr++) {
            displs[itr] = total_count;
            total_count += recvcounts[itr];
        }
        if (total_count != ntracer_global) {
            fprintf(stderr, TRACER_FP_PROGNAME ": load-balance gather mismatch total=%d expected=%d\n",
                    total_count, ntracer_global);
            goto cleanup;
        }
        all_ids = (long int *)calloc((size_t)total_count, sizeof(long int));
        all_sums = (long long *)calloc((size_t)total_count, sizeof(long long));
        if (all_ids == 0 || all_sums == 0) goto cleanup;
    }

    MPI_Gatherv(send_ids, local_count, MPI_LONG,
                all_ids, recvcounts, displs, MPI_LONG,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(send_sums, local_count, MPI_LONG_LONG,
                all_sums, recvcounts, displs, MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("  load balance        : building heavy-tracer plan top_frac=%.4f\n",
               top_frac);
        fflush(stdout);
        if (tracer_fp_lb_plan_build_global(run_tracer_start, ntracer_global,
                                                          mpi_size, top_frac, output_dir,
                                                          all_ids, all_sums, total_count,
                                                          plan) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to build load-balance reports for '%s'\n",
                    (output_dir != 0) ? output_dir : "");
            goto cleanup;
        }
    }

    MPI_Bcast(&plan->enabled, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&plan->nheavy, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&plan->chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (plan->nheavy > 0) {
        if (mpi_rank != 0) {
            plan->records = (TracerLoadBalanceRecord *)calloc((size_t)plan->nheavy, sizeof(TracerLoadBalanceRecord));
            if (plan->records == 0) goto cleanup;
        }
        MPI_Bcast(plan->records, (int)(plan->nheavy * (int)sizeof(TracerLoadBalanceRecord)),
                  MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    if (mpi_rank == 0) {
        printf("  load balance        : plan ready, heavy tracers=%d\n", plan->nheavy);
        fflush(stdout);
    }

    ierr = 0;

cleanup:
    if (ierr != 0) tracer_fp_lb_plan_free(plan);
    free(recvcounts);
    free(displs);
    free(all_ids);
    free(all_sums);
    free(send_ids);
    free(send_sums);
    return ierr;
}

int tracer_fp_lb_report_baseline(long int run_tracer_start,
                                 long int tracer_start,
                                 int ntracer_global,
                                 int mpi_rank,
                                 int mpi_size,
                                 int ntracer_local,
                                 const long long *sum_nsub_local,
                                 double top_frac,
                                 const char *output_dir)
{
    int *recvcounts = 0;
    int *displs = 0;
    long int *all_ids = 0;
    long long *all_sums = 0;
    long int *send_ids = 0;
    long long *send_sums = 0;
    int local_count = ntracer_local;
    int total_count = 0;
    int itr;
    int ierr = -1;

    send_ids = (long int *)calloc((size_t)((local_count > 0) ? local_count : 1), sizeof(long int));
    send_sums = (long long *)calloc((size_t)((local_count > 0) ? local_count : 1), sizeof(long long));
    if (send_ids == 0 || send_sums == 0) goto cleanup;
    for (itr = 0; itr < local_count; itr++) {
        send_ids[itr] = tracer_start + (long int)itr;
        send_sums[itr] = (sum_nsub_local != 0) ? sum_nsub_local[itr] : 0;
    }

    if (mpi_rank == 0) {
        recvcounts = (int *)calloc((size_t)mpi_size, sizeof(int));
        displs = (int *)calloc((size_t)mpi_size, sizeof(int));
        if (recvcounts == 0 || displs == 0) goto cleanup;
    }
    MPI_Gather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        for (itr = 0; itr < mpi_size; itr++) {
            displs[itr] = total_count;
            total_count += recvcounts[itr];
        }
        if (total_count != ntracer_global) {
            fprintf(stderr, TRACER_FP_PROGNAME ": baseline gather mismatch total=%d expected=%d\n",
                    total_count, ntracer_global);
            goto cleanup;
        }
        all_ids = (long int *)calloc((size_t)total_count, sizeof(long int));
        all_sums = (long long *)calloc((size_t)total_count, sizeof(long long));
        if (all_ids == 0 || all_sums == 0) goto cleanup;
    }

    MPI_Gatherv(send_ids, local_count, MPI_LONG,
                all_ids, recvcounts, displs, MPI_LONG,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(send_sums, local_count, MPI_LONG_LONG,
                all_sums, recvcounts, displs, MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        if (write_planning_rank_totals(output_dir, run_tracer_start, ntracer_global,
                                                 mpi_size, top_frac, all_ids, all_sums,
                                                 total_count, 0) != 0) {
            goto cleanup;
        }
    }

    ierr = 0;

cleanup:
    free(recvcounts);
    free(displs);
    free(all_ids);
    free(all_sums);
    free(send_ids);
    free(send_sums);
    return ierr;
}

int tracer_fp_lb_active_capacity(long int run_tracer_start,
                                 long int tracer_start,
                                 int ntracer_local,
                                 const TracerLoadBalancePlan *plan,
                                 int mpi_rank,
                                 int *nactive_out)
{
    int local_removed = 0;
    int assigned = 0;
    int irec;

    (void)run_tracer_start;
    if (nactive_out != 0) *nactive_out = 0;
    if (nactive_out == 0 || ntracer_local < 0) return -1;

    if (plan == 0 || !plan->enabled || plan->records == 0) {
        *nactive_out = ntracer_local;
        return 0;
    }

    for (irec = 0; irec < plan->nheavy; irec++) {
        const long int rec_start = plan->records[irec].tracer_id;
        const long int rec_end = rec_start + (long int)plan->records[irec].count;
        const long int local_start = tracer_start;
        const long int local_end = tracer_start + (long int)ntracer_local;
        const long int overlap_start = (rec_start > local_start) ? rec_start : local_start;
        const long int overlap_end = (rec_end < local_end) ? rec_end : local_end;

        if (overlap_end > overlap_start) {
            local_removed += (int)(overlap_end - overlap_start);
        }
        if (plan->records[irec].target_rank == mpi_rank) {
            assigned += plan->records[irec].count;
        }
    }

    *nactive_out = ntracer_local - local_removed + assigned;
    return 0;
}

int tracer_fp_lb_select(long int run_tracer_start,
                                       long int tracer_start,
                                       int ntracer_local,
                                       const TracerLoadBalancePlan *plan,
                                       int mpi_rank,
                                       long int *active_global_ids,
                                       int *active_source_offsets,
                                       int *nactive_out)
{
    unsigned char *local_heavy_mask = 0;
    TracerLoadBalanceSelectionPair *pairs = 0;
    int local_heavy_count = 0;
    int nactive = 0;
    int iheavy;
    int itr;

    if (nactive_out != 0) *nactive_out = 0;
    if (active_global_ids == 0 || active_source_offsets == 0 || nactive_out == 0) return -1;
    if (ntracer_local < 0) return -1;

    local_heavy_mask = (unsigned char *)calloc((size_t)((ntracer_local > 0) ? ntracer_local : 1),
                                               sizeof(unsigned char));
    if (local_heavy_mask == 0) return -1;

    if (plan != 0 && plan->enabled) {
        for (iheavy = 0; iheavy < plan->nheavy; iheavy++) {
            const long int rec_start = plan->records[iheavy].tracer_id;
            const long int rec_end = rec_start + (long int)plan->records[iheavy].count;
            const long int local_start = tracer_start;
            const long int local_end = tracer_start + (long int)ntracer_local;
            const long int overlap_start = (rec_start > local_start) ? rec_start : local_start;
            const long int overlap_end = (rec_end < local_end) ? rec_end : local_end;
            long int gid;

            for (gid = overlap_start; gid < overlap_end; gid++) {
                const int local_idx = (int)(gid - tracer_start);
                if (local_idx >= 0 && local_idx < ntracer_local &&
                    !local_heavy_mask[local_idx]) {
                    local_heavy_mask[local_idx] = 1;
                    local_heavy_count++;
                }
            }
        }
    }

    for (itr = 0; itr < ntracer_local; itr++) {
        if (local_heavy_mask[itr]) continue;
        active_global_ids[nactive] = tracer_start + (long int)itr;
        active_source_offsets[nactive] = (int)((tracer_start - run_tracer_start) + (long int)itr);
        nactive++;
    }

    if (plan != 0 && plan->enabled) {
        for (iheavy = 0; iheavy < plan->nheavy; iheavy++) {
            long int gid;
            if (plan->records[iheavy].target_rank != mpi_rank) continue;
            for (gid = plan->records[iheavy].tracer_id;
                 gid < plan->records[iheavy].tracer_id + (long int)plan->records[iheavy].count;
                 gid++) {
                active_global_ids[nactive] = gid;
                active_source_offsets[nactive] = (int)(gid - run_tracer_start);
                nactive++;
            }
        }
    }

    if (nactive > 1) {
        pairs = (TracerLoadBalanceSelectionPair *)calloc((size_t)nactive, sizeof(*pairs));
        if (pairs == 0) {
            free(local_heavy_mask);
            return -1;
        }
        for (itr = 0; itr < nactive; itr++) {
            pairs[itr].source_offset = active_source_offsets[itr];
            pairs[itr].global_id = active_global_ids[itr];
        }
        qsort(pairs, (size_t)nactive, sizeof(*pairs),
              tracer_load_balance_selection_pair_compare);
        for (itr = 0; itr < nactive; itr++) {
            active_source_offsets[itr] = pairs[itr].source_offset;
            active_global_ids[itr] = pairs[itr].global_id;
        }
        free(pairs);
    }

    *nactive_out = nactive;
    free(local_heavy_mask);
    return local_heavy_count;
}

int tracer_fp_nsub_estimate(int ntracer,
                                long int tracer_start,
                                int ntracer_global,
                                int nsnap,
                                TracerFpInputMode input_mode,
                                TracerFpBackgroundMode background_mode,
                                const char *params_file,
                                long long *sum_nsub,
                                long long *snapshot_sum_nsub,
                                long long *snapshot_target_nsub)
{
    const size_t bg_slot_storage_size = 2u * (size_t)ntracer;
    CRspectrum crp_grid, cre_grid;
    double beta_p[np], gamma2e[npe];
    double *dt_snap = 0, *z_snap = 0;
    double *n_gas_hist = 0, *kbt_hist = 0, *b_field_hist = 0, *divv_hist = 0;
    double *l_turb_hist = 0, *dv_imc_hist = 0, *cs_hist = 0, *beta_pl_hist = 0;
    TracerDataHistory bg_history;
    TracerFpBackgroundSlot bg_slots[2];
    int bg_curr_slot = 0;
    int bg_next_slot = 1;
    TracerFpHdf5Meta hdf5_meta;
    double *temp_raw_ring = 0, *rho_raw_ring = 0, *bx_raw_ring = 0, *by_raw_ring = 0;
    double *bz_raw_ring = 0, *divv_raw_ring = 0, *rotv_raw_ring = 0, *lturb_raw_ring = 0;
    double *mach_raw_ring = 0, *prestemp_raw_ring = 0, *presden_raw_ring = 0;
    double *pre_density_cgs_raw_ring = 0, *upstream_speed_cgs_raw_ring = 0;
    double *shock_side_code_raw_ring = 0;
    double *dsa_trigger_raw_ring = 0, *dsa_mach_raw_ring = 0;
    double *dsa_pre_density_raw_ring = 0, *dsa_flux_raw_ring = 0;
    TracerDataStorage raw_storage;
    TracerFpRawBackgroundSlot raw_slots[3];
    int raw_prev_slot = 0;
    int raw_curr_slot = 1;
    int raw_next_slot = 2;
    long int *local_tracer_ids = 0;
    double *crp_radp_lo = 0, *crp_radp_hi = 0;
    double *cre_radp_lo = 0, *cre_radp_hi = 0;
    int *nsubsteps = 0, *target_nsubsteps = 0, *n_onsteps = 0;
    unsigned char *capped_flags = 0;
    int isnap, itr;
    int ierr = -1;

    if (ntracer <= 0 || ntracer_global <= 0 || nsnap <= 0 || sum_nsub == 0) return 0;

    memset(&crp_grid, 0, sizeof(crp_grid));
    memset(&cre_grid, 0, sizeof(cre_grid));
    memset(&bg_history, 0, sizeof(bg_history));
    memset(bg_slots, 0, sizeof(bg_slots));
    memset(&hdf5_meta, 0, sizeof(hdf5_meta));
    memset(&raw_storage, 0, sizeof(raw_storage));
    memset(raw_slots, 0, sizeof(raw_slots));

    for (itr = 0; itr < ntracer; itr++) sum_nsub[itr] = 0;
    if (snapshot_sum_nsub != 0) {
        for (itr = 0; itr < nsnap; itr++) snapshot_sum_nsub[itr] = 0;
    }
    if (snapshot_target_nsub != 0) {
        for (itr = 0; itr < nsnap; itr++) snapshot_target_nsub[itr] = 0;
    }

    init_test_params();
    if (params_file != 0 && *params_file != '\0') {
        if (read_param_file_noMPI(params_file) != SUCCESS) goto cleanup;
    }
    if (finalize_dpp_mode_config(0) != SUCCESS) {
        goto cleanup;
    }
    momentum_bin(&crp_grid, &cre_grid, beta_p, gamma2e);

    dt_snap = (double *)calloc((size_t)nsnap, sizeof(double));
    z_snap = (double *)calloc((size_t)nsnap, sizeof(double));
    n_gas_hist = (double *)calloc(bg_slot_storage_size, sizeof(double));
    kbt_hist = (double *)calloc(bg_slot_storage_size, sizeof(double));
    b_field_hist = (double *)calloc(bg_slot_storage_size, sizeof(double));
    divv_hist = (double *)calloc(bg_slot_storage_size, sizeof(double));
    l_turb_hist = (double *)calloc(bg_slot_storage_size, sizeof(double));
    dv_imc_hist = (double *)calloc(bg_slot_storage_size, sizeof(double));
    cs_hist = (double *)calloc(bg_slot_storage_size, sizeof(double));
    beta_pl_hist = (double *)calloc(bg_slot_storage_size, sizeof(double));
    local_tracer_ids = (long int *)calloc((size_t)ntracer, sizeof(long int));
    crp_radp_lo = (double *)calloc((size_t)ntracer, sizeof(double));
    crp_radp_hi = (double *)calloc((size_t)ntracer, sizeof(double));
    cre_radp_lo = (double *)calloc((size_t)ntracer, sizeof(double));
    cre_radp_hi = (double *)calloc((size_t)ntracer, sizeof(double));
    nsubsteps = (int *)calloc((size_t)ntracer, sizeof(int));
    target_nsubsteps = (int *)calloc((size_t)ntracer, sizeof(int));
    n_onsteps = (int *)calloc((size_t)ntracer, sizeof(int));
    capped_flags = (unsigned char *)calloc((size_t)ntracer, sizeof(unsigned char));
    if (dt_snap == 0 || z_snap == 0 ||
        n_gas_hist == 0 || kbt_hist == 0 || b_field_hist == 0 || divv_hist == 0 ||
        l_turb_hist == 0 || dv_imc_hist == 0 || cs_hist == 0 || beta_pl_hist == 0 ||
        local_tracer_ids == 0 || crp_radp_lo == 0 || crp_radp_hi == 0 ||
        cre_radp_lo == 0 || cre_radp_hi == 0 || nsubsteps == 0 ||
        target_nsubsteps == 0 || n_onsteps == 0 || capped_flags == 0) {
        goto cleanup;
    }

    bg_history.n_gas = n_gas_hist;
    bg_history.kbt = kbt_hist;
    bg_history.b_field = b_field_hist;
    bg_history.divv = divv_hist;
    bg_history.l_turb = l_turb_hist;
    bg_history.dv_imc = dv_imc_hist;
    bg_history.cs = cs_hist;
    bg_history.beta_pl = beta_pl_hist;

    tracer_bind_background_slot(ntracer, 0, &bg_history, &bg_slots[0]);
    tracer_bind_background_slot(ntracer, 1, &bg_history, &bg_slots[1]);

    for (itr = 0; itr < ntracer; itr++) {
        local_tracer_ids[itr] = tracer_start + (long int)itr;
    }

    if (input_mode == TRACER_FP_INPUT_HDF5) {
        temp_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        rho_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        bx_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        by_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        bz_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        divv_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        rotv_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        lturb_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        mach_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        prestemp_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        presden_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        pre_density_cgs_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        upstream_speed_cgs_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        shock_side_code_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        dsa_trigger_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        dsa_mach_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        dsa_pre_density_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        dsa_flux_raw_ring = (double *)calloc(3u * (size_t)ntracer, sizeof(double));
        if (temp_raw_ring == 0 || rho_raw_ring == 0 || bx_raw_ring == 0 || by_raw_ring == 0 ||
            bz_raw_ring == 0 || divv_raw_ring == 0 || rotv_raw_ring == 0 || lturb_raw_ring == 0 ||
            mach_raw_ring == 0 || prestemp_raw_ring == 0 || presden_raw_ring == 0 ||
            pre_density_cgs_raw_ring == 0 || upstream_speed_cgs_raw_ring == 0 ||
            shock_side_code_raw_ring == 0 || dsa_trigger_raw_ring == 0 ||
            dsa_mach_raw_ring == 0 || dsa_pre_density_raw_ring == 0 ||
            dsa_flux_raw_ring == 0) {
            goto cleanup;
        }

        raw_storage.temp = temp_raw_ring;
        raw_storage.rho = rho_raw_ring;
        raw_storage.bx = bx_raw_ring;
        raw_storage.by = by_raw_ring;
        raw_storage.bz = bz_raw_ring;
        raw_storage.divv = divv_raw_ring;
        raw_storage.rotv = rotv_raw_ring;
        raw_storage.lturb = lturb_raw_ring;
        raw_storage.mach = mach_raw_ring;
        raw_storage.prestemp = prestemp_raw_ring;
        raw_storage.presden = presden_raw_ring;
        raw_storage.pre_density_cgs = pre_density_cgs_raw_ring;
        raw_storage.upstream_speed_cgs = upstream_speed_cgs_raw_ring;
        raw_storage.shock_side_code = shock_side_code_raw_ring;
        raw_storage.dsa_trigger = dsa_trigger_raw_ring;
        raw_storage.dsa_mach = dsa_mach_raw_ring;
        raw_storage.dsa_pre_density = dsa_pre_density_raw_ring;
        raw_storage.dsa_kinetic_energy_flux_cgs = dsa_flux_raw_ring;

        tracer_bind_raw_background_slot(ntracer, 0, &raw_storage, &raw_slots[0]);
        tracer_bind_raw_background_slot(ntracer, 1, &raw_storage, &raw_slots[1]);
        tracer_bind_raw_background_slot(ntracer, 2, &raw_storage, &raw_slots[2]);

        if (tracer_init_hdf5_meta(&hdf5_meta, params_file, ntracer_global) != 0 ||
            tracer_fill_hdf5_timeline(&hdf5_meta, dt_snap, z_snap, nsnap,
                                      background_mode, params_file) != 0) {
            goto cleanup;
        }

        if (background_mode == TRACER_FP_BACKGROUND_FROZEN) {
            if (tracer_load_hdf5_raw_snapshot_slice(&hdf5_meta, nsnp_i,
                                                    tracer_start, 0, ntracer,
                                                    &raw_slots[0]) != 0 ||
                tracer_prepare_background_from_raw(ntracer,
                                                   &raw_slots[0], &raw_slots[0], &raw_slots[0],
                                                   z_snap[0], z_snap[0], z_snap[0],
                                                   0, 0,
                                                   &bg_slots[bg_curr_slot]) != 0) {
                goto cleanup;
            }
            tracer_copy_background_slot(ntracer, &bg_slots[bg_next_slot], &bg_slots[bg_curr_slot]);
        } else {
            if (tracer_load_hdf5_raw_snapshot_slice(&hdf5_meta, nsnp_i + 0,
                                                    tracer_start, 0, ntracer,
                                                    &raw_slots[raw_prev_slot]) != 0 ||
                (nsnap > 1 &&
                 tracer_load_hdf5_raw_snapshot_slice(&hdf5_meta, nsnp_i + 1,
                                                     tracer_start, 0, ntracer,
                                                     &raw_slots[raw_curr_slot]) != 0) ||
                (nsnap > 2 &&
                 tracer_load_hdf5_raw_snapshot_slice(&hdf5_meta, nsnp_i + 2,
                                                     tracer_start, 0, ntracer,
                                                     &raw_slots[raw_next_slot]) != 0)) {
                goto cleanup;
            }
            if (tracer_prepare_background_from_raw(ntracer,
                                                   &raw_slots[raw_prev_slot],
                                                   &raw_slots[raw_prev_slot],
                                                   (nsnap > 1) ? &raw_slots[raw_curr_slot]
                                                               : &raw_slots[raw_prev_slot],
                                                   z_snap[0], z_snap[0],
                                                   (nsnap > 1) ? z_snap[1] : z_snap[0],
                                                   0, 0,
                                                   &bg_slots[bg_curr_slot]) != 0) {
                goto cleanup;
            }
            if (nsnap > 1) {
                if (tracer_prepare_background_from_raw(ntracer,
                                                       &raw_slots[raw_prev_slot],
                                                       &raw_slots[raw_curr_slot],
                                                       (nsnap > 2) ? &raw_slots[raw_next_slot]
                                                                   : &raw_slots[raw_curr_slot],
                                                       z_snap[0], z_snap[1],
                                                       (nsnap > 2) ? z_snap[2] : z_snap[1],
                                                       1, (nsnap > 2),
                                                       &bg_slots[bg_next_slot]) != 0) {
                    goto cleanup;
                }
            } else {
                tracer_copy_background_slot(ntracer, &bg_slots[bg_next_slot], &bg_slots[bg_curr_slot]);
            }
        }
    } else {
        for (itr = 0; itr < ntracer; itr++) {
            local_tracer_ids[itr] = tracer_start + (long int)itr;
        }
        tracer_fp_fill_timeline_synthetic(dt_snap, z_snap, nsnap);
        if (tracer_fp_fill_background_snapshot_synthetic(0, z_snap, ntracer,
                                                         local_tracer_ids,
                                                         ntracer_global, nsnap,
                                                         &bg_slots[bg_curr_slot]) != 0) {
            goto cleanup;
        }
        if (background_mode == TRACER_FP_BACKGROUND_FROZEN || nsnap == 1) {
            tracer_copy_background_slot(ntracer, &bg_slots[bg_next_slot], &bg_slots[bg_curr_slot]);
        } else if (tracer_fp_fill_background_snapshot_synthetic(1, z_snap, ntracer,
                                                                local_tracer_ids,
                                                                ntracer_global, nsnap,
                                                                &bg_slots[bg_next_slot]) != 0) {
            goto cleanup;
        }
    }

    for (isnap = 0; isnap < nsnap; isnap++) {
        const double *n_gas_snap = bg_slots[bg_curr_slot].n_gas;
        const double *kbt_snap = bg_slots[bg_curr_slot].kbt;
        const double *b_field_snap = bg_slots[bg_curr_slot].b_field;
        const double *divv_snap = bg_slots[bg_curr_slot].divv;
        const double *lturb_snap = bg_slots[bg_curr_slot].l_turb;
        const double *dv_snap = bg_slots[bg_curr_slot].dv_imc;
        double rad_ic_row[npe];
        double rad_ic_m1_val = 0.0;
        double rad_ic_p1_val = 0.0;
        memset(capped_flags, 0, (size_t)ntracer * sizeof(unsigned char));

        if (prepare_ic_cooling_row(z_snap[isnap], &cre_grid,
                                   rad_ic_row,
                                   &rad_ic_m1_val,
                                   &rad_ic_p1_val) != 0) {
            goto cleanup;
        }
        (void)rad_ic_m1_val;
        (void)rad_ic_p1_val;
        for (itr = 0; itr < ntracer; itr++) {
            crp_radp_lo[itr] = b_Coulomb_p(n_gas_snap[itr], crp_grid.p[0], kbt_snap[itr]) +
                               adiabatic_divv(divv_snap[itr], crp_grid.p[0]);
            crp_radp_hi[itr] = b_Coulomb_p(n_gas_snap[itr], crp_grid.p[np - 1], kbt_snap[itr]) +
                               adiabatic_divv(divv_snap[itr], crp_grid.p[np - 1]);
            cre_radp_lo[itr] = b_synch(cre_grid.p[0], b_field_snap[itr]) + rad_ic_row[0] +
                               b_Coulomb_e(n_gas_snap[itr], cre_grid.p[0]) +
                               adiabatic_divv(divv_snap[itr], cre_grid.p[0]);
            cre_radp_hi[itr] = b_synch(cre_grid.p[npe - 1], b_field_snap[itr]) +
                               rad_ic_row[npe - 1] +
                               b_Coulomb_e(n_gas_snap[itr], cre_grid.p[npe - 1]) +
                               adiabatic_divv(divv_snap[itr], cre_grid.p[npe - 1]);
        }
        if (tracer_fp_compute_nsubsteps_from_endpoints(
                ntracer, dt_snap[isnap], &crp_grid, &cre_grid,
                crp_radp_lo, crp_radp_hi, cre_radp_lo, cre_radp_hi,
                lturb_snap, dv_snap,
                nsubsteps, target_nsubsteps, n_onsteps, capped_flags,
                0, 0, 0) != 0) {
            goto cleanup;
        }
        {
            long long snapshot_sum = 0;
            long long snapshot_target_sum = 0;
            for (itr = 0; itr < ntracer; itr++) {
                sum_nsub[itr] += (long long)nsubsteps[itr];
                snapshot_sum += (long long)nsubsteps[itr];
                snapshot_target_sum += (long long)target_nsubsteps[itr];
            }
            if (snapshot_sum_nsub != 0) snapshot_sum_nsub[isnap] = snapshot_sum;
            if (snapshot_target_nsub != 0) snapshot_target_nsub[isnap] = snapshot_target_sum;
        }

        if (isnap + 1 < nsnap) {
            const int old_curr_slot = bg_curr_slot;
            bg_curr_slot = bg_next_slot;
            bg_next_slot = old_curr_slot;

            if (background_mode == TRACER_FP_BACKGROUND_EVOLVING && isnap + 2 < nsnap) {
                if (input_mode == TRACER_FP_INPUT_HDF5) {
                    const int recycle_slot = raw_prev_slot;
                    raw_prev_slot = raw_curr_slot;
                    raw_curr_slot = raw_next_slot;
                    raw_next_slot = recycle_slot;

                    if (isnap + 3 < nsnap &&
                        tracer_load_hdf5_raw_snapshot_slice(&hdf5_meta,
                                                            nsnp_i + isnap + 3,
                                                            tracer_start, 0, ntracer,
                                                            &raw_slots[raw_next_slot]) != 0) {
                        goto cleanup;
                    }
                    if (tracer_prepare_background_from_raw(
                            ntracer,
                            &raw_slots[raw_prev_slot],
                            &raw_slots[raw_curr_slot],
                            (isnap + 3 < nsnap) ? &raw_slots[raw_next_slot]
                                                : &raw_slots[raw_curr_slot],
                            z_snap[isnap + 1],
                            z_snap[isnap + 2],
                            (isnap + 3 < nsnap) ? z_snap[isnap + 3] : z_snap[isnap + 2],
                            1,
                            (isnap + 3 < nsnap),
                            &bg_slots[bg_next_slot]) != 0) {
                        goto cleanup;
                    }
                } else if (tracer_fp_fill_background_snapshot_synthetic(isnap + 2, z_snap, ntracer,
                                                                       local_tracer_ids,
                                                                       ntracer_global,
                                                                       nsnap, &bg_slots[bg_next_slot]) != 0) {
                    goto cleanup;
                }
            }
        }
    }

    ierr = 0;

cleanup:
    tracer_free_hdf5_meta(&hdf5_meta);
    free(dt_snap); free(z_snap);
    free(n_gas_hist); free(kbt_hist); free(b_field_hist); free(divv_hist);
    free(l_turb_hist); free(dv_imc_hist); free(cs_hist); free(beta_pl_hist);
    free(temp_raw_ring); free(rho_raw_ring); free(bx_raw_ring); free(by_raw_ring);
    free(bz_raw_ring); free(divv_raw_ring); free(rotv_raw_ring); free(lturb_raw_ring);
    free(mach_raw_ring); free(prestemp_raw_ring); free(presden_raw_ring);
    free(pre_density_cgs_raw_ring); free(upstream_speed_cgs_raw_ring);
    free(shock_side_code_raw_ring);
    free(dsa_trigger_raw_ring); free(dsa_mach_raw_ring);
    free(dsa_pre_density_raw_ring); free(dsa_flux_raw_ring);
    free(local_tracer_ids);
    free(crp_radp_lo); free(crp_radp_hi); free(cre_radp_lo); free(cre_radp_hi);
    free(nsubsteps); free(target_nsubsteps); free(n_onsteps); free(capped_flags);
    return ierr;
}

int tracer_fp_load_est(long int run_tracer_start,
                                int ntracer,
                                int nsnap,
                                TracerFpInputMode input_mode,
                                TracerFpBackgroundMode background_mode,
                                const char *params_file,
                                const char *output_dir,
                                int virtual_ranks,
                                double top_frac)
{
    long int *all_ids = 0;
    long long *all_sums = 0;
    TracerLoadBalancePlan plan = {0, 0, 0, 0};
    int itr;
    int ierr = -1;

    if (ntracer <= 0 || virtual_ranks <= 0) return -1;
    all_ids = (long int *)calloc((size_t)ntracer, sizeof(long int));
    all_sums = (long long *)calloc((size_t)ntracer, sizeof(long long));
    if (all_ids == 0 || all_sums == 0) goto cleanup;

    printf("  load_estimate       : estimating sum_nsub for %d tracers, %d snapshots\n",
           ntracer, nsnap);
    fflush(stdout);
    for (itr = 0; itr < ntracer; itr++) {
        all_ids[itr] = run_tracer_start + (long int)itr;
    }
    if (tracer_fp_nsub_estimate(ntracer, run_tracer_start, ntracer,
                                    nsnap, input_mode, background_mode,
                                    params_file, all_sums, 0, 0) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": load_estimate sum_nsub pass failed\n");
        goto cleanup;
    }
    if (load_balancing) {
        printf("  load_estimate       : building virtual-rank plan for %d ranks\n",
               virtual_ranks);
    } else {
        printf("  load_estimate       : writing baseline virtual-rank totals for %d ranks\n",
               virtual_ranks);
    }
    fflush(stdout);
    if (load_balancing) {
        if (tracer_fp_lb_plan_build_global(run_tracer_start, ntracer, virtual_ranks,
                                                          top_frac, output_dir,
                                                          all_ids, all_sums, ntracer,
                                                          &plan) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": load_estimate plan build failed\n");
            goto cleanup;
        }
    } else if (output_dir != 0 && *output_dir != '\0') {
        if (write_planning_rank_totals(output_dir, run_tracer_start, ntracer,
                                                 virtual_ranks, top_frac, all_ids, all_sums,
                                                 ntracer, 0) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": load_estimate baseline report write failed\n");
            goto cleanup;
        }
    }

    printf(TRACER_FP_PROGNAME " load_estimate summary\n");
    printf("  virtual ranks       : %d\n", virtual_ranks);
    printf("  heavy top frac      : %.4f\n", top_frac);
    printf("  LB chunk size       : %d\n", load_balance_chunk_size);
    printf("  LB diagnostics      : %d\n", load_balance_diagnostics);
    printf("  global ntracer      : %d\n", ntracer);
    printf("  output dir          : %s\n", output_dir);
    if (load_balancing) {
        printf("  reports             : planning_rank_totals.tsv, load_balance_rank_totals.txt, heavy_assignments.txt, heavy_summary.txt%s\n",
               load_balance_diagnostics ? ", load_balance_chunk_diagnostics.tsv" : "");
    } else {
        printf("  reports             : planning_rank_totals.tsv%s\n",
               load_balance_diagnostics ? ", load_balance_chunk_diagnostics.tsv" : "");
    }
    ierr = 0;

cleanup:
    tracer_fp_lb_plan_free(&plan);
    free(all_ids);
    free(all_sums);
    return ierr;
}
