/*
    tracer_fp_bucket.c

    K. Nishiwaki, 2026-06-18
    - tracers are "bucketed" based on their current step properties (nsubsteps, n_onsteps)
    - allocation/free funcs
*/


#include <stdlib.h>
#include <string.h>

#include "tracer_fp_bucket.h"
#include "tracer_fp_setup.h"

void tracer_bucket_reset(TracerBucket *bucket)
{
    if (bucket == 0) return;
    memset(bucket, 0, sizeof(*bucket));
}

int tracer_bucket_alloc(TracerBucket *bucket, int ntracer)
{
    if (bucket == 0 || ntracer <= 0) return -1;

    bucket->steps = (int *)calloc((size_t)ntracer, sizeof(int));
    bucket->onsteps = (int *)calloc((size_t)ntracer, sizeof(int));
    bucket->indices = (int *)calloc((size_t)ntracer, sizeof(int));
    bucket->bucket_ids = (int *)calloc((size_t)ntracer, sizeof(int));
    bucket->member_offsets = (int *)calloc((size_t)ntracer, sizeof(int));
    bucket->member_counts = (int *)calloc((size_t)ntracer, sizeof(int));
    bucket->member_write = (int *)calloc((size_t)ntracer, sizeof(int));

    if (bucket->steps == 0 || bucket->onsteps == 0 || bucket->indices == 0 ||
        bucket->bucket_ids == 0 || bucket->member_offsets == 0 ||
        bucket->member_counts == 0 || bucket->member_write == 0) {
        return -1;
    }
    return 0;
}

void tracer_bucket_release(TracerBucket *bucket)
{
    if (bucket == 0) return;
    free(bucket->steps);
    free(bucket->onsteps);
    free(bucket->indices);
    free(bucket->bucket_ids);
    free(bucket->member_offsets);
    free(bucket->member_counts);
    free(bucket->member_write);
    tracer_bucket_reset(bucket);
}

void tracer_gpu_bucket_groups_reset(TracerGpuBucketGroups *groups)
{
    if (groups == 0) return;
    memset(groups, 0, sizeof(*groups));
}

int tracer_gpu_bucket_groups_alloc(TracerGpuBucketGroups *groups, int nbuckets_max)
{
    if (groups == 0 || nbuckets_max <= 0) return -1;

    groups->capacity_nbuckets = nbuckets_max;
    groups->group_offsets = (int *)calloc((size_t)nbuckets_max + 1u, sizeof(int));
    groups->group_bucket_ids = (int *)calloc((size_t)nbuckets_max, sizeof(int));
    groups->bucket_group_ids = (int *)calloc((size_t)nbuckets_max, sizeof(int));
    groups->group_bucket_counts = (int *)calloc((size_t)nbuckets_max, sizeof(int));
    groups->group_total_cells = (int *)calloc((size_t)nbuckets_max, sizeof(int));
    groups->group_coeff_segments = (int *)calloc((size_t)nbuckets_max, sizeof(int));
    groups->group_n_on = (int *)calloc((size_t)nbuckets_max, sizeof(int));
    groups->group_nstep_min = (int *)calloc((size_t)nbuckets_max, sizeof(int));
    groups->group_nstep_max = (int *)calloc((size_t)nbuckets_max, sizeof(int));

    if (groups->group_offsets == 0 || groups->group_bucket_ids == 0 ||
        groups->bucket_group_ids == 0 ||
        groups->group_bucket_counts == 0 || groups->group_total_cells == 0 ||
        groups->group_coeff_segments == 0 ||
        groups->group_n_on == 0 || groups->group_nstep_min == 0 ||
        groups->group_nstep_max == 0) {
        free(groups->group_offsets);
        free(groups->group_bucket_ids);
        free(groups->bucket_group_ids);
        free(groups->group_bucket_counts);
        free(groups->group_total_cells);
        free(groups->group_coeff_segments);
        free(groups->group_n_on);
        free(groups->group_nstep_min);
        free(groups->group_nstep_max);
        tracer_gpu_bucket_groups_reset(groups);
        return -1;
    }
    return 0;
}

void tracer_gpu_bucket_groups_release(TracerGpuBucketGroups *groups)
{
    if (groups == 0) return;
    free(groups->group_offsets);
    free(groups->group_bucket_ids);
    free(groups->bucket_group_ids);
    free(groups->group_bucket_counts);
    free(groups->group_total_cells);
    free(groups->group_coeff_segments);
    free(groups->group_n_on);
    free(groups->group_nstep_min);
    free(groups->group_nstep_max);
    tracer_gpu_bucket_groups_reset(groups);
}

int tracer_gpu_bucket_groups_build(const TracerBucket *bucket,
                                   int nbuckets,
                                   int allow_snapshot_interp,
                                   TracerGpuBucketGroups *groups)
{
    int ibucket;
    int ngroups = 0;
    int write_pos = 0;
    unsigned char *assigned = 0;

    if (bucket == 0 || groups == 0 || nbuckets < 0 ||
        groups->capacity_nbuckets < nbuckets ||
        groups->group_offsets == 0 || groups->group_bucket_ids == 0 ||
        groups->bucket_group_ids == 0 ||
        groups->group_bucket_counts == 0 || groups->group_total_cells == 0 ||
        groups->group_coeff_segments == 0 ||
        groups->group_n_on == 0 || groups->group_nstep_min == 0 ||
        groups->group_nstep_max == 0) {
        return -1;
    }

    groups->ngroups = 0;
    if (nbuckets == 0) {
        groups->group_offsets[0] = 0;
        return 0;
    }

    assigned = (unsigned char *)calloc((size_t)nbuckets, sizeof(unsigned char));
    if (assigned == 0) return -1;

    for (ibucket = 0; ibucket < nbuckets; ibucket++) {
        int jbucket;
        int ref_n_on;
        int ref_coeff_segments;
        int min_nstep;
        int max_nstep;
        int group_bucket_count = 0;
        int group_total_cells = 0;

        if (assigned[ibucket]) continue;

        ref_n_on = bucket->onsteps[ibucket];
        ref_coeff_segments =
            tracer_fp_choose_coeff_interp_segments(bucket->steps[ibucket],
                                                   allow_snapshot_interp);
        min_nstep = bucket->steps[ibucket];
        max_nstep = bucket->steps[ibucket];
        groups->group_offsets[ngroups] = write_pos;

        for (jbucket = ibucket; jbucket < nbuckets; jbucket++) {
            int cand_n_on;
            int cand_nstep;
            int cand_coeff_segments;
            int proposed_min;
            int proposed_max;

            if (assigned[jbucket]) continue;
            if (bucket->member_counts[jbucket] <= 0) continue;

            cand_n_on = bucket->onsteps[jbucket];
            cand_nstep = bucket->steps[jbucket];
            cand_coeff_segments =
                tracer_fp_choose_coeff_interp_segments(cand_nstep,
                                                       allow_snapshot_interp);
            if (cand_coeff_segments != ref_coeff_segments) continue;

            proposed_min = (cand_nstep < min_nstep) ? cand_nstep : min_nstep;
            proposed_max = (cand_nstep > max_nstep) ? cand_nstep : max_nstep;
            min_nstep = proposed_min;
            max_nstep = proposed_max;
            groups->group_bucket_ids[write_pos++] = jbucket;
            group_bucket_count++;
            group_total_cells += bucket->member_counts[jbucket];
            assigned[jbucket] = 1;
        }

        groups->group_bucket_counts[ngroups] = group_bucket_count;
        groups->group_total_cells[ngroups] = group_total_cells;
        groups->group_coeff_segments[ngroups] = ref_coeff_segments;
        groups->group_n_on[ngroups] = ref_n_on;
        groups->group_nstep_min[ngroups] = min_nstep;
        groups->group_nstep_max[ngroups] = max_nstep;
        for (jbucket = groups->group_offsets[ngroups];
             jbucket < write_pos;
             jbucket++) {
            groups->bucket_group_ids[groups->group_bucket_ids[jbucket]] = ngroups;
        }
        ngroups++;
    }

    groups->group_offsets[ngroups] = write_pos;
    groups->ngroups = ngroups;
    free(assigned);
    return 0;
}

int tracer_bucket_build_snapshot(const TracerStep *step,
                                 int ntracer,
                                 TracerBucket *bucket,
                                 int *nbuckets_out)
{
    int itr;
    int ibucket;
    int nbuckets = 0;

    if (step == 0 || bucket == 0 || nbuckets_out == 0 || ntracer <= 0 ||
        step->nsubsteps == 0 || step->n_onsteps == 0 ||
        bucket->steps == 0 || bucket->onsteps == 0 || bucket->indices == 0 ||
        bucket->bucket_ids == 0 || bucket->member_offsets == 0 ||
        bucket->member_counts == 0 || bucket->member_write == 0) {
        return -1;
    }

    memset(bucket->member_counts, 0, (size_t)ntracer * sizeof(int));

    for (itr = 0; itr < ntracer; itr++) {
        int bucket_id = -1;
        for (ibucket = 0; ibucket < nbuckets; ibucket++) {
            if (bucket->steps[ibucket] == step->nsubsteps[itr] &&
                bucket->onsteps[ibucket] == step->n_onsteps[itr]) {
                bucket_id = ibucket;
                break;
            }
        }
        if (bucket_id < 0) {
            bucket_id = nbuckets;
            bucket->steps[bucket_id] = step->nsubsteps[itr];
            bucket->onsteps[bucket_id] = step->n_onsteps[itr];
            nbuckets++;
        }
        bucket->bucket_ids[itr] = bucket_id;
        bucket->member_counts[bucket_id]++;
    }

    {
        int bucket_offset = 0;
        for (ibucket = 0; ibucket < nbuckets; ibucket++) {
            bucket->member_offsets[ibucket] = bucket_offset;
            bucket->member_write[ibucket] = bucket_offset;
            bucket_offset += bucket->member_counts[ibucket];
        }
    }

    for (itr = 0; itr < ntracer; itr++) {
        const int bucket_id = bucket->bucket_ids[itr];
        bucket->indices[bucket->member_write[bucket_id]++] = itr;
    }

    *nbuckets_out = nbuckets;
    return 0;
}
