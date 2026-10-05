#ifndef INCLUDED_tracer_fp_bucket_h_
#define INCLUDED_tracer_fp_bucket_h_

#include "tracer_fp.h"

void tracer_bucket_reset(TracerBucket *bucket);
int tracer_bucket_alloc(TracerBucket *bucket, int ntracer);
void tracer_bucket_release(TracerBucket *bucket);
int tracer_bucket_build_snapshot(const TracerStep *step,
                                 int ntracer,
                                 TracerBucket *bucket,
                                 int *nbuckets_out);
void tracer_gpu_bucket_groups_reset(TracerGpuBucketGroups *groups);
int tracer_gpu_bucket_groups_alloc(TracerGpuBucketGroups *groups, int nbuckets_max);
void tracer_gpu_bucket_groups_release(TracerGpuBucketGroups *groups);
int tracer_gpu_bucket_groups_build(const TracerBucket *bucket,
                                   int nbuckets,
                                   int allow_snapshot_interp,
                                   TracerGpuBucketGroups *groups);

#endif
