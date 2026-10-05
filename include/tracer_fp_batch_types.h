#ifndef INCLUDED_tracer_fp_batch_types_h_
#define INCLUDED_tracer_fp_batch_types_h_

typedef struct {
    int *nsubsteps;
    int *target_nsubsteps;
    int *n_onsteps;
    int *base_nsteps;
    int *fp_cadence_steps;
    unsigned char *capped_flags;
} TracerStep;

typedef struct {
    int *steps;
    int *onsteps;
    int *indices;
    int *bucket_ids;
    int *member_offsets;
    int *member_counts;
    int *member_write;
} TracerBucket;

typedef struct {
    int capacity_nbuckets;
    int ngroups;
    int *group_offsets;
    int *group_bucket_ids;
    int *bucket_group_ids;
    int *group_bucket_counts;
    int *group_total_cells;
    int *group_coeff_segments;
    int *group_n_on;
    int *group_nstep_min;
    int *group_nstep_max;
} TracerGpuBucketGroups;

typedef struct {
    double *n_gas, *kbt, *b_field, *divv;
    double *lturb, *dv, *cs, *beta;
    double *rad_ic, *rad_ic_m1, *rad_ic_p1;
} TracerFpBackgroundBuffers;

typedef struct {
    double *qpi, *qepri;
    double *crp, *cre;
    double *mass_msun;
    unsigned char *disable_adiabatic;
} TracerFpStateBuffers;

typedef struct {
    double *crp_radp, *crp_tloss, *crp_invtloss;
    double *cre_radp, *cre_tloss, *cre_invtloss;
    double *dpp, *dppe, *dpp_off, *dppe_off;
    double *crp_radpm1, *crp_radpp1, *cre_radpm1, *cre_radpp1;
    double *dppm1, *dppp1, *dppem1, *dppep1;
    double *dppm1_off, *dppp1_off, *dppem1_off, *dppep1_off;
} TracerFpCoeffBuffers;

typedef struct {
    double *ccp_a, *ccp_b, *ccp_c;
    double *cce_a, *cce_b, *cce_c;
    double *ccp_a_off, *ccp_b_off, *ccp_c_off;
    double *cce_a_off, *cce_b_off, *cce_c_off;
} TracerFpCcBuffers;

typedef struct {
    double *qe_integral, *inje;
} TracerFpSourceBuffers;

typedef struct {
    double *crp_radp, *crp_tloss, *crp_invtloss;
    double *cre_radp, *cre_tloss, *cre_invtloss;
    double *crp_radpm1, *crp_radpp1, *cre_radpm1, *cre_radpp1;
} TracerFpSnapshotLossBuffers;

typedef struct {
    int capacity_ncell;
    TracerFpSnapshotLossBuffers loss;
} TracerFpSnapshotWs;

typedef struct {
    double *b_dyn, *logb;
} TracerFpGpuBucketBuffers;

typedef struct {
    int capacity_ncell;
    TracerFpGpuBucketBuffers bucket;
} TracerFpGpuHostWs;

typedef struct {
    int capacity_ncell;
    TracerFpBackgroundBuffers bg_curr;
    TracerFpBackgroundBuffers bg_next;
    TracerFpBackgroundBuffers bg_interp;
    TracerFpStateBuffers state;
    TracerFpSnapshotWs snapshot;
    TracerFpCoeffBuffers coeff;
    TracerFpCcBuffers cc;
    TracerFpSourceBuffers source;
} TracerFpCpuWs;

#endif
