#ifndef INCLUDED_tracer_fp_loadbalance_h_
#define INCLUDED_tracer_fp_loadbalance_h_

#include <stdio.h>

#include "tracer_fp.h"

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
                                             const double *beta);

void tracer_fp_lb_plan_free(TracerLoadBalancePlan *plan);

int tracer_fp_lb_plan_build_global(long int run_tracer_start,
                                                  int ntracer_global,
                                                  int mpi_size,
                                                  double top_frac,
                                                  const char *output_dir,
                                                  const long int *all_ids,
                                                  const long long *all_sums,
                                                  int total_count,
                                                  TracerLoadBalancePlan *plan);

int tracer_fp_lb_plan_build(long int run_tracer_start,
                                      long int tracer_start,
                                      int ntracer_global,
                                      int mpi_rank,
                                      int mpi_size,
                                      int ntracer_local,
                                      const long long *sum_nsub_local,
                                      double top_frac,
                                      const char *output_dir,
                                      TracerLoadBalancePlan *plan);
int tracer_fp_lb_active_capacity(long int run_tracer_start,
                                 long int tracer_start,
                                 int ntracer_local,
                                 const TracerLoadBalancePlan *plan,
                                 int mpi_rank,
                                 int *nactive_out);
int tracer_fp_lb_report_baseline(long int run_tracer_start,
                                 long int tracer_start,
                                 int ntracer_global,
                                 int mpi_rank,
                                 int mpi_size,
                                 int ntracer_local,
                                 const long long *sum_nsub_local,
                                 double top_frac,
                                 const char *output_dir);

int tracer_fp_lb_select(long int run_tracer_start,
                                       long int tracer_start,
                                       int ntracer_local,
                                       const TracerLoadBalancePlan *plan,
                                       int mpi_rank,
                                       long int *active_global_ids,
                                       int *active_source_offsets,
                                       int *nactive_out);

int tracer_fp_nsub_estimate(int ntracer,
                                long int tracer_start,
                                int ntracer_global,
                                int nsnap,
                                TracerFpInputMode input_mode,
                                TracerFpBackgroundMode background_mode,
                                const char *params_file,
                                long long *sum_nsub,
                                long long *snapshot_sum_nsub,
                                long long *snapshot_target_nsub);

int tracer_fp_load_est(long int run_tracer_start,
                                int ntracer,
                                int nsnap,
                                TracerFpInputMode input_mode,
                                TracerFpBackgroundMode background_mode,
                                const char *params_file,
                                const char *output_dir,
                                int virtual_ranks,
                                double top_frac);

#endif
