#ifndef INCLUDED_tracer_fp_debug_h_
#define INCLUDED_tracer_fp_debug_h_

#include "tracer_fp.h"

void tracer_fp_fill_timeline_synthetic(double *dt_snap,
                                       double *z_snap,
                                       int nsnap);
int tracer_fp_fill_background_snapshot_synthetic(int isnap,
                                                 const double *z_snap,
                                                 int ntracer,
                                                 const long int *tracer_ids,
                                                 int ntracer_global,
                                                 int nsnap,
                                                 TracerFpBackgroundSlot *slot);

long tracer_fp_debug_target_global_id(void);
int tracer_fp_debug_dump_full_bucket(void);
void tracer_fp_debug_print_cre_sample(const char *tag,
                                      long int global_tracer,
                                      int snapshot_1based,
                                      const double *cre_row);
int tracer_fp_debug_report_row_change(const char *tag,
                                      long int global_tracer,
                                      int snapshot_1based,
                                      int bucket_1based,
                                      const double *before_row,
                                      const double *after_row);

#endif
