#ifndef INCLUDED_tracer_fp_synch_h_
#define INCLUDED_tracer_fp_synch_h_

#include "tracer_fp.h"

enum { TRACER_FP_SYNCH_LOGB_BINS = 512 };

#ifdef __cplusplus
extern "C" {
#endif

void tracer_fp_logb_range_from_snapshot(int ntracer,
                                        const double *b_dyn,
                                        double *logb,
                                        double *logb_min_out,
                                        double *logb_max_out,
                                        int *nout_of_range_out);

void tracer_fp_synch_data_reset(SynchData *synch);
int tracer_fp_synch_data_alloc(SynchData *synch, int nfreq);
int tracer_fp_synch_data_init_tables(SynchData *synch,
                                     int nfreq,
                                     const CRspectrum *cre_grid);
void tracer_fp_synch_data_release(SynchData *synch);

int tracer_fp_refresh_synch_kernel_table(SynchData *synch,
                                         int nfreq,
                                         double snapshot_logb_min,
                                         double snapshot_logb_max,
                                         double **kernel_table_io);
int tracer_fp_prepare_ic_cooling_snapshot(int ntracer,
                                          int allow_snapshot_interp,
                                          double z_curr,
                                          double z_next,
                                          const CRspectrum *cre_grid,
                                          double *rad_ic_zero,
                                          double *rad_ic_m1_zero,
                                          double *rad_ic_p1_zero,
                                          double *rad_ic_row_next,
                                          double *rad_ic_m1_next_val,
                                          double *rad_ic_p1_next_val);

#ifdef __cplusplus
}
#endif

#endif
