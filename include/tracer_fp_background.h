#ifndef INCLUDED_tracer_fp_background_h_
#define INCLUDED_tracer_fp_background_h_

#include "tracer_fp.h"

int tracer_fp_frozen_runtime(const char *params_file,
                                         int *runtime_nsnap,
                                         double *runtime_dt_gyr);
int tracer_fp_runtime_steps(TracerFpInputMode input_mode,
                                        TracerFpBackgroundMode background_mode,
                                        const char *params_file,
                                        int *runtime_nsnap,
                                        int *frozen_runtime_override,
                                        double *frozen_runtime_dt_gyr);
void tracer_build_snapshot_filename(int snap_index,
                                    char *filename,
                                    size_t filename_size);
int tracer_fp_count_selected(const int *selected_offsets, int nselected);
void tracer_fp_selected_offset_stats(const int *selected_offsets,
                                     int nselected,
                                     int *selected_count_out,
                                     int *run_count_out,
                                     int *min_run_len_out,
                                     int *max_run_len_out,
                                     int *max_gap_out);
int tracer_fp_hdf5_init(const char *params_file,
                              int ntracer,
                              int ntracer_global,
                              int nsnap,
                              TracerFpBackgroundMode background_mode,
                              TracerDataStorage *raw_storage,
                              TracerFpRawBackgroundSlot *raw_slots,
                              TracerFpHdf5Meta *meta,
                              double *dt_snap,
                              double *z_snap);
int tracer_fp_hdf5_load_start(const TracerFpHdf5Meta *meta,
                                          TracerFpBackgroundMode background_mode,
                                          long int tracer_start,
                                          const int *selected_offsets,
                                          int ntracer,
                                          int nsnap,
                                          TracerFpRawBackgroundSlot raw_slots[3],
                                          int raw_prev_slot,
                                          int raw_curr_slot,
                                          int raw_next_slot,
                                          int *read_calls_out);

void tracer_zero_background_history_slot(size_t off,
                                         TracerDataHistory *history);
void tracer_store_background_history_slot(size_t off,
                                          const FpBackgroundCellOutput *bg_out,
                                          TracerDataHistory *history);
void tracer_bind_background_slot(int ntracer,
                                 int slot,
                                 const TracerDataHistory *history,
                                 TracerFpBackgroundSlot *view);
void tracer_bind_raw_background_slot(int ntracer,
                                     int slot,
                                     const TracerDataStorage *storage,
                                     TracerFpRawBackgroundSlot *view);
void tracer_copy_background_slot(int ntracer,
                                 const TracerFpBackgroundSlot *dst,
                                 const TracerFpBackgroundSlot *src);

void tracer_free_hdf5_meta(TracerFpHdf5Meta *meta);
int tracer_init_hdf5_meta(TracerFpHdf5Meta *meta,
                          const char *params_file,
                          int ntracer_global);
int tracer_fill_hdf5_timeline(const TracerFpHdf5Meta *meta,
                              double *dt_snap,
                              double *z_snap,
                              int nsnap,
                              TracerFpBackgroundMode background_mode,
                              const char *params_file);
int tracer_load_hdf5_raw_snapshot_slice(const TracerFpHdf5Meta *meta,
                                        int snap_id,
                                        long int tracer_start,
                                        const int *selected_offsets,
                                        int ntracer,
                                        TracerFpRawBackgroundSlot *raw);
int tracer_fp_write_hdf5_layout_log(const char *output_dir,
                                    int snap_id);
int tracer_prepare_background_from_raw(int ntracer,
                                       const TracerFpRawBackgroundSlot *prev_raw,
                                       const TracerFpRawBackgroundSlot *curr_raw,
                                       const TracerFpRawBackgroundSlot *next_raw,
                                       double z_prev,
                                       double z_curr,
                                       double z_next,
                                       int apply_temp_floor,
                                       int allow_curl_interp,
                                       TracerFpBackgroundSlot *out);

int load_tracer_initial_mass_hdf5(double *tracer_mass,
                                  int ntracer,
                                  long int tracer_start,
                                  const int *selected_offsets,
                                  int ntracer_global,
                                  const char *params_file,
                                  double z_init);

#endif
