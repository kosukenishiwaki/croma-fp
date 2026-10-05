#ifndef INCLUDED_tracer_fp_selection_h_
#define INCLUDED_tracer_fp_selection_h_

int tracer_fp_run_part(long int total_tracers,
                                    int num_run,
                                    int run_id,
                                    long int *run_start_out,
                                    long int *run_count_out);

int tracer_fp_heavy_mask_load(const char *filename,
                                    long int tracer_start,
                                    int ntracer_local,
                                    unsigned char *is_heavy_local,
                                    int *nheavy_local_out);

void tracer_fp_local_select(long int tracer_start,
                                     int ntracer_local,
                                     const unsigned char *is_heavy_local,
                                     int skip_heavy,
                                     long int *active_global_ids,
                                     int *active_source_offsets,
                                     int *nactive_out);

#endif
