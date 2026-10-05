#ifndef INCLUDED_tracer_fp_multirate_h_
#define INCLUDED_tracer_fp_multirate_h_

int tracer_fp_select_multirate_base_nstep(int target_nsub);
int tracer_fp_apply_multirate_schedule(int ntracer,
                                       int *nsubsteps,
                                       int *n_onsteps,
                                       int max_target_nsub,
                                       int *base_nsteps,
                                       int *fp_cadence_steps,
                                       int *coarse_base_nstep_out,
                                       int *fine_base_nstep_out);

#endif
