#ifndef INCLUDED_tracer_fp_nsub_h_
#define INCLUDED_tracer_fp_nsub_h_

#include "tracer_fp.h"

#ifdef __cplusplus
extern "C" {
#endif

int tracer_fp_compute_nsubsteps(int ntracer,
                                double dt_snap,
                                const CRspectrum *crp_grid,
                                const CRspectrum *cre_grid,
                                const double *crp_radp_batch,
                                const double *cre_radp_batch,
                                const double *l_turb_mpc,
                                const double *dv_imc,
                                int *nsubsteps,
                                int *target_nsubsteps,
                                int *n_onsteps,
                                unsigned char *capped_flags,
                                int *max_nsubsteps,
                                int *capped_count_out,
                                int *max_raw_nsub_out);

int tracer_fp_compute_nsubsteps_from_endpoints(int ntracer,
                                               double dt_snap,
                                               const CRspectrum *crp_grid,
                                               const CRspectrum *cre_grid,
                                               const double *crp_radp_lo,
                                               const double *crp_radp_hi,
                                               const double *cre_radp_lo,
                                               const double *cre_radp_hi,
                                               const double *l_turb_mpc,
                                               const double *dv_imc,
                                               int *nsubsteps,
                                               int *target_nsubsteps,
                                               int *n_onsteps,
                                               unsigned char *capped_flags,
                                               int *max_nsubsteps,
                                               int *capped_count_out,
                                               int *max_raw_nsub_out);

int tracer_fp_prepare_nsub_loss_endpoints(int ntracer,
                                          const CRspectrum *crp_grid,
                                          const CRspectrum *cre_grid,
                                          const double *n_gas,
                                          const double *kbt,
                                          const double *b_field,
                                          const double *divv_gyr,
                                          const double *rad_ic_batch,
                                          double *crp_radp_lo,
                                          double *crp_radp_hi,
                                          double *cre_radp_lo,
                                          double *cre_radp_hi);

#ifdef __cplusplus
}
#endif

#endif
