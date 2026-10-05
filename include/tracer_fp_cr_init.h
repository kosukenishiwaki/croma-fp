#ifndef INCLUDED_tracer_fp_cr_init_h_
#define INCLUDED_tracer_fp_cr_init_h_

#include "tracer_fp.h"

#ifdef __cplusplus
extern "C" {
#endif

void tracer_fp_init_state(double *crp_state,
                          double *cre_state,
                          double *qpi_batch,
                          double *qepri_batch,
                          const CRspectrum *crp_grid,
                          const CRspectrum *cre_grid,
                          const double *tracer_mass,
                          double z_init,
                          int ntracer,
                          long int tracer_start,
                          int ntracer_global,
                          long int debug_target_global);

#ifdef __cplusplus
}
#endif

#endif
