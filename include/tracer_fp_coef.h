#ifndef INCLUDED_tracer_fp_coef_h_
#define INCLUDED_tracer_fp_coef_h_

#include "tracer_fp.h"

#ifdef __cplusplus
extern "C" {
#endif

void tracer_fp_cpu_ws_free(TracerFpCpuWs *ws);
int tracer_fp_cpu_ws_ensure(TracerFpCpuWs *ws,
                            int ncell);
int tracer_fp_snapshot_ws_ensure(TracerFpSnapshotWs *ws,
                                 int ncell);
void tracer_fp_snapshot_ws_free(TracerFpSnapshotWs *ws);
int tracer_fp_gpu_host_ws_ensure(TracerFpGpuHostWs *ws,
                                 int ncell);
void tracer_fp_gpu_host_ws_free(TracerFpGpuHostWs *ws);

int tracer_fp_prepare_coeff_batches(int ncell,
                                    double dt,
                                    const CRspectrum *crp_grid,
                                    const CRspectrum *cre_grid,
                                    const double *n_gas,
                                    const double *kbt,
                                    const double *b_field,
                                    const double *divv_gyr,
                                    const double *l_turb_mpc,
                                    const double *dv_imc,
                                    const double *cs,
                                    const double *beta_pl,
                                    const double *rad_ic_batch,
                                    const double *rad_ic_m1,
                                    const double *rad_ic_p1,
                                    const double *qpi_batch,
                                    const double *qepri_batch,
                                    const double *fqe_flat,
                                    const int *np_min_qe,
                                    const double *tracer_mass,
                                    const unsigned char *disable_adiabatic,
                                    const double *crp_state,
                                    const double *cre_state,
                                    TracerFpGpuTimes *times,
                                    TracerFpCpuWs *ws);

#ifdef __cplusplus
}
#endif

#endif
