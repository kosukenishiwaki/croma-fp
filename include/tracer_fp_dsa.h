#ifndef INCLUDED_tracer_fp_dsa_h_
#define INCLUDED_tracer_fp_dsa_h_

#include <stdio.h>

#include "tracer_fp.h"

int tracer_fp_apply_dsa_tracer_injection_snapshot(const TracerFpRawBackgroundSlot *raw,
                                                  int ntracer,
                                                  const long int *tracer_ids,
                                                  const double *tracer_mass,
                                                  double density_unit_cgs,
                                                  double dt_gyr,
                                                  TracerDsaInjectionMode mode,
                                                  TracerDsaReaccMode reacc_mode,
                                                  const CRspectrum *crp_grid,
                                                  const CRspectrum *cre_grid,
                                                  DSAGrid *dsa_grid,
                                                  double *crp_state,
                                                  double *cre_state,
                                                  double *qpi_batch,
                                                  double *qepri_batch,
                                                  unsigned char *disable_adiabatic,
                                                  int snapshot_1based,
                                                  int mpi_rank,
                                                  FILE *reacc_debug_fp,
                                                  int *ninjected_out);

#endif
