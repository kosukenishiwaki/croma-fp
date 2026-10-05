#ifndef INCLUDED_tracer_fp_dsa_reacc_h_
#define INCLUDED_tracer_fp_dsa_reacc_h_

#include "DSA_MODELS.h"

double tracer_dsa_reacc_positive_delta_1d(const CRspectrum *grid,
                                          const double *state_tracer,
                                          int nbin,
                                          double volume_downstream,
                                          double q,
                                          double pmin_mc,
                                          double *delta_density,
                                          DSASpecies species);

double tracer_dsa_reacc_convolution_delta_1d(const CRspectrum *grid,
                                             const double *state_tracer,
                                             int nbin,
                                             double volume_upstream,
                                             double volume_downstream,
                                             double q,
                                             double pmin_mc,
                                             double *delta_density,
                                             DSASpecies species);

double tracer_dsa_reacc_delta_energy_density_1d(const CRspectrum *grid,
                                                const double *delta_density,
                                                int nbin,
                                                double pmin_mc,
                                                DSASpecies species);

double tracer_dsa_reacc_adiabatic_delta_1d(const CRspectrum *grid,
                                           const double *state_tracer,
                                           int nbin,
                                           double volume_upstream,
                                           double volume_downstream,
                                           double compression_ratio,
                                           double pmin_mc,
                                           double *delta_density,
                                           DSASpecies species);

#endif
