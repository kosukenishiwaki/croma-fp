#ifndef TRACER_FP_CR_SOURCE_H
#define TRACER_FP_CR_SOURCE_H

#include "params.h"

double CRe_Norm(double delta, CRspectrum *cre, CRspectrum *crp, double tracer_mass);
double CRp_Norm(double delta, CRspectrum *cre, CRspectrum *crp, double tracer_mass);
double CRe_Norm_AGN(double delta,
                    CRspectrum *cre,
                    CRspectrum *crp,
                    double tracer_mass,
                    double z,
                    double dt_gyr);
double CRp_Norm_AGN(double delta,
                    CRspectrum *cre,
                    CRspectrum *crp,
                    double tracer_mass,
                    double z,
                    double dt_gyr);
double CRe_Norm_Inj_AGN(double temp_e_cre,
                        double tracer_mass,
                        double z,
                        double dt_gyr);
void spectrum_temprate(double delta,
                       CRspectrum *crp,
                       CRspectrum *cre,
                       double *np_template,
                       double *ne_template,
                       double *num_p_template,
                       double *e_p_template,
                       double *num_e_template,
                       double *e_e_template);

#endif
