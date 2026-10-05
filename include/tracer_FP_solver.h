#include "params.h"
#include "CONSTANTS.h"
#include "FP_Coef.h"
#include "HADRONIC.h"
#include "Chang_Cooper.h"
#include "COSFUNC.h"
#include "READFILE.h"


typedef struct {
    int nsnp_i;
    int nsnp_f;
    double *z_snp;
    double *z_snp_nxt;
    double *dt;
    double *density;
    double *temperature;
    double *B;
    double *divv;
    double *rotv;
    double *volume;
    uint8_t *mask_flag;
}tracer_params;


typedef struct {
    double **rad_IC;
    double *rad_IC_m1;
    double *rad_IC_p1;
}radIC_cool;


//double tracer_FP(int tracer_ID,tracer_params *param, radIC_cool *IC_cool, double *curl_v_unmask_median, CRspectrum *CRp, CRspectrum *CRe, double **Ne_buffer, double **Np_buffer);
double tracer_FP(int tracer_ID,tracer_params *param, double *curl_v_unmask_median, CRspectrum *CRp, CRspectrum *CRe, double **Ne_buffer, double **Np_buffer);
void set_IC_cooling(tracer_params *param, radIC_cool *IC_cool, CRspectrum *CRe);
   