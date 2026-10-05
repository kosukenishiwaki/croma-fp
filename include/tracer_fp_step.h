#ifndef INCLUDED_tracer_fp_step_h_
#define INCLUDED_tracer_fp_step_h_

#include "tracer_fp.h"

void tracer_step_reset(TracerStep *step);
int tracer_step_alloc(TracerStep *step, int ntracer);
void tracer_step_release(TracerStep *step);

#endif
