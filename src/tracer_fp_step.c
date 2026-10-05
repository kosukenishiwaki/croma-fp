#include <stdlib.h>
#include <string.h>

#include "tracer_fp_step.h"

void tracer_step_reset(TracerStep *step)
{
    if (step == 0) return;
    memset(step, 0, sizeof(*step));
}

int tracer_step_alloc(TracerStep *step, int ntracer)
{
    if (step == 0 || ntracer <= 0) return -1;

    step->nsubsteps = (int *)calloc((size_t)ntracer, sizeof(int));
    step->target_nsubsteps = (int *)calloc((size_t)ntracer, sizeof(int));
    step->n_onsteps = (int *)calloc((size_t)ntracer, sizeof(int));
    step->base_nsteps = (int *)calloc((size_t)ntracer, sizeof(int));
    step->fp_cadence_steps = (int *)calloc((size_t)ntracer, sizeof(int));
    step->capped_flags = (unsigned char *)calloc((size_t)ntracer, sizeof(unsigned char));

    if (step->nsubsteps == 0 || step->target_nsubsteps == 0 || step->n_onsteps == 0 ||
        step->base_nsteps == 0 || step->fp_cadence_steps == 0 || step->capped_flags == 0) {
        return -1;
    }
    return 0;
}

void tracer_step_release(TracerStep *step)
{
    if (step == 0) return;
    free(step->capped_flags);
    free(step->nsubsteps);
    free(step->target_nsubsteps);
    free(step->n_onsteps);
    free(step->base_nsteps);
    free(step->fp_cadence_steps);
    tracer_step_reset(step);
}
