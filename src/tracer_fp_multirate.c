/*
    tracer_fp_multirate.c

    !! DEPRECATED !! 
    this mode is slower than the quantized bucketing one for some reason. 
    but probably useful for mesh-based FP (incl diffusion). we will revisit here someday.

    K. Nishiwaki, 2026-06-18
    - multirate schedular for FP timestep
*/

#include <math.h>

#include "tracer_fp_multirate.h"

static int floor_power_of_two(int value)
{
    int power = 1;
    if (value <= 1) return 1;
    while (power <= value / 2) power *= 2;
    return power;
}

static int next_power_of_two(int value)
{
    int power = 1;
    if (value <= 1) return 1;
    while (power < value && power <= 1073741823) power *= 2;
    return power;
}

int tracer_fp_select_multirate_base_nstep(int target_nsub)
{
    static const int kMultirateBaseLadder[] = {64, 128, 256, 512, 1024, 2048};
    const int nlevel = (int)(sizeof(kMultirateBaseLadder) / sizeof(kMultirateBaseLadder[0]));
    int ilevel;

    if (target_nsub <= 1) return 64;
    for (ilevel = 0; ilevel < nlevel; ilevel++) {
        if (target_nsub <= kMultirateBaseLadder[ilevel]) {
            return kMultirateBaseLadder[ilevel];
        }
    }
    return next_power_of_two(target_nsub);
}

int tracer_fp_apply_multirate_schedule(int ntracer,
                                       int *nsubsteps,
                                       int *n_onsteps,
                                       int max_target_nsub,
                                       int *base_nsteps,
                                       int *fp_cadence_steps,
                                       int *coarse_base_nstep_out,
                                       int *fine_base_nstep_out)
{
    int itr;
    int min_base_nstep;
    int max_base_nstep;

    if (ntracer <= 0 || nsubsteps == 0 || n_onsteps == 0 ||
        base_nsteps == 0 || fp_cadence_steps == 0) return -1;
    if (max_target_nsub < 1) return -1;

    max_base_nstep = tracer_fp_select_multirate_base_nstep(max_target_nsub);
    if (max_base_nstep < 1) max_base_nstep = 1;
    min_base_nstep = max_base_nstep;
    if (min_base_nstep > 64) min_base_nstep = 64;
    if (min_base_nstep < 1) min_base_nstep = 1;

    for (itr = 0; itr < ntracer; itr++) {
        const int target_nsub = (nsubsteps[itr] > 0) ? nsubsteps[itr] : 1;
        const int target_n_on = (n_onsteps[itr] > 0) ? n_onsteps[itr] : target_nsub;
        const double on_fraction = (double)target_n_on / (double)target_nsub;
        int base_nstep = tracer_fp_select_multirate_base_nstep(target_nsub);
        const int ratio_floor = (int)floor((double)base_nstep / (double)target_nsub);
        int cadence;
        int effective_nsub;
        int effective_n_on;

        if (base_nstep < min_base_nstep) base_nstep = min_base_nstep;
        if (base_nstep > max_base_nstep) base_nstep = max_base_nstep;
        cadence = floor_power_of_two(ratio_floor);
        effective_nsub = base_nstep / cadence;
        effective_n_on = (int)floor((double)effective_nsub * on_fraction + 0.5);

        if (effective_n_on < 1) effective_n_on = 1;
        if (effective_n_on > effective_nsub) effective_n_on = effective_nsub;

        base_nsteps[itr] = base_nstep;
        fp_cadence_steps[itr] = cadence;
        nsubsteps[itr] = effective_nsub;
        n_onsteps[itr] = effective_n_on;
    }

    if (coarse_base_nstep_out != 0) *coarse_base_nstep_out = min_base_nstep;
    if (fine_base_nstep_out != 0) *fine_base_nstep_out = max_base_nstep;
    return 0;
}
