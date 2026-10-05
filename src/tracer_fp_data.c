/*
    tracer_fp_data.c

    K. Nishiwaki, 2026-06-18
    - TracerData allocation/free utils
    - assuming background MHD data from hdf5

*/


#include <stdlib.h>
#include <string.h>

#include "tracer_fp_data.h"

void tracer_data_history_reset(TracerDataHistory *history)
{
    if (history == 0) return;
    memset(history, 0, sizeof(*history));
}

int tracer_data_history_alloc(TracerDataHistory *history, size_t count)
{
    if (history == 0 || count == 0) return -1;

    history->n_gas = (double *)calloc(count, sizeof(double));
    history->kbt = (double *)calloc(count, sizeof(double));
    history->b_field = (double *)calloc(count, sizeof(double));
    history->divv = (double *)calloc(count, sizeof(double));
    history->l_turb = (double *)calloc(count, sizeof(double));
    history->dv_imc = (double *)calloc(count, sizeof(double));
    history->cs = (double *)calloc(count, sizeof(double));
    history->beta_pl = (double *)calloc(count, sizeof(double));

    if (history->n_gas == 0 || history->kbt == 0 || history->b_field == 0 ||
        history->divv == 0 || history->l_turb == 0 || history->dv_imc == 0 ||
        history->cs == 0 || history->beta_pl == 0) {
        tracer_data_history_release(history);
        return -1;
    }
    return 0;
}

void tracer_data_history_release(TracerDataHistory *history)
{
    if (history == 0) return;
    free(history->n_gas);
    free(history->kbt);
    free(history->b_field);
    free(history->divv);
    free(history->l_turb);
    free(history->dv_imc);
    free(history->cs);
    free(history->beta_pl);
    tracer_data_history_reset(history);
}

void tracer_data_storage_reset(TracerDataStorage *storage)
{
    if (storage == 0) return;
    memset(storage, 0, sizeof(*storage));
}

int tracer_data_storage_alloc(TracerDataStorage *storage, size_t count)
{
    if (storage == 0 || count == 0) return -1;

    storage->temp = (double *)calloc(count, sizeof(double));
    storage->rho = (double *)calloc(count, sizeof(double));
    storage->bx = (double *)calloc(count, sizeof(double));
    storage->by = (double *)calloc(count, sizeof(double));
    storage->bz = (double *)calloc(count, sizeof(double));
    storage->divv = (double *)calloc(count, sizeof(double));
    storage->rotv = (double *)calloc(count, sizeof(double));
    storage->lturb = (double *)calloc(count, sizeof(double));
    storage->mach = (double *)calloc(count, sizeof(double));
    storage->prestemp = (double *)calloc(count, sizeof(double));
    storage->presden = (double *)calloc(count, sizeof(double));
    storage->pre_density_cgs = (double *)calloc(count, sizeof(double));
    storage->upstream_speed_cgs = (double *)calloc(count, sizeof(double));
    storage->shock_side_code = (double *)calloc(count, sizeof(double));
    storage->dsa_trigger = (double *)calloc(count, sizeof(double));
    storage->dsa_mach = (double *)calloc(count, sizeof(double));
    storage->dsa_pre_density = (double *)calloc(count, sizeof(double));
    storage->dsa_kinetic_energy_flux_cgs = (double *)calloc(count, sizeof(double));

    if (storage->temp == 0 || storage->rho == 0 || storage->bx == 0 ||
        storage->by == 0 || storage->bz == 0 || storage->divv == 0 ||
        storage->rotv == 0 || storage->lturb == 0 || storage->mach == 0 ||
        storage->prestemp == 0 || storage->presden == 0 ||
        storage->pre_density_cgs == 0 || storage->upstream_speed_cgs == 0 ||
        storage->shock_side_code == 0 || storage->dsa_trigger == 0 ||
        storage->dsa_mach == 0 || storage->dsa_pre_density == 0 ||
        storage->dsa_kinetic_energy_flux_cgs == 0) {
        tracer_data_storage_release(storage);
        return -1;
    }
    return 0;
}

void tracer_data_storage_release(TracerDataStorage *storage)
{
    if (storage == 0) return;
    free(storage->temp);
    free(storage->rho);
    free(storage->bx);
    free(storage->by);
    free(storage->bz);
    free(storage->divv);
    free(storage->rotv);
    free(storage->lturb);
    free(storage->mach);
    free(storage->prestemp);
    free(storage->presden);
    free(storage->pre_density_cgs);
    free(storage->upstream_speed_cgs);
    free(storage->shock_side_code);
    free(storage->dsa_trigger);
    free(storage->dsa_mach);
    free(storage->dsa_pre_density);
    free(storage->dsa_kinetic_energy_flux_cgs);
    tracer_data_storage_reset(storage);
}
