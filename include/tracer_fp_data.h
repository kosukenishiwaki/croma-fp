#ifndef INCLUDED_tracer_fp_data_h_
#define INCLUDED_tracer_fp_data_h_

#include <stddef.h>

#include "tracer_fp.h"

void tracer_data_history_reset(TracerDataHistory *history);
int tracer_data_history_alloc(TracerDataHistory *history, size_t count);
void tracer_data_history_release(TracerDataHistory *history);

void tracer_data_storage_reset(TracerDataStorage *storage);
int tracer_data_storage_alloc(TracerDataStorage *storage, size_t count);
void tracer_data_storage_release(TracerDataStorage *storage);

#endif
