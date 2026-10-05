/*
    tracer_fp_selection.c

    K. Nishiwaki, 2026-06-18
    - select tracers for this run/job
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "read_grid_hdf5.h"
#include "tracer_fp.h"
#include "tracer_fp_selection.h"

int tracer_fp_run_part(long int total_tracers,
                                    int num_run,
                                    int run_id,
                                    long int *run_start_out,
                                    long int *run_count_out)
{
    long int run_start = 0;
    long int run_count = 0;

    if (run_start_out == 0 || run_count_out == 0) return -1;
    *run_start_out = 0;
    *run_count_out = 0;

    if (num_run <= 1) {
        if (run_id != 0) return -1;
        *run_start_out = 0;
        *run_count_out = total_tracers;
        return 0;
    }
    if (total_tracers <= 0 || run_id < 0 || run_id >= num_run) return -1;

    grid_cell_range(total_tracers, run_id, num_run, &run_start, &run_count);
    *run_start_out = run_start;
    *run_count_out = run_count;
    return 0;
}

int tracer_fp_heavy_mask_load(const char *filename,
                                    long int tracer_start,
                                    int ntracer_local,
                                    unsigned char *is_heavy_local,
                                    int *nheavy_local_out)
{
    FILE *fp = 0;
    char line[MAX_LINE_LENGTH];
    int nheavy_local = 0;

    if (nheavy_local_out != 0) *nheavy_local_out = 0;
    if (filename == 0 || *filename == '\0' || ntracer_local < 0 || is_heavy_local == 0) return -1;
    if (ntracer_local == 0) return 0;

    memset(is_heavy_local, 0, (size_t)ntracer_local * sizeof(unsigned char));
    fp = fopen(filename, "r");
    if (fp == 0) return -1;

    while (fgets(line, sizeof(line), fp) != 0) {
        char *cursor = line;
        char *endptr = 0;
        long int tracer_id;

        while (*cursor == ' ' || *cursor == '\t') cursor++;
        if (*cursor == '\0' || *cursor == '\n' || *cursor == '#') continue;

        tracer_id = strtol(cursor, &endptr, 10);
        if (endptr == cursor) continue;
        if (tracer_id >= tracer_start &&
            tracer_id < tracer_start + (long int)ntracer_local) {
            const int local_idx = (int)(tracer_id - tracer_start);
            if (!is_heavy_local[local_idx]) {
                is_heavy_local[local_idx] = 1;
                nheavy_local++;
            }
        }
    }

    fclose(fp);
    if (nheavy_local_out != 0) *nheavy_local_out = nheavy_local;
    return 0;
}

void tracer_fp_local_select(long int tracer_start,
                                     int ntracer_local,
                                     const unsigned char *is_heavy_local,
                                     int skip_heavy,
                                     long int *active_global_ids,
                                     int *active_source_offsets,
                                     int *nactive_out)
{
    int itr;
    int nactive = 0;

    if (nactive_out != 0) *nactive_out = 0;
    if (active_global_ids == 0 || active_source_offsets == 0) return;

    for (itr = 0; itr < ntracer_local; itr++) {
        if (skip_heavy && is_heavy_local != 0 && is_heavy_local[itr]) continue;
        active_global_ids[nactive] = tracer_start + (long int)itr;
        active_source_offsets[nactive] = itr;
        nactive++;
    }

    if (nactive_out != 0) *nactive_out = nactive;
}
