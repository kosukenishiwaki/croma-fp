/*
    tracer_fp_restart.c

    Restart/checkpoint helpers for tracer_fp.
*/

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "FP_Coef.h"
#include "READFILE.h"
#include "tracer_fp_background.h"
#include "tracer_fp_debug.h"
#include "tracer_fp_output.h"
#include "tracer_fp_restart.h"
#include "read_grid_hdf5.h"

#ifdef FP_USE_CUDA_BACKEND
#define TRACER_FP_PROGNAME "tracer_fp_cuda.out"
#else
#define TRACER_FP_PROGNAME "tracer_fp_cpu.out"
#endif

int tracer_fp_restart_bg(const TracerFpRestartWin *window)
{
    if (window == 0) return -1;
    if (window->start_snapshot <= 0 || window->start_snapshot >= window->nsnap ||
        window->background_mode != TRACER_FP_BACKGROUND_EVOLVING) {
        return 0;
    }
    if (window->input_mode == TRACER_FP_INPUT_SYNTHETIC) {
        if (tracer_fp_fill_background_snapshot_synthetic(window->start_snapshot, window->z_snap,
                                                         window->ntracer, window->tracer_ids,
                                                         window->ntracer_global, window->nsnap,
                                                         &window->bg_slots[window->bg_curr_slot]) != 0) {
            return -1;
        }
        if (window->start_snapshot + 1 < window->nsnap) {
            if (tracer_fp_fill_background_snapshot_synthetic(window->start_snapshot + 1,
                                                             window->z_snap, window->ntracer,
                                                             window->tracer_ids, window->ntracer_global,
                                                             window->nsnap,
                                                             &window->bg_slots[window->bg_next_slot]) != 0) {
                return -1;
            }
        } else {
            tracer_copy_background_slot(window->ntracer, &window->bg_slots[window->bg_next_slot],
                                        &window->bg_slots[window->bg_curr_slot]);
        }
        return 0;
    }

    if (window->input_mode != TRACER_FP_INPUT_HDF5 || window->hdf5_meta == 0 ||
        window->raw_slots == 0 || window->raw_prev_slot == 0 ||
        window->raw_curr_slot == 0 || window->raw_next_slot == 0 ||
        window->source_offsets == 0) {
        return -1;
    }

    if (tracer_load_hdf5_raw_snapshot_slice(window->hdf5_meta, nsnp_i + window->start_snapshot - 1,
                                            window->tracer_start, window->source_offsets, window->ntracer,
                                            &window->raw_slots[0]) != 0 ||
        tracer_load_hdf5_raw_snapshot_slice(window->hdf5_meta, nsnp_i + window->start_snapshot,
                                            window->tracer_start, window->source_offsets, window->ntracer,
                                            &window->raw_slots[1]) != 0 ||
        tracer_load_hdf5_raw_snapshot_slice(window->hdf5_meta,
                                            nsnp_i + ((window->start_snapshot + 1 < window->nsnap)
                                                      ? window->start_snapshot + 1 : window->start_snapshot),
                                            window->tracer_start, window->source_offsets, window->ntracer,
                                            &window->raw_slots[2]) != 0) {
        return -1;
    }
    if (tracer_prepare_background_from_raw(
            window->ntracer,
            &window->raw_slots[0],
            &window->raw_slots[1],
            (window->start_snapshot + 1 < window->nsnap) ? &window->raw_slots[2] : &window->raw_slots[1],
            window->z_snap[window->start_snapshot - 1],
            window->z_snap[window->start_snapshot],
            (window->start_snapshot + 1 < window->nsnap)
                ? window->z_snap[window->start_snapshot + 1] : window->z_snap[window->start_snapshot],
            1,
            (window->start_snapshot + 1 < window->nsnap),
            &window->bg_slots[window->bg_curr_slot]) != 0) {
        return -1;
    }

    if (window->start_snapshot + 1 >= window->nsnap) {
        tracer_copy_background_slot(window->ntracer, &window->bg_slots[window->bg_next_slot],
                                    &window->bg_slots[window->bg_curr_slot]);
        *window->raw_prev_slot = 1;
        *window->raw_curr_slot = 2;
        *window->raw_next_slot = 0;
        return 0;
    }

    if (tracer_load_hdf5_raw_snapshot_slice(window->hdf5_meta, nsnp_i + window->start_snapshot,
                                            window->tracer_start, window->source_offsets, window->ntracer,
                                            &window->raw_slots[0]) != 0 ||
        tracer_load_hdf5_raw_snapshot_slice(window->hdf5_meta, nsnp_i + window->start_snapshot + 1,
                                            window->tracer_start, window->source_offsets, window->ntracer,
                                            &window->raw_slots[1]) != 0 ||
        tracer_load_hdf5_raw_snapshot_slice(window->hdf5_meta,
                                            nsnp_i + ((window->start_snapshot + 2 < window->nsnap)
                                                      ? window->start_snapshot + 2 : window->start_snapshot + 1),
                                            window->tracer_start, window->source_offsets, window->ntracer,
                                            &window->raw_slots[2]) != 0) {
        return -1;
    }
    if (tracer_prepare_background_from_raw(
            window->ntracer,
            &window->raw_slots[0],
            &window->raw_slots[1],
            (window->start_snapshot + 2 < window->nsnap) ? &window->raw_slots[2] : &window->raw_slots[1],
            window->z_snap[window->start_snapshot],
            window->z_snap[window->start_snapshot + 1],
            (window->start_snapshot + 2 < window->nsnap)
                ? window->z_snap[window->start_snapshot + 2] : window->z_snap[window->start_snapshot + 1],
            1,
            (window->start_snapshot + 2 < window->nsnap),
            &window->bg_slots[window->bg_next_slot]) != 0) {
        return -1;
    }
    *window->raw_prev_slot = 0;
    *window->raw_curr_slot = 1;
    *window->raw_next_slot = 2;
    return 0;
}

static void checkpoint_path(char *path, size_t path_size,
                            const char *dir, int mpi_rank)
{
    if (path == 0 || path_size == 0) return;
    snprintf(path, path_size, "%s/checkpoint_rank%02d.bin", dir, mpi_rank);
}

int tracer_fp_ckpt_write(const TracerFpCkptWrite *checkpoint)
{
    char path[MAX_LINE_LENGTH];
    char tmp_path[MAX_LINE_LENGTH];
    FILE *fp = 0;
    TracerFpCheckpointHeader h;
    size_t wrote;

    if (checkpoint == 0 || checkpoint->checkpoint_dir == 0 ||
        *checkpoint->checkpoint_dir == '\0' || checkpoint->tracer_ids == 0 ||
        checkpoint->crp_state == 0 || checkpoint->cre_state == 0) {
        return -1;
    }
    if (ensure_output_dir(checkpoint->checkpoint_dir) != 0) return -1;

    checkpoint_path(path, sizeof(path), checkpoint->checkpoint_dir, checkpoint->mpi_rank);
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    memset(&h, 0, sizeof(h));
    snprintf(h.magic, sizeof(h.magic), "TFPCHK1");
    h.version = 1;
    h.mpi_rank = checkpoint->mpi_rank;
    h.world_size = checkpoint->world_size;
    h.ntracer = checkpoint->ntracer;
    h.ntracer_global = checkpoint->ntracer_global;
    h.nsnap = checkpoint->nsnap;
    h.nfreq = checkpoint->nfreq;
    h.np_bins = np;
    h.npe_bins = npe;
    h.nsnp_i_value = nsnp_i;
    h.nsnp_f_value = nsnp_f;
    h.completed_snapshot = checkpoint->completed_snapshot;
    h.tracer_start = checkpoint->tracer_start;

    fp = fopen(tmp_path, "wb");
    if (fp == 0) return -1;
    wrote = fwrite(&h, sizeof(h), 1, fp);
    if (wrote != 1) goto fail;
    if (fwrite(checkpoint->tracer_ids, sizeof(int), (size_t)checkpoint->ntracer, fp) !=
        (size_t)checkpoint->ntracer) {
        goto fail;
    }
    if (fwrite(checkpoint->crp_state, sizeof(double), (size_t)checkpoint->ntracer * (size_t)np, fp) !=
        (size_t)checkpoint->ntracer * (size_t)np) {
        goto fail;
    }
    if (fwrite(checkpoint->cre_state, sizeof(double), (size_t)checkpoint->ntracer * (size_t)npe, fp) !=
        (size_t)checkpoint->ntracer * (size_t)npe) {
        goto fail;
    }
    if (fflush(fp) != 0) goto fail;
    if (fsync(fileno(fp)) != 0) goto fail;
    if (fclose(fp) != 0) {
        fp = 0;
        return -1;
    }
    fp = 0;
    if (rename(tmp_path, path) != 0) return -1;
    return 0;

fail:
    if (fp != 0) fclose(fp);
    return -1;
}

int tracer_fp_ckpt_read(const TracerFpCkptRead *checkpoint)
{
    char path[MAX_LINE_LENGTH];
    FILE *fp = 0;
    TracerFpCheckpointHeader h;
    int *checkpoint_ids = 0;
    int ierr = -1;

    if (checkpoint == 0 || checkpoint->restart_dir == 0 || *checkpoint->restart_dir == '\0' ||
        checkpoint->tracer_ids == 0 || checkpoint->crp_state == 0 ||
        checkpoint->cre_state == 0 || checkpoint->completed_snapshot_out == 0) {
        return -1;
    }

    checkpoint_path(path, sizeof(path), checkpoint->restart_dir, checkpoint->mpi_rank);
    fp = fopen(path, "rb");
    if (fp == 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to open checkpoint '%s'\n", path);
        return -1;
    }
    if (fread(&h, sizeof(h), 1, fp) != 1) goto cleanup;
    if (strncmp(h.magic, "TFPCHK1", sizeof(h.magic)) != 0 || h.version != 1) goto cleanup;
    if (h.mpi_rank != checkpoint->mpi_rank || h.world_size != checkpoint->world_size ||
        h.ntracer != checkpoint->ntracer || h.ntracer_global != checkpoint->ntracer_global ||
        h.nsnap != checkpoint->nsnap || h.nfreq != checkpoint->nfreq ||
        h.np_bins != np || h.npe_bins != npe ||
        h.nsnp_i_value != nsnp_i || h.nsnp_f_value != nsnp_f ||
        h.tracer_start != checkpoint->tracer_start ||
        h.completed_snapshot < 0 || h.completed_snapshot >= checkpoint->nsnap) {
        fprintf(stderr, TRACER_FP_PROGNAME ": checkpoint metadata mismatch in '%s'\n", path);
        goto cleanup;
    }
    checkpoint_ids = (int *)calloc((size_t)checkpoint->ntracer, sizeof(int));
    if (checkpoint_ids == 0) goto cleanup;
    if (fread(checkpoint_ids, sizeof(int), (size_t)checkpoint->ntracer, fp) !=
        (size_t)checkpoint->ntracer) {
        goto cleanup;
    }
    if (memcmp(checkpoint_ids, checkpoint->tracer_ids, (size_t)checkpoint->ntracer * sizeof(int)) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": checkpoint tracer order mismatch in '%s'\n", path);
        goto cleanup;
    }
    if (fread(checkpoint->crp_state, sizeof(double), (size_t)checkpoint->ntracer * (size_t)np, fp) !=
        (size_t)checkpoint->ntracer * (size_t)np) {
        goto cleanup;
    }
    if (fread(checkpoint->cre_state, sizeof(double), (size_t)checkpoint->ntracer * (size_t)npe, fp) !=
        (size_t)checkpoint->ntracer * (size_t)npe) {
        goto cleanup;
    }
    *checkpoint->completed_snapshot_out = h.completed_snapshot;
    ierr = 0;

cleanup:
    if (fp != 0) fclose(fp);
    free(checkpoint_ids);
    return ierr;
}
