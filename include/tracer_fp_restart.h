#ifndef INCLUDED_tracer_fp_restart_h_
#define INCLUDED_tracer_fp_restart_h_

#include "tracer_fp.h"
#include "tracer_fp_background.h"

typedef struct {
    int start_snapshot;
    int nsnap;
    TracerFpInputMode input_mode;
    TracerFpBackgroundMode background_mode;
    int ntracer;
    int ntracer_global;
    long int tracer_start;
    const int *source_offsets;
    const long int *tracer_ids;
    const double *z_snap;
    const TracerFpHdf5Meta *hdf5_meta;
    TracerFpRawBackgroundSlot *raw_slots;
    int *raw_prev_slot;
    int *raw_curr_slot;
    int *raw_next_slot;
    TracerFpBackgroundSlot *bg_slots;
    int bg_curr_slot;
    int bg_next_slot;
} TracerFpRestartWin;

typedef struct {
    const char *checkpoint_dir;
    int mpi_rank;
    int world_size;
    int ntracer;
    int ntracer_global;
    int nsnap;
    int nfreq;
    int completed_snapshot;
    long int tracer_start;
    const int *tracer_ids;
    const double *crp_state;
    const double *cre_state;
} TracerFpCkptWrite;

typedef struct {
    const char *restart_dir;
    int mpi_rank;
    int world_size;
    int ntracer;
    int ntracer_global;
    int nsnap;
    int nfreq;
    long int tracer_start;
    const int *tracer_ids;
    double *crp_state;
    double *cre_state;
    int *completed_snapshot_out;
} TracerFpCkptRead;

int tracer_fp_restart_bg(const TracerFpRestartWin *window);
int tracer_fp_ckpt_write(const TracerFpCkptWrite *checkpoint);
int tracer_fp_ckpt_read(const TracerFpCkptRead *checkpoint);

#endif
