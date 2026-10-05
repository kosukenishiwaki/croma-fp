/*
    tracer_fp_background.c

    K. Nishiwaki, 2026-06-18
    - prepare background MHD quantities for each snashot
    - read from tracer hdf5 
    - "frozen" module is for non-evolving fixed background (mostly for testing)
*/


#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hdf5.h>

#include "COSFUNC.h"
#include "FP_Coef.h"
#include "READFILE.h"
#include "params.h"
#include "read_grid_hdf5.h"
#include "tracer_fp_background.h"
#include "tracer_fp_data.h"

#ifdef FP_USE_CUDA_BACKEND
#define TRACER_FP_PROGNAME "tracer_fp_cuda.out"
#else
#define TRACER_FP_PROGNAME "tracer_fp_cpu.out"
#endif

static const double kTracerGyr = 3.1536e16;
static const double kTracerKpc = 3.0857e21;

static int tracer_read_hdf5_dataset_slice(const char *filename,
                                          const char *dataset_name,
                                          hsize_t total_count,
                                          hsize_t offset,
                                          hsize_t count,
                                          double *out);

static int read_tracer_mass_hdf5_fallback(int nsnp,
                                          int dims_all,
                                          double *m_tracer,
                                          double z)
{
    (void)dims_all;
    (void)z;

    char file_name[MAX_LINE_LENGTH];

    snprintf(file_name, sizeof(file_name), "%s%s%04d%s%s",
             tracer_file_dir,
             tracer_filename_base1,
             nsnp,
             tracer_filename_base2,
             tracer_file_extension);

    printf("reading %s\n", file_name);
    printf("reading %s\n", DATASET_NAME_MASS);
    return (tracer_read_hdf5_dataset_slice(file_name, DATASET_NAME_MASS,
                                           (hsize_t)dims_all, 0,
                                           (hsize_t)dims_all, m_tracer) == 0) ? 1 : -1;
}

static int read_redshift_hdf5(int dims_all,
                              double *z_snp,
                              double *z_nxt_snp,
                              double *c_dens,
                              double *c_velo)
{
    (void)dims_all;

    for (int nsnp = nsnp_i; nsnp < nsnp_f; nsnp++) {
        char file_name[MAX_LINE_LENGTH];
        hid_t file_id;
        hid_t dataset_id;
        herr_t status;

        z_nxt_snp[nsnp] = 0.0;
        snprintf(file_name, sizeof(file_name), "%s%s%04d%s%s",
                 tracer_file_dir,
                 tracer_filename_base1,
                 nsnp,
                 tracer_filename_base2,
                 tracer_file_extension);

        file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id == H5I_INVALID_HID) {
            printf("Error: H5Fopen failed, file %s not found or inaccessible.\n", file_name);
            return -1;
        }

        dataset_id = H5Dopen(file_id, DATASET_NAME_REDSHIFT, H5P_DEFAULT);
        if (dataset_id == H5I_INVALID_HID) {
            H5Fclose(file_id);
            return -1;
        }
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, &z_snp[nsnp]);
        H5Dclose(dataset_id);
        if (status < 0) {
            H5Fclose(file_id);
            return -1;
        }

        dataset_id = H5Dopen(file_id, DATASET_NAME_DCONV, H5P_DEFAULT);
        if (dataset_id == H5I_INVALID_HID) {
            H5Fclose(file_id);
            return -1;
        }
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, &c_dens[nsnp]);
        H5Dclose(dataset_id);
        if (status < 0) {
            H5Fclose(file_id);
            return -1;
        }

        dataset_id = H5Dopen(file_id, DATASET_NAME_VCONV, H5P_DEFAULT);
        if (dataset_id == H5I_INVALID_HID) {
            H5Fclose(file_id);
            return -1;
        }
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, &c_velo[nsnp]);
        H5Dclose(dataset_id);
        if (status < 0) {
            H5Fclose(file_id);
            return -1;
        }

        H5Fclose(file_id);

        z_nxt_snp[nsnp] = z_snp[nsnp] - 0.01;
        if (nsnp > nsnp_i) {
            z_nxt_snp[nsnp - 1] = z_snp[nsnp];
        }
    }

    return 0;
}

static int read_param_hdf5(int nsnp,
                           int dims_all,
                           char *file_name,
                           const char *data_name,
                           double *param)
{
    (void)nsnp;

    return tracer_read_hdf5_dataset_slice(file_name, data_name,
                                          (hsize_t)dims_all, 0,
                                          (hsize_t)dims_all, param);
}

int tracer_fp_frozen_runtime(const char *params_file,
                                         int *runtime_nsnap,
                                         double *runtime_dt_gyr)
{
    int nout;
    double dt_gyr;

    if (params_file == 0 || runtime_nsnap == 0 || runtime_dt_gyr == 0) return -1;
    if (read_param_file_noMPI(params_file) != SUCCESS) return -1;
    if (t_fp_total <= 0.0) return -1;

    nout = (n_fp_out >= 1) ? n_fp_out : 1;
    dt_gyr = t_fp_total / (double)nout;
    if (dt_gyr <= 0.0) return -1;

    *runtime_nsnap = nout;
    *runtime_dt_gyr = dt_gyr;
    return 0;
}

int tracer_fp_runtime_steps(TracerFpInputMode input_mode,
                                        TracerFpBackgroundMode background_mode,
                                        const char *params_file,
                                        int *runtime_nsnap,
                                        int *frozen_runtime_override,
                                        double *frozen_runtime_dt_gyr)
{
    const int available_snapshots = nsnp_f - nsnp_i;

    if (runtime_nsnap == 0 || frozen_runtime_override == 0 || frozen_runtime_dt_gyr == 0) {
        return -1;
    }

    *frozen_runtime_override = 0;
    *frozen_runtime_dt_gyr = 0.0;

    if (input_mode == TRACER_FP_INPUT_HDF5 &&
        background_mode == TRACER_FP_BACKGROUND_FROZEN) {
        if (tracer_fp_frozen_runtime(params_file, frozen_runtime_override,
                                                 frozen_runtime_dt_gyr) != 0 ||
            *frozen_runtime_override <= 0) {
            return -1;
        }
        *runtime_nsnap = *frozen_runtime_override;
        return 0;
    }

    if (available_snapshots <= 0) return -1;
    *runtime_nsnap = available_snapshots;
    return 0;
}

int tracer_fp_hdf5_init(const char *params_file,
                              int ntracer,
                              int ntracer_global,
                              int nsnap,
                              TracerFpBackgroundMode background_mode,
                              TracerDataStorage *raw_storage,
                              TracerFpRawBackgroundSlot *raw_slots,
                              TracerFpHdf5Meta *meta,
                              double *dt_snap,
                              double *z_snap)
{
    if (ntracer <= 0 || nsnap <= 0 || raw_storage == 0 || raw_slots == 0 ||
        meta == 0 || dt_snap == 0 || z_snap == 0) {
        return -1;
    }

    if (tracer_data_storage_alloc(raw_storage, 3u * (size_t)ntracer) != 0) {
        return -1;
    }

    tracer_bind_raw_background_slot(ntracer, 0, raw_storage, &raw_slots[0]);
    tracer_bind_raw_background_slot(ntracer, 1, raw_storage, &raw_slots[1]);
    tracer_bind_raw_background_slot(ntracer, 2, raw_storage, &raw_slots[2]);

    if (tracer_init_hdf5_meta(meta, params_file, ntracer_global) != 0 ||
        tracer_fill_hdf5_timeline(meta, dt_snap, z_snap, nsnap,
                                  background_mode, params_file) != 0) {
        return -1;
    }

    return 0;
}

int tracer_fp_hdf5_load_start(const TracerFpHdf5Meta *meta,
                                          TracerFpBackgroundMode background_mode,
                                          long int tracer_start,
                                          const int *selected_offsets,
                                          int ntracer,
                                          int nsnap,
                                          TracerFpRawBackgroundSlot raw_slots[3],
                                          int raw_prev_slot,
                                          int raw_curr_slot,
                                          int raw_next_slot,
                                          int *read_calls_out)
{
    int read_calls = 0;

    if (meta == 0 || selected_offsets == 0 || raw_slots == 0 || read_calls_out == 0 ||
        ntracer <= 0 || nsnap <= 0) {
        return -1;
    }

    if (background_mode == TRACER_FP_BACKGROUND_FROZEN) {
        if (tracer_load_hdf5_raw_snapshot_slice(meta, nsnp_i,
                                                tracer_start, selected_offsets, ntracer,
                                                &raw_slots[0]) != 0) {
            return -1;
        }
        read_calls = 1;
        *read_calls_out = read_calls;
        return 0;
    }

    if (tracer_load_hdf5_raw_snapshot_slice(meta, nsnp_i + 0,
                                            tracer_start, selected_offsets, ntracer,
                                            &raw_slots[raw_prev_slot]) != 0) {
        return -1;
    }
    read_calls++;
    if (nsnap > 1 &&
        tracer_load_hdf5_raw_snapshot_slice(meta, nsnp_i + 1,
                                            tracer_start, selected_offsets, ntracer,
                                            &raw_slots[raw_curr_slot]) != 0) {
        return -1;
    }
    if (nsnap > 1) read_calls++;
    if (nsnap > 2 &&
        tracer_load_hdf5_raw_snapshot_slice(meta, nsnp_i + 2,
                                            tracer_start, selected_offsets, ntracer,
                                            &raw_slots[raw_next_slot]) != 0) {
        return -1;
    }
    if (nsnap > 2) read_calls++;

    *read_calls_out = read_calls;
    return 0;
}

void tracer_build_snapshot_filename(int snap_index,
                                    char *filename,
                                    size_t filename_size)
{
    snprintf(filename, filename_size, "%s%s%04d%s%s",
             tracer_file_dir, tracer_filename_base1,
             snap_index, tracer_filename_base2, tracer_file_extension);
}

void tracer_zero_background_history_slot(size_t off,
                                         TracerDataHistory *history)
{
    if (history == 0) return;
    if (history->n_gas) history->n_gas[off] = 0.0;
    if (history->kbt) history->kbt[off] = 0.0;
    if (history->b_field) history->b_field[off] = 0.0;
    if (history->divv) history->divv[off] = 0.0;
    if (history->l_turb) history->l_turb[off] = 0.0;
    if (history->dv_imc) history->dv_imc[off] = 0.0;
    if (history->cs) history->cs[off] = 0.0;
    if (history->beta_pl) history->beta_pl[off] = 0.0;
}

void tracer_store_background_history_slot(size_t off,
                                          const FpBackgroundCellOutput *bg_out,
                                          TracerDataHistory *history)
{
    if (bg_out == 0) {
        tracer_zero_background_history_slot(off, history);
        return;
    }

    if (history == 0) return;
    if (history->n_gas) history->n_gas[off] = bg_out->n_gas;
    if (history->kbt) history->kbt[off] = bg_out->kbt_GeV;
    if (history->b_field) history->b_field[off] = bg_out->b_eff_G;
    if (history->divv) history->divv[off] = bg_out->divv_gyr;
    if (history->l_turb) history->l_turb[off] = bg_out->l_turb_mpc;
    if (history->dv_imc) history->dv_imc[off] = bg_out->dv_imc_cms;
    if (history->cs) history->cs[off] = bg_out->cs_cms;
    if (history->beta_pl) history->beta_pl[off] = bg_out->beta_pl;
}

void tracer_bind_background_slot(int ntracer,
                                 int slot,
                                 const TracerDataHistory *history,
                                 TracerFpBackgroundSlot *view)
{
    const size_t off = (size_t)slot * (size_t)ntracer;
    if (view == 0) return;

    view->n_gas = (history != 0 && history->n_gas != 0) ? (history->n_gas + off) : 0;
    view->kbt = (history != 0 && history->kbt != 0) ? (history->kbt + off) : 0;
    view->b_field = (history != 0 && history->b_field != 0) ? (history->b_field + off) : 0;
    view->divv = (history != 0 && history->divv != 0) ? (history->divv + off) : 0;
    view->l_turb = (history != 0 && history->l_turb != 0) ? (history->l_turb + off) : 0;
    view->dv_imc = (history != 0 && history->dv_imc != 0) ? (history->dv_imc + off) : 0;
    view->cs = (history != 0 && history->cs != 0) ? (history->cs + off) : 0;
    view->beta_pl = (history != 0 && history->beta_pl != 0) ? (history->beta_pl + off) : 0;
}

void tracer_bind_raw_background_slot(int ntracer,
                                     int slot,
                                     const TracerDataStorage *storage,
                                     TracerFpRawBackgroundSlot *view)
{
    const size_t off = (size_t)slot * (size_t)ntracer;
    if (view == 0) return;

    view->temp = (storage != 0 && storage->temp != 0) ? (storage->temp + off) : 0;
    view->rho = (storage != 0 && storage->rho != 0) ? (storage->rho + off) : 0;
    view->bx = (storage != 0 && storage->bx != 0) ? (storage->bx + off) : 0;
    view->by = (storage != 0 && storage->by != 0) ? (storage->by + off) : 0;
    view->bz = (storage != 0 && storage->bz != 0) ? (storage->bz + off) : 0;
    view->divv = (storage != 0 && storage->divv != 0) ? (storage->divv + off) : 0;
    view->rotv = (storage != 0 && storage->rotv != 0) ? (storage->rotv + off) : 0;
    view->lturb = (storage != 0 && storage->lturb != 0) ? (storage->lturb + off) : 0;
    view->mach = (storage != 0 && storage->mach != 0) ? (storage->mach + off) : 0;
    view->prestemp = (storage != 0 && storage->prestemp != 0) ? (storage->prestemp + off) : 0;
    view->presden = (storage != 0 && storage->presden != 0) ? (storage->presden + off) : 0;
    view->pre_density_cgs = (storage != 0 && storage->pre_density_cgs != 0) ? (storage->pre_density_cgs + off) : 0;
    view->upstream_speed_cgs = (storage != 0 && storage->upstream_speed_cgs != 0) ? (storage->upstream_speed_cgs + off) : 0;
    view->shock_side_code = (storage != 0 && storage->shock_side_code != 0) ? (storage->shock_side_code + off) : 0;
    view->dsa_trigger = (storage != 0 && storage->dsa_trigger != 0) ? (storage->dsa_trigger + off) : 0;
    view->dsa_mach = (storage != 0 && storage->dsa_mach != 0) ? (storage->dsa_mach + off) : 0;
    view->dsa_pre_density = (storage != 0 && storage->dsa_pre_density != 0) ? (storage->dsa_pre_density + off) : 0;
    view->dsa_kinetic_energy_flux_cgs =
        (storage != 0 && storage->dsa_kinetic_energy_flux_cgs != 0)
            ? (storage->dsa_kinetic_energy_flux_cgs + off)
            : 0;
    view->has_shock = 0;
    view->has_dsa = 0;
}

void tracer_copy_background_slot(int ntracer,
                                 const TracerFpBackgroundSlot *dst,
                                 const TracerFpBackgroundSlot *src)
{
    const size_t bytes = (size_t)ntracer * sizeof(double);
    if (ntracer <= 0 || dst == 0 || src == 0) return;

    memcpy(dst->n_gas, src->n_gas, bytes);
    memcpy(dst->kbt, src->kbt, bytes);
    memcpy(dst->b_field, src->b_field, bytes);
    memcpy(dst->divv, src->divv, bytes);
    memcpy(dst->l_turb, src->l_turb, bytes);
    memcpy(dst->dv_imc, src->dv_imc, bytes);
    memcpy(dst->cs, src->cs, bytes);
    memcpy(dst->beta_pl, src->beta_pl, bytes);
}

static int tracer_read_hdf5_dataset_slice_data(hid_t data_id,
                                               hsize_t total_count,
                                               hsize_t offset,
                                               hsize_t count,
                                               double *out)
{
    hid_t file_space = H5I_INVALID_HID;
    hid_t mem_space = H5I_INVALID_HID;
    hsize_t nfile;
    herr_t status;
    int ierr = -1;

    if (data_id == H5I_INVALID_HID || out == 0) return -1;
    if (count == 0) return 0;

    file_space = H5Dget_space(data_id);
    if (file_space == H5I_INVALID_HID) goto cleanup;

    nfile = (hsize_t)H5Sget_simple_extent_npoints(file_space);
    if (nfile < total_count || offset > total_count || count > total_count - offset) {
        goto cleanup;
    }

    status = H5Sselect_hyperslab(file_space, H5S_SELECT_SET, &offset, NULL, &count, NULL);
    if (status < 0) goto cleanup;
    mem_space = H5Screate_simple(1, &count, NULL);
    if (mem_space == H5I_INVALID_HID) goto cleanup;
    status = H5Dread(data_id, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, out);
    if (status < 0) goto cleanup;

    ierr = 0;

cleanup:
    if (mem_space != H5I_INVALID_HID) H5Sclose(mem_space);
    if (file_space != H5I_INVALID_HID) H5Sclose(file_space);
    return ierr;
}

static int tracer_read_hdf5_dataset_slice_open(hid_t file_id,
                                               const char *dataset_name,
                                               hsize_t total_count,
                                               hsize_t offset,
                                               hsize_t count,
                                               double *out)
{
    hid_t data_id = H5I_INVALID_HID;
    int ierr = -1;

    if (file_id == H5I_INVALID_HID || dataset_name == 0 || out == 0) return -1;

    data_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (data_id == H5I_INVALID_HID) return -1;

    ierr = tracer_read_hdf5_dataset_slice_data(data_id, total_count, offset, count, out);
    if (ierr != 0) {
        hid_t space_id = H5Dget_space(data_id);
        if (space_id != H5I_INVALID_HID) {
            const hsize_t nfile = (hsize_t)H5Sget_simple_extent_npoints(space_id);
            if (nfile < total_count || offset > total_count || count > total_count - offset) {
                fprintf(stderr,
                        TRACER_FP_PROGNAME ": HDF5 dataset '%s' has %lld entries; "
                        "requested N_TRACERS=%lld, offset=%lld, count=%lld\n",
                        dataset_name,
                        (long long)nfile,
                        (long long)total_count,
                        (long long)offset,
                        (long long)count);
            }
            H5Sclose(space_id);
        }
    }

    if (data_id != H5I_INVALID_HID) H5Dclose(data_id);
    return ierr;
}

static int tracer_read_hdf5_dataset_slice(const char *filename,
                                          const char *dataset_name,
                                          hsize_t total_count,
                                          hsize_t offset,
                                          hsize_t count,
                                          double *out)
{
    hid_t file_id = H5I_INVALID_HID;
    int ierr = -1;

    if (filename == 0) return -1;
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID) return -1;
    ierr = tracer_read_hdf5_dataset_slice_open(file_id, dataset_name,
                                               total_count, offset, count, out);
    H5Fclose(file_id);
    return ierr;
}

static int tracer_read_hdf5_dataset_slice_open_optional(hid_t file_id,
                                                        const char *dataset_name,
                                                        hsize_t total_count,
                                                        hsize_t offset,
                                                        hsize_t count,
                                                        double *out)
{
    int ierr;

    H5E_BEGIN_TRY {
        ierr = tracer_read_hdf5_dataset_slice_open(file_id, dataset_name,
                                                   total_count, offset, count, out);
    } H5E_END_TRY;

    return ierr;
}

static int tracer_hdf5_dataset_exists_open(hid_t file_id, const char *dataset_name)
{
    htri_t exists;

    if (file_id == H5I_INVALID_HID || dataset_name == 0) return 0;
    H5E_BEGIN_TRY {
        exists = H5Lexists(file_id, dataset_name, H5P_DEFAULT);
    } H5E_END_TRY;
    return (exists > 0) ? 1 : 0;
}

static int dsa_spec_is_off(const char *spec)
{
    return (spec == 0 || *spec == '\0' ||
            strcmp(spec, "off") == 0 ||
            strcmp(spec, "none") == 0 ||
            strcmp(spec, "0") == 0);
}

static const char *tracer_hdf5_layout_name(H5D_layout_t layout)
{
    switch (layout) {
    case H5D_COMPACT:
        return "compact";
    case H5D_CONTIGUOUS:
        return "contiguous";
    case H5D_CHUNKED:
        return "chunked";
#ifdef H5D_VIRTUAL
    case H5D_VIRTUAL:
        return "virtual";
#endif
    default:
        return "unknown";
    }
}

static void tracer_hdf5_dims_to_string(int ndims,
                                       const hsize_t *dims,
                                       char *buffer,
                                       size_t buffer_size)
{
    size_t used = 0;
    int idim;

    if (buffer == 0 || buffer_size == 0) return;
    buffer[0] = '\0';
    if (ndims <= 0 || dims == 0) {
        snprintf(buffer, buffer_size, "-");
        return;
    }
    for (idim = 0; idim < ndims; idim++) {
        const int nwritten = snprintf(buffer + used,
                                      (used < buffer_size) ? buffer_size - used : 0,
                                      "%s%llu",
                                      (idim == 0) ? "" : "x",
                                      (unsigned long long)dims[idim]);
        if (nwritten < 0) break;
        used += (size_t)nwritten;
        if (used >= buffer_size) {
            buffer[buffer_size - 1] = '\0';
            break;
        }
    }
}

static void tracer_hdf5_log_dataset_layout(FILE *fp,
                                           hid_t file_id,
                                           int snap_id,
                                           const char *role,
                                           const char *dataset_name)
{
    hid_t data_id = H5I_INVALID_HID;
    hid_t space_id = H5I_INVALID_HID;
    hid_t dcpl_id = H5I_INVALID_HID;
    hid_t type_id = H5I_INVALID_HID;
    hsize_t dims[16];
    hsize_t chunk_dims[16];
    int ndims = -1;
    int chunk_ndims = 0;
    int filter_count = -1;
    size_t elem_size = 0;
    hsize_t storage_size = 0;
    H5D_layout_t layout = H5D_LAYOUT_ERROR;
    char dims_text[256];
    char chunk_text[256];
    int exists = 0;

    if (fp == 0 || file_id == H5I_INVALID_HID || dataset_name == 0) return;

    dims_text[0] = '\0';
    chunk_text[0] = '\0';
    H5E_BEGIN_TRY {
        exists = (H5Lexists(file_id, dataset_name, H5P_DEFAULT) > 0) ? 1 : 0;
    } H5E_END_TRY;
    if (!exists) {
        fprintf(fp, "%d\t%s\t%s\t0\tmissing\t-1\t-\t0\t0\t0\t-\t-1\n",
                snap_id, (role != 0) ? role : "unknown", dataset_name);
        return;
    }

    data_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (data_id == H5I_INVALID_HID) {
        fprintf(fp, "%d\t%s\t%s\t1\topen_failed\t-1\t-\t0\t0\t0\t-\t-1\n",
                snap_id, (role != 0) ? role : "unknown", dataset_name);
        return;
    }

    space_id = H5Dget_space(data_id);
    if (space_id != H5I_INVALID_HID) {
        ndims = H5Sget_simple_extent_ndims(space_id);
        if (ndims > 0 && ndims <= (int)(sizeof(dims) / sizeof(dims[0]))) {
            H5Sget_simple_extent_dims(space_id, dims, 0);
            tracer_hdf5_dims_to_string(ndims, dims, dims_text, sizeof(dims_text));
        } else {
            snprintf(dims_text, sizeof(dims_text), "-");
        }
    } else {
        snprintf(dims_text, sizeof(dims_text), "-");
    }

    type_id = H5Dget_type(data_id);
    if (type_id != H5I_INVALID_HID) {
        elem_size = H5Tget_size(type_id);
    }

    dcpl_id = H5Dget_create_plist(data_id);
    if (dcpl_id != H5I_INVALID_HID) {
        layout = H5Pget_layout(dcpl_id);
        filter_count = H5Pget_nfilters(dcpl_id);
        if (layout == H5D_CHUNKED) {
            chunk_ndims = H5Pget_chunk(dcpl_id,
                                       (int)(sizeof(chunk_dims) / sizeof(chunk_dims[0])),
                                       chunk_dims);
            if (chunk_ndims > 0) {
                tracer_hdf5_dims_to_string(chunk_ndims, chunk_dims,
                                           chunk_text, sizeof(chunk_text));
            } else {
                snprintf(chunk_text, sizeof(chunk_text), "-");
            }
        } else {
            snprintf(chunk_text, sizeof(chunk_text), "-");
        }
    } else {
        snprintf(chunk_text, sizeof(chunk_text), "-");
    }

    storage_size = H5Dget_storage_size(data_id);
    fprintf(fp, "%d\t%s\t%s\t1\t%s\t%d\t%s\t%llu\t%zu\t%d\t%s\t%d\n",
            snap_id,
            (role != 0) ? role : "unknown",
            dataset_name,
            tracer_hdf5_layout_name(layout),
            ndims,
            dims_text,
            (unsigned long long)storage_size,
            elem_size,
            chunk_ndims,
            chunk_text,
            filter_count);

    if (type_id != H5I_INVALID_HID) H5Tclose(type_id);
    if (dcpl_id != H5I_INVALID_HID) H5Pclose(dcpl_id);
    if (space_id != H5I_INVALID_HID) H5Sclose(space_id);
    if (data_id != H5I_INVALID_HID) H5Dclose(data_id);
}

int tracer_fp_write_hdf5_layout_log(const char *output_dir,
                                    int snap_id)
{
    typedef struct {
        const char *role;
        const char *dataset_name;
    } DatasetEntry;

    const TracerInputFieldNames *fields = &kTracerInputFields;
    const DatasetEntry entries[] = {
        {"temperature", fields->temperature},
        {"density", fields->density},
        {"B_x", fields->bx},
        {"B_y", fields->by},
        {"B_z", fields->bz},
        {"div_v", fields->divv},
        {"curl_v", fields->curlv},
        {"lturb", fields->lturb},
        {"mach", fields->mach},
        {"prestemp", fields->prestemp},
        {"presden", fields->presden},
        {"pre_density_cgs", fields->pre_density_cgs},
        {"upstream_speed_cgs", fields->upstream_speed_cgs},
        {"shock_side_code", fields->shock_side_code},
        {"dsa_trigger", fields->dsa_trigger},
        {"dsa_mach", fields->dsa_mach},
        {"dsa_pre_density", fields->dsa_pre_density},
        {"dsa_kinetic_energy_flux_cgs", fields->dsa_kinetic_energy_flux_cgs},
        {"M_tracer", fields->mass},
        {"particle_ID", fields->particle_id},
        {"Redshift", fields->redshift},
        {"time", fields->time}
    };
    char filename[MAX_LINE_LENGTH];
    char path[MAX_LINE_LENGTH];
    hid_t file_id = H5I_INVALID_HID;
    FILE *fp = 0;
    size_t i;

    if (output_dir == 0 || *output_dir == '\0') return -1;

    tracer_build_snapshot_filename(snap_id, filename, sizeof(filename));
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID) return -1;

    snprintf(path, sizeof(path), "%s/input_hdf5_layout.tsv", output_dir);
    fp = fopen(path, "w");
    if (fp == 0) {
        H5Fclose(file_id);
        return -1;
    }

    fprintf(fp,
            "physical_snapshot\trole\tdataset\texists\tlayout\tndims\tdims\tstorage_size_bytes\telement_size_bytes\tchunk_ndims\tchunk_dims\tfilter_count\n");
    for (i = 0; i < sizeof(entries) / sizeof(entries[0]); i++) {
        tracer_hdf5_log_dataset_layout(fp, file_id, snap_id,
                                       entries[i].role, entries[i].dataset_name);
    }

    fclose(fp);
    H5Fclose(file_id);
    return 0;
}

int tracer_fp_count_selected(const int *selected_offsets, int nselected)
{
    int itr;
    int count = 0;

    if (selected_offsets == 0 || nselected <= 0) return 0;
    for (itr = 0; itr < nselected; itr++) {
        if (selected_offsets[itr] >= 0) count++;
    }
    return count;
}

void tracer_fp_selected_offset_stats(const int *selected_offsets,
                                     int nselected,
                                     int *selected_count_out,
                                     int *run_count_out,
                                     int *min_run_len_out,
                                     int *max_run_len_out,
                                     int *max_gap_out)
{
    int selected_count = 0;
    int run_count = 0;
    int min_run_len = 0;
    int max_run_len = 0;
    int max_gap = 0;
    int itr;

    if (selected_offsets != 0 && nselected > 0) {
        int current_run_len = 0;
        int prev = 0;
        int have_prev = 0;

        for (itr = 0; itr < nselected; itr++) {
            const int off = selected_offsets[itr];
            if (off < 0) continue;

            selected_count++;
            if (!have_prev || off != prev + 1) {
                if (current_run_len > 0) {
                    if (min_run_len == 0 || current_run_len < min_run_len) {
                        min_run_len = current_run_len;
                    }
                    if (current_run_len > max_run_len) max_run_len = current_run_len;
                }
                if (have_prev && off > prev + 1) {
                    const int gap = off - prev - 1;
                    if (gap > max_gap) max_gap = gap;
                }
                run_count++;
                current_run_len = 1;
            } else {
                current_run_len++;
            }
            prev = off;
            have_prev = 1;
        }
        if (current_run_len > 0) {
            if (min_run_len == 0 || current_run_len < min_run_len) {
                min_run_len = current_run_len;
            }
            if (current_run_len > max_run_len) max_run_len = current_run_len;
        }
    }

    if (selected_count_out != 0) *selected_count_out = selected_count;
    if (run_count_out != 0) *run_count_out = run_count;
    if (min_run_len_out != 0) *min_run_len_out = min_run_len;
    if (max_run_len_out != 0) *max_run_len_out = max_run_len;
    if (max_gap_out != 0) *max_gap_out = max_gap;
}

static int tracer_read_hdf5_dataset_selected_open(hid_t file_id,
                                                  const char *dataset_name,
                                                  hsize_t total_count,
                                                  long int base_offset,
                                                  const int *selected_offsets,
                                                  int nselected,
                                                  double *out)
{
    hid_t data_id = H5I_INVALID_HID;
    int itr;

    if (selected_offsets == 0) {
        return tracer_read_hdf5_dataset_slice_open(file_id, dataset_name,
                                                   total_count, (hsize_t)base_offset,
                                                   (hsize_t)nselected, out);
    }

    if (nselected == 0) return 0;
    if (file_id == H5I_INVALID_HID || dataset_name == 0 || out == 0 || nselected < 0) {
        return -1;
    }

    data_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (data_id == H5I_INVALID_HID) return -1;

    for (itr = 0; itr < nselected; ) {
        const int run_begin = itr;
        long int src = base_offset + (long int)selected_offsets[itr];
        hsize_t run_count = 1;

        if (src < 0 || src >= (long int)total_count) {
            H5Dclose(data_id);
            return -1;
        }

        while ((itr + 1) < nselected) {
            const long int next_rel = (long int)selected_offsets[itr + 1];
            const long int curr_rel = (long int)selected_offsets[itr];
            if (next_rel != curr_rel + 1) break;
            itr++;
            run_count++;
        }

        if (tracer_read_hdf5_dataset_slice_data(data_id,
                                                total_count,
                                                (hsize_t)src,
                                                run_count,
                                                out + run_begin) != 0) {
            H5Dclose(data_id);
            return -1;
        }
        itr++;
    }

    H5Dclose(data_id);
    return 0;
}

static int tracer_read_hdf5_dataset_selected_open_optional(hid_t file_id,
                                                           const char *dataset_name,
                                                           hsize_t total_count,
                                                           long int base_offset,
                                                           const int *selected_offsets,
                                                           int nselected,
                                                           double *out)
{
    int ierr;

    H5E_BEGIN_TRY {
        ierr = tracer_read_hdf5_dataset_selected_open(file_id, dataset_name,
                                                      total_count, base_offset,
                                                      selected_offsets, nselected, out);
    } H5E_END_TRY;

    return ierr;
}

void tracer_free_hdf5_meta(TracerFpHdf5Meta *meta)
{
    if (meta == 0) return;
    free(meta->z_full);
    free(meta->z_nxt_full);
    free(meta->c_dens);
    free(meta->c_velo);
    free(meta->c_mag);
    memset(meta, 0, sizeof(*meta));
}

int tracer_init_hdf5_meta(TracerFpHdf5Meta *meta,
                          const char *params_file,
                          int ntracer_global)
{
    int isnap;

    if (meta == 0 || params_file == 0 || ntracer_global <= 0) return -1;
    memset(meta, 0, sizeof(*meta));

    if (read_param_file_noMPI(params_file) != SUCCESS) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to read params file '%s'\n", params_file);
        return -1;
    }
    if (N_TRACERS <= 0 || nsnp_f <= nsnp_i) {
        fprintf(stderr, TRACER_FP_PROGNAME ": invalid tracer snapshot metadata in '%s'\n", params_file);
        return -1;
    }
    if (ntracer_global > (int)N_TRACERS) {
        fprintf(stderr, TRACER_FP_PROGNAME ": requested tracer count=%d exceeds N_TRACERS=%ld\n",
                ntracer_global, N_TRACERS);
        return -1;
    }

    meta->z_full = (double *)calloc((size_t)nsnp_f, sizeof(double));
    meta->z_nxt_full = (double *)calloc((size_t)nsnp_f, sizeof(double));
    meta->c_dens = (double *)calloc((size_t)nsnp_f, sizeof(double));
    meta->c_velo = (double *)calloc((size_t)nsnp_f, sizeof(double));
    meta->c_mag = (double *)calloc((size_t)nsnp_f, sizeof(double));
    if (meta->z_full == 0 || meta->z_nxt_full == 0 || meta->c_dens == 0 ||
        meta->c_velo == 0 || meta->c_mag == 0) {
        tracer_free_hdf5_meta(meta);
        return -1;
    }

    if (read_redshift_hdf5((int)N_TRACERS, meta->z_full, meta->z_nxt_full,
                           meta->c_dens, meta->c_velo) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to read redshift/units from tracer HDF5 series\n");
        tracer_free_hdf5_meta(meta);
        return -1;
    }

    for (isnap = 0; isnap < nsnp_f; isnap++) {
        meta->c_mag[isnap] = sqrt(4.0 * M_PI * meta->c_dens[isnap]) * meta->c_velo[isnap];
    }
    return 0;
}

int tracer_fill_hdf5_timeline(const TracerFpHdf5Meta *meta,
                              double *dt_snap,
                              double *z_snap,
                              int nsnap,
                              TracerFpBackgroundMode background_mode,
                              const char *params_file)
{
    int isnap;
    int runtime_nsnap = 0;
    double runtime_dt_gyr = 0.0;

    if (meta == 0 || dt_snap == 0 || z_snap == 0 || nsnap <= 0) return -1;

    if (background_mode == TRACER_FP_BACKGROUND_FROZEN) {
        if (tracer_fp_frozen_runtime(params_file, &runtime_nsnap, &runtime_dt_gyr) != 0 ||
            runtime_nsnap != nsnap) {
            fprintf(stderr, TRACER_FP_PROGNAME ": frozen runtime mismatch for '%s'\n", params_file);
            return -1;
        }
        for (isnap = 0; isnap < nsnap; isnap++) {
            dt_snap[isnap] = runtime_dt_gyr;
            z_snap[isnap] = meta->z_full[nsnp_i];
        }
        return 0;
    }

    if (nsnap > (nsnp_f - nsnp_i)) {
        fprintf(stderr,
                TRACER_FP_PROGNAME ": runtime steps=%d exceed available snapshot range [%d,%d) from '%s'\n",
                nsnap, nsnp_i, nsnp_f, params_file);
        return -1;
    }

    for (isnap = 0; isnap < nsnap; isnap++) {
        const int snap_id = nsnp_i + isnap;
        dt_snap[isnap] = dz_to_dt(meta->z_full[snap_id] - meta->z_nxt_full[snap_id],
                                  meta->z_full[snap_id]);
        z_snap[isnap] = meta->z_full[snap_id];
    }
    return 0;
}

int tracer_load_hdf5_raw_snapshot_slice(const TracerFpHdf5Meta *meta,
                                        int snap_id,
                                        long int tracer_start,
                                        const int *selected_offsets,
                                        int ntracer,
                                        TracerFpRawBackgroundSlot *raw)
{
    char filename[MAX_LINE_LENGTH];
    hid_t file_id = H5I_INVALID_HID;
    double *temp_local = 0;
    double *rho_local = 0;
    double *bx_local = 0;
    double *by_local = 0;
    double *bz_local = 0;
    double *divv_local = 0;
    double *rotv_local = 0;
    double *lturb_local = 0;
    int has_mach = 0;
    int has_prestemp = 0;
    int has_presden = 0;
    int has_dsa_trigger = 0;
    int has_dsa_mach = 0;
    int has_dsa_pre_density = 0;
    int has_dsa_flux = 0;
    int exists_mach = 0;
    int exists_prestemp = 0;
    int exists_presden = 0;
    int exists_dsa_trigger = 0;
    int exists_dsa_mach = 0;
    int exists_dsa_pre_density = 0;
    int exists_dsa_flux = 0;
    int itr;
    const TracerInputFieldNames *fields = &kTracerInputFields;
    const int dsa_input_required =
        (!dsa_spec_is_off(DSAInjectionModeSpec) ||
         !dsa_spec_is_off(DSAReaccModeSpec));

    if (meta == 0 || raw == 0 || ntracer <= 0) return -1;

    temp_local = (double *)calloc((size_t)ntracer, sizeof(double));
    rho_local = (double *)calloc((size_t)ntracer, sizeof(double));
    bx_local = (double *)calloc((size_t)ntracer, sizeof(double));
    by_local = (double *)calloc((size_t)ntracer, sizeof(double));
    bz_local = (double *)calloc((size_t)ntracer, sizeof(double));
    divv_local = (double *)calloc((size_t)ntracer, sizeof(double));
    rotv_local = (double *)calloc((size_t)ntracer, sizeof(double));
    lturb_local = (double *)calloc((size_t)ntracer, sizeof(double));
    if (temp_local == 0 || rho_local == 0 || bx_local == 0 || by_local == 0 ||
        bz_local == 0 || divv_local == 0 || rotv_local == 0 || lturb_local == 0) {
        free(temp_local); free(rho_local); free(bx_local); free(by_local);
        free(bz_local); free(divv_local); free(rotv_local); free(lturb_local);
        return -1;
    }

    tracer_build_snapshot_filename(snap_id, filename, sizeof(filename));
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id == H5I_INVALID_HID ||
        tracer_read_hdf5_dataset_selected_open(file_id, fields->temperature,
                                               (hsize_t)N_TRACERS, tracer_start,
                                               selected_offsets, ntracer, temp_local) != 0 ||
        tracer_read_hdf5_dataset_selected_open(file_id, fields->density,
                                               (hsize_t)N_TRACERS, tracer_start,
                                               selected_offsets, ntracer, rho_local) != 0 ||
        tracer_read_hdf5_dataset_selected_open(file_id, fields->bx,
                                               (hsize_t)N_TRACERS, tracer_start,
                                               selected_offsets, ntracer, bx_local) != 0 ||
        tracer_read_hdf5_dataset_selected_open(file_id, fields->by,
                                               (hsize_t)N_TRACERS, tracer_start,
                                               selected_offsets, ntracer, by_local) != 0 ||
        tracer_read_hdf5_dataset_selected_open(file_id, fields->bz,
                                               (hsize_t)N_TRACERS, tracer_start,
                                               selected_offsets, ntracer, bz_local) != 0 ||
        tracer_read_hdf5_dataset_selected_open(file_id, fields->divv,
                                               (hsize_t)N_TRACERS, tracer_start,
                                               selected_offsets, ntracer, divv_local) != 0 ||
        tracer_read_hdf5_dataset_selected_open(file_id, fields->curlv,
                                               (hsize_t)N_TRACERS, tracer_start,
                                               selected_offsets, ntracer, rotv_local) != 0 ||
        tracer_read_hdf5_dataset_selected_open(file_id, fields->lturb,
                                               (hsize_t)N_TRACERS, tracer_start,
                                               selected_offsets, ntracer, lturb_local) != 0) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to read tracer snapshot %d slice from '%s'\n",
                snap_id, filename);
        if (file_id != H5I_INVALID_HID) H5Fclose(file_id);
        free(temp_local); free(rho_local); free(bx_local); free(by_local);
        free(bz_local); free(divv_local); free(rotv_local); free(lturb_local);
        return -1;
    }
    if (raw->mach != 0) memset(raw->mach, 0, (size_t)ntracer * sizeof(double));
    if (raw->prestemp != 0) memset(raw->prestemp, 0, (size_t)ntracer * sizeof(double));
    if (raw->presden != 0) memset(raw->presden, 0, (size_t)ntracer * sizeof(double));
    if (raw->pre_density_cgs != 0) memset(raw->pre_density_cgs, 0, (size_t)ntracer * sizeof(double));
    if (raw->upstream_speed_cgs != 0) memset(raw->upstream_speed_cgs, 0, (size_t)ntracer * sizeof(double));
    if (raw->shock_side_code != 0) memset(raw->shock_side_code, 0, (size_t)ntracer * sizeof(double));
    if (raw->dsa_trigger != 0) memset(raw->dsa_trigger, 0, (size_t)ntracer * sizeof(double));
    if (raw->dsa_mach != 0) memset(raw->dsa_mach, 0, (size_t)ntracer * sizeof(double));
    if (raw->dsa_pre_density != 0) memset(raw->dsa_pre_density, 0, (size_t)ntracer * sizeof(double));
    if (raw->dsa_kinetic_energy_flux_cgs != 0) {
        memset(raw->dsa_kinetic_energy_flux_cgs, 0, (size_t)ntracer * sizeof(double));
    }

    if (dsa_input_required) {
        exists_mach = tracer_hdf5_dataset_exists_open(file_id, fields->mach);
        exists_prestemp = tracer_hdf5_dataset_exists_open(file_id, fields->prestemp);
        exists_presden = tracer_hdf5_dataset_exists_open(file_id, fields->presden);
        exists_dsa_trigger = tracer_hdf5_dataset_exists_open(file_id, fields->dsa_trigger);
        exists_dsa_mach = tracer_hdf5_dataset_exists_open(file_id, fields->dsa_mach);
        exists_dsa_pre_density = tracer_hdf5_dataset_exists_open(file_id, fields->dsa_pre_density);
        exists_dsa_flux =
            tracer_hdf5_dataset_exists_open(file_id, fields->dsa_kinetic_energy_flux_cgs);

        has_mach = (raw->mach != 0 &&
                    tracer_read_hdf5_dataset_selected_open_optional(file_id, fields->mach,
                                                                    (hsize_t)N_TRACERS, tracer_start,
                                                                    selected_offsets, ntracer, raw->mach) == 0);
        has_prestemp = (raw->prestemp != 0 &&
                        tracer_read_hdf5_dataset_selected_open_optional(file_id, fields->prestemp,
                                                                        (hsize_t)N_TRACERS, tracer_start,
                                                                        selected_offsets, ntracer, raw->prestemp) == 0);
        has_presden = (raw->presden != 0 &&
                       tracer_read_hdf5_dataset_selected_open_optional(file_id, fields->presden,
                                                                       (hsize_t)N_TRACERS, tracer_start,
                                                                       selected_offsets, ntracer, raw->presden) == 0);
        if (raw->pre_density_cgs != 0) {
            (void)tracer_read_hdf5_dataset_selected_open_optional(file_id, fields->pre_density_cgs,
                                                                  (hsize_t)N_TRACERS, tracer_start,
                                                                  selected_offsets, ntracer, raw->pre_density_cgs);
        }
        if (raw->upstream_speed_cgs != 0) {
            (void)tracer_read_hdf5_dataset_selected_open_optional(file_id, fields->upstream_speed_cgs,
                                                                  (hsize_t)N_TRACERS, tracer_start,
                                                                  selected_offsets, ntracer, raw->upstream_speed_cgs);
        }
        if (raw->shock_side_code != 0) {
            (void)tracer_read_hdf5_dataset_selected_open_optional(file_id, fields->shock_side_code,
                                                                  (hsize_t)N_TRACERS, tracer_start,
                                                                  selected_offsets, ntracer, raw->shock_side_code);
        }
        has_dsa_trigger = (raw->dsa_trigger != 0 &&
                           tracer_read_hdf5_dataset_selected_open_optional(file_id, fields->dsa_trigger,
                                                                           (hsize_t)N_TRACERS, tracer_start,
                                                                           selected_offsets, ntracer,
                                                                           raw->dsa_trigger) == 0);
        has_dsa_mach = (raw->dsa_mach != 0 &&
                        tracer_read_hdf5_dataset_selected_open_optional(file_id, fields->dsa_mach,
                                                                        (hsize_t)N_TRACERS, tracer_start,
                                                                        selected_offsets, ntracer,
                                                                        raw->dsa_mach) == 0);
        has_dsa_pre_density =
            (raw->dsa_pre_density != 0 &&
             tracer_read_hdf5_dataset_selected_open_optional(file_id, fields->dsa_pre_density,
                                                             (hsize_t)N_TRACERS, tracer_start,
                                                             selected_offsets, ntracer,
                                                             raw->dsa_pre_density) == 0);
        has_dsa_flux =
            (raw->dsa_kinetic_energy_flux_cgs != 0 &&
             tracer_read_hdf5_dataset_selected_open_optional(file_id,
                                                             fields->dsa_kinetic_energy_flux_cgs,
                                                             (hsize_t)N_TRACERS, tracer_start,
                                                             selected_offsets, ntracer,
                                                             raw->dsa_kinetic_energy_flux_cgs) == 0);
        raw->has_shock = (has_mach && has_prestemp && has_presden) ? 1 : 0;
        raw->has_dsa = (has_dsa_trigger && has_dsa_mach && has_dsa_pre_density && has_dsa_flux) ? 1 : 0;
        if (!raw->has_shock && raw->has_dsa) raw->has_shock = 1;
        if (DSAShockRequired && !dsa_spec_is_off(DSAInjectionModeSpec) && !raw->has_shock) {
            fprintf(stderr, TRACER_FP_PROGNAME
                            ": required shock or merge-DSA datasets are missing from '%s'\n",
                    filename);
            fprintf(stderr,
                    TRACER_FP_PROGNAME
                    ": dataset status legacy(exists/read) mach=%d/%d prestemp=%d/%d presden=%d/%d "
                    "merge_dsa(exists/read) trigger=%d/%d mach=%d/%d pre_density=%d/%d flux=%d/%d\n",
                    exists_mach, has_mach,
                    exists_prestemp, has_prestemp,
                    exists_presden, has_presden,
                    exists_dsa_trigger, has_dsa_trigger,
                    exists_dsa_mach, has_dsa_mach,
                    exists_dsa_pre_density, has_dsa_pre_density,
                    exists_dsa_flux, has_dsa_flux);
            H5Fclose(file_id);
            free(temp_local); free(rho_local); free(bx_local); free(by_local);
            free(bz_local); free(divv_local); free(rotv_local); free(lturb_local);
            return -1;
        }
    }
    H5Fclose(file_id);

    for (itr = 0; itr < ntracer; itr++) {
        raw->temp[itr] = temp_local[itr];
        raw->rho[itr] = rho_local[itr] * meta->c_dens[snap_id];
        raw->bx[itr] = bx_local[itr] * meta->c_mag[snap_id];
        raw->by[itr] = by_local[itr] * meta->c_mag[snap_id];
        raw->bz[itr] = bz_local[itr] * meta->c_mag[snap_id];
        raw->divv[itr] = divv_local[itr];
        raw->rotv[itr] = rotv_local[itr];
        raw->lturb[itr] = lturb_local[itr];
    }

    free(temp_local); free(rho_local); free(bx_local); free(by_local);
    free(bz_local); free(divv_local); free(rotv_local); free(lturb_local);
    return 0;
}

int tracer_prepare_background_from_raw(int ntracer,
                                       const TracerFpRawBackgroundSlot *prev_raw,
                                       const TracerFpRawBackgroundSlot *curr_raw,
                                       const TracerFpRawBackgroundSlot *next_raw,
                                       double z_prev,
                                       double z_curr,
                                       double z_next,
                                       int apply_temp_floor,
                                       int allow_curl_interp,
                                       TracerFpBackgroundSlot *out)
{
    const double L_target = L_turb_target_kpc * kTracerKpc;
    const double div_v_Norm = 1.0;
    const double rot_v_Norm = 1.0;
    TracerDataHistory history;
    int itr;

    if (prev_raw == 0 || curr_raw == 0 || next_raw == 0 || out == 0) return -1;

    memset(&history, 0, sizeof(history));
    history.n_gas = out->n_gas;
    history.kbt = out->kbt;
    history.b_field = out->b_field;
    history.divv = out->divv;
    history.l_turb = out->l_turb;
    history.dv_imc = out->dv_imc;
    history.cs = out->cs;
    history.beta_pl = out->beta_pl;

    for (itr = 0; itr < ntracer; itr++) {
        const FpBackgroundCellInput bg_in = {
            .density_gcc = curr_raw->rho[itr],
            .temp_K = curr_raw->temp[itr],
            .bx_G = curr_raw->bx[itr],
            .by_G = curr_raw->by[itr],
            .bz_G = curr_raw->bz[itr],
            .divv_gyr = div_v_Norm * curr_raw->divv[itr] * kTracerGyr,
            .curl_v_s = rot_v_Norm * curr_raw->rotv[itr],
            .curl_v_prev_s = rot_v_Norm * prev_raw->rotv[itr],
            .curl_v_next_s = rot_v_Norm * next_raw->rotv[itr],
            .z_prev = z_prev,
            .z_curr = z_curr,
            .z_next = z_next,
            .l_turb_cm = curr_raw->lturb[itr],
            .target_l_turb_cm = L_target,
            .apply_temp_floor = apply_temp_floor,
            .allow_curl_interp = allow_curl_interp
        };
        FpBackgroundCellOutput bg_out;

        if (prepare_background_cell(&bg_in, &bg_out) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to derive background cell tracer=%d\n", itr);
            return -1;
        }
        tracer_store_background_history_slot((size_t)itr, &bg_out, &history);
    }
    return 0;
}

int load_tracer_initial_mass_hdf5(double *tracer_mass,
                                  int ntracer,
                                  long int tracer_start,
                                  const int *selected_offsets,
                                  int ntracer_global,
                                  const char *params_file,
                                  double z_init)
{
    char filename[MAX_LINE_LENGTH];
    double *mass_all = 0;
    int itr;
    int ierr = -1;

    if (tracer_mass == 0 || ntracer <= 0 || params_file == 0) return -1;

    if (read_param_file_noMPI(params_file) != SUCCESS) {
        fprintf(stderr, TRACER_FP_PROGNAME ": failed to read params file '%s' for tracer mass\n", params_file);
        return -1;
    }
    if (N_TRACERS <= 0 || nsnp_i < 0) return -1;
    if (ntracer_global > (int)N_TRACERS) return -1;

    if (!variable_tracer_mass) {
        for (itr = 0; itr < ntracer; itr++) tracer_mass[itr] = M_trc_fix;
        return 0;
    }

    tracer_build_snapshot_filename(nsnp_i, filename, sizeof(filename));
    if (selected_offsets == 0) {
        if (tracer_read_hdf5_dataset_slice(filename, DATASET_NAME_MASS,
                                           (hsize_t)N_TRACERS, (hsize_t)tracer_start,
                                           (hsize_t)ntracer, tracer_mass) == 0) {
            return 0;
        }
    } else {
        hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id != H5I_INVALID_HID) {
            if (tracer_read_hdf5_dataset_selected_open(file_id, DATASET_NAME_MASS,
                                                       (hsize_t)N_TRACERS, tracer_start,
                                                       selected_offsets, ntracer,
                                                       tracer_mass) == 0) {
                H5Fclose(file_id);
                return 0;
            }
            H5Fclose(file_id);
        }
    }

    mass_all = (double *)calloc((size_t)N_TRACERS, sizeof(double));
    if (mass_all == 0) return -1;
    if (read_param_hdf5(nsnp_i, (int)N_TRACERS, filename, DATASET_NAME_MASS, mass_all) != 0) {
        if (read_tracer_mass_hdf5_fallback(nsnp_i, (int)N_TRACERS, mass_all, z_init) <= 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": failed to read initial tracer mass from '%s'\n", filename);
            goto cleanup;
        }
    }

    if (selected_offsets == 0) {
        memcpy(tracer_mass, mass_all + tracer_start, (size_t)ntracer * sizeof(double));
    } else {
        for (itr = 0; itr < ntracer; itr++) {
            tracer_mass[itr] = mass_all[tracer_start + (long int)selected_offsets[itr]];
        }
    }
    ierr = 0;

cleanup:
    free(mass_all);
    return ierr;
}
