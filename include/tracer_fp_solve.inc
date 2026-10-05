    for (isnap = start_snapshot; isnap < nsnap; isnap++) {
        const double *n_gas_snap = bg_slots[bg_curr_slot].n_gas;
        const double *kbt_snap = bg_slots[bg_curr_slot].kbt;
        const double *b_field_snap = bg_slots[bg_curr_slot].b_field;
        const double *divv_snap = bg_slots[bg_curr_slot].divv;
        const double *lturb_snap = bg_slots[bg_curr_slot].l_turb;
        const double *dv_snap = bg_slots[bg_curr_slot].dv_imc;
        const double *cs_snap = bg_slots[bg_curr_slot].cs;
        const double *beta_snap = bg_slots[bg_curr_slot].beta_pl;
        const int has_next_snapshot = (isnap + 1 < nsnap);
        const int allow_snapshot_interp =
            (background_mode == TRACER_FP_BACKGROUND_EVOLVING) && has_next_snapshot;
        const double *n_gas_next_snap = allow_snapshot_interp ? bg_slots[bg_next_slot].n_gas : n_gas_snap;
        const double *kbt_next_snap = allow_snapshot_interp ? bg_slots[bg_next_slot].kbt : kbt_snap;
        const double *b_field_next_snap = allow_snapshot_interp ? bg_slots[bg_next_slot].b_field : b_field_snap;
        const double *divv_next_snap = allow_snapshot_interp ? bg_slots[bg_next_slot].divv : divv_snap;
        const double *lturb_next_snap = allow_snapshot_interp ? bg_slots[bg_next_slot].l_turb : lturb_snap;
        const double *dv_next_snap = allow_snapshot_interp ? bg_slots[bg_next_slot].dv_imc : dv_snap;
        const double *cs_next_snap = allow_snapshot_interp ? bg_slots[bg_next_slot].cs : cs_snap;
        const double *beta_next_snap = allow_snapshot_interp ? bg_slots[bg_next_slot].beta_pl : beta_snap;
        const TracerFpRawBackgroundSlot *shock_raw =
            (input_mode == TRACER_FP_INPUT_HDF5) ? &raw_slots[raw_prev_slot] : 0;
        FpSynchEmissionBatchInput synch_in;
        double *crp_radp_lo = 0, *crp_radp_hi = 0;
        double *cre_radp_lo = 0, *cre_radp_hi = 0;
        int nbuckets = 0;
        int snapshot_capped_count = 0;
        int snapshot_max_raw_nsub = 0;
        int ibucket;
        const int emit_emission_snapshot =
            (!bucket_stats_only && tracer_fp_should_emit_snapshot(schedule, isnap, nsnap));
        const int emit_cr_snapshot =
            (!bucket_stats_only && tracer_fp_should_emit_snapshot(schedule, isnap + 1, nsnap + 1));
        const int emit_output_snapshot = emit_emission_snapshot || emit_cr_snapshot;
        const int emit_synch_snapshot = emit_emission_snapshot;
        double rad_ic_row_next[npe];
        double rad_ic_m1_next_val = 0.0;
        double rad_ic_p1_next_val = 0.0;

        if (mpi_rank == 0) {
            const int physical_snapshot =
                (background_mode == TRACER_FP_BACKGROUND_FROZEN) ? nsnp_i : (nsnp_i + isnap);
            printf("  snapshot   : %d/%d (nsnp=%d, z=%.6f)\n",
                   isnap + 1, nsnap, physical_snapshot, z_snap[isnap]);
            fflush(stdout);
        }
        if (use_mapped_chunk && write_output_files && emit_output_snapshot &&
            mapped_chunk_count == 0 && mapped_chunk_cr_slots == 0) {
            mapped_chunk_start_snapshot = isnap;
            mapped_chunk_cr_base_slot = isnap + 1;
            tracer_fp_zero_rows(ne_chunk_core, ntracer,
                                      (size_t)(mapped_chunk_snapshots + 1) * (size_t)npe);
            tracer_fp_zero_rows(np_chunk_core, ntracer,
                                      (size_t)(mapped_chunk_snapshots + 1) * (size_t)np);
            tracer_fp_zero_rows(epssyn_chunk_core, ntracer,
                                      (size_t)mapped_chunk_snapshots * (size_t)nfreq);
            if (ic_enabled) {
                tracer_fp_zero_rows(epsic_chunk_core, ntracer,
                                          (size_t)mapped_chunk_snapshots * (size_t)bins_IC);
            }
            if (gamma_enabled) {
                tracer_fp_zero_rows(epsgamma_chunk_core, ntracer,
                                          (size_t)mapped_chunk_snapshots * (size_t)bins_gamma);
            }
            if (neutrino_enabled) {
                tracer_fp_zero_rows(epsnu_chunk_core, ntracer,
                                          (size_t)mapped_chunk_snapshots * (size_t)bins_nu);
            }
        }

        memset(&synch_in, 0, sizeof(synch_in));

        {
            const double t0_snapshot_prep = now_ms();

            if (dsa_injection_mode == TRACER_DSA_INJECTION_TRACER_SOURCE) {
                memset(qpi_batch, 0, np_batch * sizeof(double));
                memset(qepri_batch, 0, npe_batch * sizeof(double));
            }
            memset(dsa_disable_adiabatic_local, 0, (size_t)ntracer * sizeof(unsigned char));

            if (dsa_injection_mode == TRACER_DSA_INJECTION_TRACER_STATE ||
                dsa_injection_mode == TRACER_DSA_INJECTION_TRACER_SOURCE ||
                dsa_reacc_mode != TRACER_DSA_REACC_OFF) {
                int dsa_ninjected_snapshot = 0;
                const int density_snap_id =
                    (background_mode == TRACER_FP_BACKGROUND_FROZEN) ? nsnp_i : (nsnp_i + isnap);
                if (tracer_fp_apply_dsa_tracer_injection_snapshot(
                        shock_raw, ntracer, tracer_ids, tracer_mass,
                        (input_mode == TRACER_FP_INPUT_HDF5) ? hdf5_meta.c_dens[density_snap_id] : 1.0,
                        dt_snap[isnap], dsa_injection_mode, dsa_reacc_mode,
                        &crp_grid_local, &cre_grid_local, &dsa_grid_local,
                        crp_state, cre_state, qpi_batch, qepri_batch,
                        dsa_disable_adiabatic_local,
                        isnap + 1, mpi_rank, ctx->dsa_reacc_debug_fp,
                        &dsa_ninjected_snapshot) != 0) {
                    fprintf(stderr, TRACER_FP_PROGNAME ": DSA injection failed at snapshot %d\n",
                            isnap + 1);
                    goto cleanup;
                }
                total_dsa_injected += dsa_ninjected_snapshot;
            }

            if (tracer_fp_prepare_ic_cooling_snapshot(
                    ntracer,
                    allow_snapshot_interp,
                    z_snap[isnap],
                    allow_snapshot_interp ? z_snap[isnap + 1] : z_snap[isnap],
                    &cre_grid_local,
                    rad_ic_zero, rad_ic_m1_zero, rad_ic_p1_zero,
                    rad_ic_row_next, &rad_ic_m1_next_val, &rad_ic_p1_next_val) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": failed to prepare IC cooling rows at snapshot %d\n",
                        isnap + 1);
                goto cleanup;
            }

            if (emit_synch_snapshot || use_cuda_backend) {
                double snapshot_logb_min = 0.0;
                double snapshot_logb_max = 0.0;
                int snapshot_logb_out_of_range = 0;
                for (itr = 0; itr < ntracer; itr++) {
                    b_dyn[itr] = b_field_snap[itr];
                }
                tracer_fp_logb_range_from_snapshot(ntracer, b_dyn, logb,
                                                   &snapshot_logb_min, &snapshot_logb_max,
                                                   &snapshot_logb_out_of_range);
                if (log_root && snapshot_logb_out_of_range > 0) {
                    const int physical_snapshot =
                        (background_mode == TRACER_FP_BACKGROUND_FROZEN) ? nsnp_i : (nsnp_i + isnap);
                    fprintf(stderr,
                            TRACER_FP_PROGNAME ": warning synch logB range exceeded at snapshot %d (nsnp=%d): "
                            "%d/%d tracers outside [%.2f, %.2f], observed [%.3f, %.3f]\n",
                            isnap + 1, physical_snapshot,
                            snapshot_logb_out_of_range, ntracer,
                            kTracerSynchLogBHardMin, kTracerSynchLogBHardMax,
                            snapshot_logb_min, snapshot_logb_max);
                    fflush(stderr);
                }
                times_local.snapshot_prep_ms += now_ms() - t0_snapshot_prep;
                {
                    const double t0_synch_table = now_ms();
                    if (tracer_fp_refresh_synch_kernel_table(&synch_local, nfreq,
                                                             snapshot_logb_min, snapshot_logb_max,
                                                             &synch_local.kernel_table) != 0) {
                        fprintf(stderr, TRACER_FP_PROGNAME ": failed to build synch pitch-kernel table\n");
                        goto cleanup;
                    }
                    times_local.synch_table_ms += now_ms() - t0_synch_table;
                }
            } else {
                times_local.snapshot_prep_ms += now_ms() - t0_snapshot_prep;
            }
        }

        if (emit_synch_snapshot) {
            double t_synch = now_ms();
            double elapsed = 0.0;
            synch_in.ncell = ntracer;
            synch_in.nfreq = nfreq;
            synch_in.nx_tab = synch_local.nx_tab;
            synch_in.ntheta_pitch = synch_local.ntheta_pitch;
            synch_in.nlogb = TRACER_FP_SYNCH_LOGB_BINS;
            synch_in.z = z_snap[isnap];
            synch_in.xmin = synch_local.xmin;
            synch_in.logb_min = synch_local.table_logb_min;
            synch_in.inv_dlogb =
                (synch_local.table_logb_max > synch_local.table_logb_min)
                ? (double)(TRACER_FP_SYNCH_LOGB_BINS - 1) / (synch_local.table_logb_max - synch_local.table_logb_min)
                : 0.0;
            synch_in.fx_tab = synch_local.fx_tab;
            synch_in.logfx_tab = synch_local.logfx_tab;
            synch_in.logx_tab = synch_local.logx_tab;
            synch_in.logy = synch_local.logy_planes;
            synch_in.pitch_kernel_table = synch_local.kernel_table;
            synch_in.b_dyn = b_dyn;
            synch_in.logb = logb;
            synch_in.cre_batch = cre_state;
            synch_in.gamma2e = gamma2e;
            synch_in.theta = synch_local.theta;
            synch_in.dtheta = synch_local.dtheta;
            synch_in.pitch_weight = synch_local.pitch_weight;
            synch_in.nus = synch_local.nus;

            if (use_cuda_backend) {
#ifdef FP_USE_CUDA_BACKEND
                if (prepare_synch_emission_cuda_batch(&synch_in, &cre_grid_local, eps_syn, &elapsed) != 0) {
                    fprintf(stderr, TRACER_FP_PROGNAME ": CUDA synch emission failed at snapshot %d\n", isnap + 1);
                    goto cleanup;
                }
#endif
            } else {
                if (prepare_synch_emission_cell_major(&synch_in, &cre_grid_local, eps_syn) != 0) {
                    fprintf(stderr, TRACER_FP_PROGNAME ": CPU synch emission failed at snapshot %d\n", isnap + 1);
                    goto cleanup;
                }
            }
            emitted_synch++;
            last_synch_snapshot = isnap + 1;
            times_local.synch_ms += now_ms() - t_synch;
            if (write_output_files) {
                const double t0_emission_write = now_ms();
                if (use_tile_output) {
                    if (tracer_fp_tile_write_emit(&epssyn_tile_output_local, ntracer,
                                                  (size_t)isnap, nfreq, eps_syn,
                                                  raw_slots[raw_curr_slot].rho, tracer_mass,
                                                  tracer_output_per_cc) != 0) {
                        fprintf(stderr, TRACER_FP_PROGNAME ": failed to write tile synch emission at snapshot %d (%s)\n",
                                isnap + 1, strerror(errno));
                        goto cleanup;
                    }
                } else if (use_mapped_output) {
                    if (use_mapped_chunk) {
                        populate_emission_buffer(epssyn_chunk_core, ntracer,
                                                 isnap - mapped_chunk_start_snapshot,
                                                 nfreq, eps_syn,
                                                 raw_slots[raw_curr_slot].rho, tracer_mass,
                                                 tracer_output_per_cc);
                    } else {
                        if (tracer_fp_write_emit_map(&epssyn_output_local, ntracer,
                                                     isnap, nfreq, eps_syn,
                                                     raw_slots[raw_curr_slot].rho, tracer_mass,
                                                     tracer_output_per_cc) != 0) {
                            fprintf(stderr, TRACER_FP_PROGNAME ": failed to write mapped synch emission at snapshot %d (%s)\n",
                                    isnap + 1, strerror(errno));
                            goto cleanup;
                        }
                    }
                } else {
                    populate_emission_buffer(epssyn_buffer_core, ntracer, isnap, nfreq, eps_syn,
                                             raw_slots[raw_curr_slot].rho, tracer_mass,
                                             tracer_output_per_cc);
                }
                times_local.output_write_ms += now_ms() - t0_emission_write;
                times_local.output_write_calls++;
            }

            if (ic_enabled) {
                FpIcEmissionBatchInput ic_in;
                double t_ic = now_ms();
                memset(&ic_in, 0, sizeof(ic_in));
                prepare_ic_kernel_flat(&cre_grid_local, bins_IC, nu_min_ic, nu_max_ic,
                                       z_snap[isnap], fic_flat);
                ic_in.ncell = ntracer;
                ic_in.nbins = bins_IC;
                ic_in.z = z_snap[isnap];
                ic_in.nu_min = nu_min_ic;
                ic_in.nu_max = nu_max_ic;
                ic_in.cre_batch = cre_state;
                ic_in.fic_flat = fic_flat;

                if (prepare_ic_emission_cell_major(&ic_in, &cre_grid_local, eps_ic) != 0) {
                    fprintf(stderr, TRACER_FP_PROGNAME ": CPU IC emission failed at snapshot %d\n", isnap + 1);
                    goto cleanup;
                }
                times_local.ic_ms += now_ms() - t_ic;
                if (write_output_files) {
                    const double t0_emission_write = now_ms();
                    if (use_tile_output) {
                        if (tracer_fp_tile_write_emit(&epsic_tile_output_local, ntracer,
                                                      (size_t)isnap, bins_IC, eps_ic,
                                                      raw_slots[raw_curr_slot].rho, tracer_mass,
                                                      tracer_output_per_cc) != 0) {
                            fprintf(stderr, TRACER_FP_PROGNAME ": failed to write tile IC emission at snapshot %d (%s)\n",
                                    isnap + 1, strerror(errno));
                            goto cleanup;
                        }
                    } else if (use_mapped_output) {
                        if (use_mapped_chunk) {
                            populate_emission_buffer(epsic_chunk_core, ntracer,
                                                     isnap - mapped_chunk_start_snapshot,
                                                     bins_IC, eps_ic,
                                                     raw_slots[raw_curr_slot].rho, tracer_mass,
                                                     tracer_output_per_cc);
                        } else {
                            if (tracer_fp_write_emit_map(&epsic_output_local, ntracer,
                                                         isnap, bins_IC, eps_ic,
                                                         raw_slots[raw_curr_slot].rho, tracer_mass,
                                                         tracer_output_per_cc) != 0) {
                                fprintf(stderr, TRACER_FP_PROGNAME ": failed to write mapped IC emission at snapshot %d (%s)\n",
                                        isnap + 1, strerror(errno));
                                goto cleanup;
                            }
                        }
                    } else {
                        populate_emission_buffer(epsic_buffer_core, ntracer, isnap, bins_IC, eps_ic,
                                                 raw_slots[raw_curr_slot].rho, tracer_mass,
                                                 tracer_output_per_cc);
                    }
                    times_local.output_write_ms += now_ms() - t0_emission_write;
                    times_local.output_write_calls++;
                }
            }

            if (gamma_enabled) {
                FpGammaEmissionBatchInput gamma_in;
                double t_gamma = now_ms();
                memset(&gamma_in, 0, sizeof(gamma_in));
                gamma_in.ncell = ntracer;
                gamma_in.nbins = bins_gamma;
                gamma_in.z = z_snap[isnap];
                gamma_in.egamma_min = E_gamma_min;
                gamma_in.egamma_max = E_gamma_max;
                gamma_in.n_gas = n_gas_snap;
                gamma_in.crp_batch = crp_state;
                gamma_in.beta_p = beta_p;
                gamma_in.fga_flat = fga_flat;

                if (use_cuda_backend) {
#ifdef FP_USE_CUDA_BACKEND
                    if (prepare_gamma_emission_cuda_batch(&gamma_in, &crp_grid_local, eps_gamma, 0) != 0) {
                        fprintf(stderr, TRACER_FP_PROGNAME ": CUDA gamma emission failed at snapshot %d\n", isnap + 1);
                        goto cleanup;
                    }
#endif
                } else {
                    if (prepare_gamma_emission_cell_major(&gamma_in, &crp_grid_local, eps_gamma) != 0) {
                        fprintf(stderr, TRACER_FP_PROGNAME ": CPU gamma emission failed at snapshot %d\n", isnap + 1);
                        goto cleanup;
                    }
                }
                emitted_gamma++;
                times_local.gamma_ms += now_ms() - t_gamma;
                if (write_output_files) {
                    const double t0_emission_write = now_ms();
                    if (use_tile_output) {
                        if (tracer_fp_tile_write_emit(&epsgamma_tile_output_local, ntracer,
                                                      (size_t)isnap, bins_gamma, eps_gamma,
                                                      raw_slots[raw_curr_slot].rho, tracer_mass,
                                                      tracer_output_per_cc) != 0) {
                            fprintf(stderr, TRACER_FP_PROGNAME ": failed to write tile gamma emission at snapshot %d (%s)\n",
                                    isnap + 1, strerror(errno));
                            goto cleanup;
                        }
                    } else if (use_mapped_output) {
                        if (use_mapped_chunk) {
                            populate_emission_buffer(epsgamma_chunk_core, ntracer,
                                                     isnap - mapped_chunk_start_snapshot,
                                                     bins_gamma, eps_gamma,
                                                     raw_slots[raw_curr_slot].rho, tracer_mass,
                                                     tracer_output_per_cc);
                        } else {
                            if (tracer_fp_write_emit_map(&epsgamma_output_local, ntracer,
                                                         isnap, bins_gamma, eps_gamma,
                                                         raw_slots[raw_curr_slot].rho, tracer_mass,
                                                         tracer_output_per_cc) != 0) {
                                fprintf(stderr, TRACER_FP_PROGNAME ": failed to write mapped gamma emission at snapshot %d (%s)\n",
                                        isnap + 1, strerror(errno));
                                goto cleanup;
                            }
                        }
                    } else {
                        populate_emission_buffer(epsgamma_buffer_core, ntracer, isnap, bins_gamma, eps_gamma,
                                                 raw_slots[raw_curr_slot].rho, tracer_mass,
                                                 tracer_output_per_cc);
                    }
                    times_local.output_write_ms += now_ms() - t0_emission_write;
                    times_local.output_write_calls++;
                }
            }
            if (neutrino_enabled) {
                FpNeutrinoEmissionBatchInput nu_in;
                double t_nu = now_ms();
                memset(&nu_in, 0, sizeof(nu_in));
                nu_in.ncell = ntracer;
                nu_in.nbins = bins_nu;
                nu_in.z = z_snap[isnap];
                nu_in.enu_min = E_nu_min;
                nu_in.enu_max = E_nu_max;
                nu_in.n_gas = n_gas_snap;
                nu_in.crp_batch = crp_state;
                nu_in.beta_p = beta_p;
                nu_in.fnu_flat = fnu_flat;

                if (use_cuda_backend) {
#ifdef FP_USE_CUDA_BACKEND
                    if (prepare_neutrino_emission_cuda_batch(&nu_in, &crp_grid_local, eps_nu, 0) != 0) {
                        fprintf(stderr, TRACER_FP_PROGNAME ": CUDA neutrino emission failed at snapshot %d\n", isnap + 1);
                        goto cleanup;
                    }
#endif
                } else {
                    if (prepare_neutrino_emission_cell_major(&nu_in, &crp_grid_local, eps_nu) != 0) {
                        fprintf(stderr, TRACER_FP_PROGNAME ": CPU neutrino emission failed at snapshot %d\n", isnap + 1);
                        goto cleanup;
                    }
                }
                emitted_neutrino++;
                times_local.neutrino_ms += now_ms() - t_nu;
                if (write_output_files) {
                    const double t0_emission_write = now_ms();
                    if (use_tile_output) {
                        if (tracer_fp_tile_write_emit(&epsnu_tile_output_local, ntracer,
                                                      (size_t)isnap, bins_nu, eps_nu,
                                                      raw_slots[raw_curr_slot].rho, tracer_mass,
                                                      tracer_output_per_cc) != 0) {
                            fprintf(stderr, TRACER_FP_PROGNAME ": failed to write tile neutrino emission at snapshot %d (%s)\n",
                                    isnap + 1, strerror(errno));
                            goto cleanup;
                        }
                    } else if (use_mapped_output) {
                        if (use_mapped_chunk) {
                            populate_emission_buffer(epsnu_chunk_core, ntracer,
                                                     isnap - mapped_chunk_start_snapshot,
                                                     bins_nu, eps_nu,
                                                     raw_slots[raw_curr_slot].rho, tracer_mass,
                                                     tracer_output_per_cc);
                        } else {
                            if (tracer_fp_write_emit_map(&epsnu_output_local, ntracer,
                                                         isnap, bins_nu, eps_nu,
                                                         raw_slots[raw_curr_slot].rho, tracer_mass,
                                                         tracer_output_per_cc) != 0) {
                                fprintf(stderr, TRACER_FP_PROGNAME ": failed to write mapped neutrino emission at snapshot %d (%s)\n",
                                        isnap + 1, strerror(errno));
                                goto cleanup;
                            }
                        }
                    } else {
                        populate_emission_buffer(epsnu_buffer_core, ntracer, isnap, bins_nu, eps_nu,
                                                 raw_slots[raw_curr_slot].rho, tracer_mass,
                                                 tracer_output_per_cc);
                    }
                    times_local.output_write_ms += now_ms() - t0_emission_write;
                    times_local.output_write_calls++;
                }
            }
        }

        if (tracer_fp_snapshot_ws_ensure(&cpu_workspace_local.snapshot, ntracer) != 0) {
            fprintf(stderr, TRACER_FP_PROGNAME ": loss allocation failure at snapshot %d\n", isnap + 1);
            goto cleanup;
        }
        /* Reuse the four ncell-sized snapshot buffers as endpoint-only loss storage
         * for the N_subloop estimator. */
        crp_radp_lo = cpu_workspace_local.snapshot.loss.crp_radpm1;
        crp_radp_hi = cpu_workspace_local.snapshot.loss.crp_radpp1;
        cre_radp_lo = cpu_workspace_local.snapshot.loss.cre_radpm1;
        cre_radp_hi = cpu_workspace_local.snapshot.loss.cre_radpp1;

        {
            const double t0_loss_prepass = now_ms();
            if (tracer_fp_prepare_nsub_loss_endpoints(
                    ntracer, &crp_grid_local, &cre_grid_local,
                    n_gas_snap, kbt_snap, b_field_snap, divv_snap,
                    rad_ic_zero,
                    crp_radp_lo, crp_radp_hi, cre_radp_lo, cre_radp_hi) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": loss prepass failed at snapshot %d\n", isnap + 1);
                goto cleanup;
            }
            times_local.loss_prepass_ms += now_ms() - t0_loss_prepass;
        }

        {
            const double t0_nsub_estimate = now_ms();
            if (tracer_fp_compute_nsubsteps_from_endpoints(
                    ntracer, dt_snap[isnap], &crp_grid_local, &cre_grid_local,
                    crp_radp_lo, crp_radp_hi, cre_radp_lo, cre_radp_hi,
                    lturb_snap, dv_snap,
                    step_local.nsubsteps, step_local.target_nsubsteps, step_local.n_onsteps, step_local.capped_flags,
                    &max_nsubsteps, &snapshot_capped_count,
                    &snapshot_max_raw_nsub) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": N_subloop estimate failed at snapshot %d\n", isnap + 1);
                goto cleanup;
            }
            times_local.nsub_estimate_ms += now_ms() - t0_nsub_estimate;
        }
        {
            long long runtime_snapshot_sum = 0;
            long long runtime_snapshot_target_sum = 0;
            for (itr = 0; itr < ntracer; itr++) {
                runtime_snapshot_sum += (long long)step_local.nsubsteps[itr];
                runtime_snapshot_target_sum += (long long)step_local.target_nsubsteps[itr];
            }
            runtime_snapshot_nsub_local[isnap] = runtime_snapshot_sum;
            runtime_snapshot_target_nsub_local[isnap] = runtime_snapshot_target_sum;
        }
        total_capped_tracer_snapshots += (long long)snapshot_capped_count;
        if (snapshot_max_raw_nsub > max_raw_nsubsteps) {
            max_raw_nsubsteps = snapshot_max_raw_nsub;
        }
        if (mpi_rank == 0 && tracer_nsub_max > 0 && snapshot_capped_count > 0) {
            printf("  warning nsub cap    : snapshot %d/%d capped %d tracers at nsub_max=%d (raw max=%d)\n",
                   isnap + 1, nsnap, snapshot_capped_count, tracer_nsub_max, snapshot_max_raw_nsub);
            fflush(stdout);
        }
        if (bucket_stats_only) {
            const int physical_snapshot =
                (background_mode == TRACER_FP_BACKGROUND_FROZEN) ? nsnp_i : (nsnp_i + isnap);
            int world_size_local = 1;
            const double t0_bucketstats_top = now_ms();
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_local);
            if (tracer_fp_bucket_top_write(bucketstats_top_fp,
                                           mpi_rank, world_size_local, ntracer_global,
                                           isnap + 1, physical_snapshot, z_snap[isnap],
                                           ntracer, tracer_ids,
                                           step_local.nsubsteps, step_local.target_nsubsteps,
                                           n_gas_snap, kbt_snap, b_field_snap,
                                           divv_snap, lturb_snap, dv_snap, beta_snap) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": failed to write bucketstats top tracers at snapshot %d\n",
                        isnap + 1);
                goto cleanup;
            }
            times_local.bucketstats_top_ms += now_ms() - t0_bucketstats_top;
        }
        if (bucket_stats_only) {
            int local_max_nsub = 0;
            int local_max_itr = -1;
            struct {
                int value;
                int rank;
            } local_maxloc, global_maxloc;
            double max_payload[10];
            int payload_count = 10;

            for (itr = 0; itr < ntracer; itr++) {
                if (step_local.nsubsteps[itr] > local_max_nsub) {
                    local_max_nsub = step_local.nsubsteps[itr];
                    local_max_itr = itr;
                }
            }

            local_maxloc.value = local_max_nsub;
            local_maxloc.rank = mpi_rank;
            MPI_Allreduce(&local_maxloc, &global_maxloc, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

            if (local_max_itr >= 0 && mpi_rank == global_maxloc.rank) {
                max_payload[0] = (double)tracer_ids[local_max_itr];
                max_payload[1] = (double)step_local.nsubsteps[local_max_itr];
                max_payload[2] = (double)step_local.target_nsubsteps[local_max_itr];
                max_payload[3] = n_gas_snap[local_max_itr];
                max_payload[4] = kbt_snap[local_max_itr];
                max_payload[5] = b_field_snap[local_max_itr];
                max_payload[6] = divv_snap[local_max_itr];
                max_payload[7] = lturb_snap[local_max_itr];
                max_payload[8] = dv_snap[local_max_itr];
                max_payload[9] = beta_snap[local_max_itr];
            }
            MPI_Bcast(max_payload, payload_count, MPI_DOUBLE, global_maxloc.rank, MPI_COMM_WORLD);
            if (mpi_rank == 0 && global_maxloc.value > 0) {
                fprintf(stderr,
                        "  max-nsub tracer     : snapshot %d/%d id=%.0f nsub=%.0f target=%.0f n_gas=%e kbt=%e B=%e divv=%e lturb=%e dv=%e beta=%e owner_rank=%d\n",
                        isnap + 1, nsnap,
                        max_payload[0], max_payload[1], max_payload[2],
                        max_payload[3], max_payload[4], max_payload[5], max_payload[6],
                        max_payload[7], max_payload[8], max_payload[9], global_maxloc.rank);
                fflush(stderr);
            }
        }

        if (log_root && debug_target_local >= 0) {
            fprintf(stderr,
                    "[debug tracer %ld] snapshot=%d dt=%e nsub=%d non=%d "
                    "n_gas=%e kbt=%e B=%e divv=%e lturb=%e dv=%e cs=%e beta=%e\n",
                    debug_target_global, isnap + 1, dt_snap[isnap],
                    step_local.nsubsteps[debug_target_local], step_local.n_onsteps[debug_target_local],
                    n_gas_snap[debug_target_local], kbt_snap[debug_target_local],
                    b_field_snap[debug_target_local], divv_snap[debug_target_local],
                    lturb_snap[debug_target_local], dv_snap[debug_target_local],
                    cs_snap[debug_target_local], beta_snap[debug_target_local]);
        }

        /* The CPU thread-local path still accepts the old multirate shape.
         * Keep the compatibility arrays explicit while tracer_fp uses
         * quantized integration only. */
        for (itr = 0; itr < ntracer; itr++) {
            step_local.base_nsteps[itr] = step_local.nsubsteps[itr];
            step_local.fp_cadence_steps[itr] = 1;
        }

        {
            const double t0_bucket_build = now_ms();
            if (tracer_bucket_build_snapshot(&step_local, ntracer, &bucket_local, &nbuckets) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": bucket build failed at snapshot %d\n",
                        isnap + 1);
                goto cleanup;
            }
            times_local.bucket_build_ms += now_ms() - t0_bucket_build;
        }
        if (nbuckets > max_bucket_count_per_snapshot) {
            max_bucket_count_per_snapshot = nbuckets;
        }
        if (use_cuda_backend && enable_cuda_bucket_group_path) {
            const double t0_gpu_group_build = now_ms();
            if (tracer_gpu_bucket_groups_build(&bucket_local, nbuckets,
                                               allow_snapshot_interp,
                                               &gpu_bucket_groups_local) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": GPU bucket group build failed at snapshot %d\n",
                        isnap + 1);
                goto cleanup;
            }
            times_local.gpu_group_build_ms += now_ms() - t0_gpu_group_build;
            total_gpu_group_count_est += (long long)gpu_bucket_groups_local.ngroups;
            if (gpu_bucket_groups_local.ngroups > max_gpu_groups_per_snapshot) {
                max_gpu_groups_per_snapshot = gpu_bucket_groups_local.ngroups;
            }
        }
#ifdef FP_USE_CUDA_BACKEND
        if (use_cuda_rank_state_path && !bucket_stats_only) {
            double rank_h2d_ms = 0.0;
            if (cuda_rank_state_upload(cuda_workspace_local, ntracer,
                                       qpi_batch, qepri_batch,
                                       crp_state, cre_state,
                                       &rank_h2d_ms) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": CUDA rank state upload failed at snapshot %d\n",
                        isnap + 1);
                goto cleanup;
            }
            times_local.cuda_h2d_ms += rank_h2d_ms;
            times_local.cuda_total_ms += rank_h2d_ms;
        }
#endif
        if (bucket_stats_only && mpi_rank == 0) {
            printf("  bucket snapshot     : %d/%d (physical=%d, z=%.6f) buckets=%d\n",
                   isnap + 1, nsnap,
                   (background_mode == TRACER_FP_BACKGROUND_FROZEN) ? nsnp_i : (nsnp_i + isnap),
                   z_snap[isnap], nbuckets);
        }

        if (!bucket_stats_only && !use_cuda_backend &&
            tracer_fp_cpu_threads() > 1) {
            int snapshot_bucket_calls = 0;
            int snapshot_min_bucket_size = 0;
            int snapshot_max_bucket_size = 0;

            if (log_root && debug_target_local >= 0) {
                fprintf(stderr,
                        "[debug tracer %ld] snapshot=%d cpu-thread-local-buckets tracers=%d threads=%d\n",
                        debug_target_global, isnap + 1, ntracer,
                        tracer_fp_cpu_threads());
            }

            if (ntracer > 0 &&
                tracer_fp_batch_execute_parallel(
                    ntracer,
                    step_local.nsubsteps,
                    step_local.n_onsteps,
                    step_local.base_nsteps,
                    step_local.fp_cadence_steps,
                    TRACER_FP_INTEGRATION_QUANTIZED,
                    allow_snapshot_interp,
                    elapsed_gyr,
                    dt_snap[isnap],
                    z_snap[isnap],
                    allow_snapshot_interp ? z_snap[isnap + 1] : z_snap[isnap],
                    requested_t_acc_direct_gyr,
                    &crp_grid_local, &cre_grid_local,
                    n_gas_snap, kbt_snap, b_field_snap, divv_snap,
                    lturb_snap, dv_snap, cs_snap, beta_snap,
                    rad_ic_zero, rad_ic_m1_zero, rad_ic_p1_zero,
                    n_gas_next_snap, kbt_next_snap, b_field_next_snap, divv_next_snap,
                    lturb_next_snap, dv_next_snap, cs_next_snap, beta_next_snap,
                    rad_ic_row_next, rad_ic_m1_next_val, rad_ic_p1_next_val,
                    qpi_batch, qepri_batch, fqe_flat, np_min_qe,
                    tracer_mass,
                    dsa_disable_adiabatic_local,
                    crp_state, cre_state, &times_local,
                    &snapshot_bucket_calls,
                    &snapshot_min_bucket_size,
                    &snapshot_max_bucket_size) != 0) {
                fprintf(stderr,
                        TRACER_FP_PROGNAME ": CPU thread-local bucket evolve failed at snapshot %d\n",
                        isnap + 1);
                goto cleanup;
            }

            total_bucket_calls += snapshot_bucket_calls;
            total_bucket_cells += (long long)ntracer;
            if (snapshot_min_bucket_size > 0 &&
                (min_bucket_size == 0 || snapshot_min_bucket_size < min_bucket_size)) {
                min_bucket_size = snapshot_min_bucket_size;
            }
            if (snapshot_max_bucket_size > max_bucket_size_seen) {
                max_bucket_size_seen = snapshot_max_bucket_size;
            }
        } else {
            const int nbucket_exec = nbuckets;

            for (ibucket = 0; ibucket < nbucket_exec; ibucket++) {
                const int bucket_id = ibucket;
                const int bucket_offset = bucket_local.member_offsets[bucket_id];
                const int bucket_nstep = bucket_local.steps[bucket_id];
                const int bucket_n_on = bucket_local.onsteps[bucket_id];
                const int bucket_count = bucket_local.member_counts[bucket_id];
                const int *bucket_members = bucket_local.indices + (size_t)bucket_offset;
                const int pipeline_nstep = bucket_nstep;
                const int bucket_on_start = (bucket_nstep - bucket_n_on) / 2;
                const int bucket_on_end = bucket_on_start + bucket_n_on;
                const int coeff_interp_segments =
                    tracer_fp_choose_coeff_interp_segments(bucket_nstep, allow_snapshot_interp);
                int debug_target_idx_in_bucket = -1;
                int idx;
                int iseg;
                double t0_bucket_host = 0.0;
                double pack_before_ms = 0.0;
                double interp_before_ms = 0.0;
                double cuda_total_before_ms = 0.0;
                double *n_gas_bucket = 0, *kbt_bucket = 0, *b_field_bucket = 0, *divv_bucket = 0;
                double *lturb_bucket = 0, *dv_bucket = 0, *cs_bucket = 0, *beta_bucket = 0;
                double *rad_ic_bucket = 0, *rad_ic_m1_bucket = 0, *rad_ic_p1_bucket = 0;
                double *n_gas_next_bucket = 0, *kbt_next_bucket = 0, *b_field_next_bucket = 0, *divv_next_bucket = 0;
                double *lturb_next_bucket = 0, *dv_next_bucket = 0, *cs_next_bucket = 0, *beta_next_bucket = 0;
                double *rad_ic_next_bucket = 0, *rad_ic_m1_next_bucket = 0, *rad_ic_p1_next_bucket = 0;
                double *n_gas_coeff = 0, *kbt_coeff = 0, *b_field_coeff = 0, *divv_coeff = 0;
                double *lturb_coeff = 0, *dv_coeff = 0, *cs_coeff = 0, *beta_coeff = 0;
                double *rad_ic_coeff = 0, *rad_ic_m1_coeff = 0, *rad_ic_p1_coeff = 0;
                double *qpi_bucket = 0, *qepri_bucket = 0;
                double *crp_bucket = 0, *cre_bucket = 0;

                for (idx = 0; idx < bucket_count; idx++) {
                    if (bucket_members[idx] == debug_target_local) {
                        debug_target_idx_in_bucket = idx;
                        break;
                    }
                }
                if (bucket_stats_only) {
                    int target_min = 0, target_max = 0;
                    long long target_sum = 0;
                    double inflate_sum = 0.0;
                    double inflate_max = 0.0;
                    for (idx = 0; idx < bucket_count; idx++) {
                        const int src = bucket_members[idx];
                        const int target_nsub = (step_local.target_nsubsteps != 0 && step_local.target_nsubsteps[src] > 0)
                            ? step_local.target_nsubsteps[src] : bucket_nstep;
                        const double inflate = (target_nsub > 0)
                            ? (double)bucket_nstep / (double)target_nsub : 1.0;
                        if (idx == 0 || target_nsub < target_min) target_min = target_nsub;
                        if (idx == 0 || target_nsub > target_max) target_max = target_nsub;
                        target_sum += (long long)target_nsub;
                        inflate_sum += inflate;
                        if (inflate > inflate_max) inflate_max = inflate;
                    }
                    record_bucket_metrics(bucket_count, &total_bucket_calls,
                                                    &total_bucket_cells,
                                                    &min_bucket_size,
                                                    &max_bucket_size_seen,
                                                    bucket_hist_counts);
                    if (mpi_rank == 0) {
                        printf("    bucket %3d/%3d : size=%d nsub=%d non=%d coeff_segments=%d target[min,max,avg]=%d,%d,%.1f inflate[avg,max]=%.3f,%.3f\n",
                               ibucket + 1, nbuckets, bucket_count,
                               bucket_nstep, bucket_n_on, coeff_interp_segments,
                               target_min, target_max,
                               (double)target_sum / (double)bucket_count,
                               inflate_sum / (double)bucket_count, inflate_max);
                    }
                    if (bucketstats_rank_fp != 0) {
                        fprintf(bucketstats_rank_fp,
                                "%d\t%d\t%d\t%.8e\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\n",
                                mpi_rank,
                                isnap + 1,
                                (background_mode == TRACER_FP_BACKGROUND_FROZEN) ? nsnp_i : (nsnp_i + isnap),
                                z_snap[isnap],
                                ibucket + 1,
                                nbuckets,
                                bucket_count,
                                bucket_nstep,
                                bucket_n_on,
                                coeff_interp_segments,
                                target_min,
                                target_max,
                                (double)target_sum / (double)bucket_count,
                                inflate_sum / (double)bucket_count,
                                inflate_max);
                    }
                    continue;
                }

#ifdef FP_USE_CUDA_BACKEND
                if (use_cuda_backend && enable_cuda_bucket_group_path) {
                    const int group_id = gpu_bucket_groups_local.bucket_group_ids[bucket_id];
                    const int group_begin = gpu_bucket_groups_local.group_offsets[group_id];
                    const int group_end = gpu_bucket_groups_local.group_offsets[group_id + 1];
                    const int group_bucket_count = group_end - group_begin;
                    const int group_coeff_segments = gpu_bucket_groups_local.group_coeff_segments[group_id];
                    const int *group_bucket_ids = gpu_bucket_groups_local.group_bucket_ids + group_begin;
                    int use_segment_group_path =
                        (ibucket == group_begin &&
                         debug_target_local < 0);

                    if (use_segment_group_path) {
                        int ilocal;

                        for (ilocal = 0; ilocal < group_bucket_count; ilocal++) {
                            const int group_bucket_id = group_bucket_ids[ilocal];
                            if (bucket_local.member_counts[group_bucket_id] <= 0 ||
                                tracer_fp_choose_coeff_interp_segments(
                                    bucket_local.steps[group_bucket_id],
                                    allow_snapshot_interp) != group_coeff_segments) {
                                use_segment_group_path = 0;
                                break;
                            }
                        }

                        if (use_segment_group_path) {
                            FpCudaOutputSchedule none_schedule;
                            FpCudaPipelineTimes cuda_times;

                            if (cuda_group_scratch_ensure(
                                    &gpu_group_inputs_local,
                                    &gpu_group_coeff_inputs_local,
                                    &gpu_group_cell_offsets_local,
                                    &gpu_group_rad_ic_offsets_local,
                                    group_bucket_count,
                                    &gpu_group_launch_capacity) != 0) {
                                goto cleanup;
                            }

                            t0_bucket_host = now_ms();
                            pack_before_ms = times_local.pack_ms;
                            interp_before_ms = times_local.interp_ms;
                            cuda_total_before_ms = times_local.cuda_total_ms;
                            if (tracer_fp_batch_pack_group_current(
                                    group_bucket_count, group_bucket_ids,
                                    &bucket_local, ntracer,
                                    n_gas_snap, kbt_snap, b_field_snap, divv_snap,
                                    lturb_snap, dv_snap, cs_snap, beta_snap,
                                    rad_ic_zero, rad_ic_m1_zero, rad_ic_p1_zero,
                                    (group_coeff_segments > 1) ? n_gas_next_snap : 0,
                                    (group_coeff_segments > 1) ? kbt_next_snap : 0,
                                    (group_coeff_segments > 1) ? b_field_next_snap : 0,
                                    (group_coeff_segments > 1) ? divv_next_snap : 0,
                                    (group_coeff_segments > 1) ? lturb_next_snap : 0,
                                    (group_coeff_segments > 1) ? dv_next_snap : 0,
                                    (group_coeff_segments > 1) ? cs_next_snap : 0,
                                    (group_coeff_segments > 1) ? beta_next_snap : 0,
                                    (group_coeff_segments > 1) ? rad_ic_row_next : 0,
                                    rad_ic_m1_next_val, rad_ic_p1_next_val,
                                    qpi_batch, qepri_batch,
                                    tracer_mass,
                                    dsa_disable_adiabatic_local,
                                    crp_state, cre_state,
                                    0, 0,
                                    gpu_group_cell_offsets_local,
                                    gpu_group_rad_ic_offsets_local,
                                    &cpu_workspace_local, 0,
                                    &times_local) != 0) {
                                goto cleanup;
                            }

                            memset(&none_schedule, 0, sizeof(none_schedule));
                            {
                                for (iseg = 0; iseg < group_coeff_segments; iseg++) {
                                    const double alpha = (group_coeff_segments > 1)
                                        ? ((double)iseg + 0.5) / (double)group_coeff_segments : 0.0;
                                    const double z_coeff = allow_snapshot_interp
                                        ? z_snap[isnap] + alpha * (z_snap[isnap + 1] - z_snap[isnap])
                                        : z_snap[isnap];
                                    int active_group_inputs = 0;
                                    int active_group_cells = 0;
                                    long long group_fp_steps_sum = 0;
                                    long long group_cell_steps_sum = 0;
                                    double t0_interp_group = 0.0;

                                    memset(&cuda_times, 0, sizeof(cuda_times));
                                    if (group_coeff_segments > 1) {
                                        t0_interp_group = now_ms();
                                    }

                                    for (ilocal = 0; ilocal < group_bucket_count; ilocal++) {
                                        const int group_bucket_id = group_bucket_ids[ilocal];
                                        const int group_bucket_cells = bucket_local.member_counts[group_bucket_id];
                                        const int group_bucket_nstep = bucket_local.steps[group_bucket_id];
                                        const int group_bucket_n_on = bucket_local.onsteps[group_bucket_id];
                                        const int cell_offset = gpu_group_cell_offsets_local[ilocal];
                                        const int rad_ic_offset = gpu_group_rad_ic_offsets_local[ilocal];
                                        const int seg_fp_begin =
                                            (int)(((long long)iseg * (long long)group_bucket_nstep) / (long long)group_coeff_segments);
                                        const int seg_fp_end =
                                            (int)(((long long)(iseg + 1) * (long long)group_bucket_nstep) / (long long)group_coeff_segments);
                                        const int seg_fp_steps = seg_fp_end - seg_fp_begin;
                                        const int seg_on_begin =
                                            (((group_bucket_nstep - group_bucket_n_on) / 2) > seg_fp_begin)
                                            ? ((group_bucket_nstep - group_bucket_n_on) / 2) : seg_fp_begin;
                                        int seg_on_end =
                                            ((((group_bucket_nstep - group_bucket_n_on) / 2) + group_bucket_n_on) < seg_fp_end)
                                            ? (((group_bucket_nstep - group_bucket_n_on) / 2) + group_bucket_n_on) : seg_fp_end;
                                        int seg_use_windowed_reacc;
                                        int seg_on_start_local;
                                        int seg_on_end_local;
                                        const double *n_gas_coeff_ptr = cpu_workspace_local.bg_curr.n_gas + cell_offset;
                                        const double *kbt_coeff_ptr = cpu_workspace_local.bg_curr.kbt + cell_offset;
                                        const double *b_field_coeff_ptr = cpu_workspace_local.bg_curr.b_field + cell_offset;
                                        const double *divv_coeff_ptr = cpu_workspace_local.bg_curr.divv + cell_offset;
                                        const double *lturb_coeff_ptr = cpu_workspace_local.bg_curr.lturb + cell_offset;
                                        const double *dv_coeff_ptr = cpu_workspace_local.bg_curr.dv + cell_offset;
                                        const double *cs_coeff_ptr = cpu_workspace_local.bg_curr.cs + cell_offset;
                                        const double *beta_coeff_ptr = cpu_workspace_local.bg_curr.beta + cell_offset;
                                        const double *rad_ic_coeff_ptr = cpu_workspace_local.bg_curr.rad_ic + rad_ic_offset;
                                        const double *rad_ic_m1_coeff_ptr = cpu_workspace_local.bg_curr.rad_ic_m1 + cell_offset;
                                        const double *rad_ic_p1_coeff_ptr = cpu_workspace_local.bg_curr.rad_ic_p1 + cell_offset;
                                        FpCoeffBatchInput *coeff_in;
                                        FpCudaPipelineInput *cuda_in;

                                        if (seg_fp_steps <= 0) continue;
                                        if (seg_on_end < seg_on_begin) seg_on_end = seg_on_begin;
                                        seg_use_windowed_reacc = ((seg_on_end - seg_on_begin) < seg_fp_steps) ? 1 : 0;
                                        seg_on_start_local = seg_on_begin - seg_fp_begin;
                                        seg_on_end_local = seg_on_start_local + (seg_on_end - seg_on_begin);

                                        if (group_coeff_segments > 1) {
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.n_gas + cell_offset,
                                                                   cpu_workspace_local.bg_next.n_gas + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.n_gas + cell_offset);
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.kbt + cell_offset,
                                                                   cpu_workspace_local.bg_next.kbt + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.kbt + cell_offset);
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.b_field + cell_offset,
                                                                   cpu_workspace_local.bg_next.b_field + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.b_field + cell_offset);
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.divv + cell_offset,
                                                                   cpu_workspace_local.bg_next.divv + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.divv + cell_offset);
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.lturb + cell_offset,
                                                                   cpu_workspace_local.bg_next.lturb + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.lturb + cell_offset);
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.dv + cell_offset,
                                                                   cpu_workspace_local.bg_next.dv + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.dv + cell_offset);
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.cs + cell_offset,
                                                                   cpu_workspace_local.bg_next.cs + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.cs + cell_offset);
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.beta + cell_offset,
                                                                   cpu_workspace_local.bg_next.beta + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.beta + cell_offset);
                                            interp_array(group_bucket_cells * npe,
                                                                   cpu_workspace_local.bg_curr.rad_ic + rad_ic_offset,
                                                                   cpu_workspace_local.bg_next.rad_ic + rad_ic_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.rad_ic + rad_ic_offset);
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.rad_ic_m1 + cell_offset,
                                                                   cpu_workspace_local.bg_next.rad_ic_m1 + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.rad_ic_m1 + cell_offset);
                                            interp_array(group_bucket_cells,
                                                                   cpu_workspace_local.bg_curr.rad_ic_p1 + cell_offset,
                                                                   cpu_workspace_local.bg_next.rad_ic_p1 + cell_offset,
                                                                   alpha,
                                                                   cpu_workspace_local.bg_interp.rad_ic_p1 + cell_offset);
                                            n_gas_coeff_ptr = cpu_workspace_local.bg_interp.n_gas + cell_offset;
                                            kbt_coeff_ptr = cpu_workspace_local.bg_interp.kbt + cell_offset;
                                            b_field_coeff_ptr = cpu_workspace_local.bg_interp.b_field + cell_offset;
                                            divv_coeff_ptr = cpu_workspace_local.bg_interp.divv + cell_offset;
                                            lturb_coeff_ptr = cpu_workspace_local.bg_interp.lturb + cell_offset;
                                            dv_coeff_ptr = cpu_workspace_local.bg_interp.dv + cell_offset;
                                            cs_coeff_ptr = cpu_workspace_local.bg_interp.cs + cell_offset;
                                            beta_coeff_ptr = cpu_workspace_local.bg_interp.beta + cell_offset;
                                            rad_ic_coeff_ptr = cpu_workspace_local.bg_interp.rad_ic + rad_ic_offset;
                                            rad_ic_m1_coeff_ptr = cpu_workspace_local.bg_interp.rad_ic_m1 + cell_offset;
                                            rad_ic_p1_coeff_ptr = cpu_workspace_local.bg_interp.rad_ic_p1 + cell_offset;
                                        }

                                        coeff_in = gpu_group_coeff_inputs_local + active_group_inputs;
                                        cuda_in = gpu_group_inputs_local + active_group_inputs;

                                        memset(coeff_in, 0, sizeof(*coeff_in));
                                        memset(cuda_in, 0, sizeof(*cuda_in));

                                        coeff_in->ncell = group_bucket_cells;
                                        coeff_in->z = z_coeff;
                                        coeff_in->dt_gyr = dt_snap[isnap] / (double)group_bucket_nstep;
                                        coeff_in->epmax = 1.0e7;
                                        coeff_in->n_gas = n_gas_coeff_ptr;
                                        coeff_in->kbt = kbt_coeff_ptr;
                                        coeff_in->b_field = b_field_coeff_ptr;
                                        coeff_in->divv_gyr = divv_coeff_ptr;
                                        coeff_in->l_turb_mpc = lturb_coeff_ptr;
                                        coeff_in->dv_imc = dv_coeff_ptr;
                                        coeff_in->cs = cs_coeff_ptr;
                                        coeff_in->beta_pl = beta_coeff_ptr;
                                        coeff_in->tracer_mass_msun =
                                            cpu_workspace_local.state.mass_msun + cell_offset;
                                        coeff_in->disable_adiabatic =
                                            cpu_workspace_local.state.disable_adiabatic + cell_offset;
                                        coeff_in->crp_state_cell_major =
                                            cpu_workspace_local.state.crp + (size_t)cell_offset * (size_t)np;
                                        coeff_in->cre_state_cell_major =
                                            cpu_workspace_local.state.cre + (size_t)cell_offset * (size_t)npe;
                                        coeff_in->rad_ic_batch = rad_ic_coeff_ptr;
                                        coeff_in->rad_ic_m1 = rad_ic_m1_coeff_ptr;
                                        coeff_in->rad_ic_p1 = rad_ic_p1_coeff_ptr;

                                        cuda_in->ncell = group_bucket_cells;
                                        cuda_in->nstep = seg_fp_steps;
                                        cuda_in->fp_cadence_steps = 1;
                                        cuda_in->diff_cadence_steps = 1;
                                        cuda_in->adv_cadence_steps = 1;
                                        cuda_in->use_windowed_reacc = seg_use_windowed_reacc;
                                        cuda_in->on_start_step = seg_on_start_local;
                                        cuda_in->on_end_step = seg_on_end_local;
                                        cuda_in->nfreq = nfreq;
                                        cuda_in->nx_tab = synch_local.nx_tab;
                                        cuda_in->dt = dt_snap[isnap] / (double)group_bucket_nstep;
                                        cuda_in->epmax = 1.0e7;
                                        cuda_in->xmin = synch_local.xmin;
                                        cuda_in->psi_value = psi;
                                        cuda_in->mach_limit_value = mach_limit;
                                        cuda_in->transport_mode = cuda_transport_mode();
                                        cuda_in->crp_grid = &crp_grid_local;
                                        cuda_in->cre_grid = &cre_grid_local;
                                        cuda_in->coeff_in = coeff_in;
                                        cuda_in->synch_in = 0;
                                        cuda_in->qpi_batch =
                                            cpu_workspace_local.state.qpi + (size_t)cell_offset * (size_t)np;
                                        cuda_in->qepri_batch =
                                            cpu_workspace_local.state.qepri + (size_t)cell_offset * (size_t)npe;
                                        cuda_in->beta_p = beta_p;
                                        cuda_in->crp_init = cpu_workspace_local.state.crp + (size_t)cell_offset * (size_t)np;
                                        cuda_in->cre_init = cpu_workspace_local.state.cre + (size_t)cell_offset * (size_t)npe;
                                        cuda_in->crp_state_out = cpu_workspace_local.state.crp + (size_t)cell_offset * (size_t)np;
                                        cuda_in->cre_state_out = cpu_workspace_local.state.cre + (size_t)cell_offset * (size_t)npe;
                                        cuda_in->eps_syn_out = 0;
                                        cuda_in->eps_gamma_out = 0;
                                        cuda_in->eps_nu_out = 0;
                                        cuda_in->workspace = cuda_workspace_local;

                                        active_group_inputs++;
                                        active_group_cells += group_bucket_cells;
                                        group_fp_steps_sum += (long long)seg_fp_steps;
                                        group_cell_steps_sum +=
                                            (long long)group_bucket_cells * (long long)seg_fp_steps;
                                    }
                                    if (group_coeff_segments > 1) {
                                        times_local.interp_ms += now_ms() - t0_interp_group;
                                    }
                                    if (active_group_inputs <= 0) continue;
                                    total_gpu_pipeline_calls++;
                                    total_gpu_pipeline_cells += (long long)active_group_cells;
                                    total_gpu_pipeline_fp_steps += group_fp_steps_sum;
                                    total_gpu_pipeline_cell_steps += group_cell_steps_sum;

                                    if (run_tracer_pipeline_group(gpu_group_inputs_local,
                                                                  active_group_inputs,
                                                                  &none_schedule,
                                                                  &cuda_times) != 0) {
                                        fprintf(stderr,
                                                TRACER_FP_PROGNAME ": CUDA group evolve failed at snapshot %d group %d segment %d/%d\n",
                                                isnap + 1, group_id + 1, iseg + 1, group_coeff_segments);
                                        goto cleanup;
                                    }

                                    times_local.coeff_ms += cuda_times.coeff_ms;
                                    times_local.secondary_ms += cuda_times.secondary_ms;
                                    times_local.solve_ms += cuda_times.solve_ms;
                                    times_local.cuda_setup_ms += cuda_times.setup_ms;
                                    times_local.cuda_h2d_ms += cuda_times.h2d_ms;
                                    times_local.cuda_d2h_ms += cuda_times.d2h_ms;
                                    times_local.cuda_other_ms += cuda_times.other_ms;
                                    times_local.cuda_total_ms += cuda_times.total_ms;
                                }
                            }

                            tracer_fp_batch_scatter_group_state(
                                group_bucket_count, group_bucket_ids, &bucket_local,
                                gpu_group_cell_offsets_local,
                                crp_state, cre_state,
                                &cpu_workspace_local, &times_local);
                            for (ilocal = 0; ilocal < group_bucket_count; ilocal++) {
                                record_bucket_metrics(
                                    bucket_local.member_counts[group_bucket_ids[ilocal]],
                                    &total_bucket_calls,
                                    &total_bucket_cells,
                                    &min_bucket_size,
                                    &max_bucket_size_seen,
                                    bucket_hist_counts);
                            }
                            {
                                const double bucket_wall_ms = now_ms() - t0_bucket_host;
                                const double bucket_known_ms =
                                    (times_local.pack_ms - pack_before_ms) +
                                    (times_local.interp_ms - interp_before_ms) +
                                    (times_local.cuda_total_ms - cuda_total_before_ms);
                                const double bucket_host_ms = bucket_wall_ms - bucket_known_ms;
                                if (bucket_host_ms > 0.0) {
                                    times_local.bucket_host_ms += bucket_host_ms;
                                }
                            }
                            ibucket = group_end - 1;
                            continue;
                        }
                    }
                }
#endif

                if (!use_cuda_backend) {
                    /* Default CPU workspace path: pack directly into the reusable CPU
                     * workspace and skip CUDA-only bucket buffers such as synch b/logB tables. */
                    if (log_root && debug_target_idx_in_bucket >= 0) {
                        fprintf(stderr,
                                "[debug tracer %ld] snapshot=%d cpu-workspace bucket=%d/%d size=%d idx=%d "
                                "nstep=%d non=%d coeff_segments=%d\n",
                                debug_target_global, isnap + 1, ibucket + 1, nbuckets,
                                bucket_count, debug_target_idx_in_bucket,
                                bucket_nstep, bucket_n_on, coeff_interp_segments);
                        tracer_fp_debug_print_cre_sample(
                            "pre-bucket",
                            debug_target_global,
                            isnap + 1,
                            cre_state + (size_t)bucket_members[debug_target_idx_in_bucket] * (size_t)npe);
                    }
                    if (tracer_fp_batch_execute_bucket(
                            bucket_count, bucket_members,
                            bucket_nstep, bucket_n_on, bucket_nstep, 1,
                            TRACER_FP_INTEGRATION_QUANTIZED, coeff_interp_segments, allow_snapshot_interp,
                            ntracer, elapsed_gyr, dt_snap[isnap], z_snap[isnap],
                            allow_snapshot_interp ? z_snap[isnap + 1] : z_snap[isnap],
                            requested_t_acc_direct_gyr,
                            &crp_grid_local, &cre_grid_local,
                            n_gas_snap, kbt_snap, b_field_snap, divv_snap,
                            lturb_snap, dv_snap, cs_snap, beta_snap,
                            rad_ic_zero, rad_ic_m1_zero, rad_ic_p1_zero,
                            n_gas_next_snap, kbt_next_snap, b_field_next_snap, divv_next_snap,
                            lturb_next_snap, dv_next_snap, cs_next_snap, beta_next_snap,
                            rad_ic_row_next, rad_ic_m1_next_val, rad_ic_p1_next_val,
                            qpi_batch, qepri_batch, fqe_flat, np_min_qe,
                            tracer_mass,
                            dsa_disable_adiabatic_local,
                            crp_state, cre_state, &times_local, &cpu_workspace_local) != 0) {
                        fprintf(stderr,
                                TRACER_FP_PROGNAME ": CPU workspace bucket evolve failed at snapshot %d bucket %d\n",
                                isnap + 1, ibucket + 1);
                        goto cleanup;
                    }
                    if (debug_target_idx_in_bucket >= 0) {
                        tracer_fp_debug_print_cre_sample(
                            "post-bucket",
                            debug_target_global,
                            isnap + 1,
                            cre_state + (size_t)bucket_members[debug_target_idx_in_bucket] * (size_t)npe);
                    }
                    record_bucket_metrics(bucket_count, &total_bucket_calls,
                                                    &total_bucket_cells,
                                                    &min_bucket_size,
                                                    &max_bucket_size_seen,
                                                    bucket_hist_counts);
                    continue;
                }

            t0_bucket_host = now_ms();
            pack_before_ms = times_local.pack_ms;
            interp_before_ms = times_local.interp_ms;
            cuda_total_before_ms = times_local.cuda_total_ms;
            if (tracer_fp_batch_pack_bucket(bucket_count, bucket_members,
                                            coeff_interp_segments, ntracer,
                                            n_gas_snap, kbt_snap, b_field_snap, divv_snap,
                                            lturb_snap, dv_snap, cs_snap, beta_snap,
                                            rad_ic_zero, rad_ic_m1_zero, rad_ic_p1_zero,
                                            n_gas_next_snap, kbt_next_snap, b_field_next_snap,
                                            divv_next_snap, lturb_next_snap, dv_next_snap,
                                            cs_next_snap, beta_next_snap,
                                            rad_ic_row_next, rad_ic_m1_next_val, rad_ic_p1_next_val,
                                            qpi_batch, qepri_batch,
                                    tracer_mass,
                                    dsa_disable_adiabatic_local,
                                    crp_state, cre_state,
                                    0, 0,
                                    &cpu_workspace_local, 0,
                                    &times_local) != 0) {
                                goto cleanup;
                            }
            n_gas_bucket = cpu_workspace_local.bg_curr.n_gas;
            kbt_bucket = cpu_workspace_local.bg_curr.kbt;
            b_field_bucket = cpu_workspace_local.bg_curr.b_field;
            divv_bucket = cpu_workspace_local.bg_curr.divv;
            lturb_bucket = cpu_workspace_local.bg_curr.lturb;
            dv_bucket = cpu_workspace_local.bg_curr.dv;
            cs_bucket = cpu_workspace_local.bg_curr.cs;
            beta_bucket = cpu_workspace_local.bg_curr.beta;
            rad_ic_bucket = cpu_workspace_local.bg_curr.rad_ic;
            rad_ic_m1_bucket = cpu_workspace_local.bg_curr.rad_ic_m1;
            rad_ic_p1_bucket = cpu_workspace_local.bg_curr.rad_ic_p1;
            n_gas_next_bucket = cpu_workspace_local.bg_next.n_gas;
            kbt_next_bucket = cpu_workspace_local.bg_next.kbt;
            b_field_next_bucket = cpu_workspace_local.bg_next.b_field;
            divv_next_bucket = cpu_workspace_local.bg_next.divv;
            lturb_next_bucket = cpu_workspace_local.bg_next.lturb;
            dv_next_bucket = cpu_workspace_local.bg_next.dv;
            cs_next_bucket = cpu_workspace_local.bg_next.cs;
            beta_next_bucket = cpu_workspace_local.bg_next.beta;
            rad_ic_next_bucket = cpu_workspace_local.bg_next.rad_ic;
            rad_ic_m1_next_bucket = cpu_workspace_local.bg_next.rad_ic_m1;
            rad_ic_p1_next_bucket = cpu_workspace_local.bg_next.rad_ic_p1;
            n_gas_coeff = cpu_workspace_local.bg_interp.n_gas;
            kbt_coeff = cpu_workspace_local.bg_interp.kbt;
            b_field_coeff = cpu_workspace_local.bg_interp.b_field;
            divv_coeff = cpu_workspace_local.bg_interp.divv;
            lturb_coeff = cpu_workspace_local.bg_interp.lturb;
            dv_coeff = cpu_workspace_local.bg_interp.dv;
            cs_coeff = cpu_workspace_local.bg_interp.cs;
            beta_coeff = cpu_workspace_local.bg_interp.beta;
            rad_ic_coeff = cpu_workspace_local.bg_interp.rad_ic;
            rad_ic_m1_coeff = cpu_workspace_local.bg_interp.rad_ic_m1;
            rad_ic_p1_coeff = cpu_workspace_local.bg_interp.rad_ic_p1;
            qpi_bucket = cpu_workspace_local.state.qpi;
            qepri_bucket = cpu_workspace_local.state.qepri;
            crp_bucket = cpu_workspace_local.state.crp;
            cre_bucket = cpu_workspace_local.state.cre;

            if (log_root && debug_target_idx_in_bucket >= 0) {
                const int dump_all_members = tracer_fp_debug_dump_full_bucket();
                fprintf(stderr,
                        "[debug tracer %ld] snapshot=%d bucket=%d/%d size=%d idx=%d "
                        "nstep=%d non=%d coeff_segments=%d members:",
                        debug_target_global, isnap + 1, ibucket + 1, nbuckets,
                        bucket_count, debug_target_idx_in_bucket,
                        bucket_nstep, bucket_n_on, coeff_interp_segments);
                for (idx = 0; idx < bucket_count && (dump_all_members || idx < 16); idx++) {
                    fprintf(stderr, " %ld", tracer_ids[bucket_members[idx]]);
                }
                if (!dump_all_members && bucket_count > 16) fprintf(stderr, " ...");
                fprintf(stderr, "\n");
                tracer_fp_debug_print_cre_sample("pre-bucket", debug_target_global, isnap + 1,
                                                 cre_bucket + (size_t)debug_target_idx_in_bucket * (size_t)npe);
            } else if (log_root && debug_target_local >= 0) {
                memcpy(debug_target_cre_before_bucket,
                       cre_state + (size_t)debug_target_local * (size_t)npe,
                       (size_t)npe * sizeof(double));
            }

            /* Coefficient interpolation is refreshed only at a small number of
             * FP-step segments. Small buckets stay at one frozen coefficient
             * set, while long buckets can track snapshot-to-snapshot drift. */
            {
                const double base_dt = dt_snap[isnap] / (double)pipeline_nstep;
                int bucket_cuda_last_active_seg = -1;
                int bucket_cuda_state_loaded = 0;

                if (use_cuda_backend) {
                    int seg_probe;
                    for (seg_probe = 0; seg_probe < coeff_interp_segments; seg_probe++) {
                        int probe_begin = 0;
                        int probe_end = 0;

                        segment_fp_range(bucket_nstep, coeff_interp_segments, seg_probe,
                                                   &probe_begin, &probe_end);
                        if (probe_end > probe_begin) {
                            bucket_cuda_last_active_seg = seg_probe;
                        }
                    }
                }

                for (iseg = 0; iseg < coeff_interp_segments; iseg++) {
                    const double alpha = (coeff_interp_segments > 1)
                        ? ((double)iseg + 0.5) / (double)coeff_interp_segments : 0.0;
                    const double z_coeff = allow_snapshot_interp
                        ? z_snap[isnap] + alpha * (z_snap[isnap + 1] - z_snap[isnap])
                        : z_snap[isnap];
                    const double *n_gas_coeff_ptr = n_gas_bucket;
                    const double *kbt_coeff_ptr = kbt_bucket;
                    const double *b_field_coeff_ptr = b_field_bucket;
                    const double *divv_coeff_ptr = divv_bucket;
                    const double *lturb_coeff_ptr = lturb_bucket;
                    const double *dv_coeff_ptr = dv_bucket;
                    const double *cs_coeff_ptr = cs_bucket;
                    const double *beta_coeff_ptr = beta_bucket;
                    const double *rad_ic_coeff_ptr = rad_ic_bucket;
                    const double *rad_ic_m1_coeff_ptr = rad_ic_m1_bucket;
                    const double *rad_ic_p1_coeff_ptr = rad_ic_p1_bucket;
                    int seg_fp_begin = 0;
                    int seg_fp_end = 0;
                    int seg_fp_steps;
                    int seg_pipeline_steps;
                    int seg_on_begin;
                    int seg_on_end;
                    int seg_on_count;
                    int seg_use_windowed_reacc;
                    int seg_on_start_local;
                    int seg_on_end_local;

                    segment_fp_range(bucket_nstep, coeff_interp_segments, iseg,
                                               &seg_fp_begin, &seg_fp_end);
                    seg_fp_steps = seg_fp_end - seg_fp_begin;
                    if (seg_fp_steps <= 0) continue;

                    seg_pipeline_steps = seg_fp_steps;

                    seg_on_begin = (bucket_on_start > seg_fp_begin) ? bucket_on_start : seg_fp_begin;
                    seg_on_end = (bucket_on_end < seg_fp_end) ? bucket_on_end : seg_fp_end;
                    if (seg_on_end < seg_on_begin) seg_on_end = seg_on_begin;
                    seg_on_count = seg_on_end - seg_on_begin;
                    seg_use_windowed_reacc = (seg_on_count < seg_fp_steps) ? 1 : 0;
                    seg_on_start_local = seg_on_begin - seg_fp_begin;
                    seg_on_end_local = seg_on_start_local + seg_on_count;

                    if (coeff_interp_segments > 1 &&
                        (!use_cuda_backend || !enable_cuda_device_bg_interp)) {
                        const double t0_interp = now_ms();
                        interp_array(bucket_count, n_gas_bucket, n_gas_next_bucket, alpha, n_gas_coeff);
                        interp_array(bucket_count, kbt_bucket, kbt_next_bucket, alpha, kbt_coeff);
                        interp_array(bucket_count, b_field_bucket, b_field_next_bucket, alpha, b_field_coeff);
                        interp_array(bucket_count, divv_bucket, divv_next_bucket, alpha, divv_coeff);
                        interp_array(bucket_count, lturb_bucket, lturb_next_bucket, alpha, lturb_coeff);
                        interp_array(bucket_count, dv_bucket, dv_next_bucket, alpha, dv_coeff);
                        interp_array(bucket_count, cs_bucket, cs_next_bucket, alpha, cs_coeff);
                        interp_array(bucket_count, beta_bucket, beta_next_bucket, alpha, beta_coeff);
                        interp_array(bucket_count * npe, rad_ic_bucket, rad_ic_next_bucket, alpha, rad_ic_coeff);
                        interp_array(bucket_count, rad_ic_m1_bucket, rad_ic_m1_next_bucket, alpha, rad_ic_m1_coeff);
                        interp_array(bucket_count, rad_ic_p1_bucket, rad_ic_p1_next_bucket, alpha, rad_ic_p1_coeff);
                        n_gas_coeff_ptr = n_gas_coeff;
                        kbt_coeff_ptr = kbt_coeff;
                        b_field_coeff_ptr = b_field_coeff;
                        divv_coeff_ptr = divv_coeff;
                        lturb_coeff_ptr = lturb_coeff;
                        dv_coeff_ptr = dv_coeff;
                        cs_coeff_ptr = cs_coeff;
                        beta_coeff_ptr = beta_coeff;
                        rad_ic_coeff_ptr = rad_ic_coeff;
                        rad_ic_m1_coeff_ptr = rad_ic_m1_coeff;
                        rad_ic_p1_coeff_ptr = rad_ic_p1_coeff;
                        times_local.interp_ms += now_ms() - t0_interp;
                    }

                    {
                        const int sub_fp_begin = seg_fp_begin;
                        const int sub_fp_end = seg_fp_end;
                        const int sub_fp_steps = sub_fp_end - sub_fp_begin;
                        const int sub_pipeline_steps = sub_fp_steps;
                        const int sub_on_begin = (bucket_on_start > sub_fp_begin) ? bucket_on_start : sub_fp_begin;
                        int sub_on_end = (bucket_on_end < sub_fp_end) ? bucket_on_end : sub_fp_end;
                        int sub_use_windowed_reacc;
                        int sub_on_start_local;
                        int sub_on_end_local;
                        const double sub_time_mid = elapsed_gyr +
                            ((double)sub_fp_begin + 0.5 * (double)sub_fp_steps) * base_dt;
                        const double active_t_acc_gyr =
                            (direct_tacc_schedule_enabled && sub_time_mid < t_off_gyr)
                            ? t_acc_off_gyr
                            : requested_t_acc_direct_gyr;

                        if (sub_fp_steps <= 0) continue;
                        if (sub_on_end < sub_on_begin) sub_on_end = sub_on_begin;
                        sub_use_windowed_reacc = ((sub_on_end - sub_on_begin) < sub_fp_steps) ? 1 : 0;
                        sub_on_start_local = sub_on_begin - sub_fp_begin;
                        sub_on_end_local = sub_on_start_local + (sub_on_end - sub_on_begin);

                        t_acc_direct_gyr = active_t_acc_gyr;

                    if (use_cuda_backend) {
#ifdef FP_USE_CUDA_BACKEND
                        FpCoeffBatchInput coeff_in;
                        FpCudaPipelineInput cuda_in;
                        FpCudaOutputSchedule none_schedule;
                        FpCudaPipelineTimes cuda_times;
                        const int upload_bucket_state = !bucket_cuda_state_loaded;
                        const int download_bucket_state = (iseg == bucket_cuda_last_active_seg);
                        const int gather_rank_state =
                            use_cuda_rank_state_path ? upload_bucket_state : 0;
                        const int scatter_rank_state =
                            use_cuda_rank_state_path ? download_bucket_state : 0;
                        const int download_host_state =
                            (!use_cuda_rank_state_path && download_bucket_state);

                        memset(&coeff_in, 0, sizeof(coeff_in));
                        memset(&cuda_in, 0, sizeof(cuda_in));
                        memset(&none_schedule, 0, sizeof(none_schedule));
                        memset(&cuda_times, 0, sizeof(cuda_times));

                        coeff_in.ncell = bucket_count;
                        coeff_in.z = z_coeff;
                        coeff_in.dt_gyr = dt_snap[isnap] / (double)bucket_nstep;
                        coeff_in.epmax = 1.0e7;
                        coeff_in.n_gas = n_gas_coeff_ptr;
                        coeff_in.kbt = kbt_coeff_ptr;
                        coeff_in.b_field = b_field_coeff_ptr;
                        coeff_in.divv_gyr = divv_coeff_ptr;
                        coeff_in.l_turb_mpc = lturb_coeff_ptr;
                        coeff_in.dv_imc = dv_coeff_ptr;
                        coeff_in.cs = cs_coeff_ptr;
                        coeff_in.beta_pl = beta_coeff_ptr;
                        coeff_in.tracer_mass_msun = cpu_workspace_local.state.mass_msun;
                        coeff_in.disable_adiabatic = cpu_workspace_local.state.disable_adiabatic;
                        coeff_in.crp_state_cell_major = cpu_workspace_local.state.crp;
                        coeff_in.cre_state_cell_major = cpu_workspace_local.state.cre;
                        coeff_in.rad_ic_batch = rad_ic_coeff_ptr;
                        coeff_in.rad_ic_m1 = rad_ic_m1_coeff_ptr;
                        coeff_in.rad_ic_p1 = rad_ic_p1_coeff_ptr;
                        if (coeff_interp_segments > 1 && enable_cuda_device_bg_interp) {
                            coeff_in.device_interp_background = 1;
                            coeff_in.interp_alpha = alpha;
                            coeff_in.n_gas = n_gas_bucket;
                            coeff_in.kbt = kbt_bucket;
                            coeff_in.b_field = b_field_bucket;
                            coeff_in.divv_gyr = divv_bucket;
                            coeff_in.l_turb_mpc = lturb_bucket;
                            coeff_in.dv_imc = dv_bucket;
                            coeff_in.cs = cs_bucket;
                            coeff_in.beta_pl = beta_bucket;
                            coeff_in.rad_ic_batch = rad_ic_bucket;
                            coeff_in.rad_ic_m1 = rad_ic_m1_bucket;
                            coeff_in.rad_ic_p1 = rad_ic_p1_bucket;
                            coeff_in.n_gas_next = n_gas_next_bucket;
                            coeff_in.kbt_next = kbt_next_bucket;
                            coeff_in.b_field_next = b_field_next_bucket;
                            coeff_in.divv_gyr_next = divv_next_bucket;
                            coeff_in.l_turb_mpc_next = lturb_next_bucket;
                            coeff_in.dv_imc_next = dv_next_bucket;
                            coeff_in.cs_next = cs_next_bucket;
                            coeff_in.beta_pl_next = beta_next_bucket;
                            coeff_in.rad_ic_batch_next = rad_ic_next_bucket;
                            coeff_in.rad_ic_m1_next = rad_ic_m1_next_bucket;
                            coeff_in.rad_ic_p1_next = rad_ic_p1_next_bucket;
                        }

                        cuda_in.ncell = bucket_count;
                        cuda_in.nstep = sub_pipeline_steps;
                        cuda_in.fp_cadence_steps = 1;
                        cuda_in.diff_cadence_steps = 1;
                        cuda_in.adv_cadence_steps = 1;
                        cuda_in.use_windowed_reacc = sub_use_windowed_reacc;
                        cuda_in.on_start_step = sub_on_start_local;
                        cuda_in.on_end_step = sub_on_end_local;
                        cuda_in.nfreq = nfreq;
                        cuda_in.nx_tab = synch_local.nx_tab;
                        cuda_in.dt = base_dt;
                        cuda_in.epmax = 1.0e7;
                        cuda_in.xmin = synch_local.xmin;
                        cuda_in.psi_value = psi;
                        cuda_in.mach_limit_value = mach_limit;
                        cuda_in.transport_mode = cuda_transport_mode();
                        cuda_in.crp_grid = &crp_grid_local;
                        cuda_in.cre_grid = &cre_grid_local;
                        cuda_in.coeff_in = &coeff_in;
                        cuda_in.synch_in = 0;
                        /* The first active segment loads bucket state/source
                         * rows into scratch. With rank-global state this is a
                         * device gather; otherwise it is the legacy H2D copy. */
                        cuda_in.qpi_batch =
                            (!use_cuda_rank_state_path && upload_bucket_state) ? qpi_bucket : 0;
                        cuda_in.qepri_batch =
                            (!use_cuda_rank_state_path && upload_bucket_state) ? qepri_bucket : 0;
                        cuda_in.beta_p = beta_p;
                        cuda_in.crp_init =
                            (!use_cuda_rank_state_path && upload_bucket_state) ? crp_bucket : 0;
                        cuda_in.cre_init =
                            (!use_cuda_rank_state_path && upload_bucket_state) ? cre_bucket : 0;
                        cuda_in.rank_state_indices =
                            (gather_rank_state || scatter_rank_state) ? bucket_members : 0;
                        cuda_in.gather_rank_state = gather_rank_state;
                        cuda_in.scatter_rank_state = scatter_rank_state;
                        cuda_in.workspace = cuda_workspace_local;
                        total_gpu_pipeline_calls++;
                        total_gpu_pipeline_cells += (long long)bucket_count;
                        total_gpu_pipeline_fp_steps += (long long)sub_fp_steps;
                        total_gpu_pipeline_cell_steps +=
                            (long long)bucket_count * (long long)sub_fp_steps;

                        if (run_tracer_pipeline(&cuda_in, &none_schedule,
                                                download_host_state ? crp_bucket : 0,
                                                download_host_state ? cre_bucket : 0,
                                                0,
                                                NULL,
                                                NULL,
                                                &cuda_times) != 0) {
                            fprintf(stderr, TRACER_FP_PROGNAME ": CUDA bucket evolve failed at snapshot %d bucket %d segment %d/%d\n",
                                    isnap + 1, ibucket + 1, iseg + 1, coeff_interp_segments);
                            goto cleanup;
                        }
                        bucket_cuda_state_loaded = 1;
                        times_local.coeff_ms += cuda_times.coeff_ms;
                        times_local.secondary_ms += cuda_times.secondary_ms;
                        times_local.solve_ms += cuda_times.solve_ms;
                        times_local.cuda_setup_ms += cuda_times.setup_ms;
                        times_local.cuda_h2d_ms += cuda_times.h2d_ms;
                        times_local.cuda_d2h_ms += cuda_times.d2h_ms;
                        times_local.cuda_other_ms += cuda_times.other_ms;
                        times_local.cuda_total_ms += cuda_times.total_ms;
#endif
                    }
                        t_acc_direct_gyr = requested_t_acc_direct_gyr;
                    }
                }
            }

            if (debug_target_idx_in_bucket >= 0 && !use_cuda_rank_state_path) {
                tracer_fp_debug_print_cre_sample("post-bucket", debug_target_global, isnap + 1,
                                                 cre_bucket + (size_t)debug_target_idx_in_bucket * (size_t)npe);
            }

            if (!use_cuda_rank_state_path) {
                tracer_fp_batch_scatter_bucket_state(bucket_count, bucket_members,
                                                     crp_state, cre_state,
                                                     &cpu_workspace_local, &times_local);
            }

            record_bucket_metrics(bucket_count, &total_bucket_calls,
                                            &total_bucket_cells,
                                            &min_bucket_size,
                                            &max_bucket_size_seen,
                                            bucket_hist_counts);
            if (debug_target_local >= 0 && debug_target_idx_in_bucket < 0 &&
                !use_cuda_rank_state_path) {
                (void)tracer_fp_debug_report_row_change(
                    "cross-bucket",
                    debug_target_global,
                    isnap + 1,
                    ibucket + 1,
                    debug_target_cre_before_bucket,
                    cre_state + (size_t)debug_target_local * (size_t)npe
                );
            }
            {
                const double bucket_wall_ms = now_ms() - t0_bucket_host;
                const double bucket_known_ms =
                    (times_local.pack_ms - pack_before_ms) +
                    (times_local.interp_ms - interp_before_ms) +
                    (times_local.cuda_total_ms - cuda_total_before_ms);
                const double bucket_host_ms = bucket_wall_ms - bucket_known_ms;
                if (bucket_host_ms > 0.0) {
                    times_local.bucket_host_ms += bucket_host_ms;
                }
            }
        }
        }

#ifdef FP_USE_CUDA_BACKEND
        if (use_cuda_rank_state_path && !bucket_stats_only) {
            double rank_d2h_ms = 0.0;
            if (cuda_rank_state_download(cuda_workspace_local, ntracer,
                                         crp_state, cre_state,
                                         &rank_d2h_ms) != 0) {
                fprintf(stderr, TRACER_FP_PROGNAME ": CUDA rank state download failed at snapshot %d\n",
                        isnap + 1);
                goto cleanup;
            }
            times_local.cuda_d2h_ms += rank_d2h_ms;
            times_local.cuda_total_ms += rank_d2h_ms;
        }
#endif

        if (write_output_files && emit_cr_snapshot) {
            const double t0_output_write = now_ms();
            if (use_tile_output) {
                if (tracer_fp_tile_write_cr(&ne_tile_output_local, &np_tile_output_local,
                                            ntracer, (size_t)(isnap + 1),
                                            cre_state, crp_state,
                                            raw_slots[raw_curr_slot].rho, tracer_mass,
                                            tracer_output_per_cc, write_crp_output) != 0) {
                    fprintf(stderr, TRACER_FP_PROGNAME ": failed to write tile CR state at snapshot %d (%s)\n",
                            isnap + 1, strerror(errno));
                    goto cleanup;
                }
                times_local.output_write_calls++;
            } else if (use_mapped_output) {
                if (use_mapped_chunk) {
                    const int local_cr_slot = (isnap + 1) - mapped_chunk_cr_base_slot;
                    const int checkpoint_due =
                        checkpoint_enabled &&
                        (((isnap + 1) % tracer_checkpoint_interval) == 0 || isnap + 1 == nsnap);
                    populate_output_buffers(ne_chunk_core, np_chunk_core, ntracer, local_cr_slot,
                                            cre_state, crp_state,
                                            raw_slots[raw_curr_slot].rho, tracer_mass,
                                            tracer_output_per_cc);
                    if (local_cr_slot + 1 > mapped_chunk_cr_slots) {
                        mapped_chunk_cr_slots = local_cr_slot + 1;
                    }
                    mapped_chunk_count = isnap - mapped_chunk_start_snapshot + 1;
                    if (mapped_chunk_count >= mapped_chunk_snapshots ||
                        (!tracer_fp_has_later_emit_snapshot(schedule, isnap, nsnap) &&
                         !tracer_fp_has_later_emit_snapshot(schedule, isnap + 1, nsnap + 1)) ||
                        checkpoint_due) {
                        if (tracer_fp_flush_map_chunk(
                                &ne_output_local, &np_output_local, &epssyn_output_local,
                                &epsic_output_local, &epsgamma_output_local,
                                &epsnu_output_local,
                                ic_enabled, gamma_enabled, neutrino_enabled, ntracer, nfreq,
                                mapped_chunk_start_snapshot, mapped_chunk_count,
                                mapped_chunk_cr_base_slot, mapped_chunk_cr_slots,
                                ne_chunk_core, np_chunk_core,
                                epssyn_chunk_core, epsic_chunk_core,
                                epsgamma_chunk_core, epsnu_chunk_core) != 0) {
                            fprintf(stderr, TRACER_FP_PROGNAME ": failed to flush mapped output chunk ending at snapshot %d (%s)\n",
                                    isnap + 1, strerror(errno));
                            goto cleanup;
                        }
                        mapped_chunk_count = 0;
                        mapped_chunk_cr_slots = 0;
                        times_local.output_write_calls++;
                    }
                } else {
                    if (tracer_fp_write_cr_map(&ne_output_local, &np_output_local, ntracer, isnap + 1,
                                               cre_state, crp_state,
                                               raw_slots[raw_curr_slot].rho, tracer_mass,
                                               tracer_output_per_cc) != 0) {
                        fprintf(stderr, TRACER_FP_PROGNAME ": failed to write mapped CR state at snapshot %d (%s)\n",
                                isnap + 1, strerror(errno));
                        goto cleanup;
                    }
                    times_local.output_write_calls++;
                }
            } else {
                populate_output_buffers(ne_buffer_core, np_buffer_core, ntracer, isnap + 1,
                                        cre_state, crp_state,
                                        raw_slots[raw_curr_slot].rho, tracer_mass,
                                        tracer_output_per_cc);
                times_local.output_write_calls++;
            }
            times_local.output_write_ms += now_ms() - t0_output_write;
        }
        if (checkpoint_enabled &&
            (((isnap + 1) % tracer_checkpoint_interval) == 0 || isnap + 1 == nsnap)) {
            if (use_mapped_output || use_tile_output) {
                const double t0_sync = now_ms();
                if (use_mapped_output) {
                    if (tracer_fp_map_sync(&ne_output_local) != 0 ||
                        tracer_fp_map_sync(&np_output_local) != 0 ||
                        tracer_fp_map_sync(&epssyn_output_local) != 0 ||
                        (ic_enabled && tracer_fp_map_sync(&epsic_output_local) != 0) ||
                        (gamma_enabled && tracer_fp_map_sync(&epsgamma_output_local) != 0) ||
                        (neutrino_enabled && tracer_fp_map_sync(&epsnu_output_local) != 0)) {
                        fprintf(stderr, TRACER_FP_PROGNAME ": failed to sync mapped output before checkpoint\n");
                        goto cleanup;
                    }
                } else {
                    if (tracer_fp_tile_sync(&ne_tile_output_local) != 0 ||
                        (write_crp_output && tracer_fp_tile_sync(&np_tile_output_local) != 0) ||
                        tracer_fp_tile_sync(&epssyn_tile_output_local) != 0 ||
                        (ic_enabled && tracer_fp_tile_sync(&epsic_tile_output_local) != 0) ||
                        (gamma_enabled && tracer_fp_tile_sync(&epsgamma_tile_output_local) != 0) ||
                        (neutrino_enabled && tracer_fp_tile_sync(&epsnu_tile_output_local) != 0)) {
                        fprintf(stderr, TRACER_FP_PROGNAME ": failed to sync tile output before checkpoint\n");
                        goto cleanup;
                    }
                }
                times_local.output_sync_ms += now_ms() - t0_sync;
                times_local.output_sync_calls++;
            }
            {
                const double t0_checkpoint = now_ms();
                TracerFpCkptWrite checkpoint_write;

                memset(&checkpoint_write, 0, sizeof(checkpoint_write));
                checkpoint_write.checkpoint_dir = checkpoint_dir_effective;
                checkpoint_write.mpi_rank = mpi_rank;
                checkpoint_write.world_size = world_size;
                checkpoint_write.ntracer = ntracer;
                checkpoint_write.ntracer_global = ntracer_global;
                checkpoint_write.nsnap = nsnap;
                checkpoint_write.nfreq = nfreq;
                checkpoint_write.completed_snapshot = isnap;
                checkpoint_write.tracer_start = tracer_start;
                checkpoint_write.tracer_ids = tracer_id_core;
                checkpoint_write.crp_state = crp_state;
                checkpoint_write.cre_state = cre_state;
                if (tracer_fp_ckpt_write(&checkpoint_write) != 0) {
                    fprintf(stderr, TRACER_FP_PROGNAME ": failed to write checkpoint for snapshot %d\n",
                            isnap + 1);
                    goto cleanup;
                }
                times_local.checkpoint_ms += now_ms() - t0_checkpoint;
                times_local.checkpoint_calls++;
            }
            if (log_root) {
                printf("  checkpoint          : wrote snapshot %d to %s\n",
                       isnap + 1, checkpoint_dir_effective);
                fflush(stdout);
            }
        }
        if (log_root && debug_target_local >= 0) {
            const size_t debug_e_off = (size_t)debug_target_local * (size_t)npe;
            tracer_fp_debug_print_cre_sample("post-snapshot", debug_target_global, isnap + 1,
                                             cre_state + debug_e_off);
        }

        if (isnap + 1 < nsnap) {
            const int old_curr_slot = bg_curr_slot;
            bg_curr_slot = bg_next_slot;
            bg_next_slot = old_curr_slot;

            if (background_mode == TRACER_FP_BACKGROUND_EVOLVING && isnap + 2 < nsnap) {
                if (input_mode == TRACER_FP_INPUT_HDF5) {
                    const int recycle_slot = raw_prev_slot;
                    raw_prev_slot = raw_curr_slot;
                    raw_curr_slot = raw_next_slot;
                    raw_next_slot = recycle_slot;

                    if (isnap + 3 < nsnap) {
                        /* This is a raw-data lookahead for building the next
                         * background slot, not the snapshot being solved. */
                        /* if (log_root) {
                            printf("  data load           : prefetching snapshot nsnp=%d\n",
                                   nsnp_i + isnap + 3);
                            fflush(stdout);
                        } */
                        const double t0_read = now_ms();
                        const double t0_epoch = epoch_ms();
                        if (tracer_load_hdf5_raw_snapshot_slice(&hdf5_meta,
                                                                nsnp_i + isnap + 3,
                                                                tracer_start, source_offsets, ntracer,
                                                                &raw_slots[raw_next_slot]) != 0) {
                            goto cleanup;
                        }
                        {
                            const int snapshot_index = isnap + 3;
                            const int physical_snapshot = nsnp_i + snapshot_index;
                            const double read_ms = now_ms() - t0_read;
                            const double t1_epoch = epoch_ms();
                            int order = 0;

                            times_local.input_read_ms += read_ms;
                            times_local.input_read_calls++;
                            times_local.input_selected_runs += ctx->input_selected_count;
                            times_local.input_offset_runs += ctx->input_offset_runs;
                            if (ctx->input_offset_min_run_len > 0 &&
                                (times_local.input_offset_min_run_len == 0 ||
                                 ctx->input_offset_min_run_len < times_local.input_offset_min_run_len)) {
                                times_local.input_offset_min_run_len = ctx->input_offset_min_run_len;
                            }
                            if (ctx->input_offset_max_run_len > times_local.input_offset_max_run_len) {
                                times_local.input_offset_max_run_len = ctx->input_offset_max_run_len;
                            }
                            if (ctx->input_offset_max_gap > times_local.input_offset_max_gap) {
                                times_local.input_offset_max_gap = ctx->input_offset_max_gap;
                            }

                            if (input_read_order != 0) {
                                order = *input_read_order;
                                *input_read_order = order + 1;
                            }
                            if (input_read_fp != 0) {
                                fprintf(input_read_fp,
                                        "%d\t%d\t%s\t%d\t%s\t%d\t%d\t%.8e\t%.3f\t%.3f\t%.6f\t%d\t%d\t%d\t%d\t%d\t%d\t%ld\t%ld\t%ld\n",
                                        mpi_rank, local_rank,
                                        (hostname != 0 && *hostname != '\0') ? hostname : "unknown",
                                        order, "lookahead",
                                        snapshot_index, physical_snapshot,
                                        z_snap[snapshot_index],
                                        t0_epoch, t1_epoch, read_ms,
                                        ntracer, ctx->input_selected_count,
                                        ctx->input_offset_runs,
                                        ctx->input_offset_min_run_len,
                                        ctx->input_offset_max_run_len,
                                        ctx->input_offset_max_gap,
                                        ctx->input_offset_min,
                                        ctx->input_offset_max,
                                        ctx->input_offset_span);
                                fflush(input_read_fp);
                            }
                        }
                    }
                    {
                        const double t0_bg = now_ms();
                        if (tracer_prepare_background_from_raw(
                                ntracer,
                                &raw_slots[raw_prev_slot],
                                &raw_slots[raw_curr_slot],
                                (isnap + 3 < nsnap) ? &raw_slots[raw_next_slot]
                                                    : &raw_slots[raw_curr_slot],
                                z_snap[isnap + 1],
                                z_snap[isnap + 2],
                                (isnap + 3 < nsnap) ? z_snap[isnap + 3] : z_snap[isnap + 2],
                                1,
                                (isnap + 3 < nsnap),
                                &bg_slots[bg_next_slot]) != 0) {
                            goto cleanup;
                        }
                        times_local.bg_prepare_ms += now_ms() - t0_bg;
                    }
                } else if (tracer_fp_fill_background_snapshot_synthetic(isnap + 2, z_snap, ntracer,
                                                                       tracer_ids, ntracer_global,
                                                                       nsnap, &bg_slots[bg_next_slot]) != 0) {
                    goto cleanup;
                }
            }
        }
        elapsed_gyr += dt_snap[isnap];
    }
