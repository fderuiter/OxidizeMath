#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::kspace_phantom::{build_preset_mask_into, colorize_scalar_field_into};
use super::kspace_runner::{
    MriKSpaceRunner, BLOCH_STEP_DT, DELTA_B0_POCKET_TESLA, ISOCHROMAT_SPREAD_FRAC,
    PROBE_SPREAD_FRAC,
};
use super::kspace_types::{
    CompactBrushStroke, MriKSpaceSnapshot, MriKSpaceToolState, StudioBrushMode,
};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::DiagnosticReport;
use eframe::egui::Color32;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::mri::proton;
use nalgebra::Vector3;
use num_complex::Complex;
use std::f64::consts::PI;
use std::sync::Arc;

impl MriKSpaceRunner {
    pub(crate) fn probe_spins_for_idx(
        &self,
        idx: usize,
    ) -> [Vector3<f64>; crate::audio::frame::MAX_VOICES] {
        if idx == self.probe_voxel_idx {
            return self.probe_spins;
        }
        let center = self.spins[idx][1];
        let m_perp = center.x.hypot(center.y);
        let base_theta = center.x.atan2(center.y);
        let spread = PROBE_SPREAD_FRAC * self.delta_b0[idx].abs();
        let t_defocus = self
            .echo_tau
            .map_or(self.current_time, |tau| self.current_time - 2.0 * tau);
        let gamma_t = proton::GYROMAGNETIC_RATIO * t_defocus;
        std::array::from_fn(|s| {
            let theta = base_theta + gamma_t * (((s as f64) - 3.5) * spread);
            Vector3::new(m_perp * theta.sin(), m_perp * theta.cos(), center.z)
        })
    }

    pub(crate) fn reseed_probe_for_voxel(&mut self, selected_voxel: [usize; 2]) {
        let sel_r = selected_voxel[0].min(self.size.saturating_sub(1));
        let sel_c = selected_voxel[1].min(self.size.saturating_sub(1));
        let idx = flatten_2d_index(sel_c, sel_r, self.size);
        let center = self.spins[idx][1];
        let m_perp = center.x.hypot(center.y);
        let base_theta = center.x.atan2(center.y);
        let spread = PROBE_SPREAD_FRAC * self.delta_b0[idx].abs();
        let t_defocus = self
            .echo_tau
            .map_or(self.current_time, |tau| self.current_time - 2.0 * tau);
        let gamma_t = proton::GYROMAGNETIC_RATIO * t_defocus;
        for (s, slot) in self.probe_spins.iter_mut().enumerate() {
            let theta = base_theta + gamma_t * (((s as f64) - 3.5) * spread);
            *slot = Vector3::new(m_perp * theta.sin(), m_perp * theta.cos(), center.z);
        }
        self.probe_voxel_idx = idx;
    }

    pub(crate) fn apply_stroke_direct(
        &mut self,
        mode: StudioBrushMode,
        cx: i32,
        cy: i32,
        r: i32,
        tip_live_spins: bool,
    ) {
        let n_i32 = self.size as i32;
        if mode == StudioBrushMode::KSpaceRfSpike {
            if cx >= 0 && cx < n_i32 && cy >= 0 && cy < n_i32 {
                let spike_amp = 0.35 * ((self.size * self.size) as f64);
                self.kspace_spikes[(cy as usize) * self.size + (cx as usize)] =
                    Complex::new(spike_amp, 0.0);
                if tip_live_spins {
                    self.rf_spike_pending
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            return;
        }
        let r_sq = r * r;
        for y in (cy - r).max(0)..=(cy + r).min(n_i32 - 1) {
            for x in (cx - r).max(0)..=(cx + r).min(n_i32 - 1) {
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r_sq {
                    let idx = (y as usize) * self.size + (x as usize);
                    self.apply_brush_cell(mode, idx, tip_live_spins);
                }
            }
        }
    }

    fn apply_brush_cell(&mut self, mode: StudioBrushMode, idx: usize, tip_live_spins: bool) {
        match mode {
            StudioBrushMode::KSpaceMaskErase => {
                self.kspace_mask[idx] = 0.0;
                self.kspace_spikes[idx] = Complex::new(0.0, 0.0);
            }
            StudioBrushMode::KSpaceMaskPaint => self.kspace_mask[idx] = 1.0,
            StudioBrushMode::SpatialDeltaB0Pocket => {
                let prev_b0 = self.delta_b0[idx];
                self.delta_b0[idx] = DELTA_B0_POCKET_TESLA;
                if tip_live_spins && self.echo_tau.is_none() {
                    rotate_ensemble_delta_b0(
                        &mut self.spins[idx],
                        prev_b0,
                        DELTA_B0_POCKET_TESLA,
                        ISOCHROMAT_SPREAD_FRAC,
                        self.current_time,
                    );
                    if idx == self.probe_voxel_idx {
                        rotate_ensemble_delta_b0(
                            &mut self.probe_spins,
                            prev_b0,
                            DELTA_B0_POCKET_TESLA,
                            PROBE_SPREAD_FRAC,
                            self.current_time,
                        );
                    }
                }
            }
            StudioBrushMode::SpatialFlipAnglePulse => {
                let d_alpha = 45.0 * (PI / 180.0);
                self.local_flip_rad[idx] = (self.local_flip_rad[idx] + d_alpha).min(PI);
                if tip_live_spins {
                    rotate_ensemble_x(&mut self.spins[idx], d_alpha);
                    if idx == self.probe_voxel_idx {
                        rotate_ensemble_x(&mut self.probe_spins, d_alpha);
                    }
                }
            }
            StudioBrushMode::KSpaceRfSpike => {}
        }
    }

    pub(crate) fn handle_custom_action(&mut self, action: &str, cfg: &MriKSpaceToolState) {
        match action {
            "refocus_180" => self.apply_refocus_180_pulse(
                cfg.grad_gx_integral,
                cfg.grad_gy_integral,
                cfg.selected_voxel,
            ),
            "excite_90" => self.excite_to_initial_echo(cfg),
            "sync_config" => self.rebuild_from_config(cfg),
            "sync_mask" => {
                build_preset_mask_into(
                    cfg.mask_preset,
                    self.size,
                    cfg.mask_cutoff,
                    cfg.radial_spokes,
                    cfg.cartesian_stride,
                    &mut self.kspace_mask,
                );
                for s in &cfg.strokes {
                    if let Ok((m, x, y, rad)) = s.validate(self.size) {
                        if m.targets_kspace() {
                            self.apply_stroke_direct(m, x, y, rad, false);
                        }
                    }
                }
                self.sync_dft_kspace_tables();
                self.reconstruct_in_place(
                    cfg.grad_gx_integral,
                    cfg.grad_gy_integral,
                    cfg.selected_voxel,
                );
            }
            "sync_view" => {
                let sr = cfg.selected_voxel[0].min(self.size.saturating_sub(1));
                let sc = cfg.selected_voxel[1].min(self.size.saturating_sub(1));
                if flatten_2d_index(sc, sr, self.size) != self.probe_voxel_idx {
                    self.reseed_probe_for_voxel(cfg.selected_voxel);
                }
                self.reconstruct_in_place(
                    cfg.grad_gx_integral,
                    cfg.grad_gy_integral,
                    cfg.selected_voxel,
                );
            }
            _ => {}
        }
    }

    fn build_probe_audio_input(
        &self,
        sel_voxel: [usize; 2],
        rf_spike_triggered: bool,
    ) -> crate::audio::mappings::MriKSpaceProbeAudioInput {
        let sr = sel_voxel[0].min(self.size.saturating_sub(1));
        let sc = sel_voxel[1].min(self.size.saturating_sub(1));
        let sidx = flatten_2d_index(sc, sr, self.size);
        let idx = if self.rho[sidx] > 1e-4 && self.delta_b0[sidx].abs() > 0.0 {
            sidx
        } else if let Some(p_idx) =
            (0..self.rho.len()).find(|&i| self.rho[i] > 1e-4 && self.delta_b0[i].abs() > 0.0)
        {
            p_idx
        } else if self.rho[sidx] > 1e-4 {
            sidx
        } else {
            (0..self.rho.len())
                .find(|&i| self.rho[i] > 1e-4)
                .unwrap_or(sidx)
        };
        crate::audio::mappings::MriKSpaceProbeAudioInput {
            probe_spins: self.probe_spins_for_idx(idx),
            rho: self.rho[idx],
            t2: self.t2[idx],
            delta_b0: self.delta_b0[idx],
            turing_uv: self.phantom_engine.sample_turing_uv(idx, self.rho[idx]),
            current_time: self.current_time,
            echo_tau: self.echo_tau,
            k_coord_cycles_per_m: self.k_coord_cycles_per_m,
            selected_grad_phase_rad: self.selected_grad_phase_rad,
            rf_spike_triggered,
        }
    }

    fn snapshot_sonification(&self, sel_voxel: [usize; 2]) -> crate::audio::SonificationFrame {
        let rf_spike = self
            .rf_spike_pending
            .swap(false, std::sync::atomic::Ordering::Relaxed);
        if self.is_finite_state() {
            crate::audio::mappings::mri_kspace_probe_frame(
                &self.build_probe_audio_input(sel_voxel, rf_spike),
            )
        } else {
            crate::audio::SonificationFrame::silent()
        }
    }
}

fn rotate_ensemble_x<const K: usize>(ens: &mut [Vector3<f64>; K], angle_rad: f64) {
    let (s, c) = angle_rad.sin_cos();
    for m in ens {
        let my = m.y * c + m.z * s;
        let mz = -m.y * s + m.z * c;
        m.y = my;
        m.z = mz;
    }
}

fn rotate_ensemble_delta_b0<const K: usize>(
    ens: &mut [Vector3<f64>; K],
    prev_b0: f64,
    next_b0: f64,
    spread_frac: f64,
    elapsed_time: f64,
) {
    let gamma_t = proton::GYROMAGNETIC_RATIO * elapsed_time;
    let prev_spread = spread_frac * prev_b0.abs();
    let next_spread = spread_frac * next_b0.abs();
    let center = ((K.saturating_sub(1)) as f64) * 0.5;
    for (s, m) in ens.iter_mut().enumerate() {
        let offset = (s as f64) - center;
        let dbz = (next_b0 + offset * next_spread) - (prev_b0 + offset * prev_spread);
        let (sin_th, cos_th) = (gamma_t * dbz).sin_cos();
        let mx = m.x * cos_th + m.y * sin_th;
        let my = -m.x * sin_th + m.y * cos_th;
        m.x = mx;
        m.y = my;
    }
}

impl SimulationRunner for MriKSpaceRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        let cfg = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .config
            .clone();
        match cmd {
            SimCommand::Reset => self.rebuild_from_config(&cfg),
            SimCommand::ClearObstacles => {
                self.delta_b0.fill(0.0);
                self.local_flip_rad.fill(0.0);
                self.kspace_spikes.fill(Complex::new(0.0, 0.0));
                build_preset_mask_into(
                    cfg.mask_preset,
                    self.size,
                    cfg.mask_cutoff,
                    cfg.radial_spokes,
                    cfg.cartesian_stride,
                    &mut self.kspace_mask,
                );
                self.sync_dft_kspace_tables();
                self.excite_to_initial_echo(&cfg);
            }
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => {
                let mode = if !is_obstacle && cfg.brush_mode == StudioBrushMode::KSpaceMaskPaint {
                    StudioBrushMode::KSpaceMaskErase
                } else {
                    cfg.brush_mode
                };
                let stroke = CompactBrushStroke::new(mode, cx, cy, r.clamp(1, 24));
                if let Ok((m, x, y, rad)) = stroke.validate(self.size) {
                    self.apply_stroke_direct(m, x, y, rad, true);
                    if m.targets_kspace() {
                        self.sync_dft_kspace_tables();
                    }
                    self.reconstruct_in_place(
                        cfg.grad_gx_integral,
                        cfg.grad_gy_integral,
                        cfg.selected_voxel,
                    );
                }
            }
            SimCommand::Custom(ref action) => self.handle_custom_action(action, &cfg),
            _ => {}
        }
    }

    fn step(&mut self) {
        let (evolve_turing, gx, gy, sel_voxel) = self.sync_zero_alloc_step_params();
        if evolve_turing {
            self.phantom_engine
                .step_live_turing(&mut self.rho, &mut self.t1, &mut self.t2);
            self.refresh_relaxation_factors();
        }
        self.advance_spatial_bloch(BLOCH_STEP_DT, sel_voxel);
        self.record_trajectory_point(sel_voxel);
        self.reconstruct_in_place(gx, gy, sel_voxel);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let (high_contrast, sel_voxel) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            (p.high_contrast, p.config.selected_voxel)
        };
        let total = self.size * self.size;
        let kspace_log = self.shifted_kspace_log_magnitude();
        let mut phantom_rgba = vec![Color32::BLACK; total];
        let mut kspace_rgba = vec![Color32::BLACK; total];
        let mut recon_rgba = vec![Color32::BLACK; total];
        colorize_scalar_field_into(&self.rho, high_contrast, &mut phantom_rgba);
        colorize_scalar_field_into(&kspace_log, high_contrast, &mut kspace_rgba);
        colorize_scalar_field_into(&self.recon_mag, high_contrast, &mut recon_rgba);

        let sr = sel_voxel[0].min(self.size.saturating_sub(1));
        let sc = sel_voxel[1].min(self.size.saturating_sub(1));
        let sidx = flatten_2d_index(sc, sr, self.size);
        let (nrmse_pct, acceleration_r) = self.compute_kspace_metrics();
        let sonification = self.snapshot_sonification(sel_voxel);
        let structured = MriKSpaceSnapshot {
            grid_size: self.size,
            current_time: self.current_time,
            echo_tau: self.echo_tau,
            k_coord_cycles_per_m: self.k_coord_cycles_per_m,
            selected_grad_phase_rad: self.selected_grad_phase_rad,
            phantom_rgba,
            kspace_rgba,
            recon_rgba: recon_rgba.clone(),
            recon_mag: self.recon_mag.clone(),
            nrmse_pct,
            acceleration_r,
            selected_spins: self.spins[sidx],
            selected_net_m: self.net_voxel_spin(sel_voxel),
            selected_tissue: [
                self.rho[sidx],
                self.t1[sidx],
                self.t2[sidx],
                self.delta_b0[sidx],
            ],
            selected_trajectory: self.selected_trajectory.iter().copied().collect(),
        };
        StateSnapshot {
            width: self.size,
            height: self.size,
            pixels: Arc::new(std::sync::RwLock::new(recon_rgba)),
            custom_data: vec![self.current_time],
            structured_data: Some(Box::new(structured)),
            sonification,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        self.current_time.is_finite()
            && self.recon_mag.iter().all(|v| v.is_finite())
            && self
                .probe_spins
                .iter()
                .all(|m| m.x.is_finite() && m.y.is_finite() && m.z.is_finite())
            && self.spins.iter().all(|ens| {
                ens.iter()
                    .all(|m| m.x.is_finite() && m.y.is_finite() && m.z.is_finite())
            })
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        (!self.is_finite_state()).then(|| {
            DiagnosticReport::new(
                self.current_time,
                0,
                "kspace_spins",
                "Non-finite voxel spin or reconstructed k-space magnitude encountered.",
            )
        })
    }
}

impl MriKSpaceRunner {
    fn shifted_kspace_log_magnitude(&self) -> Vec<f64> {
        let n = self.size;
        let half = n / 2;
        let mut kspace_log = vec![0.0; n * n];
        for c in 0..n {
            let dc = (c + half) % n;
            for r in 0..n {
                let dr = (r + half) % n;
                let z = self.masked_kspace_rm[flatten_2d_index(r, c, n)];
                kspace_log[flatten_2d_index(dc, dr, n)] =
                    (1.0 + (z.re * z.re + z.im * z.im).sqrt()).ln();
            }
        }
        kspace_log
    }

    /// Computes zero-allocation `(nrmse_pct, acceleration_r)` against `self.rho` and `self.kspace_mask`.
    #[must_use]
    pub fn compute_kspace_metrics(&self) -> (f64, f64) {
        let total = self.rho.len();
        let mut err_sq = 0.0_f64;
        let mut ref_sq = 0.0_f64;
        for (&recon, &truth) in self.recon_mag.iter().zip(&self.rho) {
            let diff = recon - truth;
            err_sq += diff * diff;
            ref_sq += truth * truth;
        }
        let nrmse_pct = if ref_sq > 1.0e-12 {
            100.0 * (err_sq / ref_sq).sqrt()
        } else {
            0.0
        };
        let sampled: f64 = self.kspace_mask.iter().map(|&m| m.clamp(0.0, 1.0)).sum();
        let acceleration_r = if total == 0 {
            1.0
        } else {
            (total as f64) / sampled.max(1.0)
        };
        (nrmse_pct, acceleration_r)
    }
}
