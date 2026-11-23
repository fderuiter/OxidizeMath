#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::kspace_phantom::{build_preset_mask_into, colorize_scalar_field_into};
use super::kspace_runner::{
    MriKSpaceRunner, BLOCH_STEP_DT, DELTA_B0_POCKET_TESLA, ISOCHROMAT_SPREAD_FRAC,
};
use super::kspace_types::{
    CompactBrushStroke, MriKSpaceSnapshot, MriKSpaceToolState, StudioBrushMode, ISOCHROMAT_COUNT,
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
                        self.current_time,
                    );
                }
            }
            StudioBrushMode::SpatialFlipAnglePulse => {
                let d_alpha = 45.0 * (PI / 180.0);
                self.local_flip_rad[idx] = (self.local_flip_rad[idx] + d_alpha).min(PI);
                if tip_live_spins {
                    rotate_ensemble_x(&mut self.spins[idx], d_alpha);
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
            "sync_view" => self.reconstruct_in_place(
                cfg.grad_gx_integral,
                cfg.grad_gy_integral,
                cfg.selected_voxel,
            ),
            _ => {}
        }
    }
}

fn rotate_ensemble_x(ens: &mut [Vector3<f64>; ISOCHROMAT_COUNT], angle_rad: f64) {
    let (s, c) = angle_rad.sin_cos();
    for m in ens {
        let my = m.y * c + m.z * s;
        let mz = -m.y * s + m.z * c;
        m.y = my;
        m.z = mz;
    }
}

fn rotate_ensemble_delta_b0(
    ens: &mut [Vector3<f64>; ISOCHROMAT_COUNT],
    prev_b0: f64,
    next_b0: f64,
    elapsed_time: f64,
) {
    let gamma_t = proton::GYROMAGNETIC_RATIO * elapsed_time;
    let prev_spread = ISOCHROMAT_SPREAD_FRAC * prev_b0.abs();
    let next_spread = ISOCHROMAT_SPREAD_FRAC * next_b0.abs();
    for (s, m) in ens.iter_mut().enumerate() {
        let offset = (s as f64) - 1.0;
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
        let n = self.size;
        let total = n * n;
        let half = n / 2;
        let mut kspace_log = vec![0.0; total];
        for c in 0..n {
            let dc = (c + half) % n;
            for r in 0..n {
                let dr = (r + half) % n;
                let z = self.masked_kspace_rm[flatten_2d_index(r, c, n)];
                kspace_log[flatten_2d_index(dc, dr, n)] =
                    (1.0 + (z.re * z.re + z.im * z.im).sqrt()).ln();
            }
        }
        let mut phantom_rgba = vec![Color32::BLACK; total];
        let mut kspace_rgba = vec![Color32::BLACK; total];
        let mut recon_rgba = vec![Color32::BLACK; total];
        colorize_scalar_field_into(&self.rho, high_contrast, &mut phantom_rgba);
        colorize_scalar_field_into(&kspace_log, high_contrast, &mut kspace_rgba);
        colorize_scalar_field_into(&self.recon_mag, high_contrast, &mut recon_rgba);

        let sr = sel_voxel[0].min(self.size.saturating_sub(1));
        let sc = sel_voxel[1].min(self.size.saturating_sub(1));
        let sidx = sr * self.size + sc;
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
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        1
    }

    fn is_finite_state(&self) -> bool {
        self.current_time.is_finite()
            && self.recon_mag.iter().all(|v| v.is_finite())
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
