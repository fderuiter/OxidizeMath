#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:advanced_linear_algebra]

//! Parameter constraints, boundary-accurate eigenmode readouts, and antinodal lock-in for [`super::ChladniWaveTankTool`].

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use eframe::egui;
use math_explorer::pure_math::analysis::pde::wave_2d::{
    BoundaryMode2D, Eigenmode2D, Wave2DMedium, evaluate_eigenmode_amplitude, populate_eigenmodes,
    snap_to_mode_antinode,
};
use scientific_metadata::theory::ParameterConstraint;

use super::atlas::{ChladniModalCartographer, DRUMHEAD_RADIUS_NORM, MAX_CHLADNI_MODES};
use super::{
    AMP_MAX, AMP_MIN, ChladniWaveTankTool, DAMPING_MAX, DAMPING_MIN, DT_MAX, DT_MIN, FREQ_MAX,
    FREQ_MIN, GRID_SIZE, REFRACTIVE_MAX, REFRACTIVE_MIN, SLIT_SEP_MAX, SLIT_SEP_MIN, SPEED_MAX,
    SPEED_MIN, STIFFNESS_MAX, STIFFNESS_MIN, TankPreset,
};
use crate::async_sim::atlas::AtlasTeleportEvent;
use crate::reflective_ui::render_copyable_metric;

/// Canonical parameter constraints exposed by `TheoryDescribable::theory_parameters`.
#[must_use]
pub(crate) fn chladni_parameter_constraints() -> HashMap<String, ParameterConstraint> {
    [
        ("driver_frequency", FREQ_MIN, FREQ_MAX, 0.001),
        ("plate_stiffness", STIFFNESS_MIN, STIFFNESS_MAX, 1.0e-5),
        ("wave_speed", SPEED_MIN, SPEED_MAX, 0.01),
        ("damping", DAMPING_MIN, DAMPING_MAX, 0.01),
        ("driver_x", 0.0, 1.0, 0.01),
        ("driver_y", 0.0, 1.0, 0.01),
        ("driver_amplitude", AMP_MIN, AMP_MAX, 0.5),
        ("time_step", DT_MIN, DT_MAX, 0.001),
        ("slit_separation", SLIT_SEP_MIN, SLIT_SEP_MAX, 0.5),
        ("refractive_index", REFRACTIVE_MIN, REFRACTIVE_MAX, 0.05),
    ]
    .into_iter()
    .map(|(k, min, max, step)| (k.to_string(), ParameterConstraint { min, max, step }))
    .collect()
}

impl ChladniWaveTankTool {
    pub(crate) fn domain_physical_length(&self) -> f64 {
        self.solver.dx() * ((GRID_SIZE - 1) as f64)
    }

    pub(crate) fn active_eigenmodes(&self) -> ([Eigenmode2D; MAX_CHLADNI_MODES], usize) {
        let mut modes = [Eigenmode2D::default(); MAX_CHLADNI_MODES];
        let count = populate_eigenmodes(
            self.state.preset == TankPreset::CircularDrumhead,
            GRID_SIZE,
            self.domain_physical_length(),
            DRUMHEAD_RADIUS_NORM,
            &self.pde_params(),
            self.state.clamp_layout.to_pin_constraint(),
            &mut modes,
        );
        (modes, count)
    }

    pub(crate) fn boundary_fundamental_readout(&self) -> Option<(String, f64)> {
        let (modes, count) = self.active_eigenmodes();
        if count == 0 {
            return None;
        }
        let m0 = modes[0];
        let prefix = if self.state.preset == TankPreset::CircularDrumhead {
            "Bessel"
        } else {
            match self.state.boundary {
                BoundaryMode2D::Neumann => "Neumann",
                BoundaryMode2D::Clamped => "Clamped",
                BoundaryMode2D::Dirichlet | BoundaryMode2D::Sponge => "Dirichlet",
            }
        };
        Some((format!("{prefix} f({},{})", m0.m, m0.n), m0.frequency_hz))
    }

    pub(crate) fn nearest_eigenmode(&self) -> Option<(Eigenmode2D, f64)> {
        let (modes, count) = self.active_eigenmodes();
        let best = modes[..count].iter().copied().min_by(|a, b| {
            (a.frequency_hz - self.state.driver_frequency)
                .abs()
                .total_cmp(&(b.frequency_hz - self.state.driver_frequency).abs())
        })?;
        let coupling = evaluate_eigenmode_amplitude(
            &best,
            self.state.driver_x,
            self.state.driver_y,
            DRUMHEAD_RADIUS_NORM,
        )
        .abs();
        Some((best, coupling))
    }

    pub(super) fn draw_eigenmode_metric_cards(&self, ui: &mut egui::Ui) {
        if let Some((label, f0)) = self.boundary_fundamental_readout() {
            render_copyable_metric(ui, &label, &format!("{f0:.3} Hz"));
        }
        if let Some((nearest, coup)) = self.nearest_eigenmode() {
            render_copyable_metric(
                ui,
                &format!("Nearest Mode ({},{})", nearest.m, nearest.n),
                &format!("{:.3} Hz (|Φ|={coup:.2})", nearest.frequency_hz),
            );
        }
    }

    pub(crate) fn serialize_chladni_state(&self) -> Option<String> {
        let mut snap = self.state.clone();
        snap.atlas_visible = self.atlas.is_visible();
        let (xk, yk) = self.atlas.axis_keys();
        snap.atlas_axes = [xk.to_string(), yk.to_string()];
        serde_json::to_string(&snap).ok()
    }

    pub(crate) fn lock_to_eigenmode(&mut self, mode: &Eigenmode2D, ctx: Option<&egui::Context>) {
        self.enforce_chladni_challenge_locks();
        self.state.driver_frequency = mode.frequency_hz.clamp(FREQ_MIN, FREQ_MAX);
        let cur_amp = evaluate_eigenmode_amplitude(
            mode,
            self.state.driver_x,
            self.state.driver_y,
            DRUMHEAD_RADIUS_NORM,
        )
        .abs();
        if cur_amp < 0.55 {
            let (nx, ny) = snap_to_mode_antinode(
                mode,
                self.state.driver_x,
                self.state.driver_y,
                self.state.clamp_layout.to_pin_constraint(),
                DRUMHEAD_RADIUS_NORM,
            );
            self.state.driver_x = nx;
            self.state.driver_y = ny;
        }
        self.state.driver_enabled = true;
        if !self.state.barrier_enabled && !self.state.lens_enabled {
            self.state.line_source = false;
        }
        self.solver.clear_wave_state();
        self.analysis_dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
        let msg = format!(
            "Locked eigenmode ({}, {}) at {:.3} Hz, antinode ({:.2}, {:.2})",
            mode.m, mode.n, self.state.driver_frequency, self.state.driver_x, self.state.driver_y
        );
        if let Some(c) = ctx {
            crate::accessibility::announce_live_status(c, msg);
        } else {
            crate::accessibility::announce_status(&msg);
        }
    }

    pub(crate) fn snap_to_nearest_eigenmode(&mut self, ctx: Option<&egui::Context>) -> bool {
        if let Some((mode, _)) = self.nearest_eigenmode() {
            self.lock_to_eigenmode(&mode, ctx);
            return true;
        }
        false
    }

    pub(crate) fn lock_nearest_fft_eigenmode(
        &mut self,
        k_bin_click: f64,
        ctx: Option<&egui::Context>,
    ) -> Option<Eigenmode2D> {
        let (modes, count) = self.active_eigenmodes();
        let dom_len = self.domain_physical_length();
        let best = modes[..count].iter().copied().min_by(|a, b| {
            (a.fft_radial_bin(GRID_SIZE, dom_len) - k_bin_click)
                .abs()
                .total_cmp(&(b.fft_radial_bin(GRID_SIZE, dom_len) - k_bin_click).abs())
        })?;
        if (best.fft_radial_bin(GRID_SIZE, dom_len) - k_bin_click).abs() <= 2.0 {
            self.lock_to_eigenmode(&best, ctx);
            return Some(best);
        }
        None
    }

    pub(crate) fn apply_atlas_teleport(
        &mut self,
        ev: AtlasTeleportEvent,
        ctx: Option<&egui::Context>,
    ) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "driver_frequency" => self.state.driver_frequency = val.clamp(FREQ_MIN, FREQ_MAX),
                "plate_stiffness" => {
                    self.state.plate_stiffness = val.clamp(STIFFNESS_MIN, STIFFNESS_MAX);
                    if self.state.plate_stiffness > 0.0 && self.state.wave_speed == 0.0 {
                        self.state.medium = Wave2DMedium::BiharmonicPlate;
                    }
                }
                "wave_speed" => self.state.wave_speed = val.clamp(SPEED_MIN, SPEED_MAX),
                "driver_x" => self.state.driver_x = val.clamp(0.02, 0.98),
                "driver_y" => self.state.driver_y = val.clamp(0.02, 0.98),
                _ => {}
            }
        }
        self.enforce_chladni_challenge_locks();
        if (ev.x_key == "driver_frequency" || ev.y_key == "driver_frequency")
            && let Some((nearest, _)) = self.nearest_eigenmode()
        {
            let capture = (0.08 * nearest.frequency_hz).max(0.50);
            if (self.state.driver_frequency - nearest.frequency_hz).abs() <= capture {
                self.lock_to_eigenmode(&nearest, ctx);
                return;
            }
        }
        self.solver.reset_detector();
        self.analysis_dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    pub(super) fn render_chladni_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let ctx = ui.ctx().clone();
        let (xk, yk) = self.atlas.axis_keys();
        let get_val = |k: &str| match k {
            "driver_frequency" => self.state.driver_frequency,
            "plate_stiffness" => self.state.plate_stiffness,
            "wave_speed" => self.state.wave_speed,
            "driver_x" => self.state.driver_x,
            "driver_y" => self.state.driver_y,
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.state.preset as u8).hash(&mut h);
        (self.state.medium as u8).hash(&mut h);
        (self.state.boundary as u8).hash(&mut h);
        (self.state.clamp_layout as u8).hash(&mut h);
        self.state.damping.to_bits().hash(&mut h);
        if xk == "driver_x" {
            self.state.driver_frequency.to_bits().hash(&mut h);
            self.state.plate_stiffness.to_bits().hash(&mut h);
            self.state.wave_speed.to_bits().hash(&mut h);
        } else {
            self.state.driver_x.to_bits().hash(&mut h);
            self.state.driver_y.to_bits().hash(&mut h);
        }
        let params = chladni_parameter_constraints();
        let snap = self.state.clone();
        let dom_len = self.domain_physical_length();
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "chladni_wave_tank",
            get_val(xk),
            get_val(yk),
            h.finish(),
            move |keys| Arc::new(ChladniModalCartographer::new(&params, keys, &snap, dom_len)),
        ) {
            self.apply_atlas_teleport(ev, Some(&ctx));
        }
    }
}
