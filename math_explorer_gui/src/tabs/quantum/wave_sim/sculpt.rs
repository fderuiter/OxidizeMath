//! Live 1D potential $V(x)$ sculpting, resonant double-barrier cavity stamping,
//! and sparse tuple persistence for [`super::WaveSimulatorTool`] and [`super::WaveRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{
    barrier_interval_for_preset, clamp_potential_sample, evaluate_potential_grid, DEFAULT_DT,
    DEFAULT_SIGMA, GRID_POINTS, GRID_X_MAX, GRID_X_MIN, K0_MAX, K0_MIN, POTENTIAL_DISPLAY_SCALE,
    SIGMA_MAX, SIGMA_MIN, X0_MAX, X0_MIN,
};
use super::{WaveRunner, WaveSimulatorTool};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    apply_sparse_potential, encode_sparse_potential, interpolate_1d_span, BrushMode, BrushStroke,
    InteractionContext, SpatialBrushKit,
};
use eframe::egui;
use math_explorer::physics::quantum::{apply_bresenham_potential_stroke, gaussian_wavepacket_into};
use std::any::Any;
use std::sync::{Arc, RwLock};

/// Default `SpatialBrushKit` configured for 1D potential barrier/well/step sculpting on `[-5, 5]`.
#[must_use]
pub(crate) fn default_wave_brush_kit() -> SpatialBrushKit {
    SpatialBrushKit::new(BrushMode::PotentialBarrier, 0.25)
        .with_radius_bounds(0.10, 1.20, 0.05)
        .with_cursor_step(0.20, 0.04)
}

pub(crate) fn apply_discrete_brush_to_potential(
    potential: &mut [f64],
    base_potential: &[f64],
    cx: i32,
    r: i32,
    is_obstacle: bool,
) {
    let n = potential.len();
    if n == 0 {
        return;
    }
    let idx = (cx.max(0) as usize).min(n - 1);
    let half_w = (r.max(1) as usize).min(n / 4);
    let Some((lo, hi)) = interpolate_1d_span(idx, idx, half_w, n) else {
        return;
    };
    if is_obstacle {
        for v in &mut potential[lo..=hi] {
            *v = 16.0;
        }
    } else {
        for (dst, &base_v) in potential[lo..=hi].iter_mut().zip(&base_potential[lo..=hi]) {
            *dst = base_v;
        }
    }
}

pub(crate) fn x_to_grid_index(x: f64, x_min: f64, x_max: f64, n_points: usize) -> usize {
    if n_points <= 1 {
        return 0;
    }
    let frac = ((x - x_min) / (x_max - x_min)).clamp(0.0, 1.0);
    (frac * (n_points - 1) as f64).round() as usize
}

pub(crate) fn apply_stroke_to_potential_slice(
    potential: &mut [f64],
    base_potential: &[f64],
    x_min: f64,
    x_max: f64,
    stroke: &BrushStroke,
) {
    let n = potential.len();
    if n < 2 {
        return;
    }
    let dx = (x_max - x_min) / (n - 1) as f64;
    let i0 = x_to_grid_index(stroke.start[0], x_min, x_max, n);
    let i1 = x_to_grid_index(stroke.end[0], x_min, x_max, n);
    let half_w = ((stroke.radius / dx).round() as usize).clamp(1, n / 4);
    let Some((lo, hi)) = interpolate_1d_span(i0, i1, half_w, n) else {
        return;
    };

    let plot_v = stroke.end[1] / POTENTIAL_DISPLAY_SCALE;
    let is_stamp = stroke.drag_vector[0].abs() < 1e-9 && stroke.drag_vector[1].abs() < 1e-9;

    match stroke.mode {
        BrushMode::PotentialBarrier | BrushMode::Solid => {
            let v_bar = if is_stamp && plot_v.abs() <= 1.0 {
                16.0
            } else if plot_v >= 0.0 {
                clamp_potential_sample(plot_v.max(4.0))
            } else {
                clamp_potential_sample(plot_v)
            };
            for v in &mut potential[lo..=hi] {
                *v = v_bar;
            }
        }
        BrushMode::PotentialWell => {
            let v_well = if plot_v < -1.0 {
                clamp_potential_sample(plot_v)
            } else {
                -12.0
            };
            for v in &mut potential[lo..=hi] {
                *v = v_well;
            }
        }
        BrushMode::PotentialStep => {
            let v_step = if is_stamp && plot_v.abs() <= 1.0 {
                12.0
            } else {
                clamp_potential_sample(plot_v)
            };
            for v in &mut potential[lo..n] {
                *v = v_step;
            }
        }
        BrushMode::Erase => {
            for (dst, &base_v) in potential[lo..=hi].iter_mut().zip(&base_potential[lo..=hi]) {
                *dst = base_v;
            }
        }
        _ => {}
    }
}

pub(crate) fn stamp_double_barrier_into(potential: &mut [f64], x_axis: &[f64]) {
    for (v, &x) in potential.iter_mut().zip(x_axis.iter()) {
        *v = if (-0.52..=-0.28).contains(&x) || (0.28..=0.52).contains(&x) {
            16.0
        } else {
            0.0
        };
    }
}

fn detect_sculpted_interval(
    potential: &[f64],
    base_potential: &[f64],
    x_axis: &[f64],
) -> Option<(f64, f64)> {
    let n = potential.len().min(base_potential.len()).min(x_axis.len());
    let mut first = None;
    let mut last = None;
    for i in 0..n {
        if (potential[i] - base_potential[i]).abs() > 0.5 {
            if first.is_none() {
                first = Some(i);
            }
            last = Some(i);
        }
    }
    let (lo, hi) = (first?, last?);
    Some((x_axis[lo], x_axis[hi]))
}

impl FiniteStateGuard for WaveRunner {
    fn is_finite_state(&self) -> bool {
        if self.diverged {
            return false;
        }
        self.psi
            .vector
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    fn diverged(&self) -> bool {
        self.diverged || !FiniteStateGuard::is_finite_state(self)
    }

    fn reset_parameters(&mut self) {
        self.diverged = false;
        self.diagnostic_report = None;
        self.step_count = 0;
    }
}

impl WaveRunner {
    pub(crate) fn checked_dx(&self) -> Option<f64> {
        let span = self.x_max - self.x_min;
        if self.n_points < 2 || !span.is_finite() || span <= 0.0 {
            return None;
        }
        let dx = span / (self.n_points as f64 - 1.0);
        (dx.is_finite() && dx > 0.0).then_some(dx)
    }

    pub(crate) fn sync_base_and_sparse_potential(&mut self, sparse: &[(u16, f64)]) {
        let n = self.potential.len();
        self.base_potential.resize(n, 0.0);
        self.base_potential
            .copy_from_slice(self.potential.as_slice());
        if !sparse.is_empty()
            && apply_sparse_potential(self.potential.as_mut_slice(), sparse).is_ok()
        {
            if let Some((xl, xr)) = detect_sculpted_interval(
                self.potential.as_slice(),
                &self.base_potential,
                &self.x_axis,
            ) {
                self.x_left = xl;
                self.x_right = xr;
            }
        }
    }

    pub(crate) fn mark_sculpted_potential_dirty(&mut self) {
        let Some(dx) = self.checked_dx() else {
            self.record_divergence(
                "dx",
                "Invalid spatial step dx encountered while rebuilding sculpted potential.",
            );
            return;
        };
        if let Some((xl, xr)) = detect_sculpted_interval(
            self.potential.as_slice(),
            &self.base_potential,
            &self.x_axis,
        ) {
            self.x_left = xl;
            self.x_right = xr;
        } else if let Ok(cfg) = self.config.read() {
            let (xl, xr) = barrier_interval_for_preset(cfg.preset, cfg.width, dx);
            self.x_left = xl;
            self.x_right = xr;
        }
        self.hamiltonian_dirty = true;
        self.cached_dt = None;
        self.update_observables_in_place();
        let dt = self.config.read().map_or(DEFAULT_DT, |c| c.dt);
        self.refresh_sonification(dt);
    }

    fn relaunch_wavepacket_in_place(&mut self) {
        gaussian_wavepacket_into(&self.x_axis, self.x0, self.k0, self.sigma, &mut self.psi);
        self.time = 0.0;
        self.step_count = 0;
        self.entered_barrier = false;
        self.wall_arrived = false;
        self.quasi_bound_latched = false;
        self.latched_transmission = None;
        self.prev_mode_phases = [0.0; 8];
        self.update_observables_in_place();
        let dt = self.config.read().map_or(DEFAULT_DT, |c| c.dt);
        self.refresh_sonification(dt);
    }

    pub(crate) fn handle_sculpt_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::SculptStroke1D { i0, v0, i1, v1 } => {
                apply_bresenham_potential_stroke(self.potential.as_mut_slice(), i0, v0, i1, v1);
                self.mark_sculpted_potential_dirty();
            }
            SimCommand::SetWavepacket { x0, k0, sigma } => {
                if x0.is_finite() {
                    self.x0 = x0.clamp(self.x_min, self.x_max);
                }
                if k0.is_finite() {
                    self.k0 = k0;
                    self.kinetic_energy = 0.5 * k0 * k0;
                }
                self.sigma = if sigma.is_finite() {
                    sigma.clamp(SIGMA_MIN, SIGMA_MAX)
                } else {
                    DEFAULT_SIGMA
                };
                self.relaunch_wavepacket_in_place();
            }
            SimCommand::ApplyBrush {
                cx, r, is_obstacle, ..
            } => {
                let slice = self.potential.as_mut_slice();
                apply_discrete_brush_to_potential(slice, &self.base_potential, cx, r, is_obstacle);
                self.mark_sculpted_potential_dirty();
            }
            SimCommand::ApplyBrushStroke(ref stroke) => {
                let (x_min, x_max) = (self.x_min, self.x_max);
                let slice = self.potential.as_mut_slice();
                apply_stroke_to_potential_slice(slice, &self.base_potential, x_min, x_max, stroke);
                self.mark_sculpted_potential_dirty();
            }
            SimCommand::SyncSculptState(ref payload) => {
                self.potential
                    .as_mut_slice()
                    .copy_from_slice(&self.base_potential);
                let _ = apply_sparse_potential(
                    self.potential.as_mut_slice(),
                    &payload.sparse_potential,
                );
                self.mark_sculpted_potential_dirty();
            }
            SimCommand::ClearObstacles => self.restore_base_potential(),
            SimCommand::Custom(ref s) if s == "Clear Potential Edits" => {
                self.restore_base_potential();
            }
            SimCommand::Custom(ref s) if s == "stamp_double_barrier" => {
                stamp_double_barrier_into(self.potential.as_mut_slice(), &self.x_axis);
                self.mark_sculpted_potential_dirty();
            }
            SimCommand::Custom(ref s) if s == "relaunch_wavepacket" => {
                self.relaunch_wavepacket_in_place();
            }
            _ => {}
        }
    }

    fn restore_base_potential(&mut self) {
        self.potential
            .as_mut_slice()
            .copy_from_slice(&self.base_potential);
        self.mark_sculpted_potential_dirty();
    }
}

impl SimulationRunner for WaveRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                self.reset_parameters();
                let cfg = self
                    .config
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                self.init_from_config(&cfg);
            }
            SimCommand::SetSpeed(speed) => self.steps_per_frame = speed.max(1),
            other => self.handle_sculpt_command(other),
        }
    }

    fn step(&mut self) {
        let dt = self.config.read().map_or(DEFAULT_DT, |c| c.dt);
        self.step_with_dt(dt);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let sonification = if self.diverged {
            crate::audio::SonificationFrame::silent()
        } else {
            self.sonification
        };
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(RwLock::new(Vec::new())),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(self.build_wave_data()) as Box<dyn Any + Send>),
            sonification,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        self.steps_per_frame
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        self.diagnostic_report.clone()
    }
}

impl WaveSimulatorTool {
    pub(super) fn base_potential_vec(&self) -> (Vec<f64>, Vec<f64>) {
        let dx = (GRID_X_MAX - GRID_X_MIN) / (GRID_POINTS as f64 - 1.0);
        let x_axis: Vec<f64> = (0..GRID_POINTS)
            .map(|i| GRID_X_MIN + i as f64 * dx)
            .collect();
        let base = evaluate_potential_grid(
            self.preset,
            &x_axis,
            self.v0,
            self.width,
            dx,
            &self.last_valid_expr,
        );
        (x_axis, base.as_slice().to_vec())
    }

    pub(crate) fn apply_ui_brush_stroke(
        &mut self,
        stroke: BrushStroke,
        ctx: Option<&egui::Context>,
    ) {
        let i0 = x_to_grid_index(stroke.start[0], GRID_X_MIN, GRID_X_MAX, GRID_POINTS);
        let i1 = x_to_grid_index(stroke.end[0], GRID_X_MIN, GRID_X_MAX, GRID_POINTS);
        if i0 != i1
            && stroke.radius <= 0.15
            && matches!(
                stroke.mode,
                BrushMode::PotentialBarrier | BrushMode::PotentialWell
            )
        {
            let v0 = clamp_potential_sample(stroke.start[1] / POTENTIAL_DISPLAY_SCALE);
            let v1 = clamp_potential_sample(stroke.end[1] / POTENTIAL_DISPLAY_SCALE);
            self.apply_sculpt_stroke(i0, v0, i1, v1, ctx);
            return;
        }
        let (_, base) = self.base_potential_vec();
        let mut current = base.clone();
        let _ = apply_sparse_potential(&mut current, &self.sparse_potential);
        apply_stroke_to_potential_slice(&mut current, &base, GRID_X_MIN, GRID_X_MAX, &stroke);
        self.sparse_potential = encode_sparse_potential(&base, &current, 1e-6);
        if let Ok(mut cfg) = self.shared_config.write() {
            cfg.sparse_potential = self.sparse_potential.clone();
        }
        self.bump_revision();
        self.export_runner = None;
        self.export_data = None;
        let _ = self
            .worker
            .send_command(SimCommand::ApplyBrushStroke(stroke));
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn apply_sculpt_stroke(
        &mut self,
        i0: usize,
        v0: f64,
        i1: usize,
        v1: f64,
        ctx: Option<&egui::Context>,
    ) {
        let (_, base) = self.base_potential_vec();
        let mut current = base.clone();
        let _ = apply_sparse_potential(&mut current, &self.sparse_potential);
        apply_bresenham_potential_stroke(&mut current, i0, v0, i1, v1);
        self.sparse_potential = encode_sparse_potential(&base, &current, 1e-6);
        if let Ok(mut cfg) = self.shared_config.write() {
            cfg.sparse_potential = self.sparse_potential.clone();
        }
        self.bump_revision();
        self.export_runner = None;
        self.export_data = None;
        let _ = self
            .worker
            .send_command(SimCommand::SculptStroke1D { i0, v0, i1, v1 });
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn update_wavepacket_live(
        &mut self,
        x0: f64,
        k0: f64,
        sigma: f64,
        ctx: Option<&egui::Context>,
    ) {
        if x0.is_finite() {
            self.x0 = x0.clamp(X0_MIN, X0_MAX);
        }
        if k0.is_finite() {
            self.k0 = k0.clamp(K0_MIN, K0_MAX);
        }
        if sigma.is_finite() {
            self.sigma = sigma.clamp(SIGMA_MIN, SIGMA_MAX);
        }
        self.enforce_challenge_locks();
        if let Ok(mut cfg) = self.shared_config.write() {
            cfg.x0 = self.x0;
            cfg.k0 = self.k0;
            cfg.sigma = self.sigma;
        }
        self.bump_revision();
        self.export_runner = None;
        self.export_data = None;
        let _ = self.worker.send_command(SimCommand::SetWavepacket {
            x0: self.x0,
            k0: self.k0,
            sigma: self.sigma,
        });
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn apply_ui_brush_context(&mut self, ic: &InteractionContext) {
        if let Some(stroke) = ic.brush_stroke {
            self.apply_ui_brush_stroke(stroke, Some(&ic.response.ctx));
        }
    }
}
