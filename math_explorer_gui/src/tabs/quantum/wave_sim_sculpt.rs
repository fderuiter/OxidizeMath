//! Live 1D potential $V(x)$ sculpting, resonant double-barrier cavity stamping,
//! and sparse tuple persistence for [`super::wave_sim::WaveSimulatorTool`] and [`super::wave_sim::WaveRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::wave_sim::presets::{
    barrier_interval_for_preset, clamp_potential_sample, evaluate_potential_grid, GRID_POINTS,
    GRID_X_MAX, GRID_X_MIN, POTENTIAL_DISPLAY_SCALE, V0_MAX,
};
use super::wave_sim::{WaveRunner, WaveSimulatorTool};
use crate::async_sim::SimCommand;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    apply_sparse_potential, encode_sparse_potential, interpolate_1d_span, BrushInteractionMode,
    BrushMode, BrushStroke, InteractionContext, SpatialBrushKit,
};
use eframe::egui;
use math_explorer::physics::quantum::{construct_1d_hamiltonian_into, gaussian_wavepacket};

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
            let v_bar = if is_stamp {
                16.0
            } else {
                clamp_potential_sample(plot_v.clamp(6.0, V0_MAX))
            };
            for v in &mut potential[lo..=hi] {
                *v = v_bar;
            }
        }
        BrushMode::PotentialWell => {
            let v_well = if is_stamp {
                -12.0
            } else if plot_v < -1.0 {
                clamp_potential_sample(plot_v.clamp(-V0_MAX, -1.0))
            } else {
                -12.0
            };
            for v in &mut potential[lo..=hi] {
                *v = v_well;
            }
        }
        BrushMode::PotentialStep => {
            let v_step = if is_stamp {
                12.0
            } else {
                clamp_potential_sample(plot_v.clamp(4.0, V0_MAX))
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

fn detect_sculpted_interval(potential: &[f64], base_potential: &[f64], x_axis: &[f64]) -> Option<(f64, f64)> {
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
    pub(crate) fn dx(&self) -> f64 {
        (self.x_max - self.x_min) / (self.n_points.max(2) as f64 - 1.0)
    }

    pub(crate) fn sync_base_and_sparse_potential(&mut self, sparse: &[(u16, f64)]) {
        self.base_potential = self.potential.as_slice().to_vec();
        if !sparse.is_empty()
            && apply_sparse_potential(self.potential.as_mut_slice(), sparse).is_ok()
        {
            if let Some((xl, xr)) =
                detect_sculpted_interval(self.potential.as_slice(), &self.base_potential, &self.x_axis)
            {
                self.x_left = xl;
                self.x_right = xr;
            }
        }
    }

    pub(crate) fn rebuild_sculpted_hamiltonian(&mut self) {
        let dx = self.dx();
        if let Some((xl, xr)) =
            detect_sculpted_interval(self.potential.as_slice(), &self.base_potential, &self.x_axis)
        {
            self.x_left = xl;
            self.x_right = xr;
        } else if let Ok(cfg) = self.config.read() {
            let (xl, xr) = barrier_interval_for_preset(cfg.preset, cfg.width, dx);
            self.x_left = xl;
            self.x_right = xr;
        }
        construct_1d_hamiltonian_into(&self.potential, dx, 1.0, 1.0, &mut self.hamiltonian);
        self.cached_dt = None;
        self.update_observables_in_place();
    }

    pub(crate) fn handle_sculpt_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::ApplyBrush {
                cx,
                r,
                is_obstacle,
                ..
            } => {
                let slice = self.potential.as_mut_slice();
                apply_discrete_brush_to_potential(slice, &self.base_potential, cx, r, is_obstacle);
                self.rebuild_sculpted_hamiltonian();
            }
            SimCommand::ApplyBrushStroke(ref stroke) => {
                let (x_min, x_max) = (self.x_min, self.x_max);
                let slice = self.potential.as_mut_slice();
                apply_stroke_to_potential_slice(slice, &self.base_potential, x_min, x_max, stroke);
                self.rebuild_sculpted_hamiltonian();
            }
            SimCommand::SyncSculptState(ref payload) => {
                self.potential
                    .as_mut_slice()
                    .copy_from_slice(&self.base_potential);
                let _ = apply_sparse_potential(self.potential.as_mut_slice(), &payload.sparse_potential);
                self.rebuild_sculpted_hamiltonian();
            }
            SimCommand::ClearObstacles => self.restore_base_potential(),
            SimCommand::Custom(ref s) if s == "Clear Potential Edits" => {
                self.restore_base_potential();
            }
            SimCommand::Custom(ref s) if s == "stamp_double_barrier" => {
                stamp_double_barrier_into(self.potential.as_mut_slice(), &self.x_axis);
                self.rebuild_sculpted_hamiltonian();
            }
            SimCommand::Custom(ref s) if s == "relaunch_wavepacket" => {
                self.psi = gaussian_wavepacket(&self.x_axis, self.x0, self.k0, self.sigma);
                self.time = 0.0;
                self.step_count = 0;
                self.entered_barrier = false;
                self.wall_arrived = false;
                self.latched_transmission = None;
                self.update_observables_in_place();
            }
            _ => {}
        }
    }

    fn restore_base_potential(&mut self) {
        self.potential
            .as_mut_slice()
            .copy_from_slice(&self.base_potential);
        self.rebuild_sculpted_hamiltonian();
    }
}

impl WaveSimulatorTool {
    fn base_potential_vec(&self) -> (Vec<f64>, Vec<f64>) {
        let dx = (GRID_X_MAX - GRID_X_MIN) / (GRID_POINTS as f64 - 1.0);
        let x_axis: Vec<f64> = (0..GRID_POINTS).map(|i| GRID_X_MIN + i as f64 * dx).collect();
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

    pub(crate) fn apply_ui_brush_stroke(&mut self, stroke: BrushStroke, ctx: Option<&egui::Context>) {
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
        let _ = self.worker.send_command(SimCommand::ApplyBrushStroke(stroke));
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn apply_ui_brush_context(&mut self, ic: &InteractionContext) {
        if let Some(stroke) = ic.brush_stroke {
            self.apply_ui_brush_stroke(stroke, None);
        }
    }

    pub(crate) fn draw_sculpt_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Potential V(x) Sculpting");
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.brush_kit.interaction_mode,
                BrushInteractionMode::Sculpt,
                "🖌 Sculpt",
            );
            ui.selectable_value(
                &mut self.brush_kit.interaction_mode,
                BrushInteractionMode::Pan,
                "✋ Pan",
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.brush_kit.mode, BrushMode::PotentialBarrier, "Barrier");
            ui.selectable_value(&mut self.brush_kit.mode, BrushMode::PotentialWell, "Well");
            ui.selectable_value(&mut self.brush_kit.mode, BrushMode::PotentialStep, "Step");
            ui.selectable_value(&mut self.brush_kit.mode, BrushMode::Erase, "Erase");
        });
        ui.add(egui::Slider::new(&mut self.brush_kit.radius, 0.10..=1.20).text("Brush Width"));
        ui.horizontal_wrapped(|ui| {
            if ui.button("⛩ Stamp Double Barrier").clicked() {
                let (x_axis, base) = self.base_potential_vec();
                let mut target = base.clone();
                stamp_double_barrier_into(&mut target, &x_axis);
                self.sparse_potential = encode_sparse_potential(&base, &target, 1e-6);
                if let Ok(mut cfg) = self.shared_config.write() {
                    cfg.sparse_potential = self.sparse_potential.clone();
                }
                self.bump_revision();
                self.export_runner = None;
                self.export_data = None;
                let _ = self
                    .worker
                    .send_command(SimCommand::Custom("stamp_double_barrier".to_string()));
                ctx.request_repaint();
            }
            if ui.button("Clear Potential Edits").clicked() {
                self.sparse_potential.clear();
                if let Ok(mut cfg) = self.shared_config.write() {
                    cfg.sparse_potential.clear();
                }
                self.bump_revision();
                self.export_runner = None;
                self.export_data = None;
                let _ = self
                    .worker
                    .send_command(SimCommand::Custom("Clear Potential Edits".to_string()));
                ctx.request_repaint();
            }
            if ui.button("🚀 Re-launch ψ(x,0)").clicked() {
                let _ = self
                    .worker
                    .send_command(SimCommand::Custom("relaunch_wavepacket".to_string()));
                ctx.request_repaint();
            }
        });
    }
}
