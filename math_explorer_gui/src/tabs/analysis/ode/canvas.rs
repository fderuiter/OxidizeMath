#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::comparison::{compute_discrepancy, simulate_ode, OdeSimSpec, SolverTrajectory};
use super::recovery;
use super::systems::CustomOdeSpec;
use super::OdeSolverTool;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{InteractiveTool, TraceSeries};
use crate::presets::OdePreset;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::analysis::ode::IntegrationMethod;
use std::collections::HashMap;
use std::sync::Arc;

impl InteractiveTool for OdeSolverTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "ODE Solvers"
    }
    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_full_export_datasets()
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_full_state()
    }

    fn save_recovery_state(&self) -> Option<String> {
        self.serialize_recovery_state()
    }

    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        self.deserialize_state(state, true)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.deserialize_state(state, false)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let mut changed = false;

        let sweeps_before = crate::reflective_ui::slider::live_active_sweep_count(parent_ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "ode_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            changed |= self.render_sidebar_controls(ui);
        });
        let sweep_active =
            crate::reflective_ui::slider::live_active_sweep_count(parent_ui.ctx()) > sweeps_before;

        self.apply_sidebar_changes(changed, sweep_active);

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_plot_canvas(ui, &mut changed);
        });
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl OdeSolverTool {
    pub(super) fn sim_spec<'a>(
        &'a self,
        method: IntegrationMethod,
        custom_spec: Option<&'a CustomOdeSpec>,
    ) -> OdeSimSpec<'a> {
        OdeSimSpec {
            preset: self.preset,
            method,
            dt: self.dt,
            total_time: self.total_time,
            param_k: self.param_k,
            param_r: self.param_r,
            param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0,
            ic_v0: self.ic_v0,
            matrix_a: self.matrix_a,
            custom_spec,
            planar_x_expr: self.planar_x_input.cached_expr(),
            planar_y_expr: self.planar_y_input.cached_expr(),
        }
    }

    pub(super) fn recalculate_comparison(&mut self) {
        self.comparison_manager.clear();
        if !self.comparison_mode {
            return;
        }

        let custom_spec = self.build_custom_ode_spec();
        let rk4 =
            simulate_ode(&self.sim_spec(IntegrationMethod::RungeKutta4, custom_spec.as_ref()));
        let euler = simulate_ode(&self.sim_spec(IntegrationMethod::Euler, custom_spec.as_ref()));
        let heun = simulate_ode(&self.sim_spec(IntegrationMethod::Heun, custom_spec.as_ref()));

        if !self.has_custom_input_error() && (rk4.diverged || euler.diverged || heun.diverged) {
            self.diverged = true;
        }

        self.discrepancy_euler_rk4 = compute_discrepancy(&euler, &rk4);
        self.discrepancy_heun_rk4 = compute_discrepancy(&heun, &rk4);
        self.discrepancy_euler_heun = compute_discrepancy(&euler, &heun);

        self.populate_comparison_snapshots(&[rk4, euler, heun]);
    }

    pub(super) fn populate_comparison_snapshots(&mut self, trajs: &[SolverTrajectory]) {
        let (t1_lbl, t2_lbl, px_lbl, py_lbl) = self.active_state_labels();
        let phase_active = self.is_phase_plane_active();
        let has_v = self.has_second_state();
        for traj in trajs {
            let series = if phase_active {
                let pts_phase: Vec<[f64; 2]> = traj
                    .y_series
                    .iter()
                    .zip(&traj.v_series)
                    .map(|(&x, &y)| [x, y])
                    .collect();
                vec![TraceSeries::new(format!("({px_lbl}, {py_lbl})"), pts_phase)]
            } else {
                let pts_y: Vec<[f64; 2]> = traj
                    .time_series
                    .iter()
                    .zip(&traj.y_series)
                    .map(|(&t, &y)| [t, y])
                    .collect();
                let mut out = vec![TraceSeries::new(t1_lbl, pts_y)];
                if has_v && !traj.v_series.is_empty() {
                    let pts_v: Vec<[f64; 2]> = traj
                        .time_series
                        .iter()
                        .zip(&traj.v_series)
                        .map(|(&t, &v)| [t, v])
                        .collect();
                    out.push(TraceSeries::new(t2_lbl, pts_v));
                }
                out
            };
            self.comparison_manager
                .add_snapshot(traj.method.name(), "", series);
        }
    }

    pub(super) fn render_discrepancy_panel(&self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.heading("Discrepancy Metrics");
        ui.group(|ui| {
            for (label, disc) in [
                ("Euler vs RK4", self.discrepancy_euler_rk4),
                ("Heun vs RK4", self.discrepancy_heun_rk4),
                ("Euler vs Heun", self.discrepancy_euler_heun),
            ] {
                render_copyable_metric(
                    ui,
                    &format!("{label} (Max)"),
                    &format!("{:.4e}", disc.max_error),
                );
                render_copyable_metric(
                    ui,
                    &format!("{label} (RMS)"),
                    &format!("{:.4e}", disc.rms_error),
                );
            }
        });
    }

    pub(super) fn render_plot_canvas(&mut self, ui: &mut egui::Ui, changed: &mut bool) {
        if self.is_phase_plane_active() {
            self.render_phase_plane_canvas(ui, changed);
            return;
        }

        let (t1_lbl, t2_lbl, _, _) = self.active_state_labels();
        let has_v = self.has_second_state();
        ui.heading("Solution Trajectory");

        if FiniteStateGuard::diverged(self) {
            let mut scratch_y = Vec::new();
            let mut scratch_v = Vec::new();
            recovery::render_recovery_ui(ui, self, changed, &mut scratch_y, &mut scratch_v);
        }

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let line_y = Line::new(t1_lbl, PlotPoints::from_shared(Arc::clone(&self.display_y)))
            .color(palette.series_color(0));

        Plot::new("ode_plot")
            .view_aspect(2.0)
            .x_axis_label("t \\text{ (Time)}")
            .y_axis_label(t1_lbl)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                if !self.comparison_mode {
                    plot_ui.line(line_y);
                    if has_v && !self.display_v.is_empty() {
                        let line_v =
                            Line::new(t2_lbl, PlotPoints::from_shared(Arc::clone(&self.display_v)))
                                .color(palette.series_color(1));
                        plot_ui.line(line_v);
                    }
                } else {
                    self.comparison_manager.render_on_plot(plot_ui);
                }
                if self.show_annotations && !FiniteStateGuard::diverged(self) {
                    super::annotations::render_ode_callouts(plot_ui, self);
                }
                self.snapshot_manager.render_on_plot(plot_ui);
                self.callout_manager.render_on_plot(plot_ui);
            });
    }

    pub(crate) fn collect_full_xy_pairs(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let mut plot_points_y = Vec::with_capacity(self.time_series.len());
        let mut plot_points_v = if self.v_series.is_empty() {
            Vec::new()
        } else {
            Vec::with_capacity(self.v_series.len())
        };
        for (i, &t) in self.time_series.iter().enumerate() {
            if let Some(&y) = self.y_series.get(i) {
                plot_points_y.push([t, y]);
            }
            if let Some(&v) = self.v_series.get(i) {
                plot_points_v.push([t, v]);
            }
        }
        (plot_points_y, plot_points_v)
    }

    pub(crate) fn build_current_trace_series(&self) -> Vec<TraceSeries> {
        let (t1_lbl, t2_lbl, _, _) = self.active_state_labels();
        let (plot_points_y, plot_points_v) = self.collect_full_xy_pairs();
        let mut series = vec![TraceSeries::new(t1_lbl, plot_points_y)];
        if self.has_second_state() && !plot_points_v.is_empty() {
            series.push(TraceSeries::new(t2_lbl, plot_points_v));
        }
        series
    }

    pub(crate) fn build_full_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (t1_lbl, t2_lbl, px_lbl, py_lbl) = self.active_state_labels();
        let mut datasets = Vec::new();
        if self.comparison_mode && !self.comparison_manager.is_empty() {
            datasets.extend(self.comparison_manager.export_datasets());
        } else if self.is_phase_plane_active() {
            let phase_pts: Vec<[f64; 2]> = self
                .y_series
                .iter()
                .copied()
                .zip(self.v_series.iter().copied())
                .map(|(x, y)| [x, y])
                .collect();
            datasets.push((format!("Primary ({px_lbl}, {py_lbl})"), phase_pts));
            for (idx, traj) in self.seeded_trajectories.iter().enumerate() {
                let pts: Vec<[f64; 2]> = traj
                    .y_series
                    .iter()
                    .copied()
                    .zip(traj.v_series.iter().copied())
                    .filter(|(x, y)| x.is_finite() && y.is_finite())
                    .map(|(x, y)| [x, y])
                    .collect();
                if let Some(&ic) = pts.first() {
                    datasets.push((
                        format!("Orbit {} ({:.2}, {:.2})", idx + 1, ic[0], ic[1]),
                        pts,
                    ));
                }
            }
        } else {
            let (plot_points_y, plot_points_v) = self.collect_full_xy_pairs();
            datasets.push((t1_lbl.to_string(), plot_points_y));
            if self.has_second_state() && !plot_points_v.is_empty() {
                datasets.push((t2_lbl.to_string(), plot_points_v));
            }
        }
        datasets.extend(self.snapshot_manager.export_datasets());
        for (name, pts) in egui_plot::get_last_plot_datasets() {
            if !datasets.iter().any(|(existing, _)| existing == &name) {
                datasets.push((name, pts));
            }
        }
        datasets
    }
}

impl scientific_metadata::theory::TheoryDescribable for OdeSolverTool {
    fn theory_description(&self) -> String {
        match self.preset {
            OdePreset::Exponential => "First-order linear ordinary differential equation describing exponential growth or decay proportional to current quantity.".into(),
            OdePreset::HarmonicOscillator => "Second-order ordinary differential equation modeling simple harmonic motion under a Hooke's law restoring force.".into(),
            OdePreset::LogisticGrowth => "Nonlinear ordinary differential equation modeling density-dependent population growth with an environmental carrying capacity.".into(),
            OdePreset::LinearSystem2D => "Planar 2x2 linear dynamical system classified on the Poincaré trace-determinant bifurcation diagram.".into(),
            OdePreset::PlanarCustom => "Coupled 2D planar ordinary differential equations: x' = f(t, x, y), y' = g(t, x, y).".into(),
            OdePreset::Custom if self.custom_coupled => "Coupled two-state ordinary differential equation system: y' = f(t, y, v), v' = g(t, y, v)".into(),
            OdePreset::Custom => "Custom user-defined ordinary differential equation: y' = f(t, y)".into(),
        }
    }
    fn phonetic_description(&self) -> String {
        "Ordinary differential equations theory".into()
    }
    fn theory_citation(&self) -> String {
        match self.preset {
            OdePreset::Exponential
            | OdePreset::LinearSystem2D
            | OdePreset::PlanarCustom
            | OdePreset::Custom => {
                "Differential Equations and Dynamical Systems (Perko, 2001)".into()
            }
            OdePreset::HarmonicOscillator => {
                "Classical Mechanics (Goldstein et al., 2002); Perko (2001)".into()
            }
            OdePreset::LogisticGrowth => {
                "Elements of Mathematical Biology (Lotka, 1925); Verhulst (1838)".into()
            }
        }
    }
    fn theory_latex_equation(&self) -> Option<String> {
        let eq = match self.preset {
            OdePreset::Exponential => r"\frac{dy}{dt} = k y",
            OdePreset::HarmonicOscillator => r"\frac{d^2 y}{dt^2} = -k y",
            OdePreset::LogisticGrowth => r"\frac{dy}{dt} = r y (1 - \frac{y}{K})",
            OdePreset::LinearSystem2D => r"\dot{\mathbf{x}} = A \mathbf{x}",
            OdePreset::PlanarCustom | OdePreset::Custom => return None,
        };
        Some(eq.into())
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([
            (
                "Exponential".into(),
                "First-order linear ODE representing exponential growth or decay".into(),
            ),
            (
                "Harmonic Oscillator".into(),
                "Second-order linear ODE representing simple harmonic motion".into(),
            ),
            (
                "Logistic Growth".into(),
                "Nonlinear first-order ODE modeling population growth with carrying capacity"
                    .into(),
            ),
            (
                "Linear 2D".into(),
                "Planar 2x2 linear system with Poincaré stability regimes".into(),
            ),
            (
                "Planar Custom".into(),
                "Coupled 2D user-defined ODE phase-plane system".into(),
            ),
            (
                "Custom".into(),
                "Custom user-defined scalar or coupled ordinary differential equation".into(),
            ),
        ])
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "harmonic_oscillator",
        title: "Harmonic Oscillator",
        description: "y'' = -k·y: Second-order differential equation modeling undamped periodic spring motion.",
        category: "Differential Equations",
        icon: "〰️",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_HARMONIC_OSCILLATOR,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "logistic_growth",
        title: "Logistic Growth",
        description: "y' = r·y·(1 - y/K): Non-linear growth constrained by carrying capacity K.",
        category: "Population Dynamics",
        icon: "📈",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_LOGISTIC_GROWTH,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "exponential_decay",
        title: "Exponential Decay",
        description: "y' = k·y: First-order differential decay process with constant half-life.",
        category: "Calculus",
        icon: "📉",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_EXPONENTIAL_DECAY,
    }
}
