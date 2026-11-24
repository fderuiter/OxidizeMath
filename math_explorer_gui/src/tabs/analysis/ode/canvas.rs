#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::comparison::{
    build_view_trace_series, compute_series_discrepancy, simulate_ode, OdeSimSpec,
    SolverTrajectory, TrajectoryView,
};
use super::recovery;
use super::systems::CustomOdeSpec;
use super::OdeSolverTool;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{InteractiveTool, TraceSeries};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::analysis::ode::IntegrationMethod;
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

    fn url_state_revision(&self) -> u64 {
        self.compute_url_state_revision()
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::plot_with_code()
    }

    fn export_code_artifact(&self) -> Option<crate::framework::CodeArtifact> {
        self.build_code_artifact()
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
        self.sync_series_cache();

        let sweeps_before =
            crate::reflective_ui::slider::current_pass_active_sweep_count(parent_ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "ode_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            changed |= self.render_sidebar_controls(ui, sweeps_before);
        });
        self.sync_series_cache();

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
    pub(super) fn sim_spec_with_dt<'a>(
        &'a self,
        method: IntegrationMethod,
        dt: f64,
        custom_spec: Option<&'a CustomOdeSpec>,
    ) -> OdeSimSpec<'a> {
        OdeSimSpec {
            preset: self.preset,
            method,
            dt,
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

    fn simulate_if_inactive(
        &self,
        target: IntegrationMethod,
        dt: f64,
        reuse_active: bool,
        custom_spec: Option<&CustomOdeSpec>,
    ) -> Option<SolverTrajectory> {
        if reuse_active && self.method == target {
            None
        } else {
            Some(simulate_ode(&self.sim_spec_with_dt(
                target,
                dt,
                custom_spec,
            )))
        }
    }

    fn trajectory_view<'a>(
        &'a self,
        target: IntegrationMethod,
        simulated: &'a Option<SolverTrajectory>,
    ) -> TrajectoryView<'a> {
        if let Some(traj) = simulated {
            TrajectoryView {
                method: traj.method,
                time_series: &traj.time_series,
                y_series: &traj.y_series,
                v_series: &traj.v_series,
                diverged: traj.diverged,
                is_active_reused: false,
            }
        } else {
            TrajectoryView {
                method: target,
                time_series: &self.time_series,
                y_series: &self.y_series,
                v_series: &self.v_series,
                diverged: self.diverged,
                is_active_reused: true,
            }
        }
    }

    fn build_comparison_snapshot_series(
        &self,
        views: &[TrajectoryView<'_>],
    ) -> Vec<(&'static str, Vec<TraceSeries>)> {
        let labels = self.active_state_labels();
        let (t1_lbl, t2_lbl, _, _) = labels;
        let phase_active = self.is_phase_plane_active();
        let has_v = self.has_second_state();
        let cache_matches = !phase_active
            && self
                .series_cache
                .matches_series(&self.time_series, &self.y_series, &self.v_series);
        views
            .iter()
            .map(|view| {
                let series = if view.is_active_reused && cache_matches {
                    let (py, pv) = self.series_cache.shared_points();
                    let mut out = vec![TraceSeries::new(t1_lbl, py)];
                    if has_v && !pv.is_empty() {
                        out.push(TraceSeries::new(t2_lbl, pv));
                    }
                    out
                } else {
                    build_view_trace_series(view, labels, has_v, phase_active)
                };
                (view.method.name(), series)
            })
            .collect()
    }

    pub(super) fn recalculate_comparison(&mut self) {
        self.comparison_manager.clear();
        if !self.comparison_mode {
            self.budget_state.comparison_notice = None;
            return;
        }

        let primary_dt = self.budget_state.effective_dt.max(self.dt);
        let (comp_dt, comp_notice) = self.effective_comparison_dt(primary_dt);
        self.budget_state.comparison_notice = comp_notice;
        let reuse_active = (comp_dt - primary_dt).abs() <= 1e-15
            && (!self.has_custom_input_error() || !self.time_series.is_empty() || self.diverged);
        self.populate_comparison_for_dt(comp_dt, reuse_active);
    }

    fn populate_comparison_for_dt(&mut self, comp_dt: f64, reuse_active: bool) {
        let custom_spec = self.build_custom_ode_spec();
        let [rk4_sim, euler_sim, heun_sim] = [
            IntegrationMethod::RungeKutta4,
            IntegrationMethod::Euler,
            IntegrationMethod::Heun,
        ]
        .map(|m| self.simulate_if_inactive(m, comp_dt, reuse_active, custom_spec.as_ref()));

        let (any_diverged, euler_rk4, heun_rk4, euler_heun, snapshots) = {
            let rk4 = self.trajectory_view(IntegrationMethod::RungeKutta4, &rk4_sim);
            let euler = self.trajectory_view(IntegrationMethod::Euler, &euler_sim);
            let heun = self.trajectory_view(IntegrationMethod::Heun, &heun_sim);

            let any_diverged = rk4.diverged || euler.diverged || heun.diverged;
            let euler_rk4 = compute_series_discrepancy(
                euler.y_series,
                euler.v_series,
                rk4.y_series,
                rk4.v_series,
            );
            let heun_rk4 = compute_series_discrepancy(
                heun.y_series,
                heun.v_series,
                rk4.y_series,
                rk4.v_series,
            );
            let euler_heun = compute_series_discrepancy(
                euler.y_series,
                euler.v_series,
                heun.y_series,
                heun.v_series,
            );
            let snapshots = self.build_comparison_snapshot_series(&[rk4, euler, heun]);
            (any_diverged, euler_rk4, heun_rk4, euler_heun, snapshots)
        };

        if !self.has_custom_input_error() && any_diverged {
            self.diverged = true;
        }
        self.discrepancy_euler_rk4 = euler_rk4;
        self.discrepancy_heun_rk4 = heun_rk4;
        self.discrepancy_euler_heun = euler_heun;
        for (label, series) in snapshots {
            self.comparison_manager.add_snapshot(label, "", series);
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
            recovery::render_recovery_ui_in_place(ui, self, changed);
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
        if self
            .series_cache
            .matches_series(&self.time_series, &self.y_series, &self.v_series)
        {
            let (py, pv) = self.series_cache.shared_points();
            return (py.to_vec(), pv.to_vec());
        }
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
        let (py, pv) =
            if self
                .series_cache
                .matches_series(&self.time_series, &self.y_series, &self.v_series)
            {
                self.series_cache.shared_points()
            } else {
                let (y, v) = self.collect_full_xy_pairs();
                (y.into(), v.into())
            };
        let mut series = vec![TraceSeries::new(t1_lbl, py)];
        if self.has_second_state() && !pv.is_empty() {
            series.push(TraceSeries::new(t2_lbl, pv));
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
