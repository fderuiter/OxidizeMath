#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::recovery;
use super::OdeSolverTool;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
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

    fn supports_empirical_csv(&self) -> bool {
        true
    }

    fn ingest_empirical_dataset(
        &mut self,
        dataset: crate::framework::empirical_csv::EmpiricalCsvTable,
    ) -> Result<String, String> {
        self.bump_revision();
        Ok(self.empirical_csv.set_table(dataset))
    }

    fn notify_empirical_csv_error(&mut self, error: String) {
        self.empirical_csv.record_error(None, error);
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
}

impl OdeSolverTool {
    pub(super) fn render_discrepancy_panel(&self, ui: &mut egui::Ui) {
        ui.add_space(10.0);
        ui.heading("Discrepancy Metrics");
        ui.group(|ui| {
            render_copyable_metric(
                ui,
                "Euler vs RK4 (Max)",
                &format!("{:.4e}", self.discrepancy_euler_rk4.max_error),
            );
            render_copyable_metric(
                ui,
                "Euler vs RK4 (RMS)",
                &format!("{:.4e}", self.discrepancy_euler_rk4.rms_error),
            );
            render_copyable_metric(
                ui,
                "Heun vs RK4 (Max)",
                &format!("{:.4e}", self.discrepancy_heun_rk4.max_error),
            );
            render_copyable_metric(
                ui,
                "Heun vs RK4 (RMS)",
                &format!("{:.4e}", self.discrepancy_heun_rk4.rms_error),
            );
            render_copyable_metric(
                ui,
                "Euler vs Heun (Max)",
                &format!("{:.4e}", self.discrepancy_euler_heun.max_error),
            );
            render_copyable_metric(
                ui,
                "Euler vs Heun (RMS)",
                &format!("{:.4e}", self.discrepancy_euler_heun.rms_error),
            );
        });
    }

    pub(super) fn render_plot_canvas(&mut self, ui: &mut egui::Ui, changed: &mut bool) {
        ui.heading("Solution Trajectory");

        if FiniteStateGuard::diverged(self) {
            let mut scratch_y = Vec::new();
            let mut scratch_v = Vec::new();
            recovery::render_recovery_ui(ui, self, changed, &mut scratch_y, &mut scratch_v);
        }

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let line_y = Line::new("y(t)", PlotPoints::from_shared(Arc::clone(&self.display_y)))
            .color(palette.series_color(0));
        let has_v = self.has_second_state();
        let v_label = self.second_state_label();

        Plot::new("ode_plot")
            .view_aspect(2.0)
            .x_axis_label("t \\text{ (Time)}")
            .y_axis_label("y(t)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                if !self.comparison_mode {
                    plot_ui.line(line_y);
                    if has_v && !self.display_v.is_empty() {
                        let line_v =
                            Line::new(v_label, PlotPoints::from_shared(Arc::clone(&self.display_v)))
                                .color(palette.series_color(1));
                        plot_ui.line(line_v);
                    }
                } else {
                    self.comparison_manager.render_on_plot(plot_ui);
                }
                if self.show_annotations && !FiniteStateGuard::diverged(self) {
                    super::annotations::render_ode_callouts(plot_ui, self);
                }
                self.render_empirical_overlay(plot_ui, palette.series_color(2));
                self.snapshot_manager.render_on_plot(plot_ui);
                self.callout_manager.render_on_plot(plot_ui);
            });
    }
}
