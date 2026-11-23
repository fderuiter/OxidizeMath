#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Singular-value decay plot, normal-equation breakdown thresholds, and 5-solver
//! relative error comparison charts for [`super::state::ConditioningWorkbenchTool`].

use super::state::ConditioningWorkbenchTool;
use eframe::egui;
use egui_plot::{Bar, BarChart, ExportPrimitive, HLine, Legend, Line, Plot, PlotPoints, Points};

fn safe_log10(val: f64) -> f64 {
    if val.is_finite() && val > 0.0 {
        val.log10().clamp(-16.5, 4.0)
    } else {
        -16.5
    }
}

impl ConditioningWorkbenchTool {
    fn singular_value_points(&self) -> Vec<[f64; 2]> {
        self.snapshot
            .singular_values
            .iter()
            .enumerate()
            .map(|(idx, &sigma)| [(idx + 1) as f64, safe_log10(sigma)])
            .collect()
    }

    pub(crate) fn show_singular_value_panel(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(
            egui::RichText::new(format!(
                "{} ({}×{}) — Singular Value Spectrum log₁₀(σᵢ) & Breakdown Thresholds",
                self.config.family.label(),
                self.config.rows,
                self.config.cols
            ))
            .strong(),
        );

        let sv_pts = self.singular_value_points();
        let sigma1 = self
            .snapshot
            .singular_values
            .first()
            .copied()
            .unwrap_or(1.0);
        let normal_eq_floor = safe_log10(sigma1 * f64::EPSILON.sqrt());
        let mach_floor = safe_log10(sigma1 * f64::EPSILON);

        let plot_id = crate::framework::scoped_panel_id(ui, "linalg_singular_values_plot");
        Plot::new(plot_id)
            .legend(Legend::default())
            .x_axis_label("Singular Value Index i")
            .y_axis_label("log10(σ_i)")
            .include_y(0.5)
            .include_y(-16.5)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("log10(σ_i)", PlotPoints::from(sv_pts.clone()))
                        .color(palette.info)
                        .width(2.2_f32),
                );
                plot_ui.points(
                    Points::new("σ_i Modes", PlotPoints::from(sv_pts))
                        .color(palette.accent)
                        .radius(4.5_f32),
                );
                plot_ui.hline(
                    HLine::new("Normal Eq Breakdown (√ε · σ₁)", normal_eq_floor)
                        .color(palette.warning)
                        .width(1.5_f32),
                );
                plot_ui.hline(
                    HLine::new("QR / SVD Precision Floor (ε · σ₁)", mach_floor)
                        .color(palette.error)
                        .width(1.5_f32),
                );
            });
    }

    fn render_solver_summary_strip(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            for m in &self.snapshot.solvers {
                let text = if m.succeeded {
                    format!(
                        "{}: err={:.2e} (min x={:.2})",
                        m.label, m.forward_rel_error, m.min_entry
                    )
                } else {
                    format!("{}: SINGULAR / FAILED", m.label)
                };
                let col = if !m.succeeded || m.forward_rel_error > 0.1 {
                    palette.error
                } else if m.forward_rel_error < 1e-6 {
                    palette.success
                } else {
                    palette.info
                };
                ui.colored_label(col, text);
                ui.separator();
            }
        });
    }

    pub(crate) fn show_solver_errors_panel(&self, parent_ui: &mut egui::Ui) {
        let bottom_id = crate::framework::scoped_panel_id(parent_ui, "linalg_solver_errors_panel");
        egui::TopBottomPanel::bottom(bottom_id)
            .resizable(true)
            .default_height(235.0)
            .min_height(160.0)
            .show_inside(parent_ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Solver Accuracy Comparison: log₁₀ Forward Error ‖x̂ - x*‖₂/‖x*‖₂ vs Residual ‖Ax̂ - b‖₂/‖b‖₂",
                    )
                    .strong(),
                );
                self.render_solver_summary_strip(ui);

                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                let mut fwd_bars = Vec::with_capacity(5);
                let mut res_bars = Vec::with_capacity(5);
                for (idx, m) in self.snapshot.solvers.iter().enumerate() {
                    let x = idx as f64;
                    let fill = if m.succeeded {
                        palette.info
                    } else {
                        palette.error
                    };
                    fwd_bars.push(
                        Bar::new(x - 0.18, safe_log10(m.forward_rel_error))
                            .name(format!("{} Forward", m.label))
                            .width(0.32)
                            .fill(fill),
                    );
                    res_bars.push(
                        Bar::new(x + 0.18, safe_log10(m.residual_rel_error))
                            .name(format!("{} Residual", m.label))
                            .width(0.32)
                            .fill(palette.accent),
                    );
                }

                let plot_id = crate::framework::scoped_panel_id(ui, "linalg_solver_errors_plot");
                Plot::new(plot_id)
                    .legend(Legend::default())
                    .x_axis_label("0: Normal Eq | 1: Householder QR | 2: Thin SVD | 3: Ridge | 4: NNLS")
                    .y_axis_label("log10 Relative Error (more negative = better)")
                    .include_y(0.5)
                    .include_y(-16.5)
                    .show(ui, |plot_ui| {
                        plot_ui.bar_chart(BarChart::new("log10 Forward Error", fwd_bars));
                        plot_ui.bar_chart(BarChart::new("log10 Residual Error", res_bars));
                    });
            });
    }

    pub(crate) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let fwd: Vec<[f64; 2]> = self
            .snapshot
            .solvers
            .iter()
            .enumerate()
            .map(|(i, m)| [i as f64, safe_log10(m.forward_rel_error)])
            .collect();
        let res: Vec<[f64; 2]> = self
            .snapshot
            .solvers
            .iter()
            .enumerate()
            .map(|(i, m)| [i as f64, safe_log10(m.residual_rel_error)])
            .collect();
        vec![
            (
                "Singular Values log10(sigma_i)".to_string(),
                self.singular_value_points(),
            ),
            ("Solver Forward Error log10".to_string(), fwd),
            ("Solver Residual Error log10".to_string(), res),
        ]
    }

    pub(crate) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::SemanticPalette::from_visuals(&egui::Visuals::dark());
        let mut prims = Vec::new();
        let sv_pts = self.singular_value_points();
        if sv_pts.len() >= 2 {
            prims.push(ExportPrimitive::Polyline {
                name: "Singular Values log10(sigma_i)".to_string(),
                points: sv_pts,
                color: palette.info,
                stroke_width: 2.2,
                closed: false,
                fill: None,
            });
        }
        for (idx, m) in self.snapshot.solvers.iter().enumerate() {
            let x = (idx + 1) as f64;
            let y = safe_log10(m.forward_rel_error);
            prims.push(ExportPrimitive::Bar {
                name: format!("{} log10 Forward Error", m.label),
                min: [x - 0.28, y.min(0.0)],
                max: [x + 0.28, y.max(0.0)],
                fill: if m.succeeded {
                    palette.accent
                } else {
                    palette.error
                },
                stroke_color: palette.info,
                stroke_width: 1.0,
            });
        }
        prims
    }
}

// [cite:advanced_linear_algebra]
