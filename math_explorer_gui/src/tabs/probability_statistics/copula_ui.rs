#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, Fréchet-Hoeffding envelope plots, conditional SGP curve, and
//! deferred exports for [`CopulaDependenceTool`].

use super::copula_tool::CopulaDependenceTool;
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{ExportPrimitive, Legend, Line, Plot, PlotPoints, VLine};

impl CopulaDependenceTool {
    fn show_copula_structure_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let mut dim = self.config.dimension;
        if ui
            .add(egui::Slider::new(&mut dim, 2..=5).text("Dimension d"))
            .accessible_hover_text("Multivariate Gaussian copula dimension (2D exact Plackett, 3D-5D Genz QMC)")
            .changed()
        {
            self.config.dimension = dim;
            changed = true;
        }
        let mut rho = self.config.correlation_rho;
        if ui
            .add(egui::Slider::new(&mut rho, -0.48..=0.92).text("Equicorrelation rho"))
            .accessible_hover_text("Off-diagonal correlation rho validated via Cholesky factorization")
            .changed()
        {
            self.config.correlation_rho = rho;
            changed = true;
        }
        let mut u1 = self.config.marginal_u1;
        if ui
            .add(egui::Slider::new(&mut u1, 0.05..=0.95).text("Marginal u1"))
            .accessible_hover_text("First marginal cumulative probability u1 (baseline percentile)")
            .changed()
        {
            self.config.marginal_u1 = u1;
            changed = true;
        }
        let mut u2 = self.config.marginal_u2;
        if ui
            .add(egui::Slider::new(&mut u2, 0.05..=0.95).text("Marginal u2"))
            .accessible_hover_text("Second marginal cumulative probability u2 (target percentile)")
            .changed()
        {
            self.config.marginal_u2 = u2;
            changed = true;
        }
        changed
    }

    fn show_copula_sliders(&mut self, ui: &mut egui::Ui) {
        let mut changed = self.show_copula_structure_sliders(ui);
        let mut qmc = self.config.qmc_samples;
        if ui
            .add(egui::Slider::new(&mut qmc, 64..=1_024).text("QMC Pairs N"))
            .accessible_hover_text("Antithetic Genz separation-of-variables QMC sample pairs")
            .changed()
        {
            self.config.qmc_samples = qmc;
            changed = true;
        }
        let mut zip_p = self.config.zip_zero_prob;
        if ui
            .add(egui::Slider::new(&mut zip_p, 0.05..=0.85).text("ZIP Zero Prob pi"))
            .accessible_hover_text("Structural zero-inflation rate for companion EM count regression")
            .changed()
        {
            self.config.zip_zero_prob = zip_p;
            changed = true;
        }
        let mut seed = self.config.seed;
        if ui
            .add(egui::Slider::new(&mut seed, 1..=1_000).text("RNG Seed"))
            .accessible_hover_text("Deterministic OxidizeRng seed for QMC shifts and ZIP data")
            .changed()
        {
            self.config.seed = seed;
            changed = true;
        }
        if changed {
            self.recompute();
            self.bump_revision();
        }
    }

    fn show_copula_diagnostics(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let s = &self.snapshot;
        ui.group(|ui| {
            ui.label(egui::RichText::new("Cholesky & Fréchet-Hoeffding Audit").strong());
            match &s.cholesky_error {
                None => {
                    ui.colored_label(
                        palette.success,
                        format!(
                            "SPD Verified (d = {}, rho = {:.3})",
                            self.config.dimension, self.config.correlation_rho
                        ),
                    );
                }
                Some(msg) => {
                    ui.colored_label(palette.error, format!("Rejected: {msg}"));
                }
            }
            if let Some(joint) = s.active_joint_cdf {
                ui.colored_label(palette.accent, format!("Copula Joint CDF C_R(u): {joint:.5}"));
            }
            ui.label(format!("Independence prod u_i:   {:.5}", s.independence_cdf));
            ui.label(format!(
                "Fréchet Bounds [W, M]:   [{:.4}, {:.4}]",
                s.frechet_bounds.0, s.frechet_bounds.1
            ));
        });
        ui.add_space(4.0);
        ui.group(|ui| {
            ui.label(egui::RichText::new("ZIP Covariate EM Regression").strong());
            ui.label(&s.zip_em_summary);
        });
    }

    pub(crate) fn render_workbench(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "copula_lab_controls");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Multivariate Copula Lab");
                ui.separator();
                self.show_copula_sliders(ui);
                ui.separator();
                self.show_copula_diagnostics(ui);
            });
        });

        let bottom_id = crate::framework::scoped_panel_id(parent_ui, "copula_sgp_panel");
        egui::TopBottomPanel::bottom(bottom_id)
            .resizable(true)
            .default_height(205.0)
            .min_height(135.0)
            .show_inside(parent_ui, |ui| {
                self.render_sgp_plot(ui);
            });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_envelope_plot(ui);
        });
    }

    fn render_envelope_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(
            egui::RichText::new(format!(
                "{}D Gaussian Copula Joint CDF vs Independence & Fréchet-Hoeffding Bounds",
                self.config.dimension
            ))
            .strong(),
        );
        let plot_id = crate::framework::scoped_panel_id(ui, "copula_envelope_plot");
        Plot::new(plot_id)
            .legend(Legend::default())
            .x_axis_label("Equicorrelation rho")
            .y_axis_label("Joint Cumulative Probability C_R(u)")
            .include_y(0.0)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Gaussian Copula C_R(u)", PlotPoints::from(self.snapshot.sweep_copula.clone()))
                        .color(palette.accent)
                        .width(2.5_f32),
                );
                plot_ui.line(
                    Line::new("Independence prod u_i", PlotPoints::from(self.snapshot.sweep_independence.clone()))
                        .color(palette.info)
                        .width(1.8_f32),
                );
                plot_ui.line(
                    Line::new("Fréchet Lower Bound W(u)", PlotPoints::from(self.snapshot.sweep_frechet_low.clone()))
                        .color(palette.warning),
                );
                plot_ui.line(
                    Line::new("Fréchet Upper Bound M(u)", PlotPoints::from(self.snapshot.sweep_frechet_high.clone()))
                        .color(palette.success),
                );
                plot_ui.vline(
                    VLine::new("Selected rho", self.config.correlation_rho).color(palette.accent),
                );
            });
    }

    fn render_sgp_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(
            egui::RichText::new(format!(
                "Student Growth Percentile Conditioning P(U2 <= u2 | U1 = {:.2})",
                self.config.marginal_u1
            ))
            .strong(),
        );
        let plot_id = crate::framework::scoped_panel_id(ui, "copula_sgp_plot");
        Plot::new(plot_id)
            .legend(Legend::default())
            .x_axis_label("Follow-up Quantile u2")
            .y_axis_label("Conditional Growth Percentile")
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Conditional SGP Curve",
                        PlotPoints::from(self.snapshot.sgp_conditional_curve.clone()),
                    )
                    .color(palette.success)
                    .width(2.2_f32),
                );
            });
    }

    pub(crate) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            (
                "Gaussian Copula Joint CDF".to_string(),
                self.snapshot.sweep_copula.clone(),
            ),
            (
                "Independence Baseline".to_string(),
                self.snapshot.sweep_independence.clone(),
            ),
            (
                "Conditional SGP Curve".to_string(),
                self.snapshot.sgp_conditional_curve.clone(),
            ),
        ]
    }

    pub(crate) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::SemanticPalette::from_visuals(&egui::Visuals::dark());
        vec![
            ExportPrimitive::Polyline {
                name: "Gaussian Copula Joint CDF".to_string(),
                points: self.snapshot.sweep_copula.clone(),
                color: palette.accent,
                stroke_width: 2.4,
                closed: false,
                fill: None,
            },
            ExportPrimitive::Polyline {
                name: "Conditional SGP Curve".to_string(),
                points: self.snapshot.sgp_conditional_curve.clone(),
                color: palette.success,
                stroke_width: 2.0,
                closed: false,
                fill: None,
            },
        ]
    }
}

// [cite:clinical_trials_statistics]
