#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::engine::{MAX_OU_PATHS, MAX_OU_STEPS, MAX_SCENARIO_LABEL_BYTES};
use super::{CorrelatedRiskScenarioTool, RiskAnalysisMode};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, Points};

impl CorrelatedRiskScenarioTool {
    pub(super) fn show_controls_panel(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Correlated Risk & OU");
            ui.horizontal(|ui| {
                ui.label("Scenario:");
                if ui
                    .text_edit_singleline(&mut self.scenario_label)
                    .accessible_hover_text("Scenario label included in CSV and Markdown exports")
                    .changed()
                {
                    if self.scenario_label.len() > MAX_SCENARIO_LABEL_BYTES {
                        self.scenario_label.truncate(MAX_SCENARIO_LABEL_BYTES);
                    }
                    if self.scenario_label.trim().is_empty() {
                        self.scenario_label = String::from("Scenario");
                    }
                }
            });
            ui.separator();
            changed |= self.show_copula_controls(ui);
            ui.separator();
            changed |= self.show_ou_controls(ui);
            ui.separator();
            self.show_export_and_reset_controls(ui);
        });
        if changed {
            self.recalculate_if_needed();
        }
    }

    fn show_copula_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut c = false;
        ui.strong("Bivariate Copula / Same-Game Parlay");
        c |= ui
            .add(egui::Slider::new(&mut self.prob_a, 0.01..=0.99).text("Event A P(A)"))
            .accessible_hover_text("Marginal probability of Event A in [0.01, 0.99]")
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.prob_b, 0.01..=0.99).text("Event B P(B)"))
            .accessible_hover_text("Marginal probability of Event B in [0.01, 0.99]")
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.rho, -0.99..=0.99).text("Correlation ρ"))
            .accessible_hover_text("Latent Gaussian correlation rho in [-0.99, 0.99]")
            .changed();
        c
    }

    fn show_ou_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut c = false;
        ui.strong("Ornstein-Uhlenbeck Mean Reversion");
        c |= ui
            .add(egui::Slider::new(&mut self.ou_mu, -5.0..=5.0).text("Equilibrium μ"))
            .accessible_hover_text("Long-term mean equilibrium level mu")
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.ou_theta, 0.05..=10.0).text("Reversion θ"))
            .accessible_hover_text("Mean reversion speed theta (> 0)")
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.ou_sigma, 0.0..=2.5).text("Volatility σ"))
            .accessible_hover_text("Diffusion volatility sigma (>= 0)")
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.ou_x0, -5.0..=5.0).text("Initial X₀"))
            .accessible_hover_text("Starting performance or state X_0")
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.ou_dt, 0.005..=0.25).text("Step Δt"))
            .accessible_hover_text("Euler-Maruyama integration step size dt")
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.ou_steps, 10..=MAX_OU_STEPS).text("Steps N"))
            .accessible_hover_text("Number of Euler-Maruyama steps (max 2,000)")
            .changed();
        c |= ui
            .add(egui::Slider::new(&mut self.ou_paths, 1..=MAX_OU_PATHS).text("MC Paths"))
            .accessible_hover_text("Number of seeded Monte Carlo paths (max 200)")
            .changed();
        ui.horizontal(|ui| {
            ui.label("Seed:");
            c |= ui
                .add(egui::DragValue::new(&mut self.seed).speed(1))
                .accessible_hover_text("Deterministic OxidizeRng seed")
                .changed();
        });
        c
    }

    fn show_export_and_reset_controls(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📋 Copy CSV")
                .accessible_hover_text("Copy formula-safe CSV scenario series to clipboard")
                .clicked()
            {
                self.copy_scenario_to_clipboard(ui.ctx(), false);
            }
            if ui
                .button("💾 Save CSV")
                .accessible_hover_text("Save formula-safe CSV scenario series")
                .clicked()
            {
                self.save_scenario_file(false);
            }
            if ui
                .button("📋 Copy MD")
                .accessible_hover_text("Copy Markdown scenario summary to clipboard")
                .clicked()
            {
                self.copy_scenario_to_clipboard(ui.ctx(), true);
            }
            if ui
                .button("💾 Save MD")
                .accessible_hover_text("Save Markdown scenario summary")
                .clicked()
            {
                self.save_scenario_file(true);
            }
            if ui
                .button("↺ Reset")
                .accessible_hover_text("Reset parameters to baseline scenario")
                .clicked()
            {
                self.reset_defaults();
            }
        });
        if let Some((ref msg, is_err)) = self.io_status {
            let color = if is_err { palette.error } else { palette.info };
            ui.colored_label(color, msg);
        }
    }

    pub(super) fn show_main_workbench(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.mode,
                RiskAnalysisMode::CopulaCorrelation,
                "🔗 Bivariate Copula Correlation Sweep",
            )
            .accessible_hover_text(
                "Inspect joint event probability across correlation rho in [-0.99, 0.99]",
            );
            ui.selectable_value(
                &mut self.mode,
                RiskAnalysisMode::OrnsteinUhlenbeck,
                "📈 Ornstein-Uhlenbeck Monte Carlo Paths",
            )
            .accessible_hover_text(
                "Inspect seeded Euler-Maruyama trajectories around equilibrium mu",
            );
        });
        ui.separator();
        self.show_kpi_summary_strip(ui, &palette);
        ui.separator();

        match self.mode {
            RiskAnalysisMode::CopulaCorrelation => self.show_copula_plot(ui, &palette),
            RiskAnalysisMode::OrnsteinUhlenbeck => self.show_ou_plot(ui, &palette),
        }
    }

    fn show_kpi_summary_strip(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        let delta = self.active_sgp_prob - self.naive_prob;
        let fair_odds = if self.active_sgp_prob > 1e-6 {
            1.0 / self.active_sgp_prob
        } else {
            f64::INFINITY
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("Naive P(A)·P(B): {:.4}", self.naive_prob));
            ui.separator();
            ui.strong(format!(
                "Copula P(A ∩ B; ρ={:+.2}): {:.4}",
                self.rho, self.active_sgp_prob
            ));
            let delta_color = if delta >= 0.0 {
                palette.success
            } else {
                palette.warning
            };
            ui.colored_label(
                delta_color,
                format!("(Δ {:+.4}, Fair {:.2}x)", delta, fair_odds),
            );
            if let Some(st) = self.ou_stats {
                ui.separator();
                ui.label(format!(
                    "OU [{}]: E[X_T]={:.3} ± {:.3}",
                    self.ou_momentum.description(),
                    st.mean_final,
                    st.std_final
                ));
            }
        });
        if let Some(ref err) = self.copula_error {
            ui.colored_label(palette.error, err);
        }
        if let Some(ref err) = self.ou_error {
            ui.colored_label(palette.error, err);
        }
    }

    fn show_copula_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        Plot::new("correlated_risk_copula_plot")
            .legend(Legend::default())
            .x_axis_label("Latent Normal Correlation (ρ)")
            .y_axis_label("Joint Event Probability P(A ∩ B)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Naive Independence P(A)·P(B)", &self.naive_line_points[..])
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 6.0 }),
                );
                plot_ui.line(
                    Line::new(
                        "Gaussian Copula Joint P(A ∩ B; ρ)",
                        &self.copula_curve_points[..],
                    )
                    .color(palette.series_color(0))
                    .width(2.2_f32),
                );
                if !self.active_rho_marker.is_empty() {
                    plot_ui.points(
                        Points::new("Active SGP Scenario", &self.active_rho_marker[..])
                            .color(palette.series_color(3))
                            .radius(6.0_f32)
                            .shape(MarkerShape::Circle),
                    );
                }
            });
    }

    fn show_ou_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        Plot::new("correlated_risk_ou_plot")
            .legend(Legend::default())
            .x_axis_label("Time (t)")
            .y_axis_label("Process State X(t)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Equilibrium Mean (μ)", &self.ou_equilibrium_line[..])
                        .color(palette.warning)
                        .width(2.0_f32)
                        .style(LineStyle::Dashed { length: 6.0 }),
                );
                for (idx, (label, pts)) in self.ou_sample_paths.iter().enumerate() {
                    let c = palette.series_color(idx + 1).gamma_multiply(0.55);
                    let name = if idx < 3 { label.as_str() } else { "" };
                    plot_ui.line(Line::new(name, &pts[..]).color(c).width(1.1_f32));
                }
                plot_ui.line(
                    Line::new("Ensemble Mean Trajectory", &self.ou_mean_path[..])
                        .color(palette.series_color(0))
                        .width(2.6_f32),
                );
            });
    }
}
