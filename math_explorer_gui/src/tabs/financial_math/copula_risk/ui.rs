//! Interactive UI controls, correlation matrix editor, and plot views for [`super::CopulaRiskTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::engine::{decimal_to_american_odds, pair_flat_index};
use super::{CopulaPlotView, CopulaRiskTool, CorrelationStructureMode};
use crate::accessibility::AccessibleHoverText;
use crate::framework::bounded_drag_value;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, MarkerShape, Plot, Points};
use math_explorer::pure_math::statistics::copula::MAX_COPULA_DIMENSION;

impl CopulaRiskTool {
    fn show_top_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Legs (2..=8):");
            let mut legs = self.num_legs;
            if ui
                .add(egui::DragValue::new(&mut legs).range(2..=MAX_COPULA_DIMENSION))
                .accessible_hover_text("Number of correlated parlay legs (2 to 8)")
                .changed()
            {
                self.resize_legs(legs);
                changed = true;
            }
            ui.separator();
            ui.label("Structure:");
            changed |= ui
                .radio_value(
                    &mut self.mode,
                    CorrelationStructureMode::Equicorrelated,
                    "Equicorrelated",
                )
                .changed();
            changed |= ui
                .radio_value(
                    &mut self.mode,
                    CorrelationStructureMode::PairwiseMatrix,
                    "Pairwise Matrix",
                )
                .changed();
            ui.separator();
            ui.label("Book Vig (%):");
            changed |= ui
                .add(bounded_drag_value(&mut self.book_vig_pct, 0.0..=25.0, 0.1).suffix("%"))
                .accessible_hover_text("Sportsbook overround margin applied to fair odds")
                .changed();
            ui.label("RNG Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.rng_seed).speed(1.0))
                .accessible_hover_text("Deterministic OxidizeRng seed for Cholesky sampling")
                .changed();
        });

        if self.mode == CorrelationStructureMode::Equicorrelated {
            let min_rho = (-1.0 / (self.num_legs.saturating_sub(1) as f64) + 0.02).max(-0.95);
            if self.equicorrelation < min_rho {
                self.equicorrelation = min_rho;
                changed = true;
            }
            ui.horizontal(|ui| {
                ui.label("Pairwise Equicorrelation (ρ):");
                changed |= ui
                    .add(egui::Slider::new(&mut self.equicorrelation, min_rho..=0.95))
                    .accessible_hover_text("Common off-diagonal correlation across all legs")
                    .changed();
            });
        }
        changed
    }

    fn show_leg_editor(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.collapsing("Leg Marginals & Labels", |ui| {
            egui::Grid::new("copula_legs_grid")
                .num_columns(3)
                .striped(true)
                .spacing([18.0, 6.0])
                .show(ui, |ui| {
                    ui.strong("Leg");
                    ui.strong("Label");
                    ui.strong("Marginal Probability");
                    ui.end_row();
                    for idx in 0..self.num_legs {
                        ui.label(format!("#{}", idx + 1));
                        if ui.text_edit_singleline(&mut self.leg_labels[idx]).changed() {
                            if self.leg_labels[idx].len() > 64 {
                                self.leg_labels[idx].truncate(64);
                            }
                            changed = true;
                        }
                        changed |= ui
                            .add(bounded_drag_value(
                                &mut self.marginals[idx],
                                0.01..=0.99,
                                0.005,
                            ))
                            .changed();
                        ui.end_row();
                    }
                });
        });
        changed
    }

    fn show_pairwise_matrix_editor(&mut self, ui: &mut egui::Ui) -> bool {
        if self.mode != CorrelationStructureMode::PairwiseMatrix {
            return false;
        }
        let mut changed = false;
        ui.collapsing("Symmetric Pairwise Correlation Matrix R", |ui| {
            egui::Grid::new("copula_pairwise_grid")
                .striped(true)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    ui.label("");
                    for j in 0..self.num_legs {
                        ui.strong(format!("L{}", j + 1));
                    }
                    ui.end_row();
                    for i in 0..self.num_legs {
                        ui.strong(format!("L{}", i + 1));
                        for j in 0..self.num_legs {
                            if i == j {
                                ui.label("1.00");
                            } else if i < j {
                                let flat = pair_flat_index(self.num_legs, i, j);
                                changed |= ui
                                    .add(bounded_drag_value(
                                        &mut self.pairwise_upper[flat],
                                        -0.95..=0.95,
                                        0.01,
                                    ))
                                    .changed();
                            } else {
                                let flat = pair_flat_index(self.num_legs, i, j);
                                ui.label(format!("{:+.2}", self.pairwise_upper[flat]));
                            }
                        }
                        ui.end_row();
                    }
                });
        });
        changed
    }

    fn show_pricing_banner(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        if let Some(ref err) = self.calc_error {
            ui.colored_label(palette.error, err);
        } else if let Some(jp) = self.joint_probability {
            let fair_dec = if jp > 1e-6 { 1.0 / jp } else { 999_999.0 };
            let naive_dec = if self.naive_probability > 1e-6 {
                1.0 / self.naive_probability
            } else {
                999_999.0
            };
            let amer = decimal_to_american_odds(fair_dec);
            let edge_delta = (jp - self.naive_probability) * 100.0;
            ui.horizontal_wrapped(|ui| {
                ui.strong(format!(
                    "Copula Joint P: {:.2}% (Fair {:.2}x / {:+.0})",
                    jp * 100.0,
                    fair_dec,
                    amer
                ));
                ui.separator();
                ui.label(format!(
                    "Naive Independent: {:.2}% ({:.2}x)",
                    self.naive_probability * 100.0,
                    naive_dec
                ));
                ui.separator();
                let color = if edge_delta >= 0.0 {
                    palette.success
                } else {
                    palette.warning
                };
                ui.colored_label(color, format!("Correlation Shift: {:+.2} pp", edge_delta));
            });
        }
    }

    fn show_summary_and_exports(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        self.show_pricing_banner(ui, &palette);
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📋 Copy CSV Sheet")
                .accessible_hover_text("Copy fair-odds CSV pricing sheet to clipboard")
                .clicked()
            {
                let csv = self.export_pricing_csv();
                self.copy_payload(ui.ctx(), "copula CSV pricing sheet", csv);
            }
            if ui
                .button("💾 Save CSV")
                .accessible_hover_text("Download or save CSV fair-odds pricing sheet")
                .clicked()
            {
                self.trigger_save_csv();
            }
            if ui
                .button("📋 Copy JSON Sheet")
                .accessible_hover_text("Copy JSON fair-odds pricing sheet to clipboard")
                .clicked()
            {
                if let Ok(json) = self.export_pricing_json() {
                    self.copy_payload(ui.ctx(), "copula JSON pricing sheet", json);
                }
            }
            if ui
                .button("💾 Save JSON")
                .accessible_hover_text("Download or save JSON fair-odds pricing sheet")
                .clicked()
            {
                self.trigger_save_json();
            }
            if ui
                .button("🦀 Copy Rust")
                .accessible_hover_text("Copy calibrated Rust sgp_joint_probability snippet")
                .clicked()
            {
                let snippet = self.rust_snippet();
                self.copy_payload(ui.ctx(), "Rust copula snippet", snippet);
            }
        });
        if let Some((ref msg, is_err)) = self.io_state.status {
            let color = if is_err { palette.error } else { palette.info };
            ui.colored_label(color, msg);
        }
    }

    fn show_plot_view_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label("Plot View:");
            changed |= ui
                .selectable_value(
                    &mut self.plot_view,
                    CopulaPlotView::CorrelationSweep,
                    "Correlation Sweep",
                )
                .changed();
            changed |= ui
                .selectable_value(
                    &mut self.plot_view,
                    CopulaPlotView::BivariateContours,
                    "Bivariate Copula Contours & Cloud",
                )
                .changed();
            if self.plot_view == CopulaPlotView::BivariateContours && self.num_legs > 2 {
                ui.separator();
                ui.label("Pair:");
                for idx in 0..self.num_legs {
                    if idx != self.contour_leg_b
                        && ui
                            .selectable_label(self.contour_leg_a == idx, format!("A:L{}", idx + 1))
                            .clicked()
                    {
                        self.contour_leg_a = idx;
                        changed = true;
                    }
                }
                for idx in 0..self.num_legs {
                    if idx != self.contour_leg_a
                        && ui
                            .selectable_label(self.contour_leg_b == idx, format!("B:L{}", idx + 1))
                            .clicked()
                    {
                        self.contour_leg_b = idx;
                        changed = true;
                    }
                }
            }
        });
        changed
    }

    fn show_sweep_canvas(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        Plot::new("copula_sweep_plot")
            .view_aspect(2.1)
            .legend(Legend::default())
            .x_axis_label("Equicorrelation Coefficient (ρ)")
            .y_axis_label("Joint Parlay Probability C(u₁..u_d)")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Naive Independent Product", &self.sweep_naive_points[..])
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 6.0 }),
                );
                if !self.sweep_copula_points.is_empty() {
                    plot_ui.line(
                        Line::new(
                            "Cholesky Gaussian Copula Joint P",
                            &self.sweep_copula_points[..],
                        )
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                    );
                }
                if !self.active_marker_points.is_empty() {
                    plot_ui.points(
                        Points::new("Active Configuration", &self.active_marker_points[..])
                            .color(palette.series_color(3))
                            .radius(6.0_f32)
                            .shape(MarkerShape::Diamond),
                    );
                }
            });
    }

    fn show_contours_canvas(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        Plot::new("copula_contours_plot")
            .view_aspect(1.6)
            .legend(Legend::default())
            .x_axis_label(format!("Leg {} Uniform Margin u", self.contour_leg_a + 1))
            .y_axis_label(format!("Leg {} Uniform Margin v", self.contour_leg_b + 1))
            .show(ui, |plot_ui| {
                if !self.sampled_cloud_points.is_empty() {
                    plot_ui.points(
                        Points::new(
                            "Cholesky Correlated Uniform Draws",
                            &self.sampled_cloud_points[..],
                        )
                        .color(palette.series_color(1))
                        .radius(2.8_f32)
                        .shape(MarkerShape::Circle),
                    );
                }
                for (idx, (label, pts)) in self.contour_curves.iter().enumerate() {
                    plot_ui.line(
                        Line::new(label.as_str(), &pts[..])
                            .color(palette.series_color(idx + 2))
                            .width(1.8_f32),
                    );
                }
            });
    }

    fn show_copula_plot(&mut self, ui: &mut egui::Ui) -> bool {
        let changed = self.show_plot_view_selector(ui);
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        match self.plot_view {
            CopulaPlotView::CorrelationSweep => self.show_sweep_canvas(ui, &palette),
            CopulaPlotView::BivariateContours => self.show_contours_canvas(ui, &palette),
        }
        changed
    }

    pub(super) fn render_workbench_ui(&mut self, ui: &mut egui::Ui) {
        self.poll_io(ui.ctx());
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Multivariate Copula Tail Risk & SGP Pricing");
            ui.label(
                "Price 2 to 8 correlated legs via positive-definite Cholesky Gaussian copulas and inspect correlation sweeps, contours, and fair-odds schedules.",
            );
            ui.separator();

            let mut dirty = self.show_top_controls(ui);
            dirty |= self.show_leg_editor(ui);
            dirty |= self.show_pairwise_matrix_editor(ui);
            if dirty {
                self.bump_revision();
                self.recalculate();
            }
            ui.separator();
            self.show_summary_and_exports(ui);
            ui.separator();
            if self.show_copula_plot(ui) {
                self.bump_revision();
                self.recalculate();
            }
        });
    }
}

// [cite:clinical_trials_statistics]
