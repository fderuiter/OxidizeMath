//! UI controls, Efficient Frontier plot, optimal allocation bar chart, and exports for [`super::MarkowitzFrontierTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MarkowitzFrontierTool;
use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{
    Bar, BarChart, ExportPrimitive, HLine, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoint,
    Points,
};
use math_explorer::applied::portfolio::MAX_PORTFOLIO_ASSETS;

fn to_pairs(pts: &[PlotPoint]) -> Vec<[f64; 2]> {
    pts.iter().map(|p| [p.x, p.y]).collect()
}

fn upper_index(n: usize, r: usize, c: usize) -> usize {
    r * n - r * (r + 1) / 2 + (c - r - 1)
}

impl MarkowitzFrontierTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        egui::SidePanel::left("markowitz_controls")
            .resizable(true)
            .default_width(305.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if self.render_controls(ui) {
                        self.recompute();
                    }
                });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_summary_banner(ui, &palette);
                ui.add_space(6.0);
                self.render_frontier_plot(ui, &palette);
                ui.add_space(8.0);
                self.render_weights_plot(ui, &palette);
                ui.add_space(6.0);
                self.render_weights_table(ui);
            });
        });
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.heading("Portfolio Universe");
        ui.small(format!(
            "{} assets | solved in {:.3} ms",
            self.expected_returns.len(),
            self.last_compute_ms
        ));
        ui.separator();

        changed |= self.render_universe_buttons(ui);
        ui.separator();

        let n = self.expected_returns.len();
        let min_cap = (1.0 / (n as f64) + 1e-3).min(1.0);
        if self.max_asset_weight < min_cap {
            self.max_asset_weight = min_cap;
        }
        ui.label("Risk-Free Rate (r_f):");
        changed |= ui
            .add(egui::Slider::new(&mut self.risk_free_rate, 0.0..=0.15))
            .changed();
        ui.label("Max Per-Asset Weight Cap (u_i):");
        changed |= ui
            .add(egui::Slider::new(&mut self.max_asset_weight, min_cap..=1.0))
            .changed();

        ui.separator();
        ui.strong("Asset Moments (Annualized μ, σ)");
        changed |= self.render_asset_moment_rows(ui);
        ui.separator();
        changed |= self.render_correlation_controls(ui);
        changed
    }

    fn render_universe_buttons(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let n = self.expected_returns.len();
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(n < MAX_PORTFOLIO_ASSETS, egui::Button::new("+ Add Asset"))
                .clicked()
            {
                self.append_asset();
                changed = true;
            }
            if ui
                .add_enabled(n > 2, egui::Button::new("− Remove Asset"))
                .clicked()
            {
                self.pop_asset();
                changed = true;
            }
            if ui.button("8-Asset Endowment").clicked() {
                self.apply_eight_asset_endowment();
                changed = true;
            }
        });
        changed
    }

    fn render_asset_moment_rows(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("markowitz_asset_moments_grid")
            .num_columns(3)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.small("Asset");
                ui.small("E[R]");
                ui.small("Vol σ");
                ui.end_row();
                for i in 0..self.expected_returns.len() {
                    ui.label(&self.asset_names[i]);
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.expected_returns[i])
                                .speed(0.002)
                                .range(-0.25..=0.50),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.volatilities[i])
                                .speed(0.002)
                                .range(0.01..=0.90),
                        )
                        .changed();
                    ui.end_row();
                }
            });
        changed
    }

    fn render_correlation_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let n = self.expected_returns.len();
        egui::CollapsingHeader::new("Pairwise Correlations (ρ_ij)")
            .default_open(n <= 5)
            .show(ui, |ui| {
                if self.last_error.is_some() && ui.button("Regularize SPD Matrix").clicked() {
                    self.regularize_correlations();
                    changed = true;
                }
                for r in 0..n {
                    for c in (r + 1)..n {
                        let idx = upper_index(n, r, c);
                        ui.horizontal(|ui| {
                            ui.small(format!(
                                "{} × {}:",
                                self.asset_names[r], self.asset_names[c]
                            ));
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut self.upper_correlations[idx])
                                        .speed(0.01)
                                        .range(-0.95..=0.95),
                                )
                                .changed();
                        });
                    }
                }
            });
        changed
    }

    fn render_summary_banner(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        if let Some(err) = self.last_error {
            ui.colored_label(
                palette.error,
                format!("Covariance constraint diagnostic: {err} (shrink correlations to restore positive-definiteness)."),
            );
            return;
        }
        ui.horizontal_wrapped(|ui| {
            if let Some(gmv) = self.con_gmv {
                ui.colored_label(
                    palette.series_color(0),
                    format!(
                        "Constrained GMV: E[R] {:.2}%, σ {:.2}%, Sharpe {:+.2}",
                        gmv.expected_return * 100.0,
                        gmv.volatility * 100.0,
                        gmv.sharpe_ratio
                    ),
                );
                ui.separator();
            }
            if let Some(tan) = self.con_tangency {
                ui.colored_label(
                    palette.success,
                    format!(
                        "Constrained Tangency: E[R] {:.2}%, σ {:.2}%, Sharpe {:+.2}",
                        tan.expected_return * 100.0,
                        tan.volatility * 100.0,
                        tan.sharpe_ratio
                    ),
                );
                ui.separator();
            }
            if let Some(unc) = self.unc_tangency {
                ui.colored_label(
                    palette.series_color(1),
                    format!("Unconstrained Sharpe: {:+.2}", unc.sharpe_ratio),
                );
            }
        });
    }

    fn render_frontier_plot(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.strong("Mean-Variance Efficient Frontier & Capital Market Line");
        let gmv_pt = self.con_gmv.map(|a| {
            [PlotPoint::new(
                a.volatility * 100.0,
                a.expected_return * 100.0,
            )]
        });
        let tan_pt = self.con_tangency.map(|a| {
            [PlotPoint::new(
                a.volatility * 100.0,
                a.expected_return * 100.0,
            )]
        });
        let resp = Plot::new("markowitz_efficient_frontier_plot")
            .height(255.0)
            .x_axis_label("Annualized Portfolio Volatility σ (%)")
            .y_axis_label("Annualized Expected Return E[R] (%)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.draw_frontier_series(plot_ui, palette, &gmv_pt, &tan_pt);
            });
        let desc = self.con_tangency.map_or_else(
            || "Markowitz efficient frontier".to_string(),
            |t| {
                format!(
                    "{} assets; constrained tangency Sharpe {:.2} at volatility {:.2}%",
                    self.expected_returns.len(),
                    t.sharpe_ratio,
                    t.volatility * 100.0
                )
            },
        );
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Markowitz Unconstrained and Long-Only Efficient Frontier",
            Some(&desc),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn draw_frontier_series<'a>(
        &'a mut self,
        plot_ui: &mut egui_plot::PlotUi<'a>,
        palette: &SemanticPalette,
        gmv_pt: &'a Option<[PlotPoint; 1]>,
        tan_pt: &'a Option<[PlotPoint; 1]>,
    ) {
        plot_ui.line(
            Line::new(
                "Unconstrained Lagrangian Frontier",
                &self.unc_frontier_pts[..],
            )
            .color(palette.series_color(1))
            .style(LineStyle::Dashed { length: 6.0 }),
        );
        plot_ui.line(
            Line::new(
                "Constrained Active-Set Frontier",
                &self.con_frontier_pts[..],
            )
            .color(palette.series_color(0))
            .width(2.4_f32),
        );
        if self.cml_pts.len() >= 2 {
            plot_ui.line(
                Line::new("Capital Market Line (CML)", &self.cml_pts[..])
                    .color(palette.warning)
                    .style(LineStyle::Dotted { spacing: 5.0 }),
            );
        }
        plot_ui.points(
            Points::new("Individual Assets", &self.asset_pts[..])
                .color(palette.accent)
                .shape(MarkerShape::Circle)
                .radius(4.5_f32),
        );
        if let Some(ref pt) = gmv_pt {
            plot_ui.points(
                Points::new("Constrained GMV", &pt[..])
                    .color(palette.series_color(2))
                    .shape(MarkerShape::Diamond)
                    .radius(6.0_f32),
            );
        }
        if let Some(ref pt) = tan_pt {
            plot_ui.points(
                Points::new("Max-Sharpe Tangency", &pt[..])
                    .color(palette.success)
                    .shape(MarkerShape::Up)
                    .radius(6.5_f32),
            );
        }
        self.callout_manager.render_on_plot(plot_ui);
    }

    fn render_weights_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.strong("Optimal Capital Weights by Asset Index (%)");
        let n = self.expected_returns.len();
        let gmv_bars: Vec<Bar> = self
            .con_gmv
            .map(|a| {
                (0..n)
                    .map(|i| Bar::new(i as f64 - 0.16, a.weights[i] * 100.0).width(0.26))
                    .collect()
            })
            .unwrap_or_default();
        let tan_bars: Vec<Bar> = self
            .con_tangency
            .map(|a| {
                (0..n)
                    .map(|i| Bar::new(i as f64 + 0.16, a.weights[i] * 100.0).width(0.26))
                    .collect()
            })
            .unwrap_or_default();

        let resp = Plot::new("markowitz_weights_bar_plot")
            .height(175.0)
            .x_axis_label("Asset Index i (0-based)")
            .y_axis_label("Portfolio Weight (%)")
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.hline(
                    HLine::new("Long-Only Floor (0%)", 0.0)
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 5.0 }),
                );
                if self.max_asset_weight < 0.999 {
                    plot_ui.hline(
                        HLine::new("Weight Cap", self.max_asset_weight * 100.0)
                            .color(palette.warning)
                            .style(LineStyle::Dotted { spacing: 5.0 }),
                    );
                }
                if !gmv_bars.is_empty() {
                    plot_ui.bar_chart(
                        BarChart::new("Constrained GMV (%)", gmv_bars)
                            .color(palette.series_color(2)),
                    );
                }
                if !tan_bars.is_empty() {
                    plot_ui.bar_chart(
                        BarChart::new("Constrained Tangency (%)", tan_bars).color(palette.success),
                    );
                }
            });
        register_canvas_accessibility(
            ui,
            &resp.response,
            Role::GraphicsDocument,
            "Optimal GMV and Tangency Portfolio Weights Bar Chart",
            Some("Capital allocation percentage across active assets"),
        );
        paint_canvas_focus_ring(ui, &resp.response);
    }

    fn render_weights_table(&self, ui: &mut egui::Ui) {
        let (Some(gmv), Some(tan)) = (self.con_gmv, self.con_tangency) else {
            return;
        };
        egui::Grid::new("markowitz_weights_table")
            .num_columns(4)
            .spacing([20.0, 4.0])
            .striped(true)
            .show(ui, |ui| {
                ui.strong("#");
                ui.strong("Asset");
                ui.strong("Constrained GMV");
                ui.strong("Constrained Tangency");
                ui.end_row();
                for i in 0..self.asset_names.len() {
                    ui.small(format!("{i}"));
                    ui.label(&self.asset_names[i]);
                    ui.label(format!("{:.2}%", gmv.weights[i] * 100.0));
                    ui.label(format!("{:.2}%", tan.weights[i] * 100.0));
                    ui.end_row();
                }
            });
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut sets = vec![
            (
                "Constrained Active-Set Frontier".into(),
                to_pairs(&self.con_frontier_pts),
            ),
            (
                "Unconstrained Lagrangian Frontier".into(),
                to_pairs(&self.unc_frontier_pts),
            ),
            ("Individual Assets".into(), to_pairs(&self.asset_pts)),
        ];
        if let Some(tan) = self.con_tangency {
            let weights: Vec<[f64; 2]> = (0..tan.asset_count)
                .map(|i| [i as f64, tan.weights[i] * 100.0])
                .collect();
            sets.push(("Constrained Tangency Weights (%)".into(), weights));
        }
        sets
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        let mut prims = Vec::new();
        if self.unc_frontier_pts.len() >= 2 {
            prims.push(ExportPrimitive::Polyline {
                name: "Unconstrained Lagrangian Frontier".into(),
                points: to_pairs(&self.unc_frontier_pts),
                color: pal.series[1],
                stroke_width: 1.8,
                closed: false,
                fill: None,
            });
        }
        if self.con_frontier_pts.len() >= 2 {
            prims.push(ExportPrimitive::Polyline {
                name: "Constrained Active-Set Frontier".into(),
                points: to_pairs(&self.con_frontier_pts),
                color: pal.series[0],
                stroke_width: 2.4,
                closed: false,
                fill: None,
            });
        }
        if !self.asset_pts.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "Individual Assets".into(),
                points: to_pairs(&self.asset_pts),
                color: pal.accent,
                radius: 4.5,
            });
        }
        prims
    }
}

// [cite:clinical_trials_statistics]
