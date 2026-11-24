//! Side-panel controls, WCAG status badges, and central plot views for [`super::MyersonAuctionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::challenge::SHADING_FIXED_POINT_TOL;
use super::math::{AuctionDistributionKind, AuctionPlotView, AuctionRuleMode};
use super::MyersonAuctionTool;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoints, Points, Polygon, VLine};

impl MyersonAuctionTool {
    pub(super) fn show_settings_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Myerson Mechanism Design");
            let mut changed = false;
            changed |= self.render_distribution_controls(ui);
            ui.separator();
            changed |= self.render_auction_controls(ui);
            if changed {
                self.record_history_and_recalculate();
            }
            ui.separator();
            self.render_equilibrium_readouts(ui);
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                if self.atlas.show_toggle_button(ui) {
                    self.bump_revision();
                }
            });
            self.render_auction_atlas_panel(ui);
            ui.separator();
            let summary = self.param_summary();
            let count = self.snapshot_manager.len();
            let series = self.current_trace_series();
            if self.snapshot_manager.render_ui(ui, move || {
                Some((format!("Auction {}", count + 1), summary, series))
            }) {
                self.bump_revision();
            }
        });
    }

    fn render_distribution_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let env_locked = self.is_env_locked();
        let mut changed = false;
        ui.strong("Valuation Distribution F(v)");
        ui.add_enabled_ui(!env_locked, |ui| {
            egui::ComboBox::from_id_salt("myerson_dist_selector")
                .selected_text(self.distribution.label())
                .show_ui(ui, |ui| {
                    for kind in [
                        AuctionDistributionKind::Uniform,
                        AuctionDistributionKind::Exp,
                        AuctionDistributionKind::Beta,
                    ] {
                        if ui
                            .selectable_value(&mut self.distribution, kind, kind.label())
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            match self.distribution {
                AuctionDistributionKind::Uniform => {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.uniform_min, 0.0..=200.0)
                                .text("Min Valuation (v_min)"),
                        )
                        .changed();
                    let min_hi = (self.uniform_min + 1.0).min(500.0);
                    if self.uniform_max < min_hi {
                        self.uniform_max = min_hi;
                    }
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.uniform_max, min_hi..=500.0)
                                .text("Max Valuation (v_max)"),
                        )
                        .changed();
                }
                AuctionDistributionKind::Exp => {
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut self.exp_rate, 0.05..=5.0)
                                .text("Hazard Rate (λ)"),
                        )
                        .changed();
                }
                AuctionDistributionKind::Beta => {
                    changed |= ui
                        .add(egui::Slider::new(&mut self.beta_alpha, 0.5..=8.0).text("Shape α"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut self.beta_beta, 0.5..=8.0).text("Shape β"))
                        .changed();
                }
            }
        });
        changed
    }

    fn render_auction_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let env_locked = self.is_env_locked();
        let reserve_locked = self.is_reserve_locked();
        let (v_min, v_max) = self.dist_eval.plot_bounds();
        let mut changed = false;

        ui.strong("Mechanism & Bidding Rules");
        ui.add_enabled_ui(!env_locked, |ui| {
            ui.horizontal_wrapped(|ui| {
                for mode in [AuctionRuleMode::SecondPrice, AuctionRuleMode::FirstPrice] {
                    if ui
                        .selectable_value(&mut self.rule_mode, mode, mode.label())
                        .changed()
                    {
                        changed = true;
                    }
                }
            });
            let mut n_u32 = self.n_bidders as u32;
            if ui
                .add(egui::Slider::new(&mut n_u32, 1..=20).text("Bidders (n)"))
                .changed()
            {
                self.n_bidders = n_u32 as usize;
                changed = true;
            }
        });

        ui.add_enabled_ui(!reserve_locked, |ui| {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.reserve_price, v_min..=v_max)
                        .text("Seller Reserve (r)"),
                )
                .changed();
            let snap_lbl = format!(
                "Snap r to Myerson Optimal r* ({:.2})",
                self.summary.optimal_reserve
            );
            if ui
                .button(snap_lbl)
                .accessible_hover_text(
                    "Set seller reserve price to the root of virtual valuation J(r*) = 0.",
                )
                .clicked()
            {
                self.reserve_price = self.summary.optimal_reserve.clamp(v_min, v_max);
                changed = true;
            }
        });
        changed | self.render_bidding_and_mc_controls(ui, v_min, v_max, env_locked)
    }

    fn render_bidding_and_mc_controls(
        &mut self,
        ui: &mut egui::Ui,
        v_min: f64,
        v_max: f64,
        env_locked: bool,
    ) -> bool {
        let mut changed = false;
        ui.add_enabled_ui(!env_locked, |ui| {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.focal_valuation, v_min..=v_max)
                        .text("Focal Bidder v"),
                )
                .changed();
        });
        changed |= ui
            .add(egui::Slider::new(&mut self.candidate_bid, 0.0..=v_max).text("Candidate Bid b(v)"))
            .changed();
        ui.horizontal_wrapped(|ui| {
            let eq_lbl = format!("Snap to b*(v) = {:.2}", self.summary.equilibrium_bid);
            if ui.button(eq_lbl).clicked() {
                self.candidate_bid = self.summary.equilibrium_bid;
                changed = true;
            }
            if ui.button("Truthful b = v").clicked() {
                self.candidate_bid = self.summary.truthful_bid;
                changed = true;
            }
        });
        let mut mc_u32 = self.mc_samples as u32;
        if ui
            .add(egui::Slider::new(&mut mc_u32, 200..=10_000).text("MC Draws (OxidizeRng)"))
            .changed()
        {
            self.mc_samples = mc_u32 as usize;
            changed = true;
        }
        changed
    }

    fn render_equilibrium_badge(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let s = &self.summary;
        let (eq_col, eq_txt) = if s.is_equilibrium_fixed_point {
            (
                palette.success,
                format!(
                    "✓ Bayes-Nash Fixed Point Verified (|b - b*| ≤ {SHADING_FIXED_POINT_TOL:.2})"
                ),
            )
        } else {
            (
                palette.warning,
                format!(
                    "⚠ Off-Equilibrium Bid: |b - b*| = {:.2} (b* = {:.2})",
                    (self.candidate_bid - s.equilibrium_bid).abs(),
                    s.equilibrium_bid
                ),
            )
        };
        ui.colored_label(eq_col, eq_txt);
    }

    fn render_equilibrium_readouts(&self, ui: &mut egui::Ui) {
        self.render_equilibrium_badge(ui);
        let s = &self.summary;
        let eff_pct = if s.efficient_welfare > 1e-9 {
            100.0 * (s.social_welfare / s.efficient_welfare).clamp(0.0, 1.0)
        } else {
            100.0
        };
        let rows = [
            (
                "Myerson Optimal Reserve r*",
                format!("{:.4}", s.optimal_reserve),
            ),
            (
                "Expected Seller Revenue R(r)",
                format!(
                    "{:.4} (Optimal: {:.4})",
                    s.expected_revenue, s.optimal_revenue
                ),
            ),
            (
                "Monte Carlo Optimal Revenue",
                format!("{:.4} ({} draws)", s.mc_optimal_revenue, self.mc_samples),
            ),
            (
                "Trade Prob / Blocked Trades",
                format!(
                    "{:.2}% traded  |  {:.2}% blocked",
                    s.trade_probability * 100.0,
                    (1.0 - s.trade_probability) * 100.0
                ),
            ),
            (
                "Social Welfare & Efficiency",
                format!("{:.4} ({eff_pct:.2}% of first-best)", s.social_welfare),
            ),
            (
                "Equilibrium Bid & Utility",
                format!(
                    "b*(v) = {:.4}, U(b|v) = {:.4} (opt {:.4})",
                    s.equilibrium_bid, s.candidate_utility, s.equilibrium_utility
                ),
            ),
        ];
        for (lbl, val) in rows {
            render_copyable_metric(ui, lbl, &val);
        }
    }

    pub(super) fn show_central_plot(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.strong("Curve View:");
            for (view, label) in [
                (AuctionPlotView::VirtualValuation, "Virtual Valuation J(v)"),
                (AuctionPlotView::BidFunctions, "First- vs Second-Price Bids"),
                (AuctionPlotView::RevenueWelfare, "Revenue & Welfare vs r"),
            ] {
                if ui
                    .selectable_value(&mut self.plot_view, view, label)
                    .changed()
                {
                    self.sync_auction_callouts();
                    self.bump_revision();
                }
            }
        });
        ui.separator();

        let palette = SemanticPalette::from_visuals(ui.visuals());
        let (x_label, y_label) = match self.plot_view {
            AuctionPlotView::VirtualValuation => ("Bidder Valuation (v)", "Virtual Valuation J(v)"),
            AuctionPlotView::BidFunctions => ("Bidder Valuation (v)", "Submitted Bid b(v)"),
            AuctionPlotView::RevenueWelfare => ("Seller Reserve Price (r)", "Expected Surplus"),
        };

        Plot::new("myerson_auction_central_plot")
            .x_axis_label(x_label)
            .y_axis_label(y_label)
            .legend(Legend::default())
            .show(ui, |plot_ui| {
                self.snapshot_manager.render_on_plot(plot_ui);
                self.draw_active_view_curves(plot_ui, &palette);
                self.callout_manager.render_on_plot(plot_ui);
            });
    }

    fn draw_active_view_curves(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
    ) {
        match self.plot_view {
            AuctionPlotView::VirtualValuation => self.draw_jv_view(plot_ui, palette),
            AuctionPlotView::BidFunctions => self.draw_bid_shading_view(plot_ui, palette),
            AuctionPlotView::RevenueWelfare => self.draw_revenue_welfare_view(plot_ui, palette),
        }
    }

    fn draw_jv_view(&self, plot_ui: &mut egui_plot::PlotUi<'_>, palette: &SemanticPalette) {
        if self.exclusion_polygon.len() >= 3 {
            plot_ui.polygon(
                Polygon::new(
                    "J(v) < 0 Exclusion Zone",
                    PlotPoints::from(self.exclusion_polygon.clone()),
                )
                .fill_color(palette.error.gamma_multiply(0.18_f32))
                .stroke(egui::Stroke::new(1.0_f32, palette.error)),
            );
        }
        plot_ui.line(
            Line::new(
                "Virtual Valuation J(v)",
                PlotPoints::from(self.jv_points.clone()),
            )
            .color(palette.series_color(0))
            .width(2.4_f32),
        );
        plot_ui.hline(HLine::new("Zero Threshold J = 0", 0.0).color(palette.warning));
        plot_ui.vline(
            VLine::new("Myerson Optimal r*", self.summary.optimal_reserve)
                .color(palette.success)
                .style(LineStyle::Dashed { length: 6.0_f32 }),
        );
        plot_ui.vline(VLine::new("Active Reserve r", self.reserve_price).color(palette.info));
    }

    fn draw_bid_shading_view(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
    ) {
        plot_ui.line(
            Line::new(
                "Second-Price Truthful b(v) = v",
                PlotPoints::from(self.second_price_points.clone()),
            )
            .color(palette.series_color(1))
            .style(LineStyle::Dashed { length: 6.0_f32 })
            .width(2.0_f32),
        );
        plot_ui.line(
            Line::new(
                "First-Price Bayes-Nash b*(v)",
                PlotPoints::from(self.first_price_points.clone()),
            )
            .color(palette.series_color(0))
            .width(2.5_f32),
        );
        plot_ui.vline(VLine::new("Reserve Price r", self.reserve_price).color(palette.warning));
        let pt_col = if self.summary.is_equilibrium_fixed_point {
            palette.success
        } else {
            palette.error
        };
        plot_ui.points(
            Points::new(
                "Candidate Bid (v, b)",
                vec![[self.focal_valuation, self.candidate_bid]],
            )
            .radius(5.5_f32)
            .color(pt_col),
        );
    }

    fn draw_revenue_welfare_view(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &SemanticPalette,
    ) {
        plot_ui.line(
            Line::new(
                "Expected Seller Revenue R(r)",
                PlotPoints::from(self.revenue_curve.clone()),
            )
            .color(palette.series_color(0))
            .width(2.5_f32),
        );
        plot_ui.line(
            Line::new(
                "Expected Social Welfare W(r)",
                PlotPoints::from(self.welfare_curve.clone()),
            )
            .color(palette.series_color(2))
            .width(2.1_f32),
        );
        plot_ui.line(
            Line::new(
                "Efficiency Loss (Deadweight)",
                PlotPoints::from(self.deadweight_curve.clone()),
            )
            .color(palette.error)
            .style(LineStyle::Dashed { length: 5.0_f32 })
            .width(1.8_f32),
        );
        plot_ui.vline(
            VLine::new("Myerson Optimal r*", self.summary.optimal_reserve)
                .color(palette.success)
                .style(LineStyle::Dashed { length: 6.0_f32 }),
        );
        plot_ui.vline(VLine::new("Active Reserve r", self.reserve_price).color(palette.warning));
    }
}
