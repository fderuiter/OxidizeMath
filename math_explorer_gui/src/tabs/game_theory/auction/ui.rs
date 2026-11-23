//! Side panel controls and plots for [`super::OptimalAuctionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{presets, AuctionDistKind, OptimalAuctionTool};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints, VLine};

impl OptimalAuctionTool {
    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Myerson Auctions");
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("⚖️ 2-Bidder Uniform")
                .accessible_hover_text("Load 2-bidder Uniform[0, 100] auction with r* = 50")
                .clicked()
            {
                self.cfg = presets::default_config();
                self.recompute();
            }
            if ui
                .button("🔨 Monopoly Reserve")
                .accessible_hover_text("Load 1-bidder Uniform[0, 200] monopoly reserve problem")
                .clicked()
            {
                self.cfg = presets::monopoly_config();
                self.recompute();
            }
            if ui
                .button("📈 Exponential Hazard")
                .accessible_hover_text("Load Exponential(0.25) constant-hazard auction with r* = 4")
                .clicked()
            {
                self.cfg = presets::exponential_config();
                self.recompute();
            }
        });
        ui.separator();

        let mut changed = false;
        ui.horizontal(|ui| {
            if ui
                .selectable_label(self.cfg.dist_kind == AuctionDistKind::Uniform, "Uniform")
                .clicked()
            {
                self.cfg.dist_kind = AuctionDistKind::Uniform;
                changed = true;
            }
            if ui
                .selectable_label(
                    self.cfg.dist_kind == AuctionDistKind::Exponential,
                    "Exponential",
                )
                .clicked()
            {
                self.cfg.dist_kind = AuctionDistKind::Exponential;
                changed = true;
            }
        });

        changed |= self.show_parameter_sliders(ui);
        if changed {
            self.recompute();
        }
    }

    fn show_parameter_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        match self.cfg.dist_kind {
            AuctionDistKind::Uniform => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.cfg.upper_bound, 5.0..=500.0)
                            .text("Max Valuation b"),
                    )
                    .changed();
                let max_lo = (self.cfg.upper_bound - 1.0).max(0.0);
                self.cfg.lower_bound = self.cfg.lower_bound.clamp(0.0, max_lo);
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.cfg.lower_bound, 0.0..=max_lo)
                            .text("Min Valuation a"),
                    )
                    .changed();
            }
            AuctionDistKind::Exponential => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.cfg.exp_rate, 0.05..=2.0).text("Hazard Rate λ"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.cfg.upper_bound, 4.0..=100.0)
                            .text("Plot Horizon"),
                    )
                    .changed();
            }
        }
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.n_bidders, 1..=16).text("Bidders n"))
            .changed();
        let hi = self.cfg.upper_bound;
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.custom_reserve, 0.0..=hi).text("Test Reserve r"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.mc_trials, 500..=10_000).text("MC Auctions"))
            .changed();
        ui.horizontal(|ui| {
            ui.label("RNG Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.cfg.seed).speed(1.0))
                .changed();
        });
        changed
    }

    pub(super) fn show_main_view(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let uplift = self.mc_optimal_revenue - self.no_reserve_revenue;
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!(
                "Myerson Optimal Reserve r*: {:.3}",
                self.optimal_reserve
            ));
            ui.separator();
            ui.strong(format!(
                "Optimal E[Rev] (MC): {:.3}",
                self.mc_optimal_revenue
            ));
            ui.separator();
            ui.label(format!("No-Reserve E[Rev]: {:.3}", self.no_reserve_revenue));
            ui.separator();
            ui.label(format!(
                "Test Reserve r={:.2} E[Rev]: {:.3}",
                self.cfg.custom_reserve, self.custom_reserve_revenue
            ));
            ui.separator();
            ui.colored_label(palette.success, format!("Reserve ΔRev: {:+.3}", uplift));
        });

        Plot::new("myerson_auction_plot")
            .legend(Legend::default())
            .include_y(0.0)
            .show(ui, |plot_ui| {
                let j_pts: PlotPoints = self.virtual_curve.iter().copied().collect();
                let r_pts: PlotPoints = self.revenue_curve.iter().copied().collect();
                plot_ui.line(
                    Line::new("Virtual Valuation J(v)", j_pts).color(palette.series_color(0)),
                );
                plot_ui
                    .line(Line::new("Expected Revenue R(r)", r_pts).color(palette.series_color(1)));
                plot_ui.vline(
                    VLine::new("Optimal Reserve r*", self.optimal_reserve)
                        .color(palette.series_color(2)),
                );
                plot_ui.vline(
                    VLine::new("Test Reserve r", self.cfg.custom_reserve)
                        .color(palette.series_color(3)),
                );
                plot_ui.hline(HLine::new("J(v) = 0", 0.0).color(palette.warning));
            });
    }
}
