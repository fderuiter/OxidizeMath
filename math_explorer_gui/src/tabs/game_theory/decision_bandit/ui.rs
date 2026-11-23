//! Controls and visualizations for [`super::DecisionBanditTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{presets, DecisionBanditTool, DecisionWorkbenchMode, RiskMode};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoints, Points};

impl DecisionBanditTool {
    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Decision & Bandits");
        ui.horizontal(|ui| {
            if ui
                .selectable_label(
                    self.cfg.mode == DecisionWorkbenchMode::DecisionTreeVoi,
                    "Tree & VOI",
                )
                .clicked()
            {
                self.cfg.mode = DecisionWorkbenchMode::DecisionTreeVoi;
                self.recompute();
            }
            if ui
                .selectable_label(
                    self.cfg.mode == DecisionWorkbenchMode::BanditRegret,
                    "UCB1 & Thompson",
                )
                .clicked()
            {
                self.cfg.mode = DecisionWorkbenchMode::BanditRegret;
                self.recompute();
            }
        });
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🌳 CARA Seismic VOI")
                .accessible_hover_text("Load risk-averse CARA drilling & EVSI model")
                .clicked()
            {
                self.cfg = presets::cara_seismic_config();
                self.recompute();
            }
            if ui
                .button("🎰 Bandit Regret Race")
                .accessible_hover_text("Load 4-arm Thompson Sampling vs UCB1 benchmark")
                .clicked()
            {
                self.cfg = presets::thompson_vs_ucb1_config();
                self.recompute();
            }
        });
        ui.separator();

        match self.cfg.mode {
            DecisionWorkbenchMode::DecisionTreeVoi => self.show_tree_voi_controls(ui),
            DecisionWorkbenchMode::BanditRegret => self.show_bandit_controls(ui),
        }
    }

    fn show_risk_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Utility:");
            for (mode, label) in [
                (RiskMode::Neutral, "Neutral"),
                (RiskMode::Cara, "CARA"),
                (RiskMode::Crra, "CRRA"),
            ] {
                if ui
                    .selectable_label(self.cfg.risk_mode == mode, label)
                    .clicked()
                {
                    self.cfg.risk_mode = mode;
                    changed = true;
                }
            }
        });
        if self.cfg.risk_mode != RiskMode::Neutral {
            let max_r = if self.cfg.risk_mode == RiskMode::Cara {
                0.1
            } else {
                3.0
            };
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.cfg.risk_param, 0.001..=max_r)
                        .text("Risk Parameter"),
                )
                .changed();
        }
        changed
    }

    fn show_tree_voi_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = self.show_risk_selector(ui);
        let min_w = if self.cfg.risk_mode == RiskMode::Crra {
            1.0
        } else {
            -200.0
        };
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.safe_payoff, min_w..=200.0).text("Safe Payoff"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.cfg.success_payoff, min_w..=300.0)
                    .text("Success Payoff"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.cfg.failure_payoff, min_w..=150.0)
                    .text("Failure Payoff"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.prior_good, 0.05..=0.95).text("Prior P(Good)"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.cfg.signal_sensitivity, 0.5..=0.99).text("Sensitivity"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.cfg.signal_specificity, 0.5..=0.99).text("Specificity"),
            )
            .changed();
        if changed {
            self.recompute();
        }
    }

    fn show_bandit_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        let mut arm_count = self.cfg.arm_probs.len();
        if ui
            .add(egui::Slider::new(&mut arm_count, 2..=6).text("Arms"))
            .changed()
        {
            self.cfg.arm_probs.resize(arm_count, 0.5);
            changed = true;
        }
        for (i, p) in self.cfg.arm_probs.iter_mut().enumerate() {
            changed |= ui
                .add(egui::Slider::new(p, 0.01..=0.99).text(format!("Arm {} μ", i + 1)))
                .changed();
        }
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.ucb_c, 0.1..=3.0).text("UCB1 c"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.horizon, 50..=800).text("Horizon T"))
            .changed();
        ui.horizontal(|ui| {
            ui.label(format!("Seed: {}", self.cfg.rng_seed));
            if ui
                .button("🎲 Reseed")
                .accessible_hover_text("Advance deterministic OxidizeRng seed")
                .clicked()
            {
                self.cfg.rng_seed = self.cfg.rng_seed.wrapping_add(137);
                changed = true;
            }
        });
        if changed {
            self.recompute();
        }
    }

    pub(super) fn show_main_view(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        match self.cfg.mode {
            DecisionWorkbenchMode::DecisionTreeVoi => self.show_tree_voi_plot(ui, &palette),
            DecisionWorkbenchMode::BanditRegret => self.show_bandit_regret_plot(ui, &palette),
        }
    }

    fn show_tree_voi_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        let choice = if self.optimal_branch == 0 {
            "Safe Option"
        } else {
            "Risky Project"
        };
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!("Optimal Policy: {choice}"));
            ui.separator();
            ui.label(format!("Root CE: {:.2}", self.root_ce));
            ui.separator();
            ui.label(format!("Risky Branch CE: {:.2}", self.risky_ce));
            ui.separator();
            ui.label(format!("EVSI: {:.2}", self.evsi_val));
            ui.separator();
            ui.label(format!("EVPI Ceiling: {:.2}", self.evpi_val));
        });

        Plot::new("decision_voi_plot")
            .legend(Legend::default())
            .include_y(0.0)
            .show(ui, |plot_ui| {
                let pts: PlotPoints = self.evsi_curve.iter().copied().collect();
                plot_ui.line(
                    Line::new("Symmetric Signal EVSI(q)", pts)
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                plot_ui.hline(
                    HLine::new("EVPI Upper Bound", self.evpi_val)
                        .color(palette.series_color(1))
                        .style(LineStyle::Dashed { length: 8.0_f32 }),
                );
                let avg_acc = 0.5 * (self.cfg.signal_sensitivity + self.cfg.signal_specificity);
                plot_ui.points(
                    Points::new("Current Signal EVSI", vec![[avg_acc, self.evsi_val]])
                        .radius(5.0_f32)
                        .color(palette.series_color(2)),
                );
            });
    }

    fn show_bandit_regret_plot(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        let ucb_r = self.ucb_regret.last().map_or(0.0, |p| p[1]);
        let ts_r = self.thompson_regret.last().map_or(0.0, |p| p[1]);
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!("UCB1 Regret: {ucb_r:.2}"));
            ui.separator();
            ui.strong(format!("Thompson Regret: {ts_r:.2}"));
            ui.separator();
            ui.label(format!("Thompson Pulls: {:?}", self.thompson_pulls));
        });

        Plot::new("bandit_regret_plot")
            .legend(Legend::default())
            .include_y(0.0)
            .show(ui, |plot_ui| {
                let ucb_pts: PlotPoints = self.ucb_regret.iter().copied().collect();
                let ts_pts: PlotPoints = self.thompson_regret.iter().copied().collect();
                plot_ui.line(
                    Line::new("UCB1 Cumulative Regret", ucb_pts)
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("Thompson Cumulative Regret", ts_pts)
                        .color(palette.series_color(1))
                        .width(2.0_f32),
                );
            });
    }
}
