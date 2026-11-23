//! Side panel controls and plots for [`super::NashShapleyTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    presets, NashShapleyTool, NashShapleyViewMode, MAX_UI_COALITION_PLAYERS, MAX_UI_STRATEGIES,
};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints};
use math_commons::math_kernel::types::flatten_2d_index;

fn format_coalition_mask(mask: usize, n_players: usize) -> String {
    let mut members = Vec::new();
    for i in 0..n_players {
        if (mask & (1usize << i)) != 0 {
            members.push(format!("P{}", i + 1));
        }
    }
    format!("{{{}}}", members.join(","))
}

impl NashShapleyTool {
    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Nash & Shapley");
        ui.horizontal(|ui| {
            if ui
                .selectable_label(
                    self.cfg.mode == NashShapleyViewMode::BimatrixNash,
                    "Bimatrix Nash",
                )
                .clicked()
            {
                self.cfg.mode = NashShapleyViewMode::BimatrixNash;
                self.recompute();
            }
            if ui
                .selectable_label(
                    self.cfg.mode == NashShapleyViewMode::CoalitionalShapley,
                    "Coalition Shapley",
                )
                .clicked()
            {
                self.cfg.mode = NashShapleyViewMode::CoalitionalShapley;
                self.recompute();
            }
        });
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🎭 Battle of Sexes")
                .accessible_hover_text("Load 2x2 Battle of the Sexes with 3 Nash equilibria")
                .clicked()
            {
                self.cfg = presets::default_config();
                self.recompute();
            }
            if ui
                .button("✂️ Zero-Sum RPS")
                .accessible_hover_text("Load 3x3 zero-sum Rock-Paper-Scissors minimax game")
                .clicked()
            {
                self.cfg = presets::rps_zero_sum_config();
                self.recompute();
            }
            if ui
                .button("🤝 Convex Coalition")
                .accessible_hover_text("Load 4-player convex synergy game in the Core")
                .clicked()
            {
                self.cfg = presets::convex_synergy_config();
                self.recompute();
            }
        });
        ui.separator();

        match self.cfg.mode {
            NashShapleyViewMode::BimatrixNash => self.show_bimatrix_controls(ui),
            NashShapleyViewMode::CoalitionalShapley => self.show_coalition_controls(ui),
        }
    }

    fn show_bimatrix_controls(&mut self, ui: &mut egui::Ui) {
        let mut r = self.cfg.rows;
        let mut c = self.cfg.cols;
        ui.horizontal(|ui| {
            ui.label("Rows:");
            ui.add(egui::DragValue::new(&mut r).range(2..=MAX_UI_STRATEGIES));
            ui.label("Cols:");
            ui.add(egui::DragValue::new(&mut c).range(2..=MAX_UI_STRATEGIES));
        });
        if r != self.cfg.rows || c != self.cfg.cols {
            self.resize_bimatrix(r, c);
        }

        let mut changed = false;
        changed |= ui
            .checkbox(&mut self.cfg.zero_sum_lock, "Zero-Sum Minimax (B = −A)")
            .changed();
        changed |= ui
            .checkbox(&mut self.cfg.weak_dominance, "Weak Dominance (IEWDS)")
            .changed();
        if !self.cfg.zero_sum_lock {
            let max_lbl = self.cfg.rows + self.cfg.cols - 1;
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.cfg.initial_label, 0..=max_lbl)
                        .text("LH Initial Label"),
                )
                .changed();
        }

        ui.label("Row Payoffs A / Col Payoffs B:");
        let cols = self.cfg.cols;
        egui::Grid::new("nash_bimatrix_grid")
            .striped(true)
            .show(ui, |ui| {
                for row in 0..self.cfg.rows {
                    for col in 0..cols {
                        let idx = flatten_2d_index(col, row, cols);
                        ui.horizontal(|ui| {
                            changed |= ui
                                .add(
                                    egui::DragValue::new(&mut self.cfg.row_payoffs[idx])
                                        .speed(0.25)
                                        .range(-100.0..=100.0),
                                )
                                .changed();
                            if !self.cfg.zero_sum_lock {
                                changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut self.cfg.col_payoffs[idx])
                                            .speed(0.25)
                                            .range(-100.0..=100.0),
                                    )
                                    .changed();
                            }
                        });
                    }
                    ui.end_row();
                }
            });

        if changed {
            self.recompute();
        }
    }

    fn show_coalition_controls(&mut self, ui: &mut egui::Ui) {
        let mut n = self.cfg.coalition_players;
        if ui
            .add(egui::Slider::new(&mut n, 2..=MAX_UI_COALITION_PLAYERS).text("Players"))
            .changed()
        {
            self.resize_coalition(n);
        }

        let mut changed = false;
        let np = self.cfg.coalition_players;
        egui::ScrollArea::vertical()
            .max_height(280.0)
            .show(ui, |ui| {
                for (mask, val) in self
                    .cfg
                    .characteristic_values
                    .iter_mut()
                    .enumerate()
                    .take(1usize << np)
                    .skip(1)
                {
                    ui.horizontal(|ui| {
                        ui.label(format!("v({}):", format_coalition_mask(mask, np)));
                        changed |= ui
                            .add(egui::DragValue::new(val).speed(0.25).range(-100.0..=500.0))
                            .changed();
                    });
                }
            });
        if changed {
            self.recompute();
        }
    }

    pub(super) fn show_main_view(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        match self.cfg.mode {
            NashShapleyViewMode::BimatrixNash => self.show_bimatrix_main_view(ui, &palette),
            NashShapleyViewMode::CoalitionalShapley => self.show_coalition_main_view(ui, &palette),
        }
    }

    fn show_bimatrix_main_view(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!("Row E[u]: {:.3}", self.row_payoff));
            ui.separator();
            ui.strong(format!("Col E[v]: {:.3}", self.col_payoff));
            ui.separator();
            ui.label(format!("Max Regret: {:.2e}", self.max_regret));
            ui.separator();
            ui.label(format!("Pivots: {}", self.pivots));
            ui.separator();
            ui.label(format!("Support Equilibria: {}", self.enumerated_count));
            ui.separator();
            let surv_r = self.row_alive.iter().filter(|&&b| b).count();
            let surv_c = self.col_alive.iter().filter(|&&b| b).count();
            ui.label(format!(
                "Dominance Survivors: {surv_r}x{surv_c} ({} rds)",
                self.dom_rounds
            ));
        });

        Plot::new("nash_mixed_strategy_plot")
            .legend(Legend::default())
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                let row_bars: Vec<Bar> = self
                    .row_strategy
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| Bar::new(i as f64 - 0.16, p).width(0.28))
                    .collect();
                let col_bars: Vec<Bar> = self
                    .col_strategy
                    .iter()
                    .enumerate()
                    .map(|(j, &q)| Bar::new(j as f64 + 0.16, q).width(0.28))
                    .collect();
                plot_ui.bar_chart(
                    BarChart::new("Row Strategy p*", row_bars).color(palette.series_color(0)),
                );
                plot_ui.bar_chart(
                    BarChart::new("Col Strategy q*", col_bars).color(palette.series_color(1)),
                );
                let p_line: PlotPoints = self
                    .row_strategy
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| [i as f64, p])
                    .collect();
                plot_ui.line(Line::new("p* Envelope", p_line).color(palette.series_color(2)));
            });
    }

    fn show_coalition_main_view(&self, ui: &mut egui::Ui, palette: &crate::app::SemanticPalette) {
        ui.horizontal_wrapped(|ui| {
            let grand = self
                .cfg
                .characteristic_values
                .last()
                .copied()
                .unwrap_or(0.0);
            ui.strong(format!("Grand Worth v(N): {grand:.2}"));
            ui.separator();
            ui.label(format!("Superadditive: {}", self.superadditive));
            ui.separator();
            ui.label(format!("Convex: {}", self.convex));
            ui.separator();
            let status_color = if self.in_core {
                palette.success
            } else {
                palette.warning
            };
            ui.colored_label(
                status_color,
                format!(
                    "In Core: {} (Max Excess: {:.3})",
                    self.in_core, self.core_excess
                ),
            );
        });

        Plot::new("shapley_allocation_plot")
            .legend(Legend::default())
            .include_y(0.0)
            .show(ui, |plot_ui| {
                let bars: Vec<Bar> = self
                    .shapley
                    .iter()
                    .enumerate()
                    .map(|(i, &phi)| Bar::new(i as f64, phi).width(0.45))
                    .collect();
                plot_ui.bar_chart(
                    BarChart::new("Shapley Value φ_i", bars).color(palette.series_color(0)),
                );
                let pts: PlotPoints = self
                    .shapley
                    .iter()
                    .enumerate()
                    .map(|(i, &phi)| [i as f64, phi])
                    .collect();
                plot_ui.line(Line::new("Shapley Profile", pts).color(palette.series_color(1)));
            });
    }
}
