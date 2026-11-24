#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{
    AssetConfig, BsmPlotMetric, PortfolioWorkbenchState, WorkbenchView, MAX_WORKBENCH_ASSETS,
    MIN_WORKBENCH_ASSETS,
};
use super::PortfolioCopulaWorkbenchTool;
use crate::accessibility::AccessibleHoverText;
use eframe::egui;

impl PortfolioCopulaWorkbenchTool {
    pub(super) fn show_controls_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Portfolio & Copula Risk");
            ui.label("Multi-asset Kelly, QMC Gaussian copula, OU cones & BSM Greeks.");
            ui.separator();

            let mut dirty = self.show_view_selector(ui);
            ui.separator();
            self.show_file_io_controls(ui);
            ui.separator();

            dirty |= self.show_common_portfolio_controls(ui);
            ui.separator();

            match self.state.view {
                WorkbenchView::KellyFrontier => dirty |= self.show_asset_list_controls(ui),
                WorkbenchView::CopulaTailCrash => dirty |= self.show_copula_controls(ui),
                WorkbenchView::OuMeanReversion => dirty |= self.show_ou_controls(ui),
                WorkbenchView::BlackScholesGreeks => dirty |= self.show_bsm_controls(ui),
            }

            if dirty {
                self.recalculate_all();
            }
        });
    }

    fn show_view_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Workbench Lens");
        ui.horizontal_wrapped(|ui| {
            for mode in [
                WorkbenchView::KellyFrontier,
                WorkbenchView::CopulaTailCrash,
                WorkbenchView::OuMeanReversion,
                WorkbenchView::BlackScholesGreeks,
            ] {
                changed |= ui
                    .selectable_value(&mut self.state.view, mode, mode.label())
                    .accessible_hover_text("Switch active risk engine visualization")
                    .changed();
            }
        });
        changed
    }

    fn show_file_io_controls(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.strong("Return Series & Risk Tear-Sheets");
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📂 Import .csv/.json")
                .accessible_hover_text(
                    "Import historical asset return matrix (up to 2 MiB, 10,000 rows, 1..=12 assets)",
                )
                .clicked()
            {
                self.trigger_open_returns_dialog();
            }
            if self.imported.is_some()
                && ui
                    .button("✖ Clear Import")
                    .accessible_hover_text("Revert to manual parametric benchmark")
                    .clicked()
            {
                self.imported = None;
                self.state.corr_upper.clear();
                self.recalculate_all();
            }
        });
        self.show_tearsheet_export_buttons(ui);
        if let Some(ref imp) = self.imported {
            ui.colored_label(
                palette.success,
                format!(
                    "Loaded {} ({} rows: {})",
                    imp.source_name,
                    imp.row_count,
                    imp.column_names.join(", ")
                ),
            );
        }
        if let Some((ref msg, is_err)) = self.io_status {
            let color = if is_err { palette.error } else { palette.info };
            ui.colored_label(color, msg);
        }
    }

    fn show_tearsheet_export_buttons(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("💾 Save CSV")
                .accessible_hover_text("Save formula-safe CSV risk tear-sheet")
                .clicked()
            {
                self.trigger_save_tearsheet_dialog(false);
            }
            if ui
                .button("💾 Save JSON")
                .accessible_hover_text("Save structured JSON risk tear-sheet")
                .clicked()
            {
                self.trigger_save_tearsheet_dialog(true);
            }
            if ui
                .button("📋 Copy CSV")
                .accessible_hover_text("Copy CSV tear-sheet to clipboard")
                .clicked()
            {
                self.copy_tearsheet_to_clipboard(ui.ctx(), false);
            }
            if ui
                .button("📋 Copy JSON")
                .accessible_hover_text("Copy JSON tear-sheet to clipboard")
                .clicked()
            {
                self.copy_tearsheet_to_clipboard(ui.ctx(), true);
            }
        });
    }

    fn show_common_portfolio_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let min_rho = PortfolioWorkbenchState::min_equicorrelation_for(
            self.state.assets.len().max(self.state.copula_legs),
        );
        egui::Grid::new("pw_common_grid")
            .num_columns(2)
            .spacing([14.0, 6.0])
            .show(ui, |ui| {
                changed |= self.show_bankroll_and_rate_rows(ui);
                ui.label("Equicorrelation ρ:");
                let rho_edited = ui
                    .add(egui::Slider::new(&mut self.state.rho, min_rho..=0.95))
                    .accessible_hover_text("Pairwise correlation across assets and copula legs")
                    .changed();
                if rho_edited {
                    self.state.corr_upper.clear();
                    changed = true;
                }
                ui.end_row();
                changed |= self.show_horizon_and_seed_row(ui);
            });
        changed
    }

    fn show_bankroll_and_rate_rows(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Bankroll ($):");
        changed |= ui
            .add(
                egui::DragValue::new(&mut self.state.bankroll)
                    .speed(25.0)
                    .range(1.0..=10_000_000.0)
                    .prefix("$"),
            )
            .accessible_hover_text("Portfolio capital for VaR and CVaR sizing")
            .changed();
        ui.end_row();

        ui.label("Risk-Free Rate r:");
        changed |= ui
            .add(
                egui::DragValue::new(&mut self.state.risk_free_rate)
                    .speed(0.002)
                    .range(-0.20..=0.50),
            )
            .accessible_hover_text("Annualized continuously compounded risk-free rate")
            .changed();
        ui.end_row();
        changed
    }

    fn show_horizon_and_seed_row(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Horizon T / Seed:");
        ui.horizontal(|ui| {
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.state.horizon)
                        .speed(0.05)
                        .range(0.05..=25.0)
                        .suffix(" yr"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.state.seed)
                        .speed(1.0)
                        .range(0..=1_000_000),
                )
                .changed();
        });
        ui.end_row();
        changed
    }

    fn show_asset_list_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let mut count = self.state.assets.len();
        ui.horizontal(|ui| {
            ui.strong("Assets (N):");
            if ui
                .add(
                    egui::DragValue::new(&mut count)
                        .speed(0.2)
                        .range(MIN_WORKBENCH_ASSETS..=MAX_WORKBENCH_ASSETS),
                )
                .accessible_hover_text("Number of correlated portfolio legs (1..=12)")
                .changed()
            {
                while self.state.assets.len() < count {
                    let idx = self.state.assets.len();
                    self.state.assets.push(AssetConfig::benchmark_leg(idx));
                }
                self.state.assets.truncate(count);
                self.state.corr_upper.clear();
                let min_r = PortfolioWorkbenchState::min_equicorrelation_for(
                    count.max(self.state.copula_legs),
                );
                if self.state.rho < min_r {
                    self.state.rho = min_r;
                }
                changed = true;
            }
            ui.label("Long Cap:");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.state.leverage_cap)
                        .speed(0.05)
                        .range(0.10..=10.0)
                        .suffix("x"),
                )
                .accessible_hover_text("Gross weight cap for projected-gradient long-only Kelly")
                .changed();
        });
        changed | self.show_asset_grid(ui)
    }

    fn show_asset_grid(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::Grid::new("pw_assets_grid")
            .num_columns(4)
            .striped(true)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                for hdr in ["Leg", "μ", "σ", "u_sgp"] {
                    ui.strong(hdr);
                }
                ui.end_row();
                for asset in &mut self.state.assets {
                    ui.label(&asset.label);
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut asset.mu)
                                .speed(0.005)
                                .range(-0.95..=2.50),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut asset.sigma)
                                .speed(0.005)
                                .range(0.01..=2.50),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut asset.marginal_u)
                                .speed(0.01)
                                .range(0.01..=0.99),
                        )
                        .changed();
                    ui.end_row();
                }
            });
        changed
    }

    fn show_copula_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Gaussian Copula QMC Parameters");
        egui::Grid::new("pw_copula_grid")
            .num_columns(2)
            .spacing([14.0, 6.0])
            .show(ui, |ui| {
                ui.label("Parlay / Crash Legs:");
                if ui
                    .add(egui::Slider::new(
                        &mut self.state.copula_legs,
                        2..=MAX_WORKBENCH_ASSETS,
                    ))
                    .accessible_hover_text("Number of correlated legs (2..=12) integrated via QMC")
                    .changed()
                {
                    let min_r = PortfolioWorkbenchState::min_equicorrelation_for(
                        self.state.assets.len().max(self.state.copula_legs),
                    );
                    if self.state.rho < min_r {
                        self.state.rho = min_r;
                    }
                    changed = true;
                }
                ui.end_row();

                ui.label("Tail Crash Quantile u:");
                changed |= ui
                    .add(egui::Slider::new(
                        &mut self.state.crash_quantile,
                        0.005..=0.35,
                    ))
                    .accessible_hover_text(
                        "Marginal crash threshold u (e.g., 0.05 = 5% tail event)",
                    )
                    .changed();
                ui.end_row();
            });
        changed
    }

    fn show_ou_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Ornstein-Uhlenbeck Forecast Cone");
        egui::Grid::new("pw_ou_grid")
            .num_columns(2)
            .spacing([14.0, 6.0])
            .show(ui, |ui| {
                ui.label("Equilibrium μ / Start X₀:");
                changed |= Self::drag_f64_pair(
                    ui,
                    (&mut self.state.ou_mu, 0.01, -5.0..=5.0),
                    (&mut self.state.ou_x0, 0.01, -5.0..=5.0),
                );
                ui.end_row();

                ui.label("Reversion θ / Vol σ:");
                changed |= Self::drag_f64_pair(
                    ui,
                    (&mut self.state.ou_theta, 0.05, 0.05..=20.0),
                    (&mut self.state.ou_sigma, 0.01, 0.01..=5.0),
                );
                ui.end_row();

                ui.label("Step dt / Horizon:");
                changed |= Self::drag_f64_pair(
                    ui,
                    (&mut self.state.ou_dt, 0.002, 0.002..=0.25),
                    (&mut self.state.ou_horizon, 0.1, 0.10..=20.0),
                );
                ui.end_row();

                ui.label("MC Paths:");
                changed |= ui
                    .add(egui::Slider::new(&mut self.state.ou_paths, 32..=1_024))
                    .changed();
                ui.end_row();
            });
        changed
    }

    fn show_bsm_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.strong("Black-Scholes-Merton Option Engine");
        ui.horizontal_wrapped(|ui| {
            for metric in BsmPlotMetric::ALL {
                changed |= ui
                    .selectable_value(&mut self.state.bsm_metric, metric, metric.label())
                    .changed();
            }
        });
        egui::Grid::new("pw_bsm_grid")
            .num_columns(2)
            .spacing([14.0, 6.0])
            .show(ui, |ui| {
                ui.label("Spot S / Strike K:");
                changed |= Self::drag_f64_pair(
                    ui,
                    (&mut self.state.bsm_spot, 0.5, 1.0..=10_000.0),
                    (&mut self.state.bsm_strike, 0.5, 1.0..=10_000.0),
                );
                ui.end_row();

                ui.label("Volatility σ / Expiry T:");
                changed |= Self::drag_f64_pair(
                    ui,
                    (&mut self.state.bsm_vol, 0.01, 0.01..=3.0),
                    (&mut self.state.bsm_expiry, 0.05, 0.02..=20.0),
                );
                ui.end_row();

                ui.label("Dividend Yield q:");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.state.bsm_dividend)
                            .speed(0.002)
                            .range(0.0..=0.50),
                    )
                    .changed();
                ui.end_row();
            });
        changed
    }

    fn drag_f64_pair(
        ui: &mut egui::Ui,
        left: (&mut f64, f64, std::ops::RangeInclusive<f64>),
        right: (&mut f64, f64, std::ops::RangeInclusive<f64>),
    ) -> bool {
        ui.horizontal(|ui| {
            let c1 = ui
                .add(egui::DragValue::new(left.0).speed(left.1).range(left.2))
                .changed();
            let c2 = ui
                .add(egui::DragValue::new(right.0).speed(right.1).range(right.2))
                .changed();
            c1 | c2
        })
        .inner
    }
}

// [cite:clinical_trials_statistics]
