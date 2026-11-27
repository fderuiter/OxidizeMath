//! Interactive side-panel controls and `egui_plot` views for [`super::WinRatioStressTesterTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::mpsc;

use eframe::egui;
use egui_plot::{Bar, BarChart, HLine, Legend, Line, Plot, PlotPoints, Points};
use math_explorer::applied::clinical_trials::survival_analysis::TimePoint;
use math_explorer::applied::win_ratio::power_sweep::{
    MAX_ARM_SIZE, MAX_SWEEP_REPLICATES, MAX_SWEEP_STEPS,
};

use super::cohort_io::{load_and_analyze_cohort, SAMPLE_COHORT_CSV};
use super::protocol_export::{trigger_save_protocol_csv, trigger_save_protocol_json};
use super::{ChartMode, WinRatioStressTesterTool};
use crate::accessibility::AccessibleHoverText;
use crate::reflective_ui::render_copyable_metric;
use crate::web_io::{spawn_open_file_dialog, OpenDialogConfig};

const COHORT_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Two-Arm Clinical Cohort (.csv, .json)",
    extensions: &["csv", "json"],
    accept: ".csv,.json,text/csv,application/json",
    read_error_prefix: "Failed to read cohort file",
};

impl WinRatioStressTesterTool {
    pub(super) fn poll_async_io(&mut self, ctx: &egui::Context) {
        let cohort_msg = self
            .open_cohort_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());
        if let Some(outcome) = cohort_msg {
            self.open_cohort_rx = None;
            match outcome {
                Ok((name, body)) => match self.apply_cohort_content(&name, &body) {
                    Ok(()) => {
                        self.status_banner =
                            Some((format!("Loaded cohort '{name}' successfully."), false));
                    }
                    Err(err) => {
                        self.status_banner = Some((format!("Cohort rejected: {err}"), true));
                    }
                },
                Err(err) => self.status_banner = Some((err, true)),
            }
            ctx.request_repaint();
        }
        let save_msg = self
            .save_protocol_rx
            .as_ref()
            .and_then(|rx| rx.try_recv().ok());
        if let Some(outcome) = save_msg {
            self.save_protocol_rx = None;
            match outcome {
                Ok(name) => {
                    self.status_banner =
                        Some((format!("Saved protocol takeaway '{name}'."), false));
                }
                Err(err) => self.status_banner = Some((err, true)),
            }
            ctx.request_repaint();
        }
    }

    pub(super) fn apply_cohort_content(&mut self, name: &str, content: &str) -> Result<(), String> {
        let horizon = self
            .apply_horizon_to_cohort
            .then_some(self.sweep_cfg.censor_horizon);
        let loaded = load_and_analyze_cohort(name, content, self.sweep_cfg.sig_alpha, horizon)?;
        self.uploaded_cohort = Some(loaded);
        Ok(())
    }

    pub(super) fn render_side_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Composite Stress Controls");
        let mut sweep_dirty = self.render_weibull_controls(ui);
        ui.add_space(4.0);
        sweep_dirty |= self.render_censoring_and_grid_controls(ui);
        if sweep_dirty {
            self.recompute_parametric_sweep();
            self.refresh_uploaded_cohort_if_needed();
        }
        ui.separator();
        self.render_cohort_io_section(ui);
        ui.separator();
        self.render_protocol_download_section(ui);
    }

    fn render_weibull_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.strong("Bivariate Weibull Hazards");
            changed |= ui
                .add(egui::Slider::new(&mut self.params.lambda1, 0.02..=1.0).text("Fatal λ₁"))
                .accessible_hover_text("Control baseline fatal hazard scale parameter λ₁")
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.params.lambda2, 0.02..=1.5).text("Non-Fatal λ₂"))
                .accessible_hover_text("Control baseline non-fatal hazard scale parameter λ₂")
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.params.alpha, 0.20..=1.0).text("Copula α"))
                .accessible_hover_text(
                    "Association shape α in (0, 1]; smaller α increases correlation",
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.params.theta, 0.25..=1.5).text("Effect θ"))
                .accessible_hover_text("Hazard multiplier θ (< 1 favors treatment; 1.0 = null)")
                .changed();
        });
        changed
    }

    fn render_censoring_and_grid_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.strong("Censoring & Monte Carlo Sweep");
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.sweep_cfg.censor_horizon, 0.5..=20.0)
                        .text("Follow-Up C"),
                )
                .accessible_hover_text("Administrative follow-up censoring horizon")
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.sweep_cfg.dropout_rate, 0.0..=0.50)
                        .text("Dropout γ"),
                )
                .accessible_hover_text("Exponential random loss-to-follow-up hazard rate")
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.sweep_cfg.sig_alpha, 0.01..=0.10).text("Alpha α"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.sweep_cfg.n_min, 10..=MAX_ARM_SIZE).text("N min"))
                .changed();
            if self.sweep_cfg.n_max < self.sweep_cfg.n_min {
                self.sweep_cfg.n_max = self.sweep_cfg.n_min;
            }
            changed |= ui
                .add(
                    egui::Slider::new(
                        &mut self.sweep_cfg.n_max,
                        self.sweep_cfg.n_min..=MAX_ARM_SIZE,
                    )
                    .text("N max"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.sweep_cfg.n_steps, 2..=MAX_SWEEP_STEPS)
                        .text("Grid Steps"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.sweep_cfg.replicates, 20..=MAX_SWEEP_REPLICATES)
                        .text("Replicates"),
                )
                .changed();
            ui.horizontal(|ui| {
                ui.label("Seed:");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.seed).speed(1.0))
                    .changed();
            });
        });
        changed
    }

    fn render_cohort_io_section(&mut self, ui: &mut egui::Ui) {
        ui.strong("Local Two-Arm Cohort (.csv / .json)");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Import Cohort…").clicked() {
                let (tx, rx) = mpsc::channel();
                self.open_cohort_rx = Some(rx);
                spawn_open_file_dialog(tx, COHORT_OPEN_CONFIG, |path| {
                    path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("cohort")
                        .to_string()
                });
            }
            if ui.button("Load Sample Cohort").clicked()
                && self
                    .apply_cohort_content("sample_cardio_cohort.csv", SAMPLE_COHORT_CSV)
                    .is_ok()
            {
                self.status_banner = Some((
                    "Loaded built-in two-arm cardiovascular sample cohort.".into(),
                    false,
                ));
            }
            if self.uploaded_cohort.is_some() && ui.button("Clear Cohort").clicked() {
                self.uploaded_cohort = None;
                self.status_banner =
                    Some(("Reverted to parametric simulation view.".into(), false));
            }
        });
        if self.uploaded_cohort.is_some()
            && ui
                .checkbox(
                    &mut self.apply_horizon_to_cohort,
                    "Apply Follow-Up Horizon C to uploaded cohort",
                )
                .changed()
        {
            self.refresh_uploaded_cohort_if_needed();
        }
    }

    fn render_protocol_download_section(&mut self, ui: &mut egui::Ui) {
        ui.strong("Takeaway Protocol Tables");
        ui.horizontal(|ui| {
            if ui.button("Save Protocol CSV…").clicked() {
                let (tx, rx) = mpsc::channel();
                self.save_protocol_rx = Some(rx);
                trigger_save_protocol_csv(tx, self.build_protocol_csv());
            }
            if ui.button("Save Protocol JSON…").clicked() {
                let (tx, rx) = mpsc::channel();
                self.save_protocol_rx = Some(rx);
                trigger_save_protocol_json(tx, self.build_protocol_json());
            }
        });
        if let Some((msg, is_err)) = &self.status_banner {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let color = if *is_err {
                palette.error
            } else {
                palette.success
            };
            ui.colored_label(color, msg);
        }
    }

    pub(super) fn render_central_view(&mut self, ui: &mut egui::Ui) {
        ui.heading("Composite Win Ratio Stress Tester");
        self.render_summary_strip(ui);
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Visualization:");
            ui.selectable_value(
                &mut self.chart_mode,
                ChartMode::PowerCurve,
                "Power-by-N Sweep",
            );
            ui.selectable_value(
                &mut self.chart_mode,
                ChartMode::TierWaterfall,
                "Hierarchical Tier Waterfall",
            );
            if self.uploaded_cohort.is_some() {
                ui.selectable_value(
                    &mut self.chart_mode,
                    ChartMode::CohortSurvival,
                    "Uploaded Cohort Kaplan-Meier",
                );
            } else if self.chart_mode == ChartMode::CohortSurvival {
                self.chart_mode = ChartMode::PowerCurve;
            }
        });
        ui.add_space(4.0);
        match self.chart_mode {
            ChartMode::PowerCurve => self.render_power_sweep_plot(ui),
            ChartMode::TierWaterfall => self.render_tier_waterfall_plot(ui),
            ChartMode::CohortSurvival => self.render_cohort_km_plot(ui),
        }
    }

    fn render_summary_strip(&self, ui: &mut egui::Ui) {
        let rep = self.report();
        let (s, _) = self.active_summary_and_tiers();
        ui.horizontal_wrapped(|ui| {
            render_copyable_metric(
                ui,
                "Win Ratio [95% CI]",
                &format!("{:.2} [{:.2}, {:.2}]", s.win_ratio, s.ci_low, s.ci_high),
            );
            render_copyable_metric(ui, "U-Stat p-value", &format!("{:.4}", s.p_value));
            render_copyable_metric(ui, "Win Odds", &format!("{:.2}", s.win_odds));
            render_copyable_metric(
                ui,
                "Net Benefit",
                &format!("{:+.1}%", s.net_benefit * 100.0),
            );
            render_copyable_metric(
                ui,
                "Uncensored PR_W",
                &format!("{:.2}", rep.theoretical_uncensored_wr),
            );
        });
        if let Some(c) = &self.uploaded_cohort {
            ui.horizontal_wrapped(|ui| {
                ui.strong(format!(
                    "Cohort '{}' (n₁={}, n₂={}):",
                    c.source_name,
                    c.treatment.len(),
                    c.control.len()
                ));
                if let Ok(hr) = c.fatal_hazard_ratio {
                    render_copyable_metric(ui, "Fatal HR", &format!("{hr:.3}"));
                }
                render_copyable_metric(
                    ui,
                    "Risk Diff",
                    &format!("{:+.1}%", c.risk_difference * 100.0),
                );
                if let Ok(rm) = &c.risk_metrics {
                    render_copyable_metric(ui, "RR", &format!("{:.2}", rm.relative_risk));
                    render_copyable_metric(ui, "OR", &format!("{:.2}", rm.odds_ratio));
                }
            });
        }
    }

    fn render_power_sweep_plot(&self, ui: &mut egui::Ui) {
        let rep = self.report();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let pwr_pts: Vec<[f64; 2]> = rep
            .points
            .iter()
            .map(|p| [p.n_per_arm as f64, p.empirical_power])
            .collect();
        let tie_pts: Vec<[f64; 2]> = rep
            .points
            .iter()
            .map(|p| [p.n_per_arm as f64, p.tie_fraction])
            .collect();

        Plot::new("win_ratio_power_sweep_plot")
            .legend(Legend::default())
            .x_axis_label("Subjects per Arm (N)")
            .y_axis_label("Probability / Fraction")
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "Empirical Power Pr(p < α)",
                        PlotPoints::new(pwr_pts.clone()),
                    )
                    .color(palette.accent)
                    .width(2.5_f32),
                );
                plot_ui.points(
                    Points::new("Sweep Grid Points", PlotPoints::new(pwr_pts))
                        .color(palette.accent)
                        .radius(4.5_f32),
                );
                plot_ui.line(
                    Line::new("Composite Tie Fraction", PlotPoints::new(tie_pts))
                        .color(palette.warning)
                        .width(2.0_f32),
                );
                plot_ui.hline(
                    HLine::new("80% Target Power", 0.80)
                        .color(palette.success)
                        .style(egui_plot::LineStyle::dashed_dense()),
                );
            });
    }

    fn render_tier_waterfall_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let (summary, tiers) = self.active_summary_and_tiers();
        let denom = (summary.total_pairs.max(1)) as f64 / 100.0;
        let win_bars = vec![
            Bar::new(1.0, (tiers[0].wins as f64) / denom).name("Tier 1 Fatal Wins"),
            Bar::new(2.0, (tiers[1].wins as f64) / denom).name("Tier 2 Non-Fatal Wins"),
        ];
        let loss_bars = vec![
            Bar::new(1.32, (tiers[0].losses as f64) / denom).name("Tier 1 Fatal Losses"),
            Bar::new(2.32, (tiers[1].losses as f64) / denom).name("Tier 2 Non-Fatal Losses"),
        ];
        let tie_bars =
            vec![Bar::new(3.0, (tiers[1].ties_remaining as f64) / denom).name("Unresolved Ties")];

        Plot::new("win_ratio_tier_waterfall_plot")
            .legend(Legend::default())
            .x_axis_label("Composite Priority Tier (1=Fatal, 2=Non-Fatal, 3=Unresolved Ties)")
            .y_axis_label("Pairwise Share (%)")
            .include_y(0.0)
            .show(ui, |plot_ui| {
                plot_ui.bar_chart(
                    BarChart::new("Treatment Wins (%)", win_bars)
                        .width(0.28)
                        .color(palette.success),
                );
                plot_ui.bar_chart(
                    BarChart::new("Treatment Losses (%)", loss_bars)
                        .width(0.28)
                        .color(palette.error),
                );
                plot_ui.bar_chart(
                    BarChart::new("Unresolved Ties (%)", tie_bars)
                        .width(0.32)
                        .color(palette.warning),
                );
            });
    }

    fn render_cohort_km_plot(&self, ui: &mut egui::Ui) {
        let Some(cohort) = &self.uploaded_cohort else {
            return;
        };
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let trt = km_step_points(&cohort.trt_km);
        let ctrl = km_step_points(&cohort.ctrl_km);
        Plot::new("win_ratio_uploaded_km_plot")
            .legend(Legend::default())
            .x_axis_label("Follow-up Time")
            .y_axis_label("Tier-1 Event-Free Survival")
            .include_y(0.0)
            .include_y(1.05)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Treatment Survival", PlotPoints::new(trt))
                        .color(palette.accent)
                        .width(2.5_f32),
                );
                plot_ui.line(
                    Line::new("Control Survival", PlotPoints::new(ctrl))
                        .color(palette.error)
                        .width(2.2_f32),
                );
            });
    }
}

pub(super) fn km_step_points(curve: &[TimePoint]) -> Vec<[f64; 2]> {
    let mut pts = Vec::with_capacity(curve.len() * 2 + 2);
    pts.push([0.0, 1.0]);
    for pt in curve {
        if let Some(last) = pts.last() {
            pts.push([pt.time, last[1]]);
        }
        pts.push([pt.time, pt.survival_probability]);
    }
    pts
}
