//! Interactive sidebar controls, metrics cards, and `egui_plot` canvas for `WinRatioTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cohort::{
    ComparatorKind, PairingMode, WinRatioPlotMode, MAX_HORIZON_C, MAX_PRC_SAMPLES, MAX_THRESHOLD,
    MAX_WEIBULL_ALPHA, MAX_WEIBULL_LAMBDA, MAX_WEIBULL_THETA, MIN_HORIZON_C, MIN_PRC_SAMPLES,
    MIN_WEIBULL_ALPHA, MIN_WEIBULL_LAMBDA, MIN_WEIBULL_THETA,
};
use super::{export, WinRatioTool};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, Points};

/// Renders the side control panel and central visualization panel for [`WinRatioTool`].
pub(crate) fn show_win_ratio_inside(tool: &mut WinRatioTool, ui: &mut egui::Ui) {
    tool.poll_io(ui.ctx());
    let side_id = scoped_panel_id(ui, "win_ratio_controls_panel");
    PaneSidePanel::left(side_id)
        .default_width(370.0)
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_sidebar(tool, ui);
            });
        });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            render_central_panel(tool, ui);
        });
    });
}

fn render_sidebar(tool: &mut WinRatioTool, ui: &mut egui::Ui) {
    ui.heading("Composite Win Ratio Workbench");
    let mut changed = render_cohort_section(tool, ui);
    ui.add_space(4.0);
    changed |= render_hierarchy_section(tool, ui);
    ui.add_space(4.0);
    changed |= render_ttest_and_bmi_section(tool, ui);
    ui.add_space(4.0);
    changed |= render_weibull_section(tool, ui);

    if changed {
        tool.recalculate();
    }

    if let Some(ref err) = tool.error_message {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.colored_label(palette.error, format!("⚠ {err}"));
    }

    ui.separator();
    render_metrics_summary(tool, ui);
    ui.add_space(4.0);
    render_report_export_box(tool, ui);
}

fn render_cohort_section(tool: &mut WinRatioTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Local Patient Cohort (CSV / TSV)").strong());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📂 Import Cohort (.csv / .tsv)")
                .accessible_hover_text("Load local patient cohort without uploading data")
                .clicked()
            {
                tool.io_state.trigger_open_cohort();
            }
            if ui
                .button("🔄 Reset Sample")
                .accessible_hover_text("Restore default 12-patient composite cohort")
                .clicked()
            {
                tool.cohort_text = super::cohort::DEFAULT_COHORT_CSV.to_string();
                tool.derive_bmi_tier3 = false;
                changed = true;
            }
        });
        ui.small("Columns: Arm, Tier1 (Fatal), Tier2 (Hosp)[, Tier3 | WeightKg, HeightM]");
        changed |= ui
            .checkbox(
                &mut tool.derive_bmi_tier3,
                "Derive Tier 3 BMI from Weight (kg) & Height (m)",
            )
            .accessible_hover_text("Compute Tier 3 BMI = weight_kg / height_m^2 on 5-column rows")
            .changed();
        let editor = egui::TextEdit::multiline(&mut tool.cohort_text)
            .desired_width(f32::INFINITY)
            .desired_rows(6);
        changed |= ui.add(editor).changed();
    });
    changed
}

fn render_comparator_row(ui: &mut egui::Ui, label: &str, kind: &mut ComparatorKind) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.strong(label);
        if ui
            .selectable_label(matches!(kind, ComparatorKind::HigherIsBetter), "Higher ↑")
            .accessible_hover_text("Higher outcome value wins")
            .clicked()
        {
            *kind = ComparatorKind::HigherIsBetter;
            changed = true;
        }
        if ui
            .selectable_label(matches!(kind, ComparatorKind::LowerIsBetter), "Lower ↓")
            .accessible_hover_text("Lower outcome value wins")
            .clicked()
        {
            *kind = ComparatorKind::LowerIsBetter;
            changed = true;
        }
        let is_thresh = matches!(kind, ComparatorKind::Threshold { .. });
        if ui
            .selectable_label(is_thresh, "Threshold Δ")
            .accessible_hover_text("Higher wins only when difference exceeds threshold")
            .clicked()
            && !is_thresh
        {
            *kind = ComparatorKind::Threshold { threshold: 5.0 };
            changed = true;
        }
    });
    if let ComparatorKind::Threshold { ref mut threshold } = kind {
        changed |= ui
            .add(
                egui::Slider::new(threshold, 0.0..=MAX_THRESHOLD.min(250.0))
                    .text(format!("{label} Δ")),
            )
            .changed();
    }
    changed
}

fn render_hierarchy_section(tool: &mut WinRatioTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Prioritized Hierarchy & Pairing").strong());
        ui.horizontal_wrapped(|ui| {
            changed |= ui
                .radio_value(
                    &mut tool.pairing_mode,
                    PairingMode::Unmatched,
                    "Unmatched (All-Pairs)",
                )
                .changed();
            changed |= ui
                .radio_value(
                    &mut tool.pairing_mode,
                    PairingMode::Matched,
                    "Matched Pairs (1:1)",
                )
                .changed();
        });
        changed |= render_comparator_row(ui, "Tier 1 (Fatal):", &mut tool.tier1_kind);
        changed |= render_comparator_row(ui, "Tier 2 (Hosp):", &mut tool.tier2_kind);
        changed |= render_comparator_row(ui, "Tier 3 (Biomarker):", &mut tool.tier3_kind);
    });
    changed
}

fn render_ttest_and_bmi_section(tool: &mut WinRatioTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Secondary t-Test & BMI Covariate").strong());
        ui.horizontal_wrapped(|ui| {
            ui.label("t-Test Endpoint:");
            for (idx, lbl) in [(0, "Tier 1"), (1, "Tier 2"), (2, "Tier 3")] {
                changed |= ui
                    .selectable_value(&mut tool.ttest_tier_index, idx, lbl)
                    .changed();
            }
        });
        changed |= ui
            .add(egui::Slider::new(&mut tool.ttest_alpha, 0.001..=0.20).text("t-Test α"))
            .changed();
        ui.collapsing("BMI Covariate Calculator (kg / m²)", |ui| {
            changed |= ui
                .add(egui::Slider::new(&mut tool.treat_weight_kg, 30.0..=200.0).text("Treat kg"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut tool.treat_height_m, 1.20..=2.20).text("Treat m"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut tool.ctrl_weight_kg, 30.0..=200.0).text("Ctrl kg"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut tool.ctrl_height_m, 1.20..=2.20).text("Ctrl m"))
                .changed();
        });
    });
    changed
}

fn render_weibull_section(tool: &mut WinRatioTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.label(egui::RichText::new("Bivariate Weibull PR(c) Integral").strong());
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut tool.weibull_lambda1,
                    MIN_WEIBULL_LAMBDA..=MAX_WEIBULL_LAMBDA,
                )
                .text("Fatal Scale λ₁"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut tool.weibull_lambda2,
                    MIN_WEIBULL_LAMBDA..=MAX_WEIBULL_LAMBDA,
                )
                .text("Non-Fatal Scale λ₂"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut tool.weibull_alpha,
                    MIN_WEIBULL_ALPHA..=MAX_WEIBULL_ALPHA,
                )
                .text("Correlation α"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut tool.weibull_theta,
                    MIN_WEIBULL_THETA..=MAX_WEIBULL_THETA,
                )
                .text("Hazard Ratio θ"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.horizon_c, MIN_HORIZON_C..=MAX_HORIZON_C)
                    .text("Horizon c_max"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut tool.prc_samples, MIN_PRC_SAMPLES..=MAX_PRC_SAMPLES)
                    .text("Grid Samples"),
            )
            .changed();
    });
    changed
}

fn render_metrics_summary(tool: &WinRatioTool, ui: &mut egui::Ui) {
    let ev = &tool.eval;
    ui.label(egui::RichText::new("Statistical Summary").strong());
    render_copyable_metric(
        ui,
        "Wins / Losses / Ties",
        &format!("{} / {} / {}", ev.wins, ev.losses, ev.ties),
    );
    if let Some(stats) = ev.stats {
        let wr_str = if stats.win_ratio.is_finite() {
            format!(
                "{:.3} [{:.3}, {:.3}]",
                stats.win_ratio, stats.ci_low, stats.ci_high
            )
        } else {
            format!("∞ [{:.3}, ∞]", stats.ci_low)
        };
        render_copyable_metric(ui, "Win Ratio [95% CI]", &wr_str);
        render_copyable_metric(ui, "WR P-Value", &format!("{:.4}", stats.p_value));
    }
    render_copyable_metric(
        ui,
        "Pocock 2-Tier Z",
        &format!(
            "{:.3} (pw = {:.3})",
            ev.pocock.z_statistic, ev.pocock.win_proportion
        ),
    );
    if let Some(ref tt) = ev.ttest_result {
        render_copyable_metric(
            ui,
            "Independent t-Test",
            &format!("t = {:.3}, p = {:.4}", tt.statistic, tt.p_value),
        );
    } else if let Some(ref note) = ev.ttest_note {
        ui.small(format!("t-Test: {note}"));
    }
    render_copyable_metric(
        ui,
        "Weibull PR_W / PR(c)",
        &format!("{:.3} / {:.3}", ev.weibull_pr_w, ev.terminal_pr_c),
    );
    render_copyable_metric(
        ui,
        "BMI (Treat / Ctrl)",
        &format!("{:.1} / {:.1} kg/m²", ev.treat_ref_bmi, ev.ctrl_ref_bmi),
    );
}

fn render_report_export_box(tool: &mut WinRatioTool, ui: &mut egui::Ui) {
    ui.group(|ui| {
        ui.label(egui::RichText::new("HIPAA-Safe Win Ratio Reports").strong());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📋 Copy .md Report")
                .accessible_hover_text("Copy Markdown composite Win Ratio report to clipboard")
                .clicked()
            {
                ui.ctx().copy_text(export::format_markdown_report(tool));
                tool.io_state
                    .record_info("Copied Markdown Win Ratio report to clipboard".into());
            }
            if ui
                .button("💾 Save .md Report")
                .accessible_hover_text("Download or save Markdown composite Win Ratio report")
                .clicked()
            {
                let md = export::format_markdown_report(tool);
                tool.io_state
                    .trigger_save_markdown("win_ratio_report.md", md);
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📋 Copy .csv Report")
                .accessible_hover_text("Copy CSV Win Ratio and Weibull PR(c) table to clipboard")
                .clicked()
            {
                ui.ctx().copy_text(export::format_csv_report(tool));
                tool.io_state
                    .record_info("Copied CSV Win Ratio report to clipboard".into());
            }
            if ui
                .button("💾 Save .csv Report")
                .accessible_hover_text("Download or save CSV Win Ratio report")
                .clicked()
            {
                let csv = export::format_csv_report(tool);
                tool.io_state.trigger_save_csv("win_ratio_report.csv", csv);
            }
        });
        tool.io_state.render_status(ui);
    });
}

fn render_central_panel(tool: &mut WinRatioTool, ui: &mut egui::Ui) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.horizontal_wrapped(|ui| {
        ui.heading("Composite Endpoint & Probability Win Ratio Analysis");
        ui.separator();
        ui.selectable_value(
            &mut tool.plot_mode,
            WinRatioPlotMode::WeibullPrc,
            "Weibull PR(c) & Survival",
        );
        ui.selectable_value(
            &mut tool.plot_mode,
            WinRatioPlotMode::EmpiricalHierarchy,
            "Hierarchy Tier Forest",
        );
    });
    ui.add_space(4.0);

    match tool.plot_mode {
        WinRatioPlotMode::WeibullPrc => render_prc_plot(tool, ui, palette),
        WinRatioPlotMode::EmpiricalHierarchy => render_hierarchy_plot(tool, ui, palette),
    }

    ui.add_space(8.0);
    render_tier_breakdown_table(tool, ui);
}

fn render_prc_plot(tool: &WinRatioTool, ui: &mut egui::Ui, palette: SemanticPalette) {
    Plot::new("win_ratio_prc_plot")
        .height(380.0)
        .x_axis_label("Follow-up Censoring Horizon c")
        .y_axis_label("Win Ratio / Survival Probability")
        .include_y(0.0)
        .legend(Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("PR(c) (Clenshaw-Curtis)", tool.prc_points.clone())
                    .color(palette.series_color(0))
                    .width(2.5_f32),
            );
            plot_ui.line(
                Line::new("Treatment S₁(t)", tool.surv_treat_points.clone())
                    .color(palette.series_color(1))
                    .style(LineStyle::Dashed { length: 6.0 }),
            );
            plot_ui.line(
                Line::new("Control S₀(t)", tool.surv_ctrl_points.clone())
                    .color(palette.series_color(2))
                    .style(LineStyle::Dotted { spacing: 5.0 }),
            );
            plot_ui.hline(
                HLine::new("Uncensored PR_W = 1/θ^α", tool.eval.weibull_pr_w)
                    .color(palette.success)
                    .style(LineStyle::Dashed { length: 8.0 }),
            );
            plot_ui.hline(
                HLine::new("Null Hypothesis (WR = 1.0)", 1.0)
                    .color(palette.muted_text)
                    .style(LineStyle::Dotted { spacing: 4.0 }),
            );
        });
}

fn render_hierarchy_plot(tool: &WinRatioTool, ui: &mut egui::Ui, palette: SemanticPalette) {
    let count = tool.eval.active_tier_count;
    Plot::new("win_ratio_hierarchy_plot")
        .height(380.0)
        .x_axis_label("Cumulative Hierarchy Tier")
        .y_axis_label("Cumulative Win Ratio (95% CI)")
        .include_y(0.0)
        .legend(Legend::default())
        .show(ui, |plot_ui| {
            let mut est_pts = Vec::with_capacity(count);
            for (k, &t) in tool.eval.tiers[..count].iter().enumerate() {
                let x = (k + 1) as f64;
                let y = if t.cumulative_wr.is_finite() {
                    t.cumulative_wr
                } else {
                    10.0
                };
                est_pts.push([x, y]);
                let hi = t.cumulative_ci_high.min(y.max(2.0) * 2.5);
                plot_ui.line(
                    Line::new(
                        format!("Tier {} 95% CI", k + 1),
                        vec![[x, t.cumulative_ci_low], [x, hi]],
                    )
                    .color(palette.series_color(k + 1))
                    .width(3.0_f32),
                );
            }
            plot_ui.line(
                Line::new("Cumulative WR Trajectory", est_pts.clone())
                    .color(palette.series_color(0))
                    .width(2.2_f32),
            );
            plot_ui.points(
                Points::new("Tier WR Estimate", est_pts)
                    .radius(5.5_f32)
                    .color(palette.series_color(0)),
            );
            plot_ui.hline(
                HLine::new("Null Effect (WR = 1.0)", 1.0)
                    .color(palette.warning)
                    .style(LineStyle::Dashed { length: 8.0 }),
            );
        });
}

fn render_tier_breakdown_table(tool: &WinRatioTool, ui: &mut egui::Ui) {
    ui.group(|ui| {
        ui.label(egui::RichText::new("Prioritized Tier Attribution Table").strong());
        egui::Grid::new("win_ratio_tier_table")
            .striped(true)
            .spacing([18.0, 6.0])
            .show(ui, |ui| {
                ui.strong("Hierarchy Tier");
                ui.strong("Strategy");
                ui.strong("Tier Wins");
                ui.strong("Tier Losses");
                ui.strong("Remaining Ties");
                ui.strong("Cumulative WR");
                ui.end_row();

                let kinds = [tool.tier1_kind, tool.tier2_kind, tool.tier3_kind];
                for (k, kind) in kinds.iter().enumerate().take(tool.eval.active_tier_count) {
                    let t = tool.eval.tiers[k];
                    ui.label(format!("Tier {}", t.tier_index));
                    ui.label(kind.label());
                    ui.label(t.tier_wins.to_string());
                    ui.label(t.tier_losses.to_string());
                    ui.label(t.remaining_ties.to_string());
                    if t.cumulative_wr.is_finite() {
                        ui.label(format!("{:.3}", t.cumulative_wr));
                    } else {
                        ui.label("∞");
                    }
                    ui.end_row();
                }
            });
    });
}

// [cite:win_ratio]
// [cite:clinical_trials_statistics]
