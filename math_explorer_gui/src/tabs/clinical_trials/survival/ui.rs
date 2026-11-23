//! Sidebar controls, multi-arm Kaplan-Meier `egui_plot::Plot`, and `TimePoint` risk-set table UI.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::export::{
    format_markdown_summary, format_risk_set_csv, format_risk_table_csv, render_survival_svg,
};
use super::{SurvivalAnalysisTool, TWO_ARM_EXAMPLE_CSV, TWO_ARM_SAMPLE_TEXT};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoints, Points};

/// Renders the complete `SurvivalAnalysisTool` split-pane workspace.
pub fn show_survival_inside(tool: &mut SurvivalAnalysisTool, ui: &mut egui::Ui) {
    tool.poll_io(ui.ctx());
    let panel_id = crate::framework::scoped_panel_id(ui, "survival_input_panel");
    crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            render_sidebar(tool, ui);
        });
    });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            render_km_plot(tool, ui);
            ui.add_space(8.0);
            ui.separator();
            render_risk_set_table(tool, ui);
        });
    });
}

fn render_sidebar(tool: &mut SurvivalAnalysisTool, ui: &mut egui::Ui) {
    ui.heading("Survival Cohort Input");
    ui.label("Single-arm: Time, Event (1=Event, 0=Censored)");
    ui.label("Multi-arm CSV/TSV: Arm, Time, Event (or Time, Event, Arm)");
    ui.add_space(4.0);
    render_cohort_preset_buttons(tool, ui);
    if render_cohort_text_inputs(tool, ui) {
        tool.recalculate();
    }
    if let Some(err) = &tool.error_message {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.colored_label(palette.error, format!("⚠ {err}"));
    }
    ui.separator();
    render_summary_metrics(tool, ui);
    ui.separator();
    render_export_actions(tool, ui);
    tool.io_state.render_status(ui);
}

fn render_cohort_preset_buttons(tool: &mut SurvivalAnalysisTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        if ui
            .button("📂 Import Cohort (.csv / .tsv)")
            .accessible_hover_text(
                "Import a local single-arm or multi-arm CSV/TSV cohort completely in memory",
            )
            .clicked()
        {
            tool.io_state.trigger_open_cohort();
        }
        if ui
            .button("📊 Load Two-Arm Example")
            .accessible_hover_text("Load Treatment vs Control survival cohort (Arm, Time, Event)")
            .clicked()
        {
            tool.input_text = TWO_ARM_EXAMPLE_CSV.to_string();
            tool.control_text.clear();
            tool.recalculate();
        }
        if ui
            .button("🧪 Load Sample (time,event,arm)")
            .accessible_hover_text("Load two-arm oncology sample in time,event,arm column order")
            .clicked()
        {
            tool.input_text = TWO_ARM_SAMPLE_TEXT.to_string();
            tool.control_text.clear();
            tool.recalculate();
        }
    });
}

fn render_cohort_text_inputs(tool: &mut SurvivalAnalysisTool, ui: &mut egui::Ui) -> bool {
    let mut changed = ui
        .add(
            egui::TextEdit::multiline(&mut tool.input_text)
                .desired_width(f32::INFINITY)
                .desired_rows(11),
        )
        .changed();

    egui::CollapsingHeader::new("Optional Separate Control Arm (time, event)")
        .default_open(!tool.control_text.is_empty())
        .show(ui, |ui| {
            changed |= ui
                .add(
                    egui::TextEdit::multiline(&mut tool.control_text)
                        .desired_width(f32::INFINITY)
                        .desired_rows(5),
                )
                .changed();
            if !tool.control_text.is_empty() && ui.button("Clear Separate Control").clicked() {
                tool.control_text.clear();
                changed = true;
            }
        });
    changed
}

fn render_summary_metrics(tool: &SurvivalAnalysisTool, ui: &mut egui::Ui) {
    render_copyable_metric(ui, "Observations", &tool.total_observations().to_string());
    render_copyable_metric(ui, "Study Arms", &tool.arms.len().to_string());

    if tool.arms.len() <= 1 {
        let median_text = tool
            .median_survival_time()
            .map_or_else(|| "Not reached".to_owned(), |t| format!("{t:.2}"));
        render_copyable_metric(ui, "Median Survival", &median_text);
    } else {
        for arm in &tool.arms {
            let med = arm
                .median_survival_time()
                .map_or_else(|| "Not reached".to_owned(), |t| format!("{t:.2}"));
            render_copyable_metric(
                ui,
                &format!("{} Median (N={})", arm.name, arm.observations.len()),
                &med,
            );
        }
    }

    if let Some(ref hr_res) = tool.hazard_ratio {
        let hr_str = match hr_res {
            Ok(hr) if hr.is_finite() => format!("{hr:.3}"),
            Ok(hr) => format!("{hr}"),
            Err(err) => format!("N/A ({err})"),
        };
        render_copyable_metric(
            ui,
            &format!("Hazard Ratio ({}/{})", tool.arms[0].name, tool.arms[1].name),
            &hr_str,
        );
    }

    if let Some(ref lr_res) = tool.log_rank {
        match lr_res {
            Ok(lr) => {
                render_copyable_metric(ui, "Log-Rank χ²", &format!("{:.3}", lr.statistic));
                render_copyable_metric(ui, "Log-Rank p-value", &format!("{:.4}", lr.p_value));
            }
            Err(err) => {
                let palette = SemanticPalette::from_visuals(ui.visuals());
                ui.colored_label(palette.warning, err);
            }
        }
    }
}

fn render_export_actions(tool: &mut SurvivalAnalysisTool, ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("HIPAA-Safe Local Exports").strong());
    ui.horizontal_wrapped(|ui| {
        if ui
            .button("⬇ Risk-Set (.csv)")
            .accessible_hover_text("Download Kaplan-Meier TimePoint risk-set table as CSV")
            .clicked()
        {
            let csv = format_risk_set_csv(tool);
            tool.io_state
                .trigger_save_csv("kaplan_meier_risk_set.csv", csv);
        }
        if ui
            .button("⬇ Summary (.md)")
            .accessible_hover_text("Download Markdown survival & hazard ratio summary card")
            .clicked()
        {
            let md = format_markdown_summary(tool);
            tool.io_state
                .trigger_save_markdown("kaplan_meier_summary.md", md);
        }
        if ui
            .button("⬇ Figure (.svg)")
            .accessible_hover_text("Download standalone vector SVG Kaplan-Meier plot")
            .clicked()
        {
            match render_survival_svg(tool) {
                Ok(svg) => tool
                    .io_state
                    .trigger_save_svg("kaplan_meier_curve.svg", svg),
                Err(err) => tool.io_state.record_error(err),
            }
        }
        if ui
            .button("📋 Copy Risk CSV")
            .accessible_hover_text("Copy Kaplan-Meier risk table CSV to clipboard")
            .clicked()
        {
            ui.ctx().copy_text(format_risk_table_csv(&tool.arms));
            tool.io_state
                .record_info("Copied risk-table CSV to clipboard".into());
        }
        if ui
            .button("📋 Copy Summary")
            .accessible_hover_text("Copy Markdown summary card to clipboard")
            .clicked()
        {
            ui.ctx().copy_text(format_markdown_summary(tool));
            tool.io_state
                .record_info("Copied survival summary card to clipboard".into());
        }
    });
}

fn render_km_plot(tool: &SurvivalAnalysisTool, ui: &mut egui::Ui) {
    ui.heading("Kaplan-Meier Survival Curve");
    let palette = SemanticPalette::from_visuals(ui.visuals());

    Plot::new("kaplan_meier_plot")
        .height(320.0)
        .legend(Legend::default())
        .x_axis_label("Time")
        .y_axis_label("Survival Probability S(t)")
        .include_y(0.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            for (idx, arm) in tool.arms.iter().enumerate() {
                let color = palette.series_color(idx * 2);
                let label = if tool.arms.len() == 1 {
                    "Survival Probability".to_string()
                } else {
                    format!("{} S(t)", arm.name)
                };
                plot_ui.line(
                    Line::new(label, PlotPoints::new(arm.build_step_points()))
                        .color(color)
                        .width(2.2_f32),
                );
                let censor_pts = arm.build_censor_points();
                if !censor_pts.is_empty() {
                    plot_ui.points(
                        Points::new(format!("{} Censored", arm.name), censor_pts)
                            .shape(MarkerShape::Plus)
                            .radius(4.5_f32)
                            .color(color),
                    );
                }
            }
            plot_ui.hline(
                HLine::new("S(t) = 0.50", 0.50)
                    .color(palette.warning)
                    .style(LineStyle::Dashed { length: 8.0 }),
            );
        });
}

fn render_risk_set_table(tool: &SurvivalAnalysisTool, ui: &mut egui::Ui) {
    ui.heading("Risk-Set Trajectory Table (TimePoint)");
    if tool.arms.is_empty() {
        ui.label("No valid survival observations to tabulate.");
        return;
    }
    egui::ScrollArea::vertical()
        .id_salt("km_risk_set_scroll")
        .max_height(210.0)
        .show(ui, |ui| {
            egui::Grid::new("km_risk_set_grid")
                .striped(true)
                .spacing([18.0, 4.0])
                .show(ui, |ui| {
                    ui.strong("Arm");
                    ui.strong("Time (t)");
                    ui.strong("At Risk (n_i)");
                    ui.strong("Events (d_i)");
                    ui.strong("Censored (c_i)");
                    ui.strong("Survival S(t)");
                    ui.end_row();
                    for arm in &tool.arms {
                        for pt in &arm.curve {
                            ui.label(&arm.name);
                            ui.label(format!("{:.2}", pt.time));
                            ui.label(pt.n_at_risk.to_string());
                            ui.label(pt.n_events.to_string());
                            ui.label(pt.n_censored.to_string());
                            ui.label(format!("{:.4}", pt.survival_probability));
                            ui.end_row();
                        }
                    }
                });
        });
}

// [cite:clinical_trials_statistics]
