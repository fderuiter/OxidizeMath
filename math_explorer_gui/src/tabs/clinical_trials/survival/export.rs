//! Risk-set CSV tables, Markdown summary cards, and standalone SVG export helpers for [`SurvivalAnalysisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SurvivalAnalysisTool;
use std::fmt::Write as _;

/// Formats all arm `TimePoint` risk-set steps as a HIPAA-safe CSV table.
#[must_use]
pub fn format_risk_set_csv(tool: &SurvivalAnalysisTool) -> String {
    let mut out = String::with_capacity(1024);
    let _ = writeln!(
        out,
        "arm,time,n_at_risk,n_events,n_censored,survival_probability"
    );
    for arm in &tool.arms {
        for pt in &arm.curve {
            let _ = writeln!(
                out,
                "{},{:.4},{},{},{},{:.6}",
                sanitize_csv_cell(&arm.name),
                pt.time,
                pt.n_at_risk,
                pt.n_events,
                pt.n_censored,
                pt.survival_probability
            );
        }
    }
    out
}

fn sanitize_csv_cell(raw: &str) -> String {
    if raw.contains(',') || raw.contains('"') {
        format!("\"{}\"", raw.replace('"', "\"\""))
    } else {
        raw.to_string()
    }
}

/// Formats a standalone Markdown Kaplan-Meier & Hazard Ratio summary card.
#[must_use]
pub fn format_markdown_summary(tool: &SurvivalAnalysisTool) -> String {
    let mut out = String::with_capacity(1536);
    let _ = writeln!(out, "# Kaplan-Meier Survival & Risk-Set Summary Card\n");
    let _ = writeln!(
        out,
        "- **Total Observations**: {} across {} study arm(s)",
        tool.observations.len(),
        tool.arms.len()
    );
    if let Some(ref hr_res) = tool.hazard_ratio {
        match hr_res {
            Ok(hr) if hr.is_finite() => {
                let _ = writeln!(
                    out,
                    "- **Incidence-Density Hazard Ratio ({} vs {})**: {:.4}",
                    tool.arms[0].name, tool.arms[1].name, hr
                );
            }
            Ok(hr) => {
                let _ = writeln!(
                    out,
                    "- **Incidence-Density Hazard Ratio ({} vs {})**: {hr}",
                    tool.arms[0].name, tool.arms[1].name
                );
            }
            Err(err) => {
                let _ = writeln!(out, "- **Hazard Ratio Status**: {err}");
            }
        }
    }
    let _ = writeln!(out);
    append_markdown_arm_summaries(&mut out, tool);
    append_markdown_risk_set_table(&mut out, tool);
    out
}

fn append_markdown_arm_summaries(out: &mut String, tool: &SurvivalAnalysisTool) {
    let _ = writeln!(out, "## Per-Arm Cohort Summary\n");
    let _ = writeln!(
        out,
        "| Arm | Subjects | Events | Censored | Median Survival |"
    );
    let _ = writeln!(out, "| :--- | :--- | :--- | :--- | :--- |");
    for arm in &tool.arms {
        let events = arm.observations.iter().filter(|o| o.event_occurred).count();
        let censored = arm.observations.len().saturating_sub(events);
        let median_str = arm
            .median_survival_time()
            .map_or_else(|| "Not reached".to_string(), |m| format!("{m:.2}"));
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} |",
            arm.name,
            arm.observations.len(),
            events,
            censored,
            median_str
        );
    }
    let _ = writeln!(out);
}

fn append_markdown_risk_set_table(out: &mut String, tool: &SurvivalAnalysisTool) {
    let _ = writeln!(out, "## Risk-Set Trajectory (`TimePoint`)\n");
    let _ = writeln!(
        out,
        "| Arm | Time | At Risk (`n_i`) | Events (`d_i`) | Censored (`c_i`) | `S(t)` |"
    );
    let _ = writeln!(out, "| :--- | :--- | :--- | :--- | :--- | :--- |");
    for arm in &tool.arms {
        for pt in &arm.curve {
            let _ = writeln!(
                out,
                "| {} | {:.2} | {} | {} | {} | {:.4} |",
                arm.name,
                pt.time,
                pt.n_at_risk,
                pt.n_events,
                pt.n_censored,
                pt.survival_probability
            );
        }
    }
}

/// Collects step-curve `[time, S(t)]` series across all study arms for plot/SVG/CSV exports.
#[must_use]
pub fn collect_export_datasets(tool: &SurvivalAnalysisTool) -> Vec<(String, Vec<[f64; 2]>)> {
    if tool.arms.len() <= 1 {
        vec![("Survival Probability".to_string(), tool.build_step_points())]
    } else {
        tool.arms
            .iter()
            .map(|arm| (format!("{} S(t)", arm.name), arm.build_step_points()))
            .collect()
    }
}

/// Renders a standalone vector SVG of the active Kaplan-Meier curve(s).
pub fn render_survival_svg(tool: &SurvivalAnalysisTool) -> Result<String, String> {
    let datasets = collect_export_datasets(tool);
    let opts = egui_plot::export::ExportOptions::default();
    egui_plot::export::export_custom_svg(&datasets, &opts)
}

// [cite:clinical_trials_statistics]
