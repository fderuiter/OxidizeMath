//! Risk-set CSV tables, Markdown summary cards, and standalone SVG export helpers for [`SurvivalAnalysisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ArmSurvival, SurvivalAnalysisTool};
use std::fmt::Write as _;

/// Formats all arm `TimePoint` records (`survival_probability`, `n_at_risk`, `n_events`, `n_censored`) as CSV.
#[must_use]
pub fn format_risk_table_csv(arms: &[ArmSurvival]) -> String {
    let mut out = String::from("arm,time,survival_probability,n_at_risk,n_events,n_censored\n");
    for arm in arms {
        let safe_arm = sanitize_csv_cell(&arm.name);
        for pt in &arm.curve {
            let _ = writeln!(
                out,
                "{safe_arm},{:.6},{:.6},{},{},{}",
                pt.time, pt.survival_probability, pt.n_at_risk, pt.n_events, pt.n_censored
            );
        }
    }
    out
}

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
        tool.total_observations(),
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
    if let Some(ref lr_res) = tool.log_rank {
        match lr_res {
            Ok(lr) => {
                let _ = writeln!(
                    out,
                    "- **Mantel-Haenszel Log-Rank Test**: χ² = {:.4}, p = {:.4}",
                    lr.statistic, lr.p_value
                );
            }
            Err(err) => {
                let _ = writeln!(out, "- **Log-Rank Test Status**: {err}");
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

/// Collects Kaplan-Meier step curves and `TimePoint` risk-set series (`n_at_risk`, `n_events`, `n_censored`) for dataset exports.
#[must_use]
pub fn collect_export_datasets(tool: &SurvivalAnalysisTool) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut datasets = Vec::with_capacity(tool.arms.len().saturating_mul(4));
    for arm in &tool.arms {
        if arm.curve.is_empty() {
            continue;
        }
        datasets.push((
            format!("{} Survival Probability", arm.name),
            arm.build_step_points(),
        ));
        datasets.push((
            format!("{} n_at_risk", arm.name),
            arm.curve
                .iter()
                .map(|p| [p.time, p.n_at_risk as f64])
                .collect(),
        ));
        datasets.push((
            format!("{} n_events", arm.name),
            arm.curve
                .iter()
                .map(|p| [p.time, p.n_events as f64])
                .collect(),
        ));
        datasets.push((
            format!("{} n_censored", arm.name),
            arm.curve
                .iter()
                .map(|p| [p.time, p.n_censored as f64])
                .collect(),
        ));
    }
    datasets
}

/// Renders a standalone vector SVG of the active Kaplan-Meier survival step curve(s).
pub fn render_survival_svg(tool: &SurvivalAnalysisTool) -> Result<String, String> {
    let step_datasets: Vec<(String, Vec<[f64; 2]>)> = if tool.arms.len() <= 1 {
        vec![("Survival Probability".to_string(), tool.build_step_points())]
    } else {
        tool.arms
            .iter()
            .map(|arm| (format!("{} S(t)", arm.name), arm.build_step_points()))
            .collect()
    };
    let opts = egui_plot::export::ExportOptions::default();
    egui_plot::export::export_custom_svg(&step_datasets, &opts)
}

// [cite:clinical_trials_statistics]
