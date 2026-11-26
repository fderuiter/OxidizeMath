//! Protocol sample-size sizing table `.csv`/`.md` formatters and plot dataset exporters.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CalculationMode, SampleSizeCalculatorTool};
use std::fmt::Write as _;

/// Formats a standalone CSV protocol sample-size & power schedule.
#[must_use]
pub fn format_protocol_csv(tool: &SampleSizeCalculatorTool) -> String {
    let mut out = String::with_capacity(1024);
    let _ = writeln!(
        out,
        "mode,alpha,effect_param_1,effect_param_2,target_power,n_per_arm,total_subjects"
    );
    let (mode_tag, p1, p2) = match tool.mode {
        CalculationMode::Means => ("Means", tool.delta, tool.sigma),
        CalculationMode::Proportions => ("Proportions", tool.p1, tool.p2),
    };
    if let Ok(active_n) = tool.result {
        let _ = writeln!(
            out,
            "{mode_tag},{:.4},{p1:.4},{p2:.4},{:.4},{active_n},{}",
            tool.alpha,
            tool.power,
            active_n.saturating_mul(2)
        );
    }
    for row in &tool.protocol_rows {
        let _ = writeln!(
            out,
            "{mode_tag},{:.4},{p1:.4},{p2:.4},{:.4},{},{}",
            tool.alpha, row.power, row.n_per_arm, row.total_n
        );
    }
    out
}

/// Formats a standalone Markdown clinical protocol sample-size sizing card.
#[must_use]
pub fn format_protocol_markdown(tool: &SampleSizeCalculatorTool) -> String {
    let mut out = String::with_capacity(1280);
    let _ = writeln!(
        out,
        "# Clinical Trial Sample Size & Statistical Power Protocol Card\n"
    );
    let _ = writeln!(out, "- **Hypothesis Test Mode**: {}", tool.mode.label());
    let _ = writeln!(
        out,
        "- **Significance Level (`α`, Two-Sided)**: {:.4}",
        tool.alpha
    );
    let _ = writeln!(
        out,
        "- **Selected Target Power (`1 - β`)**: {:.1}%",
        tool.power * 100.0
    );
    match tool.mode {
        CalculationMode::Means => {
            let _ = writeln!(
                out,
                "- **Continuous Assumptions**: Effect Size `δ = {:.3}`, Common SD `σ = {:.3}` (Standardized `δ/σ = {:.3}`)",
                tool.delta,
                tool.sigma,
                tool.delta / tool.sigma.max(1e-12)
            );
        }
        CalculationMode::Proportions => {
            let _ = writeln!(
                out,
                "- **Binary Assumptions**: Control `p₁ = {:.3}`, Treatment `p₂ = {:.3}` (`|p₂ - p₁| = {:.3}`)",
                tool.p1,
                tool.p2,
                (tool.p2 - tool.p1).abs()
            );
        }
    }
    match &tool.result {
        Ok(n) => {
            let _ = writeln!(
                out,
                "- **Required Enrollment (1:1 Allocation)**: **{n} per arm** (**{} total subjects**)\n",
                n.saturating_mul(2)
            );
        }
        Err(err) => {
            let _ = writeln!(out, "- **Status**: Error — {err}\n");
        }
    }
    append_protocol_milestone_table(&mut out, tool);
    out
}

fn append_protocol_milestone_table(out: &mut String, tool: &SampleSizeCalculatorTool) {
    let _ = writeln!(out, "## Protocol Power Milestones (`1 - β` vs `N`)\n");
    let _ = writeln!(
        out,
        "| Target Power (`1 - β`) | Subjects per Arm (`N`) | Total Enrollment (`2N`) |"
    );
    let _ = writeln!(out, "| :--- | :--- | :--- |");
    for row in &tool.protocol_rows {
        let _ = writeln!(
            out,
            "| {:.0}% ({:.2}) | {} | {} |",
            row.power * 100.0,
            row.power,
            row.n_per_arm,
            row.total_n
        );
    }
}

/// Collects analytical power and required sample-size curve datasets for `ExportCapabilities::plot()`.
#[must_use]
pub fn collect_export_datasets(tool: &SampleSizeCalculatorTool) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut ds = Vec::with_capacity(3);
    if !tool.power_curve.is_empty() {
        ds.push((
            "Analytical Power Curve".to_string(),
            tool.power_curve.clone(),
        ));
    }
    if !tool.per_arm_curve.is_empty() {
        ds.push((
            "Sample Size per Group (n)".to_string(),
            tool.per_arm_curve.clone(),
        ));
        ds.push((
            "Total Study Size (2n)".to_string(),
            tool.total_curve.clone(),
        ));
    }
    ds
}

// [cite:clinical_trials_statistics]
