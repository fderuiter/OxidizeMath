//! HIPAA-safe Markdown and CSV composite Win Ratio report generators and plot dataset exporters.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cohort::WinRatioPlotMode;
use super::WinRatioTool;
use std::fmt::Write as _;

fn format_finite_or_inf(val: f64) -> String {
    if val.is_finite() {
        format!("{val:.4}")
    } else if val.is_sign_positive() {
        "Inf".to_string()
    } else {
        "N/A".to_string()
    }
}

/// Formats a standalone HIPAA-safe Markdown composite Win Ratio & Weibull `PR(c)` report.
#[must_use]
pub fn format_markdown_report(tool: &WinRatioTool) -> String {
    let mut out = String::with_capacity(1536);
    let _ = writeln!(
        out,
        "# Composite Endpoint Win Ratio & Weibull PR(c) Report\n"
    );
    let _ = writeln!(
        out,
        "- **Pairing Scheme**: {} (Treatment N = {}, Control N = {}, Total Pairs = {})",
        tool.pairing_mode.label(),
        tool.parsed_cohort.treatment.len(),
        tool.parsed_cohort.control.len(),
        tool.eval.total_pairs
    );
    let _ = writeln!(
        out,
        "- **Tier Strategies**: Tier 1 = {}, Tier 2 = {}, Tier 3 = {}\n",
        tool.tier1_kind.label(),
        tool.tier2_kind.label(),
        tool.tier3_kind.label()
    );

    append_markdown_empirical_section(&mut out, tool);
    append_markdown_secondary_and_weibull(&mut out, tool);
    out
}

fn append_markdown_empirical_section(out: &mut String, tool: &WinRatioTool) {
    let ev = &tool.eval;
    let _ = writeln!(out, "## Empirical Composite Win Ratio Summary\n");
    let _ = writeln!(out, "| Metric | Value |");
    let _ = writeln!(out, "| :--- | :--- |");
    let _ = writeln!(
        out,
        "| Wins / Losses / Ties | {} / {} / {} |",
        ev.wins, ev.losses, ev.ties
    );
    if let Some(stats) = ev.stats {
        let _ = writeln!(
            out,
            "| Composite Win Ratio (WR) | {} [95% CI: {}, {}] |",
            format_finite_or_inf(stats.win_ratio),
            format_finite_or_inf(stats.ci_low),
            format_finite_or_inf(stats.ci_high)
        );
        let _ = writeln!(out, "| Two-Sided P-Value | {:.4} |", stats.p_value);
    }
    if let Some(counts) = ev.pocock.counts {
        let _ = writeln!(
            out,
            "| Pocock 2-Tier Counts (Na, Nb, Nc, Nd) | ({}, {}, {}, {}) |",
            counts.n_a, counts.n_b, counts.n_c, counts.n_d
        );
        let _ = writeln!(
            out,
            "| Pocock Sample WR & Z-Stat | {} [95% CI: {}, {}] (pw = {:.3}, Z = {:.3}) |\n",
            format_finite_or_inf(ev.pocock.sample_wr),
            format_finite_or_inf(ev.pocock.ci.0),
            format_finite_or_inf(ev.pocock.ci.1),
            ev.pocock.win_proportion,
            ev.pocock.z_statistic
        );
    }

    let _ = writeln!(out, "### Per-Tier Hierarchy Attribution\n");
    let _ = writeln!(
        out,
        "| Tier | Wins | Losses | Remaining Ties | Cumulative WR [95% CI] |"
    );
    let _ = writeln!(out, "| :--- | :--- | :--- | :--- | :--- |");
    for &t in &ev.tiers[..ev.active_tier_count] {
        let _ = writeln!(
            out,
            "| Tier {} | {} | {} | {} | {} [{}, {}] |",
            t.tier_index,
            t.tier_wins,
            t.tier_losses,
            t.remaining_ties,
            format_finite_or_inf(t.cumulative_wr),
            format_finite_or_inf(t.cumulative_ci_low),
            format_finite_or_inf(t.cumulative_ci_high)
        );
    }
    let _ = writeln!(out);
}

fn append_markdown_secondary_and_weibull(out: &mut String, tool: &WinRatioTool) {
    let ev = &tool.eval;
    let _ = writeln!(
        out,
        "## Secondary Continuous Endpoint & Bivariate Weibull PR(c)\n"
    );
    if let Some(ref tt) = ev.ttest_result {
        let (ci_lo, ci_hi) = tt.confidence_interval.unwrap_or((0.0, 0.0));
        let _ = writeln!(
            out,
            "- **Independent Two-Sample t-Test (Tier {}, α = {:.3})**: t = {:.4}, p = {:.4}, Δ Mean 95% CI = [{:.3}, {:.3}], significant = {}",
            tool.ttest_tier_index + 1,
            tool.ttest_alpha,
            tt.statistic,
            tt.p_value,
            ci_lo,
            ci_hi,
            tt.is_significant
        );
    }
    let _ = writeln!(
        out,
        "- **Reference BMI Covariate**: Treatment = {:.2} kg/m², Control = {:.2} kg/m²",
        ev.treat_ref_bmi, ev.ctrl_ref_bmi
    );
    let _ = writeln!(
        out,
        "- **Bivariate Weibull Parameters**: λ₁ = {:.3}, λ₂ = {:.3}, α = {:.3}, θ = {:.3}, Horizon c = {:.2}",
        tool.weibull_lambda1,
        tool.weibull_lambda2,
        tool.weibull_alpha,
        tool.weibull_theta,
        tool.horizon_c
    );
    let _ = writeln!(
        out,
        "- **Probability Win Ratio**: Uncensored PR_W = {:.4}, PR(c_max) = {:.4} (W(c) = {:.4}, L(c) = {:.4})",
        ev.weibull_pr_w, ev.terminal_pr_c, ev.terminal_w_c, ev.terminal_l_c
    );
}

/// Formats a standalone CSV composite Win Ratio & Weibull `PR(c)` report.
#[must_use]
pub fn format_csv_report(tool: &WinRatioTool) -> String {
    let ev = &tool.eval;
    let mut out = String::with_capacity(2048);
    let _ = writeln!(out, "section,key,x_or_tier,value,ci_low,ci_high,extra");
    let _ = writeln!(out, "summary,wins,,{},,,", ev.wins);
    let _ = writeln!(out, "summary,losses,,{},,,", ev.losses);
    let _ = writeln!(out, "summary,ties,,{},,,", ev.ties);
    if let Some(stats) = ev.stats {
        let _ = writeln!(
            out,
            "summary,win_ratio,,{},{},{},p={:.6}",
            format_finite_or_inf(stats.win_ratio),
            format_finite_or_inf(stats.ci_low),
            format_finite_or_inf(stats.ci_high),
            stats.p_value
        );
    }
    if let Some(ref tt) = ev.ttest_result {
        let (lo, hi) = tt.confidence_interval.unwrap_or((0.0, 0.0));
        let _ = writeln!(
            out,
            "ttest,tier_{}_t_stat,{},{:.6},{:.6},{:.6},p={:.6}",
            tool.ttest_tier_index + 1,
            tool.ttest_tier_index + 1,
            tt.statistic,
            lo,
            hi,
            tt.p_value
        );
    }
    let _ = writeln!(out, "bmi,treatment_ref,,{:.4},,,kg/m2", ev.treat_ref_bmi);
    let _ = writeln!(out, "bmi,control_ref,,{:.4},,,kg/m2", ev.ctrl_ref_bmi);
    for &t in &ev.tiers[..ev.active_tier_count] {
        let _ = writeln!(
            out,
            "tier,tier_{},{},{},{},{},wins={};losses={};ties={}",
            t.tier_index,
            t.tier_index,
            format_finite_or_inf(t.cumulative_wr),
            format_finite_or_inf(t.cumulative_ci_low),
            format_finite_or_inf(t.cumulative_ci_high),
            t.tier_wins,
            t.tier_losses,
            t.remaining_ties
        );
    }
    for &[c, pr] in &tool.prc_points {
        let _ = writeln!(
            out,
            "weibull_prc,PR(c),{c:.4},{pr:.6},,,PR_W={:.6}",
            ev.weibull_pr_w
        );
    }
    out
}

/// Collects 2D `(X, Y)` curve series for `ExportCapabilities::plot()` (`CSV`, `JSON`, `PNG`, `SVG`).
#[must_use]
pub fn collect_export_datasets(tool: &WinRatioTool) -> Vec<(String, Vec<[f64; 2]>)> {
    let mut ds = Vec::new();
    match tool.plot_mode {
        WinRatioPlotMode::WeibullPrc => {
            if !tool.prc_points.is_empty() {
                ds.push((
                    "Probability Win Ratio PR(c)".into(),
                    tool.prc_points.clone(),
                ));
            }
            if !tool.surv_treat_points.is_empty() {
                ds.push((
                    "Treatment Survival S1(t)".into(),
                    tool.surv_treat_points.clone(),
                ));
            }
            if !tool.surv_ctrl_points.is_empty() {
                ds.push((
                    "Control Survival S0(t)".into(),
                    tool.surv_ctrl_points.clone(),
                ));
            }
        }
        WinRatioPlotMode::EmpiricalHierarchy => {
            let mut pts = Vec::with_capacity(tool.eval.active_tier_count);
            for (k, &t) in tool.eval.tiers[..tool.eval.active_tier_count]
                .iter()
                .enumerate()
            {
                let wr = t.cumulative_wr;
                pts.push([(k + 1) as f64, if wr.is_finite() { wr } else { 10.0 }]);
            }
            if !pts.is_empty() {
                ds.push(("Cumulative Composite Win Ratio".into(), pts));
            }
            if !tool.prc_points.is_empty() {
                ds.push(("Weibull PR(c) Reference".into(), tool.prc_points.clone()));
            }
        }
    }
    ds
}

// [cite:win_ratio]
// [cite:clinical_trials_statistics]
