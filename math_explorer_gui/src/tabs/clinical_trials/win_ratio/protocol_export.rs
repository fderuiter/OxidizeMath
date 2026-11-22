//! Structured `.csv` and `.json` protocol takeaway formatters and `web_io` save helpers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::fmt::Write as _;
use std::sync::mpsc::Sender;

use math_explorer::applied::win_ratio::censoring::TierCount;
use math_explorer::applied::win_ratio::power_sweep::{PowerSweepConfig, SampleSizePowerPoint};
use math_explorer::applied::win_ratio::simulation::SimulationParams;
use math_explorer::applied::win_ratio::u_statistic::UnmatchedWinRatioSummary;

use super::cohort_io::LoadedCohortAnalysis;
use crate::web_io::{spawn_save_as_file_dialog, SaveAsDialogConfig};

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Protocol Table",
    extensions: &["csv"],
    fallback_filename: "win_ratio_protocol.csv",
    write_error_prefix: "Failed to save protocol CSV",
};

const JSON_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "JSON Protocol Report",
    extensions: &["json"],
    fallback_filename: "win_ratio_protocol.json",
    write_error_prefix: "Failed to save protocol JSON",
};

/// Borrowed inputs for formatting structured `.csv` and `.json` protocol takeaway artifacts.
#[derive(Clone, Copy)]
pub struct ProtocolExportContext<'a> {
    pub params: &'a SimulationParams,
    pub cfg: &'a PowerSweepConfig,
    pub seed: u64,
    pub theoretical_wr: f64,
    pub active_summary: &'a UnmatchedWinRatioSummary,
    pub active_tiers: &'a [TierCount; 2],
    pub cohort: Option<&'a LoadedCohortAnalysis>,
    pub sweep_points: &'a [SampleSizePowerPoint],
}

/// Formats a complete composite win-ratio protocol report as CSV containing summary parameters,
/// hierarchical tier waterfall breakdown, and Power-by-`N` sweep rows.
#[must_use]
pub fn format_protocol_csv(ctx: &ProtocolExportContext<'_>) -> String {
    let mut out = String::with_capacity(2048);
    let source = ctx
        .cohort
        .map_or("parametric_simulation", |c| c.source_name.as_str());
    let s = ctx.active_summary;
    let _ = writeln!(out, "# Composite Win Ratio Stress Tester Protocol Export");
    let _ = writeln!(
        out,
        "# source={source},seed={},lambda1={:.4},lambda2={:.4},alpha={:.4},theta={:.4},horizon={:.4},dropout={:.4},sig_alpha={:.4},theoretical_uncensored_wr={:.6}",
        ctx.seed,
        ctx.params.lambda1,
        ctx.params.lambda2,
        ctx.params.alpha,
        ctx.params.theta,
        ctx.cfg.censor_horizon,
        ctx.cfg.dropout_rate,
        ctx.cfg.sig_alpha,
        ctx.theoretical_wr
    );
    let _ = writeln!(
        out,
        "# win_ratio={:.6},ci_low={:.6},ci_high={:.6},win_odds={:.6},net_benefit={:.6},log_wr_se={:.6},z_score={:.6},p_value={:.6}",
        s.win_ratio,
        s.ci_low,
        s.ci_high,
        s.win_odds,
        s.net_benefit,
        s.log_wr_se,
        s.z_score,
        s.p_value
    );
    write_csv_tier_section(&mut out, s.total_pairs, ctx.active_tiers);
    write_csv_power_section(&mut out, ctx.sweep_points);
    out
}

fn write_csv_tier_section(out: &mut String, total_pairs: usize, tiers: &[TierCount; 2]) {
    let denom = (total_pairs.max(1)) as f64;
    let labels = ["Tier 1: Fatal Event", "Tier 2: Non-Fatal Event"];
    let _ = writeln!(
        out,
        "section,tier_index,tier_label,wins,losses,ties_remaining,win_fraction,loss_fraction"
    );
    for (idx, tier) in tiers.iter().enumerate() {
        let _ = writeln!(
            out,
            "tier_waterfall,{idx},\"{}\",{},{},{},{:.6},{:.6}",
            labels[idx],
            tier.wins,
            tier.losses,
            tier.ties_remaining,
            (tier.wins as f64) / denom,
            (tier.losses as f64) / denom,
        );
    }
}

fn write_csv_power_section(out: &mut String, points: &[SampleSizePowerPoint]) {
    let _ = writeln!(
        out,
        "\nsection,n_per_arm,total_subjects,empirical_power,mean_win_ratio,tier1_win_fraction,tier1_loss_fraction,tier2_win_fraction,tier2_loss_fraction,tie_fraction"
    );
    for pt in points {
        let _ = writeln!(
            out,
            "power_by_n,{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            pt.n_per_arm,
            pt.n_per_arm * 2,
            pt.empirical_power,
            pt.mean_win_ratio,
            pt.tier1_win_fraction,
            pt.tier1_loss_fraction,
            pt.tier2_win_fraction,
            pt.tier2_loss_fraction,
            pt.tie_fraction,
        );
    }
}

fn build_tier_json_rows(total_pairs: usize, tiers: &[TierCount; 2]) -> Vec<serde_json::Value> {
    let denom = (total_pairs.max(1)) as f64;
    let labels = ["Tier 1: Fatal Event", "Tier 2: Non-Fatal Event"];
    tiers
        .iter()
        .enumerate()
        .map(|(i, t)| {
            serde_json::json!({
                "tier_index": i,
                "label": labels[i],
                "wins": t.wins,
                "losses": t.losses,
                "ties_remaining": t.ties_remaining,
                "win_fraction": (t.wins as f64) / denom,
                "loss_fraction": (t.losses as f64) / denom,
            })
        })
        .collect()
}

fn build_power_json_rows(points: &[SampleSizePowerPoint]) -> Vec<serde_json::Value> {
    points
        .iter()
        .map(|pt| {
            serde_json::json!({
                "n_per_arm": pt.n_per_arm,
                "total_subjects": pt.n_per_arm * 2,
                "empirical_power": pt.empirical_power,
                "mean_win_ratio": pt.mean_win_ratio,
                "tier1_win_fraction": pt.tier1_win_fraction,
                "tier1_loss_fraction": pt.tier1_loss_fraction,
                "tier2_win_fraction": pt.tier2_win_fraction,
                "tier2_loss_fraction": pt.tier2_loss_fraction,
                "tie_fraction": pt.tie_fraction,
            })
        })
        .collect()
}

fn build_cohort_json(cohort: Option<&LoadedCohortAnalysis>) -> Option<serde_json::Value> {
    cohort.map(|c| {
        serde_json::json!({
            "source_name": c.source_name,
            "n_treatment": c.treatment.len(),
            "n_control": c.control.len(),
            "fatal_hazard_ratio": c.fatal_hazard_ratio.as_ref().ok(),
            "risk_difference": c.risk_difference,
            "relative_risk": c.risk_metrics.as_ref().ok().map(|r| r.relative_risk),
            "odds_ratio": c.risk_metrics.as_ref().ok().map(|r| r.odds_ratio),
        })
    })
}

/// Formats a structured JSON protocol report containing parameters, U-statistic summary,
/// optional uploaded cohort hazard/risk metrics, tier waterfall, and Power-by-`N` table.
#[must_use]
pub fn format_protocol_json(ctx: &ProtocolExportContext<'_>) -> String {
    let s = ctx.active_summary;
    let payload = serde_json::json!({
        "schema_version": "1.0",
        "tool": "WinRatioStressTesterTool",
        "parameters": {
            "lambda1": ctx.params.lambda1,
            "lambda2": ctx.params.lambda2,
            "alpha": ctx.params.alpha,
            "theta": ctx.params.theta,
            "censor_horizon": ctx.cfg.censor_horizon,
            "dropout_rate": ctx.cfg.dropout_rate,
            "sig_alpha": ctx.cfg.sig_alpha,
            "seed": ctx.seed,
            "theoretical_uncensored_win_ratio": ctx.theoretical_wr,
        },
        "u_statistic_summary": {
            "n_treatment": s.n_treatment,
            "n_control": s.n_control,
            "total_pairs": s.total_pairs,
            "wins": s.wins,
            "losses": s.losses,
            "ties": s.ties,
            "win_ratio": finite_or_null(s.win_ratio),
            "ci_low": finite_or_null(s.ci_low),
            "ci_high": finite_or_null(s.ci_high),
            "win_odds": finite_or_null(s.win_odds),
            "net_benefit": s.net_benefit,
            "log_wr_se": s.log_wr_se,
            "z_score": finite_or_null(s.z_score),
            "p_value": s.p_value,
        },
        "uploaded_cohort_secondary": build_cohort_json(ctx.cohort),
        "tier_waterfall": build_tier_json_rows(s.total_pairs, ctx.active_tiers),
        "power_by_n": build_power_json_rows(ctx.sweep_points),
    });
    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string())
}

fn finite_or_null(val: f64) -> Option<f64> {
    val.is_finite().then_some(val)
}

/// Triggers a native `rfd` save dialog or browser download for a `.csv` protocol table.
pub fn trigger_save_protocol_csv(tx: Sender<Result<String, String>>, csv_body: String) {
    spawn_save_as_file_dialog(
        tx,
        CSV_SAVE_CONFIG,
        "win_ratio_protocol.csv",
        csv_body,
        |path, fallback| {
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(fallback)
                .to_string()
        },
    );
}

/// Triggers a native `rfd` save dialog or browser download for a `.json` protocol report.
pub fn trigger_save_protocol_json(tx: Sender<Result<String, String>>, json_body: String) {
    spawn_save_as_file_dialog(
        tx,
        JSON_SAVE_CONFIG,
        "win_ratio_protocol.json",
        json_body,
        |path, fallback| {
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(fallback)
                .to_string()
        },
    );
}
