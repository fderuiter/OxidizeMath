//! Inverse challenge `win_ratio_severity_reversal` for `WinRatioParadoxLab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{WinRatioParadoxConfig, WinRatioParadoxLab};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};

/// Canonical challenge ID for the composite-endpoint severity reversal inverse problem.
pub const CHALLENGE_WIN_RATIO_REVERSAL: &str = "win_ratio_severity_reversal";

const LOCKED_SAMPLE_SIZE: usize = 20;
const LOCKED_HORIZON_MONTHS: f64 = 36.0;
const LOCKED_CONTROL_MORTALITY: f64 = 0.60;

impl WinRatioParadoxLab {
    pub(crate) fn enforce_challenge_locks(&mut self) {
        if self.active_challenge == Some(CHALLENGE_WIN_RATIO_REVERSAL) {
            self.config.sample_size_per_arm = LOCKED_SAMPLE_SIZE;
            self.config.horizon_months = LOCKED_HORIZON_MONTHS;
            self.config.control_mortality_rate = LOCKED_CONTROL_MORTALITY;
            self.config.matched_pairing = false;
            self.config.seed_counter = 0;
        }
    }

    pub(crate) fn set_challenge_mode(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.recalculate();
                Ok(())
            }
            Some(CHALLENGE_WIN_RATIO_REVERSAL) => {
                self.active_challenge = Some(CHALLENGE_WIN_RATIO_REVERSAL);
                self.recalculate();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for WinRatioParadoxLab"
            )),
        }
    }

    pub(crate) fn eval_active_challenge(&self) -> Option<ChallengeEvaluation> {
        if self.active_challenge != Some(CHALLENGE_WIN_RATIO_REVERSAL) {
            return None;
        }
        let m = &self.metrics;
        let hr_deficit = (2.50 - m.pooled_hazard_ratio).max(0.0);
        let wr_deficit = (1.50 - m.win_ratio).max(0.0);
        let p_excess = (m.win_ratio_p_value - 0.001).max(0.0) * 100.0;
        let residual = hr_deficit + wr_deficit + p_excess;
        let passed =
            m.pooled_hazard_ratio >= 2.50 && m.win_ratio >= 1.50 && m.win_ratio_p_value <= 0.001;
        let score = if passed {
            100.0
        } else {
            (100.0 * (1.0 - 0.55 * residual)).clamp(0.0, 99.0)
        };
        Some(ChallengeEvaluation::new(
            score,
            passed,
            "ΔHR + ΔWR + Δp",
            residual,
            format!(
                "Pooled HR = {:.2} (need >= 2.50), Prioritized WR = {:.2} (need >= 1.50), p = {:.2e} (need <= 1e-3)",
                m.pooled_hazard_ratio, m.win_ratio, m.win_ratio_p_value
            ),
        ))
    }
}

fn generate_win_ratio_challenge_baseline() -> String {
    let mut lab = WinRatioParadoxLab {
        config: WinRatioParadoxConfig {
            sample_size_per_arm: LOCKED_SAMPLE_SIZE,
            horizon_months: LOCKED_HORIZON_MONTHS,
            control_mortality_rate: LOCKED_CONTROL_MORTALITY,
            mortality_reduction: 0.15,
            early_hosp_rate: 0.20,
            early_hosp_mean_months: 12.0,
            matched_pairing: false,
            seed_counter: 0,
        },
        active_challenge: Some(CHALLENGE_WIN_RATIO_REVERSAL),
        ..WinRatioParadoxLab::default()
    };
    lab.recalculate();
    lab.save_state().unwrap_or_default()
}

fn clamp_win_ratio_challenge_state(raw_json: &str) -> Result<String, String> {
    let parsed: WinRatioParadoxConfig =
        serde_json::from_str(raw_json).map_err(|e| format!("Invalid WinRatio JSON: {e}"))?;
    let mut lab = WinRatioParadoxLab {
        config: parsed,
        active_challenge: Some(CHALLENGE_WIN_RATIO_REVERSAL),
        ..WinRatioParadoxLab::default()
    };
    lab.clamp_config_in_place();
    lab.save_state()
        .ok_or_else(|| "Failed to serialize clamped WinRatio state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_WIN_RATIO_REVERSAL,
        title: "Composite Severity Reversal",
        description: "Configure mortality reduction and early non-fatal hospitalization dynamics so pooled time-to-first-event HR >= 2.50 penalizes treatment while prioritized Win Ratio >= 1.50 (p <= 0.001) proves significant benefit.",
        category: "Clinical Trials",
        icon: "⚖",
        domain: "clinical_trials",
        tool_name: "WinRatioParadoxLab",
        locked_inputs: "N = 20/arm, Follow-up = 36 mo, Control mortality = 60%, Unmatched all-pairs",
        tunable_budget: "Mortality reduction ∈ [0, 0.90], Early hosp rate ∈ [0, 0.95], Mean hosp onset ∈ [1, 24] mo",
        target_summary: "Pooled HR >= 2.50 AND Prioritized WR >= 1.50 with p <= 0.001",
        baseline_state_generator: generate_win_ratio_challenge_baseline,
        clamp_locked_state: clamp_win_ratio_challenge_state,
    }
}

// [cite:win_ratio]
