//! Inverse challenge `copula_heavy_tail_kelly` for `CopulaTailRiskLab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CopulaTailRiskConfig, CopulaTailRiskLab};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};

/// Canonical challenge ID for the heavy-tail copula SGP & Kelly sizing inverse problem.
pub const CHALLENGE_COPULA_TAIL_KELLY: &str = "copula_heavy_tail_kelly";

const LOCKED_LEG1_PROB: f64 = 0.20;
const LOCKED_LEG2_PROB: f64 = 0.20;
const LOCKED_TAIL_U: f64 = 0.999;
const LOCKED_DOF: f64 = 3.0;
const LOCKED_DECIMAL_ODDS: f64 = 18.0;

impl CopulaTailRiskLab {
    pub(crate) fn enforce_challenge_locks(&mut self) {
        if self.active_challenge == Some(CHALLENGE_COPULA_TAIL_KELLY) {
            self.config.leg1_prob = LOCKED_LEG1_PROB;
            self.config.leg2_prob = LOCKED_LEG2_PROB;
            self.config.tail_quantile_u = LOCKED_TAIL_U;
            self.config.degrees_of_freedom = LOCKED_DOF;
            self.config.offered_decimal_odds = LOCKED_DECIMAL_ODDS;
        }
    }

    pub(crate) fn set_challenge_mode(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.recalculate();
                Ok(())
            }
            Some(CHALLENGE_COPULA_TAIL_KELLY) => {
                self.active_challenge = Some(CHALLENGE_COPULA_TAIL_KELLY);
                self.recalculate();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for CopulaTailRiskLab"
            )),
        }
    }

    pub(crate) fn eval_active_challenge(&self) -> Option<ChallengeEvaluation> {
        if self.active_challenge != Some(CHALLENGE_COPULA_TAIL_KELLY) {
            return None;
        }
        let m = &self.metrics;
        let growth_deficit = (0.010 - m.true_expected_growth).max(0.0) * 50.0;
        let tail_deficit = (3.50 - m.tail_multiplier).max(0.0) * 0.4;
        let overbet_excess = (self.config.stake_multiplier - 0.85).max(0.0) * 2.0;
        let residual = growth_deficit + tail_deficit + overbet_excess;
        let passed = m.true_expected_growth >= 0.010
            && m.tail_multiplier >= 3.50
            && self.config.stake_multiplier <= 0.85;
        let score = if passed {
            100.0
        } else {
            (100.0 * (1.0 - 0.65 * residual)).clamp(0.0, 99.0)
        };
        Some(ChallengeEvaluation::new(
            score,
            passed,
            "Δg + ΔTail + Overbet",
            residual,
            format!(
                "g(f) = {:.4} (need >= 0.0100), Tail ratio = {:.2}x (need >= 3.50x), Stake mult = {:.2}x (need <= 0.85x)",
                m.true_expected_growth, m.tail_multiplier, self.config.stake_multiplier
            ),
        ))
    }
}

fn generate_copula_challenge_baseline() -> String {
    let mut lab = CopulaTailRiskLab {
        config: CopulaTailRiskConfig {
            leg1_prob: LOCKED_LEG1_PROB,
            leg2_prob: LOCKED_LEG2_PROB,
            correlation_rho: -0.40,
            tail_quantile_u: LOCKED_TAIL_U,
            degrees_of_freedom: LOCKED_DOF,
            offered_decimal_odds: LOCKED_DECIMAL_ODDS,
            stake_multiplier: 1.80,
        },
        active_challenge: Some(CHALLENGE_COPULA_TAIL_KELLY),
        ..CopulaTailRiskLab::default()
    };
    lab.recalculate();
    lab.save_state().unwrap_or_default()
}

fn clamp_copula_challenge_state(raw_json: &str) -> Result<String, String> {
    let parsed: CopulaTailRiskConfig =
        serde_json::from_str(raw_json).map_err(|e| format!("Invalid Copula JSON: {e}"))?;
    let mut lab = CopulaTailRiskLab {
        config: parsed,
        active_challenge: Some(CHALLENGE_COPULA_TAIL_KELLY),
        ..CopulaTailRiskLab::default()
    };
    lab.clamp_config_in_place();
    lab.save_state()
        .ok_or_else(|| "Failed to serialize clamped Copula state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_COPULA_TAIL_KELLY,
        title: "Heavy-Tail SGP & Sub-Kelly Calibration",
        description: "Identify positive latent correlation ρ and a disciplined sub-Kelly stake multiplier (<= 0.85x) on an 18.0x decimal parlay under ν=3 Student-t tail clustering so g(f) >= 0.010 with >= 3.50x tail amplification.",
        category: "Quantitative Finance",
        icon: "🎲",
        domain: "financial_math",
        tool_name: "CopulaTailRiskLab",
        locked_inputs: "p₁ = p₂ = 0.20, Odds = 18.0x, Student-t ν = 3.0, Tail quantile u = 0.999",
        tunable_budget: "Correlation ρ ∈ [-0.95, 0.95], Kelly stake multiplier ∈ [0.10, 2.50]",
        target_summary: "Expected growth g(f) >= 0.010, Tail ratio >= 3.50x, Stake mult <= 0.85x",
        baseline_state_generator: generate_copula_challenge_baseline,
        clamp_locked_state: clamp_copula_challenge_state,
    }
}

// [cite:clinical_trials_statistics]
