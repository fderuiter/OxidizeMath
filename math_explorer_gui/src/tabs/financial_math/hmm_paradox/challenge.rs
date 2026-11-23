//! Inverse challenge `hmm_trellis_branch_rescue` for `HiddenMarkovParadoxTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HiddenMarkovParadoxTool, HmmParadoxConfig};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};

/// Canonical challenge ID for the HMM forbidden-trellis time-travel rescue problem.
pub const CHALLENGE_HMM_FORBIDDEN_FORK: &str = "hmm_trellis_branch_rescue";

const LOCKED_OBSERVATIONS: [usize; 3] = [0, 1, 1];
const LOCKED_SHOCK_SELF_LOOP: f64 = 0.0;

impl HiddenMarkovParadoxTool {
    pub(crate) fn enforce_challenge_locks(&mut self) {
        if self.active_challenge == Some(CHALLENGE_HMM_FORBIDDEN_FORK) {
            self.config.observations = LOCKED_OBSERVATIONS.to_vec();
            self.config.shock_self_loop_prob = LOCKED_SHOCK_SELF_LOOP;
            self.config.fork_step = self.config.fork_step.min(LOCKED_OBSERVATIONS.len() - 1);
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
            Some(CHALLENGE_HMM_FORBIDDEN_FORK) => {
                self.active_challenge = Some(CHALLENGE_HMM_FORBIDDEN_FORK);
                self.recalculate();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for HiddenMarkovParadoxTool"
            )),
        }
    }

    pub(crate) fn eval_active_challenge(&self) -> Option<ChallengeEvaluation> {
        if self.active_challenge != Some(CHALLENGE_HMM_FORBIDDEN_FORK) {
            return None;
        }
        let baseline_paradox_preserved = self.baseline.has_forbidden_marginal_step;
        let branched_reconciled = !self.branched.has_forbidden_marginal_step
            && self.branched.marginal_path == self.branched.viterbi_path;
        let likelihood_deficit = (0.12 - self.branched.forward_likelihood).max(0.0) * 8.0;
        let penalty_base = if baseline_paradox_preserved { 0.0 } else { 1.0 };
        let penalty_branch = if branched_reconciled { 0.0 } else { 1.0 };
        let residual = likelihood_deficit + penalty_base + penalty_branch;
        let passed = baseline_paradox_preserved
            && branched_reconciled
            && self.branched.forward_likelihood >= 0.12;
        let score = if passed {
            100.0
        } else {
            (100.0 * (1.0 - 0.45 * residual)).clamp(0.0, 99.0)
        };
        Some(ChallengeEvaluation::new(
            score,
            passed,
            "ΔL_branch + PathMismatch",
            residual,
            format!(
                "Baseline paradox = {} (need true), Branched MAP==Viterbi = {} (need true), Branched P(Y') = {:.4} (need >= 0.1200)",
                baseline_paradox_preserved,
                branched_reconciled,
                self.branched.forward_likelihood
            ),
        ))
    }
}

fn generate_hmm_challenge_baseline() -> String {
    let mut tool = HiddenMarkovParadoxTool {
        config: HmmParadoxConfig {
            observations: LOCKED_OBSERVATIONS.to_vec(),
            fork_step: 2,
            fork_token: 1, // Identical to baseline [0, 1, 1], so branched still hits forbidden MAP!
            shock_self_loop_prob: LOCKED_SHOCK_SELF_LOOP,
            bridge_to_shock_prob: 0.95,
            shock_emission_focus: 0.85,
            seed_counter: 0,
        },
        active_challenge: Some(CHALLENGE_HMM_FORBIDDEN_FORK),
        ..HiddenMarkovParadoxTool::default()
    };
    tool.recalculate();
    tool.save_state().unwrap_or_default()
}

fn clamp_hmm_challenge_state(raw_json: &str) -> Result<String, String> {
    let parsed: HmmParadoxConfig =
        serde_json::from_str(raw_json).map_err(|e| format!("Invalid HMM JSON: {e}"))?;
    let mut tool = HiddenMarkovParadoxTool {
        config: parsed,
        active_challenge: Some(CHALLENGE_HMM_FORBIDDEN_FORK),
        ..HiddenMarkovParadoxTool::default()
    };
    tool.clamp_config_in_place();
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped HMM state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_HMM_FORBIDDEN_FORK,
        title: "HMM Time-Travel Trellis Rescue",
        description: "With forbidden A[1,1] = 0 and baseline Y = [0, 1, 1] locked, choose a single observation fork (t_fork, y'_t) and emission/bridge focus so thebaseline preserves the forbidden MAP trap while the forked timeline reconciles MAP == Viterbi with P(Y') >= 0.12.",
        category: "Stochastic Processes",
        icon: "🔀",
        domain: "financial_math",
        tool_name: "HiddenMarkovParadoxTool",
        locked_inputs: "Baseline Y = [0, 1, 1], Forbidden self-loop A[1, 1] = 0.0",
        tunable_budget: "Fork step t_fork ∈ {0,1,2}, Fork token y' ∈ {0,1,2}, A[2,1] ∈ [0.10, 0.95], B[1,1] ∈ [0.30, 0.90]",
        target_summary: "Baseline MAP forbidden, Branched MAP == Viterbi (valid), and Branched P(Y') >= 0.12",
        baseline_state_generator: generate_hmm_challenge_baseline,
        clamp_locked_state: clamp_hmm_challenge_state,
    }
}

// [cite:clinical_trials_statistics]
