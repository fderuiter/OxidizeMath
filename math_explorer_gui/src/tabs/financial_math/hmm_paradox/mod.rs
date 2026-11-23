//! Standalone `HiddenMarkovParadoxTool` workbench in `FinancialMathTab` demonstrating why
//! pointwise marginal smoothing (`posterior_probabilities`) can select a zero-probability
//! forbidden state path while dynamic-programming `viterbi` recovers the valid joint path,
//! paired with interactive time-travel observation branching.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod challenge;
pub mod model;
pub mod plots;
pub mod ui;

#[cfg(test)]
mod tests;

use crate::framework::{
    CalloutManager, ChallengeEvaluation, ExportCapabilities, InteractiveTool, ToolStateBounds,
};
use eframe::egui;
use math_explorer::pure_math::statistics::markov::hmm::{HiddenMarkovModel, HmmScratch};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Number of hidden market states (`0 = Calm`, `1 = Shock`, `2 = Bridge`).
pub const NUM_HMM_STATES: usize = 3;
/// Number of discrete observation tokens (`0 = LowVol`, `1 = Spike`, `2 = Reversal`).
pub const NUM_HMM_OBS: usize = 3;
/// Maximum observation sequence length supported without reallocating `HmmScratch`.
pub const MAX_HMM_STEPS: usize = 12;

/// Human-readable hidden state labels.
pub const STATE_NAMES: [&str; NUM_HMM_STATES] = ["0: Calm", "1: Shock", "2: Bridge"];
/// Human-readable observation token labels.
pub const OBS_NAMES: [&str; NUM_HMM_OBS] = ["0: CalmTick", "1: VolSpike", "2: HedgeFlow"];

/// Serializable configuration for [`HiddenMarkovParadoxTool`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HmmParadoxConfig {
    /// Baseline observation token sequence (`len` in `2..=12`, each token in `0..3`).
    pub observations: Vec<usize>,
    /// Time index $t_{\text{fork}}$ where the branched timeline mutates the observation token.
    pub fork_step: usize,
    /// Replacement observation token $y'_{t_{\text{fork}}} \in \{0, 1, 2\}$ on the forked timeline.
    pub fork_token: usize,
    /// Self-transition probability $A_{1,1}$ of the `Shock` state (`0.0` = strictly forbidden).
    pub shock_self_loop_prob: f64,
    /// Transition probability $A_{2,1}$ from `Bridge` into `Shock` (`0.10..=0.95`).
    pub bridge_to_shock_prob: f64,
    /// Emission probability $B_{1,1}$ of `VolSpike` in `Shock` (`0.30..=0.90`).
    pub shock_emission_focus: f64,
    /// Deterministic `OxidizeRng` seed counter for synthetic observation rollouts.
    pub seed_counter: u64,
}

impl Default for HmmParadoxConfig {
    fn default() -> Self {
        Self {
            observations: vec![0, 1, 1],
            fork_step: 2,
            fork_token: 0,
            shock_self_loop_prob: 0.0,
            bridge_to_shock_prob: 0.95,
            shock_emission_focus: 0.85,
            seed_counter: 0,
        }
    }
}

/// Decoded metrics for a single observation timeline (Baseline or Forked).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimelineSummary {
    /// Exact unscaled forward likelihood $P(Y_{0:T-1})$.
    pub forward_likelihood: f64,
    /// Exact forward log-likelihood $\ln P(Y_{0:T-1})$.
    pub forward_log_likelihood: f64,
    /// Viterbi joint optimal state sequence $S^*_{0:T-1}$.
    pub viterbi_path: Vec<usize>,
    /// Exact joint probability $P(S^*_{0:T-1}, Y_{0:T-1})$ of the Viterbi path.
    pub viterbi_joint_prob: f64,
    /// Pointwise marginal MAP state sequence $\hat{S}^{\text{MAP}}_t = \arg\max_i \gamma_t(i)$.
    pub marginal_path: Vec<usize>,
    /// Exact joint probability $P(\hat{S}^{\text{MAP}}_{0:T-1}, Y_{0:T-1})$ (`0.0` if forbidden).
    pub marginal_joint_prob: f64,
    /// Filtered final-step belief $P(X_{T-1} = i \mid Y_{0:T-1})$.
    pub final_filter_belief: [f64; NUM_HMM_STATES],
    /// Whether the pointwise marginal path traverses a zero-probability $A_{i,j} = 0$ transition.
    pub has_forbidden_marginal_step: bool,
    /// First transition step $t$ where $A_{\hat{s}_t, \hat{s}_{t+1}} = 0$.
    pub first_forbidden_step: Option<usize>,
}

/// Interactive HMM forbidden-path trellis and time-travel branching workbench.
pub struct HiddenMarkovParadoxTool {
    /// Editable HMM topology and observation branching configuration.
    pub config: HmmParadoxConfig,
    /// Compiled `HiddenMarkovModel<f64>`.
    pub hmm: HiddenMarkovModel<f64>,
    /// Preallocated zero-allocation scratch workspace for the baseline timeline.
    pub baseline_scratch: HmmScratch<f64>,
    /// Preallocated zero-allocation scratch workspace for the forked timeline.
    pub branched_scratch: HmmScratch<f64>,
    /// Preallocated buffer holding the forked observation sequence.
    pub branched_observations: Vec<usize>,
    /// Decoded metrics for the baseline timeline.
    pub baseline: TimelineSummary,
    /// Decoded metrics for the forked timeline.
    pub branched: TimelineSummary,
    /// Callout manager flagging forbidden $A_{i,j} = 0$ marginal steps on the trellis.
    pub callout_manager: CalloutManager,
    /// Active inverse challenge identifier, if any.
    pub active_challenge: Option<&'static str>,
}

impl Default for HiddenMarkovParadoxTool {
    fn default() -> Self {
        let cfg = HmmParadoxConfig::default();
        let hmm = model::build_hmm_from_config(&cfg);
        let mut tool = Self {
            config: cfg,
            hmm,
            baseline_scratch: HmmScratch::new(NUM_HMM_STATES, MAX_HMM_STEPS),
            branched_scratch: HmmScratch::new(NUM_HMM_STATES, MAX_HMM_STEPS),
            branched_observations: Vec::with_capacity(MAX_HMM_STEPS),
            baseline: TimelineSummary::default(),
            branched: TimelineSummary::default(),
            callout_manager: CalloutManager::new(),
            active_challenge: None,
        };
        tool.recalculate();
        tool
    }
}

impl HiddenMarkovParadoxTool {
    fn compute_revision(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.config.observations.hash(&mut h);
        self.config.fork_step.hash(&mut h);
        self.config.fork_token.hash(&mut h);
        self.config.shock_self_loop_prob.to_bits().hash(&mut h);
        self.config.bridge_to_shock_prob.to_bits().hash(&mut h);
        self.config.shock_emission_focus.to_bits().hash(&mut h);
        self.config.seed_counter.hash(&mut h);
        self.active_challenge.hash(&mut h);
        h.finish()
    }
}

impl InteractiveTool for HiddenMarkovParadoxTool {
    fn name(&self) -> &'static str {
        "HMM Trellis & Forbidden-Path Lab"
    }

    fn domain(&self) -> &'static str {
        "financial_math"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let viterbi_pts: Vec<[f64; 2]> = self
            .baseline
            .viterbi_path
            .iter()
            .enumerate()
            .map(|(t, &s)| [t as f64, s as f64])
            .collect();
        let map_pts: Vec<[f64; 2]> = self
            .baseline
            .marginal_path
            .iter()
            .enumerate()
            .map(|(t, &s)| [t as f64, s as f64])
            .collect();
        let branch_pts: Vec<[f64; 2]> = self
            .branched
            .viterbi_path
            .iter()
            .enumerate()
            .map(|(t, &s)| [t as f64, s as f64])
            .collect();
        vec![
            ("Baseline Viterbi Path".into(), viterbi_pts),
            ("Baseline Marginal MAP Path".into(), map_pts),
            ("Branched Viterbi Path".into(), branch_pts),
        ]
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn state_revision(&self) -> u64 {
        self.compute_revision()
    }

    fn has_finite_parameters(&self) -> bool {
        let c = &self.config;
        c.shock_self_loop_prob.is_finite()
            && c.bridge_to_shock_prob.is_finite()
            && c.shock_emission_focus.is_finite()
            && (2..=MAX_HMM_STEPS).contains(&c.observations.len())
            && c.observations.iter().all(|&o| o < NUM_HMM_OBS)
    }

    fn clamp_parameters(&mut self) -> bool {
        let before = self.config.clone();
        self.clamp_config_in_place();
        if self.config != before {
            self.recalculate();
            true
        } else {
            false
        }
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: HmmParadoxConfig =
            serde_json::from_str(state).map_err(|e| format!("Invalid HMM state: {e}"))?;
        self.config = parsed;
        self.clamp_config_in_place();
        self.recalculate();
        Ok(())
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        self.set_challenge_mode(challenge_id)
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.eval_active_challenge()
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        ui::show_hmm_paradox_tool(self, ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

impl scientific_metadata::theory::TheoryDescribable for HiddenMarkovParadoxTool {
    fn theory_description(&self) -> String {
        format!(
            "Hidden Markov Model smoothing $\\gamma_t(i) = P(X_t = i \\mid Y_{{0:T-1}})$ vs Viterbi joint decoding $S^* = \\arg\\max_S P(S, Y)$: on observation likelihood $P(Y) = {:.7}$ ($\\ln L = {:.4}$), pointwise marginal decoding selects {:?} ($P = {:.7}$) across forbidden $A_{{1,1}} = 0$, whereas Viterbi recovers {:?} ($P = {:.7}$).",
            self.baseline.forward_likelihood,
            self.baseline.forward_log_likelihood,
            self.baseline.marginal_path,
            self.baseline.marginal_joint_prob,
            self.baseline.viterbi_path,
            self.baseline.viterbi_joint_prob
        )
    }

    fn phonetic_description(&self) -> String {
        "Hidden Markov Model forward-backward posterior smoothing versus Viterbi joint sequence decoding".into()
    }

    fn theory_citation(&self) -> String {
        "Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286; Viterbi, A. (1967).".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Pointwise Marginal Trap".into(),
            "Maximizing γ_t(i) independently at each step can concatenate states connected by A[i,j] = 0".into(),
        );
        map.insert(
            "Viterbi Trellis Recovery".into(),
            "Dynamic programming maximizes full joint path probability P(X_{0:T-1}, Y_{0:T-1})"
                .into(),
        );
        map
    }
}

/// Starter preset JSON payload reproducing `[0, 1, 1]` (`P = 0.0`), `[0, 2, 1]` (`P = 0.0282625`), and `forward = 0.0693765`.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Forbidden Marginal Path vs Viterbi Recovery",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "observations": [0, 1, 1],
    "fork_step": 2,
    "fork_token": 0,
    "shock_self_loop_prob": 0.0,
    "bridge_to_shock_prob": 0.95,
    "shock_emission_focus": 0.85,
    "seed_counter": 0
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "HiddenMarkovParadoxTool",
        display_name: "HMM Trellis & Forbidden-Path Lab",
        domain: "financial_math",
        tags: &["hmm", "viterbi", "forward-backward", "markov", "trellis", "paradox"],
        build: || Box::new(HiddenMarkovParadoxTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "hmm_forbidden_marginal_path",
        title: "Forbidden Marginal Path vs Viterbi",
        description: "Pointwise posterior smoothing selects [0, 1, 1] (P = 0.0 across A[1,1] = 0), while Viterbi recovers [0, 2, 1] (P = 0.0282625) out of forward likelihood 0.0693765.",
        category: "Stochastic Processes",
        icon: "🔀",
        domain: "financial_math",
        tool_name: Some("HiddenMarkovParadoxTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]
