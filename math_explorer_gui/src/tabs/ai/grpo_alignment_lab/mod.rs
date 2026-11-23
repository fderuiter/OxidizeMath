#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod presets;
mod ui;

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use math_explorer::ai::self_calibration::{
    calculate_answer_entropy, calculate_kl_divergence_loss, calculate_soft_self_consistency_scores,
    calculate_target_distribution, map_entropy_to_temperature, Response,
};
use math_explorer::applied::grpo::formulas::{
    clipped_surrogate_objective, kl_divergence_lower_bound, normalize_group_advantages,
    solver_empirical_accuracy,
};
use math_explorer::applied::grpo::rewards::{
    composite_reward, repetition_penalty, uncertainty_reward,
};
use scientific_metadata::theory::TheoryDescribable;

pub(crate) const EPSILON_MIN: f64 = 0.05;
pub(crate) const EPSILON_MAX: f64 = 0.50;
pub(crate) const BETA_MIN: f64 = 0.0;
pub(crate) const BETA_MAX: f64 = 0.50;
pub(crate) const ACCURACY_MIN: f64 = 0.0;
pub(crate) const ACCURACY_MAX: f64 = 1.0;
pub(crate) const LAMBDA_MIN: f64 = 0.0;
pub(crate) const LAMBDA_MAX: f64 = 2.0;
pub(crate) const SKEW_MIN: f64 = 0.10;
pub(crate) const SKEW_MAX: f64 = 0.90;
pub(crate) const GROUP_SIZE: usize = 6;

/// Preset candidate response scenarios for GRPO + Self-Calibration exploration.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum GrpoScenarioPreset {
    /// Majority consensus on a reasoning answer with moderate minority disagreement.
    #[default]
    ReasoningConsensus,
    /// High answer-space entropy across competing reasoning paths.
    HighEntropyExploration,
    /// Repetitive near-duplicate phrasing triggering cluster repetition penalties.
    RepetitionCollapse,
}

impl GrpoScenarioPreset {
    /// All selectable scenarios in UI display order.
    pub const ALL: [Self; 3] = [
        Self::ReasoningConsensus,
        Self::HighEntropyExploration,
        Self::RepetitionCollapse,
    ];

    /// Human-readable display label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::ReasoningConsensus => "Reasoning Consensus",
            Self::HighEntropyExploration => "High-Entropy Exploration",
            Self::RepetitionCollapse => "Repetition Collapse",
        }
    }
}

/// Snapshot of computed GRPO alignment, Challenger, and self-calibration diagnostics.
#[derive(Debug, Clone)]
pub struct GrpoCalibrationSnapshot {
    /// Group-standardized rollout advantages $\hat{A}_i$.
    pub advantages: [f64; GROUP_SIZE],
    /// Scalar GRPO clipped loss objective $L_{\mathrm{GRPO}}(\theta)$.
    pub grpo_loss: f64,
    /// Challenger uncertainty reward $1 - 2|\hat{p} - 0.5|$ at active `solver_accuracy`.
    pub active_uncertainty_reward: f64,
    /// Challenger repetition penalty at active `cluster_size` and `repetition_lambda`.
    pub active_repetition_penalty: f64,
    /// Challenger composite reward $\max(0, R_{\text{unc}} - r_{\text{rep}})$.
    pub active_composite_reward: f64,
    /// Lower bound on KL divergence at active `solver_accuracy` and positive `beta`.
    pub kl_lower_bound: f64,
    /// Empirical Solver consensus accuracy across the candidate response set.
    pub empirical_solver_acc: f64,
    /// Soft self-consistency scores $S(y_i)$ for candidate responses.
    pub consistency_scores: Vec<f64>,
    /// Answer-group Shannon entropy $H(A)$.
    pub answer_entropy: f64,
    /// Dynamic Boltzmann calibration temperature $T(H)$.
    pub dynamic_temperature: f64,
    /// Entropy-scaled target probability distribution $P(y_i)$.
    pub target_distribution: Vec<f64>,
    /// Normalized raw model confidence distribution $Q(y_i)$.
    pub predicted_distribution: Vec<f64>,
    /// Distillation KL divergence $D_{\mathrm{KL}}(Q \parallel P)$.
    pub calibration_kl_loss: f64,
}

/// Interactive laboratory for GRPO surrogate clipping, Challenger rewards, and KL self-calibration.
pub struct GrpoAlignmentLabTool {
    /// Symmetric PPO/GRPO policy ratio clipping parameter $\epsilon$.
    pub epsilon: f64,
    /// Reference-policy KL divergence weight $\beta$.
    pub beta: f64,
    /// Challenger-probed Solver accuracy $\hat{p} \in [0, 1]$.
    pub solver_accuracy: f64,
    /// Repetition penalty strength $\lambda$.
    pub repetition_lambda: f64,
    /// Duplicate cluster cardinality $k \in \{1 \dots G\}$.
    pub cluster_size: usize,
    /// Raw policy confidence concentration skew for KL self-calibration comparison.
    pub confidence_skew: f64,
    /// Active candidate rollout preset.
    pub scenario: GrpoScenarioPreset,
    rollout_rewards: [f64; GROUP_SIZE],
    rollout_advantages: [f64; GROUP_SIZE],
}

impl Default for GrpoAlignmentLabTool {
    fn default() -> Self {
        Self {
            epsilon: 0.20,
            beta: 0.04,
            solver_accuracy: 0.50,
            repetition_lambda: 0.50,
            cluster_size: 2,
            confidence_skew: 0.55,
            scenario: GrpoScenarioPreset::ReasoningConsensus,
            rollout_rewards: [0.0; GROUP_SIZE],
            rollout_advantages: [0.0; GROUP_SIZE],
        }
    }
}

impl GrpoAlignmentLabTool {
    /// Builds candidate [`Response`] items for the active scenario and `confidence_skew`.
    pub fn build_candidate_responses(&self) -> (Vec<Response>, &'static str) {
        let skew = self.confidence_skew.clamp(SKEW_MIN, SKEW_MAX);
        let raw_weights: [(&str, f64, &str); 4] = match self.scenario {
            GrpoScenarioPreset::ReasoningConsensus => [
                ("Chain 1: factor -> 42", skew, "42"),
                ("Chain 2: modular -> 42", 0.55 * (1.0 - skew), "42"),
                ("Chain 3: algebra -> 42", 0.25 * (1.0 - skew), "42"),
                ("Chain 4: sign slip -> 18", 0.20 * (1.0 - skew), "18"),
            ],
            GrpoScenarioPreset::HighEntropyExploration => [
                ("Path A -> 42", 0.45 * skew + 0.10, "42"),
                ("Path B -> 17", 0.35 * (1.0 - skew) + 0.15, "17"),
                ("Path C -> 29", 0.35 * skew + 0.10, "29"),
                ("Path D -> 64", 0.25 * (1.0 - skew) + 0.10, "64"),
            ],
            GrpoScenarioPreset::RepetitionCollapse => [
                ("Repeat 1 -> 42", 0.60 * skew + 0.15, "42"),
                ("Repeat 2 -> 42", 0.20 * (1.0 - skew) + 0.05, "42"),
                ("Alt -> 11", 0.45 * (1.0 - skew) + 0.08, "11"),
                ("Alt -> 9", 0.25 * (1.0 - skew) + 0.05, "9"),
            ],
        };
        let sum_w: f64 = raw_weights.iter().map(|(_, w, _)| *w).sum();
        let inv = if sum_w > 0.0 { 1.0 / sum_w } else { 0.25 };
        let responses = raw_weights
            .iter()
            .map(|(text, w, ans)| Response {
                text: (*text).to_string(),
                probability: (*w * inv).max(1e-6),
                answer: (*ans).to_string(),
            })
            .collect();
        (responses, "42")
    }

    fn compute_rollout_surrogate(&mut self, p_base: f64) -> (f64, f64) {
        let offsets = [-0.28, -0.14, -0.04, 0.04, 0.16, 0.30];
        let clusters = match self.scenario {
            GrpoScenarioPreset::RepetitionCollapse => [
                self.cluster_size,
                self.cluster_size,
                2,
                1,
                2,
                self.cluster_size,
            ],
            _ => [1, 1, self.cluster_size, 1, 2, 1],
        };
        for i in 0..GROUP_SIZE {
            let p_i = (p_base + offsets[i]).clamp(0.0, 1.0);
            let r_unc = uncertainty_reward(p_i);
            let r_rep = repetition_penalty(
                clusters[i].clamp(1, GROUP_SIZE),
                GROUP_SIZE,
                self.repetition_lambda,
            );
            self.rollout_rewards[i] = composite_reward(r_unc, r_rep);
        }
        normalize_group_advantages(&self.rollout_rewards, &mut self.rollout_advantages);

        let pi_olds = [1.0 / (GROUP_SIZE as f64); GROUP_SIZE];
        let mut pi_thetas = [0.0; GROUP_SIZE];
        for i in 0..GROUP_SIZE {
            let ratio = (1.0 + 0.22 * self.rollout_advantages[i]).clamp(0.25, 1.95);
            pi_thetas[i] = pi_olds[i] * ratio;
        }
        let kl_lb =
            kl_divergence_lower_bound(p_base.clamp(0.01, 0.99), self.beta.max(1e-3)).min(12.0);
        let grpo_loss = clipped_surrogate_objective(
            &pi_thetas,
            &pi_olds,
            &self.rollout_advantages,
            self.epsilon,
            self.beta,
            kl_lb,
        );
        (grpo_loss, kl_lb)
    }

    /// Evaluates the zero-allocation GRPO rollout and self-calibration distillation snapshot.
    pub fn evaluate_snapshot(&mut self) -> GrpoCalibrationSnapshot {
        let p_base = self.solver_accuracy.clamp(ACCURACY_MIN, ACCURACY_MAX);
        let (grpo_loss, kl_lb) = self.compute_rollout_surrogate(p_base);
        let active_unc = uncertainty_reward(p_base);
        let active_rep = repetition_penalty(
            self.cluster_size.clamp(1, GROUP_SIZE),
            GROUP_SIZE,
            self.repetition_lambda,
        );
        let active_comp = composite_reward(active_unc, active_rep);

        let (responses, consensus) = self.build_candidate_responses();
        let ans_strings: Vec<String> = responses.iter().map(|r| r.answer.clone()).collect();
        let empirical_solver_acc = solver_empirical_accuracy(&ans_strings, consensus);
        let consistency_scores = calculate_soft_self_consistency_scores(&responses);
        let answer_entropy = calculate_answer_entropy(&responses);
        let dynamic_temperature = map_entropy_to_temperature(answer_entropy);
        let target_distribution = calculate_target_distribution(&responses);
        let predicted_distribution: Vec<f64> = responses.iter().map(|r| r.probability).collect();
        let calibration_kl_loss =
            calculate_kl_divergence_loss(&responses, &predicted_distribution).unwrap_or(0.0);

        GrpoCalibrationSnapshot {
            advantages: self.rollout_advantages,
            grpo_loss,
            active_uncertainty_reward: active_unc,
            active_repetition_penalty: active_rep,
            active_composite_reward: active_comp,
            kl_lower_bound: kl_lb,
            empirical_solver_acc,
            consistency_scores,
            answer_entropy,
            dynamic_temperature,
            target_distribution,
            predicted_distribution,
            calibration_kl_loss,
        }
    }

    /// Generates surrogate clipping and Challenger reward curves for plotting and dataset export.
    pub fn compute_surrogate_and_reward_series(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let eps = self.epsilon.clamp(EPSILON_MIN, EPSILON_MAX);
        let adv = 1.0_f64;
        let mut unclipped = Vec::with_capacity(65);
        let mut clipped = Vec::with_capacity(65);
        for idx in 0..=64 {
            let r_t = 0.2 + (idx as f64 / 64.0) * 1.6;
            let u_val = r_t * adv;
            let c_val = u_val.min(r_t.clamp(1.0 - eps, 1.0 + eps) * adv);
            unclipped.push([r_t, u_val]);
            clipped.push([r_t, c_val]);
        }

        let rep = repetition_penalty(
            self.cluster_size.clamp(1, GROUP_SIZE),
            GROUP_SIZE,
            self.repetition_lambda.clamp(LAMBDA_MIN, LAMBDA_MAX),
        );
        let mut challenger_curve = Vec::with_capacity(51);
        for idx in 0..=50 {
            let p_hat = idx as f64 / 50.0;
            let r_u = uncertainty_reward(p_hat);
            challenger_curve.push([p_hat, composite_reward(r_u, rep)]);
        }

        vec![
            ("Clipped Surrogate".to_string(), clipped),
            ("Unclipped Surrogate".to_string(), unclipped),
            ("Challenger Composite Reward".to_string(), challenger_curve),
        ]
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GrpoAlignmentState {
    epsilon: f64,
    beta: f64,
    solver_accuracy: f64,
    repetition_lambda: f64,
    cluster_size: usize,
    confidence_skew: f64,
    #[serde(default)]
    scenario: GrpoScenarioPreset,
}

impl InteractiveTool for GrpoAlignmentLabTool {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "GRPO Alignment Lab"
    }

    fn domain(&self) -> &'static str {
        "ai"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.compute_surrogate_and_reward_series()
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.epsilon.to_bits().hash(&mut h);
        self.beta.to_bits().hash(&mut h);
        self.solver_accuracy.to_bits().hash(&mut h);
        self.repetition_lambda.to_bits().hash(&mut h);
        self.cluster_size.hash(&mut h);
        self.confidence_skew.to_bits().hash(&mut h);
        self.scenario.hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.epsilon.is_finite()
            && (EPSILON_MIN..=EPSILON_MAX).contains(&self.epsilon)
            && self.beta.is_finite()
            && (BETA_MIN..=BETA_MAX).contains(&self.beta)
            && self.solver_accuracy.is_finite()
            && (ACCURACY_MIN..=ACCURACY_MAX).contains(&self.solver_accuracy)
            && self.repetition_lambda.is_finite()
            && (LAMBDA_MIN..=LAMBDA_MAX).contains(&self.repetition_lambda)
            && (1..=GROUP_SIZE).contains(&self.cluster_size)
            && self.confidence_skew.is_finite()
            && (SKEW_MIN..=SKEW_MAX).contains(&self.confidence_skew)
    }

    fn clamp_parameters(&mut self) -> bool {
        self.epsilon = if self.epsilon.is_finite() {
            self.epsilon.clamp(EPSILON_MIN, EPSILON_MAX)
        } else {
            0.20
        };
        self.beta = if self.beta.is_finite() {
            self.beta.clamp(BETA_MIN, BETA_MAX)
        } else {
            0.04
        };
        self.solver_accuracy = if self.solver_accuracy.is_finite() {
            self.solver_accuracy.clamp(ACCURACY_MIN, ACCURACY_MAX)
        } else {
            0.50
        };
        self.repetition_lambda = if self.repetition_lambda.is_finite() {
            self.repetition_lambda.clamp(LAMBDA_MIN, LAMBDA_MAX)
        } else {
            0.50
        };
        self.cluster_size = self.cluster_size.clamp(1, GROUP_SIZE);
        self.confidence_skew = if self.confidence_skew.is_finite() {
            self.confidence_skew.clamp(SKEW_MIN, SKEW_MAX)
        } else {
            0.55
        };
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&GrpoAlignmentState {
            epsilon: self.epsilon,
            beta: self.beta,
            solver_accuracy: self.solver_accuracy,
            repetition_lambda: self.repetition_lambda,
            cluster_size: self.cluster_size,
            confidence_skew: self.confidence_skew,
            scenario: self.scenario,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: GrpoAlignmentState =
            serde_json::from_str(state).map_err(|e| format!("Invalid GRPO state: {e}"))?;
        if !parsed.epsilon.is_finite() || !(EPSILON_MIN..=EPSILON_MAX).contains(&parsed.epsilon) {
            return Err("epsilon out of bounds".into());
        }
        if !parsed.beta.is_finite() || !(BETA_MIN..=BETA_MAX).contains(&parsed.beta) {
            return Err("beta out of bounds".into());
        }
        if !parsed.solver_accuracy.is_finite()
            || !(ACCURACY_MIN..=ACCURACY_MAX).contains(&parsed.solver_accuracy)
        {
            return Err("solver_accuracy out of bounds".into());
        }
        if !parsed.repetition_lambda.is_finite()
            || !(LAMBDA_MIN..=LAMBDA_MAX).contains(&parsed.repetition_lambda)
        {
            return Err("repetition_lambda out of bounds".into());
        }
        if !(1..=GROUP_SIZE).contains(&parsed.cluster_size) {
            return Err("cluster_size out of bounds".into());
        }
        if !parsed.confidence_skew.is_finite()
            || !(SKEW_MIN..=SKEW_MAX).contains(&parsed.confidence_skew)
        {
            return Err("confidence_skew out of bounds".into());
        }
        self.epsilon = parsed.epsilon;
        self.beta = parsed.beta;
        self.solver_accuracy = parsed.solver_accuracy;
        self.repetition_lambda = parsed.repetition_lambda;
        self.cluster_size = parsed.cluster_size;
        self.confidence_skew = parsed.confidence_skew;
        self.scenario = parsed.scenario;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.draw_side_panel(ui);
        self.draw_central_plots(ui);
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

// [cite:grpo]
// [cite:self_calibration_paper]

#[cfg(test)]
mod tests;
