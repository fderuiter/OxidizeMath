#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! State serialization, presets, theory metadata, and plot exports for [`super::HmmOuWorkbenchTool`].

use super::{
    HmmOuWorkbenchTool, MAX_EM_ITERS, MAX_OU_STEPS, MAX_SEQ_LEN, MIN_EM_ITERS, MIN_OU_STEPS,
    MIN_SEQ_LEN,
};
use crate::framework::{StarterPresetMetadata, ToolMetadata};
use egui_plot::ExportPrimitive;
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Active inference & calibration mode in [`HmmOuWorkbenchTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CalibrationMode {
    /// Log-space Viterbi decoding + floored Baum-Welch EM up to 600 observations.
    #[default]
    HmmViterbiBaumWelch,
    /// Exact conditional Gaussian OU transitions vs Euler-Maruyama + AR(1) OU calibration.
    OuExactAndCalibration,
}

impl CalibrationMode {
    pub(super) const ALL: [Self; 2] = [Self::HmmViterbiBaumWelch, Self::OuExactAndCalibration];

    /// Short UI label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HmmViterbiBaumWelch => "HMM Viterbi & Baum-Welch",
            Self::OuExactAndCalibration => "Exact OU & Calibration",
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct HmmOuState {
    pub(super) mode: CalibrationMode,
    pub(super) seq_len: usize,
    pub(super) stay_prob: f64,
    pub(super) emission_clarity: f64,
    pub(super) em_iters: usize,
    pub(super) prob_floor: f64,
    pub(super) ou_mu: f64,
    pub(super) ou_theta: f64,
    pub(super) ou_sigma: f64,
    pub(super) ou_x0: f64,
    pub(super) ou_dt: f64,
    pub(super) ou_steps: usize,
    pub(super) seed: u64,
}

impl HmmOuState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let floats_ok = [
            self.stay_prob,
            self.emission_clarity,
            self.prob_floor,
            self.ou_mu,
            self.ou_theta,
            self.ou_sigma,
            self.ou_x0,
            self.ou_dt,
        ]
        .into_iter()
        .all(f64::is_finite);
        if !floats_ok {
            return Err("Non-finite HMM/OU parameter".to_string());
        }
        if !(MIN_SEQ_LEN..=MAX_SEQ_LEN).contains(&self.seq_len)
            || !(0.55..=0.97).contains(&self.stay_prob)
            || !(0.55..=0.95).contains(&self.emission_clarity)
            || !(MIN_EM_ITERS..=MAX_EM_ITERS).contains(&self.em_iters)
            || !(0.001..=0.10).contains(&self.prob_floor)
            || !(-2.0..=2.0).contains(&self.ou_mu)
            || !(0.10..=5.0).contains(&self.ou_theta)
            || !(0.05..=1.50).contains(&self.ou_sigma)
            || !(-2.0..=2.0).contains(&self.ou_x0)
            || !(0.01..=0.25).contains(&self.ou_dt)
            || !(MIN_OU_STEPS..=MAX_OU_STEPS).contains(&self.ou_steps)
        {
            return Err("HMM/OU parameter out of bounds".to_string());
        }
        Ok(())
    }
}

impl TheoryDescribable for HmmOuWorkbenchTool {
    fn theory_description(&self) -> String {
        match self.mode {
            CalibrationMode::HmmViterbiBaumWelch => format!(
                "Hidden Markov Model over $T = {}$ observations: accumulates scaling-factor log-likelihood $\\sum_t \\ln c_t$, decodes MAP paths via log-space Viterbi without underflow, and learns $(\\pi, A, B)$ via floored Baum-Welch EM.",
                self.seq_len
            ),
            CalibrationMode::OuExactAndCalibration => format!(
                "Ornstein-Uhlenbeck process ($\\mu = {:.2}, \\theta = {:.2}, \\sigma = {:.2}$): compares exact Gaussian transitions $X_{{t+\\Delta t}} \\mid X_t$ against Euler-Maruyama and recovers $(\\hat{{\\mu}}, \\hat{{\\theta}}, \\hat{{\\sigma}})$ via exact discrete AR(1) innovation variance scaling.",
                self.ou_mu, self.ou_theta, self.ou_sigma
            ),
        }
    }

    fn phonetic_description(&self) -> String {
        "Hidden Markov Model log space Viterbi decoding Baum Welch expectation maximization and Ornstein Uhlenbeck exact Gaussian transition calibration".to_string()
    }

    fn theory_citation(&self) -> String {
        "Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE; Uhlenbeck, G. E., & Ornstein, L. S. (1930). On the theory of the Brownian motion. Physical Review.".to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("HMM & OU Calibration".to_string(), self.theory_description());
        map
    }
}

inventory::submit! {
    ToolMetadata {
        id: "HmmOuWorkbenchTool",
        display_name: "HMM & OU Calibration",
        domain: "stochastic_processes",
        tags: &["hmm", "viterbi", "baum-welch", "ornstein-uhlenbeck", "sde", "calibration"],
        build: || Box::new(HmmOuWorkbenchTool::default()),
    }
}

const PRESET_HMM_600_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "600-Step Log-Space Viterbi & Baum-Welch",
    "domain": "stochastic_processes",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "HmmViterbiBaumWelch",
    "seq_len": 600,
    "stay_prob": 0.9,
    "emission_clarity": 0.85,
    "em_iters": 20,
    "prob_floor": 0.01,
    "ou_mu": 0.5,
    "ou_theta": 2.5,
    "ou_sigma": 0.35,
    "ou_x0": 1.2,
    "ou_dt": 0.05,
    "ou_steps": 300,
    "seed": 1337
  }
}"#;

const PRESET_OU_EXACT_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Exact OU Transitions & Volatility Recovery",
    "domain": "stochastic_processes",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "mode": "OuExactAndCalibration",
    "seq_len": 240,
    "stay_prob": 0.88,
    "emission_clarity": 0.82,
    "em_iters": 15,
    "prob_floor": 0.01,
    "ou_mu": 0.45,
    "ou_theta": 4.0,
    "ou_sigma": 0.3,
    "ou_x0": 1.15,
    "ou_dt": 0.05,
    "ou_steps": 400,
    "seed": 2026
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "hmm_600_step_viterbi_em_starter",
        title: "600-Step Log-Space Viterbi & Baum-Welch",
        description: "Decodes a 600-observation regime chain in log-space without underflow and trains parameters via floored Baum-Welch EM.",
        category: "Inference & Calibration",
        icon: "🧬",
        domain: "stochastic_processes",
        tool_name: Some("HmmOuWorkbenchTool"),
        preset_generator: || PRESET_HMM_600_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "ou_exact_calibration_starter",
        title: "Exact OU Transitions & Volatility Recovery",
        description: "Compares ExactOuSolver vs Euler-Maruyama at fast mean reversion (theta=4.0) and calibrates sigma via exact AR(1) variance.",
        category: "Inference & Calibration",
        icon: "📈",
        domain: "stochastic_processes",
        tool_name: Some("HmmOuWorkbenchTool"),
        preset_generator: || PRESET_OU_EXACT_JSON,
    }
}

impl HmmOuWorkbenchTool {
    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.mode {
            CalibrationMode::HmmViterbiBaumWelch => vec![
                (
                    "Baum-Welch Log-Likelihood".to_string(),
                    self.em_log_liks.iter().map(|p| [p.x, p.y]).collect(),
                ),
                (
                    "Viterbi Decoded State".to_string(),
                    self.viterbi_curve.iter().map(|p| [p.x, p.y]).collect(),
                ),
            ],
            CalibrationMode::OuExactAndCalibration => vec![
                (
                    "ExactOuSolver Path".to_string(),
                    self.ou_exact_curve.iter().map(|p| [p.x, p.y]).collect(),
                ),
                (
                    "Euler-Maruyama Path".to_string(),
                    self.ou_euler_curve.iter().map(|p| [p.x, p.y]).collect(),
                ),
                (
                    "Conditional Mean E[X_t]".to_string(),
                    self.ou_mean_curve.iter().map(|p| [p.x, p.y]).collect(),
                ),
            ],
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        match self.mode {
            CalibrationMode::HmmViterbiBaumWelch => vec![
                ExportPrimitive::Polyline {
                    name: "Baum-Welch Log-Likelihood".to_string(),
                    points: self.em_log_liks.iter().map(|p| [p.x, p.y]).collect(),
                    color: pal.series[0],
                    stroke_width: 2.2_f32,
                    closed: false,
                    fill: None,
                },
                ExportPrimitive::Polyline {
                    name: "Viterbi Path".to_string(),
                    points: self.viterbi_curve.iter().map(|p| [p.x, p.y]).collect(),
                    color: pal.series[1],
                    stroke_width: 1.8_f32,
                    closed: false,
                    fill: None,
                },
            ],
            CalibrationMode::OuExactAndCalibration => vec![
                ExportPrimitive::Polyline {
                    name: "ExactOuSolver".to_string(),
                    points: self.ou_exact_curve.iter().map(|p| [p.x, p.y]).collect(),
                    color: pal.series[0],
                    stroke_width: 2.0_f32,
                    closed: false,
                    fill: None,
                },
                ExportPrimitive::Polyline {
                    name: "Euler-Maruyama".to_string(),
                    points: self.ou_euler_curve.iter().map(|p| [p.x, p.y]).collect(),
                    color: pal.series[1],
                    stroke_width: 1.6_f32,
                    closed: false,
                    fill: None,
                },
                ExportPrimitive::Polyline {
                    name: "Conditional Mean".to_string(),
                    points: self.ou_mean_curve.iter().map(|p| [p.x, p.y]).collect(),
                    color: pal.series[2],
                    stroke_width: 2.0_f32,
                    closed: false,
                    fill: None,
                },
            ],
        }
    }
}

// [cite:clinical_trials_statistics]
