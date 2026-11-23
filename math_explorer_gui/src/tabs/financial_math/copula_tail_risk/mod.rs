//! Standalone `CopulaTailRiskLab` workbench in `FinancialMathTab` contrasting naive independent
//! parlay multiplication, `GaussianCopula` SGP pricing, heavy-tailed `StudentTCopula` crash
//! amplification, and correlated Kelly bankroll sizing.

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
use math_explorer::pure_math::statistics::copula::BivariateCopulaScratch;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Grid dimension along each axis of the 2D `(rho, u)` tail-multiplier heatmap.
pub const COPULA_GRID_DIM: usize = 24;
/// Number of correlation sweep points for SGP parlay curves.
pub const SGP_CURVE_POINTS: usize = 49;

/// Serializable configuration for [`CopulaTailRiskLab`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CopulaTailRiskConfig {
    /// Marginal probability of Parlay Leg 1 (`0.02..=0.95`).
    pub leg1_prob: f64,
    /// Marginal probability of Parlay Leg 2 (`0.02..=0.95`).
    pub leg2_prob: f64,
    /// Latent correlation $\rho \in [-0.95, 0.95]$.
    pub correlation_rho: f64,
    /// Extreme upper-tail crash quantile $u \in [0.85, 0.9995]$.
    pub tail_quantile_u: f64,
    /// Student-t degrees of freedom $\nu \in [1.5, 30.0]$.
    pub degrees_of_freedom: f64,
    /// Sportsbook offered decimal odds $O > 1$ for the two-leg parlay (`1.2..=200.0`).
    pub offered_decimal_odds: f64,
    /// Multiplier applied to optimal correlated Kelly fraction (`0.10..=2.50`).
    pub stake_multiplier: f64,
}

impl Default for CopulaTailRiskConfig {
    fn default() -> Self {
        Self {
            leg1_prob: 0.20,
            leg2_prob: 0.20,
            correlation_rho: 0.50,
            tail_quantile_u: 0.999,
            degrees_of_freedom: 3.0,
            offered_decimal_odds: 18.0,
            stake_multiplier: 0.50,
        }
    }
}

/// Computed SGP mispricing, extreme-tail amplification, and Kelly growth observables.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CopulaTailRiskMetrics {
    /// Naive independent joint probability $p_1 \times p_2$.
    pub naive_independent_prob: f64,
    /// Gaussian copula SGP joint probability via `sgp_joint_probability`.
    pub sgp_gaussian_prob: f64,
    /// Heavy-tailed Student-t copula joint probability via `StudentTCopula::cdf`.
    pub student_t_joint_prob: f64,
    /// Reference Gaussian SGP probability at $\rho = -0.50$ (`0.84%` on `0.20 x 0.20`).
    pub sgp_neg50_prob: f64,
    /// Reference Gaussian SGP probability at $\rho = +0.70$ (`11.29%` on `0.20 x 0.20`).
    pub sgp_pos70_prob: f64,
    /// Conditional Gaussian upper-tail crash probability $P(U_2 > u \mid U_1 > u)$.
    pub gaussian_cond_tail: f64,
    /// Conditional Student-t upper-tail crash probability $P(U_2 > u \mid U_1 > u)$.
    pub student_t_cond_tail: f64,
    /// Student-t vs Gaussian upper-tail conditional crash ratio (`5.83x` at `u = 0.999, rho = 0.50, nu = 3`).
    pub tail_multiplier: f64,
    /// Asymptotic Student-t tail dependence coefficient $\lambda_U(\rho, \nu)$.
    pub asymptotic_lambda_u: f64,
    /// Naive independent Kelly fraction (`0` if negative EV).
    pub naive_kelly_fraction: f64,
    /// True correlated Kelly fraction under Student-t joint probability.
    pub copula_kelly_fraction: f64,
    /// Applied bankroll fraction (`stake_multiplier * copula_kelly_fraction` or naive overbet).
    pub applied_stake: f64,
    /// True expected logarithmic bankroll growth rate $g(f)$ under `StudentTCopula` probability.
    pub true_expected_growth: f64,
    /// True expected growth rate if sizing naively by independent probability.
    pub naive_stake_true_growth: f64,
    /// True when overbetting or mispricing drives expected log-growth negative on a staked bet.
    pub is_overbet_ruin: bool,
}

/// Interactive Same-Game Parlay (SGP), extreme tail-dependence, and Kelly mispricing workbench.
pub struct CopulaTailRiskLab {
    /// Editable copula and wager configuration.
    pub config: CopulaTailRiskConfig,
    /// Latest computed metrics.
    pub metrics: CopulaTailRiskMetrics,
    /// Preallocated zero-allocation scratch buffers for 2D `(rho, u)` regime heatmaps.
    pub scratch: BivariateCopulaScratch,
    /// Sampled $\rho$ grid coordinates for the heatmap.
    pub rho_axis: [f64; COPULA_GRID_DIM],
    /// Sampled quantile $u$ grid coordinates for the heatmap.
    pub quantile_axis: [f64; COPULA_GRID_DIM],
    /// Naive independent SGP curve across $\rho$.
    pub curve_naive: Vec<[f64; 2]>,
    /// Gaussian copula SGP curve across $\rho$.
    pub curve_gaussian: Vec<[f64; 2]>,
    /// Student-t copula SGP curve across $\rho$.
    pub curve_student_t: Vec<[f64; 2]>,
    /// Cached RGBA pixels for the 2D `(rho, u)` tail-multiplier heatmap.
    pub regime_pixels: Vec<egui::Color32>,
    /// Uploaded GPU texture handle for the heatmap.
    pub regime_texture: Option<egui::TextureHandle>,
    /// Dirty flag for texture re-upload.
    pub texture_dirty: bool,
    /// Callout overlay manager.
    pub callout_manager: CalloutManager,
    /// Active inverse challenge ID, if any.
    pub active_challenge: Option<&'static str>,
}

impl Default for CopulaTailRiskLab {
    fn default() -> Self {
        let rho_axis =
            std::array::from_fn(|i| -0.85 + 1.75 * (i as f64) / ((COPULA_GRID_DIM - 1) as f64));
        let quantile_axis =
            std::array::from_fn(|j| 0.85 + 0.149 * (j as f64) / ((COPULA_GRID_DIM - 1) as f64));
        let mut lab = Self {
            config: CopulaTailRiskConfig::default(),
            metrics: CopulaTailRiskMetrics::default(),
            scratch: BivariateCopulaScratch::new(COPULA_GRID_DIM, COPULA_GRID_DIM),
            rho_axis,
            quantile_axis,
            curve_naive: Vec::with_capacity(SGP_CURVE_POINTS),
            curve_gaussian: Vec::with_capacity(SGP_CURVE_POINTS),
            curve_student_t: Vec::with_capacity(SGP_CURVE_POINTS),
            regime_pixels: vec![egui::Color32::BLACK; COPULA_GRID_DIM * COPULA_GRID_DIM],
            regime_texture: None,
            texture_dirty: true,
            callout_manager: CalloutManager::new(),
            active_challenge: None,
        };
        lab.recalculate();
        lab
    }
}

impl CopulaTailRiskLab {
    fn compute_revision(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.config.leg1_prob.to_bits().hash(&mut h);
        self.config.leg2_prob.to_bits().hash(&mut h);
        self.config.correlation_rho.to_bits().hash(&mut h);
        self.config.tail_quantile_u.to_bits().hash(&mut h);
        self.config.degrees_of_freedom.to_bits().hash(&mut h);
        self.config.offered_decimal_odds.to_bits().hash(&mut h);
        self.config.stake_multiplier.to_bits().hash(&mut h);
        self.active_challenge.hash(&mut h);
        h.finish()
    }
}

impl InteractiveTool for CopulaTailRiskLab {
    fn name(&self) -> &'static str {
        "Copula Tail-Risk & SGP Lab"
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
        vec![
            (
                "Naive Independent Joint Prob".into(),
                self.curve_naive.clone(),
            ),
            (
                "Gaussian Copula SGP Prob".into(),
                self.curve_gaussian.clone(),
            ),
            (
                "Student-t Copula SGP Prob".into(),
                self.curve_student_t.clone(),
            ),
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
        c.leg1_prob.is_finite()
            && c.leg2_prob.is_finite()
            && c.correlation_rho.is_finite()
            && c.tail_quantile_u.is_finite()
            && c.degrees_of_freedom.is_finite()
            && c.offered_decimal_odds.is_finite()
            && c.stake_multiplier.is_finite()
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
        let parsed: CopulaTailRiskConfig =
            serde_json::from_str(state).map_err(|e| format!("Invalid Copula state: {e}"))?;
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
        ui::show_copula_tail_risk_lab(self, ui);
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

impl scientific_metadata::theory::TheoryDescribable for CopulaTailRiskLab {
    fn theory_description(&self) -> String {
        format!(
            "Gaussian copula $C_\\rho(u_1, u_2) = \\Phi_\\rho(\\Phi^{{-1}}(u_1), \\Phi^{{-1}}(u_2)) = {:.4}$ vs Student-t copula $C_{{\\rho,\\nu}}(u_1, u_2) = {:.4}$ ($\\nu = {:.1}$): Gaussian upper-tail dependence vanishes asymptotically ($\\lambda_U = 0$) whereas Student-t retains $\\lambda_U = {:.3}$, amplifying conditional crash probability at $u = {:.4}$ by ${:.2}\\times$ and exposing Kelly overbetting ruin.",
            self.metrics.sgp_gaussian_prob,
            self.metrics.student_t_joint_prob,
            self.config.degrees_of_freedom,
            self.metrics.asymptotic_lambda_u,
            self.config.tail_quantile_u,
            self.metrics.tail_multiplier
        )
    }

    fn phonetic_description(&self) -> String {
        "Gaussian versus Student-t copula tail dependence, Same Game Parlay pricing, and Kelly bankroll risk".into()
    }

    fn theory_citation(&self) -> String {
        "Joe, H. (2014). Dependence Modeling with Copulas; Genz, A. (2004). Numerical computation of rectangular bivariate and trivariate normal and t probabilities; Kelly, J. L. (1956).".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Gaussian SGP Pricing".into(),
            "Maps marginal leg probabilities through inverse normal quantiles and bivariate correlation".into(),
        );
        map.insert(
            "Student-t Tail Amplification".into(),
            "Shared chi-square volatility scale induces non-zero asymptotic upper/lower tail clustering".into(),
        );
        map
    }
}

/// Starter preset JSON payload reproducing `0.84%` (`rho=-0.50`), `11.29%` (`rho=0.70`), and `5.83x` at `u=0.999`.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Correlated SGP & 5.83x Student-t Tail Crash",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "leg1_prob": 0.2,
    "leg2_prob": 0.2,
    "correlation_rho": 0.5,
    "tail_quantile_u": 0.999,
    "degrees_of_freedom": 3.0,
    "offered_decimal_odds": 18.0,
    "stake_multiplier": 1.0
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CopulaTailRiskLab",
        display_name: "Copula Tail-Risk & SGP Lab",
        domain: "financial_math",
        tags: &["copula", "parlay", "sgp", "tail risk", "student-t", "kelly"],
        build: || Box::new(CopulaTailRiskLab::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "copula_sgp_tail_crash",
        title: "Correlated SGP & 5.83x Tail Crash",
        description: "C(u₁,u₂; ρ) swings a 20%×20% parlay from 0.84% (ρ=-0.50) to 11.29% (ρ=0.70), while Student-t (ν=3) amplifies u=0.999 tail crashes by 5.83× over Gaussian.",
        category: "Quantitative Finance",
        icon: "🎲",
        domain: "financial_math",
        tool_name: Some("CopulaTailRiskLab"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]
