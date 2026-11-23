#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Conjugate posterior updating and pre-allocated zero-allocation Metropolis-Hastings MCMC
//! engine for [`BayesianInferenceTool`].

use super::bayesian_analytical::{compute_analytical, eval_log_posterior};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::pure_math::statistics::bayesian::{
    BayesFactor, CredibleInterval, JeffreysEvidence, McmcSummary, MetropolisHastings1D,
};
use math_explorer::rng::OxidizeRng;
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Maximum retained MCMC posterior samples in the pre-allocated chain buffer.
pub const MAX_MCMC_SAMPLES: usize = 2_048;

/// Maximum Metropolis-Hastings transitions stepped per UI frame.
pub const MAX_MCMC_STEPS_PER_FRAME: usize = 64;

/// Conjugate model family paired with 1D Metropolis-Hastings verification.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConjugateFamily {
    BetaBinomial,
    GammaPoisson,
    NormalNormal,
    NormalInverseGamma,
}

impl ConjugateFamily {
    pub const ALL: [Self; 4] = [
        Self::BetaBinomial,
        Self::GammaPoisson,
        Self::NormalNormal,
        Self::NormalInverseGamma,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BetaBinomial => "Beta-Binomial",
            Self::GammaPoisson => "Gamma-Poisson",
            Self::NormalNormal => "Normal-Normal",
            Self::NormalInverseGamma => "Normal-Inv-Gamma",
        }
    }
}

/// Serializable configuration for [`BayesianInferenceTool`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BayesianToolConfig {
    /// Gaussian random-walk proposal standard deviation (`0.02..=2.50`).
    pub proposal_scale: f64,
    /// Credible interval probability mass (`0.50..=0.99`).
    pub credible_level: f64,
    /// Primary prior hyperparameter (`-5.0..=20.0`).
    pub prior_a: f64,
    /// Secondary prior hyperparameter (`0.2..=20.0`).
    pub prior_b: f64,
    /// Observed sufficient statistic / sample mean (`-5.0..=60.0`).
    pub obs_stat: f64,
    /// Observed trial / sample count (`1..=100`).
    pub obs_count: usize,
    /// MCMC transitions executed per live UI frame (`1..=64`).
    pub steps_per_frame: usize,
    /// Deterministic warmup draws executed on reset (`32..=1024`).
    pub warmup_draws: usize,
    /// Deterministic `OxidizeRng` seed (`1..=1_000_000`).
    pub seed: u64,
    /// Selected conjugate model.
    pub family: ConjugateFamily,
}

impl Default for BayesianToolConfig {
    fn default() -> Self {
        Self {
            proposal_scale: 0.14,
            credible_level: 0.95,
            prior_a: 2.0,
            prior_b: 2.0,
            obs_stat: 14.0,
            obs_count: 20,
            steps_per_frame: 24,
            warmup_draws: 320,
            seed: 101,
            family: ConjugateFamily::BetaBinomial,
        }
    }
}

impl BayesianToolConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.proposal_scale.is_finite() || !(0.02..=2.50).contains(&self.proposal_scale) {
            return Err("proposal_scale must be finite within [0.02, 2.50]".to_string());
        }
        if !self.credible_level.is_finite() || !(0.50..=0.99).contains(&self.credible_level) {
            return Err("credible_level must be finite within [0.50, 0.99]".to_string());
        }
        if !self.prior_a.is_finite() || !(-5.0..=20.0).contains(&self.prior_a) {
            return Err("prior_a must be finite within [-5.0, 20.0]".to_string());
        }
        if !self.prior_b.is_finite() || !(0.2..=20.0).contains(&self.prior_b) {
            return Err("prior_b must be finite within [0.2, 20.0]".to_string());
        }
        if !self.obs_stat.is_finite() || !(-5.0..=60.0).contains(&self.obs_stat) {
            return Err("obs_stat must be finite within [-5.0, 60.0]".to_string());
        }
        if !(1..=100).contains(&self.obs_count) {
            return Err("obs_count must be within 1..=100".to_string());
        }
        if !(1..=MAX_MCMC_STEPS_PER_FRAME).contains(&self.steps_per_frame) {
            return Err("steps_per_frame must be within 1..=64".to_string());
        }
        if !(32..=1_024).contains(&self.warmup_draws) {
            return Err("warmup_draws must be within 32..=1024".to_string());
        }
        if self.seed == 0 || self.seed > 1_000_000 {
            return Err("seed must be within 1..=1_000_000".to_string());
        }
        Ok(())
    }
}

/// Cached analytical conjugate curves, intervals, and Bayes factor.
#[derive(Clone, Debug)]
pub struct BayesianAnalyticalCache {
    pub prior_curve: Vec<[f64; 2]>,
    pub posterior_curve: Vec<[f64; 2]>,
    pub plot_bounds: (f64, f64),
    pub posterior_mean: f64,
    pub posterior_std: f64,
    pub equal_tailed: CredibleInterval,
    pub hpd: CredibleInterval,
    pub bayes_factor: BayesFactor,
}

/// Interactive Conjugate & Zero-Allocation MCMC Bayesian Inference Workbench.
pub struct BayesianInferenceTool {
    pub(crate) config: BayesianToolConfig,
    pub(crate) analytical: BayesianAnalyticalCache,
    pub(crate) sampler: MetropolisHastings1D,
    pub(crate) rng: OxidizeRng,
    pub(crate) chain_buffer: Vec<f64>,
    pub(crate) chain_cursor: usize,
    pub(crate) chain_len: usize,
    pub(crate) mcmc_summary: McmcSummary,
    pub(crate) is_playing: bool,
    pub(crate) revision: u64,
}

impl Default for BayesianInferenceTool {
    fn default() -> Self {
        let cfg = BayesianToolConfig::default();
        let analytical = compute_analytical(&cfg);
        let sampler = MetropolisHastings1D::new(analytical.posterior_mean, cfg.proposal_scale)
            .expect("valid default MH sampler");
        let mut tool = Self {
            rng: OxidizeRng::from_seed(cfg.seed),
            chain_buffer: vec![0.0_f64; MAX_MCMC_SAMPLES],
            chain_cursor: 0,
            chain_len: 0,
            mcmc_summary: McmcSummary {
                mean: analytical.posterior_mean,
                variance: analytical.posterior_std * analytical.posterior_std,
                acceptance_rate: 0.0,
                effective_sample_size: 0.0,
            },
            analytical,
            sampler,
            is_playing: false,
            config: cfg,
            revision: 1,
        };
        tool.rebuild_chain();
        tool
    }
}

impl BayesianInferenceTool {
    #[inline]
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(crate) fn rebuild_chain(&mut self) {
        self.analytical = compute_analytical(&self.config);
        self.rng = OxidizeRng::from_seed(self.config.seed);
        let init = self.analytical.posterior_mean;
        self.sampler = MetropolisHastings1D::new(init, self.config.proposal_scale.clamp(0.02, 2.50))
            .expect("clamped proposal scale");
        self.chain_cursor = 0;
        self.chain_len = 0;
        let warmup = self.config.warmup_draws.clamp(32, 1_024);
        self.step_mcmc_draws(128, warmup);
        self.bump_revision();
    }

    /// Advances the pre-allocated MCMC chain by `n_draws` with zero heap allocations.
    pub fn step_mcmc_draws(&mut self, burn_in: usize, n_draws: usize) {
        let cfg = self.config.clone();
        for _ in 0..burn_in {
            let _ = self
                .sampler
                .step(|theta| eval_log_posterior(&cfg, theta), &mut self.rng);
        }
        let mut batch = [0.0_f64; MAX_MCMC_STEPS_PER_FRAME];
        let mut remaining = n_draws;
        while remaining > 0 {
            let chunk = remaining.min(MAX_MCMC_STEPS_PER_FRAME);
            if self
                .sampler
                .sample_into(
                    |theta| eval_log_posterior(&cfg, theta),
                    0,
                    &mut batch[..chunk],
                    &mut self.rng,
                )
                .is_ok()
            {
                for &val in &batch[..chunk] {
                    self.chain_buffer[self.chain_cursor] = val;
                    self.chain_cursor = (self.chain_cursor + 1) % MAX_MCMC_SAMPLES;
                    self.chain_len = (self.chain_len + 1).min(MAX_MCMC_SAMPLES);
                }
            }
            remaining -= chunk;
        }
        self.refresh_mcmc_summary();
    }

    fn refresh_mcmc_summary(&mut self) {
        let active = &self.chain_buffer[..self.chain_len];
        if active.is_empty() {
            return;
        }
        let n = active.len() as f64;
        let mean = active.iter().sum::<f64>() / n;
        let variance = if active.len() > 1 {
            active.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        let lag1 = if active.len() > 2 && variance > 1e-16 {
            let cov: f64 = active
                .windows(2)
                .map(|w| (w[0] - mean) * (w[1] - mean))
                .sum::<f64>()
                / (n - 1.0);
            (cov / variance).clamp(-0.95, 0.99)
        } else {
            0.0
        };
        let ess = (n * (1.0 - lag1) / (1.0 + lag1).max(1e-3)).clamp(1.0, n);
        self.mcmc_summary = McmcSummary {
            mean,
            variance,
            acceptance_rate: self.sampler.acceptance_rate(),
            effective_sample_size: ess,
        };
    }

    pub fn apply_config(&mut self, cfg: BayesianToolConfig) -> Result<(), String> {
        cfg.validate()?;
        self.config = cfg;
        self.is_playing = false;
        self.rebuild_chain();
        Ok(())
    }

    #[must_use]
    pub fn jeffreys_label(&self) -> &'static str {
        match self.analytical.bayes_factor.evidence {
            JeffreysEvidence::SupportsNull => "Supports Null M0",
            JeffreysEvidence::Anecdotal => "Anecdotal for M1",
            JeffreysEvidence::Substantial => "Substantial for M1",
            JeffreysEvidence::Strong => "Strong for M1",
            JeffreysEvidence::VeryStrong => "Very Strong for M1",
            JeffreysEvidence::Decisive => "Decisive for M1",
        }
    }
}

impl TheoryDescribable for BayesianInferenceTool {
    fn theory_description(&self) -> String {
        "Conjugate Bayesian updating maps prior beliefs p(theta) and observed sufficient \
         statistics D onto closed-form posteriors p(theta | D), exact Highest Posterior Density \
         (HPD) credible intervals, and Jeffreys marginal Bayes factors BF10. Simultaneously, a \
         zero-allocation Random-Walk Metropolis-Hastings Markov chain samples the unnormalized \
         log-posterior into a pre-allocated ring buffer to verify asymptotic convergence."
            .to_string()
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(r"p(\theta \mid D) = \frac{p(D \mid \theta)\,p(\theta)}{p(D)}, \quad BF_{10} = \frac{p(D \mid M_1)}{p(D \mid M_0)}".to_string())
    }

    fn phonetic_description(&self) -> String {
        "Posterior p of theta given data D equals likelihood times prior divided by marginal \
         evidence, with Bayes factor B F ten comparing model one against model zero"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Jeffreys, H. (1961). Theory of Probability (3rd ed.); Hastings, W. K. (1970). Monte Carlo \
         Sampling Methods Using Markov Chains, Biometrika 57(1), 97-109."
            .to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("overview".to_string(), self.theory_description());
        map.insert("citation".to_string(), self.theory_citation());
        map
    }
}

impl InteractiveTool for BayesianInferenceTool {
    fn name(&self) -> &'static str {
        "Bayesian Inference Lab"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn domain(&self) -> &'static str {
        "probability_statistics"
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 8 * 1024,
            max_depth: 6,
            max_array_len: 32,
        }
    }

    fn has_finite_parameters(&self) -> bool {
        self.config.validate().is_ok()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.config.proposal_scale = if self.config.proposal_scale.is_finite() {
            self.config.proposal_scale.clamp(0.02, 2.50)
        } else {
            0.20
        };
        self.config.credible_level = if self.config.credible_level.is_finite() {
            self.config.credible_level.clamp(0.50, 0.99)
        } else {
            0.95
        };
        self.config.prior_a = if self.config.prior_a.is_finite() {
            self.config.prior_a.clamp(-5.0, 20.0)
        } else {
            2.0
        };
        self.config.prior_b = if self.config.prior_b.is_finite() {
            self.config.prior_b.clamp(0.2, 20.0)
        } else {
            2.0
        };
        self.config.obs_stat = if self.config.obs_stat.is_finite() {
            self.config.obs_stat.clamp(-5.0, 60.0)
        } else {
            5.0
        };
        self.config.obs_count = self.config.obs_count.clamp(1, 100);
        self.config.steps_per_frame = self.config.steps_per_frame.clamp(1, MAX_MCMC_STEPS_PER_FRAME);
        self.config.warmup_draws = self.config.warmup_draws.clamp(32, 1_024);
        self.config.seed = self.config.seed.clamp(1, 1_000_000);
        self.rebuild_chain();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: BayesianToolConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid BayesianInference state: {e}"))?;
        self.apply_config(parsed)
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }

    fn step_simulation(&mut self) {
        self.step_mcmc_draws(0, 1);
        self.bump_revision();
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if self.is_playing {
            self.step_mcmc_draws(0, self.config.steps_per_frame);
            self.bump_revision();
            parent_ui.ctx().request_repaint();
        }
        self.render_workbench(parent_ui);
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

// [cite:clinical_trials_statistics]
