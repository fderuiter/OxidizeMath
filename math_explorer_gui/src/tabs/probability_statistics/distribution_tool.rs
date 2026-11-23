#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! State model, parameter validation, and `InteractiveTool` implementation for [`DistributionExplorerTool`].

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Maximum synthetic sample count per evaluation.
pub const MAX_DIST_SAMPLE_SIZE: usize = 512;

/// Maximum deterministic RNG seed exposed in UI controls.
pub const MAX_RNG_SEED: u64 = 1_000_000;

/// Supported continuous and discrete probability distribution families.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionFamily {
    Normal,
    Gamma,
    Beta,
    Weibull,
    StudentT,
    LogNormal,
    Exponential,
    ChiSquared,
    Uniform,
    Poisson,
    Binomial,
    NegativeBinomial,
    Geometric,
    Bernoulli,
    ZeroInflatedPoisson,
}

impl DistributionFamily {
    /// Ordered list of continuous families.
    pub const CONTINUOUS: [Self; 9] = [
        Self::Normal,
        Self::Gamma,
        Self::Beta,
        Self::Weibull,
        Self::StudentT,
        Self::LogNormal,
        Self::Exponential,
        Self::ChiSquared,
        Self::Uniform,
    ];

    /// Ordered list of discrete families.
    pub const DISCRETE: [Self; 6] = [
        Self::Poisson,
        Self::Binomial,
        Self::NegativeBinomial,
        Self::Geometric,
        Self::Bernoulli,
        Self::ZeroInflatedPoisson,
    ];

    /// Human-readable display label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Gamma => "Gamma",
            Self::Beta => "Beta",
            Self::Weibull => "Weibull",
            Self::StudentT => "Student-t",
            Self::LogNormal => "Log-Normal",
            Self::Exponential => "Exponential",
            Self::ChiSquared => "Chi-Squared",
            Self::Uniform => "Uniform",
            Self::Poisson => "Poisson",
            Self::Binomial => "Binomial",
            Self::NegativeBinomial => "Negative Binomial",
            Self::Geometric => "Geometric",
            Self::Bernoulli => "Bernoulli",
            Self::ZeroInflatedPoisson => "Zero-Inflated Poisson",
        }
    }

    /// Returns `true` when this family is discrete-valued on non-negative integers.
    #[must_use]
    pub const fn is_discrete(self) -> bool {
        matches!(
            self,
            Self::Poisson
                | Self::Binomial
                | Self::NegativeBinomial
                | Self::Geometric
                | Self::Bernoulli
                | Self::ZeroInflatedPoisson
        )
    }

    /// Recommended default `(param_a, param_b)` pair when switching families.
    #[must_use]
    pub const fn default_params(self) -> (f64, f64) {
        match self {
            Self::Normal => (0.0, 1.2),
            Self::Gamma => (3.0, 1.5),
            Self::Beta => (2.5, 4.0),
            Self::Weibull => (2.2, 1.8),
            Self::StudentT => (4.5, 1.0),
            Self::LogNormal => (0.2, 0.55),
            Self::Exponential => (1.2, 1.0),
            Self::ChiSquared => (5.0, 1.0),
            Self::Uniform => (-2.0, 3.0),
            Self::Poisson => (4.2, 1.0),
            Self::Binomial => (16.0, 0.35),
            Self::NegativeBinomial => (4.0, 0.42),
            Self::Geometric => (0.32, 1.0),
            Self::Bernoulli => (0.62, 1.0),
            Self::ZeroInflatedPoisson => (0.35, 3.6),
        }
    }
}

/// Serializable configuration for [`DistributionExplorerTool`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DistributionToolConfig {
    /// Target probability level $p \in [0.01, 0.99]$ for quantile evaluation.
    pub quantile_p: f64,
    /// Primary distribution parameter (`-8.0..=30.0`).
    pub param_a: f64,
    /// Secondary distribution parameter (`0.05..=30.0`).
    pub param_b: f64,
    /// Synthetic sample count drawn from `OxidizeRng` (`16..=512`).
    pub sample_size: usize,
    /// Deterministic RNG seed (`1..=1_000_000`).
    pub seed: u64,
    /// Selected parametric family.
    pub family: DistributionFamily,
}

impl Default for DistributionToolConfig {
    fn default() -> Self {
        Self {
            quantile_p: 0.50,
            param_a: 0.0,
            param_b: 1.2,
            sample_size: 160,
            seed: 42,
            family: DistributionFamily::Normal,
        }
    }
}

impl DistributionToolConfig {
    /// Validates numeric finiteness and parameter bounds.
    pub fn validate(&self) -> Result<(), String> {
        if !self.quantile_p.is_finite() || !(0.01..=0.99).contains(&self.quantile_p) {
            return Err("quantile_p must be finite within [0.01, 0.99]".to_string());
        }
        if !self.param_a.is_finite() || !(-8.0..=30.0).contains(&self.param_a) {
            return Err("param_a must be finite within [-8.0, 30.0]".to_string());
        }
        if !self.param_b.is_finite() || !(0.05..=30.0).contains(&self.param_b) {
            return Err("param_b must be finite within [0.05, 30.0]".to_string());
        }
        if !(16..=MAX_DIST_SAMPLE_SIZE).contains(&self.sample_size) {
            return Err(format!(
                "sample_size must be within 16..={MAX_DIST_SAMPLE_SIZE}"
            ));
        }
        if self.seed == 0 || self.seed > MAX_RNG_SEED {
            return Err(format!("seed must be within 1..={MAX_RNG_SEED}"));
        }
        Ok(())
    }
}

/// Evaluated curves, moments, quantile, and fitted parameter summaries.
#[derive(Clone, Debug, Default)]
pub struct DistributionSnapshot {
    pub density_curve: Vec<[f64; 2]>,
    pub cdf_curve: Vec<[f64; 2]>,
    pub histogram_bars: Vec<[f64; 3]>,
    pub discrete_bars: Vec<(u64, f64)>,
    pub mean: Option<f64>,
    pub variance: Option<f64>,
    pub quantile_x: f64,
    pub mle_summary: String,
    pub mom_summary: String,
    pub param_labels: (&'static str, Option<&'static str>),
}

/// Interactive parametric probability distribution and MLE/MoM fitting workbench.
pub struct DistributionExplorerTool {
    pub(crate) config: DistributionToolConfig,
    pub(crate) snapshot: DistributionSnapshot,
    pub(crate) revision: u64,
}

impl Default for DistributionExplorerTool {
    fn default() -> Self {
        let mut tool = Self {
            config: DistributionToolConfig::default(),
            snapshot: DistributionSnapshot::default(),
            revision: 1,
        };
        tool.recompute();
        tool
    }
}

impl DistributionExplorerTool {
    #[inline]
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Validates and applies a new [`DistributionToolConfig`].
    pub fn apply_config(&mut self, cfg: DistributionToolConfig) -> Result<(), String> {
        cfg.validate()?;
        self.config = cfg;
        self.recompute();
        self.bump_revision();
        Ok(())
    }
}

impl TheoryDescribable for DistributionExplorerTool {
    fn theory_description(&self) -> String {
        "Parametric continuous and discrete probability laws map real or count outcomes onto \
         exact density f(x), cumulative probability F(x), and safeguarded Newton-bisection \
         quantiles Q(p). Seeded synthetic draws via OxidizeRng allow side-by-side comparison \
         of Maximum Likelihood Estimation (MLE) and Method-of-Moments (MoM) parameter recovery."
            .to_string()
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(r"F(x) = \int_{-\infty}^{x} f(t)\,dt, \quad \ell(\theta) = \sum_{i=1}^{n} \ln f(x_i \mid \theta)".to_string())
    }

    fn phonetic_description(&self) -> String {
        "Cumulative probability F of x equals the integral of density f of t, with log-likelihood \
         sum of log f of x sub i given theta"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Casella, G., & Berger, R. L. (2002). Statistical Inference (2nd ed.); Lambert, D. (1992). \
         Zero-Inflated Poisson Regression, Technometrics 34(1), 1-14."
            .to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("overview".to_string(), self.theory_description());
        map.insert("citation".to_string(), self.theory_citation());
        map
    }
}

impl InteractiveTool for DistributionExplorerTool {
    fn name(&self) -> &'static str {
        "Distribution Explorer"
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
        self.config.quantile_p = if self.config.quantile_p.is_finite() {
            self.config.quantile_p.clamp(0.01, 0.99)
        } else {
            0.50
        };
        self.config.param_a = if self.config.param_a.is_finite() {
            self.config.param_a.clamp(-8.0, 30.0)
        } else {
            0.0
        };
        self.config.param_b = if self.config.param_b.is_finite() {
            self.config.param_b.clamp(0.05, 30.0)
        } else {
            1.0
        };
        self.config.sample_size = self.config.sample_size.clamp(16, MAX_DIST_SAMPLE_SIZE);
        self.config.seed = self.config.seed.clamp(1, MAX_RNG_SEED);
        self.recompute();
        self.bump_revision();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: DistributionToolConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid DistributionExplorer state: {e}"))?;
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

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
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
