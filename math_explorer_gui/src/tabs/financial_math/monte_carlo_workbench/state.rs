//! State serialization, analytical benchmark kernels, theory metadata, and starter presets.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    MonteCarloWorkbenchTool, HISTOGRAM_BINS, MAX_REPLICATES, MAX_SAMPLES, MIN_REPLICATES,
    MIN_SAMPLES,
};
use crate::framework::{StarterPresetMetadata, ToolMetadata};
use eframe::egui;
use egui_plot::{ExportPrimitive, PlotPoint};
use math_explorer::pure_math::statistics::monte_carlo::{normal_cdf_fast, normal_inv_cdf_fast};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Analytical benchmark family evaluated inside [`MonteCarloWorkbenchTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WorkbenchBenchmark {
    /// 4D discounted geometric basket call with exact Black-Scholes closed-form price.
    #[default]
    DiscountedBasketCall4D,
    /// 3D smooth separable exponential $\prod_{j=1}^3 e^{u_j}$ over $[0, 1]^3$.
    SmoothSeparable3D,
    /// Strongly right-skewed lognormal return $\exp(\sigma Z - \tfrac{1}{2}\sigma^2)$ with `E[X] = 1`.
    SkewedWealthReturn,
}

impl WorkbenchBenchmark {
    pub(super) const ALL: [Self; 3] = [
        Self::DiscountedBasketCall4D,
        Self::SmoothSeparable3D,
        Self::SkewedWealthReturn,
    ];

    /// Human-readable benchmark label for side-panel selection.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::DiscountedBasketCall4D => "4D Geometric Basket Call",
            Self::SmoothSeparable3D => "3D Smooth Exp Product",
            Self::SkewedWealthReturn => "Skewed Lognormal Return",
        }
    }

    /// Integration / sampling dimension $d$.
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self {
            Self::DiscountedBasketCall4D => 4,
            Self::SmoothSeparable3D => 3,
            Self::SkewedWealthReturn => 1,
        }
    }

    /// Exact expectation $\mathbb{E}[f(U)]$ used for RMSE convergence curves.
    #[must_use]
    pub fn exact_value(self) -> f64 {
        match self {
            Self::SmoothSeparable3D => (std::f64::consts::E - 1.0).powi(3),
            Self::SkewedWealthReturn => 1.0,
            Self::DiscountedBasketCall4D => {
                let (s0, strike, rate, sigma) = (100.0_f64, 100.0_f64, 0.05_f64, 0.30_f64);
                let sigma_g = 0.5 * sigma;
                let fwd_g = s0 * (rate - 0.5 * sigma * sigma + 0.5 * sigma_g * sigma_g).exp();
                let d1 = ((fwd_g / strike).ln() + 0.5 * sigma_g * sigma_g) / sigma_g;
                let d2 = d1 - sigma_g;
                (-rate).exp() * (fwd_g * normal_cdf_fast(d1) - strike * normal_cdf_fast(d2))
            }
        }
    }

    /// Known expectation $\mu_g = \mathbb{E}[g(U)]$ of the benchmark's control variate.
    #[must_use]
    pub const fn control_mean(self) -> f64 {
        match self {
            Self::SmoothSeparable3D => 1.5,
            Self::DiscountedBasketCall4D | Self::SkewedWealthReturn => 0.0,
        }
    }
}

/// Active layout and export view for [`MonteCarloWorkbenchTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WorkbenchViewMode {
    /// Side-by-side / stacked log-log convergence and `Percentile` vs `BCa` histogram.
    #[default]
    SplitOverview,
    /// Full-width log-log RMSE convergence decay across MC, variance reduction, and QMC.
    ErrorDecayLogPlot,
    /// Full-width bootstrap replicate density histogram with `Percentile` vs `BCa` intervals.
    BootstrapHistogram,
}

impl WorkbenchViewMode {
    pub(super) const ALL: [Self; 3] = [
        Self::SplitOverview,
        Self::ErrorDecayLogPlot,
        Self::BootstrapHistogram,
    ];

    /// Display label for the workspace view selector.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SplitOverview => "Split Overview",
            Self::ErrorDecayLogPlot => "Log-Log Error Decay",
            Self::BootstrapHistogram => "Percentile vs BCa",
        }
    }
}

/// Resampled functional statistic compared between `Percentile` and `BCa` bootstrap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BootstrapStatKind {
    /// Sample mean $\bar{X}$.
    #[default]
    Mean,
    /// Sample median $\operatorname{med}(X)$.
    Median,
}

impl BootstrapStatKind {
    /// Display label for bootstrap statistic radio buttons.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mean => "Mean",
            Self::Median => "Median",
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkbenchState {
    pub(super) benchmark: WorkbenchBenchmark,
    pub(super) view_mode: WorkbenchViewMode,
    pub(super) num_samples: usize,
    pub(super) bootstrap_replicates: usize,
    pub(super) confidence_level: f64,
    pub(super) importance_tilt: f64,
    pub(super) bootstrap_statistic: BootstrapStatKind,
    pub(super) seed: u64,
}

impl WorkbenchState {
    pub(super) fn validate(&self) -> Result<(), String> {
        if !(MIN_SAMPLES..=MAX_SAMPLES).contains(&self.num_samples) {
            return Err(format!(
                "num_samples {} outside {}..={}",
                self.num_samples, MIN_SAMPLES, MAX_SAMPLES
            ));
        }
        if !(MIN_REPLICATES..=MAX_REPLICATES).contains(&self.bootstrap_replicates) {
            return Err(format!(
                "bootstrap_replicates {} outside {}..={}",
                self.bootstrap_replicates, MIN_REPLICATES, MAX_REPLICATES
            ));
        }
        if !self.confidence_level.is_finite() || !(0.80..=0.99).contains(&self.confidence_level) {
            return Err("confidence_level must be finite within 0.80..=0.99".to_string());
        }
        if !self.importance_tilt.is_finite() || !(0.0..=3.5).contains(&self.importance_tilt) {
            return Err("importance_tilt must be finite within 0.0..=3.5".to_string());
        }
        Ok(())
    }
}

pub(super) fn evaluate_iid(bench: WorkbenchBenchmark, u: &[f64]) -> f64 {
    match bench {
        WorkbenchBenchmark::SmoothSeparable3D => u.iter().take(3).map(|&x| x.exp()).product(),
        WorkbenchBenchmark::SkewedWealthReturn => {
            let z = normal_inv_cdf_fast(u[0].clamp(1e-12, 1.0 - 1e-12));
            let sigma = 1.15;
            (sigma * z - 0.5 * sigma * sigma).exp()
        }
        WorkbenchBenchmark::DiscountedBasketCall4D => {
            let (s0, strike, rate, sigma) = (100.0_f64, 100.0_f64, 0.05_f64, 0.30_f64);
            let drift = rate - 0.5 * sigma * sigma;
            let mut sum_z = 0.0;
            for &uj in u.iter().take(4) {
                sum_z += normal_inv_cdf_fast(uj.clamp(1e-12, 1.0 - 1e-12));
            }
            let geo_spot = s0 * (drift + 0.25 * sigma * sum_z).exp();
            (-rate).exp() * (geo_spot - strike).max(0.0)
        }
    }
}

pub(super) fn evaluate_control(bench: WorkbenchBenchmark, u: &[f64]) -> (f64, f64) {
    let fx = evaluate_iid(bench, u);
    let gx = match bench {
        WorkbenchBenchmark::SmoothSeparable3D => u.iter().take(3).sum(),
        WorkbenchBenchmark::SkewedWealthReturn => {
            normal_inv_cdf_fast(u[0].clamp(1e-12, 1.0 - 1e-12))
        }
        WorkbenchBenchmark::DiscountedBasketCall4D => {
            0.25 * u
                .iter()
                .take(4)
                .map(|&uj| normal_inv_cdf_fast(uj.clamp(1e-12, 1.0 - 1e-12)))
                .sum::<f64>()
        }
    };
    (fx, gx)
}

pub(super) fn evaluate_snis(bench: WorkbenchBenchmark, u: &[f64], tilt: f64) -> (f64, f64) {
    match bench {
        WorkbenchBenchmark::SmoothSeparable3D => {
            if tilt.abs() < 1e-6 {
                return (evaluate_iid(bench, u), 1.0);
            }
            let exp_m1 = tilt.exp() - 1.0;
            let mut x = [0.0_f64; 3];
            let mut sum_x = 0.0;
            for (dst, &uj) in x.iter_mut().zip(u.iter().take(3)) {
                let xi = (1.0 + uj * exp_m1).ln() / tilt;
                *dst = xi;
                sum_x += xi;
            }
            let unnorm_weight = (-tilt * sum_x).exp();
            (sum_x.exp(), unnorm_weight)
        }
        WorkbenchBenchmark::SkewedWealthReturn => {
            let z_prop = normal_inv_cdf_fast(u[0].clamp(1e-12, 1.0 - 1e-12)) + tilt;
            let sigma = 1.15;
            let fx = (sigma * z_prop - 0.5 * sigma * sigma).exp();
            let weight = (-tilt * z_prop + 0.5 * tilt * tilt).exp();
            (fx, weight)
        }
        WorkbenchBenchmark::DiscountedBasketCall4D => {
            let (s0, strike, rate, sigma) = (100.0_f64, 100.0_f64, 0.05_f64, 0.30_f64);
            let drift = rate - 0.5 * sigma * sigma;
            let shift = 0.5 * tilt;
            let mut sum_z = 0.0;
            let mut log_w = 0.0;
            for &uj in u.iter().take(4) {
                let zj = normal_inv_cdf_fast(uj.clamp(1e-12, 1.0 - 1e-12)) + shift;
                sum_z += zj;
                log_w += -shift * zj + 0.5 * shift * shift;
            }
            let geo_spot = s0 * (drift + 0.25 * sigma * sum_z).exp();
            let fx = (-rate).exp() * (geo_spot - strike).max(0.0);
            (fx, log_w.exp())
        }
    }
}

impl TheoryDescribable for MonteCarloWorkbenchTool {
    fn theory_description(&self) -> String {
        format!(
            "Compares preallocated crude Monte Carlo $O(N^{{-1/2}})$, antithetic/control/LHS variance reduction, self-normalized importance sampling with Kong effective sample size $\\mathrm{{ESS}} = (\\sum w_i)^2 / \\sum w_i^2$, Cranley-Patterson randomized Halton/Joe-Kuo Sobol QMC $O(N^{{-1}}(\\log N)^d)$, and jackknife-accelerated $\\mathrm{{BCa}}$ bootstrap intervals ($N = {}$, $B = {}$).",
            self.num_samples, self.bootstrap_replicates
        )
    }

    fn phonetic_description(&self) -> String {
        "Monte Carlo variance reduction, Kong effective sample size, randomized Halton and Joe Kuo Sobol quasi Monte Carlo, and bias corrected accelerated bootstrap".to_string()
    }

    fn theory_citation(&self) -> String {
        "Joe, S. & Kuo, F. Y. (2008). Constructing Sobol sequences with better two-dimensional projections. SIAM J. Sci. Comput., 30(5), 2635-2654; Kong, A. (1992). Technical Report 348, U. Chicago; Efron, B. (1987). Better bootstrap confidence intervals. JASA, 82(397), 171-185.".to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Variance Reduction & Randomized QMC".to_string(),
            "Antithetic pairs, optimal-beta control variates, and Latin Hypercube reduce the variance constant, while Cranley-Patterson shifted Sobol nets accelerate smooth convergence toward O(N^-1).".to_string(),
        );
        map.insert(
            "Importance Sampling & Kong ESS".to_string(),
            "Tilting the proposal distribution toward rare high-payoff regions lowers variance until excessive tilt causes weight degeneracy and collapses ESS/N.".to_string(),
        );
        map.insert(
            "Percentile vs BCa Bootstrap".to_string(),
            "Efron's BCa adjusts empirical replicate quantiles using median-bias z0 and jackknife skewness acceleration a to correct asymmetric payoff coverage.".to_string(),
        );
        map
    }
}

inventory::submit! {
    ToolMetadata {
        id: "MonteCarloWorkbenchTool",
        display_name: "Monte Carlo & Bootstrap Workbench",
        domain: "financial_math",
        tags: &["monte_carlo", "qmc", "sobol", "bootstrap", "bca", "ess"],
        build: || Box::new(MonteCarloWorkbenchTool::default()),
    }
}

const PRESET_QMC_CONVERGENCE_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "QMC & Variance Reduction Convergence",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "benchmark": "SmoothSeparable3D",
    "view_mode": "SplitOverview",
    "num_samples": 1024,
    "bootstrap_replicates": 800,
    "confidence_level": 0.95,
    "importance_tilt": 0.95,
    "bootstrap_statistic": "Mean",
    "seed": 1337
  }
}"#;

const PRESET_BCA_SKEWED_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "BCa vs Percentile Skewed Payoff",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "benchmark": "SkewedWealthReturn",
    "view_mode": "BootstrapHistogram",
    "num_samples": 256,
    "bootstrap_replicates": 1200,
    "confidence_level": 0.95,
    "importance_tilt": 0.65,
    "bootstrap_statistic": "Mean",
    "seed": 2026
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "mc_qmc_variance_reduction_starter",
        title: "QMC & Variance Reduction Convergence",
        description: "Compare O(N^-1/2) crude Monte Carlo against control variates, LHS, and O(N^-1) randomized Joe-Kuo Sobol nets.",
        category: "Monte Carlo & QMC",
        icon: "🎯",
        domain: "financial_math",
        tool_name: Some("MonteCarloWorkbenchTool"),
        preset_generator: || PRESET_QMC_CONVERGENCE_JSON,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "mc_bca_skewed_payoff_starter",
        title: "BCa vs Percentile Skewed Payoff",
        description: "Inspect jackknife skewness acceleration and median-bias correction on heavy-tailed lognormal payoffs.",
        category: "Bootstrap Inference",
        icon: "📊",
        domain: "financial_math",
        tool_name: Some("MonteCarloWorkbenchTool"),
        preset_generator: || PRESET_BCA_SKEWED_JSON,
    }
}

impl MonteCarloWorkbenchTool {
    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.view_mode {
            WorkbenchViewMode::SplitOverview | WorkbenchViewMode::ErrorDecayLogPlot => vec![
                ("IID MC log10(RMSE)".into(), pts_to_pairs(&self.decay_iid)),
                (
                    "Antithetic log10(RMSE)".into(),
                    pts_to_pairs(&self.decay_antithetic),
                ),
                (
                    "Control Variate log10(RMSE)".into(),
                    pts_to_pairs(&self.decay_control),
                ),
                ("LHS log10(RMSE)".into(), pts_to_pairs(&self.decay_lhs)),
                (
                    "Halton RQMC log10(RMSE)".into(),
                    pts_to_pairs(&self.decay_halton),
                ),
                (
                    "Sobol RQMC log10(RMSE)".into(),
                    pts_to_pairs(&self.decay_sobol),
                ),
            ],
            WorkbenchViewMode::BootstrapHistogram => {
                let mut out = vec![(
                    "Bootstrap Replicate Density".into(),
                    self.hist_bins.iter().map(|&(x, y)| [x, y]).collect(),
                )];
                if let Some(comp) = self.boot_comparison {
                    let top = self.hist_max_density;
                    out.push((
                        "Percentile CI".into(),
                        vec![
                            [comp.percentile.ci_lower, 0.0],
                            [comp.percentile.ci_lower, top],
                            [comp.percentile.ci_upper, top],
                            [comp.percentile.ci_upper, 0.0],
                        ],
                    ));
                    out.push((
                        "BCa CI".into(),
                        vec![
                            [comp.bca.ci_lower, 0.0],
                            [comp.bca.ci_lower, top],
                            [comp.bca.ci_upper, top],
                            [comp.bca.ci_upper, 0.0],
                        ],
                    ));
                }
                out
            }
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let pal = crate::app::palette::DARK_PALETTE;
        match self.view_mode {
            WorkbenchViewMode::SplitOverview | WorkbenchViewMode::ErrorDecayLogPlot => vec![
                polyline_prim("IID MC", &self.decay_iid, pal.series[0], 2.0),
                polyline_prim("Antithetic", &self.decay_antithetic, pal.series[1], 1.8),
                polyline_prim("Control Variate", &self.decay_control, pal.series[2], 1.8),
                polyline_prim("Latin Hypercube", &self.decay_lhs, pal.series[3], 1.8),
                polyline_prim("Halton RQMC", &self.decay_halton, pal.series[4], 2.0),
                polyline_prim("Sobol RQMC", &self.decay_sobol, pal.success, 2.4),
            ],
            WorkbenchViewMode::BootstrapHistogram => {
                let mut prims = Vec::with_capacity(HISTOGRAM_BINS + 4);
                let half_w = 0.45 * self.hist_bin_width;
                for (i, &(cx, density)) in self.hist_bins.iter().enumerate() {
                    prims.push(ExportPrimitive::Bar {
                        name: format!("Bin[{i}]"),
                        min: [cx - half_w, 0.0],
                        max: [cx + half_w, density],
                        fill: pal.series[2],
                        stroke_color: pal.accent,
                        stroke_width: 1.0,
                    });
                }
                if let Some(comp) = self.boot_comparison {
                    let top = self.hist_max_density;
                    for (name, x, col) in [
                        ("Percentile Lower", comp.percentile.ci_lower, pal.warning),
                        ("Percentile Upper", comp.percentile.ci_upper, pal.warning),
                        ("BCa Lower", comp.bca.ci_lower, pal.success),
                        ("BCa Upper", comp.bca.ci_upper, pal.success),
                    ] {
                        prims.push(ExportPrimitive::Polyline {
                            name: name.into(),
                            points: vec![[x, 0.0], [x, top]],
                            color: col,
                            stroke_width: 2.2,
                            closed: false,
                            fill: None,
                        });
                    }
                }
                prims
            }
        }
    }
}

fn pts_to_pairs(pts: &[PlotPoint]) -> Vec<[f64; 2]> {
    pts.iter().map(|p| [p.x, p.y]).collect()
}

fn polyline_prim(
    name: &str,
    pts: &[PlotPoint],
    color: egui::Color32,
    stroke_width: f32,
) -> ExportPrimitive {
    ExportPrimitive::Polyline {
        name: name.to_string(),
        points: pts_to_pairs(pts),
        color,
        stroke_width,
        closed: false,
        fill: None,
    }
}

// [cite:clinical_trials_statistics]
