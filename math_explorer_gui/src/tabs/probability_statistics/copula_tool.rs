#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Multivariate Gaussian Copula (2D-5D QMC), Fréchet-Hoeffding bounds, Student Growth
//! Percentile conditioning, and Zero-Inflated Poisson EM regression workbench.

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::statistics::copula::{
    CorrelationMatrix, GaussianCopula, Probability,
};
use math_explorer::pure_math::statistics::distributions::{ContinuousDistribution, NormalDist};
use math_explorer::pure_math::statistics::zip_regression::{
    ZipDistribution, ZipParams, ZipRegression,
};
use math_explorer::rng::OxidizeRng;
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Serializable configuration for [`CopulaDependenceTool`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CopulaToolConfig {
    /// Equicorrelation parameter $\rho \in [-0.45, 0.92]$.
    pub correlation_rho: f64,
    /// First marginal cumulative probability $u_1 \in [0.05, 0.95]$.
    pub marginal_u1: f64,
    /// Second marginal cumulative probability $u_2 \in [0.05, 0.95]$.
    pub marginal_u2: f64,
    /// Zero-Inflated Poisson baseline zero-inflation $\pi \in [0.05, 0.85]$.
    pub zip_zero_prob: f64,
    /// Multivariate copula dimension $d \in 2..=5$.
    pub dimension: usize,
    /// Antithetic Genz QMC sample pairs $N \in 64..=1024$.
    pub qmc_samples: usize,
    /// Deterministic `OxidizeRng` seed (`1..=1_000_000`).
    pub seed: u64,
}

impl Default for CopulaToolConfig {
    fn default() -> Self {
        Self {
            correlation_rho: 0.45,
            marginal_u1: 0.65,
            marginal_u2: 0.70,
            zip_zero_prob: 0.35,
            dimension: 3,
            qmc_samples: 256,
            seed: 7,
        }
    }
}

impl CopulaToolConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.correlation_rho.is_finite() || !(-0.48..=0.95).contains(&self.correlation_rho) {
            return Err("correlation_rho must be finite within [-0.48, 0.95]".to_string());
        }
        if !self.marginal_u1.is_finite() || !(0.05..=0.95).contains(&self.marginal_u1) {
            return Err("marginal_u1 must be finite within [0.05, 0.95]".to_string());
        }
        if !self.marginal_u2.is_finite() || !(0.05..=0.95).contains(&self.marginal_u2) {
            return Err("marginal_u2 must be finite within [0.05, 0.95]".to_string());
        }
        if !self.zip_zero_prob.is_finite() || !(0.05..=0.85).contains(&self.zip_zero_prob) {
            return Err("zip_zero_prob must be finite within [0.05, 0.85]".to_string());
        }
        if !(2..=5).contains(&self.dimension) {
            return Err("dimension must be within 2..=5".to_string());
        }
        if !(64..=1_024).contains(&self.qmc_samples) {
            return Err("qmc_samples must be within 64..=1024".to_string());
        }
        if self.seed == 0 || self.seed > 1_000_000 {
            return Err("seed must be within 1..=1_000_000".to_string());
        }
        Ok(())
    }
}

/// Evaluated curves, Fréchet-Hoeffding bounds, Cholesky status, and ZIP EM regression readout.
#[derive(Clone, Debug, Default)]
pub struct CopulaSnapshot {
    pub sweep_copula: Vec<[f64; 2]>,
    pub sweep_independence: Vec<[f64; 2]>,
    pub sweep_frechet_low: Vec<[f64; 2]>,
    pub sweep_frechet_high: Vec<[f64; 2]>,
    pub sgp_conditional_curve: Vec<[f64; 2]>,
    pub active_joint_cdf: Option<f64>,
    pub independence_cdf: f64,
    pub frechet_bounds: (f64, f64),
    pub cholesky_error: Option<String>,
    pub zip_em_summary: String,
}

/// Interactive Multivariate Copula, SGP Conditioning, and ZIP EM Regression Workbench.
pub struct CopulaDependenceTool {
    pub(crate) config: CopulaToolConfig,
    pub(crate) snapshot: CopulaSnapshot,
    pub(crate) revision: u64,
}

impl Default for CopulaDependenceTool {
    fn default() -> Self {
        let mut tool = Self {
            config: CopulaToolConfig::default(),
            snapshot: CopulaSnapshot::default(),
            revision: 1,
        };
        tool.recompute();
        tool
    }
}

impl CopulaDependenceTool {
    #[inline]
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub fn apply_config(&mut self, cfg: CopulaToolConfig) -> Result<(), String> {
        cfg.validate()?;
        self.config = cfg;
        self.recompute();
        self.bump_revision();
        Ok(())
    }

    pub(crate) fn recompute(&mut self) {
        let d = self.config.dimension.clamp(2, 5);
        let u_vec = build_marginals(d, self.config.marginal_u1, self.config.marginal_u2);
        let indep: f64 = u_vec.iter().map(Probability::value).product();
        let sum_u: f64 = u_vec.iter().map(Probability::value).sum();
        let fh_low = (sum_u - (d as f64) + 1.0).max(0.0);
        let fh_high = u_vec
            .iter()
            .map(Probability::value)
            .fold(1.0_f64, f64::min);

        let cholesky = build_equicorrelation(d, self.config.correlation_rho);
        let (active_joint_cdf, cholesky_error) = match cholesky {
            Ok(corr) => {
                let cop = GaussianCopula::new(corr);
                let mut rng = OxidizeRng::from_seed(self.config.seed);
                let val = cop
                    .cdf_with_rng(&u_vec, &mut rng, self.config.qmc_samples)
                    .map(|p| p.value())
                    .ok();
                (val, None)
            }
            Err(err_msg) => (None, Some(err_msg)),
        };

        let (sweep_copula, sweep_independence, sweep_frechet_low, sweep_frechet_high) =
            compute_rho_sweep(d, &u_vec, indep, fh_low, fh_high, self.config.seed, self.config.qmc_samples);
        let sgp_conditional_curve =
            compute_sgp_conditional_curve(self.config.marginal_u1, self.config.correlation_rho);
        let zip_em_summary =
            compute_zip_em_summary(self.config.zip_zero_prob, self.config.seed);

        self.snapshot = CopulaSnapshot {
            sweep_copula,
            sweep_independence,
            sweep_frechet_low,
            sweep_frechet_high,
            sgp_conditional_curve,
            active_joint_cdf,
            independence_cdf: indep,
            frechet_bounds: (fh_low, fh_high),
            cholesky_error,
            zip_em_summary,
        };
    }
}

fn build_marginals(d: usize, u1: f64, u2: f64) -> Vec<Probability> {
    let mid = (0.5 * (u1 + u2)).clamp(0.05, 0.95);
    (0..d)
        .map(|i| {
            let v = if i == 0 {
                u1
            } else if i == 1 {
                u2
            } else {
                mid
            };
            Probability::new(v.clamp(0.05, 0.95)).expect("clamped probability")
        })
        .collect()
}

fn build_equicorrelation(d: usize, rho: f64) -> Result<CorrelationMatrix, String> {
    let mat = DMatrix::from_fn(d, d, |r, c| if r == c { 1.0 } else { rho });
    CorrelationMatrix::new(mat).map_err(|e| e.to_string())
}

type RhoSweepSeries = (Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<[f64; 2]>);

fn compute_rho_sweep(
    d: usize,
    u_vec: &[Probability],
    indep: f64,
    fh_low: f64,
    fh_high: f64,
    seed: u64,
    qmc_samples: usize,
) -> RhoSweepSeries {
    let rho_min = (-1.0 / ((d - 1) as f64)) + 0.03;
    let rho_max = 0.92;
    let steps = 36;
    let mut cop_pts = Vec::with_capacity(steps);
    let mut ind_pts = Vec::with_capacity(steps);
    let mut lo_pts = Vec::with_capacity(steps);
    let mut hi_pts = Vec::with_capacity(steps);

    for idx in 0..steps {
        let rho = rho_min + (idx as f64 / ((steps - 1) as f64)) * (rho_max - rho_min);
        if let Ok(corr) = build_equicorrelation(d, rho) {
            let cop = GaussianCopula::new(corr);
            let mut rng = OxidizeRng::from_seed(seed);
            if let Ok(val) = cop.cdf_with_rng(u_vec, &mut rng, qmc_samples.min(256)) {
                cop_pts.push([rho, val.value()]);
                ind_pts.push([rho, indep]);
                lo_pts.push([rho, fh_low]);
                hi_pts.push([rho, fh_high]);
            }
        }
    }
    (cop_pts, ind_pts, lo_pts, hi_pts)
}

fn compute_sgp_conditional_curve(u1: f64, rho_raw: f64) -> Vec<[f64; 2]> {
    let rho = rho_raw.clamp(-0.92, 0.92);
    let std_norm = NormalDist::standard();
    let z1 = std_norm.quantile(u1.clamp(0.05, 0.95)).unwrap_or(0.0);
    let denom = (1.0 - rho * rho).max(1e-6).sqrt();
    (1..64)
        .map(|i| {
            let u2 = i as f64 / 64.0;
            let z2 = std_norm.quantile(u2).unwrap_or(0.0);
            let cond = std_norm.cdf((z2 - rho * z1) / denom);
            [u2, cond]
        })
        .collect()
}

fn compute_zip_em_summary(zero_prob: f64, seed: u64) -> String {
    let mut rng = OxidizeRng::from_seed(seed.wrapping_add(900));
    let n = 48_usize;
    let mut counts = Vec::with_capacity(n);
    let mut x_rows = Vec::with_capacity(n * 2);
    let mut z_rows = Vec::with_capacity(n * 2);
    let logit_base = (zero_prob / (1.0 - zero_prob)).ln();

    for i in 0..n {
        let cov = -1.0 + 2.0 * (i as f64 / ((n - 1) as f64));
        let lam = (1.1 + 0.45 * cov).exp();
        let pi = 1.0 / (1.0 + (-(logit_base - 0.5 * cov)).exp());
        let dist = ZipDistribution::new(
            ZipParams::from_values(pi.clamp(0.01, 0.98), lam).expect("valid synthetic ZIP"),
        );
        counts.push(dist.sample(&mut rng));
        x_rows.push(1.0);
        x_rows.push(cov);
        z_rows.push(1.0);
        z_rows.push(cov);
    }

    let x_mat = DMatrix::from_row_slice(n, 2, &x_rows);
    let z_mat = DMatrix::from_row_slice(n, 2, &z_rows);
    match ZipRegression::new(counts, x_mat, z_mat).and_then(|reg| reg.fit(120, 1e-6)) {
        Ok(res) => format!(
            "EM converged in {} iters | logL={:.2} | beta=[{:.2}, {:.2}] | alpha=[{:.2}, {:.2}]",
            res.iterations,
            res.log_likelihood,
            first_or_zero(&res.beta, 0),
            first_or_zero(&res.beta, 1),
            first_or_zero(&res.alpha, 0),
            first_or_zero(&res.alpha, 1),
        ),
        Err(e) => format!("EM status: {e}"),
    }
}

fn first_or_zero(v: &DVector<f64>, idx: usize) -> f64 {
    v.get(idx).copied().unwrap_or(0.0)
}

impl TheoryDescribable for CopulaDependenceTool {
    fn theory_description(&self) -> String {
        "Sklar's theorem decomposes any multivariate joint distribution into its univariate \
         marginals u_i = F_i(x_i) coupled by a dependence copula C_R(u_1, ..., u_d). Positive \
         definiteness of R is certified via Cholesky factorization, bivariate integrals use \
         16-point Gauss-Legendre Plackett quadrature, and dimensions 3 through 5 use Genz \
         separation-of-variables Quasi-Monte Carlo bounded by the Frechet-Hoeffding envelope."
            .to_string()
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(r"\max\!\Big(\sum_{i=1}^{d} u_i - d + 1, 0\Big) \le C_R(u_1, \ldots, u_d) \le \min_{i} u_i".to_string())
    }

    fn phonetic_description(&self) -> String {
        "Gaussian copula joint cumulative probability C sub R of u one through u sub d bounded \
         between the Frechet Hoeffding lower and upper envelopes"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Genz, A. (1992). Numerical Computation of Multivariate Normal Probabilities, JCGS 1(2), \
         141-149; Nelsen, R. B. (2006). An Introduction to Copulas (2nd ed.)."
            .to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("overview".to_string(), self.theory_description());
        map.insert("citation".to_string(), self.theory_citation());
        map
    }
}

impl InteractiveTool for CopulaDependenceTool {
    fn name(&self) -> &'static str {
        "Multivariate Copula Lab"
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
        self.config.correlation_rho = if self.config.correlation_rho.is_finite() {
            self.config.correlation_rho.clamp(-0.48, 0.95)
        } else {
            0.35
        };
        self.config.marginal_u1 = if self.config.marginal_u1.is_finite() {
            self.config.marginal_u1.clamp(0.05, 0.95)
        } else {
            0.65
        };
        self.config.marginal_u2 = if self.config.marginal_u2.is_finite() {
            self.config.marginal_u2.clamp(0.05, 0.95)
        } else {
            0.70
        };
        self.config.zip_zero_prob = if self.config.zip_zero_prob.is_finite() {
            self.config.zip_zero_prob.clamp(0.05, 0.85)
        } else {
            0.35
        };
        self.config.dimension = self.config.dimension.clamp(2, 5);
        self.config.qmc_samples = self.config.qmc_samples.clamp(64, 1_024);
        self.config.seed = self.config.seed.clamp(1, 1_000_000);
        self.recompute();
        self.bump_revision();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: CopulaToolConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid CopulaDependence state: {e}"))?;
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
