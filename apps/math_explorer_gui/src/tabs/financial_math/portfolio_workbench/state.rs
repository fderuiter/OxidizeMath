#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::nalgebra::DMatrix;
use math_explorer::pure_math::statistics::copula::{Correlation, CorrelationMatrix};

pub const MIN_WORKBENCH_ASSETS: usize = 1;
pub const MAX_WORKBENCH_ASSETS: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WorkbenchView {
    KellyFrontier,
    CopulaTailCrash,
    OuMeanReversion,
    BlackScholesGreeks,
}

impl WorkbenchView {
    pub fn label(self) -> &'static str {
        match self {
            Self::KellyFrontier => "Kelly & VaR/CVaR",
            Self::CopulaTailCrash => "Copula Tail Crash",
            Self::OuMeanReversion => "OU Forecast Cone",
            Self::BlackScholesGreeks => "Black-Scholes Greeks",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BsmPlotMetric {
    Prices,
    Delta,
    Gamma,
    Vega,
    Theta,
    Rho,
}

impl BsmPlotMetric {
    pub const ALL: [Self; 6] = [
        Self::Prices,
        Self::Delta,
        Self::Gamma,
        Self::Vega,
        Self::Theta,
        Self::Rho,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Prices => "Price ($)",
            Self::Delta => "Delta (∂V/∂S)",
            Self::Gamma => "Gamma (∂²V/∂S²)",
            Self::Vega => "Vega (∂V/∂σ)",
            Self::Theta => "Theta (∂V/∂t / yr)",
            Self::Rho => "Rho (∂V/∂r)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetConfig {
    pub label: String,
    pub mu: f64,
    pub sigma: f64,
    pub marginal_u: f64,
}

impl AssetConfig {
    pub fn benchmark_leg(index: usize) -> Self {
        Self {
            label: format!("Asset {}", index + 1),
            mu: 0.12,
            sigma: 0.20,
            marginal_u: 0.55,
        }
    }

    pub fn has_finite_fields(&self) -> bool {
        self.mu.is_finite() && self.sigma.is_finite() && self.marginal_u.is_finite()
    }

    pub fn clamp_in_place(&mut self, idx: usize) {
        if self.label.trim().is_empty() || self.label.len() > 32 {
            self.label = format!("Asset {}", idx + 1);
        }
        self.mu = if self.mu.is_finite() {
            self.mu.clamp(-0.95, 2.50)
        } else {
            0.12
        };
        self.sigma = if self.sigma.is_finite() {
            self.sigma.clamp(0.01, 2.50)
        } else {
            0.20
        };
        self.marginal_u = if self.marginal_u.is_finite() {
            self.marginal_u.clamp(0.001, 0.999)
        } else {
            0.55
        };
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PortfolioWorkbenchState {
    pub view: WorkbenchView,
    pub assets: Vec<AssetConfig>,
    pub rho: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corr_upper: Vec<f64>,
    pub risk_free_rate: f64,
    pub bankroll: f64,
    pub horizon: f64,
    pub leverage_cap: f64,
    pub copula_legs: usize,
    pub crash_quantile: f64,
    pub ou_mu: f64,
    pub ou_theta: f64,
    pub ou_sigma: f64,
    pub ou_x0: f64,
    pub ou_dt: f64,
    pub ou_horizon: f64,
    pub ou_paths: usize,
    pub bsm_spot: f64,
    pub bsm_strike: f64,
    pub bsm_vol: f64,
    pub bsm_expiry: f64,
    pub bsm_dividend: f64,
    pub bsm_metric: BsmPlotMetric,
    pub seed: u64,
}

impl Default for PortfolioWorkbenchState {
    fn default() -> Self {
        Self {
            view: WorkbenchView::KellyFrontier,
            assets: vec![
                AssetConfig::benchmark_leg(0),
                AssetConfig::benchmark_leg(1),
                AssetConfig::benchmark_leg(2),
            ],
            rho: 0.60,
            corr_upper: Vec::new(),
            risk_free_rate: 0.04,
            bankroll: 1_000.0,
            horizon: 1.0,
            leverage_cap: 1.0,
            copula_legs: 3,
            crash_quantile: 0.05,
            ou_mu: 0.08,
            ou_theta: 2.20,
            ou_sigma: 0.14,
            ou_x0: 0.18,
            ou_dt: 0.02,
            ou_horizon: 2.0,
            ou_paths: 256,
            bsm_spot: 100.0,
            bsm_strike: 100.0,
            bsm_vol: 0.20,
            bsm_expiry: 1.0,
            bsm_dividend: 0.0,
            bsm_metric: BsmPlotMetric::Delta,
            seed: 42,
        }
    }
}

impl PortfolioWorkbenchState {
    pub fn min_equicorrelation_for(n: usize) -> f64 {
        if n <= 1 {
            -0.95
        } else {
            (-1.0 / ((n - 1) as f64) + 0.02).clamp(-0.95, 0.0)
        }
    }

    pub fn has_finite_parameters(&self) -> bool {
        let scalars_ok = [
            self.rho,
            self.risk_free_rate,
            self.bankroll,
            self.horizon,
            self.leverage_cap,
            self.crash_quantile,
            self.ou_mu,
            self.ou_theta,
            self.ou_sigma,
            self.ou_x0,
            self.ou_dt,
            self.ou_horizon,
            self.bsm_spot,
            self.bsm_strike,
            self.bsm_vol,
            self.bsm_expiry,
            self.bsm_dividend,
        ]
        .iter()
        .all(|v| v.is_finite());

        scalars_ok
            && !self.assets.is_empty()
            && self.assets.len() <= MAX_WORKBENCH_ASSETS
            && self.assets.iter().all(AssetConfig::has_finite_fields)
            && self.corr_upper.iter().all(|v| v.is_finite())
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.has_finite_parameters() {
            return Err("Workbench parameters must be finite with 1..=12 assets".into());
        }
        let n = self.assets.len();
        let expected_upper = n * (n - 1) / 2;
        if !self.corr_upper.is_empty() && self.corr_upper.len() != expected_upper {
            return Err("Custom correlation upper-triangle length mismatch".into());
        }
        let min_rho = Self::min_equicorrelation_for(n.max(self.copula_legs));
        if self.corr_upper.is_empty() && !(min_rho..=0.98).contains(&self.rho) {
            return Err(format!(
                "Equicorrelation rho must be in [{min_rho:.3}, 0.98]"
            ));
        }
        if self.assets.iter().any(|a| {
            !(-0.95..=2.50).contains(&a.mu)
                || !(0.01..=2.50).contains(&a.sigma)
                || !(0.001..=0.999).contains(&a.marginal_u)
                || a.label.is_empty()
                || a.label.len() > 32
        }) {
            return Err("Asset parameters out of bounds".into());
        }
        self.validate_aux_ranges()
    }

    fn validate_aux_ranges(&self) -> Result<(), String> {
        if !(-0.20..=0.50).contains(&self.risk_free_rate)
            || !(1.0..=10_000_000.0).contains(&self.bankroll)
            || !(0.05..=25.0).contains(&self.horizon)
            || !(0.10..=10.0).contains(&self.leverage_cap)
            || !(2..=MAX_WORKBENCH_ASSETS).contains(&self.copula_legs)
            || !(0.001..=0.50).contains(&self.crash_quantile)
            || !(-5.0..=5.0).contains(&self.ou_mu)
            || !(0.05..=20.0).contains(&self.ou_theta)
            || !(0.01..=5.0).contains(&self.ou_sigma)
            || !(-5.0..=5.0).contains(&self.ou_x0)
            || !(0.002..=0.25).contains(&self.ou_dt)
            || !(0.10..=20.0).contains(&self.ou_horizon)
            || !(32..=2_000).contains(&self.ou_paths)
            || !(1.0..=10_000.0).contains(&self.bsm_spot)
            || !(1.0..=10_000.0).contains(&self.bsm_strike)
            || !(0.01..=3.0).contains(&self.bsm_vol)
            || !(0.02..=20.0).contains(&self.bsm_expiry)
            || !(0.0..=0.50).contains(&self.bsm_dividend)
        {
            return Err("Risk, OU, or Black-Scholes parameters out of bounds".into());
        }
        Ok(())
    }

    pub fn clamp_in_place(&mut self) {
        if self.assets.is_empty() {
            self.assets.push(AssetConfig::benchmark_leg(0));
        }
        self.assets.truncate(MAX_WORKBENCH_ASSETS);
        for (idx, asset) in self.assets.iter_mut().enumerate() {
            asset.clamp_in_place(idx);
        }
        self.copula_legs = self.copula_legs.clamp(2, MAX_WORKBENCH_ASSETS);
        let min_rho = Self::min_equicorrelation_for(self.assets.len().max(self.copula_legs));
        self.rho = if self.rho.is_finite() {
            self.rho.clamp(min_rho, 0.95)
        } else {
            0.60
        };
        let expected_upper = self.assets.len() * (self.assets.len() - 1) / 2;
        if self.corr_upper.len() != expected_upper || self.corr_upper.iter().any(|v| !v.is_finite())
        {
            self.corr_upper.clear();
        }
        self.clamp_scalars();
    }

    fn clamp_scalars(&mut self) {
        let clamp_or = |v: f64, lo: f64, hi: f64, def: f64| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                def
            }
        };
        self.risk_free_rate = clamp_or(self.risk_free_rate, -0.20, 0.50, 0.04);
        self.bankroll = clamp_or(self.bankroll, 1.0, 10_000_000.0, 1_000.0);
        self.horizon = clamp_or(self.horizon, 0.05, 25.0, 1.0);
        self.leverage_cap = clamp_or(self.leverage_cap, 0.10, 10.0, 1.0);
        self.crash_quantile = clamp_or(self.crash_quantile, 0.001, 0.50, 0.05);
        self.ou_mu = clamp_or(self.ou_mu, -5.0, 5.0, 0.08);
        self.ou_theta = clamp_or(self.ou_theta, 0.05, 20.0, 2.20);
        self.ou_sigma = clamp_or(self.ou_sigma, 0.01, 5.0, 0.14);
        self.ou_x0 = clamp_or(self.ou_x0, -5.0, 5.0, 0.18);
        self.ou_dt = clamp_or(self.ou_dt, 0.002, 0.25, 0.02);
        self.ou_horizon = clamp_or(self.ou_horizon, 0.10, 20.0, 2.0);
        self.ou_paths = self.ou_paths.clamp(32, 2_000);
        self.bsm_spot = clamp_or(self.bsm_spot, 1.0, 10_000.0, 100.0);
        self.bsm_strike = clamp_or(self.bsm_strike, 1.0, 10_000.0, 100.0);
        self.bsm_vol = clamp_or(self.bsm_vol, 0.01, 3.0, 0.20);
        self.bsm_expiry = clamp_or(self.bsm_expiry, 0.02, 20.0, 1.0);
        self.bsm_dividend = clamp_or(self.bsm_dividend, 0.0, 0.50, 0.0);
    }

    pub fn build_portfolio_matrices(&self) -> Result<(CorrelationMatrix, DMatrix<f64>), String> {
        let n = self.assets.len();
        let corr = if !self.corr_upper.is_empty() && self.corr_upper.len() == n * (n - 1) / 2 {
            let mut mat = DMatrix::identity(n, n);
            let mut k = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let c = self.corr_upper[k];
                    mat[(i, j)] = c;
                    mat[(j, i)] = c;
                    k += 1;
                }
            }
            CorrelationMatrix::new(mat).map_err(|e| format!("Invalid correlation matrix: {e}"))?
        } else if n == 1 {
            CorrelationMatrix::new(DMatrix::identity(1, 1))
                .map_err(|e| format!("Correlation error: {e}"))?
        } else {
            let c = Correlation::new(self.rho).map_err(|e| format!("Invalid rho: {e}"))?;
            CorrelationMatrix::equicorrelated(n, c)
                .map_err(|e| format!("Correlation matrix not positive definite: {e}"))?
        };

        let mut cov = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let r_ij = corr.matrix()[(i, j)];
                cov[(i, j)] = r_ij * self.assets[i].sigma * self.assets[j].sigma;
            }
        }
        Ok((corr, cov))
    }
}

// [cite:clinical_trials_statistics]
