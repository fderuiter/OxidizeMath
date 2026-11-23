//! State persistence, universe mutators, theory metadata, and starter preset for [`super::MarkowitzFrontierTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MarkowitzFrontierTool;
use math_explorer::applied::portfolio::{PortfolioSpec, MAX_PORTFOLIO_ASSETS};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MarkowitzFrontierState {
    risk_free_rate: f64,
    max_asset_weight: f64,
    asset_names: Vec<String>,
    expected_returns: Vec<f64>,
    volatilities: Vec<f64>,
    upper_correlations: Vec<f64>,
}

impl MarkowitzFrontierTool {
    pub(super) fn collect_observables(&self) -> Vec<(&'static str, f64)> {
        let mut obs = vec![("asset_count", self.expected_returns.len() as f64)];
        if let Some(gmv) = self.con_gmv {
            obs.push(("gmv_volatility", gmv.volatility));
            obs.push(("gmv_expected_return", gmv.expected_return));
        }
        if let Some(tan) = self.con_tangency {
            obs.push(("tangency_volatility", tan.volatility));
            obs.push(("tangency_expected_return", tan.expected_return));
            obs.push(("tangency_sharpe", tan.sharpe_ratio));
        }
        if let Some(unc) = self.unc_tangency {
            obs.push(("unconstrained_tangency_sharpe", unc.sharpe_ratio));
        }
        obs
    }

    pub(super) fn save_state_json(&self) -> Option<String> {
        serde_json::to_string(&MarkowitzFrontierState {
            risk_free_rate: self.risk_free_rate,
            max_asset_weight: self.max_asset_weight,
            asset_names: self.asset_names.clone(),
            expected_returns: self.expected_returns.clone(),
            volatilities: self.volatilities.clone(),
            upper_correlations: self.upper_correlations.clone(),
        })
        .ok()
    }

    pub(super) fn load_state_json(&mut self, state: &str) -> Result<(), String> {
        let s: MarkowitzFrontierState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Markowitz state: {e}"))?;
        let n = s.expected_returns.len();
        if !(2..=MAX_PORTFOLIO_ASSETS).contains(&n)
            || s.asset_names.len() != n
            || s.volatilities.len() != n
            || s.upper_correlations.len() != n * (n - 1) / 2
            || s.asset_names.iter().any(|name| name.trim().is_empty())
        {
            return Err("Portfolio asset arrays have invalid or mismatched lengths".into());
        }
        if !s.risk_free_rate.is_finite()
            || !s.max_asset_weight.is_finite()
            || !(0.0..=0.25).contains(&s.risk_free_rate)
            || !(0.05..=1.0).contains(&s.max_asset_weight)
            || (s.max_asset_weight * (n as f64)) < 1.0 - 1e-9
            || s.expected_returns
                .iter()
                .any(|r| !r.is_finite() || !(-0.50..=1.00).contains(r))
            || s.volatilities
                .iter()
                .any(|v| !v.is_finite() || !(0.005..=1.50).contains(v))
        {
            return Err("Portfolio parameters out of valid financial bounds".into());
        }
        PortfolioSpec::from_upper_correlations(
            &s.expected_returns,
            &s.volatilities,
            &s.upper_correlations,
        )
        .map_err(|e| format!("Invalid covariance matrix: {e}"))?;

        self.risk_free_rate = s.risk_free_rate;
        self.max_asset_weight = s.max_asset_weight;
        self.asset_names = s.asset_names;
        self.expected_returns = s.expected_returns;
        self.volatilities = s.volatilities;
        self.upper_correlations = s.upper_correlations;
        self.recompute();
        Ok(())
    }

    pub(super) fn append_asset(&mut self) {
        let n = self.expected_returns.len();
        if n >= MAX_PORTFOLIO_ASSETS {
            return;
        }
        let idx = n + 1;
        self.asset_names.push(format!("Asset {idx}"));
        self.expected_returns
            .push((0.055 + 0.008 * ((idx % 5) as f64)).min(0.25));
        self.volatilities
            .push((0.10 + 0.018 * ((idx % 6) as f64)).min(0.40));

        // Insert zero correlation column at index `n` across rows `0..n` so SPD is preserved!
        let mut expanded = Vec::with_capacity((n + 1) * n / 2);
        let mut cursor = 0usize;
        for r in 0..n {
            let row_len = n - 1 - r;
            expanded.extend_from_slice(&self.upper_correlations[cursor..cursor + row_len]);
            expanded.push(0.05);
            cursor += row_len;
        }
        self.upper_correlations = expanded;
        if PortfolioSpec::from_upper_correlations(
            &self.expected_returns,
            &self.volatilities,
            &self.upper_correlations,
        )
        .is_err()
        {
            self.regularize_correlations();
        }
    }

    pub(super) fn pop_asset(&mut self) {
        let n = self.expected_returns.len();
        if n <= 2 {
            return;
        }
        self.asset_names.pop();
        self.expected_returns.pop();
        self.volatilities.pop();
        let mut shrunk = Vec::with_capacity((n - 1) * (n - 2) / 2);
        let mut cursor = 0usize;
        for r in 0..(n - 1) {
            let old_row_len = n - 1 - r;
            shrunk.extend_from_slice(&self.upper_correlations[cursor..cursor + old_row_len - 1]);
            cursor += old_row_len;
        }
        self.upper_correlations = shrunk;
        let min_cap = (1.0 / ((n - 1) as f64) + 1e-4).min(1.0);
        if self.max_asset_weight < min_cap {
            self.max_asset_weight = min_cap;
        }
    }

    pub(super) fn regularize_correlations(&mut self) {
        for _ in 0..24 {
            if PortfolioSpec::from_upper_correlations(
                &self.expected_returns,
                &self.volatilities,
                &self.upper_correlations,
            )
            .is_ok()
            {
                return;
            }
            for rho in &mut self.upper_correlations {
                *rho *= 0.80;
            }
        }
        self.upper_correlations.fill(0.0);
    }

    pub(super) fn apply_eight_asset_endowment(&mut self) {
        self.risk_free_rate = 0.03;
        self.max_asset_weight = 0.35;
        self.asset_names = vec![
            "US Equities".into(),
            "Intl Equities".into(),
            "Emerging Mkts".into(),
            "Gov Bonds".into(),
            "IG Credit".into(),
            "TIPS".into(),
            "Commodities".into(),
            "Real Estate".into(),
        ];
        self.expected_returns = vec![0.105, 0.092, 0.115, 0.042, 0.054, 0.046, 0.070, 0.082];
        self.volatilities = vec![0.165, 0.175, 0.230, 0.060, 0.075, 0.065, 0.195, 0.145];
        self.upper_correlations = vec![
            0.22, 0.20, -0.15, 0.08, -0.05, 0.06, 0.18, // row 0 (7)
            0.24, -0.12, 0.07, -0.04, 0.08, 0.15, // row 1 (6)
            -0.10, 0.06, -0.02, 0.12, 0.14, // row 2 (5)
            0.25, 0.22, -0.08, -0.05, // row 3 (4)
            0.18, 0.02, 0.09, // row 4 (3)
            0.14, 0.06, // row 5 (2)
            0.10, // row 6 (1)
        ];
    }
}

impl scientific_metadata::theory::TheoryDescribable for MarkowitzFrontierTool {
    fn theory_description(&self) -> String {
        format!(
            "Markowitz mean-variance quadratic program $\\min_w \\frac{{1}}{{2}} w^T \\Sigma w$ subject to $\\mathbf{{1}}^T w = 1,\\; \\mu^T w = \\mu_p,\\; 0 \\le w_i \\le {:.2}$ across $n = {}$ correlated assets vs unconstrained Merton two-fund frontier.",
            self.max_asset_weight,
            self.expected_returns.len()
        )
    }

    fn phonetic_description(&self) -> String {
        "Markowitz mean variance efficient frontier, global minimum variance, and maximum Sharpe tangency allocation".into()
    }

    fn theory_citation(&self) -> String {
        "Portfolio Selection (Markowitz, 1952); An Analytic Derivation of the Efficient Portfolio Frontier (Merton, 1972)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Multi-Asset Efficient Frontier",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "risk_free_rate": 0.035,
    "max_asset_weight": 0.60,
    "asset_names": [
      "US Large Cap",
      "Intl Developed",
      "Sovereign Treasury",
      "Gold Bullion",
      "Real Estate"
    ],
    "expected_returns": [0.108, 0.092, 0.044, 0.062, 0.084],
    "volatilities": [0.175, 0.185, 0.058, 0.155, 0.148],
    "upper_correlations": [0.35, -0.20, 0.05, 0.25, -0.15, 0.10, 0.20, 0.12, -0.10, 0.08]
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "markowitz_frontier_starter",
        title: "Multi-Asset Efficient Frontier",
        description: "Compare unconstrained Lagrangian and weight-capped long-only active-set frontiers, GMV, and Maximum-Sharpe tangency weights.",
        category: "Portfolio Optimization",
        icon: "📊",
        domain: "financial_math",
        tool_name: Some("MarkowitzFrontierTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]
