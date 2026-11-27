//! State serialization, theory metadata, observables, and starter preset for [`super::TailRiskDrawdownTool`].

use super::{TailRiskDrawdownTool, MAX_TAIL_HORIZON};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TailRiskState {
    annual_volatility: f64,
    annual_drift: f64,
    tail_shock_prob: f64,
    tail_shock_scale: f64,
    confidence_level: f64,
    risk_free_rate: f64,
    horizon_periods: usize,
    seed: u64,
}

impl TailRiskDrawdownTool {
    pub(super) fn collect_observables(&self) -> Vec<(&'static str, f64)> {
        if let Some(s) = self.summary {
            vec![
                ("historical_var", s.historical_var),
                ("historical_cvar", s.historical_cvar),
                ("gaussian_var", s.gaussian_var),
                ("cornish_fisher_var", s.cornish_fisher_var),
                ("sharpe_ratio", s.sharpe),
                ("sortino_ratio", s.sortino),
                ("calmar_ratio", s.calmar),
                ("max_drawdown", s.max_drawdown),
                ("max_underwater_duration", s.max_underwater_duration as f64),
            ]
        } else {
            Vec::new()
        }
    }

    pub(super) fn save_state_json(&self) -> Option<String> {
        serde_json::to_string(&TailRiskState {
            annual_volatility: self.annual_volatility,
            annual_drift: self.annual_drift,
            tail_shock_prob: self.tail_shock_prob,
            tail_shock_scale: self.tail_shock_scale,
            confidence_level: self.confidence_level,
            risk_free_rate: self.risk_free_rate,
            horizon_periods: self.horizon_periods,
            seed: self.seed,
        })
        .ok()
    }

    pub(super) fn load_state_json(&mut self, state: &str) -> Result<(), String> {
        let s: TailRiskState =
            serde_json::from_str(state).map_err(|e| format!("Invalid tail-risk state: {e}"))?;
        let scalars = [
            s.annual_volatility,
            s.annual_drift,
            s.tail_shock_prob,
            s.tail_shock_scale,
            s.confidence_level,
            s.risk_free_rate,
        ];
        if !scalars.iter().all(|v| v.is_finite())
            || !(0.02..=1.20).contains(&s.annual_volatility)
            || !(-0.50..=0.80).contains(&s.annual_drift)
            || !(0.0..=0.35).contains(&s.tail_shock_prob)
            || !(0.0..=0.45).contains(&s.tail_shock_scale)
            || !(0.55..=0.995).contains(&s.confidence_level)
            || !(0.0..=0.20).contains(&s.risk_free_rate)
            || !(30..=MAX_TAIL_HORIZON).contains(&s.horizon_periods)
        {
            return Err("Tail-risk parameters out of valid simulation bounds".into());
        }
        self.annual_volatility = s.annual_volatility;
        self.annual_drift = s.annual_drift;
        self.tail_shock_prob = s.tail_shock_prob;
        self.tail_shock_scale = s.tail_shock_scale;
        self.confidence_level = s.confidence_level;
        self.risk_free_rate = s.risk_free_rate;
        self.horizon_periods = s.horizon_periods;
        self.seed = s.seed;
        self.recompute();
        Ok(())
    }
}

impl scientific_metadata::theory::TheoryDescribable for TailRiskDrawdownTool {
    fn theory_description(&self) -> String {
        format!(
            "Coherent Expected Shortfall ($\\text{{CVaR}}_\\alpha$) vs Value-at-Risk ($\\text{{VaR}}_\\alpha$) at $\\alpha = {:.1}\\%$: compares empirical Acerbi-Tasche tail averages, Gaussian quantiles, and 4th-order Cornish-Fisher skewness/kurtosis expansions alongside Sharpe, Sortino, and Calmar drawdown ratios.",
            self.confidence_level * 100.0
        )
    }

    fn phonetic_description(&self) -> String {
        "Value at Risk, Conditional Value at Risk, Cornish Fisher tail expansion, and underwater drawdown analysis".into()
    }

    fn theory_citation(&self) -> String {
        "On the Coherence of Expected Shortfall (Acerbi & Tasche, 2002); Optimization of Conditional Value-at-Risk (Rockafellar & Uryasev, 2000)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Fat-Tailed Drawdown & Expected Shortfall",
    "domain": "financial_math",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "annual_volatility": 0.24,
    "annual_drift": 0.11,
    "tail_shock_prob": 0.08,
    "tail_shock_scale": 0.11,
    "confidence_level": 0.95,
    "risk_free_rate": 0.03,
    "horizon_periods": 252,
    "seed": 7
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "tail_risk_starter",
        title: "Fat-Tailed Drawdown & Expected Shortfall",
        description: "Compare Historical, Gaussian, and Cornish-Fisher VaR/CVaR under negatively skewed downside shocks and inspect underwater recovery.",
        category: "Risk Management",
        icon: "📉",
        domain: "financial_math",
        tool_name: Some("TailRiskDrawdownTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]
