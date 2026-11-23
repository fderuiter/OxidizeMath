//! Persistence, theory metadata, and starter preset registration for [`super::BankrollGrowthTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::bankroll_ensemble::{
    BankrollLabMode, VolatilityHypothesis, DEFAULT_CUSTOM_FRACTION, MAX_CUSTOM_FRACTION,
};
use super::{is_default_stake_multiplier, BankrollGrowthTool};
use math_explorer::pure_math::statistics::kelly::{kelly_fraction, variants, Odds, UnitInterval};

const fn default_stake_multiplier() -> f64 {
    1.0
}

fn default_custom_fraction() -> f64 {
    DEFAULT_CUSTOM_FRACTION
}

fn is_default_mode(mode: &BankrollLabMode) -> bool {
    *mode == BankrollLabMode::KellyStrategies
}

fn is_default_fraction(val: &f64) -> bool {
    (*val - DEFAULT_CUSTOM_FRACTION).abs() < 1e-12
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct BankrollState {
    initial_bankroll: f64,
    probability: f64,
    odds: f64,
    num_bets: usize,
    #[serde(
        default = "default_stake_multiplier",
        skip_serializing_if = "is_default_stake_multiplier"
    )]
    stake_multiplier: f64,
    #[serde(default, skip_serializing_if = "is_default_mode")]
    mode: BankrollLabMode,
    #[serde(
        default = "default_custom_fraction",
        skip_serializing_if = "is_default_fraction"
    )]
    custom_fraction: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    hypothesis: Option<VolatilityHypothesis>,
}

impl BankrollGrowthTool {
    pub(super) fn save_state_json(&self) -> Option<String> {
        serde_json::to_string(&BankrollState {
            initial_bankroll: self.initial_bankroll,
            probability: self.probability,
            odds: self.odds,
            num_bets: self.num_bets,
            stake_multiplier: self.stake_multiplier,
            mode: self.mode,
            custom_fraction: self.ensemble_lab.custom_fraction,
            hypothesis: self.ensemble_lab.hypothesis,
        })
        .ok()
    }

    pub(super) fn load_state_json(&mut self, state: &str) -> Result<(), String> {
        let s: BankrollState =
            serde_json::from_str(state).map_err(|e| format!("Invalid bankroll state: {e}"))?;
        if ![
            s.initial_bankroll,
            s.probability,
            s.odds,
            s.stake_multiplier,
            s.custom_fraction,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !(100.0..=10_000.0).contains(&s.initial_bankroll)
            || !(0.01..=0.99).contains(&s.probability)
            || !(0.1..=10.0).contains(&s.odds)
            || !(10..=1000).contains(&s.num_bets)
            || !(0.0..=3.0).contains(&s.stake_multiplier)
            || !(0.0..=MAX_CUSTOM_FRACTION).contains(&s.custom_fraction)
        {
            return Err("Bankroll simulation parameters out of bounds".into());
        }
        let p = UnitInterval::new(s.probability).map_err(|e| format!("Error: {e}"))?;
        let o = Odds::from_net_odds(s.odds).map_err(|e| format!("Error: {e}"))?;
        if s.mode == BankrollLabMode::KellyStrategies {
            kelly_fraction(&p, &o).map_err(|e| format!("Error: {e}"))?;
            variants::half_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
            variants::quarter_kelly(&p, &o).map_err(|e| format!("Error: {e}"))?;
        }
        self.initial_bankroll = s.initial_bankroll;
        self.probability = s.probability;
        self.odds = s.odds;
        self.num_bets = s.num_bets;
        self.stake_multiplier = s.stake_multiplier;
        self.seed_counter = 0;
        self.mode = s.mode;
        self.ensemble_lab.custom_fraction = s.custom_fraction.clamp(0.0, MAX_CUSTOM_FRACTION);
        self.ensemble_lab.hypothesis = s.hypothesis;
        self.recalculate();
        Ok(())
    }

    pub(super) fn export_kelly_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let map_pts = |pts: &[egui_plot::PlotPoint]| -> Vec<[f64; 2]> {
            pts.iter().map(|p| [p.x, p.y]).collect()
        };
        let mut out = vec![
            ("Full Kelly".to_string(), map_pts(&self.full_kelly_points)),
            ("Half Kelly".to_string(), map_pts(&self.half_kelly_points)),
            (
                "Quarter Kelly".to_string(),
                map_pts(&self.quarter_kelly_points),
            ),
        ];
        if !is_default_stake_multiplier(&self.stake_multiplier) {
            out.push((
                "Active Stake".to_string(),
                map_pts(&self.active_stake_points),
            ));
        }
        out
    }

    pub(super) fn check_finite_params(&self) -> bool {
        [
            self.initial_bankroll,
            self.probability,
            self.odds,
            self.stake_multiplier,
            self.ensemble_lab.custom_fraction,
        ]
        .iter()
        .all(|v| v.is_finite())
            && self.error_msg.is_none()
    }

    pub(super) fn clamp_bankroll_params(&mut self) -> bool {
        self.initial_bankroll = if self.initial_bankroll.is_finite() {
            self.initial_bankroll.clamp(100.0, 10_000.0)
        } else {
            1000.0
        };
        self.probability = if self.probability.is_finite() {
            self.probability.clamp(0.01, 0.99)
        } else {
            0.55
        };
        self.odds = if self.odds.is_finite() {
            self.odds.clamp(0.1, 10.0)
        } else {
            2.0
        };
        self.stake_multiplier = if self.stake_multiplier.is_finite() {
            self.stake_multiplier.clamp(0.0, 3.0)
        } else {
            1.0
        };
        self.num_bets = self.num_bets.clamp(10, 1000);
        self.recalculate();
        true
    }

    pub(super) fn build_sensitivity_specs(&self) -> Vec<crate::framework::SensitivityParamSpec> {
        use crate::framework::SensitivityParamSpec;
        vec![
            SensitivityParamSpec::new(
                "probability",
                "Win Probability (p)",
                self.probability,
                0.05,
                0.95,
            ),
            SensitivityParamSpec::new("odds", "Net Odds (b)", self.odds, 0.2, 10.0),
            SensitivityParamSpec::new(
                "stake_multiplier",
                "Stake Multiplier (× f*)",
                self.stake_multiplier,
                0.05,
                2.5,
            ),
        ]
    }

    pub(super) fn eval_sensitivity_from_pairs(
        &self,
        params: &[(&str, f64)],
    ) -> Option<(&'static str, f64)> {
        let mut prob = self.probability;
        let mut odds = self.odds;
        let mut mult = self.stake_multiplier;
        for &(k, v) in params {
            if !v.is_finite() {
                return None;
            }
            match k {
                "probability" => prob = v.clamp(0.01, 0.99),
                "odds" => odds = v.clamp(0.1, 10.0),
                "stake_multiplier" => mult = v.clamp(0.0, 3.0),
                _ => {}
            }
        }
        let p = UnitInterval::new(prob).ok()?;
        let o = Odds::from_net_odds(odds).ok()?;
        let fk = kelly_fraction(&p, &o).ok()?;
        let active_f = (fk.value() * mult).clamp(0.0, 0.9999);
        let g = super::finite_kelly_growth_rate(&p, &o, active_f);
        g.is_finite().then_some(("Expected Log-Growth Rate (g)", g))
    }

    pub(super) fn calibrate_from_csv(
        &mut self,
        datasets: &[egui_plot::ImportedDataset],
    ) -> Result<bool, String> {
        use crate::framework::{
            extract_calibration_series, fit_bounded_parameters, CalibratableParamSpec,
        };
        let points = extract_calibration_series(datasets)?;
        let w0 = points
            .first()
            .filter(|pt| pt[0].abs() < 1.0e-6 && (100.0..=10_000.0).contains(&pt[1]))
            .map_or(self.initial_bankroll, |pt| pt[1]);
        let odds = self.odds.clamp(0.1, 10.0);
        let mult = self.stake_multiplier.clamp(0.0, 3.0);
        let specs = [CalibratableParamSpec::new("probability", 0.52, 0.05, 0.95)];
        let fitted = fit_bounded_parameters(points, &specs, |bet_idx, theta| {
            let p = UnitInterval::new(theta[0]).ok()?;
            let o = Odds::from_net_odds(odds).ok()?;
            let fk = kelly_fraction(&p, &o).ok()?;
            let active_f = (fk.value() * mult).clamp(0.0, 0.9999);
            let g = super::finite_kelly_growth_rate(&p, &o, active_f);
            Some(w0 * (g * bet_idx.max(0.0)).exp())
        })?;
        self.initial_bankroll = w0;
        self.probability = fitted[0].clamp(0.05, 0.95);
        self.recalculate();
        Ok(true)
    }
}

impl scientific_metadata::theory::TheoryDescribable for BankrollGrowthTool {
    fn theory_description(&self) -> String {
        let b = self.odds.max(1e-6);
        let q = 1.0 - self.probability;
        let f_kelly = ((b * self.probability - q) / b).clamp(-10.0, 1.0);
        format!(
            "Capital compounding trajectory under Kelly criterion betting: $W_T = W_0 \\prod_{{t=1}}^T (1 + f R_t)$ with initial capital $W_0 = {:.1}$, win probability $p = {:.2}$, net odds $b = {:.2}$, and optimal fraction $f^* = {:.3}$.",
            self.initial_bankroll, self.probability, self.odds, f_kelly
        )
    }
    fn phonetic_description(&self) -> String {
        "Kelly criterion bankroll growth and geometric compounding theory".into()
    }
    fn theory_citation(&self) -> String {
        "A New Interpretation of Information Rate (Kelly, 1956); The Kelly Capital Growth Investment Criterion (MacLean et al., 2011)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Full Kelly".into(),
            "Maximizes expected logarithmic growth rate".into(),
        );
        map.insert(
            "Half Kelly".into(),
            "Reduces variance and drawdowns by half while retaining 75% of growth rate".into(),
        );
        map.insert(
            "Quarter Kelly".into(),
            "Conservative wagering for high volatility protection".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Geometric Bankroll Growth",
    "domain": "financial_math",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "initial_bankroll": 5000.0,
    "probability": 0.6,
    "odds": 2.5,
    "num_bets": 150
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bankroll_growth_starter",
        title: "Geometric Bankroll Growth",
        description: "Log-wealth expectation maximizing capital trajectory under continuous compound yield.",
        category: "Portfolio Theory",
        icon: "💰",
        domain: "financial_math",
        tool_name: Some("BankrollGrowthTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:clinical_trials_statistics]
