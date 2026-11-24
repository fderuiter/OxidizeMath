//! Cooperative sensitivity hooks, dataset exports, parameter clamping, and CSV calibration for [`super::BankrollGrowthTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::simulation::finite_binary_growth_rate;
use super::state::is_default_stake_multiplier;
use super::BankrollGrowthTool;
use crate::framework::{
    extract_calibration_series, fit_bounded_parameters, CalibratableParamSpec, SensitivityParamSpec,
};
use math_explorer::pure_math::statistics::kelly::{kelly_fraction, Odds, UnitInterval};

impl BankrollGrowthTool {
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
        if !self.risk_constrained_points.is_empty() {
            out.push((
                "Risk-Constrained Kelly".to_string(),
                map_pts(&self.risk_constrained_points),
            ));
        }
        if !is_default_stake_multiplier(&self.stake_multiplier) {
            out.push((
                "Active Stake".to_string(),
                map_pts(&self.active_stake_points),
            ));
        }
        out
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

    pub(super) fn build_sensitivity_specs(&self) -> Vec<SensitivityParamSpec> {
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
        let g = finite_binary_growth_rate(&p, &o, active_f);
        g.is_finite().then_some(("Expected Log-Growth Rate (g)", g))
    }

    pub(super) fn calibrate_from_csv(
        &mut self,
        datasets: &[egui_plot::ImportedDataset],
    ) -> Result<bool, String> {
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
            let g = finite_binary_growth_rate(&p, &o, active_f);
            Some(w0 * (g * bet_idx.max(0.0)).exp())
        })?;
        self.initial_bankroll = w0;
        self.probability = fitted[0].clamp(0.05, 0.95);
        self.recalculate();
        Ok(true)
    }
}
