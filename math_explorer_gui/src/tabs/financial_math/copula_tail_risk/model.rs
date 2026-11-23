//! Copula SGP pricing, Student-t tail comparison, 2D regime sweep, and Kelly sizing for `CopulaTailRiskLab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    CopulaTailRiskConfig, CopulaTailRiskLab, CopulaTailRiskMetrics, COPULA_GRID_DIM,
    SGP_CURVE_POINTS,
};
use crate::async_sim::atlas::atlas_scalar_color;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::statistics::copula::{
    sgp_joint_probability, Correlation, CorrelationMatrix, GaussianCopula, Probability,
    StudentTCopula,
};
use math_explorer::pure_math::statistics::kelly::{
    expected_growth_rate, kelly_fraction, BankrollFraction, Odds, UnitInterval,
};

impl CopulaTailRiskLab {
    /// Clamps configuration values to valid domain bounds and active challenge locks.
    pub fn clamp_config_in_place(&mut self) {
        let c = &mut self.config;
        c.leg1_prob = sanitize(c.leg1_prob, 0.20, 0.02, 0.95);
        c.leg2_prob = sanitize(c.leg2_prob, 0.20, 0.02, 0.95);
        c.correlation_rho = sanitize(c.correlation_rho, 0.50, -0.95, 0.95);
        c.tail_quantile_u = sanitize(c.tail_quantile_u, 0.999, 0.85, 0.9995);
        c.degrees_of_freedom = sanitize(c.degrees_of_freedom, 3.0, 1.5, 30.0);
        c.offered_decimal_odds = sanitize(c.offered_decimal_odds, 18.0, 1.2, 200.0);
        c.stake_multiplier = sanitize(c.stake_multiplier, 1.0, 0.10, 2.50);
        if self.active_challenge.is_some() {
            self.enforce_challenge_locks();
        }
    }

    /// Resets configuration to the canonical SGP + `5.83x` tail-crash preset.
    pub fn reset_starter_preset(&mut self) {
        self.config = CopulaTailRiskConfig::default();
        if self.active_challenge.is_some() {
            self.enforce_challenge_locks();
        }
        self.recalculate();
    }

    /// Recomputes `GaussianCopula`, `sgp_joint_probability`, `StudentTCopula`, Kelly sizing, and the 2D heatmap.
    pub fn recalculate(&mut self) {
        self.clamp_config_in_place();
        let c = &self.config;
        let p1 = Probability::new(c.leg1_prob).unwrap_or_else(|_| Probability::new(0.20).unwrap());
        let p2 = Probability::new(c.leg2_prob).unwrap_or_else(|_| Probability::new(0.20).unwrap());
        let rho =
            Correlation::new(c.correlation_rho).unwrap_or_else(|_| Correlation::new(0.50).unwrap());

        let gauss_copula = GaussianCopula::bivariate(rho).unwrap();
        let corr_mat = CorrelationMatrix::bivariate(rho).unwrap();
        let sgp_prob = sgp_joint_probability(&[p1, p2], &corr_mat)
            .or_else(|_| gauss_copula.cdf(&[p1, p2]))
            .map(|p| p.value())
            .unwrap_or(c.leg1_prob * c.leg2_prob);

        let neg50_mat = CorrelationMatrix::bivariate(Correlation::new(-0.50).unwrap()).unwrap();
        let pos70_mat = CorrelationMatrix::bivariate(Correlation::new(0.70).unwrap()).unwrap();
        let sgp_neg50 = sgp_joint_probability(&[p1, p2], &neg50_mat)
            .map(|p| p.value())
            .unwrap_or(0.0);
        let sgp_pos70 = sgp_joint_probability(&[p1, p2], &pos70_mat)
            .map(|p| p.value())
            .unwrap_or(0.0);

        let t_copula = StudentTCopula::new(rho, c.degrees_of_freedom).unwrap();
        let student_joint = t_copula.cdf(p1, p2).map(|p| p.value()).unwrap_or(sgp_prob);

        let tail_cmp = self.scratch.evaluate_tail_comparison(
            c.leg1_prob,
            c.leg2_prob,
            c.tail_quantile_u,
            c.correlation_rho,
            c.degrees_of_freedom,
        );

        let naive_prob = c.leg1_prob * c.leg2_prob;
        let (naive_f, copula_f, applied_f, true_g, naive_true_g) = compute_kelly_metrics(
            naive_prob,
            student_joint,
            c.offered_decimal_odds,
            c.stake_multiplier,
        );

        self.metrics = CopulaTailRiskMetrics {
            naive_independent_prob: naive_prob,
            sgp_gaussian_prob: sgp_prob,
            student_t_joint_prob: student_joint,
            sgp_neg50_prob: sgp_neg50,
            sgp_pos70_prob: sgp_pos70,
            gaussian_cond_tail: tail_cmp.gaussian_upper_tail,
            student_t_cond_tail: tail_cmp.student_t_upper_tail,
            tail_multiplier: tail_cmp.tail_multiplier,
            asymptotic_lambda_u: t_copula.tail_dependence_coefficient(),
            naive_kelly_fraction: naive_f,
            copula_kelly_fraction: copula_f,
            applied_stake: applied_f,
            true_expected_growth: true_g,
            naive_stake_true_growth: naive_true_g,
            is_overbet_ruin: (applied_f > 0.0 && true_g < 0.0)
                || (naive_f > 0.0 && naive_true_g < 0.0),
        };

        self.rebuild_sgp_curves();
        self.populate_regime_heatmap(false);
        self.sync_callouts();
    }

    fn rebuild_sgp_curves(&mut self) {
        let u1 = self.config.leg1_prob;
        let u2 = self.config.leg2_prob;
        let nu = self.config.degrees_of_freedom;
        let naive = u1 * u2;
        self.curve_naive.clear();
        self.curve_gaussian.clear();
        self.curve_student_t.clear();
        for k in 0..SGP_CURVE_POINTS {
            let r = -0.90 + 1.80 * (k as f64) / ((SGP_CURVE_POINTS - 1) as f64);
            let g = self.scratch.evaluate_gaussian_joint(u1, u2, r);
            let t = self.scratch.evaluate_student_t_joint(u1, u2, r, nu);
            self.curve_naive.push([r, naive]);
            self.curve_gaussian.push([r, g]);
            self.curve_student_t.push([r, t]);
        }
    }

    /// Populates the 2D `(rho, u)` tail-multiplier heatmap in place with zero heap allocations.
    pub fn populate_regime_heatmap(&mut self, high_contrast: bool) {
        self.scratch.populate_tail_regime_grid(
            &self.rho_axis,
            &self.quantile_axis,
            self.config.degrees_of_freedom,
        );
        for row in 0..COPULA_GRID_DIM {
            let data_row = COPULA_GRID_DIM - 1 - row;
            for col in 0..COPULA_GRID_DIM {
                let mult = self.scratch.multiplier_at(col, data_row);
                // Log-compress multiplier [0.5x .. 10x] into [0, 1] for perceptual contrast.
                let norm = (mult.max(0.5).ln() - 0.5_f64.ln()) / (10.0_f64.ln() - 0.5_f64.ln());
                let px_idx = flatten_2d_index(col, row, COPULA_GRID_DIM);
                self.regime_pixels[px_idx] = atlas_scalar_color(norm, high_contrast);
            }
        }
        self.texture_dirty = true;
    }

    fn sync_callouts(&mut self) {
        self.callout_manager.clear_auto();
        let m = &self.metrics;
        let text = if m.is_overbet_ruin {
            format!(
                "Overbet / Mispricing Trap! {:.2}x Tail Crash (g = {:.4})",
                m.tail_multiplier, m.true_expected_growth
            )
        } else {
            format!(
                "Student-t {:.2}x vs Gaussian @ u={:.3}",
                m.tail_multiplier, self.config.tail_quantile_u
            )
        };
        self.callout_manager.set_auto_callout(
            "copula_tail_cursor",
            [self.config.correlation_rho, self.config.tail_quantile_u],
            text,
            [18.0, -20.0],
        );
    }
}

fn compute_kelly_metrics(
    naive_prob: f64,
    true_prob: f64,
    decimal_odds: f64,
    stake_multiplier: f64,
) -> (f64, f64, f64, f64, f64) {
    let Ok(odds) = Odds::new(decimal_odds.max(1.01)) else {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    };
    let naive_p = UnitInterval::new(naive_prob.clamp(1e-6, 0.999_999)).unwrap();
    let true_p = UnitInterval::new(true_prob.clamp(1e-6, 0.999_999)).unwrap();

    let naive_f = kelly_fraction(&naive_p, &odds)
        .map(|f| f.value())
        .unwrap_or(0.0);
    let copula_f = kelly_fraction(&true_p, &odds)
        .map(|f| f.value())
        .unwrap_or(0.0);

    let base_f = if copula_f > 0.0 {
        (copula_f * stake_multiplier).clamp(0.0, 0.95)
    } else {
        // Expose overbetting penalty when user forces stake into negative-EV correlated regime
        (0.04 * stake_multiplier).clamp(0.0, 0.95)
    };

    let true_g = safe_growth(&true_p, &odds, base_f);
    let naive_true_g = safe_growth(&true_p, &odds, naive_f);
    (naive_f, copula_f, base_f, true_g, naive_true_g)
}

fn safe_growth(p: &UnitInterval, odds: &Odds, f: f64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    if let Ok(bf) = BankrollFraction::new(f.clamp(0.0, 0.9999)) {
        let g = expected_growth_rate(p, odds, &bf);
        if g.is_finite() {
            return g;
        }
    }
    -10.0
}

fn sanitize(val: f64, fallback: f64, min: f64, max: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

// [cite:clinical_trials_statistics]
