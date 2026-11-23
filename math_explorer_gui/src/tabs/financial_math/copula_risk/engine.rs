//! Cholesky multivariate copula evaluation, sweep curves, and bivariate contour generation.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CopulaRiskState, CopulaRiskTool, CorrelationStructureMode, SweepPricingRow};
use egui_plot::PlotPoint;
use math_explorer::nalgebra::DMatrix;
use math_explorer::pure_math::statistics::copula::{
    sgp_joint_probability, Correlation, CorrelationMatrix, GaussianCopula, Probability,
    MAX_COPULA_DIMENSION,
};
use oxidize_core::rng::OxidizeRng;

pub(crate) fn decimal_to_american_odds(dec: f64) -> f64 {
    if dec >= 2.0 {
        (dec - 1.0) * 100.0
    } else if dec > 1.0001 {
        -100.0 / (dec - 1.0)
    } else {
        -10_000.0
    }
}

pub(crate) fn pair_flat_index(n: usize, i: usize, j: usize) -> usize {
    let (lo, hi) = if i < j { (i, j) } else { (j, i) };
    let mut idx = 0;
    for r in 0..lo {
        idx += n - 1 - r;
    }
    idx + (hi - lo - 1)
}

fn solve_bivariate_contour_u2(
    copula: &GaussianCopula,
    u1: Probability,
    target_c: f64,
) -> Option<f64> {
    if u1.value() <= target_c + 1e-4 {
        return None;
    }
    let mut lo = target_c.max(1e-4);
    let mut hi = 0.999;
    let p_lo = Probability::new(lo).ok()?;
    let p_hi = Probability::new(hi).ok()?;
    let f_lo = copula.cdf(&[u1, p_lo]).ok()?.value() - target_c;
    let f_hi = copula.cdf(&[u1, p_hi]).ok()?.value() - target_c;
    if f_lo > 0.0 || f_hi < 0.0 {
        return None;
    }
    for _ in 0..22 {
        let mid = 0.5 * (lo + hi);
        let p_mid = Probability::new(mid).ok()?;
        let val = copula.cdf(&[u1, p_mid]).ok()?.value();
        if val < target_c {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

impl CopulaRiskTool {
    pub(crate) fn expected_upper_len(n: usize) -> usize {
        n.saturating_mul(n.saturating_sub(1)) / 2
    }

    pub(crate) fn resize_legs(&mut self, new_legs: usize) {
        let n = new_legs.clamp(2, MAX_COPULA_DIMENSION);
        let old_n = self.num_legs;
        let old_upper = self.pairwise_upper.clone();
        self.num_legs = n;
        self.marginals.resize(n, 0.60);
        while self.leg_labels.len() < n {
            let k = self.leg_labels.len() + 1;
            self.leg_labels.push(format!("Leg {k}"));
        }
        self.leg_labels.truncate(n);

        let mut next_upper =
            vec![self.equicorrelation.clamp(-0.9, 0.9); Self::expected_upper_len(n)];
        for i in 0..n.min(old_n) {
            for j in (i + 1)..n.min(old_n) {
                let old_idx = pair_flat_index(old_n, i, j);
                let new_idx = pair_flat_index(n, i, j);
                if let Some(&val) = old_upper.get(old_idx) {
                    next_upper[new_idx] = val;
                }
            }
        }
        self.pairwise_upper = next_upper;
        if self.contour_leg_a >= n {
            self.contour_leg_a = 0;
        }
        if self.contour_leg_b >= n || self.contour_leg_b == self.contour_leg_a {
            self.contour_leg_b = (self.contour_leg_a + 1) % n;
        }
    }

    pub(crate) fn validate_state(s: &CopulaRiskState) -> Result<(), String> {
        if !(2..=MAX_COPULA_DIMENSION).contains(&s.num_legs) {
            return Err(format!("num_legs must be in 2..={MAX_COPULA_DIMENSION}"));
        }
        if s.marginals.len() != s.num_legs || s.leg_labels.len() != s.num_legs {
            return Err("marginals and leg_labels must match num_legs".into());
        }
        if s.pairwise_upper.len() != Self::expected_upper_len(s.num_legs) {
            return Err("pairwise_upper length mismatch for num_legs".into());
        }
        if !s.equicorrelation.is_finite()
            || !(-0.98..=0.98).contains(&s.equicorrelation)
            || !s.book_vig_pct.is_finite()
            || !(0.0..=25.0).contains(&s.book_vig_pct)
            || s.marginals
                .iter()
                .any(|&m| !m.is_finite() || !(0.01..=0.99).contains(&m))
            || s.pairwise_upper
                .iter()
                .any(|&r| !r.is_finite() || !(-0.98..=0.98).contains(&r))
        {
            return Err("Copula risk parameters out of valid bounds".into());
        }
        if s.contour_leg_a >= s.num_legs
            || s.contour_leg_b >= s.num_legs
            || s.contour_leg_a == s.contour_leg_b
            || s.leg_labels.iter().any(|l| l.len() > 64)
        {
            return Err("Invalid contour leg selection or label length".into());
        }
        Ok(())
    }

    pub(crate) fn apply_state(&mut self, s: CopulaRiskState) -> Result<(), String> {
        Self::validate_state(&s)?;
        self.equicorrelation = s.equicorrelation;
        self.book_vig_pct = s.book_vig_pct;
        self.num_legs = s.num_legs;
        self.marginals = s.marginals;
        self.leg_labels = s.leg_labels;
        self.mode = s.mode;
        self.pairwise_upper = s.pairwise_upper;
        self.contour_leg_a = s.contour_leg_a;
        self.contour_leg_b = s.contour_leg_b;
        self.plot_view = s.plot_view;
        self.rng_seed = s.rng_seed;
        self.bump_revision();
        self.recalculate();
        Ok(())
    }

    pub(crate) fn build_active_matrix(&self) -> Result<CorrelationMatrix, String> {
        match self.mode {
            CorrelationStructureMode::Equicorrelated => {
                let rho = Correlation::new(self.equicorrelation).map_err(|e| e.to_string())?;
                CorrelationMatrix::equicorrelated(self.num_legs, rho).map_err(|e| e.to_string())
            }
            CorrelationStructureMode::PairwiseMatrix => {
                let n = self.num_legs;
                let mat = DMatrix::from_fn(n, n, |i, j| {
                    if i == j {
                        1.0
                    } else {
                        self.pairwise_upper[pair_flat_index(n, i, j)]
                    }
                });
                CorrelationMatrix::new(mat)
                    .map_err(|e| format!("Correlation matrix is not positive definite ({e})"))
            }
        }
    }

    fn populate_sweep_and_sheet(&mut self, marginals: &[Probability]) {
        self.sweep_copula_points.clear();
        self.pricing_rows.clear();
        let min_rho = (-1.0 / (self.num_legs.saturating_sub(1) as f64) + 0.03).max(-0.95);
        let max_rho = 0.95;
        let steps = 48_usize;
        let vig_factor = 1.0 / (1.0 + self.book_vig_pct / 100.0);

        for idx in 0..=steps {
            let rho_val = min_rho + (max_rho - min_rho) * (idx as f64 / steps as f64);
            let Ok(rho) = Correlation::new(rho_val) else {
                continue;
            };
            let Ok(corr) = CorrelationMatrix::equicorrelated(self.num_legs, rho) else {
                continue;
            };
            if let Ok(joint) = sgp_joint_probability(marginals, &corr) {
                let jp = joint.value();
                self.sweep_copula_points.push(PlotPoint::new(rho_val, jp));
                if idx % 6 == 0 || idx == steps {
                    let fair_dec = if jp > 1e-6 { 1.0 / jp } else { 999_999.0 };
                    let offered_dec = (fair_dec * vig_factor).max(1.01);
                    self.pricing_rows.push(SweepPricingRow {
                        rho: rho_val,
                        joint_prob: jp,
                        naive_prob: self.naive_probability,
                        fair_decimal_odds: fair_dec,
                        offered_decimal_odds: offered_dec,
                        fair_american_odds: decimal_to_american_odds(fair_dec),
                    });
                }
            }
        }
        self.sweep_naive_points = [
            PlotPoint::new(min_rho, self.naive_probability),
            PlotPoint::new(max_rho, self.naive_probability),
        ];
    }

    fn populate_contours_and_cloud(&mut self, matrix: &CorrelationMatrix) {
        self.contour_curves.clear();
        self.sampled_cloud_points.clear();
        let pair_rho = matrix
            .get_correlation(self.contour_leg_a, self.contour_leg_b)
            .unwrap_or(self.equicorrelation)
            .clamp(-0.98, 0.98);
        self.effective_pair_rho = pair_rho;

        if let Ok(rho) = Correlation::new(pair_rho) {
            if let Ok(bi_copula) = GaussianCopula::bivariate(rho) {
                for &level in &[0.10_f64, 0.25, 0.50, 0.75] {
                    let mut pts = Vec::with_capacity(48);
                    for step in 1..=48 {
                        let u1_val = level + (0.99 - level) * (step as f64 / 49.0);
                        if let Ok(u1) = Probability::new(u1_val) {
                            if let Some(u2_val) = solve_bivariate_contour_u2(&bi_copula, u1, level)
                            {
                                pts.push(PlotPoint::new(u1_val, u2_val));
                            }
                        }
                    }
                    if !pts.is_empty() {
                        self.contour_curves
                            .push((format!("Iso-CDF C(u, v) = {level:.2}"), pts));
                    }
                }
            }
        }

        let copula = GaussianCopula::new(matrix.clone());
        let mut rng = OxidizeRng::new(self.rng_seed);
        let mut sample_u = [0.0_f64; MAX_COPULA_DIMENSION];
        for _ in 0..160 {
            if copula
                .sample_uniform_into(&mut rng, &mut sample_u[..self.num_legs])
                .is_ok()
            {
                self.sampled_cloud_points.push(PlotPoint::new(
                    sample_u[self.contour_leg_a],
                    sample_u[self.contour_leg_b],
                ));
            }
        }
    }

    pub fn recalculate(&mut self) {
        self.calc_error = None;
        self.joint_probability = None;
        self.active_marker_points.clear();

        let marginals: Result<Vec<Probability>, _> = self
            .marginals
            .iter()
            .copied()
            .map(Probability::new)
            .collect();
        let marginals = match marginals {
            Ok(m) => m,
            Err(e) => {
                self.calc_error = Some(format!("Invalid marginal probability: {e}"));
                return;
            }
        };
        self.naive_probability = marginals.iter().map(Probability::value).product();
        self.populate_sweep_and_sheet(&marginals);

        let matrix = match self.build_active_matrix() {
            Ok(m) => m,
            Err(err) => {
                self.calc_error = Some(format!("Correlation matrix error: {err}"));
                return;
            }
        };

        let copula = GaussianCopula::new(matrix.clone());
        match copula.multivariate_cdf(&marginals) {
            Ok(joint) => {
                let jp = joint.value();
                self.joint_probability = Some(jp);
                self.populate_contours_and_cloud(&matrix);
                let marker_x = match self.mode {
                    CorrelationStructureMode::Equicorrelated => self.equicorrelation,
                    CorrelationStructureMode::PairwiseMatrix => self.effective_pair_rho,
                };
                self.active_marker_points.push(PlotPoint::new(marker_x, jp));
            }
            Err(e) => {
                self.calc_error = Some(format!("Copula evaluation failed: {e}"));
            }
        }
    }
}

// [cite:clinical_trials_statistics]
