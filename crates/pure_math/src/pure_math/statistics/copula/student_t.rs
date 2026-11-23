//! Gauss-Legendre bivariate Gaussian & Student-t copula kernels and zero-allocation scratch workspace.

use super::bivariate::{bivariate_copula_cdf, bivariate_normal_cdf};
use super::core::{Correlation, Probability};
use crate::error::CopulaError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::flatten_2d_index;
use statrs::distribution::{ContinuousCDF, StudentsT};

/// 16-point Gauss-Legendre nodes on `[-1, 1]`.
const GL16_NODES: [f64; 16] = [
    -0.989_400_934_991_649_9,
    -0.944_575_023_073_232_6,
    -0.865_631_202_387_831_8,
    -0.755_404_408_355_003,
    -0.617_876_244_402_643_8,
    -0.458_016_777_657_227_37,
    -0.281_603_550_779_258_9,
    -0.095_012_509_837_637_44,
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_37,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];

/// 16-point Gauss-Legendre weights on `[-1, 1]`.
const GL16_WEIGHTS: [f64; 16] = [
    0.027_152_459_411_754_095,
    0.062_253_523_938_647_894,
    0.095_158_511_682_492_79,
    0.124_628_971_255_533_88,
    0.149_595_988_816_576_74,
    0.169_156_519_395_002_54,
    0.182_603_415_044_923_58,
    0.189_450_610_455_068_5,
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_58,
    0.169_156_519_395_002_54,
    0.149_595_988_816_576_74,
    0.124_628_971_255_533_88,
    0.095_158_511_682_492_79,
    0.062_253_523_938_647_894,
    0.027_152_459_411_754_095,
];

/// Allocation-free bivariate standard normal CDF $\Phi_2(z_1, z_2; \rho)$ delegating to Genz's
/// 20-point Gauss-Legendre bivariate normal solver ([`bivariate_normal_cdf`]).
#[verified_engine::verified]
pub fn gauss_legendre_bivariate_normal_cdf(z1: f64, z2: f64, rho: f64) -> f64 {
    if !rho.is_finite() {
        return 0.0;
    }
    bivariate_normal_cdf(z1, z2, rho.clamp(-1.0, 1.0)).unwrap_or(0.0)
}

/// Allocation-free bivariate Student-t CDF $T_2(b_1, b_2; \rho, \nu)$ via Genz (2004) 16-point
/// Gauss-Legendre trigonometric quadrature from the Fréchet-Hoeffding boundary $\rho = \pm 1$.
#[verified_engine::verified]
pub fn gauss_legendre_bivariate_student_t_cdf(b1: f64, b2: f64, rho: f64, nu: f64) -> f64 {
    bivariate_student_t_cdf_inner(b1, b2, rho, nu)
}

fn bivariate_student_t_cdf_inner(b1: f64, b2: f64, rho: f64, nu: f64) -> f64 {
    let dof = nu.max(0.5);
    let Ok(t_dist) = StudentsT::new(0.0, 1.0, dof) else {
        return 0.0;
    };
    let u1 = t_dist.cdf(b1);
    let u2 = t_dist.cdf(b2);
    let r = rho.clamp(-1.0, 1.0);
    if (r - 1.0).abs() <= 1e-12 {
        return u1.min(u2);
    }
    if (r + 1.0).abs() <= 1e-12 {
        return (u1 + u2 - 1.0).max(0.0);
    }

    let half_pi = 0.5 * std::f64::consts::PI;
    let (th0, th1, positive_branch) = if r >= 0.0 {
        (r.asin(), half_pi, true)
    } else {
        (-half_pi, r.asin(), false)
    };
    let mid = 0.5 * (th0 + th1);
    let half = 0.5 * (th1 - th0);
    let b_sq = b1 * b1 + b2 * b2;
    let two_b1_b2 = 2.0 * b1 * b2;
    let mut sum = 0.0;
    for i in 0..16 {
        let theta = mid + half * GL16_NODES[i];
        let s = theta.sin();
        let c2 = theta.cos().powi(2);
        if c2 > 1e-30 {
            let base = 1.0 + (b_sq - s * two_b1_b2) / (dof * c2);
            sum += GL16_WEIGHTS[i] * base.powf(-0.5 * dof);
        }
    }
    let integral = sum * half / (2.0 * std::f64::consts::PI);
    let raw = if positive_branch {
        u1.min(u2) - integral
    } else {
        (u1 + u2 - 1.0).max(0.0) + integral
    };
    raw.clamp(0.0, 1.0)
}

/// Bivariate Student-t copula modeling heavy-tailed symmetric joint dependence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StudentTCopula {
    correlation: Correlation,
    degrees_of_freedom: f64,
}

impl StudentTCopula {
    /// Creates a validated bivariate Student-t copula with correlation `rho` and `degrees_of_freedom > 0`.
    #[verified_engine::verified]
    pub fn new(correlation: Correlation, degrees_of_freedom: f64) -> Result<Self, CopulaError> {
        if !degrees_of_freedom.is_finite() || degrees_of_freedom <= 0.0 {
            return Err(CopulaError::Math(MathError::NumericalError {
                reason: "Degrees of freedom must be finite and positive".to_string(),
            }));
        }
        Ok(Self {
            correlation,
            degrees_of_freedom,
        })
    }

    /// Returns the correlation parameter $\rho$.
    #[verified_engine::verified]
    pub fn correlation(&self) -> Correlation {
        self.correlation
    }

    /// Returns the degrees of freedom $\nu$.
    #[verified_engine::verified]
    pub fn degrees_of_freedom(&self) -> f64 {
        self.degrees_of_freedom
    }

    /// Evaluates the bivariate Student-t copula CDF $C_{\nu, \rho}(u_1, u_2)$ without heap allocation.
    #[verified_engine::verified]
    pub fn cdf(&self, u1: Probability, u2: Probability) -> Result<Probability, CopulaError> {
        let val = student_t_copula_cdf_raw(
            u1.value(),
            u2.value(),
            self.correlation.value(),
            self.degrees_of_freedom,
        );
        Probability::new(val)
    }

    /// Asymptotic upper/lower tail-dependence coefficient $\lambda_U = \lambda_L = 2 t_{\nu+1}\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)$.
    #[verified_engine::verified]
    pub fn tail_dependence_coefficient(&self) -> f64 {
        asymptotic_tail_dependence_inner(self.correlation.value(), self.degrees_of_freedom)
    }
}

fn gaussian_copula_cdf_raw(u1: f64, u2: f64, rho: f64) -> f64 {
    if !rho.is_finite() || u1 <= 0.0 || u2 <= 0.0 {
        return 0.0;
    }
    if u1 >= 1.0 {
        return u2.clamp(0.0, 1.0);
    }
    if u2 >= 1.0 {
        return u1.clamp(0.0, 1.0);
    }
    let (Ok(p1), Ok(p2), Ok(r)) = (
        Probability::new(u1),
        Probability::new(u2),
        Correlation::new(rho.clamp(-1.0, 1.0)),
    ) else {
        return 0.0;
    };
    bivariate_copula_cdf(p1, p2, r).map_or(0.0, |p| p.value())
}

fn student_t_copula_cdf_raw(u1: f64, u2: f64, rho: f64, nu: f64) -> f64 {
    if u1 <= 0.0 || u2 <= 0.0 {
        return 0.0;
    }
    if u1 >= 1.0 {
        return u2.clamp(0.0, 1.0);
    }
    if u2 >= 1.0 {
        return u1.clamp(0.0, 1.0);
    }
    let dof = nu.max(0.5);
    let Ok(t_dist) = StudentsT::new(0.0, 1.0, dof) else {
        return 0.0;
    };
    let b1 = t_dist.inverse_cdf(u1);
    let b2 = t_dist.inverse_cdf(u2);
    bivariate_student_t_cdf_inner(b1, b2, rho, dof)
}

fn asymptotic_tail_dependence_inner(rho: f64, nu: f64) -> f64 {
    let r = rho.clamp(-1.0, 1.0);
    if (r - 1.0).abs() <= 1e-12 {
        return 1.0;
    }
    if (r + 1.0).abs() <= 1e-12 {
        return 0.0;
    }
    let dof_plus_one = nu.max(0.5) + 1.0;
    let Ok(t_dist) = StudentsT::new(0.0, 1.0, dof_plus_one) else {
        return 0.0;
    };
    let arg = -(dof_plus_one * (1.0 - r) / (1.0 + r)).sqrt();
    (2.0 * t_dist.cdf(arg)).clamp(0.0, 1.0)
}

/// Side-by-side Gaussian vs Student-t joint and conditional extreme-tail probabilities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CopulaTailComparison {
    /// Gaussian copula joint CDF $C^{\text{Gauss}}_\rho(u_1, u_2)$.
    pub gaussian_joint: f64,
    /// Student-t copula joint CDF $C^{t}_{\rho, \nu}(u_1, u_2)$.
    pub student_t_joint: f64,
    /// Conditional Gaussian upper-tail crash probability $P(U_2 > u \mid U_1 > u)$.
    pub gaussian_upper_tail: f64,
    /// Conditional Student-t upper-tail crash probability $P(U_2 > u \mid U_1 > u)$.
    pub student_t_upper_tail: f64,
    /// Heavy-tail amplification ratio $P_t(U_2 > u \mid U_1 > u) / P_{\text{Gauss}}(U_2 > u \mid U_1 > u)$.
    pub tail_multiplier: f64,
    /// Asymptotic Student-t tail dependence $\lambda_U(\rho, \nu)$.
    pub asymptotic_lambda_u: f64,
}

/// Preallocated scratch buffers for zero-allocation 2D $(\rho, u)$ copula regime heatmaps.
#[derive(Debug, Clone)]
pub struct BivariateCopulaScratch {
    max_cols: usize,
    max_rows: usize,
    gaussian_tail_grid: Vec<f64>,
    student_t_tail_grid: Vec<f64>,
    multiplier_grid: Vec<f64>,
}

impl BivariateCopulaScratch {
    /// Preallocates 2D $(\rho, u)$ regime buffers for up to `max_cols * max_rows` cells.
    #[verified_engine::verified]
    pub fn new(max_cols: usize, max_rows: usize) -> Self {
        let cap = max_cols.saturating_mul(max_rows);
        Self {
            max_cols,
            max_rows,
            gaussian_tail_grid: vec![0.0; cap],
            student_t_tail_grid: vec![0.0; cap],
            multiplier_grid: vec![1.0; cap],
        }
    }

    /// Evaluates bivariate Gaussian copula CDF $C^{\text{Gauss}}_\rho(u_1, u_2)$ with zero heap allocations.
    #[verified_engine::verified]
    pub fn evaluate_gaussian_joint(&self, u1: f64, u2: f64, rho: f64) -> f64 {
        gaussian_copula_cdf_raw(u1.clamp(0.0, 1.0), u2.clamp(0.0, 1.0), rho)
    }

    /// Evaluates bivariate Student-t copula CDF $C^{t}_{\rho, \nu}(u_1, u_2)$ with zero heap allocations.
    #[verified_engine::verified]
    pub fn evaluate_student_t_joint(&self, u1: f64, u2: f64, rho: f64, nu: f64) -> f64 {
        student_t_copula_cdf_raw(u1.clamp(0.0, 1.0), u2.clamp(0.0, 1.0), rho, nu)
    }

    /// Evaluates joint probabilities and extreme upper-tail conditional crash amplification at quantile `tail_u`.
    #[verified_engine::verified]
    pub fn evaluate_tail_comparison(
        &self,
        u1: f64,
        u2: f64,
        tail_u: f64,
        rho: f64,
        nu: f64,
    ) -> CopulaTailComparison {
        let gaussian_joint = gaussian_copula_cdf_raw(u1.clamp(0.0, 1.0), u2.clamp(0.0, 1.0), rho);
        let student_t_joint =
            student_t_copula_cdf_raw(u1.clamp(0.0, 1.0), u2.clamp(0.0, 1.0), rho, nu);
        let alpha = (1.0 - tail_u.clamp(0.5, 0.999_999)).max(1e-9);
        // By radial symmetry of both elliptical copulas, P(U1 > u, U2 > u) = C(1 - u, 1 - u).
        let gaussian_upper_tail =
            (gaussian_copula_cdf_raw(alpha, alpha, rho) / alpha).clamp(0.0, 1.0);
        let student_t_upper_tail =
            (student_t_copula_cdf_raw(alpha, alpha, rho, nu) / alpha).clamp(0.0, 1.0);
        let tail_multiplier = if gaussian_upper_tail > 1e-12 {
            student_t_upper_tail / gaussian_upper_tail
        } else {
            1.0
        };
        CopulaTailComparison {
            gaussian_joint,
            student_t_joint,
            gaussian_upper_tail,
            student_t_upper_tail,
            tail_multiplier,
            asymptotic_lambda_u: asymptotic_tail_dependence_inner(rho, nu),
        }
    }

    /// Populates the 2D $(\rho, u)$ conditional tail-multiplier grid in place without heap allocations.
    #[verified_engine::verified]
    pub fn populate_tail_regime_grid(&mut self, rhos: &[f64], quantiles: &[f64], nu: f64) {
        let cols = rhos.len().min(self.max_cols);
        let rows = quantiles.len().min(self.max_rows);
        for (row, &q) in quantiles.iter().enumerate().take(rows) {
            let alpha = (1.0 - q.clamp(0.5, 0.999_999)).max(1e-9);
            for (col, &rho) in rhos.iter().enumerate().take(cols) {
                let idx = flatten_2d_index(col, row, self.max_cols);
                let g_tail = (gaussian_copula_cdf_raw(alpha, alpha, rho) / alpha).clamp(0.0, 1.0);
                let t_tail =
                    (student_t_copula_cdf_raw(alpha, alpha, rho, nu) / alpha).clamp(0.0, 1.0);
                self.gaussian_tail_grid[idx] = g_tail;
                self.student_t_tail_grid[idx] = t_tail;
                self.multiplier_grid[idx] = if g_tail > 1e-12 { t_tail / g_tail } else { 1.0 };
            }
        }
    }

    /// Reads the precomputed Student-t / Gaussian tail-multiplier cell at `(col, row)`.
    #[verified_engine::verified]
    pub fn multiplier_at(&self, col: usize, row: usize) -> f64 {
        let idx = flatten_2d_index(
            col.min(self.max_cols - 1),
            row.min(self.max_rows - 1),
            self.max_cols,
        );
        self.multiplier_grid[idx]
    }

    /// Reads the precomputed Gaussian conditional upper-tail probability at `(col, row)`.
    #[verified_engine::verified]
    pub fn gaussian_tail_at(&self, col: usize, row: usize) -> f64 {
        let idx = flatten_2d_index(
            col.min(self.max_cols - 1),
            row.min(self.max_rows - 1),
            self.max_cols,
        );
        self.gaussian_tail_grid[idx]
    }

    /// Reads the precomputed Student-t conditional upper-tail probability at `(col, row)`.
    #[verified_engine::verified]
    pub fn student_t_tail_at(&self, col: usize, row: usize) -> f64 {
        let idx = flatten_2d_index(
            col.min(self.max_cols - 1),
            row.min(self.max_rows - 1),
            self.max_cols,
        );
        self.student_t_tail_grid[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::super::core::CorrelationMatrix;
    use super::super::gaussian::sgp_joint_probability;
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_sgp_parlay_swing_and_student_t_tail_multiplier_zero_alloc() {
        let p1 = Probability::new(0.20).unwrap();
        let p2 = Probability::new(0.20).unwrap();

        let neg_corr = CorrelationMatrix::bivariate(Correlation::new(-0.50).unwrap()).unwrap();
        let pos_corr = CorrelationMatrix::bivariate(Correlation::new(0.70).unwrap()).unwrap();

        let p_neg = sgp_joint_probability(&[p1, p2], &neg_corr).unwrap().value();
        let p_pos = sgp_joint_probability(&[p1, p2], &pos_corr).unwrap().value();
        // Reproduces 0.84% (rho = -0.50) vs 11.29% (rho = 0.70) around naive 4.00%
        assert_relative_eq!(p_neg, 0.008_438, epsilon = 1e-5);
        assert_relative_eq!(p_pos, 0.112_902, epsilon = 1e-5);

        let t_copula = StudentTCopula::new(Correlation::new(0.50).unwrap(), 3.0).unwrap();
        assert!(t_copula.tail_dependence_coefficient() > 0.30);

        let mut scratch = BivariateCopulaScratch::new(8, 8);
        let rhos = [-0.50, 0.0, 0.50, 0.70];
        let quantiles = [0.90, 0.99, 0.999];

        let (cmp, _cert) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            scratch.populate_tail_regime_grid(&rhos, &quantiles, 3.0);
            scratch.evaluate_tail_comparison(0.20, 0.20, 0.999, 0.50, 3.0)
        });

        // Reproduces 5.83x Student-t vs Gaussian upper-tail conditional crash ratio at u = 0.999
        assert_relative_eq!(cmp.tail_multiplier, 5.8254, epsilon = 1e-2);
        assert_relative_eq!(
            scratch.multiplier_at(2, 2),
            cmp.tail_multiplier,
            epsilon = 1e-12
        );
    }
}

// [cite:clinical_trials_statistics]
