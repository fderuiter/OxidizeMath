//! `#![no_std]` Richtmyer, Halton, and Sobol multidimensional cubature with shift error bounds.

use crate::error::ProbError;
use crate::math::Z_95_TWO_SIDED;
use crate::workspace::SolverWorkspace;
use oxidize_core::rng::{
    HaltonSequence, OxidizeRng, RichtmyerSequence, SobolSequence, BASE_SEED, MAX_QMC_DIM,
    MAX_SOBOL_DIM,
};

#[path = "cubature_mvn.rs"]
mod cubature_mvn;
#[path = "cubature_semidef.rs"]
mod cubature_semidef;

pub use cubature_mvn::{
    gaussian_copula_richtmyer_cdf, mvn_genz_richtmyer_cdf, workspace_floats_for_mvn_cdf,
};
pub use cubature_semidef::genz_semidefinite_richtmyer_qmc;

/// Quasi-Monte Carlo low-discrepancy point set used for multidimensional cubature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CubatureRule {
    /// Kronecker irrational-rotation sequence $u_{n,j} = \{n\sqrt{p_j} + \Delta_j\}$ with optional Baker's tent.
    Richtmyer {
        /// Whether to apply Baker's periodizing tent transformation $1 - |2u - 1|$.
        baker_tent: bool,
    },
    /// Radical-inverse Halton sequence in prime bases.
    Halton,
    /// Base-2 Joe-Kuo Sobol sequence with Gray-code stepping.
    Sobol,
}

/// Configuration for [`integrate_unit_cube`] and [`integrate_hyperrect`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CubatureConfig {
    /// Selected low-discrepancy cubature rule.
    pub rule: CubatureRule,
    /// Points evaluated per shift replicate (`>= 1`).
    pub n_points: usize,
    /// Number of independent Cranley-Patterson shifts (`0` = single unshifted pass; `>= 2` = shift SE).
    pub n_shifts: usize,
    /// Deterministic seed used to sample Cranley-Patterson shifts when `n_shifts >= 1`.
    pub seed: u64,
}

impl CubatureConfig {
    /// Creates a randomized QMC configuration with `8` Cranley-Patterson shifts for shift error bounds.
    #[must_use]
    pub const fn new(rule: CubatureRule, n_points: usize) -> Self {
        Self {
            rule,
            n_points,
            n_shifts: 8,
            seed: BASE_SEED,
        }
    }

    /// Creates a deterministic unshifted single-pass QMC configuration (`n_shifts = 0`, `std_error = 0.0`).
    #[must_use]
    pub const fn unshifted(rule: CubatureRule, n_points: usize) -> Self {
        Self {
            rule,
            n_points,
            n_shifts: 0,
            seed: BASE_SEED,
        }
    }

    /// Overrides the number of Cranley-Patterson shifts (`>= 2` enables empirical shift standard errors).
    #[must_use]
    pub const fn with_shifts(mut self, n_shifts: usize) -> Self {
        self.n_shifts = n_shifts;
        self
    }

    /// Overrides the deterministic shift seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Multidimensional integral estimate and Cranley-Patterson shift error summary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubatureEstimate {
    /// Estimated multidimensional integral value.
    pub value: f64,
    /// Empirical standard error across independent Cranley-Patterson shifts (`0.0` if `n_shifts < 2`).
    pub std_error: f64,
    /// Lower 95% shift confidence bound (`value - 1.96 * std_error`).
    pub lower_95: f64,
    /// Upper 95% shift confidence bound (`value + 1.96 * std_error`).
    pub upper_95: f64,
    /// Number of QMC points evaluated per shift.
    pub n_points: usize,
    /// Number of shifts executed (`1` when unshifted).
    pub n_shifts: usize,
    /// Total integrand evaluations (`n_points * effective_shifts`).
    pub n_evaluations: usize,
}

/// Minimum `f64` scratch slots required in [`SolverWorkspace`] for `dim`-dimensional cubature.
#[inline]
#[must_use]
pub const fn workspace_floats_for_cubature(dim: usize) -> usize {
    3 * dim
}

fn max_dim_for_rule(rule: CubatureRule) -> usize {
    match rule {
        CubatureRule::Richtmyer { .. } | CubatureRule::Halton => MAX_QMC_DIM,
        CubatureRule::Sobol => MAX_SOBOL_DIM,
    }
}

fn checked_eval<F: FnMut(&[f64]) -> f64>(pt: &[f64], f: &mut F) -> Result<f64, ProbError> {
    let val = f(pt);
    if !val.is_finite() {
        return Err(ProbError::NonFiniteInput {
            context: "cubature integrand",
        });
    }
    Ok(val)
}

fn eval_single_shift<F: FnMut(&[f64]) -> f64>(
    dim: usize,
    rule: CubatureRule,
    n_points: usize,
    shift: Option<&[f64]>,
    digital_shift: Option<&[u32; MAX_SOBOL_DIM]>,
    point_buf: &mut [f64],
    f: &mut F,
) -> Result<f64, ProbError> {
    let mut sum = 0.0;
    match rule {
        CubatureRule::Richtmyer { baker_tent } => {
            let mut seq = RichtmyerSequence::with_baker_tent(dim, baker_tent).ok_or(
                ProbError::InvalidDimension {
                    dim,
                    max_dim: MAX_QMC_DIM,
                },
            )?;
            for _ in 0..n_points {
                if let Some(s) = shift {
                    seq.next_shifted_into(s, point_buf)
                } else {
                    seq.next_into(point_buf)
                }
                sum += checked_eval(&point_buf[..dim], f)?;
            }
        }
        CubatureRule::Halton => {
            let mut seq = HaltonSequence::new(dim).ok_or(ProbError::InvalidDimension {
                dim,
                max_dim: MAX_QMC_DIM,
            })?;
            for _ in 0..n_points {
                if let Some(s) = shift {
                    seq.next_shifted_into(s, point_buf)
                } else {
                    seq.next_into(point_buf)
                }
                sum += checked_eval(&point_buf[..dim], f)?;
            }
        }
        CubatureRule::Sobol => {
            let mut seq = SobolSequence::new(dim).ok_or(ProbError::InvalidDimension {
                dim,
                max_dim: MAX_SOBOL_DIM,
            })?;
            for _ in 0..n_points {
                if let Some(ds) = digital_shift {
                    seq.next_digital_shifted_into(&ds[..dim], point_buf);
                } else {
                    seq.next_into(point_buf);
                }
                sum += checked_eval(&point_buf[..dim], f)?;
            }
        }
    }
    Ok(sum / (n_points as f64))
}

fn sample_shift_buffers(
    rule: CubatureRule,
    dim: usize,
    rng: &mut OxidizeRng,
    shift_buf: &mut [f64],
    digital_shift: &mut [u32; MAX_SOBOL_DIM],
) {
    if rule == CubatureRule::Sobol {
        for slot in digital_shift.iter_mut().take(dim) {
            *slot = rng.raw_u32();
        }
    } else {
        for slot in shift_buf.iter_mut().take(dim) {
            *slot = rng.next_f64_open01();
        }
    }
}

/// Integrates `f` over $[0, 1]^d$ using Richtmyer, Halton, or Sobol cubature over [`SolverWorkspace`].
pub fn integrate_unit_cube<F: FnMut(&[f64]) -> f64>(
    dim: usize,
    cfg: &CubatureConfig,
    workspace: &mut SolverWorkspace<'_>,
    mut f: F,
) -> Result<CubatureEstimate, ProbError> {
    let max_dim = max_dim_for_rule(cfg.rule);
    if dim == 0 || dim > max_dim {
        return Err(ProbError::InvalidDimension { dim, max_dim });
    }
    if cfg.n_points == 0 {
        return Err(ProbError::InvalidCount {
            name: "n_points",
            value: 0,
        });
    }
    let (point_buf, mut rem) = workspace.split_floats(dim)?;
    let (shift_buf, _) = rem.split_floats(dim)?;

    if cfg.n_shifts == 0 {
        let value = eval_single_shift(dim, cfg.rule, cfg.n_points, None, None, point_buf, &mut f)?;
        return Ok(CubatureEstimate {
            value,
            std_error: 0.0,
            lower_95: value,
            upper_95: value,
            n_points: cfg.n_points,
            n_shifts: 1,
            n_evaluations: cfg.n_points,
        });
    }

    let mut rng = OxidizeRng::from_seed(cfg.seed);
    let mut digital_shift = [0_u32; MAX_SOBOL_DIM];
    let (mut mean, mut m2) = (0.0, 0.0);
    for s_idx in 0..cfg.n_shifts {
        sample_shift_buffers(cfg.rule, dim, &mut rng, shift_buf, &mut digital_shift);
        let est = eval_single_shift(
            dim,
            cfg.rule,
            cfg.n_points,
            Some(shift_buf),
            Some(&digital_shift),
            point_buf,
            &mut f,
        )?;
        let delta = est - mean;
        mean += delta / ((s_idx + 1) as f64);
        m2 += delta * (est - mean);
    }

    let std_error = if cfg.n_shifts > 1 {
        libm::sqrt((m2 / ((cfg.n_shifts - 1) as f64)) / (cfg.n_shifts as f64))
    } else {
        0.0
    };
    let half_width = Z_95_TWO_SIDED * std_error;
    Ok(CubatureEstimate {
        value: mean,
        std_error,
        lower_95: mean - half_width,
        upper_95: mean + half_width,
        n_points: cfg.n_points,
        n_shifts: cfg.n_shifts,
        n_evaluations: cfg.n_points.saturating_mul(cfg.n_shifts),
    })
}

/// Integrates `f` over a hyperrectangle $\prod_{j=0}^{d-1} [a_j, b_j]$ using [`SolverWorkspace`].
pub fn integrate_hyperrect<F: FnMut(&[f64]) -> f64>(
    lower: &[f64],
    upper: &[f64],
    cfg: &CubatureConfig,
    workspace: &mut SolverWorkspace<'_>,
    mut f: F,
) -> Result<CubatureEstimate, ProbError> {
    let dim = lower.len();
    let max_dim = max_dim_for_rule(cfg.rule);
    if dim == 0 || dim > max_dim || upper.len() != dim {
        return Err(ProbError::InvalidDimension { dim, max_dim });
    }
    let mut volume = 1.0;
    for j in 0..dim {
        let lo = lower[j];
        let hi = upper[j];
        if !lo.is_finite() || !hi.is_finite() {
            return Err(ProbError::NonFiniteInput {
                context: "hyperrectangle bounds",
            });
        }
        if lo > hi {
            return Err(ProbError::InvalidBounds {
                dim_index: j,
                lower: lo,
                upper: hi,
            });
        }
        volume *= hi - lo;
    }
    if volume == 0.0 {
        return Ok(CubatureEstimate {
            value: 0.0,
            std_error: 0.0,
            lower_95: 0.0,
            upper_95: 0.0,
            n_points: cfg.n_points,
            n_shifts: cfg.n_shifts.max(1),
            n_evaluations: 0,
        });
    }
    let (scaled_buf, mut sub_ws) = workspace.split_floats(dim)?;
    let unit_est = integrate_unit_cube(dim, cfg, &mut sub_ws, |u| {
        for j in 0..dim {
            scaled_buf[j] = lower[j] + (upper[j] - lower[j]) * u[j];
        }
        f(scaled_buf)
    })?;
    let val = unit_est.value * volume;
    let se = unit_est.std_error * volume;
    let half = Z_95_TWO_SIDED * se;
    Ok(CubatureEstimate {
        value: val,
        std_error: se,
        lower_95: val - half,
        upper_95: val + half,
        ..unit_est
    })
}
