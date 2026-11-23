//! Multidimensional quasi-Monte Carlo cubature strategies backed by [`engine_prob::cubature`].
//!
//! Exposes Richtmyer, Halton, and Sobol low-discrepancy cubature with deterministic
//! Cranley-Patterson / digital-shift error bounds alongside 1D quadrature in
//! [`super::integration`].

pub use engine_prob::cubature::{
    CubatureConfig, CubatureEstimate, CubatureRule, integrate_hyperrect, integrate_unit_cube,
    workspace_floats_for_cubature,
};
pub use engine_prob::error::ProbError;
pub use engine_prob::workspace::SolverWorkspace;

use oxidize_core::rng::{BASE_SEED, MAX_QMC_DIM};

const STACK_CUBATURE_FLOATS: usize = workspace_floats_for_cubature(MAX_QMC_DIM);

/// Common interface for multidimensional quasi-Monte Carlo cubature strategies.
pub trait CubatureIntegrator: Send + Sync {
    /// Returns the underlying [`CubatureConfig`].
    #[verified_engine::verified]
    fn config(&self) -> CubatureConfig;

    /// Integrates `f` over $[0, 1]^d$ using caller-supplied [`SolverWorkspace`] with zero allocations.
    #[verified_engine::verified]
    fn integrate_unit_cube_into<F: FnMut(&[f64]) -> f64>(
        &self,
        dim: usize,
        workspace: &mut SolverWorkspace<'_>,
        f: F,
    ) -> Result<CubatureEstimate, ProbError> {
        integrate_unit_cube(dim, &self.config(), workspace, f)
    }

    /// Integrates `f` over $\prod_{j=0}^{d-1} [a_j, b_j]$ using [`SolverWorkspace`] with zero allocations.
    #[verified_engine::verified]
    fn integrate_hyperrect_into<F: FnMut(&[f64]) -> f64>(
        &self,
        lower: &[f64],
        upper: &[f64],
        workspace: &mut SolverWorkspace<'_>,
        f: F,
    ) -> Result<CubatureEstimate, ProbError> {
        integrate_hyperrect(lower, upper, &self.config(), workspace, f)
    }

    /// Integrates `f` over $[0, 1]^d$ using a stack-allocated workspace (`d <= 64`).
    #[verified_engine::verified]
    fn integrate_cube<F: FnMut(&[f64]) -> f64>(
        &self,
        dim: usize,
        f: F,
    ) -> Result<CubatureEstimate, ProbError> {
        let mut stack_buf = [0.0_f64; STACK_CUBATURE_FLOATS];
        let mut ws = SolverWorkspace::from_floats(&mut stack_buf);
        self.integrate_unit_cube_into(dim, &mut ws, f)
    }

    /// Integrates `f` over $\prod_{j=0}^{d-1} [a_j, b_j]$ using a stack-allocated workspace (`d <= 64`).
    #[verified_engine::verified]
    fn integrate_box<F: FnMut(&[f64]) -> f64>(
        &self,
        lower: &[f64],
        upper: &[f64],
        f: F,
    ) -> Result<CubatureEstimate, ProbError> {
        let mut stack_buf = [0.0_f64; STACK_CUBATURE_FLOATS];
        let mut ws = SolverWorkspace::from_floats(&mut stack_buf);
        self.integrate_hyperrect_into(lower, upper, &mut ws, f)
    }
}

/// Kronecker-Richtmyer multidimensional cubature with optional Baker's tent periodization.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::cubature::{CubatureIntegrator, RichtmyerCubature};
///
/// let integrator = RichtmyerCubature::new(2048);
/// let est = integrator.integrate_cube(3, |u| u[0] + u[1] + u[2]).unwrap();
/// assert!((est.value - 1.5).abs() < 1e-3);
/// assert!(est.std_error >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RichtmyerCubature {
    /// Points evaluated per shift replicate.
    pub n_points: usize,
    /// Number of Cranley-Patterson shifts (`0` = unshifted; `>= 2` = shift SE).
    pub n_shifts: usize,
    /// Whether to apply Baker's tent periodization.
    pub baker_tent: bool,
    /// Deterministic shift seed.
    pub seed: u64,
}

impl RichtmyerCubature {
    /// Creates a Baker-tent Richtmyer cubature integrator with `8` Cranley-Patterson shifts.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(n_points: usize) -> Self {
        Self {
            n_points,
            n_shifts: 8,
            baker_tent: true,
            seed: BASE_SEED,
        }
    }

    /// Overrides the number of Cranley-Patterson shifts.
    #[must_use]
    #[verified_engine::verified]
    pub const fn with_shifts(mut self, n_shifts: usize) -> Self {
        self.n_shifts = n_shifts;
        self
    }
}

impl Default for RichtmyerCubature {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(4096)
    }
}

impl CubatureIntegrator for RichtmyerCubature {
    #[verified_engine::verified]
    fn config(&self) -> CubatureConfig {
        CubatureConfig {
            rule: CubatureRule::Richtmyer {
                baker_tent: self.baker_tent,
            },
            n_points: self.n_points,
            n_shifts: self.n_shifts,
            seed: self.seed,
        }
    }
}

/// Radical-inverse Halton multidimensional cubature with Cranley-Patterson shift error bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HaltonCubature {
    /// Points evaluated per shift replicate.
    pub n_points: usize,
    /// Number of Cranley-Patterson shifts.
    pub n_shifts: usize,
    /// Deterministic shift seed.
    pub seed: u64,
}

impl HaltonCubature {
    /// Creates a Halton cubature integrator with `8` Cranley-Patterson shifts.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(n_points: usize) -> Self {
        Self {
            n_points,
            n_shifts: 8,
            seed: BASE_SEED,
        }
    }
}

impl Default for HaltonCubature {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(4096)
    }
}

impl CubatureIntegrator for HaltonCubature {
    #[verified_engine::verified]
    fn config(&self) -> CubatureConfig {
        CubatureConfig {
            rule: CubatureRule::Halton,
            n_points: self.n_points,
            n_shifts: self.n_shifts,
            seed: self.seed,
        }
    }
}

/// Base-2 Joe-Kuo Sobol multidimensional cubature with digital-shift error bounds (`d <= 32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SobolCubature {
    /// Points evaluated per digital-shift replicate.
    pub n_points: usize,
    /// Number of independent base-2 digital shifts.
    pub n_shifts: usize,
    /// Deterministic shift seed.
    pub seed: u64,
}

impl SobolCubature {
    /// Creates a Sobol cubature integrator with `8` digital shifts.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new(n_points: usize) -> Self {
        Self {
            n_points,
            n_shifts: 8,
            seed: BASE_SEED,
        }
    }
}

impl Default for SobolCubature {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(4096)
    }
}

impl CubatureIntegrator for SobolCubature {
    #[verified_engine::verified]
    fn config(&self) -> CubatureConfig {
        CubatureConfig {
            rule: CubatureRule::Sobol,
            n_points: self.n_points,
            n_shifts: self.n_shifts,
            seed: self.seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_cubature_rules_converge_zero_alloc() {
        let ((r, h, s), _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let r = RichtmyerCubature::new(1024)
                .integrate_box(&[0.0, 1.0], &[2.0, 3.0], |x| x[0] * x[1])
                .unwrap();
            let h = HaltonCubature::new(1024)
                .integrate_box(&[0.0, 1.0], &[2.0, 3.0], |x| x[0] * x[1])
                .unwrap();
            let s = SobolCubature::new(1024)
                .integrate_box(&[0.0, 1.0], &[2.0, 3.0], |x| x[0] * x[1])
                .unwrap();
            (r, h, s)
        });

        // Exact integral of x * y on [0, 2] x [1, 3] is 2 * 4 = 8.0
        assert!((r.value - 8.0).abs() < 0.02);
        assert!((h.value - 8.0).abs() < 0.02);
        assert!((s.value - 8.0).abs() < 0.01);
        assert!(s.std_error > 0.0);
    }
}
