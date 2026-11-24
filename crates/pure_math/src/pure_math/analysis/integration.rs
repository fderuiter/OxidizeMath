//! Numerical Integration Strategies (1D quadrature and hyperrectangle QMC/MC integration).
//!
//! Provides a common interface for scalar quadrature (`ClenshawCurtis`, `Trapezoidal` with
//! Richardson $O(h^2)$ error estimation) alongside a preallocated [`MultivariateIntegrator`]
//! over hyperrectangles $[a_1, b_1] \times \dots \times [a_d, b_d]$ backed by Joe-Kuo Sobol,
//! prime-base Halton, Latin Hypercube, or IID Monte Carlo sampling.

use crate::pure_math::statistics::monte_carlo::{
    MonteCarloError, MonteCarloWorkspace, SamplingMethod,
};
use crate::pure_math::statistics::qmc::{MAX_HALTON_DIM, MAX_SOBOL_DIM};
use oxidize_core::rng::OxidizeRng;
use quadrature::clenshaw_curtis;

/// Result of an integration operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegrationResult {
    /// The estimated value of the integral.
    pub value: f64,
    /// The estimated absolute or standard error.
    pub error: f64,
}

/// A trait for scalar numerical integration strategies.
pub trait Integrator: Send + Sync {
    /// Integrates the function `f` over the interval `[min, max]` with a target error `eps`.
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64;
}

/// Clenshaw-Curtis Quadrature Integration.
///
/// Uses the `quadrature` crate.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClenshawCurtis;

impl Integrator for ClenshawCurtis {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        let result = clenshaw_curtis::integrate(f, min, max, eps);
        IntegrationResult {
            value: result.integral,
            error: result.error_estimate,
        }
    }
}

/// Trapezoidal Rule Integration with $O(h^2)$ Richardson extrapolation error estimation.
#[derive(Debug, Clone, Copy)]
pub struct Trapezoidal {
    /// Number of steps (segments) to use.
    pub steps: usize,
}

impl Trapezoidal {
    /// Creates a new Trapezoidal integrator with the specified number of steps.
    #[verified_engine::verified]
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }
}

impl Default for Trapezoidal {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { steps: 100 }
    }
}

impl Integrator for Trapezoidal {
    #[verified_engine::verified]
    fn integrate<F>(&self, f: F, min: f64, max: f64, _eps: f64) -> IntegrationResult
    where
        F: Fn(f64) -> f64,
    {
        if self.steps == 0 || (max - min).abs() == 0.0 {
            return IntegrationResult {
                value: 0.0,
                error: 0.0,
            };
        }

        let h = (max - min) / (self.steps as f64);
        let ends = 0.5 * (f(min) + f(max));
        if self.steps >= 2 && self.steps.is_multiple_of(2) {
            let mut even_sum = 0.0;
            let mut odd_sum = 0.0;
            for i in 1..self.steps {
                let y = f(min + (i as f64) * h);
                if i.is_multiple_of(2) {
                    even_sum += y;
                } else {
                    odd_sum += y;
                }
            }
            let fine = (ends + even_sum + odd_sum) * h;
            let coarse = (ends + even_sum) * (2.0 * h);
            return IntegrationResult {
                value: fine,
                error: (fine - coarse).abs() / 3.0,
            };
        }

        let mut interior = 0.0;
        for i in 1..self.steps {
            interior += f(min + (i as f64) * h);
        }
        let coarse = (ends + interior) * h;
        let mut mid_sum = 0.0;
        for i in 0..self.steps {
            mid_sum += f(min + ((i as f64) + 0.5) * h);
        }
        let refined = 0.5 * (coarse + h * mid_sum);
        IntegrationResult {
            value: coarse,
            error: (4.0 / 3.0) * (refined - coarse).abs(),
        }
    }
}

/// Sampling engine backing [`MultivariateIntegrator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultivariateRule {
    /// Cranley-Patterson randomized Joe-Kuo Sobol low-discrepancy integration (`1..=32` dims).
    Sobol,
    /// Cranley-Patterson randomized prime-base Halton low-discrepancy integration (`1..=64` dims).
    Halton,
    /// Stratified Latin Hypercube integration (`1..=64` dims).
    LatinHypercube,
    /// Standard IID Monte Carlo integration (`1..=64` dims).
    MonteCarlo,
}

/// Preallocated hyperrectangle integrator mapping `(0, 1)^d` QMC/MC rules onto $[a_1, b_1] \times \dots \times [a_d, b_d]$.
#[derive(Debug, Clone)]
pub struct MultivariateIntegrator {
    rule: MultivariateRule,
    n_samples: usize,
    seed: u64,
    scaled_buf: Vec<f64>,
    widths_buf: Vec<f64>,
    workspace: MonteCarloWorkspace,
}

impl MultivariateIntegrator {
    /// Constructs a preallocated hyperrectangle integrator with the specified [`MultivariateRule`].
    #[verified_engine::verified]
    pub fn with_rule(
        rule: MultivariateRule,
        dimension: usize,
        n_samples: usize,
    ) -> Result<Self, MonteCarloError> {
        if matches!(rule, MultivariateRule::Sobol) && (dimension == 0 || dimension > MAX_SOBOL_DIM)
        {
            return Err(MonteCarloError::InvalidDimension {
                dimension,
                max_dimension: MAX_SOBOL_DIM,
            });
        }
        if dimension == 0 || dimension > MAX_HALTON_DIM {
            return Err(MonteCarloError::InvalidDimension {
                dimension,
                max_dimension: MAX_HALTON_DIM,
            });
        }
        let workspace = MonteCarloWorkspace::new(n_samples, dimension)?;
        Ok(Self {
            rule,
            n_samples,
            seed: 0x5EED_C0DE_2026,
            scaled_buf: vec![0.0; dimension],
            widths_buf: vec![0.0; dimension],
            workspace,
        })
    }

    /// Constructs a Joe-Kuo Sobol hyperrectangle integrator (`1..=32` dimensions).
    #[verified_engine::verified]
    pub fn new_sobol(dimension: usize, n_samples: usize) -> Result<Self, MonteCarloError> {
        Self::with_rule(MultivariateRule::Sobol, dimension, n_samples)
    }

    /// Constructs a prime-base Halton hyperrectangle integrator (`1..=64` dimensions).
    #[verified_engine::verified]
    pub fn new_halton(dimension: usize, n_samples: usize) -> Result<Self, MonteCarloError> {
        Self::with_rule(MultivariateRule::Halton, dimension, n_samples)
    }

    /// Constructs a stratified Latin Hypercube hyperrectangle integrator.
    #[verified_engine::verified]
    pub fn new_latin_hypercube(
        dimension: usize,
        n_samples: usize,
    ) -> Result<Self, MonteCarloError> {
        Self::with_rule(MultivariateRule::LatinHypercube, dimension, n_samples)
    }

    /// Constructs a standard IID Monte Carlo hyperrectangle integrator.
    #[verified_engine::verified]
    pub fn new_monte_carlo(dimension: usize, n_samples: usize) -> Result<Self, MonteCarloError> {
        Self::with_rule(MultivariateRule::MonteCarlo, dimension, n_samples)
    }

    /// Overrides the deterministic `OxidizeRng` seed used for Cranley-Patterson shifts or draws.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Coordinate dimensionality of the integrator.
    #[inline]
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        self.workspace.dimension()
    }

    /// Configured sample budget per integration call.
    #[inline]
    #[verified_engine::verified]
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Integrates `f` directly over the open unit hypercube `(0, 1)^d` with zero heap allocations.
    #[verified_engine::verified]
    pub fn integrate_unit_cube<F>(&mut self, f: F) -> Result<IntegrationResult, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        let mut rng = OxidizeRng::new(self.seed);
        let method = match self.rule {
            MultivariateRule::Sobol => SamplingMethod::RandomizedSobol,
            MultivariateRule::Halton => SamplingMethod::RandomizedHalton,
            MultivariateRule::LatinHypercube => SamplingMethod::LatinHypercube,
            MultivariateRule::MonteCarlo => SamplingMethod::Iid,
        };
        let est = self
            .workspace
            .estimate_with_method(method, self.n_samples, &mut rng, f)?;
        Ok(IntegrationResult {
            value: est.mean,
            error: est.standard_error,
        })
    }

    /// Integrates `f` over the hyperrectangle `[lower[0], upper[0]] x ... x [lower[d-1], upper[d-1]]`.
    #[verified_engine::verified]
    pub fn integrate<F>(
        &mut self,
        lower: &[f64],
        upper: &[f64],
        mut f: F,
    ) -> Result<IntegrationResult, MonteCarloError>
    where
        F: FnMut(&[f64]) -> f64,
    {
        let dim = self.dimension();
        if lower.len() != dim {
            return Err(MonteCarloError::BufferLengthMismatch {
                expected: dim,
                actual: lower.len(),
            });
        }
        if upper.len() != dim {
            return Err(MonteCarloError::BufferLengthMismatch {
                expected: dim,
                actual: upper.len(),
            });
        }

        let mut volume = 1.0;
        for j in 0..dim {
            let lo = lower[j];
            let hi = upper[j];
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(MonteCarloError::InvalidBounds {
                    dimension: j,
                    lower: lo,
                    upper: hi,
                });
            }
            let width = hi - lo;
            self.widths_buf[j] = width;
            volume *= width;
        }
        if volume == 0.0 {
            return Ok(IntegrationResult {
                value: 0.0,
                error: 0.0,
            });
        }

        let mut rng = OxidizeRng::new(self.seed);
        let method = match self.rule {
            MultivariateRule::Sobol => SamplingMethod::RandomizedSobol,
            MultivariateRule::Halton => SamplingMethod::RandomizedHalton,
            MultivariateRule::LatinHypercube => SamplingMethod::LatinHypercube,
            MultivariateRule::MonteCarlo => SamplingMethod::Iid,
        };
        let scaled = &mut self.scaled_buf;
        let widths = &self.widths_buf;
        let est = self
            .workspace
            .estimate_with_method(method, self.n_samples, &mut rng, |u| {
                for j in 0..dim {
                    scaled[j] = lower[j] + u[j] * widths[j];
                }
                f(scaled)
            })?;
        Ok(IntegrationResult {
            value: est.mean * volume,
            error: est.standard_error * volume,
        })
    }
}

#[cfg(test)]
mod tests;

// [cite:graph_parameters_rust]
