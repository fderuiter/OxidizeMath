use crate::error::DistributionError;
use oxidize_core::rng::OxidizeRng;

/// Contract for univariate continuous probability distributions with stack-only evaluation.
///
/// Implementations guarantee deterministic seeded sampling via [`OxidizeRng`] and zero heap
/// allocations across density, cumulative, quantile, moment, and sampling operations.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::distributions::{
///     ContinuousDistribution, NormalDist,
/// };
/// use math_explorer::rng::OxidizeRng;
///
/// # fn main() -> math_explorer::error::Result<()> {
/// let normal = NormalDist::new(0.0, 1.0)?;
/// assert!((normal.cdf(0.0) - 0.5).abs() < 1e-12);
/// let median = normal.quantile(0.5)?;
/// assert!(median.abs() < 1e-12);
///
/// let mut rng = OxidizeRng::from_seed(42);
/// let draw = normal.sample(&mut rng);
/// assert!(draw.is_finite());
/// # Ok(())
/// # }
/// ```
pub trait ContinuousDistribution {
    /// Evaluates the probability density function $f(x)$ at `x`.
    fn pdf(&self, x: f64) -> f64;

    /// Evaluates the natural log-density $\ln f(x)$ at `x` (`f64::NEG_INFINITY` outside support).
    fn ln_pdf(&self, x: f64) -> f64;

    /// Evaluates the cumulative distribution function $F(x) = P(X \le x)$.
    fn cdf(&self, x: f64) -> f64;

    /// Evaluates the quantile (inverse CDF) $F^{-1}(p)$ for $p \in [0, 1]$.
    ///
    /// # Errors
    /// Returns [`DistributionError::InvalidProbability`] if `p` is NaN or outside `[0.0, 1.0]`.
    fn quantile(&self, p: f64) -> Result<f64, DistributionError>;

    /// Returns the analytical expectation $E(X)$ when defined (`None` if undefined, e.g. Cauchy $t_1$).
    fn mean(&self) -> Option<f64>;

    /// Returns the analytical variance $\mathrm{Var}(X)$ when defined (`None` if undefined).
    fn variance(&self) -> Option<f64>;

    /// Returns the analytical standard deviation $\sqrt{\mathrm{Var}(X)}$ when defined.
    #[verified_engine::verified]
    fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Returns the closed or open support endpoints `(lower, upper)`.
    fn support(&self) -> (f64, f64);

    /// Draws a single deterministic variate using `rng` with zero heap allocations.
    fn sample(&self, rng: &mut OxidizeRng) -> f64;
}

/// Contract for non-negative integer discrete probability distributions with stack-only evaluation.
///
/// # Examples
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::distributions::{
///     DiscreteDistribution, PoissonDist,
/// };
/// use math_explorer::rng::OxidizeRng;
///
/// # fn main() -> math_explorer::error::Result<()> {
/// let poisson = PoissonDist::new(3.5)?;
/// assert!((poisson.mean() - 3.5).abs() < 1e-12);
/// let q50 = poisson.quantile(0.5)?;
/// assert!(poisson.cdf(q50) >= 0.5);
///
/// let mut rng = OxidizeRng::from_seed(7);
/// let count = poisson.sample(&mut rng);
/// assert!(poisson.pmf(count) > 0.0);
/// # Ok(())
/// # }
/// ```
pub trait DiscreteDistribution {
    /// Evaluates the probability mass function $P(X = k)$.
    fn pmf(&self, k: u64) -> f64;

    /// Evaluates the natural log-mass $\ln P(X = k)$ (`f64::NEG_INFINITY` outside support).
    fn ln_pmf(&self, k: u64) -> f64;

    /// Evaluates the cumulative distribution function $F(k) = P(X \le k)$.
    fn cdf(&self, k: u64) -> f64;

    /// Evaluates the smallest count $k$ satisfying $F(k) \ge p$ for $p \in [0, 1]$.
    ///
    /// # Errors
    /// Returns [`DistributionError::InvalidProbability`] if `p` is NaN or outside `[0.0, 1.0]`.
    fn quantile(&self, p: f64) -> Result<u64, DistributionError>;

    /// Returns the analytical expectation $E(X)$.
    fn mean(&self) -> f64;

    /// Returns the analytical variance $\mathrm{Var}(X)$.
    fn variance(&self) -> f64;

    /// Returns the analytical standard deviation $\sqrt{\mathrm{Var}(X)}$.
    #[verified_engine::verified]
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Draws a single deterministic count variate using `rng` with zero heap allocations.
    fn sample(&self, rng: &mut OxidizeRng) -> u64;
}

// [cite:clinical_trials_statistics]
