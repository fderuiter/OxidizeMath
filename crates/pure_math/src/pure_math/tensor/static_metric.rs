//! Stack-allocated metric tensors and zero-heap-allocation Christoffel symbols.
//!
//! Provides [`StaticMetric`] and [`static_christoffel_symbols`] backed by fixed-size
//! `nalgebra` [`SVector`] and [`SMatrix`] types so geodesic solvers can run inside
//! allocation-locked WebAssembly and verification loops.

use nalgebra::{SMatrix, SVector};

use super::types::TensorError;

/// Fixed-dimension pseudo-Riemannian or Riemannian metric evaluated without heap allocation.
///
/// Implementations supply [`StaticMetric::metric_at`] and may optionally override
/// [`StaticMetric::inverse_metric_at`], [`StaticMetric::christoffel_at`], and
/// [`StaticMetric::is_regular_at`] with closed-form analytical expressions while retaining
/// [`static_christoffel_symbols`] for central-difference cross-validation.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::SVector;
/// use math_explorer::pure_math::tensor::{StaticMetric, TwoSphereMetric};
///
/// let sphere = TwoSphereMetric::new(1.0).unwrap();
/// let point = SVector::<f64, 2>::new(std::f64::consts::FRAC_PI_3, 0.4);
/// let g = sphere.metric_at(&point).unwrap();
/// assert!((g[(0, 0)] - 1.0).abs() < 1e-12);
/// assert!((g[(1, 1)] - 0.75).abs() < 1e-12);
/// ```
pub trait StaticMetric<const N: usize> {
    /// Evaluates the covariant metric tensor $g_{\mu\nu}(x)$ at `point`.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if `point` lies on a coordinate or curvature
    /// singularity or produces non-finite entries.
    fn metric_at(&self, point: &SVector<f64, N>) -> Result<SMatrix<f64, N, N>, TensorError>;

    /// Evaluates the contravariant inverse metric $g^{\mu\nu}(x)$ at `point`.
    ///
    /// Defaults to fixed-size `SMatrix::try_inverse` and can be overridden with closed-form
    /// expressions for diagonal or block-diagonal metrics.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if $g_{\mu\nu}(x)$ is non-invertible.
    #[verified_engine::verified]
    fn inverse_metric_at(
        &self,
        point: &SVector<f64, N>,
    ) -> Result<SMatrix<f64, N, N>, TensorError> {
        let g = self.metric_at(point)?;
        if !g.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        g.try_inverse().ok_or(TensorError::SingularMetric)
    }

    /// Evaluates Christoffel symbols of the second kind $\Gamma^\mu_{\alpha\beta}(x)$ at `point`.
    ///
    /// Element `gamma[mu][(alpha, beta)]` stores $\Gamma^\mu_{\alpha\beta}$. By default this
    /// calls [`static_christoffel_symbols`]; implementations with closed-form connection
    /// formulas can override this method directly.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if the metric or its finite-difference stencil
    /// encounters a singular point.
    #[verified_engine::verified]
    fn christoffel_at(
        &self,
        point: &SVector<f64, N>,
    ) -> Result<[SMatrix<f64, N, N>; N], TensorError>
    where
        Self: Sized,
    {
        static_christoffel_symbols(self, point)
    }

    /// Returns `true` when `point` is safely away from horizon and coordinate singularities.
    #[verified_engine::verified]
    fn is_regular_at(&self, point: &SVector<f64, N>) -> bool {
        point.iter().all(|v| v.is_finite())
    }

    /// Computes the quadratic line-element norm $g_{\mu\nu}(x)\dot{x}^\mu\dot{x}^\nu$.
    ///
    /// For signature $(-, +, +, +)$, timelike four-velocities normalized by $c\tau$ have norm
    /// $-1$, null rays have norm $0$, and spacelike tangents have positive norm.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if the metric evaluation fails or overflows.
    #[verified_engine::verified]
    fn metric_norm(
        &self,
        point: &SVector<f64, N>,
        tangent: &SVector<f64, N>,
    ) -> Result<f64, TensorError> {
        if !tangent.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let g = self.metric_at(point)?;
        let mut sum = 0.0;
        for mu in 0..N {
            for nu in 0..N {
                sum += g[(mu, nu)] * tangent[mu] * tangent[nu];
            }
        }
        if sum.is_finite() {
            Ok(sum)
        } else {
            Err(TensorError::SingularMetric)
        }
    }
}

/// Closure-backed [`StaticMetric`] adapter for custom fixed-dimension manifolds.
#[derive(Debug, Clone, Copy)]
pub struct StaticRiemannianMetric<F, const N: usize> {
    metric_fn: F,
}

impl<F, const N: usize> StaticRiemannianMetric<F, N>
where
    F: Fn(&SVector<f64, N>) -> Result<SMatrix<f64, N, N>, TensorError>,
{
    /// Wraps a stack-allocated metric evaluation closure.
    #[verified_engine::verified]
    pub fn new(metric_fn: F) -> Self {
        Self { metric_fn }
    }
}

impl<F, const N: usize> StaticMetric<N> for StaticRiemannianMetric<F, N>
where
    F: Fn(&SVector<f64, N>) -> Result<SMatrix<f64, N, N>, TensorError>,
{
    #[verified_engine::verified]
    fn metric_at(&self, point: &SVector<f64, N>) -> Result<SMatrix<f64, N, N>, TensorError> {
        (self.metric_fn)(point)
    }
}

/// Standard 2-sphere manifold $S^2$ of radius $R$ in $(\theta, \varphi)$ coordinates.
///
/// Line element: $ds^2 = R^2 d\theta^2 + R^2 \sin^2\theta\,d\varphi^2$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoSphereMetric {
    /// Sphere radius $R > 0$.
    pub radius: f64,
}

impl TwoSphereMetric {
    /// Constructs a 2-sphere metric with radius $R > 0$.
    ///
    /// # Errors
    /// Returns [`TensorError::SingularMetric`] if `radius` is non-positive or non-finite.
    #[verified_engine::verified]
    pub fn new(radius: f64) -> Result<Self, TensorError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(TensorError::SingularMetric);
        }
        Ok(Self { radius })
    }
}

impl StaticMetric<2> for TwoSphereMetric {
    #[verified_engine::verified]
    fn metric_at(&self, point: &SVector<f64, 2>) -> Result<SMatrix<f64, 2, 2>, TensorError> {
        if !self.radius.is_finite() || self.radius <= 0.0 || !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let sin_theta = point[0].sin();
        let r2 = self.radius * self.radius;
        Ok(SMatrix::<f64, 2, 2>::new(
            r2,
            0.0,
            0.0,
            r2 * sin_theta * sin_theta,
        ))
    }

    #[verified_engine::verified]
    fn inverse_metric_at(
        &self,
        point: &SVector<f64, 2>,
    ) -> Result<SMatrix<f64, 2, 2>, TensorError> {
        if !self.radius.is_finite() || self.radius <= 0.0 || !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let sin_theta = point[0].sin();
        if sin_theta.abs() <= 1e-12 {
            return Err(TensorError::SingularMetric);
        }
        let inv_r2 = 1.0 / (self.radius * self.radius);
        Ok(SMatrix::<f64, 2, 2>::new(
            inv_r2,
            0.0,
            0.0,
            inv_r2 / (sin_theta * sin_theta),
        ))
    }

    #[verified_engine::verified]
    fn christoffel_at(
        &self,
        point: &SVector<f64, 2>,
    ) -> Result<[SMatrix<f64, 2, 2>; 2], TensorError> {
        if !self.radius.is_finite() || self.radius <= 0.0 || !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let (sin_theta, cos_theta) = point[0].sin_cos();
        if sin_theta.abs() <= 1e-12 {
            return Err(TensorError::SingularMetric);
        }
        let cot_theta = cos_theta / sin_theta;
        let gamma_theta = SMatrix::<f64, 2, 2>::new(0.0, 0.0, 0.0, -sin_theta * cos_theta);
        let gamma_phi = SMatrix::<f64, 2, 2>::new(0.0, cot_theta, cot_theta, 0.0);
        Ok([gamma_theta, gamma_phi])
    }

    #[verified_engine::verified]
    fn is_regular_at(&self, point: &SVector<f64, 2>) -> bool {
        let theta = point[0];
        self.radius.is_finite()
            && self.radius > 0.0
            && point.iter().all(|v| v.is_finite())
            && theta > 1e-4
            && theta < std::f64::consts::PI - 1e-4
            && theta.sin() > 1e-4
    }
}

#[verified_engine::verified]
fn compute_metric_partials<M: StaticMetric<N> + ?Sized, const N: usize>(
    metric: &M,
    point: &SVector<f64, N>,
) -> Result<[SMatrix<f64, N, N>; N], TensorError> {
    let mut partials = [SMatrix::<f64, N, N>::zeros(); N];
    for (sigma, slot) in partials.iter_mut().enumerate() {
        let h = 1e-5 * point[sigma].abs().max(1.0);
        let mut x_plus = *point;
        let mut x_minus = *point;
        x_plus[sigma] += h;
        x_minus[sigma] -= h;
        let g_plus = metric.metric_at(&x_plus)?;
        let g_minus = metric.metric_at(&x_minus)?;
        let diff = (g_plus - g_minus) / (2.0 * h);
        if !diff.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        *slot = diff;
    }
    Ok(partials)
}

/// Computes Christoffel symbols $\Gamma^\mu_{\alpha\beta}(x)$ via second-order central
/// differences using exclusively stack-allocated [`SVector`] and [`SMatrix`] buffers.
///
/// Formula:
/// $$\Gamma^\mu_{\alpha\beta} = \frac{1}{2} g^{\mu\nu} \left(
/// \partial_\alpha g_{\nu\beta} + \partial_\beta g_{\nu\alpha} - \partial_\nu g_{\alpha\beta}
/// \right)$$
///
/// # Errors
/// Returns [`TensorError::SingularMetric`] if `point` is non-finite or the metric is singular
/// at `point` or any central-difference stencil point.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::SVector;
/// use math_explorer::pure_math::tensor::{
///     StaticMetric, TwoSphereMetric, static_christoffel_symbols,
/// };
///
/// let sphere = TwoSphereMetric::new(2.0).unwrap();
/// let pt = SVector::<f64, 2>::new(0.9, 1.2);
/// let exact = sphere.christoffel_at(&pt).unwrap();
/// let approx = static_christoffel_symbols(&sphere, &pt).unwrap();
/// assert!((exact[0][(1, 1)] - approx[0][(1, 1)]).abs() < 1e-8);
/// ```
#[verified_engine::verified]
pub fn static_christoffel_symbols<M: StaticMetric<N> + ?Sized, const N: usize>(
    metric: &M,
    point: &SVector<f64, N>,
) -> Result<[SMatrix<f64, N, N>; N], TensorError> {
    if !point.iter().all(|v| v.is_finite()) {
        return Err(TensorError::SingularMetric);
    }
    let g_inv = metric.inverse_metric_at(point)?;
    let dg = compute_metric_partials(metric, point)?;
    let mut gamma = [SMatrix::<f64, N, N>::zeros(); N];

    for mu in 0..N {
        for alpha in 0..N {
            for beta in alpha..N {
                let mut val = 0.0;
                for nu in 0..N {
                    let bracket =
                        dg[alpha][(nu, beta)] + dg[beta][(nu, alpha)] - dg[nu][(alpha, beta)];
                    val += 0.5 * g_inv[(mu, nu)] * bracket;
                }
                if !val.is_finite() {
                    return Err(TensorError::SingularMetric);
                }
                gamma[mu][(alpha, beta)] = val;
                gamma[mu][(beta, alpha)] = val;
            }
        }
    }
    Ok(gamma)
}
