//! Stack-allocated metric tensors and zero-heap-allocation Christoffel symbols.
//!
//! Provides [`StaticMetric`], [`StaticAdMetric`], and [`static_christoffel_symbols`] backed by
//! fixed-size `nalgebra` [`SVector`] and [`SMatrix`] types so geodesic solvers run inside
//! allocation-locked WebAssembly and verification loops with machine-precision AD derivatives.

use math_scalars::{Dual, Scalar};
use nalgebra::{SMatrix, SVector};

use super::types::TensorError;

/// Fixed-dimension pseudo-Riemannian or Riemannian metric evaluated without heap allocation.
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
    fn metric_at(&self, point: &SVector<f64, N>) -> Result<SMatrix<f64, N, N>, TensorError>;

    /// Evaluates coordinate partial derivatives $\partial_\sigma g_{\mu\nu}(x)$ on the stack.
    ///
    /// [`StaticAdMetric`] and [`TwoSphereMetric`] override this with exact [`Dual<f64>`] tangents;
    /// closure metrics fall back to scale-aware $\varepsilon^{1/3}$ central differences.
    #[verified_engine::verified]
    fn metric_partials_at(
        &self,
        point: &SVector<f64, N>,
    ) -> Result<[SMatrix<f64, N, N>; N], TensorError> {
        let mut partials = [SMatrix::<f64, N, N>::zeros(); N];
        for (sigma, slot) in partials.iter_mut().enumerate() {
            let h = f64::EPSILON.cbrt() * point[sigma].abs().max(1.0);
            let mut x_plus = *point;
            let mut x_minus = *point;
            x_plus[sigma] = point[sigma] + h;
            x_minus[sigma] = point[sigma] - h;
            let span = x_plus[sigma] - x_minus[sigma];
            let g_plus = self.metric_at(&x_plus)?;
            let g_minus = self.metric_at(&x_minus)?;
            let diff = (g_plus - g_minus) / span;
            if !diff.iter().all(|v| v.is_finite()) {
                return Err(TensorError::SingularMetric);
            }
            *slot = diff;
        }
        Ok(partials)
    }

    /// Evaluates the contravariant inverse metric $g^{\mu\nu}(x)$ at `point`.
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

/// Zero-allocation forward-AD [`StaticMetric`] adapter backed by `[Dual<f64>; N]`.
#[derive(Debug, Clone, Copy)]
pub struct StaticAdMetric<F, const N: usize> {
    metric_fn: F,
}

impl<F, const N: usize> StaticAdMetric<F, N>
where
    F: Fn(&[Dual<f64>; N]) -> Result<[[Dual<f64>; N]; N], TensorError>,
{
    /// Wraps a stack-allocated [`Dual<f64>`] metric evaluator (`g[row][col]`).
    #[verified_engine::verified]
    pub const fn new(metric_fn: F) -> Self {
        Self { metric_fn }
    }
}

impl<F, const N: usize> StaticMetric<N> for StaticAdMetric<F, N>
where
    F: Fn(&[Dual<f64>; N]) -> Result<[[Dual<f64>; N]; N], TensorError>,
{
    #[verified_engine::verified]
    fn metric_at(&self, point: &SVector<f64, N>) -> Result<SMatrix<f64, N, N>, TensorError> {
        let mut coords = [Dual::ZERO; N];
        for i in 0..N {
            coords[i] = Dual::constant(point[i]);
        }
        let raw = (self.metric_fn)(&coords)?;
        let mut g = SMatrix::<f64, N, N>::zeros();
        for r in 0..N {
            for c in 0..N {
                if !raw[r][c].val.is_finite() {
                    return Err(TensorError::SingularMetric);
                }
                g[(r, c)] = raw[r][c].val;
            }
        }
        Ok(g)
    }

    #[verified_engine::verified]
    fn metric_partials_at(
        &self,
        point: &SVector<f64, N>,
    ) -> Result<[SMatrix<f64, N, N>; N], TensorError> {
        let mut coords = [Dual::ZERO; N];
        for i in 0..N {
            coords[i] = Dual::constant(point[i]);
        }
        let mut partials = [SMatrix::<f64, N, N>::zeros(); N];
        for (sigma, slot) in partials.iter_mut().enumerate() {
            coords[sigma] = Dual::variable(point[sigma]);
            let raw = (self.metric_fn)(&coords)?;
            coords[sigma] = Dual::constant(point[sigma]);
            for r in 0..N {
                for c in 0..N {
                    if !raw[r][c].eps.is_finite() {
                        return Err(TensorError::SingularMetric);
                    }
                    slot[(r, c)] = raw[r][c].eps;
                }
            }
        }
        Ok(partials)
    }
}

/// Standard 2-sphere manifold $S^2$ of radius $R$ in $(\theta, \varphi)$ coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoSphereMetric {
    /// Sphere radius $R > 0$.
    pub radius: f64,
}

impl TwoSphereMetric {
    /// Constructs a 2-sphere metric with radius $R > 0$.
    #[verified_engine::verified]
    pub fn new(radius: f64) -> Result<Self, TensorError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(TensorError::SingularMetric);
        }
        Ok(Self { radius })
    }

    #[inline]
    fn eval_scalar<S: Scalar>(&self, point: &[S; 2]) -> [[S; 2]; 2] {
        let r = S::from_f64(self.radius);
        let r2 = r * r;
        let st = point[0].sin();
        [[r2, S::ZERO], [S::ZERO, r2 * st * st]]
    }
}

impl StaticMetric<2> for TwoSphereMetric {
    #[verified_engine::verified]
    fn metric_at(&self, point: &SVector<f64, 2>) -> Result<SMatrix<f64, 2, 2>, TensorError> {
        if !self.radius.is_finite() || self.radius <= 0.0 || !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let m = self.eval_scalar(&[point[0], point[1]]);
        Ok(SMatrix::<f64, 2, 2>::new(
            m[0][0], m[0][1], m[1][0], m[1][1],
        ))
    }

    #[verified_engine::verified]
    fn metric_partials_at(
        &self,
        point: &SVector<f64, 2>,
    ) -> Result<[SMatrix<f64, 2, 2>; 2], TensorError> {
        if !self.radius.is_finite() || self.radius <= 0.0 || !point.iter().all(|v| v.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        let m0 = self.eval_scalar(&[Dual::variable(point[0]), Dual::constant(point[1])]);
        let m1 = self.eval_scalar(&[Dual::constant(point[0]), Dual::variable(point[1])]);
        Ok([
            SMatrix::<f64, 2, 2>::new(m0[0][0].eps, m0[0][1].eps, m0[1][0].eps, m0[1][1].eps),
            SMatrix::<f64, 2, 2>::new(m1[0][0].eps, m1[0][1].eps, m1[1][0].eps, m1[1][1].eps),
        ])
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

/// Computes Christoffel symbols $\Gamma^\mu_{\alpha\beta}(x)$ using [`StaticMetric::metric_partials_at`]
/// on stack-allocated [`SVector`] and [`SMatrix`] buffers.
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
/// assert!((exact[0][(1, 1)] - approx[0][(1, 1)]).abs() < 1e-14);
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
    let dg = metric.metric_partials_at(point)?;
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
