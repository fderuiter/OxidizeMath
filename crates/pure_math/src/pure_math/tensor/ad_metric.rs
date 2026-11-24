// [cite:general_relativity_geodesics]
//! Forward-mode automatic differentiation metric adapters and covariant derivative operators.

use math_commons::math_kernel::types::flatten_2d_index;
use math_scalars::{Dual, DualVec, Scalar};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

use super::christoffel::christoffel_symbols;
use super::metric::Metric;
use super::static_metric::StaticMetric;
use super::types::{Rank2Tensor, TensorError};

/// Dynamic-dimension Riemannian metric evaluated over generic `Dual<f64>` coordinates.
///
/// Computes exact `Dual<f64>` metric partial derivatives $\partial_k g_{ij}(x)$ without
/// finite-difference truncation or horizon stencil breakdown.
pub struct AdRiemannianMetric<F> {
    metric_fn: F,
}

impl<F> AdRiemannianMetric<F>
where
    F: Fn(&[Dual<f64>]) -> Result<Vec<Dual<f64>>, TensorError>,
{
    /// Wraps a row-major `dim * dim` dual metric callback.
    #[must_use]
    pub const fn new(metric_fn: F) -> Self {
        Self { metric_fn }
    }
}

impl<F> Metric for AdRiemannianMetric<F>
where
    F: Fn(&[Dual<f64>]) -> Result<Vec<Dual<f64>>, TensorError>,
{
    #[verified_engine::verified]
    fn metric_at(&self, point: &DVector<f64>) -> Result<DMatrix<f64>, TensorError> {
        let dim = point.len();
        debug_assert!(dim > 0);
        let const_pt: Vec<Dual<f64>> = point.iter().copied().map(Dual::constant).collect();
        let raw = (self.metric_fn)(&const_pt)?;
        if raw.len() != dim * dim || !raw.iter().all(|v| v.val.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        Ok(DMatrix::from_fn(dim, dim, |i, j| {
            raw[flatten_2d_index(j, i, dim)].val
        }))
    }

    fn metric_partials(
        &self,
        point: &DVector<f64>,
    ) -> Option<Result<Vec<DMatrix<f64>>, TensorError>> {
        let dim = point.len();
        let mut seeded: Vec<Dual<f64>> = point.iter().copied().map(Dual::constant).collect();
        let mut partials = Vec::with_capacity(dim);
        for k in 0..dim {
            seeded[k] = Dual::variable(point[k]);
            let raw = match (self.metric_fn)(&seeded) {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };
            seeded[k] = Dual::constant(point[k]);
            if raw.len() != dim * dim || !raw.iter().all(|v| v.is_finite_val()) {
                return Some(Err(TensorError::SingularMetric));
            }
            partials.push(DMatrix::from_fn(dim, dim, |i, j| {
                raw[flatten_2d_index(j, i, dim)].eps
            }));
        }
        Some(Ok(partials))
    }
}

/// Zero-allocation fixed-dimension metric adapter evaluating all partial derivatives
/// $\partial_\sigma g_{\mu\nu}(x)$ in a single `DualVec<f64, N>` forward pass.
#[derive(Debug, Clone, Copy)]
pub struct AdStaticRiemannianMetric<F, const N: usize> {
    metric_fn: F,
}

impl<F, const N: usize> AdStaticRiemannianMetric<F, N>
where
    F: Fn(&[DualVec<f64, N>; N]) -> Result<[[DualVec<f64, N>; N]; N], TensorError>,
{
    /// Wraps a stack-allocated `DualVec<f64, N>` metric evaluation closure.
    #[must_use]
    pub const fn new(metric_fn: F) -> Self {
        Self { metric_fn }
    }
}

impl<F, const N: usize> StaticMetric<N> for AdStaticRiemannianMetric<F, N>
where
    F: Fn(&[DualVec<f64, N>; N]) -> Result<[[DualVec<f64, N>; N]; N], TensorError>,
{
    #[verified_engine::verified]
    fn metric_at(&self, point: &SVector<f64, N>) -> Result<SMatrix<f64, N, N>, TensorError> {
        let mut pt = [DualVec::<f64, N>::ZERO; N];
        for i in 0..N {
            pt[i] = DualVec::constant(point[i]);
        }
        let raw = (self.metric_fn)(&pt)?;
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

    fn metric_partials_at(
        &self,
        point: &SVector<f64, N>,
    ) -> Option<Result<[SMatrix<f64, N, N>; N], TensorError>> {
        let mut seeded = [DualVec::<f64, N>::ZERO; N];
        for i in 0..N {
            seeded[i] = DualVec::variable(point[i], i);
        }
        let raw = match (self.metric_fn)(&seeded) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };
        let mut partials = [SMatrix::<f64, N, N>::zeros(); N];
        for r in 0..N {
            for c in 0..N {
                if !raw[r][c].is_finite_val() {
                    return Some(Err(TensorError::SingularMetric));
                }
                for (sigma, slot) in partials.iter_mut().enumerate() {
                    slot[(r, c)] = raw[r][c].eps[sigma];
                }
            }
        }
        Some(Ok(partials))
    }
}

/// Computes the exact covariant derivative $A^i_{;j} = \partial_j A^i + \Gamma^i_{kj} A^k$
/// using `Dual<f64>` forward-mode partials and exact metric connection symbols.
#[verified_engine::verified]
pub fn covariant_derivative_contravariant_ad<F>(
    field: F,
    metric: &impl Metric,
    point: &DVector<f64>,
) -> Result<Rank2Tensor, TensorError>
where
    F: Fn(&[Dual<f64>]) -> Vec<Dual<f64>>,
{
    let dim = point.len();
    let gammas = christoffel_symbols(metric, point)?;
    let mut seeded: Vec<Dual<f64>> = point.iter().copied().map(Dual::constant).collect();
    let base = field(&seeded);
    let mut partials = DMatrix::zeros(dim, dim);
    for j in 0..dim {
        seeded[j] = Dual::variable(point[j]);
        let col = field(&seeded);
        seeded[j] = Dual::constant(point[j]);
        for i in 0..dim {
            partials[(i, j)] = col[i].eps;
        }
    }
    let mut out = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut conn = 0.0;
            for k in 0..dim {
                conn += gammas[i][(k, j)] * base[k].val;
            }
            out[(i, j)] = partials[(i, j)] + conn;
        }
    }
    Ok(Rank2Tensor(out))
}

/// Computes the exact covariant derivative $A_{i;j} = \partial_j A_i - \Gamma^k_{ij} A_k$
/// using `Dual<f64>` forward-mode partials and exact metric connection symbols.
#[verified_engine::verified]
pub fn covariant_derivative_covariant_ad<F>(
    field: F,
    metric: &impl Metric,
    point: &DVector<f64>,
) -> Result<Rank2Tensor, TensorError>
where
    F: Fn(&[Dual<f64>]) -> Vec<Dual<f64>>,
{
    let dim = point.len();
    let gammas = christoffel_symbols(metric, point)?;
    let mut seeded: Vec<Dual<f64>> = point.iter().copied().map(Dual::constant).collect();
    let base = field(&seeded);
    let mut partials = DMatrix::zeros(dim, dim);
    for j in 0..dim {
        seeded[j] = Dual::variable(point[j]);
        let col = field(&seeded);
        seeded[j] = Dual::constant(point[j]);
        for i in 0..dim {
            partials[(i, j)] = col[i].eps;
        }
    }
    let mut out = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut conn = 0.0;
            for (k, gamma_k) in gammas.iter().enumerate().take(dim) {
                conn += gamma_k[(i, j)] * base[k].val;
            }
            out[(i, j)] = partials[(i, j)] - conn;
        }
    }
    Ok(Rank2Tensor(out))
}
