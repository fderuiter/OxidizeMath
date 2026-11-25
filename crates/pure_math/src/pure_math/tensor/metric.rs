use super::types::{ContravariantVector, CovariantVector, TensorError};
use math_scalars::Dual;
use nalgebra::{DMatrix, DVector};

/// A trait representing a metric tensor field $g_{ij}(x)$.
pub trait Metric {
    /// Computes the covariant metric tensor $g_{ij}$ at a given point.
    ///
    /// # Errors
    /// Returns a `TensorError` if the metric calculation fails for the given point.
    #[verified_engine::verified]
    fn metric_at(&self, point: &DVector<f64>) -> Result<DMatrix<f64>, TensorError>;

    /// Computes coordinate partial derivatives $\partial_k g_{ij}(x)$ as `result[k][(i, j)]`.
    ///
    /// Built-in [`AdRiemannianMetric`] overrides this with exact forward-mode [`Dual<f64>`]
    /// tangents; custom closure metrics fall back to scale-aware $\varepsilon^{1/3}$ central differences.
    #[verified_engine::verified]
    fn metric_partials_at(&self, point: &DVector<f64>) -> Result<Vec<DMatrix<f64>>, TensorError> {
        let dim = point.len();
        let mut partial_g = vec![DMatrix::zeros(dim, dim); dim];
        let mut p_eval = point.clone();
        for (k, slot) in partial_g.iter_mut().enumerate().take(dim) {
            let h = f64::EPSILON.cbrt() * point[k].abs().max(1.0);
            let x_plus = point[k] + h;
            let x_minus = point[k] - h;
            let span = x_plus - x_minus;
            p_eval[k] = x_plus;
            let g_plus = self.metric_at(&p_eval)?;
            p_eval[k] = x_minus;
            let g_minus = self.metric_at(&p_eval)?;
            p_eval[k] = point[k];
            *slot = (g_plus - g_minus) / span;
        }
        Ok(partial_g)
    }

    /// Computes the inverse (contravariant) metric tensor $g^{ij}$ at a given point.
    #[verified_engine::verified]
    fn inverse_metric_at(&self, point: &DVector<f64>) -> Result<DMatrix<f64>, TensorError> {
        debug_assert!(!point.is_empty());
        let g_cov = self.metric_at(point)?;
        g_cov.try_inverse().ok_or(TensorError::SingularMetric)
    }

    /// Lowers the index of a contravariant vector: $A_i = g_{ij} A^j$.
    #[verified_engine::verified]
    fn lower_index(
        &self,
        vec: &ContravariantVector,
        point: &DVector<f64>,
    ) -> Result<CovariantVector, TensorError> {
        debug_assert!(!point.is_empty());
        let g = self.metric_at(point)?;
        if g.nrows() != vec.dim() || g.ncols() != vec.dim() {
            return Err(crate::pure_math::tensor::TensorError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(g.nrows()),
                    actual: math_commons::math_kernel::types::Dimension(vec.dim()),
                },
            ));
        }
        Ok(CovariantVector::new(&g * &vec.0))
    }

    /// Raises the index of a covariant vector: $A^i = g^{ij} A_j$.
    #[verified_engine::verified]
    fn raise_index(
        &self,
        vec: &CovariantVector,
        point: &DVector<f64>,
    ) -> Result<ContravariantVector, TensorError> {
        debug_assert!(!point.is_empty());
        let g_inv = self.inverse_metric_at(point)?;
        if g_inv.nrows() != vec.dim() || g_inv.ncols() != vec.dim() {
            return Err(crate::pure_math::tensor::TensorError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(g_inv.nrows()),
                    actual: math_commons::math_kernel::types::Dimension(vec.dim()),
                },
            ));
        }
        Ok(ContravariantVector::new(&g_inv * &vec.0))
    }
}

/// A generic implementation of a Riemannian metric defined by a closure.
pub struct RiemannianMetric<F>
where
    F: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    metric_fn: F,
}

impl<F> RiemannianMetric<F>
where
    F: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    /// Wraps a coordinate-dependent metric-tensor closure `metric_fn`.
    #[verified_engine::verified]
    pub fn new(metric_fn: F) -> Self {
        debug_assert!(std::mem::size_of::<Self>() == std::mem::size_of::<F>());
        Self { metric_fn }
    }
}

impl<F> Metric for RiemannianMetric<F>
where
    F: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    #[verified_engine::verified]
    fn metric_at(&self, point: &DVector<f64>) -> Result<DMatrix<f64>, TensorError> {
        debug_assert!(!point.is_empty());
        Ok((self.metric_fn)(point))
    }
}

/// Forward-mode [`Dual<f64>`]-differentiable Riemannian metric evaluating exact $g_{ij}(x)$ and
/// $\partial_k g_{ij}(x)$ without finite-difference truncation or cancellation.
///
/// Closure `metric_fn` receives `&[Dual<f64>]` of length `dim` and returns row-major
/// `Vec<Dual<f64>>` of length `dim * dim` (`g[(i, j)] = out[i * dim + j]`).
pub struct AdRiemannianMetric<F>
where
    F: Fn(&[Dual<f64>]) -> Vec<Dual<f64>>,
{
    metric_fn: F,
}

impl<F> AdRiemannianMetric<F>
where
    F: Fn(&[Dual<f64>]) -> Vec<Dual<f64>>,
{
    /// Wraps a forward-AD metric evaluator returning row-major `dim * dim` [`Dual<f64>`] entries.
    #[verified_engine::verified]
    pub const fn new(metric_fn: F) -> Self {
        Self { metric_fn }
    }
}

impl<F> Metric for AdRiemannianMetric<F>
where
    F: Fn(&[Dual<f64>]) -> Vec<Dual<f64>>,
{
    #[verified_engine::verified]
    fn metric_at(&self, point: &DVector<f64>) -> Result<DMatrix<f64>, TensorError> {
        let dim = point.len();
        let coords: Vec<Dual<f64>> = point.iter().map(|&v| Dual::constant(v)).collect();
        let entries = (self.metric_fn)(&coords);
        if entries.len() != dim * dim || !entries.iter().all(|d| d.val.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        Ok(DMatrix::from_fn(dim, dim, |i, j| {
            entries[math_commons::math_kernel::types::flatten_2d_index(i, j, dim)].val
        }))
    }

    #[verified_engine::verified]
    fn metric_partials_at(&self, point: &DVector<f64>) -> Result<Vec<DMatrix<f64>>, TensorError> {
        let dim = point.len();
        let mut coords: Vec<Dual<f64>> = point.iter().map(|&v| Dual::constant(v)).collect();
        let mut partials = Vec::with_capacity(dim);
        for k in 0..dim {
            coords[k] = Dual::variable(point[k]);
            let entries = (self.metric_fn)(&coords);
            coords[k] = Dual::constant(point[k]);
            if entries.len() != dim * dim || !entries.iter().all(|d| d.eps.is_finite()) {
                return Err(TensorError::SingularMetric);
            }
            partials.push(DMatrix::from_fn(dim, dim, |i, j| {
                entries[math_commons::math_kernel::types::flatten_2d_index(i, j, dim)].eps
            }));
        }
        Ok(partials)
    }
}
