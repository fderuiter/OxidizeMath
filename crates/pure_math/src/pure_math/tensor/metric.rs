use super::types::{ContravariantVector, CovariantVector, TensorError};
use engine_symbolic::Dual;
use nalgebra::{DMatrix, DVector};

/// A trait representing a metric tensor field $g_{ij}(x)$.
pub trait Metric {
    /// Computes the covariant metric tensor $g_{ij}$ at a given point.
    ///
    /// # Errors
    ///
    /// Returns a `TensorError` if the metric calculation fails for the given point.
    #[verified_engine::verified]
    fn metric_at(&self, point: &DVector<f64>) -> Result<DMatrix<f64>, TensorError>;

    /// Evaluates the covariant metric tensor $g_{ij}(x)$ over forward-mode [`Dual<f64>`]
    /// coordinates when analytical `Dual` lifting is supported by the metric.
    #[verified_engine::verified]
    fn metric_dual_at(
        &self,
        _point: &[Dual<f64>],
    ) -> Option<Result<DMatrix<Dual<f64>>, TensorError>> {
        None
    }

    /// Computes the inverse (contravariant) metric tensor $g^{ij}$ at a given point.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::SingularMetric` if the covariant metric tensor is not invertible.
    /// May also return a `TensorError` propagated from `metric_at`.
    #[verified_engine::verified]
    fn inverse_metric_at(&self, point: &DVector<f64>) -> Result<DMatrix<f64>, TensorError> {
        debug_assert!(!point.is_empty());
        let g_cov = self.metric_at(point)?;
        g_cov.try_inverse().ok_or(TensorError::SingularMetric)
    }

    /// Lowers the index of a contravariant vector: $A_i = g_{ij} A^j$.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::DimensionMismatch` if the vector's dimension does not match the metric's dimensions.
    /// May also return a `TensorError` propagated from `metric_at`.
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
        let lowered = &g * &vec.0;
        Ok(CovariantVector::new(lowered))
    }

    /// Raises the index of a covariant vector: $A^i = g^{ij} A_j$.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::DimensionMismatch` if the vector's dimension does not match the inverse metric's dimensions.
    /// May also return a `TensorError` propagated from `inverse_metric_at`.
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
        let raised = &g_inv * &vec.0;
        Ok(ContravariantVector::new(raised))
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
    #[allow(missing_docs)]
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

/// Forward-mode [`Dual<f64>`]-lifted Riemannian metric evaluating exact metric partial derivatives
/// without finite-difference truncation or horizon-crossing stencils.
pub struct DualRiemannianMetric<F>
where
    F: Fn(&[Dual<f64>]) -> DMatrix<Dual<f64>>,
{
    metric_fn: F,
}

impl<F> DualRiemannianMetric<F>
where
    F: Fn(&[Dual<f64>]) -> DMatrix<Dual<f64>>,
{
    /// Wraps a forward-mode `Dual<f64>` metric tensor closure.
    #[verified_engine::verified]
    pub fn new(metric_fn: F) -> Self {
        Self { metric_fn }
    }
}

impl<F> Metric for DualRiemannianMetric<F>
where
    F: Fn(&[Dual<f64>]) -> DMatrix<Dual<f64>>,
{
    #[verified_engine::verified]
    fn metric_at(&self, point: &DVector<f64>) -> Result<DMatrix<f64>, TensorError> {
        let dual_pt: Vec<Dual<f64>> = point.iter().copied().map(Dual::constant).collect();
        let g_dual = (self.metric_fn)(&dual_pt);
        Ok(DMatrix::from_fn(g_dual.nrows(), g_dual.ncols(), |r, c| {
            g_dual[(r, c)].re
        }))
    }

    #[verified_engine::verified]
    fn metric_dual_at(
        &self,
        point: &[Dual<f64>],
    ) -> Option<Result<DMatrix<Dual<f64>>, TensorError>> {
        Some(Ok((self.metric_fn)(point)))
    }
}
