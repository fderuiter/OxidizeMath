// [cite:tensors]
//! # Tensor Types
//!
//! Core data structures representing various tensor types on a manifold, such
//! as contravariant and covariant vectors, rank-2 tensors, and their associated
//! error states.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Errors related to tensor operations.
#[derive(Error, Debug)]
pub enum TensorError {
    /// Indicates that an operation was attempted with tensors or vectors of incompatible dimensions.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The dimension size expected by the operation.
        expected: usize,
        /// The actual dimension size provided.
        got: usize,
    },
    /// Indicates that the metric tensor determinant is zero, preventing inversion.
    #[error("Singular metric tensor, cannot invert")]
    SingularMetric,
    /// Indicates an attempt to access an element beyond the allocated bounds.
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error(transparent)]
    #[allow(missing_docs)]
    Math(#[from] math_commons::error::MathError),
}

/// A contravariant vector ($A^\mu$).
///
/// Components transform like coordinate differentials.
#[derive(Debug, Clone, PartialEq)]
pub struct ContravariantVector(pub DVector<f64>);

impl ContravariantVector {
    /// Creates a new contravariant vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use pure_math::pure_math::tensor::ContravariantVector;
    /// use nalgebra::DVector;
    ///
    /// let vec = ContravariantVector::new(DVector::from_vec(vec![1.0, 2.0]));
    /// assert_eq!(vec.dim(), 2);
    /// ```
    #[verified_engine::verified]
    pub fn new(data: DVector<f64>) -> Self {
        debug_assert!(!data.is_empty());
        Self(data)
    }

    /// Returns the dimensionality (number of components) of the vector.
    #[verified_engine::verified]
    pub fn dim(&self) -> usize {
        let d = self.0.len();
        debug_assert!(d > 0);
        d
    }
}

/// A covariant vector ($A_\mu$).
///
/// Components transform inversely to contravariant vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct CovariantVector(pub DVector<f64>);

impl CovariantVector {
    /// Creates a new covariant vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use pure_math::pure_math::tensor::CovariantVector;
    /// use nalgebra::DVector;
    ///
    /// let vec = CovariantVector::new(DVector::from_vec(vec![3.0, 4.0, 5.0]));
    /// assert_eq!(vec.dim(), 3);
    /// ```
    #[verified_engine::verified]
    pub fn new(data: DVector<f64>) -> Self {
        debug_assert!(!data.is_empty());
        Self(data)
    }

    /// Returns the dimensionality (number of components) of the vector.
    #[verified_engine::verified]
    pub fn dim(&self) -> usize {
        let d = self.0.len();
        debug_assert!(d > 0);
        d
    }
}

/// A rank-2 tensor (can be covariant, contravariant, or mixed).
///
/// For now, we represent it as a matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Rank2Tensor(pub DMatrix<f64>);

impl Rank2Tensor {
    /// Creates a new rank-2 tensor from a given matrix representing its components.
    ///
    /// # Examples
    ///
    /// ```
    /// use pure_math::pure_math::tensor::types::Rank2Tensor;
    /// use nalgebra::DMatrix;
    ///
    /// let tensor = Rank2Tensor::new(DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]));
    /// assert_eq!(tensor.0.nrows(), 2);
    /// ```
    #[verified_engine::verified]
    pub fn new(data: DMatrix<f64>) -> Self {
        debug_assert!(data.nrows() > 0 && data.ncols() > 0);
        Self(data)
    }
}
