//! Mathematical Optimization Algorithms and Preallocated Solver Suite.
//!
//! Organizes first-order gradient optimizers (`SGD`, `Adam`), proximal `L1` solvers
//! (`ISTA`, `FISTA`), two-phase Simplex linear programming, active-set convex
//! quadratic programming, box-constrained `L-BFGS-B`, projected gradient,
//! augmented Lagrangian nonlinear solvers, derivative-free `NelderMead` / `BrentMinimizer`,
//! and multi-objective Pareto frontier extraction.

pub mod augmented_lagrangian;
mod brent;
pub mod derivative_free;
pub mod first_order;
pub mod lbfgsb;
pub mod linear_programming;
mod lp_tableau;
pub mod pareto;
pub mod projected_gradient;
pub mod proximal;
mod qp_kkt;
pub mod quadratic_programming;

use super::evolution::EvolutionError;
use super::roots::AnalysisError;
use nalgebra::{DMatrix, DVector, RealField};
use thiserror::Error;

pub use super::roots::{BrentMinimizer, OneDimMinimizer};
pub use augmented_lagrangian::{
    AugmentedLagrangianConfig, AugmentedLagrangianSolution, AugmentedLagrangianSolver,
    AugmentedLagrangianWorkspace,
};
pub use derivative_free::{NelderMead, NelderMeadConfig, NelderMeadSolution, NelderMeadWorkspace};
pub use first_order::{Adam, AdamState, AdamState2D, SGD};
pub use lbfgsb::{LbfgsBConfig, LbfgsBSolver, LbfgsBWorkspace};
pub use linear_programming::{
    ConstraintSense, LinearProgram, LpSense, LpSolution, LpStatus, SimplexSolver, SimplexWorkspace,
};
pub use pareto::{
    ParetoPoint, ParetoWorkspace, dominates, epsilon_constraint_sweep, extract_pareto_frontier,
    weighted_sum_sweep,
};
pub use projected_gradient::{
    BoxBounds, ConstrainedSolution, ProjectedGradientConfig, ProjectedGradientSolver,
    ProjectedGradientWorkspace,
};
pub use proximal::{
    L1RegularizedLeastSquares, ProximalConfig, ProximalSolution, ProximalWorkspace, soft_threshold,
    soft_threshold_in_place, soft_threshold_scalar,
};
pub use quadratic_programming::{
    ActiveSetQpSolver, QpSolution, QpStatus, QpWorkspace, QuadraticProgram,
};

/// Errors that can occur during mathematical optimization.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OptimizationError {
    /// Failed to convert a generic scalar constant.
    #[error("Failed to convert numerical value")]
    ConversionError,
    /// Wrapped evolution engine error preserving causal source.
    #[error("Evolution solver error: {0}")]
    Evolution(#[from] EvolutionError),
    /// Wrapped 1D root/minimization analysis error preserving causal source.
    #[error("Scalar analysis error: {0}")]
    Analysis(#[from] AnalysisError),
    /// Vector or matrix dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected vector/matrix dimension.
        expected: usize,
        /// Actual vector/matrix dimension provided.
        actual: usize,
    },
    /// Invalid solver or problem configuration parameter.
    #[error("Invalid optimization parameter: {0}")]
    InvalidParameter(&'static str),
    /// Lower bound exceeds upper bound or bounds are non-finite.
    #[error("Invalid box bounds: lower bound exceeds upper bound or is non-finite")]
    InvalidBounds,
    /// Objective or gradient evaluated to NaN or infinity.
    #[error("Encountered non-finite objective, constraint, or gradient value")]
    NonFiniteValue,
    /// Linear KKT or basis subproblem was singular.
    #[error("Singular linear system encountered during optimization")]
    SingularSystem,
    /// Iteration cap reached prior to satisfying convergence criteria.
    #[error("Optimization exceeded maximum iterations ({0})")]
    MaxIterationsExceeded(usize),
}

/// Strategy for updating model parameters in-place.
pub trait Optimizer<T: RealField + Copy, Key = u64> {
    /// Updates a matrix parameter in-place given its gradient.
    fn update_matrix(
        &mut self,
        key: Key,
        param: &mut DMatrix<T>,
        grad: &DMatrix<T>,
    ) -> Result<(), OptimizationError>;

    /// Updates a vector parameter in-place given its gradient.
    fn update_vector(
        &mut self,
        key: Key,
        param: &mut DVector<T>,
        grad: &DVector<T>,
    ) -> Result<(), OptimizationError>;
}

/// Identifies the type of neural/parametric model tensor being updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamType {
    /// Multiplicative weight matrix.
    Weight,
    /// Additive bias vector.
    Bias,
}

/// Adapter trait for layered model optimizers keyed by `(layer_index, ParamType)`.
pub trait ModelOptimizer<T: RealField + Copy>: Optimizer<T, (usize, ParamType)> {}

impl<T: RealField + Copy, O> ModelOptimizer<T> for O where O: Optimizer<T, (usize, ParamType)> {}

/// Defines a trainable model that performs forward passes and in-place backpropagation updates.
pub trait Trainable<T: nalgebra::RealField + Copy = f64> {
    /// Performs a forward pass through the model and returns output activations/logits.
    #[verified_engine::verified]
    fn forward(&self, x: &nalgebra::DVector<T>) -> nalgebra::DVector<T>;

    /// Performs the backward pass and updates parameters via `optimizer`.
    ///
    /// # Errors
    /// Returns [`OptimizationError`] if any parameter update fails.
    #[verified_engine::verified]
    fn backward_update(
        &mut self,
        x: &nalgebra::DVector<T>,
        loss_grad: &nalgebra::DVector<T>,
        optimizer: &mut dyn ModelOptimizer<T>,
    ) -> Result<(), OptimizationError>;
}
