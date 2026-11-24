//! Ordinary Differential Equation (ODE) solvers.
//!
//! This module provides a generic `OdeSystem` trait for defining ODEs and solvers
//! like `RungeKutta4` to integrate them numerically.
//!
//! The design relies on the `VectorOperations` trait to allow the solvers to work
//! with any vector-like type (e.g., `Vec<f64>`, `nalgebra::Vector3<f64>`, etc.),
//! avoiding heap allocations when fixed-size arrays are sufficient.

pub mod implicit_midpoint;
#[allow(missing_docs)]
pub mod macros;
#[allow(missing_docs)]
pub mod model;
#[allow(missing_docs)]
pub mod solvers;
#[allow(missing_docs)]
pub mod state;
#[allow(missing_docs)]
pub mod stepper;
pub mod symplectic;
#[allow(missing_docs)]
pub mod traits;

/// Supported numerical integration methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IntegrationMethod {
    /// 4th-order Runge-Kutta method (default).
    #[default]
    RungeKutta4,
    /// Forward Euler method.
    Euler,
    /// Heun's method (Runge-Kutta 2nd order).
    Heun,
    /// 2nd-order symplectic Implicit Midpoint rule.
    ImplicitMidpoint,
    /// 4th-order symplectic 2-stage Gauss-Legendre collocation method.
    GaussLegendre4,
}

impl IntegrationMethod {
    /// Array containing all supported integration methods.
    pub const ALL: [Self; 5] = [
        Self::RungeKutta4,
        Self::Euler,
        Self::Heun,
        Self::ImplicitMidpoint,
        Self::GaussLegendre4,
    ];

    /// Returns the human-readable display name of the integration method.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RungeKutta4 => "Runge-Kutta 4",
            Self::Euler => "Euler",
            Self::Heun => "Heun (RK2)",
            Self::ImplicitMidpoint => "Implicit Midpoint",
            Self::GaussLegendre4 => "Gauss-Legendre 4",
        }
    }

    /// Returns `true` when the method preserves canonical symplectic phase volume.
    #[must_use]
    #[verified_engine::verified]
    pub const fn is_symplectic(&self) -> bool {
        matches!(self, Self::ImplicitMidpoint | Self::GaussLegendre4)
    }
}

pub use implicit_midpoint::{ImplicitMidpoint, MAX_FIXED_POINT_ITERS};
pub use model::OdeModel;
pub use solvers::{Euler, Heun, RungeKutta4};
pub use state::{ArrayState, VecState};
pub use stepper::TimeStepper;
pub use symplectic::{GaussLegendre4, StormerVerlet, SymplecticMethod, Yoshida4};
pub use traits::{
    HamiltonianSystem, OdeSystem, SeparableHamiltonian, Solver, SolverExt, VectorOperations,
};

// [cite:graph_parameters_rust]
