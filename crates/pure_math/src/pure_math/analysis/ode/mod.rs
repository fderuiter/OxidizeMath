//! Ordinary Differential Equation (ODE) solvers.
//!
//! This module provides a generic `OdeSystem` trait for defining ODEs and solvers
//! like `RungeKutta4` to integrate them numerically.
//!
//! The design relies on the `VectorOperations` trait to allow the solvers to work
//! with any vector-like type (e.g., `Vec<f64>`, `nalgebra::Vector3<f64>`, etc.),
//! avoiding heap allocations when fixed-size arrays are sufficient.

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
}

impl IntegrationMethod {
    /// Array containing all supported integration methods.
    pub const ALL: [Self; 3] = [Self::RungeKutta4, Self::Euler, Self::Heun];

    /// Returns the human-readable display name of the integration method.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RungeKutta4 => "Runge-Kutta 4",
            Self::Euler => "Euler",
            Self::Heun => "Heun (RK2)",
        }
    }
}

pub use model::OdeModel;
pub use solvers::{Euler, Heun, RungeKutta4};
pub use state::{ArrayState, VecState};
pub use stepper::TimeStepper;
pub use symplectic::{GaussLegendre4, StormerVerlet, SymplecticMethod, Yoshida4};
pub use traits::{
    HamiltonianSystem, OdeSystem, SeparableHamiltonian, Solver, SolverExt, VectorOperations,
};

// [cite:graph_parameters_rust]
