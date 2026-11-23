//! Ordinary Differential Equation (ODE) solvers.
//!
//! This module provides a generic `OdeSystem` trait for defining ODEs and solvers
//! like `RungeKutta4` to integrate them numerically.
//!
//! The design relies on the `VectorOperations` trait to allow the solvers to work
//! with any vector-like type (e.g., `Vec<f64>`, `nalgebra::Vector3<f64>`, etc.),
//! avoiding heap allocations when fixed-size arrays are sufficient.

pub mod dormand_prince;
pub mod implicit;
#[allow(missing_docs)]
pub mod macros;
#[allow(missing_docs)]
pub mod model;
pub mod rosenbrock;
#[allow(missing_docs)]
pub mod solvers;
#[allow(missing_docs)]
pub mod state;
#[allow(missing_docs)]
pub mod stepper;
#[allow(missing_docs)]
pub mod traits;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

/// Supported numerical integration methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum IntegrationMethod {
    /// 4th-order Runge-Kutta method (default).
    #[default]
    RungeKutta4,
    /// Forward Euler method.
    Euler,
    /// Heun's method (Runge-Kutta 2nd order).
    Heun,
    /// Adaptive 5th(4th)-order Dormand-Prince Runge-Kutta method (`ode45`).
    DormandPrince45,
    /// 1st-order L-stable implicit Backward Euler method.
    BackwardEuler,
    /// 2nd-order A-stable Implicit Trapezoidal (Crank-Nicolson) method.
    ImplicitTrapezoidal,
    /// Adaptive L-stable 2nd(3rd)-order linearly implicit Rosenbrock method (`ode23s`).
    Rosenbrock23,
}

impl IntegrationMethod {
    /// Array containing all supported integration methods.
    pub const ALL: [Self; 7] = [
        Self::RungeKutta4,
        Self::Euler,
        Self::Heun,
        Self::DormandPrince45,
        Self::BackwardEuler,
        Self::ImplicitTrapezoidal,
        Self::Rosenbrock23,
    ];

    /// Returns the human-readable display name of the integration method.
    #[verified_engine::verified]
    pub fn name(&self) -> &'static str {
        match self {
            Self::RungeKutta4 => "Runge-Kutta 4",
            Self::Euler => "Euler",
            Self::Heun => "Heun (RK2)",
            Self::DormandPrince45 => "Dormand-Prince 4(5)",
            Self::BackwardEuler => "Backward Euler",
            Self::ImplicitTrapezoidal => "Implicit Trapezoidal",
            Self::Rosenbrock23 => "Rosenbrock 2(3)",
        }
    }

    /// Returns `true` when the solver uses embedded local truncation error control.
    #[verified_engine::verified]
    pub fn is_adaptive(&self) -> bool {
        matches!(self, Self::DormandPrince45 | Self::Rosenbrock23)
    }

    /// Returns `true` when the solver solves implicit or linearly-implicit stiff stage systems.
    #[verified_engine::verified]
    pub fn is_implicit(&self) -> bool {
        matches!(
            self,
            Self::BackwardEuler | Self::ImplicitTrapezoidal | Self::Rosenbrock23
        )
    }
}

pub use dormand_prince::DormandPrince45;
pub use implicit::{BackwardEuler, ImplicitTrapezoidal};
pub use model::OdeModel;
pub use rosenbrock::Rosenbrock23;
pub use solvers::{Euler, Heun, RungeKutta4};
pub use state::{ArrayState, VecState};
pub use stepper::TimeStepper;
pub use traits::{IndexedState, OdeSystem, Solver, SolverExt, VectorOperations};

// [cite:graph_parameters_rust]
