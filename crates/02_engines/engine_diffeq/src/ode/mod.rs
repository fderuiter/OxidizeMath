//! Ordinary Differential Equation (ODE) solvers, state vectors, and time-stepping containers.
// [cite:engine_diffeq]

/// Two-step Backward Differentiation Formula (`Bdf2`) stiff solver.
pub mod bdf2;
/// Explicit fixed-step ODE solvers (`Euler`, `Heun`, `RungeKutta4`).
pub mod explicit;
/// Generic `OdeModel` container tracking elapsed time `t`.
pub mod model;
/// Adaptive Dormand-Prince `RK45` solver with proportional-integral step control.
pub mod rk45;
/// L-stable Rosenbrock-Wanner `ode23s` stiff solver.
pub mod rosenbrock;
/// Stack and optional heap/nalgebra state implementations.
pub mod state;
/// Unified `TimeStepper` interface with defaulted time accessors.
pub mod stepper;
mod stiff_common;
/// Core `VectorOperations`, `OdeState`, `OdeSystem`, and `Solver` traits.
pub mod traits;

use serde::{Deserialize, Serialize};

/// Supported numerical ODE integration methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum IntegrationMethod {
    /// Classic fourth-order Runge-Kutta method (default).
    #[default]
    RungeKutta4,
    /// Embedded adaptive Dormand-Prince 4(5) (`RK45`) method with PI step control.
    DormandPrince45,
    /// L-stable second/third-order Rosenbrock-Wanner (`ode23s`) method for stiff systems.
    Rosenbrock23,
    /// Variable-step two-step Backward Differentiation Formula (`BDF2`) method for stiff systems.
    Bdf2,
    /// Forward Euler first-order explicit method.
    Euler,
    /// Heun's second-order explicit Runge-Kutta method.
    Heun,
}

impl IntegrationMethod {
    /// Array containing all supported ODE integration methods.
    pub const ALL: [Self; 6] = [
        Self::RungeKutta4,
        Self::DormandPrince45,
        Self::Rosenbrock23,
        Self::Bdf2,
        Self::Euler,
        Self::Heun,
    ];

    /// Returns the human-readable display name of the integration method.
    #[must_use]
    #[verified_engine::verified]
    pub const fn name(&self) -> &'static str {
        match *self {
            Self::RungeKutta4 => "Runge-Kutta 4",
            Self::DormandPrince45 => "Dormand-Prince (RK45)",
            Self::Rosenbrock23 => "Rosenbrock 2(3) (Stiff)",
            Self::Bdf2 => "BDF2 (Stiff)",
            Self::Euler => "Euler",
            Self::Heun => "Heun (RK2)",
        }
    }
}

pub use bdf2::Bdf2;
pub use explicit::{Euler, Heun, RungeKutta4, step_slice_rk4};
pub use model::OdeModel;
pub use rk45::{DormandPrince45, step_slice_rk45};
pub use rosenbrock::Rosenbrock23;
pub use state::ArrayState;
#[cfg(feature = "alloc")]
pub use state::VecState;
pub use stepper::TimeStepper;
pub use traits::{OdeState, OdeSystem, SliceOdeSystem, Solver, SolverExt, VectorOperations};
