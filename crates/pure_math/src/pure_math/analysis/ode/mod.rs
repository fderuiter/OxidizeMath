//! Ordinary Differential Equation (ODE) solvers.
//!
//! Re-exports the `#![no_std]` ODE engine (`engine_diffeq::ode`) alongside symplectic
//! integrators for conservative Hamiltonian systems.

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

pub use engine_diffeq::ode::IntegrationMethod;
pub use engine_diffeq::{DiffeqError, SolverWorkspace, StackWorkspace};
pub use model::OdeModel;
pub use solvers::{
    Bdf2, DormandPrince45, Euler, Heun, Rosenbrock23, RungeKutta4, step_slice_rk4, step_slice_rk45,
};
pub use state::{ArrayState, VecState};
pub use stepper::TimeStepper;
pub use symplectic::{GaussLegendre4, StormerVerlet, SymplecticMethod, Yoshida4};
pub use traits::{
    HamiltonianSystem, OdeState, OdeSystem, SeparableHamiltonian, SliceOdeSystem, Solver,
    SolverExt, VectorOperations,
};

// [cite:graph_parameters_rust]
