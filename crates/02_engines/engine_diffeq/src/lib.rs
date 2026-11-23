#![no_std]
#![doc = include_str!("../README.md")]
// [cite:engine_diffeq]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

/// Typed `Copy` error definitions for ODE, quadrature, and Hermite routines.
pub mod error;
/// Safeguarded cubic Hermite interpolation and Newton-bisection root-bracketing.
pub mod hermite;
/// Pure-`core` IEEE-754 elementary math helpers (`sqrt`, `hypot`, `ln`, `exp`, `pow`).
pub mod math;
/// Explicit, adaptive (`DormandPrince45`), and stiff (`Rosenbrock23`, `Bdf2`) ODE solvers.
pub mod ode;
/// Composite (`Trapezoidal`, `Simpson`) and adaptive (`GaussKronrod15`) quadrature rules.
pub mod quadrature;
/// Caller-owned slice and stack `SolverWorkspace` buffers plus in-place partial-pivoting LU.
pub mod workspace;

pub use error::DiffeqError;
pub use hermite::{
    clamp_fritsch_carlson_slopes, cubic_hermite_scalar, cubic_hermite_slope,
    cubic_hermite_val_deriv, solve_hermite_root, solve_hermite_unit_bracket,
    solve_monotone_hermite_crossing,
};
#[cfg(feature = "alloc")]
pub use ode::VecState;
pub use ode::{
    ArrayState, Bdf2, DormandPrince45, Euler, Heun, IntegrationMethod, OdeModel, OdeState,
    OdeSystem, Rosenbrock23, RungeKutta4, SliceOdeSystem, Solver, SolverExt, TimeStepper,
    VectorOperations, step_slice_rk4, step_slice_rk45,
};
pub use quadrature::{
    GK15_STACK_SUBINTERVALS, GaussKronrod15, IntegrationResult, Integrator, Simpson, Trapezoidal,
    gk15_panel, trapezoidal_nonuniform, trapezoidal_uniform_grid, trapezoidal_uniform_mapped,
};
pub use workspace::{SolverWorkspace, StackWorkspace, lu_factor_in_place, lu_solve_in_place};

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[cfg(test)]
mod tests_ode;
#[cfg(test)]
mod tests_quad_hermite;
