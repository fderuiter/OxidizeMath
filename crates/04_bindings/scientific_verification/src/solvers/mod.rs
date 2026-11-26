/// Epidemiology, computational biology, applied PDE/game-theory, AI, and climate solver certificates (15 solvers).
pub mod bio_epi_applied_ai;
/// Potential-flow tracer RK4, SU(1,1) hyperbolic billiards, Platt Newton-Raphson, and PAVA isotonic certificates (4 solvers).
pub mod calibration_and_hyperbolic;
/// Linear algebra, OLS, non-autonomous ODE, and symplectic phase-space solver certificates (12 solvers).
pub mod commons_and_ode;
mod neuro_applied_ai;
/// Astrophysics, GR, chaos, optics, fluid dynamics, MRI, quantum, and solid-state solver certificates (13 solvers).
pub mod physics;
/// 1D/2D PDE, differential-geometry curvature flow, geodesic, and OU solver certificates (11 solvers).
pub mod pure_pde_and_geom;

use crate::mms::SolverCertificate;

/// Executes manufactured-solution, exact-eigenmode, and grid-refinement convergence checks across
/// all 55 workspace numerical solvers, returning all 55 [`SolverCertificate`] records or an error
/// describing the first failing solver.
pub fn verify_all_solvers() -> Result<Vec<SolverCertificate>, String> {
    let mut certs = Vec::with_capacity(55);
    certs.extend(commons_and_ode::certify_commons_and_ode()?);
    certs.extend(pure_pde_and_geom::certify_pure_pde_and_geom()?);
    certs.extend(physics::certify_physics_solvers()?);
    certs.extend(bio_epi_applied_ai::certify_bio_epi_applied_ai()?);
    certs.extend(calibration_and_hyperbolic::certify_calibration_and_hyperbolic()?);

    for cert in &certs {
        if !cert.passed {
            return Err(format!(
                "Solver `{}` ({}) failed MMS convergence: observed_order={:.3} (min {:.3}), fine_error={:.3e} (max {:.3e})",
                cert.name,
                cert.crate_name,
                cert.observed_order,
                cert.expected_min_order,
                cert.fine_error,
                cert.max_allowed_error
            ));
        }
    }
    Ok(certs)
}
