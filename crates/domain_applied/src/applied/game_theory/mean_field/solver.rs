use super::physics::{Hamiltonian, QuadraticHamiltonian};
use super::types::{Density, MFGConfig, Position};
use math_commons::math_kernel::types::StepSize;
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::analysis::pde::fused_stepper::FusedStencilStepper;

/// Strategy trait for solving Mean Field Games.
///
/// Implementors of this trait provide a specific algorithm (e.g., Fixed Point, Newton-Raphson, Deep Learning)
/// to solve the coupled HJB-FP system defined by the `MFGConfig`.
pub trait MFGSolver {
    /// Solves the Mean Field Game coupled system.
    ///
    /// # Parameters
    /// - `config`: The physical and grid configuration.
    /// - `cost_function`: $F(x, m)$ - Running cost.
    /// - `terminal_cost`: $G(x, m)$ - Terminal cost.
    /// - `initial_distribution`: $m_0(x)$ - Initial population distribution.
    ///
    /// # Returns
    /// Tuple `(u, m)` containing the value function and distribution matrices.
    #[verified_engine::verified]
    fn solve(
        &self,
        config: &MFGConfig,
        cost_function: &impl Fn(Position, Density) -> f64,
        terminal_cost: &impl Fn(Position, Density) -> f64,
        initial_distribution: &impl Fn(Position) -> f64,
    ) -> (DMatrix<f64>, DMatrix<f64>);
}

/// A Fixed-Point Iteration solver for Mean Field Games.
///
/// Iterates back and forth between the HJB (backward) and Fokker-Planck (forward) equations
/// until convergence (or for a fixed number of iterations).
#[derive(Debug, Clone, PartialEq)]
pub struct FixedPointSolver<H: Hamiltonian> {
    #[allow(missing_docs)]
    pub iterations: usize,
    #[allow(missing_docs)]
    pub hamiltonian: H,
}

impl FixedPointSolver<QuadraticHamiltonian> {
    /// Creates a new solver with a default Quadratic Hamiltonian ($H = p^2/2$).
    #[verified_engine::verified]
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            hamiltonian: QuadraticHamiltonian::default(),
        }
    }
}

impl<H: Hamiltonian> FixedPointSolver<H> {
    /// Creates a new solver with a custom Hamiltonian strategy.
    #[verified_engine::verified]
    pub fn new_with_hamiltonian(iterations: usize, hamiltonian: H) -> Self {
        Self {
            iterations,
            hamiltonian,
        }
    }
}

impl<H: Hamiltonian> MFGSolver for FixedPointSolver<H> {
    #[verified_engine::verified]
    fn solve(
        &self,
        config: &MFGConfig,
        cost_function: &impl Fn(Position, Density) -> f64,
        terminal_cost: &impl Fn(Position, Density) -> f64,
        initial_distribution: &impl Fn(Position) -> f64,
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let nx = config.grid_points.get();
        let nt = config.time_steps.get();

        // Initialize m (density) and u (value)
        let mut m = DMatrix::zeros(nx, nt + 1);
        let mut u = DMatrix::zeros(nx, nt + 1);

        // Precompute x values
        let xs: Vec<f64> = (0..nx)
            .map(|i| config.space_min + (i as f64) * config.dx)
            .collect();

        // Initialize m at t=0
        for i in 0..nx {
            m[(i, 0)] = initial_distribution(Position(xs[i]));
        }

        // Normalize initial distribution
        let sum_0: f64 = m.column(0).sum();
        if sum_0 > math_commons::registry::TOLERANCE_STANDARD {
            m.column_mut(0).scale_mut(1.0 / sum_0);
        }

        // Make an initial guess for m for all t (copy m0)
        let m0 = m.column(0).clone_owned();
        for n in 1..=nt {
            m.column_mut(n).copy_from(&m0);
        }

        let stepper = FusedStencilStepper::new(StepSize(config.dx));
        let mut scratch = DVector::zeros(nx);

        for _iter in 0..self.iterations {
            // 1. Solve HJB Backward
            // Terminal condition
            for i in 0..nx {
                u[(i, nt)] = terminal_cost(Position(xs[i]), Density(m[(i, nt)]));
            }

            // Backward in time
            for n in (0..nt).rev() {
                stepper.step_1d_slice(
                    u.column(n + 1).as_slice(),
                    scratch.as_mut_slice(),
                    config.dt,
                    -1.0, // Backward
                    |i, prev, curr, next, ops| {
                        let du_dx = ops.central_diff_1st(prev, next);
                        let d2u_dx2 = ops.central_diff_2nd(prev, curr, next);
                        let hamiltonian = self.hamiltonian.evaluate(du_dx);
                        let running_cost = cost_function(Position(xs[i]), Density(m[(i, n + 1)]));
                        hamiltonian - config.viscosity * d2u_dx2 - running_cost
                    },
                );

                // Boundary conditions (Neumann 0)
                scratch[0] = scratch[1];
                scratch[nx - 1] = scratch[nx - 2];
                u.column_mut(n).copy_from(&scratch);
            }

            // 2. Solve Fokker-Planck Forward
            for n in 0..nt {
                let u_col = u.column(n);
                let u_curr = u_col.as_slice();

                stepper.step_1d_slice(
                    m.column(n).as_slice(),
                    scratch.as_mut_slice(),
                    config.dt,
                    1.0, // Forward
                    |i, prev, curr, next, ops| {
                        let u_prev = u_curr[i - 1];
                        let u_next = u_curr[i + 1];
                        let du_dx = ops.central_diff_1st(u_prev, u_next);
                        let v = -self.hamiltonian.derivative(du_dx);
                        let drift_flux = ops.upwind_flux(v, prev, curr, next);
                        let d2m_dx2 = ops.central_diff_2nd(prev, curr, next);
                        -drift_flux + config.viscosity * d2m_dx2
                    },
                );

                // Boundary conditions
                scratch[0] = scratch[1];
                scratch[nx - 1] = scratch[nx - 2];

                // Normalize mass to prevent explosion/vanishing
                let sum: f64 = scratch.sum();
                if sum > math_commons::registry::TOLERANCE_STANDARD {
                    scratch.scale_mut(1.0 / sum);
                }

                m.column_mut(n + 1).copy_from(&scratch);
            }
        }

        (u, m)
    }
}
