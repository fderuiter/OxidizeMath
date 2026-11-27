//! # Evolutionary Replicator Dynamics
//!
//! Provides the [`FitnessStrategy`] abstraction, linear [`MatrixPayoff`] games,
//! and simplex-preserving [`ReplicatorDynamics`] ODE trajectories.

use crate::error::GameTheoryError;
use crate::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver};
use nalgebra::{DMatrix, DVector};

/// Defines the strategy for calculating the fitness of a population.
///
/// In Evolutionary Game Theory, the fitness of a strategy depends on the composition
/// of the population. This trait abstracts the fitness calculation, allowing for
/// both linear (matrix-based) and non-linear (frequency-dependent) games.
pub trait FitnessStrategy {
    /// Computes the fitness vector $f(x)$ for the population state $x$.
    ///
    /// # Arguments
    /// * `x` - The current population state (proportions of each strategy).
    /// * `out` - The output vector to store the calculated fitness values.
    #[verified_engine::verified]
    fn fitness(&self, x: &DVector<f64>, out: &mut DVector<f64>);
}

/// A standard linear fitness landscape defined by a Payoff Matrix.
///
/// For a population state $x$ and payoff matrix $A$, the fitness vector is:
/// $$ f(x) = Ax $$
#[derive(Debug, Clone)]
pub struct MatrixPayoff {
    payoff_matrix: DMatrix<f64>,
}

impl MatrixPayoff {
    /// Creates a new [`MatrixPayoff`] strategy.
    ///
    /// # Errors
    /// Returns [`GameTheoryError::NonSquarePayoffMatrix`] if the matrix is not square.
    #[verified_engine::verified]
    pub fn new(payoff_matrix: DMatrix<f64>) -> Result<Self, GameTheoryError> {
        if payoff_matrix.nrows() != payoff_matrix.ncols() {
            return Err(GameTheoryError::NonSquarePayoffMatrix {
                rows: payoff_matrix.nrows(),
                cols: payoff_matrix.ncols(),
            });
        }
        Ok(Self { payoff_matrix })
    }

    /// Returns a reference to the underlying payoff matrix.
    #[verified_engine::verified]
    pub fn payoff_matrix(&self) -> &DMatrix<f64> {
        &self.payoff_matrix
    }
}

impl FitnessStrategy for MatrixPayoff {
    #[verified_engine::verified]
    fn fitness(&self, x: &DVector<f64>, out: &mut DVector<f64>) {
        self.payoff_matrix.mul_to(x, out);
    }
}

const MAX_REPLICATOR_STEPS: usize = 200_000;

/// Represents an Evolutionary Game solved via **Replicator Dynamics**:
///
/// $$ \dot{x}_i = x_i \left( f_i(x) - \bar{f}(x) \right), \quad \bar{f}(x) = \sum_j x_j f_j(x) $$
pub struct ReplicatorDynamics<S: FitnessStrategy = MatrixPayoff> {
    strategy: S,
}

impl ReplicatorDynamics<MatrixPayoff> {
    /// Creates a new system with the given square payoff matrix $A$.
    #[verified_engine::verified]
    pub fn new(payoff_matrix: DMatrix<f64>) -> Result<Self, GameTheoryError> {
        Ok(Self {
            strategy: MatrixPayoff::new(payoff_matrix)?,
        })
    }

    /// Accessor for the underlying payoff matrix.
    #[verified_engine::verified]
    pub fn payoff_matrix(&self) -> &DMatrix<f64> {
        self.strategy.payoff_matrix()
    }
}

impl<S: FitnessStrategy> ReplicatorDynamics<S> {
    /// Creates a new system with a custom fitness strategy.
    #[verified_engine::verified]
    pub fn new_with_fitness(strategy: S) -> Self {
        Self { strategy }
    }

    /// Computes the time derivative $\dot{x}$ for the population state $x$.
    #[verified_engine::verified]
    pub fn derivative(&self, x: &DVector<f64>) -> DVector<f64> {
        <Self as OdeSystem<DVector<f64>>>::derivative(self, 0.0, x)
    }

    /// Simulates the dynamics over time using the **Runge-Kutta 4** method by default.
    #[verified_engine::verified]
    pub fn simulate(
        &self,
        initial_population: DVector<f64>,
        time_horizon: f64,
        dt: f64,
    ) -> Vec<(f64, DVector<f64>)> {
        let mut solver = RungeKutta4::new(&initial_population);
        self.simulate_with_strategy(initial_population, time_horizon, dt, &mut solver)
    }

    /// Simulates the dynamics over time using a provided [`Solver`] strategy.
    #[verified_engine::verified]
    pub fn simulate_with_strategy<Solve>(
        &self,
        initial_population: DVector<f64>,
        time_horizon: f64,
        dt: f64,
        solver: &mut Solve,
    ) -> Vec<(f64, DVector<f64>)>
    where
        Solve: Solver<DVector<f64>>,
    {
        let steps =
            if !time_horizon.is_finite() || time_horizon <= 0.0 || !dt.is_finite() || dt <= 0.0 {
                0
            } else {
                let raw_steps = (time_horizon / dt).ceil();
                if raw_steps.is_nan() || raw_steps <= 0.0 {
                    0
                } else if raw_steps >= MAX_REPLICATOR_STEPS as f64 {
                    MAX_REPLICATOR_STEPS
                } else {
                    raw_steps as usize
                }
            };
        let mut trajectory = Vec::with_capacity(steps.saturating_add(1));
        let mut current_x = initial_population;
        let mut current_t = 0.0;

        trajectory.push((current_t, current_x.clone()));

        for _ in 0..steps {
            solver.step(self, current_t, &mut current_x, dt);
            current_t += dt;

            let sum = current_x.sum();
            if sum.is_finite()
                && sum > 0.0
                && (sum - 1.0).abs() > math_commons::registry::TOLERANCE_STANDARD
            {
                current_x /= sum;
            }

            trajectory.push((current_t, current_x.clone()));
        }

        trajectory
    }
}

impl<S: FitnessStrategy> OdeSystem<DVector<f64>> for ReplicatorDynamics<S> {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, x: &DVector<f64>) -> DVector<f64> {
        let mut out = DVector::zeros(x.len());
        self.derivative_in_place(t, x, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, x: &DVector<f64>, out: &mut DVector<f64>) {
        self.strategy.fitness(x, out);
        let average_fitness = x.dot(out);
        for (o, xi) in out.iter_mut().zip(x.iter()) {
            *o = *xi * (*o - average_fitness);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_rock_paper_scissors() {
        let payoff =
            DMatrix::from_row_slice(3, 3, &[0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0]);
        let system = ReplicatorDynamics::new(payoff).unwrap();
        let equilibrium = DVector::from_vec(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let deriv = system.derivative(&equilibrium);
        assert!(deriv.norm() < math_commons::registry::TOLERANCE_STANDARD);

        let init = DVector::from_vec(vec![0.4, 0.3, 0.3]);
        let trajectory = system.simulate(init, 10.0, 0.01);
        let last_state = &trajectory.last().unwrap().1;
        assert!((last_state.sum() - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_non_linear_fitness() {
        struct QuadraticFitness;
        impl FitnessStrategy for QuadraticFitness {
            #[verified_engine::verified]
            fn fitness(&self, x: &DVector<f64>, out: &mut DVector<f64>) {
                out.copy_from(x);
            }
        }

        let system = ReplicatorDynamics::new_with_fitness(QuadraticFitness);
        let init = DVector::from_vec(vec![0.8, 0.2]);
        let deriv = system.derivative(&init);
        assert!((deriv[0] - 0.096).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((deriv[1] - -0.096).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_simulate_guards_zero_dt_and_extreme_step_counts() {
        let payoff = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, 0.0, 1.0]);
        let system = ReplicatorDynamics::new(payoff).unwrap();
        let init = DVector::from_vec(vec![0.25, 0.75]);

        for (horizon, dt) in [
            (20.0, 0.0),
            (20.0, -0.05),
            (20.0, f64::NAN),
            (20.0, f64::INFINITY),
            (0.0, 0.05),
            (-10.0, 0.05),
            (f64::NAN, 0.05),
            (f64::INFINITY, 0.05),
        ] {
            let traj = system.simulate(init.clone(), horizon, dt);
            assert_eq!(traj.len(), 1);
            assert_eq!(traj[0].0, 0.0);
            assert_eq!(traj[0].1, init);
        }

        let capped = system.simulate(init, 20.0, 1e-12);
        assert_eq!(capped.len(), MAX_REPLICATOR_STEPS + 1);
        let last = &capped.last().unwrap().1;
        assert!(last.iter().all(|v| v.is_finite()));
    }
}
