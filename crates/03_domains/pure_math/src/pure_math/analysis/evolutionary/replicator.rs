// [cite:pure_math]
use super::strategies::MatrixPayoff;
use super::traits::{FitnessStrategy, GameTheoryError};
use crate::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver};
use nalgebra::{DMatrix, DVector};

/// Continuous-time evolutionary game ODE system governed by **Replicator Dynamics**.
///
/// Evolves strategy frequencies $\mathbf{x}(t) \in \Delta^{n-1}$ according to excess fitness over the
/// population mean $\bar{f}(\mathbf{x}) = \mathbf{x}^\top \mathbf{f}(\mathbf{x})$:
///
/// $$\dot{x}_i = x_i \left( f_i(\mathbf{x}) - \bar{f}(\mathbf{x}) \right), \quad \bar{f}(\mathbf{x}) = \sum_{j=1}^{n} x_j f_j(\mathbf{x})$$
///
/// Reference: Taylor & Jonker (1978), *Math. Biosci.* 40(1–2), 145–156, DOI: 10.1016/0025-5564(78)90077-9; see
/// [`docs/math_specs/pure_math.md`](../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// The probability simplex $\Delta^{n-1} = \{\mathbf{x} \ge 0 : \sum_i x_i = 1\}$ and its faces are
/// forward-invariant; explicit RK4 stepping re-normalizes $\sum_i x_i = 1$ when rounding drift exceeds
/// `TOLERANCE_STANDARD` and caps total steps at `200_000`.
///
/// # Domain of Validity
/// Non-negative population vector $\mathbf{x} \in \Delta^{n-1}$ summing to $1$ with finite fitness
/// evaluations and positive finite `time_horizon` and `dt`.
pub struct ReplicatorDynamics<S: FitnessStrategy = MatrixPayoff> {
    strategy: S,
}

impl ReplicatorDynamics<MatrixPayoff> {
    /// Creates a [`ReplicatorDynamics`] system backed by a linear [`MatrixPayoff`] matrix $\mathbf{A}$.
    ///
    /// # Errors
    /// Returns [`GameTheoryError::NonSquarePayoffMatrix`] if `payoff_matrix` is not square.
    #[verified_engine::verified]
    pub fn new(payoff_matrix: DMatrix<f64>) -> Result<Self, GameTheoryError> {
        Ok(Self {
            strategy: MatrixPayoff::new(payoff_matrix)?,
        })
    }

    /// Returns a reference to the underlying payoff matrix $\mathbf{A}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn payoff_matrix(&self) -> &DMatrix<f64> {
        self.strategy.payoff_matrix()
    }
}

const MAX_REPLICATOR_STEPS: usize = 200_000;

impl<S: FitnessStrategy> ReplicatorDynamics<S> {
    /// Creates a [`ReplicatorDynamics`] system with a custom [`FitnessStrategy`].
    #[must_use]
    #[verified_engine::verified]
    pub fn new_with_fitness(strategy: S) -> Self {
        Self { strategy }
    }

    /// Computes the instantaneous replicator time derivative $\dot{\mathbf{x}}$ at state `x`.
    #[must_use]
    #[verified_engine::verified]
    pub fn derivative(&self, x: &DVector<f64>) -> DVector<f64> {
        <Self as OdeSystem<DVector<f64>>>::derivative(self, 0.0, x)
    }

    /// Simulates replicator trajectories over `time_horizon` with step `dt` using 4th-order Runge-Kutta.
    #[must_use]
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

    /// Simulates replicator trajectories over `time_horizon` with step `dt` using `solver`.
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
