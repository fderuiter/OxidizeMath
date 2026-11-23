use super::strategies::MatrixPayoff;
use super::traits::FitnessStrategy;
use crate::error::GameTheoryError;
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver};

/// Represents an Evolutionary Game solved via **Replicator Dynamics**.
///
/// In evolutionary game theory, we study how the composition of a population changes over time
/// based on the fitness of different strategies. The governing equation is the Replicator Equation:
///
/// $$ \dot{x}_i = x_i \left( f_i(x) - \bar{f}(x) \right) $$
///
/// Where:
/// - $x_i$ is the proportion of the population playing strategy $i$.
/// - $f_i(x)$ is the fitness of strategy $i$ against population state $x$.
/// - $\bar{f}(x) = \sum x_j f_j(x)$ is the average fitness of the population.
///
/// This struct uses the **Strategy Pattern** for fitness calculation, allowing
/// for both standard Matrix Games (linear) and complex Non-Linear Games.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplicatorDynamics<S: FitnessStrategy = MatrixPayoff> {
    /// The strategy used to compute fitness values.
    strategy: S,
}

impl ReplicatorDynamics<MatrixPayoff> {
    /// Creates a new system with the given payoff matrix $A$.
    ///
    /// This constructor is preserved for backward compatibility.
    /// It instantiates the default `MatrixPayoff` strategy.
    #[verified_engine::verified]
    pub fn new(payoff_matrix: DMatrix<f64>) -> Result<Self, GameTheoryError> {
        Ok(Self {
            strategy: MatrixPayoff::new(payoff_matrix)?,
        })
    }

    /// Accessor for the underlying payoff matrix.
    ///
    /// Useful for inspecting the game structure.
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
    ///
    /// This is a helper method that delegates to the `OdeSystem` implementation.
    #[verified_engine::verified]
    pub fn derivative(&self, x: &DVector<f64>) -> DVector<f64> {
        <Self as OdeSystem<DVector<f64>>>::derivative(self, 0.0, x)
    }

    /// Simulates the dynamics over time using the **Runge-Kutta 4** method by default.
    ///
    /// # Parameters
    /// - `initial_population`: Vector of proportions summing to 1.0.
    /// - `time_horizon`: Total time to simulate.
    /// - `dt`: Time step size.
    ///
    /// # Returns
    /// A time-series of population states: `Vec<(Time, State)>`.
    #[verified_engine::verified]
    pub fn simulate(
        &self,
        initial_population: DVector<f64>,
        time_horizon: f64,
        dt: f64,
    ) -> Vec<(f64, DVector<f64>)> {
        // Create a solver initialized with the structure of the population vector
        let mut solver = RungeKutta4::new(&initial_population);
        self.simulate_with_strategy(initial_population, time_horizon, dt, &mut solver)
    }

    /// Simulates the dynamics over time using a provided **Solver** strategy.
    ///
    /// This allows for different integration schemes (e.g., Euler vs Runge-Kutta)
    /// to be used based on accuracy or performance requirements.
    ///
    /// # Parameters
    /// - `initial_population`: Vector of proportions summing to 1.0.
    /// - `time_horizon`: Total time to simulate.
    /// - `dt`: Time step size.
    /// - `solver`: The numerical integrator to use.
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
        let steps = (time_horizon / dt).ceil() as usize;
        let mut trajectory = Vec::with_capacity(steps + 1);
        let mut current_x = initial_population;
        let mut current_t = 0.0;

        trajectory.push((current_t, current_x.clone()));

        for _ in 0..steps {
            // Bolt Optimization: Use step in-place to avoid the allocation inside solve()
            solver.step(self, current_t, &mut current_x, dt);
            current_t += dt;

            // Normalize to prevent numerical drift from simplex
            let sum = current_x.sum();
            if (sum - 1.0).abs() > math_commons::registry::TOLERANCE_STANDARD {
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
        // 1. Calculate Fitness Vector f(x)
        // Store directly in 'out' to avoid allocation
        self.strategy.fitness(x, out);

        // 2. Calculate Average Fitness \bar{f} = x . f(x)
        // 'out' holds f(x), so we dot with x
        let average_fitness = x.dot(out);

        // 3. Calculate Replicator Equation: dx_i/dt = x_i * (f_i - \bar{f})
        // 'out' holds f_i, so we update it in-place.
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
        // Rock Paper Scissors matrix (Zero-Sum)
        let payoff =
            DMatrix::from_row_slice(3, 3, &[0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0]);

        let system = ReplicatorDynamics::new(payoff).unwrap();

        // Equilibrium check
        let equilibrium = DVector::from_vec(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let deriv = system.derivative(&equilibrium);
        assert!(deriv.norm() < math_commons::registry::TOLERANCE_STANDARD);

        // Simulation check
        let init = DVector::from_vec(vec![0.4, 0.3, 0.3]);
        let trajectory = system.simulate(init, 10.0, 0.01);
        let last_state = &trajectory.last().unwrap().1;
        assert!((last_state.sum() - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_non_linear_fitness() {
        // Implement a custom non-linear strategy
        // Example: Frequency-dependent selection where fitness depends on x squared.
        struct QuadraticFitness;
        impl FitnessStrategy for QuadraticFitness {
            #[verified_engine::verified]
            fn fitness(&self, x: &DVector<f64>, out: &mut DVector<f64>) {
                // f_i = x_i
                out.copy_from(x);
            }
        }

        let system = ReplicatorDynamics::new_with_fitness(QuadraticFitness);
        let init = DVector::from_vec(vec![0.8, 0.2]); // x1 > x2 implies f1 > f2

        // Replicator eq: dx1/dt = x1 * (x1 - (x1^2 + x2^2))
        // Since x1 > x2, x1 should grow (if x1 > 0.5 in this case? No wait.)
        // Avg fitness = x1^2 + x2^2.
        // f1 - avg = x1 - (x1^2 + x2^2).

        let deriv = system.derivative(&init);
        // x1=0.8, x2=0.2. Avg = 0.64 + 0.04 = 0.68.
        // f1 = 0.8. f1 - avg = 0.12. dx1 = 0.8 * 0.12 = 0.096.
        // f2 = 0.2. f2 - avg = 0.2 - 0.68 = -0.48. dx2 = 0.2 * -0.48 = -0.096.

        assert!((deriv[0] - 0.096).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((deriv[1] - -0.096).abs() < math_commons::registry::TOLERANCE_FAST);
    }
}
