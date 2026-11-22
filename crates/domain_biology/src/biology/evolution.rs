//! # Evolutionary Dynamics (Hawk-Dove)
//!
//! This module models population dynamics where individuals adopt strategies that compete for resources.
//! The classic example implemented here is the **Hawk-Dove Game**.
//!
//! ##  The Hawk-Dove Game
//!
//! - **Hawk**: Aggressive strategy. Fights for the resource.
//! - **Dove**: Passive strategy. Shares the resource or retreats.
//!
//! The payoff matrix is:
//!
//! | vs | Hawk | Dove |
//! |---|---|---|
//! | **Hawk** | $(V-C)/2$ | $V$ |
//! | **Dove** | $0$ | $V/2$ |
//!
//! Where:
//! - $V$: Value of the resource.
//! - $C$: Cost of injury from fighting.
//!
//! ##  Quick Start
//!
//! Simulate a population where the cost of fighting is high ($C > V$).
//! In this case, neither strategy is an Evolutionarily Stable Strategy (ESS),
//! and the population should converge to a mixed equilibrium.
//!
//! ```rust
//! # use domain_biology::__doctest_facade as math_explorer;
//! use math_explorer::biology::evolution::HawkDovePopulation;
//! use math_explorer::primitives::UnitInterval;
//!
//! // 1. Define the environment
//! // Value = 2.0, Cost = 10.0 (High cost of fighting)
//! let population = HawkDovePopulation::new(2.0, 10.0);
//!
//! // 2. Initial State: Mostly Hawks (90%)
//! let mut hawk_freq = UnitInterval::new(0.9).unwrap();
//! let dt = 0.1;
//!
//! // 3. Evolve over time
//! for _ in 0..100 {
//!     hawk_freq = population.update_frequencies(hawk_freq, dt).expect("Invalid probability");
//! }
//!
//! // 4. Check convergence
//! // Theoretical Equilibrium: p = V/C = 2/10 = 0.2
//! println!("Final Hawk Frequency: {:.3}", hawk_freq.value());
//! assert!((hawk_freq.value() - 0.2).abs() < 0.05);
//! ```

use domain_applied::applied::game_theory::evolutionary::ReplicatorDynamics;
use domain_applied::error::GameTheoryError;
use math_commons::primitives::UnitInterval;
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::analysis::ode::{ArrayState, Euler, OdeSystem, Solver, SolverExt};

/// Represents a population playing the Hawk-Dove game.
pub struct HawkDovePopulation {
    /// Value of the resource
    pub v: f64,
    /// Cost of injury
    pub c: f64,
}

impl HawkDovePopulation {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(v: f64, c: f64) -> Self {
        Self { v, c }
    }

    /// Constructs the Replicator Dynamics system for this game.
    ///
    /// The payoff matrix is:
    ///
    /// $$
    /// \begin{pmatrix}
    /// (V-C)/2 & V \\
    /// 0 & V/2
    /// \end{pmatrix}
    /// $$
    ///
    /// Row 0 / Col 0: Hawk
    /// Row 1 / Col 1: Dove
    #[verified_engine::verified]
    pub fn to_replicator_dynamics(&self) -> Result<ReplicatorDynamics, GameTheoryError> {
        let e_hh = (self.v - self.c) / 2.0;
        let e_hd = self.v;
        let e_dh = 0.0;
        let e_dd = self.v / 2.0;

        let payoff = DMatrix::from_row_slice(2, 2, &[e_hh, e_hd, e_dh, e_dd]);

        ReplicatorDynamics::new(payoff)
    }

    /// Updates the frequency of the Hawk strategy using the Replicator Equation.
    ///
    /// # Arguments
    /// * `hawk_freq` - Current frequency of Hawks ($p_H$). (Dove freq is $1 - p_H$).
    /// * `dt` - Time step.
    ///
    /// # Returns
    /// The new frequency of Hawks.
    #[verified_engine::verified]
    pub fn update_frequencies(
        &self,
        hawk_freq: UnitInterval,
        dt: f64,
    ) -> Result<UnitInterval, GameTheoryError> {
        // Use ArrayState (stack allocated) for zero-overhead simulation
        let current_state = ArrayState([hawk_freq.value()]);
        let mut solver = Euler::new(&current_state);

        // Solve using the efficient OdeSystem<ArrayState<1>> implementation
        let next_state = solver.solve(self, 0.0, &current_state, dt);

        let new_p_h = next_state.0[0];

        // Return error if out of bounds instead of manual clamping
        UnitInterval::new(new_p_h).map_err(|_| GameTheoryError::InvalidParameter {
            name: "hawk_freq".to_string(),
            value: new_p_h,
        })
    }

    /// Updates the frequency of the Hawk strategy using a provided solver strategy.
    ///
    /// This method allows for Dependency Injection of different numerical integrators
    /// (e.g., Euler, Runge-Kutta 4), adhering to the Dependency Inversion Principle.
    ///
    /// # Arguments
    /// * `hawk_freq` - Current frequency of Hawks.
    /// * `dt` - Time step.
    /// * `solver` - A mutable reference to a generic `Solver`.
    #[verified_engine::verified]
    pub fn update_frequencies_with_solver<S>(
        &self,
        hawk_freq: UnitInterval,
        dt: f64,
        solver: &mut S,
    ) -> Result<UnitInterval, GameTheoryError>
    where
        S: Solver<DVector<f64>>,
    {
        let p_h = hawk_freq.value();
        let p_d = 1.0 - p_h;

        let current_state = DVector::from_vec(vec![p_h, p_d]);
        let system = self.to_replicator_dynamics()?;

        // Use the injected solver strategy
        let next_state = solver.solve(&system, 0.0, &current_state, dt);
        let new_p_h = next_state[0];

        // Return error if out of bounds instead of manual clamping
        UnitInterval::new(new_p_h).map_err(|_| GameTheoryError::InvalidParameter {
            name: "hawk_freq".to_string(),
            value: new_p_h,
        })
    }
}

/// Implement OdeSystem for ArrayState<1> to allow zero-allocation simulation.
///
/// The state vector contains `[hawk_freq]`.
///
/// Equation: $\dot{p}_H = p_H (1 - p_H) \frac{(V - p_H C)}{2}$
impl OdeSystem<ArrayState<1>> for HawkDovePopulation {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        let p_h = state.0[0];

        // Analytical derivative derived from the replicator equation
        // f_H = p_H * (V-C)/2 + (1-p_H) * V
        // f_D = p_H * 0 + (1-p_H) * V/2
        // \dot{p}_H = p_H * (1 - p_H) * (f_H - f_D)
        // Simplifies to: p_H * (1 - p_H) * (V - p_H * C) / 2

        let dp_dt = p_h * (1.0 - p_h) * (self.v - p_h * self.c) / 2.0;

        ArrayState([dp_dt])
    }
}
