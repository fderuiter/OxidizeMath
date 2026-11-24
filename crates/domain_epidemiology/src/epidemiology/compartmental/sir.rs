use super::common::{validate_initial_infected, validate_population, validate_rate};
use crate::error::EpidemiologyError;
use crate::impl_compartmental_ops;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, TimeStepper};
use verified_engine::Theory;

/// Compartmental state vector $(S, I, R)$ conserving $S + I + R = N$ (`@kermack1927contribution`, `spec:epidemiology_v1`).
/// Numerical stability: componentwise `f64` vector arithmetic incurs $O(\varepsilon_{\text{mach}} N)$ drift per RK4 step. Domain of validity: non-negative counts $S, I, R \ge 0$ with $S + I + R = N > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SIRState {
    #[allow(missing_docs)]
    pub s: f64,
    #[allow(missing_docs)]
    pub i: f64,
    #[allow(missing_docs)]
    pub r: f64,
}

impl_compartmental_ops!(SIRState, s, i, r);

/// Stateless right-hand side evaluating $(\dot{S}, \dot{I}, \dot{R}) = (-\beta S I / N, \beta S I / N - \gamma I, \gamma I)$ (`@kermack1927contribution`, `spec:epidemiology_v1`).
/// Numerical stability: invariant simplex preserved under explicit RK4 when $\Delta t (\beta + \gamma) < 2.78$. Domain of validity: finite $N > 0$ and $\beta, \gamma \ge 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SIRDynamics {
    #[allow(missing_docs)]
    pub n: f64,
    #[allow(missing_docs)]
    pub beta: f64,
    #[allow(missing_docs)]
    pub gamma: f64,
}

impl OdeSystem<SIRState> for SIRDynamics {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &SIRState) -> SIRState {
        let s = state.s;
        let i = state.i;

        let ds = -self.beta * s * i / self.n;
        let di = self.beta * s * i / self.n - self.gamma * i;
        let dr = self.gamma * i;

        SIRState {
            s: ds,
            i: di,
            r: dr,
        }
    }
}

/// SIR Model: Susceptible, Infectious, Recovered.
///
/// Equations:
/// $$dS/dt = -\beta S I / N$$
/// $$dI/dt = \beta S I / N - \gamma I$$
/// $$dR/dt = \gamma I$$
///
/// - **Citation**: `@kermack1927contribution` (`spec:epidemiology_v1`).
/// - **Numerical stability**: Fourth-order RK4 preserves total mass $S + I + R = N$ to machine precision when $\Delta t (\beta + \gamma) \le 1$.
/// - **Domain of validity**: Closed population $N > 0$, initial infected $0 \le I_0 \le N$, and finite rates $\beta, \gamma \ge 0$.
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "The SIR model is an epidemiological model that computes the theoretical number of people infected with a contagious illness in a closed population over time.",
    citation = "kermack1927contribution"
)]
pub struct SIRModel<S: Solver<SIRState> = RungeKutta4<SIRState>> {
    state: SIRState,
    /// The underlying dynamics model (parameters + equations).
    pub dynamics: SIRDynamics,
    /// The numerical solver strategy.
    solver: S,
}

impl<S: Solver<SIRState>> TimeStepper<SIRState> for SIRModel<S> {
    #[verified_engine::verified]
    fn get_state(&self) -> &SIRState {
        &self.state
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut SIRState {
        &mut self.state
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        // Delegate stepping to the injected solver strategy.
        // pass &self.dynamics to avoid partial borrow of self.
        self.solver.step(&self.dynamics, 0.0, &mut self.state, dt);
    }
}

/// Parameter-validating builder for $\dot{S} = -\beta S I / N, \dot{I} = \beta S I / N - \gamma I, \dot{R} = \gamma I$ (`@kermack1927contribution`, `spec:epidemiology_v1`).
/// Numerical stability: rejects non-finite or zero-population inputs prior to division by $N$. Domain of validity: $N > 0, 0 \le I_0 \le N, \beta \ge 0, \gamma \ge 0$.
#[derive(Debug, Default, Clone)]
pub struct SIRModelBuilder {
    n: Option<f64>,
    i0: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
}

impl SIRModelBuilder {
    /// Sets closed population size $N = S + I + R$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: exact `f64` capture. Domain of validity: finite $N > 0$.
    #[verified_engine::verified]
    pub fn n(mut self, n: f64) -> Self {
        self.n = Some(n);
        self
    }

    /// Sets initial infectious count $I_0 = I(0)$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: exact `f64` capture; $S(0) = N - I_0$. Domain of validity: finite $0 \le I_0 \le N$.
    #[verified_engine::verified]
    pub fn i0(mut self, i0: f64) -> Self {
        self.i0 = Some(i0);
        self
    }

    /// Sets transmission rate $\beta$ in $\dot{S} = -\beta S I / N$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: exact `f64` capture. Domain of validity: finite $\beta \ge 0$.
    #[verified_engine::verified]
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = Some(beta);
        self
    }

    /// Sets recovery rate $\gamma$ in $\dot{R} = \gamma I$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: exact `f64` capture. Domain of validity: finite $\gamma \ge 0$.
    #[verified_engine::verified]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Validates $(N, I_0, \beta, \gamma)$ and constructs an RK4 [`SIRModel`] with $S_0 = N - I_0$ (`@kermack1927contribution`, `spec:epidemiology_v1`).
    /// Numerical stability: guarantees $N > 0$ so $-\beta S I / N$ cannot divide by zero. Domain of validity: $N > 0, 0 \le I_0 \le N, \beta \ge 0, \gamma \ge 0$.
    #[verified_engine::verified]
    pub fn build(self) -> Result<SIRModel<RungeKutta4<SIRState>>, EpidemiologyError> {
        let n = self.n.ok_or(EpidemiologyError::MissingParameter {
            name: "n (population)".to_string(),
        })?;
        let i0 = self.i0.ok_or(EpidemiologyError::MissingParameter {
            name: "i0 (initial infected)".to_string(),
        })?;
        let beta = self.beta.ok_or(EpidemiologyError::MissingParameter {
            name: "beta (transmission rate)".to_string(),
        })?;
        let gamma = self.gamma.ok_or(EpidemiologyError::MissingParameter {
            name: "gamma (recovery rate)".to_string(),
        })?;

        validate_population(n)?;
        validate_initial_infected(i0, n)?;
        validate_rate("beta (transmission rate)", beta)?;
        validate_rate("gamma (recovery rate)", gamma)?;

        let state = SIRState {
            s: n - i0,
            i: i0,
            r: 0.0,
        };

        Ok(SIRModel {
            state,
            dynamics: SIRDynamics { n, beta, gamma },
            solver: RungeKutta4::new(&state),
        })
    }
}

impl SIRModel<RungeKutta4<SIRState>> {
    /// Returns a fresh builder for $\dot{S} = -\beta S I / N, \dot{I} = \beta S I / N - \gamma I, \dot{R} = \gamma I$ (`@kermack1927contribution`, `spec:epidemiology_v1`).
    /// Numerical stability: defers division by $N$ until validated `build()`. Domain of validity: populate $N > 0, 0 \le I_0 \le N, \beta \ge 0, \gamma \ge 0$.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_epidemiology::__doctest_facade as math_explorer;
    /// use math_explorer::epidemiology::compartmental::SIRModel;
    /// use math_explorer::error::EpidemiologyError;
    ///
    /// // Missing required parameters returns EpidemiologyError::MissingParameter
    /// let incomplete = SIRModel::builder().n(1000.0).build();
    /// assert!(matches!(incomplete, Err(EpidemiologyError::MissingParameter { .. })));
    ///
    /// let model = SIRModel::builder()
    ///     .n(1000.0)
    ///     .i0(10.0)
    ///     .beta(0.4)
    ///     .gamma(0.1)
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(model.state().s, 990.0);
    /// ```
    #[verified_engine::verified]
    pub fn builder() -> SIRModelBuilder {
        SIRModelBuilder::default()
    }

    /// Constructs a validated RK4 [`SIRModel`] with $R_0 = \beta / \gamma$ (`@kermack1927contribution`, `spec:epidemiology_v1`).
    /// Numerical stability: $O(\Delta t^4)$ local truncation error under `RungeKutta4`. Domain of validity: finite $N > 0, 0 \le I_0 \le N, \beta \ge 0, \gamma \ge 0$.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_epidemiology::__doctest_facade as math_explorer;
    /// use math_explorer::epidemiology::compartmental::SIRModel;
    ///
    /// let mut model = SIRModel::new(1000.0, 5.0, 0.5, 0.1).unwrap();
    /// model.step(0.5);
    /// let st = model.state();
    /// assert!((st.s + st.i + st.r - 1000.0).abs() < 1e-9);
    /// assert!(st.i > 5.0);
    /// ```
    #[verified_engine::verified]
    pub fn new(
        n: f64,
        i0: f64,
        beta: f64,
        gamma: f64,
    ) -> Result<SIRModel<RungeKutta4<SIRState>>, EpidemiologyError> {
        Self::builder().n(n).i0(i0).beta(beta).gamma(gamma).build()
    }
}

impl<S: Solver<SIRState>> SIRModel<S> {
    /// Advances $(S, I, R)$ by $\Delta t$ under $\dot{S} = -\beta S I / N$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: preserves simplex for $\Delta t (\beta + \gamma) < 2.78$ (RK4). Domain of validity: finite $\Delta t \ge 0$.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        <Self as TimeStepper<SIRState>>::step(self, dt);
    }

    /// Advances $(S, I, R)$ by $\Delta t$ using caller-supplied integrator $\Phi_{\Delta t}$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: governed by `OtherS` stability region. Domain of validity: finite $\Delta t \ge 0$.
    pub fn step_with<OtherS: Solver<SIRState>>(&mut self, solver: &mut OtherS, dt: f64) {
        <Self as TimeStepper<SIRState>>::step_with(self, solver, dt);
    }

    /// Replaces the internal ODE integrator advancing $(S, I, R)(t + \Delta t)$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: lossless state transfer without rounding. Domain of validity: any [`Solver<SIRState>`].
    pub fn with_solver<NewS: Solver<SIRState>>(self, new_solver: NewS) -> SIRModel<NewS> {
        SIRModel {
            state: self.state,
            dynamics: self.dynamics,
            solver: new_solver,
        }
    }

    /// Returns transmission rate $\beta \ge 0$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: exact scalar copy. Domain of validity: non-negative rate $\beta \ge 0$.
    #[verified_engine::verified]
    pub fn beta(&self) -> f64 {
        self.dynamics.beta
    }

    /// Returns recovery rate $\gamma \ge 0$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: exact scalar copy. Domain of validity: non-negative rate $\gamma \ge 0$.
    #[verified_engine::verified]
    pub fn gamma(&self) -> f64 {
        self.dynamics.gamma
    }

    /// Returns total closed population $N = S + I + R > 0$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: exact scalar copy. Domain of validity: strictly positive $N > 0$.
    #[verified_engine::verified]
    pub fn n(&self) -> f64 {
        self.dynamics.n
    }

    /// Borrows current compartmental state $(S, I, R)$ (`@kermack1927contribution`, `spec:epidemiology_v1`). Numerical stability: zero-cost reference borrow. Domain of validity: $S, I, R \ge 0$ with $S + I + R = N$.
    #[verified_engine::verified]
    pub fn state(&self) -> &SIRState {
        &self.state
    }
}

impl<S: Solver<SIRState>> OdeSystem<SIRState> for SIRModel<S> {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &SIRState) -> SIRState {
        // Delegate to the pure dynamics component
        self.dynamics.derivative(t, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pure_math::pure_math::analysis::ode::{Euler, RungeKutta4};

    #[test]
    #[verified_engine::verified]
    fn test_builder() {
        let model = SIRModel::builder()
            .n(1000.0)
            .i0(10.0)
            .beta(0.5)
            .gamma(0.1)
            .build();
        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.n(), 1000.0);
        assert_eq!(model.beta(), 0.5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_builder_missing_param() {
        let model = SIRModel::builder().n(1000.0).build();
        assert!(matches!(
            model,
            Err(EpidemiologyError::MissingParameter { .. })
        ));
    }

    #[test]
    #[verified_engine::verified]
    fn test_threshold_theorem() {
        let n = 1000.0;
        let i0 = 10.0;
        // R0 = beta / gamma = 0.5 / 1.0 = 0.5 < 1
        let mut model = SIRModel::new(n, i0, 0.5, 1.0).unwrap();

        let initial_i = model.state().i;
        model.step(0.1);

        assert!(
            model.state().i < initial_i,
            "Infected should decrease when R0 < 1"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_sir_step_with_rk4() {
        let n = 1000.0;
        let i0 = 10.0;
        let mut model_std = SIRModel::new(n, i0, 0.5, 0.1).unwrap();
        let mut model_with = SIRModel::new(n, i0, 0.5, 0.1).unwrap();

        let dt = 0.1;
        model_std.step(dt);
        let state = *model_with.state();

        // Use external solver
        model_with.step_with(&mut RungeKutta4::new(&state), dt);

        assert_eq!(
            model_std.state(),
            model_with.state(),
            "step and step_with(RK4) should yield identical results"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_sir_step_with_euler() {
        let n = 1000.0;
        let i0 = 10.0;
        let mut model = SIRModel::new(n, i0, 0.5, 0.1).unwrap();

        // Euler is less accurate but should still run without panic
        let state = *model.state();
        model.step_with(&mut Euler::new(&state), 0.1);

        assert!(model.state().s <= n);
        assert!(model.state().i >= 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_independent_dynamics() {
        // Demonstrate usage of SIRDynamics without SIRModel (pure strategy pattern)
        let dynamics = SIRDynamics {
            n: 1000.0,
            beta: 0.5,
            gamma: 0.1,
        };

        // Initial State
        let mut state = SIRState {
            s: 990.0,
            i: 10.0,
            r: 0.0,
        };

        // Use a generic solver directly with the dynamics and the state
        let mut solver = RungeKutta4::new(&state);
        let dt = 0.1;

        // Step forward
        solver.step(&dynamics, 0.0, &mut state, dt);

        // Verify state changed appropriately
        // S should decrease (infection spreads)
        assert!(state.s < 990.0);
        // I should increase (R0 = 5 > 1)
        assert!(state.i > 10.0);
        // R should increase (recovery)
        assert!(state.r > 0.0);

        // Ensure conservation of mass (population size constant)
        let total = state.s + state.i + state.r;
        assert!((total - 1000.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_swapping_solver() {
        let n = 1000.0;
        let i0 = 10.0;
        let model = SIRModel::new(n, i0, 0.5, 0.1).unwrap();

        // Swap from RK4 (default) to Euler
        let state = *model.state();
        let mut model_euler = model.with_solver(Euler::new(&state));

        model_euler.step(0.1);

        // Just check it ran
        assert!(model_euler.state().s <= n);
    }
}
