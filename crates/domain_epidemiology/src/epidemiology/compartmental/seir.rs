use super::common::{validate_initial_infected, validate_population, validate_rate};
use crate::error::EpidemiologyError;
use crate::impl_compartmental_ops;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, TimeStepper};
use verified_engine::Theory;

/// Compartmental state vector $(S, E, I, R)$ conserving $S + E + I + R = N$ (`@anderson1991infectious`, `spec:epidemiology_v1`).
/// Numerical stability: componentwise `f64` vector arithmetic incurs $O(\varepsilon_{\text{mach}} N)$ drift per step. Domain of validity: non-negative counts $S, E, I, R \ge 0$ with $S + E + I + R = N > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SEIRState {
    #[allow(missing_docs)]
    pub s: f64,
    #[allow(missing_docs)]
    pub e: f64,
    #[allow(missing_docs)]
    pub i: f64,
    #[allow(missing_docs)]
    pub r: f64,
}

impl_compartmental_ops!(SEIRState, s, e, i, r);

/// Stateless SEIR dynamics $(\dot{S}, \dot{E}, \dot{I}, \dot{R}) = (-\beta S I / N, \beta S I / N - \sigma E, \sigma E - \gamma I, \gamma I)$ (`@anderson1991infectious`, `spec:epidemiology_v1`).
/// Numerical stability: explicit RK4 preserves positivity when $\Delta t (\beta + \sigma + \gamma) < 2.78$. Domain of validity: finite $N > 0$ and $\beta, \sigma, \gamma \ge 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SEIRDynamics {
    #[allow(missing_docs)]
    pub n: f64,
    #[allow(missing_docs)]
    pub beta: f64,
    #[allow(missing_docs)]
    pub sigma: f64,
    #[allow(missing_docs)]
    pub gamma: f64,
}

impl OdeSystem<SEIRState> for SEIRDynamics {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &SEIRState) -> SEIRState {
        let s = state.s;
        let e = state.e;
        let i = state.i;

        let new_exposed = self.beta * s * i / self.n;
        let ds = -new_exposed;
        let de = new_exposed - self.sigma * e;
        let di = self.sigma * e - self.gamma * i;
        let dr = self.gamma * i;

        SEIRState {
            s: ds,
            e: de,
            i: di,
            r: dr,
        }
    }
}

/// SEIR Model: Susceptible, Exposed, Infectious, Recovered.
///
/// Equations:
/// $$dE/dt = \beta S I / N - \sigma E$$
/// $$dI/dt = \sigma E - \gamma I$$
///
/// - **Citation**: `@anderson1991infectious` (`spec:epidemiology_v1`).
/// - **Numerical stability**: Fourth-order RK4 conserves $S + E + I + R = N$ up to $O(\varepsilon_{\text{mach}} N)$ for $\Delta t (\beta + \sigma + \gamma) \le 1$.
/// - **Domain of validity**: Closed population $N > 0$, initial infected $0 \le I_0 \le N$, and finite rates $\beta, \sigma, \gamma \ge 0$.
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "The SEIR model is a compartmental epidemic model that incorporates an exposed (latent) period before individuals become infectious.",
    citation = "anderson1991infectious"
)]
pub struct SEIRModel<S: Solver<SEIRState> = RungeKutta4<SEIRState>> {
    state: SEIRState,
    /// The underlying dynamics model (parameters + equations).
    pub dynamics: SEIRDynamics,
    /// The numerical solver strategy.
    solver: S,
}

impl<S: Solver<SEIRState>> TimeStepper<SEIRState> for SEIRModel<S> {
    #[verified_engine::verified]
    fn get_state(&self) -> &SEIRState {
        &self.state
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut SEIRState {
        &mut self.state
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        // Delegate stepping to the injected solver strategy.
        // pass &self.dynamics to avoid partial borrow of self.
        self.solver.step(&self.dynamics, 0.0, &mut self.state, dt);
    }
}

/// Parameter-validating builder for SEIR dynamics $\dot{E} = \beta S I / N - \sigma E$ (`@anderson1991infectious`, `spec:epidemiology_v1`).
/// Numerical stability: rejects zero/negative $N$ prior to division by $N$. Domain of validity: $N > 0, 0 \le I_0 \le N, \beta, \sigma, \gamma \ge 0$.
#[derive(Debug, Default, Clone)]
pub struct SEIRModelBuilder {
    n: Option<f64>,
    i0: Option<f64>,
    beta: Option<f64>,
    sigma: Option<f64>,
    gamma: Option<f64>,
}

impl SEIRModelBuilder {
    /// Sets total closed population $N = S + E + I + R$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: exact `f64` capture. Domain of validity: finite $N > 0$.
    #[verified_engine::verified]
    pub fn n(mut self, n: f64) -> Self {
        self.n = Some(n);
        self
    }

    /// Sets initial infectious count $I_0 = I(0)$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: exact `f64` capture; $S_0 = N - I_0, E_0 = 0$. Domain of validity: finite $0 \le I_0 \le N$.
    #[verified_engine::verified]
    pub fn i0(mut self, i0: f64) -> Self {
        self.i0 = Some(i0);
        self
    }

    /// Sets transmission rate $\beta$ in $\dot{S} = -\beta S I / N$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: exact `f64` capture. Domain of validity: finite $\beta \ge 0$.
    #[verified_engine::verified]
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = Some(beta);
        self
    }

    /// Sets latent progression rate $\sigma$ in $\dot{I} = \sigma E - \gamma I$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: exact `f64` capture. Domain of validity: finite $\sigma \ge 0$.
    #[verified_engine::verified]
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }

    /// Sets recovery rate $\gamma$ in $\dot{R} = \gamma I$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: exact `f64` capture. Domain of validity: finite $\gamma \ge 0$.
    #[verified_engine::verified]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Validates $(N, I_0, \beta, \sigma, \gamma)$ and constructs an RK4 [`SEIRModel`] with $S_0 = N - I_0$ (`@anderson1991infectious`, `spec:epidemiology_v1`).
    /// Numerical stability: guarantees $N > 0$ before evaluating $\beta S I / N$. Domain of validity: $N > 0, 0 \le I_0 \le N, \beta, \sigma, \gamma \ge 0$.
    #[verified_engine::verified]
    pub fn build(self) -> Result<SEIRModel<RungeKutta4<SEIRState>>, EpidemiologyError> {
        let n = self.n.ok_or(EpidemiologyError::MissingParameter {
            name: "n (population)".to_string(),
        })?;
        let i0 = self.i0.ok_or(EpidemiologyError::MissingParameter {
            name: "i0 (initial infected)".to_string(),
        })?;
        let beta = self.beta.ok_or(EpidemiologyError::MissingParameter {
            name: "beta (transmission rate)".to_string(),
        })?;
        let sigma = self.sigma.ok_or(EpidemiologyError::MissingParameter {
            name: "sigma (incubation rate)".to_string(),
        })?;
        let gamma = self.gamma.ok_or(EpidemiologyError::MissingParameter {
            name: "gamma (recovery rate)".to_string(),
        })?;

        validate_population(n)?;
        validate_initial_infected(i0, n)?;
        validate_rate("beta (transmission rate)", beta)?;
        validate_rate("sigma (incubation rate)", sigma)?;
        validate_rate("gamma (recovery rate)", gamma)?;

        let state = SEIRState {
            s: n - i0,
            e: 0.0,
            i: i0,
            r: 0.0,
        };

        Ok(SEIRModel {
            state,
            dynamics: SEIRDynamics {
                n,
                beta,
                sigma,
                gamma,
            },
            solver: RungeKutta4::new(&state),
        })
    }
}

impl SEIRModel<RungeKutta4<SEIRState>> {
    /// Returns a fresh builder for $\dot{E} = \beta S I / N - \sigma E, \dot{I} = \sigma E - \gamma I$ (`@anderson1991infectious`, `spec:epidemiology_v1`).
    /// Numerical stability: defers division by $N$ until `build()`. Domain of validity: populate $N > 0, 0 \le I_0 \le N, \beta, \sigma, \gamma \ge 0$.
    #[verified_engine::verified]
    pub fn builder() -> SEIRModelBuilder {
        SEIRModelBuilder::default()
    }

    /// Constructs a validated RK4 [`SEIRModel`] for $(S, E, I, R)$ (`@anderson1991infectious`, `spec:epidemiology_v1`).
    /// Numerical stability: fourth-order $O(\Delta t^4)$ local truncation error under `RungeKutta4`. Domain of validity: finite $N > 0, 0 \le I_0 \le N, \beta, \sigma, \gamma \ge 0$.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_epidemiology::__doctest_facade as math_explorer;
    /// use math_explorer::epidemiology::compartmental::SEIRModel;
    ///
    /// let mut model = SEIRModel::new(10_000.0, 10.0, 0.6, 0.2, 0.1).unwrap();
    /// model.step(1.0);
    /// let st = model.state();
    /// assert!(st.e > 0.0);
    /// assert!((st.s + st.e + st.i + st.r - 10_000.0).abs() < 1e-8);
    /// ```
    #[verified_engine::verified]
    pub fn new(
        n: f64,
        i0: f64,
        beta: f64,
        sigma: f64,
        gamma: f64,
    ) -> Result<SEIRModel<RungeKutta4<SEIRState>>, EpidemiologyError> {
        Self::builder()
            .n(n)
            .i0(i0)
            .beta(beta)
            .sigma(sigma)
            .gamma(gamma)
            .build()
    }
}

impl<S: Solver<SEIRState>> SEIRModel<S> {
    /// Advances $(S, E, I, R)$ by $\Delta t$ under $\dot{E} = \beta S I / N - \sigma E$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: stable under RK4 for $\Delta t (\beta + \sigma + \gamma) < 2.78$. Domain of validity: finite $\Delta t \ge 0$.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        <Self as TimeStepper<SEIRState>>::step(self, dt);
    }

    /// Advances $(S, E, I, R)$ by $\Delta t$ using external integrator $\Phi_{\Delta t}$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: governed by `OtherS` stability region. Domain of validity: finite $\Delta t \ge 0$.
    pub fn step_with<OtherS: Solver<SEIRState>>(&mut self, solver: &mut OtherS, dt: f64) {
        <Self as TimeStepper<SEIRState>>::step_with(self, solver, dt);
    }

    /// Substitutes internal integrator advancing $(S, E, I, R)(t + \Delta t)$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: lossless state move without rounding. Domain of validity: any [`Solver<SEIRState>`].
    pub fn with_solver<NewS: Solver<SEIRState>>(self, new_solver: NewS) -> SEIRModel<NewS> {
        SEIRModel {
            state: self.state,
            dynamics: self.dynamics,
            solver: new_solver,
        }
    }

    /// Returns transmission rate $\beta \ge 0$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: exact scalar copy. Domain of validity: non-negative rate $\beta \ge 0$.
    #[verified_engine::verified]
    pub fn beta(&self) -> f64 {
        self.dynamics.beta
    }

    /// Returns latent progression rate $\sigma \ge 0$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: exact scalar copy. Domain of validity: non-negative rate $\sigma \ge 0$.
    #[verified_engine::verified]
    pub fn sigma(&self) -> f64 {
        self.dynamics.sigma
    }

    /// Returns recovery rate $\gamma \ge 0$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: exact scalar copy. Domain of validity: non-negative rate $\gamma \ge 0$.
    #[verified_engine::verified]
    pub fn gamma(&self) -> f64 {
        self.dynamics.gamma
    }

    /// Returns closed population size $N = S + E + I + R > 0$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: exact scalar copy. Domain of validity: strictly positive $N > 0$.
    #[verified_engine::verified]
    pub fn n(&self) -> f64 {
        self.dynamics.n
    }

    /// Borrows current compartmental state $(S, E, I, R)$ (`@anderson1991infectious`, `spec:epidemiology_v1`). Numerical stability: zero-cost reference borrow. Domain of validity: $S, E, I, R \ge 0$ with $S + E + I + R = N$.
    #[verified_engine::verified]
    pub fn state(&self) -> &SEIRState {
        &self.state
    }
}

impl<S: Solver<SEIRState>> OdeSystem<SEIRState> for SEIRModel<S> {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &SEIRState) -> SEIRState {
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
        let model = SEIRModel::builder()
            .n(1000.0)
            .i0(10.0)
            .beta(0.5)
            .sigma(0.2)
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
        let model = SEIRModel::builder().n(1000.0).build();
        assert!(matches!(
            model,
            Err(EpidemiologyError::MissingParameter { .. })
        ));
    }

    #[test]
    #[verified_engine::verified]
    fn test_seir_step_with_rk4() {
        let n = 1000.0;
        let i0 = 10.0;
        let mut model_std = SEIRModel::new(n, i0, 0.5, 0.2, 0.1).unwrap();
        let mut model_with = SEIRModel::new(n, i0, 0.5, 0.2, 0.1).unwrap();

        let dt = 0.1;
        model_std.step(dt);
        let state = *model_with.state();
        model_with.step_with(&mut RungeKutta4::new(&state), dt);

        assert_eq!(
            model_std.state(),
            model_with.state(),
            "step and step_with(RK4) should yield identical results"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_swapping_solver() {
        let n = 1000.0;
        let i0 = 10.0;
        let model = SEIRModel::new(n, i0, 0.5, 0.2, 0.1).unwrap();

        // Swap from RK4 (default) to Euler
        let state = *model.state();
        let mut model_euler = model.with_solver(Euler::new(&state));

        model_euler.step(0.1);

        // Just check it ran
        assert!(model_euler.state().s <= n);
    }
}
