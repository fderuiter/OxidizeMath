use super::common::{validate_initial_infected, validate_population, validate_rate};
use crate::error::EpidemiologyError;
use crate::impl_compartmental_ops;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, TimeStepper};
use verified_engine::Theory;

/// State for the SIR Model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SIRState {
    /// Number of susceptible individuals $S(t)$.
    pub s: f64,
    /// Number of actively infectious individuals $I(t)$.
    pub i: f64,
    /// Number of recovered or removed individuals $R(t)$.
    pub r: f64,
}

impl_compartmental_ops!(SIRState, s, i, r);

/// Pure dynamics of the SIR Model.
///
/// This struct holds the parameters and defines the differential equations,
/// but does not hold the simulation state. This allows it to be used
/// as a stateless Strategy or Flyweight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SIRDynamics {
    /// Constant closed population size $N = S + I + R > 0$.
    pub n: f64,
    /// Effective transmission rate $\beta \ge 0$.
    pub beta: f64,
    /// Per-capita recovery rate $\gamma \ge 0$.
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

struct ForcedDynamics<'a, D, F> {
    base: &'a D,
    forcing: F,
}

impl<'a, D: OdeSystem<SIRState>, F: Fn(f64, &SIRState) -> SIRState> OdeSystem<SIRState>
    for ForcedDynamics<'a, D, F>
{
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &SIRState) -> SIRState {
        self.base.derivative(t, state) + (self.forcing)(t, state)
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &SIRState, out: &mut SIRState) {
        self.base.derivative_in_place(t, state, out);
        let source = (self.forcing)(t, state);
        *out += source;
    }
}

/// SIR Model: Susceptible, Infectious, Recovered.
///
/// Equations:
/// $$dS/dt = -\beta S I / N$$
/// $$dI/dt = \beta S I / N - \gamma I$$
/// $$dR/dt = \gamma I$$
///
/// Reference: Kermack & McKendrick (1927), DOI: `10.1098/rspa.1927.0118`; see
/// [`docs/math_specs/domain_epidemiology.md`](../../../../../docs/math_specs/domain_epidemiology.md#kermack1927contribution).
///
/// # Stability
/// Preserves total mass $S(t) + I(t) + R(t) = N$ up to rounding error; explicit RK4 integration
/// requires $\Delta t \cdot \max(\beta, \gamma) \lesssim 1$ to keep compartments non-negative.
///
/// # Domain of Validity
/// Closed homogeneous-mixing populations with $N > 0$, $0 \le I_0 \le N$, and finite rates
/// $\beta \ge 0$, $\gamma \ge 0$.
///
/// Use `SIRModel::builder()` or `SIRModel::new()` to construct.
#[derive(Debug, Clone, PartialEq, Theory)]
#[theory(
    description = "The SIR model is an epidemiological model that computes the theoretical number of people infected with a contagious illness in a closed population over time.",
    citation = "A Contribution to the Mathematical Theory of Epidemics (Kermack & McKendrick, 1927)"
)]
pub struct SIRModel<S: Solver<SIRState> = RungeKutta4<SIRState>> {
    state: SIRState,
    /// The underlying dynamics model (parameters + equations).
    pub dynamics: SIRDynamics,
    /// The numerical solver strategy.
    solver: S,
    time: f64,
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
    fn current_time(&self) -> f64 {
        self.time
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        debug_assert!(dt.is_finite());
        self.time += dt;
    }

    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        debug_assert!(t.is_finite() && dt.is_finite());
        self.solver.step(&self.dynamics, t, &mut self.state, dt);
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let t = self.time;
        self.step_at(t, dt);
        self.advance_time(dt);
    }

    fn step_with_at<OtherS: Solver<SIRState>>(&mut self, solver: &mut OtherS, t: f64, dt: f64) {
        debug_assert!(t.is_finite() && dt.is_finite());
        solver.step(&self.dynamics, t, &mut self.state, dt);
    }

    fn step_with<OtherS: Solver<SIRState>>(&mut self, solver: &mut OtherS, dt: f64) {
        let t = self.time;
        self.step_with_at(solver, t, dt);
        self.advance_time(dt);
    }
}

/// Builder for SIRModel to ensure valid parameter configuration.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SIRModelBuilder {
    n: Option<f64>,
    i0: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
}

impl SIRModelBuilder {
    /// Sets the total population size N.
    #[verified_engine::verified]
    pub fn n(mut self, n: f64) -> Self {
        self.n = Some(n);
        self
    }

    /// Sets the initial infected count I0.
    #[verified_engine::verified]
    pub fn i0(mut self, i0: f64) -> Self {
        self.i0 = Some(i0);
        self
    }

    /// Sets the transmission rate beta.
    #[verified_engine::verified]
    pub fn beta(mut self, beta: f64) -> Self {
        self.beta = Some(beta);
        self
    }

    /// Sets the recovery rate gamma.
    #[verified_engine::verified]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Builds the SIRModel, validating all parameters.
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
            time: 0.0,
        })
    }
}

impl SIRModel<RungeKutta4<SIRState>> {
    /// Returns a new builder for the SIRModel.
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

    /// Constructs a new SIRModel with the given parameters using RungeKutta4.
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
    /// Advances the state by dt using the configured solver.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        <Self as TimeStepper<SIRState>>::step(self, dt);
    }

    /// Advances the state by dt using a provided solver strategy (temporarily ignoring the internal solver).
    pub fn step_with<OtherS: Solver<SIRState>>(&mut self, solver: &mut OtherS, dt: f64) {
        <Self as TimeStepper<SIRState>>::step_with(self, solver, dt);
    }

    /// Advances the state by `dt` with an explicit time-dependent source term $Q(t, \mathbf{y})$ and zero heap allocation.
    #[verified_engine::verified]
    pub fn step_forced<F: Fn(f64, &SIRState) -> SIRState>(&mut self, dt: f64, forcing: F) {
        debug_assert!(dt.is_finite() && self.time.is_finite());
        let forced = ForcedDynamics {
            base: &self.dynamics,
            forcing,
        };
        self.solver.step(&forced, self.time, &mut self.state, dt);
        self.time += dt;
    }

    /// Returns the elapsed simulation time $t$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn time(&self) -> f64 {
        self.time
    }

    /// Sets the simulation clock to `t`.
    #[verified_engine::verified]
    pub fn set_time(&mut self, t: f64) {
        debug_assert!(t.is_finite());
        self.time = t;
    }

    /// Resets the simulation clock to `0.0`.
    #[verified_engine::verified]
    pub fn reset_time(&mut self) {
        self.time = 0.0;
    }

    /// Replaces the current solver with a new one.
    pub fn with_solver<NewS: Solver<SIRState>>(self, new_solver: NewS) -> SIRModel<NewS> {
        SIRModel {
            state: self.state,
            dynamics: self.dynamics,
            solver: new_solver,
            time: self.time,
        }
    }

    /// Returns the transmission rate beta.
    #[verified_engine::verified]
    pub fn beta(&self) -> f64 {
        self.dynamics.beta
    }

    /// Returns the recovery rate gamma.
    #[verified_engine::verified]
    pub fn gamma(&self) -> f64 {
        self.dynamics.gamma
    }

    /// Returns the total population size N.
    #[verified_engine::verified]
    pub fn n(&self) -> f64 {
        self.dynamics.n
    }

    /// Returns the current state.
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
#[path = "sir_tests.rs"]
mod tests;
