use super::common::{validate_initial_infected, validate_population, validate_rate};
use crate::error::EpidemiologyError;
use crate::impl_compartmental_ops;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, TimeStepper};
use verified_engine::Theory;

/// State for the SIR Model.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SIRState {
    #[allow(missing_docs)]
    pub s: f64,
    #[allow(missing_docs)]
    pub i: f64,
    #[allow(missing_docs)]
    pub r: f64,
}

impl_compartmental_ops!(SIRState, s, i, r);

/// Pure dynamics of the SIR Model.
///
/// This struct holds the parameters and defines the differential equations,
/// but does not hold the simulation state. This allows it to be used
/// as a stateless Strategy or Flyweight.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
/// Use `SIRModel::builder()` or `SIRModel::new()` to construct.
#[derive(Debug, Clone, PartialEq, Theory)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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

/// Builder for SIRModel to ensure valid parameter configuration.
#[derive(Debug, Default, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        })
    }
}

impl SIRModel<RungeKutta4<SIRState>> {
    /// Returns a new builder for the SIRModel.
    #[verified_engine::verified]
    pub fn builder() -> SIRModelBuilder {
        SIRModelBuilder::default()
    }

    /// Constructs a new SIRModel with the given parameters using RungeKutta4.
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

    /// Replaces the current solver with a new one.
    pub fn with_solver<NewS: Solver<SIRState>>(self, new_solver: NewS) -> SIRModel<NewS> {
        SIRModel {
            state: self.state,
            dynamics: self.dynamics,
            solver: new_solver,
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

#[cfg(feature = "serde")]
impl<'de, S> serde::Deserialize<'de> for SIRModel<S>
where
    S: Solver<SIRState> + serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawSIRModel<S> {
            state: SIRState,
            dynamics: SIRDynamics,
            solver: S,
        }

        let raw = RawSIRModel::<S>::deserialize(deserializer)?;
        validate_population(raw.dynamics.n).map_err(serde::de::Error::custom)?;
        validate_rate("beta (transmission rate)", raw.dynamics.beta)
            .map_err(serde::de::Error::custom)?;
        validate_rate("gamma (recovery rate)", raw.dynamics.gamma)
            .map_err(serde::de::Error::custom)?;
        if !raw.state.s.is_finite()
            || !raw.state.i.is_finite()
            || !raw.state.r.is_finite()
            || raw.state.s < 0.0
            || raw.state.i < 0.0
            || raw.state.r < 0.0
        {
            return Err(serde::de::Error::custom(
                "SIRState compartments must be finite and non-negative",
            ));
        }
        Ok(Self {
            state: raw.state,
            dynamics: raw.dynamics,
            solver: raw.solver,
        })
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
