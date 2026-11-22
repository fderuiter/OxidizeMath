use super::common::{validate_initial_infected, validate_population, validate_rate};
use crate::error::EpidemiologyError;
use crate::impl_compartmental_ops;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, TimeStepper};
use verified_engine::Theory;

/// State for the SEIR Model.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Pure dynamics of the SEIR Model.
///
/// This struct holds the parameters and defines the differential equations,
/// but does not hold the simulation state.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
/// Use `SEIRModel::builder()` or `SEIRModel::new()` to construct.
#[derive(Debug, Clone, PartialEq, Theory)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[theory(
    description = "The SEIR model is a compartmental epidemic model that incorporates an exposed (latent) period before individuals become infectious.",
    citation = "Mathematical epidemiology of infectious diseases (Diekmann & Heesterbeek, 2000)"
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

/// Builder for SEIRModel to ensure valid parameter configuration.
#[derive(Debug, Default, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SEIRModelBuilder {
    n: Option<f64>,
    i0: Option<f64>,
    beta: Option<f64>,
    sigma: Option<f64>,
    gamma: Option<f64>,
}

impl SEIRModelBuilder {
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

    /// Sets the incubation rate sigma.
    #[verified_engine::verified]
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = Some(sigma);
        self
    }

    /// Sets the recovery rate gamma.
    #[verified_engine::verified]
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Builds the SEIRModel, validating all parameters.
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
    /// Returns a new builder for the SEIRModel.
    #[verified_engine::verified]
    pub fn builder() -> SEIRModelBuilder {
        SEIRModelBuilder::default()
    }

    /// Constructs a new SEIRModel with the given parameters using RungeKutta4.
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
    /// Advances the state by dt using the configured solver.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        <Self as TimeStepper<SEIRState>>::step(self, dt);
    }

    /// Advances the state by dt using a provided solver strategy.
    pub fn step_with<OtherS: Solver<SEIRState>>(&mut self, solver: &mut OtherS, dt: f64) {
        <Self as TimeStepper<SEIRState>>::step_with(self, solver, dt);
    }

    /// Replaces the current solver with a new one.
    pub fn with_solver<NewS: Solver<SEIRState>>(self, new_solver: NewS) -> SEIRModel<NewS> {
        SEIRModel {
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

    /// Returns the incubation rate sigma.
    #[verified_engine::verified]
    pub fn sigma(&self) -> f64 {
        self.dynamics.sigma
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

#[cfg(feature = "serde")]
impl<'de, S> serde::Deserialize<'de> for SEIRModel<S>
where
    S: Solver<SEIRState> + serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct RawSEIRModel<S> {
            state: SEIRState,
            dynamics: SEIRDynamics,
            solver: S,
        }

        let raw = RawSEIRModel::<S>::deserialize(deserializer)?;
        validate_population(raw.dynamics.n).map_err(serde::de::Error::custom)?;
        validate_rate("beta (transmission rate)", raw.dynamics.beta)
            .map_err(serde::de::Error::custom)?;
        validate_rate("sigma (incubation rate)", raw.dynamics.sigma)
            .map_err(serde::de::Error::custom)?;
        validate_rate("gamma (recovery rate)", raw.dynamics.gamma)
            .map_err(serde::de::Error::custom)?;
        if !raw.state.s.is_finite()
            || !raw.state.e.is_finite()
            || !raw.state.i.is_finite()
            || !raw.state.r.is_finite()
            || raw.state.s < 0.0
            || raw.state.e < 0.0
            || raw.state.i < 0.0
            || raw.state.r < 0.0
        {
            return Err(serde::de::Error::custom(
                "SEIRState compartments must be finite and non-negative",
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
