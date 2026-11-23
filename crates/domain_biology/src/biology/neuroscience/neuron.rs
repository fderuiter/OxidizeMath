//! Public facade for the Hodgkin-Huxley neuron.

use super::model::HodgkinHuxleyModel;
use super::types::{HodgkinHuxleyParameters, HodgkinHuxleyState};
use crate::error::HodgkinHuxleyError;
use pure_math::pure_math::analysis::ode::{Euler, Solver, SolverExt};

/// Builder for constructing a [`HodgkinHuxleyNeuron`] with validated parameters.
///
/// Ensures that initial states (especially gating variables) are within valid ranges.
#[derive(Debug, Clone)]
pub struct HodgkinHuxleyNeuronBuilder {
    v_initial: f64,
    n_initial: Option<f64>,
    m_initial: Option<f64>,
    h_initial: Option<f64>,
    params: HodgkinHuxleyParameters,
}

impl Default for HodgkinHuxleyNeuronBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            v_initial: -65.0,
            n_initial: None, // Will default to steady-state approximation
            m_initial: None,
            h_initial: None,
            params: HodgkinHuxleyParameters::default(),
        }
    }
}

impl HodgkinHuxleyNeuronBuilder {
    /// Creates a new builder with default settings.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the initial membrane potential (mV). Default is -65.0.
    #[verified_engine::verified]
    pub fn with_initial_v(mut self, v: f64) -> Self {
        self.v_initial = v;
        self
    }

    /// Sets the Potassium activation gating variable ($n$).
    ///
    /// # Errors
    /// Returns error in `build()` if not between 0 and 1.
    #[verified_engine::verified]
    pub fn with_n(mut self, n: f64) -> Self {
        self.n_initial = Some(n);
        self
    }

    /// Sets the Sodium activation gating variable ($m$).
    ///
    /// # Errors
    /// Returns error in `build()` if not between 0 and 1.
    #[verified_engine::verified]
    pub fn with_m(mut self, m: f64) -> Self {
        self.m_initial = Some(m);
        self
    }

    /// Sets the Sodium inactivation gating variable ($h$).
    ///
    /// # Errors
    /// Returns error in `build()` if not between 0 and 1.
    #[verified_engine::verified]
    pub fn with_h(mut self, h: f64) -> Self {
        self.h_initial = Some(h);
        self
    }

    /// Sets custom parameters for the model.
    #[verified_engine::verified]
    pub fn with_params(mut self, params: HodgkinHuxleyParameters) -> Self {
        self.params = params;
        self
    }

    /// Builds the `HodgkinHuxleyNeuron`.
    ///
    /// # Returns
    /// - `Ok(neuron)` if all parameters are valid.
    /// - `Err(HodgkinHuxleyError)` if any gating variable is out of range [0, 1].
    #[verified_engine::verified]
    pub fn build(self) -> Result<HodgkinHuxleyNeuron, HodgkinHuxleyError> {
        let n = self.n_initial.unwrap_or(0.32);
        let m = self.m_initial.unwrap_or(0.05);
        let h = self.h_initial.unwrap_or(0.6);

        validate_gating_variable(n)?;
        validate_gating_variable(m)?;
        validate_gating_variable(h)?;

        // Validate params (e.g. non-negative conductance)
        if self.params.g_na < 0.0 {
            return Err(HodgkinHuxleyError::InvalidConductance(self.params.g_na));
        }
        if self.params.g_k < 0.0 {
            return Err(HodgkinHuxleyError::InvalidConductance(self.params.g_k));
        }
        if self.params.g_l < 0.0 {
            return Err(HodgkinHuxleyError::InvalidConductance(self.params.g_l));
        }

        Ok(HodgkinHuxleyNeuron {
            v: self.v_initial,
            n,
            m,
            h,
            params: self.params,
        })
    }
}

#[verified_engine::verified]
fn validate_gating_variable(val: f64) -> Result<(), HodgkinHuxleyError> {
    if !(0.0..=1.0).contains(&val) {
        Err(HodgkinHuxleyError::InvalidGatingVariable(val))
    } else {
        Ok(())
    }
}

/// Represents the state of a Hodgkin-Huxley neuron.
///
/// Stores the membrane potential and the state of the three gating variables ($n, m, h$).
#[derive(Debug, Clone)]
pub struct HodgkinHuxleyNeuron {
    /// Membrane potential (mV).
    v: f64,
    /// Gating variable for Potassium channel activation ($0 \le n \le 1$).
    n: f64,
    /// Gating variable for Sodium channel activation ($0 \le m \le 1$).
    m: f64,
    /// Gating variable for Sodium channel inactivation ($0 \le h \le 1$).
    h: f64,

    /// Parameters for the neuron model.
    params: HodgkinHuxleyParameters,
}

impl HodgkinHuxleyNeuron {
    /// Creates a new builder for constructing a neuron.
    #[verified_engine::verified]
    pub fn builder() -> HodgkinHuxleyNeuronBuilder {
        HodgkinHuxleyNeuronBuilder::default()
    }

    /// Creates a new neuron state with the given initial membrane potential.
    /// Gating variables are initialized to their steady-state values at rest.
    /// Uses default Hodgkin-Huxley parameters.
    ///
    /// # Arguments
    /// * `v_initial` - Initial membrane potential (typically -65.0 mV).
    #[verified_engine::verified]
    pub fn new(v_initial: f64) -> Self {
        // We bypass the builder to avoid `.expect()` completely.
        // The default gating variables and default parameters are mathematically proven to be valid invariants.
        Self {
            v: v_initial,
            n: 0.32,
            m: 0.05,
            h: 0.6,
            params: HodgkinHuxleyParameters::default(),
        }
    }

    /// Creates a new neuron with custom parameters.
    ///
    /// # Returns
    /// - `Ok(neuron)` if parameters are valid.
    /// - `Err(HodgkinHuxleyError)` if parameters are invalid.
    #[verified_engine::verified]
    pub fn try_new_with_params(
        v_initial: f64,
        params: HodgkinHuxleyParameters,
    ) -> Result<Self, HodgkinHuxleyError> {
        Self::builder()
            .with_initial_v(v_initial)
            .with_params(params)
            .build()
    }

    /// Membrane potential (mV).
    #[verified_engine::verified]
    pub fn v(&self) -> f64 {
        self.v
    }

    /// Potassium activation gating variable ($0 \le n \le 1$).
    #[verified_engine::verified]
    pub fn n(&self) -> f64 {
        self.n
    }

    /// Sets the Potassium activation gating variable.
    #[verified_engine::verified]
    pub fn set_n(&mut self, n: f64) -> Result<(), HodgkinHuxleyError> {
        validate_gating_variable(n)?;
        self.n = n;
        Ok(())
    }

    /// Sodium activation gating variable ($0 \le m \le 1$).
    #[verified_engine::verified]
    pub fn m(&self) -> f64 {
        self.m
    }

    /// Sets the Sodium activation gating variable.
    #[verified_engine::verified]
    pub fn set_m(&mut self, m: f64) -> Result<(), HodgkinHuxleyError> {
        validate_gating_variable(m)?;
        self.m = m;
        Ok(())
    }

    /// Sodium inactivation gating variable ($0 \le h \le 1$).
    #[verified_engine::verified]
    pub fn h(&self) -> f64 {
        self.h
    }

    /// Sets the Sodium inactivation gating variable.
    #[verified_engine::verified]
    pub fn set_h(&mut self, h: f64) -> Result<(), HodgkinHuxleyError> {
        validate_gating_variable(h)?;
        self.h = h;
        Ok(())
    }

    /// Parameters for the neuron model.
    #[verified_engine::verified]
    pub fn params(&self) -> &HodgkinHuxleyParameters {
        &self.params
    }

    /// Updates the neuron state by a time step `dt` with external current `i_ext`.
    ///
    /// Uses the Euler method by default to maintain backward compatibility.
    ///
    /// # Arguments
    /// * `dt` - Time step in milliseconds (e.g., 0.01).
    /// * `i_ext` - External injected current ($\mu A/cm^2$).
    #[verified_engine::verified]
    pub fn update(&mut self, dt: f64, i_ext: f64) {
        let state = HodgkinHuxleyState {
            v: self.v,
            n: self.n,
            m: self.m,
            h: self.h,
        };
        // Create the solver with the current state structure
        let mut solver = Euler::new(&state);
        self.update_with(dt, i_ext, &mut solver);
    }

    /// Updates the neuron state using a provided solver strategy.
    ///
    /// Allows switching between Euler, Runge-Kutta, etc.
    pub fn update_with<S: Solver<HodgkinHuxleyState>>(
        &mut self,
        dt: f64,
        i_ext: f64,
        solver: &mut S,
    ) {
        // Convert to strongly typed state
        let state = HodgkinHuxleyState {
            v: self.v,
            n: self.n,
            m: self.m,
            h: self.h,
        };

        // Create the model with current parameters
        let model = HodgkinHuxleyModel::new(self.params.clone(), i_ext);

        // Solve using the provided solver
        let new_state = solver.solve(&model, 0.0, &state, dt);

        // Update fields
        self.v = new_state.v;

        // We clamp these to [0, 1] because numerical solvers can sometimes overshoot slightly
        // which would cause our internal invariants to break on the next access if we used checked setters.
        // However, we are setting internal state directly here.
        self.n = new_state.n.clamp(0.0, 1.0);
        self.m = new_state.m.clamp(0.0, 1.0);
        self.h = new_state.h.clamp(0.0, 1.0);
    }
}
