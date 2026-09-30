//! Classical Dynamics Simulator using Bloch Equations.

use super::proton;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{OdeSystem, TimeStepper};

/// Classical Dynamics Simulator using Bloch Equations.
pub struct BlochSimulator {
    /// Current magnetization vector $\vec{M} = (M_x, M_y, M_z)$.
    pub magnetization: Vector3<f64>,
    /// Equilibrium magnetization $M_0$ (aligned with z-axis).
    pub m0: f64,
    /// Longitudinal relaxation time ($T_1$) in seconds.
    pub t1: f64,
    /// Transverse relaxation time ($T_2$) in seconds.
    pub t2: f64,
    /// External magnetic field vector $\vec{B}$ in Tesla.
    pub b_field: Vector3<f64>,
    /// Time step `dt`.
    pub dt: f64,
    /// Numerical integration method.
    pub integration_method: pure_math::pure_math::analysis::ode::IntegrationMethod,
}

impl OdeSystem<Vector3<f64>> for BlochSimulator {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        let gamma = proton::GYROMAGNETIC_RATIO;

        // Precession term: M x (gamma B)
        let precession = state.cross(&(self.b_field * gamma));

        // Relaxation terms
        // Transverse relaxation (x and y components decay with T2)
        let transverse_decay = if self.t2.is_infinite() {
            Vector3::zeros()
        } else {
            Vector3::new(state.x / self.t2, state.y / self.t2, 0.0)
        };

        // Longitudinal relaxation (z component recovers to M0 with T1)
        let longitudinal_recovery = if self.t1.is_infinite() {
            Vector3::zeros()
        } else {
            Vector3::new(0.0, 0.0, (state.z - self.m0) / self.t1)
        };

        // Total derivative dM/dt
        precession - transverse_decay - longitudinal_recovery
    }
}

impl TimeStepper<Vector3<f64>> for BlochSimulator {
    #[verified_engine::verified]
    fn get_state(&self) -> &Vector3<f64> {
        &self.magnetization
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut Vector3<f64> {
        &mut self.magnetization
    }
}

impl BlochSimulator {
    /// Creates a new BlochSimulator.
    pub fn new(initial_magnetization: Vector3<f64>, m0: f64) -> Self {
        Self {
            magnetization: initial_magnetization,
            m0,
            t1: f64::INFINITY,
            t2: f64::INFINITY,
            b_field: Vector3::zeros(),
            dt: 0.01,
            integration_method: pure_math::pure_math::analysis::ode::IntegrationMethod::RungeKutta4,
        }
    }

    /// Sets the magnetic field vector $\vec{B}$.
    #[verified_engine::verified]
    pub fn set_b_field(&mut self, b_field: Vector3<f64>) {
        self.b_field = b_field;
    }

    /// Sets the relaxation times $T_1$ and $T_2$.
    #[verified_engine::verified]
    pub fn set_relaxation(&mut self, t1: f64, t2: f64) {
        self.t1 = t1;
        self.t2 = t2;
    }
}

use oxidize_core::{ModelConfig, ModelState, SimulationModel};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct BlochConfig {
    #[allow(missing_docs)]
    pub t1: f64,
    #[allow(missing_docs)]
    pub t2: f64,
    #[allow(missing_docs)]
    pub b0: f64, // Not directly used in simple relaxation, but typical
    #[allow(missing_docs)]
    pub dt: f64,
    #[allow(missing_docs)]
    pub m0: f64,
    #[serde(default)]
    #[allow(missing_docs)]
    pub integration_method: pure_math::pure_math::analysis::ode::IntegrationMethod,
}

impl ModelConfig for BlochConfig {}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct BlochState {
    #[allow(missing_docs)]
    pub x: f64,
    #[allow(missing_docs)]
    pub y: f64,
    #[allow(missing_docs)]
    pub z: f64,
}

impl ModelState for BlochState {}

impl SimulationModel for BlochSimulator {
    type Config = BlochConfig;
    type State = BlochState;
    type Error = std::io::Error;

    #[verified_engine::verified]
    fn initialize<R: rand::RngCore>(
        config: Self::Config,
        _provider: R,
    ) -> Result<Self, Self::Error> {
        let initial_m = Vector3::new(0.0, 1.0, 0.0); // Default 90 deg flipped
        let mut sim = BlochSimulator::new(initial_m, config.m0);
        sim.set_relaxation(config.t1, config.t2);
        sim.dt = config.dt;
        sim.integration_method = config.integration_method;
        Ok(sim)
    }

    #[verified_engine::verified]
    fn step(&mut self) -> Result<(), Self::Error> {
        let dt = self.dt;
        match self.integration_method {
            pure_math::pure_math::analysis::ode::IntegrationMethod::Euler => {
                let mut solver =
                    pure_math::pure_math::analysis::ode::Euler::new(&self.magnetization);
                <Self as pure_math::pure_math::analysis::ode::TimeStepper<Vector3<f64>>>::step_with(
                    self,
                    &mut solver,
                    dt,
                );
            }
            pure_math::pure_math::analysis::ode::IntegrationMethod::RungeKutta4 => {
                <Self as pure_math::pure_math::analysis::ode::TimeStepper<Vector3<f64>>>::step(
                    self, dt,
                );
            }
        }
        Ok(())
    }

    #[verified_engine::verified]
    fn get_state(&self) -> Self::State {
        BlochState {
            x: self.magnetization.x,
            y: self.magnetization.y,
            z: self.magnetization.z,
        }
    }
}
