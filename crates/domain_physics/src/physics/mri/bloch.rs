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
