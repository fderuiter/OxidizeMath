//! Classical Dynamics Simulator using Bloch Equations.

use super::proton;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{
    Euler, Heun, IntegrationMethod, OdeSystem, RungeKutta4, Solver, TimeStepper,
};

#[derive(Clone, Copy)]
struct BlochSolverBuffers {
    euler: Euler<Vector3<f64>>,
    heun: Heun<Vector3<f64>>,
    rk4: RungeKutta4<Vector3<f64>>,
}

impl BlochSolverBuffers {
    #[verified_engine::verified]
    fn new(example: &Vector3<f64>) -> Self {
        Self {
            euler: Euler::new(example),
            heun: Heun::new(example),
            rk4: RungeKutta4::new(example),
        }
    }
}

impl PartialEq for BlochSolverBuffers {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for BlochSolverBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlochSolverBuffers").finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
struct BlochDynamics {
    m0: f64,
    t1: f64,
    t2: f64,
    b_field: Vector3<f64>,
}

impl OdeSystem<Vector3<f64>> for BlochDynamics {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        let mut out = Vector3::zeros();
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &Vector3<f64>, out: &mut Vector3<f64>) {
        let gamma = proton::GYROMAGNETIC_RATIO;
        let precession = state.cross(&(self.b_field * gamma));
        let transverse_decay = if self.t2.is_infinite() {
            Vector3::zeros()
        } else {
            Vector3::new(state.x / self.t2, state.y / self.t2, 0.0)
        };
        let longitudinal_recovery = if self.t1.is_infinite() {
            Vector3::zeros()
        } else {
            Vector3::new(0.0, 0.0, (state.z - self.m0) / self.t1)
        };
        *out = precession - transverse_decay - longitudinal_recovery;
    }
}

/// Classical Dynamics Simulator using Bloch Equations.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    integration_method: IntegrationMethod,
    time: f64,
    solvers: BlochSolverBuffers,
}

impl OdeSystem<Vector3<f64>> for BlochSimulator {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        self.dynamics().derivative(t, state)
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &Vector3<f64>, out: &mut Vector3<f64>) {
        self.dynamics().derivative_in_place(t, state, out);
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

    #[verified_engine::verified]
    fn current_time(&self) -> f64 {
        self.time
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        self.time += dt;
    }

    fn step_with_at<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, t: f64, dt: f64) {
        let dynamics = self.dynamics();
        solver.step(&dynamics, t, &mut self.magnetization, dt);
    }

    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        self.step_configured_at(t, dt);
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        self.step_configured(dt);
    }

    fn step_with<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, dt: f64) {
        let t = self.time;
        let dynamics = self.dynamics();
        solver.step(&dynamics, t, &mut self.magnetization, dt);
        self.time += dt;
    }
}

impl BlochSimulator {
    #[verified_engine::verified]
    fn dynamics(&self) -> BlochDynamics {
        BlochDynamics {
            m0: self.m0,
            t1: self.t1,
            t2: self.t2,
            b_field: self.b_field,
        }
    }

    #[verified_engine::verified]
    fn step_configured_at(&mut self, t: f64, dt: f64) {
        let dynamics = self.dynamics();
        match self.integration_method {
            IntegrationMethod::RungeKutta4 => {
                self.solvers
                    .rk4
                    .step(&dynamics, t, &mut self.magnetization, dt);
            }
            IntegrationMethod::Euler => {
                self.solvers
                    .euler
                    .step(&dynamics, t, &mut self.magnetization, dt);
            }
            IntegrationMethod::Heun => {
                self.solvers
                    .heun
                    .step(&dynamics, t, &mut self.magnetization, dt);
            }
        }
    }

    #[verified_engine::verified]
    fn step_configured(&mut self, dt: f64) {
        let t = self.time;
        self.step_configured_at(t, dt);
        self.time += dt;
    }

    /// Creates a new BlochSimulator with pre-allocated solver buffers.
    pub fn new(initial_magnetization: Vector3<f64>, m0: f64) -> Self {
        let solvers = BlochSolverBuffers::new(&initial_magnetization);
        Self {
            magnetization: initial_magnetization,
            m0,
            t1: f64::INFINITY,
            t2: f64::INFINITY,
            b_field: Vector3::zeros(),
            integration_method: IntegrationMethod::RungeKutta4,
            time: 0.0,
            solvers,
        }
    }

    /// Configures the numerical [`IntegrationMethod`] in builder style.
    #[verified_engine::verified]
    pub fn with_integration_method(mut self, method: IntegrationMethod) -> Self {
        self.integration_method = method;
        self
    }

    /// Returns the active [`IntegrationMethod`] used when stepping.
    #[verified_engine::verified]
    pub fn integration_method(&self) -> IntegrationMethod {
        self.integration_method
    }

    /// Updates the active [`IntegrationMethod`] without reallocating scratch buffers.
    #[verified_engine::verified]
    pub fn set_integration_method(&mut self, method: IntegrationMethod) {
        self.integration_method = method;
    }

    /// Returns the elapsed simulation time `t`.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Sets the current simulation time `t`.
    #[verified_engine::verified]
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
    }

    /// Advances the magnetization at explicit simulation time `t` by `dt` using the configured integrator,
    /// without mutating [`Self::time`].
    #[verified_engine::verified]
    pub fn step_at(&mut self, t: f64, dt: f64) {
        self.step_configured_at(t, dt);
    }

    /// Advances the magnetization by `dt` using the configured [`IntegrationMethod`] and pre-allocated buffers.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        self.step_configured(dt);
    }

    /// Advances the magnetization by `dt` using a caller-provided solver strategy.
    pub fn step_with<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, dt: f64) {
        <Self as TimeStepper<Vector3<f64>>>::step_with(self, solver, dt);
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

    /// Advances a pre-allocated grid of `K`-isochromat voxel spin ensembles by `dt` under
    /// longitudinal (`exp_t1`), transverse (`exp_t2`), and rotating-frame z-field offsets `delta_b0`.
    #[verified_engine::verified]
    #[allow(clippy::too_many_arguments)]
    pub fn advance_isochromat_grid<const K: usize>(
        &mut self,
        spins: &mut [[Vector3<f64>; K]],
        rho: &[f64],
        exp_t1: &[f64],
        exp_t2: &[f64],
        delta_b0: &[f64],
        spread_frac: f64,
        dt: f64,
    ) {
        advance_isochromat_grid_kernel(spins, rho, exp_t1, exp_t2, delta_b0, spread_frac, dt);
        if let Some(first) = spins.first().and_then(|ens| ens.first()) {
            self.magnetization = *first;
        }
    }
}

fn advance_isochromat_grid_kernel<const K: usize>(
    spins: &mut [[Vector3<f64>; K]],
    rho: &[f64],
    exp_t1: &[f64],
    exp_t2: &[f64],
    delta_b0: &[f64],
    spread_frac: f64,
    dt: f64,
) {
    let gamma_dt = proton::GYROMAGNETIC_RATIO * dt;
    let center = ((K.saturating_sub(1)) as f64) * 0.5;
    for ((((ens, &m0), &e1), &e2), &b0) in spins
        .iter_mut()
        .zip(rho)
        .zip(exp_t1)
        .zip(exp_t2)
        .zip(delta_b0)
    {
        if m0 <= 0.0 {
            continue;
        }
        if b0 == 0.0 {
            if K == 3 && ens[0] == ens[1] && ens[1] == ens[2] {
                let m = ens[0];
                let next_m = Vector3::new(m.x * e2, m.y * e2, m0 + (m.z - m0) * e1);
                *ens = [next_m; K];
            } else {
                for m in ens.iter_mut() {
                    *m = Vector3::new(m.x * e2, m.y * e2, m0 + (m.z - m0) * e1);
                }
            }
        } else {
            let spread = spread_frac * b0.abs();
            for (s, m) in ens.iter_mut().enumerate() {
                let bz = b0 + ((s as f64) - center) * spread;
                let (sin_th, cos_th) = (gamma_dt * bz).sin_cos();
                *m = Vector3::new(
                    e2 * (m.x * cos_th + m.y * sin_th),
                    e2 * (-m.x * sin_th + m.y * cos_th),
                    m0 + (m.z - m0) * e1,
                );
            }
        }
    }
}

/// Computes the effective magnetic field $\vec{B}_{\text{eff}}$ in the frame rotating at
/// carrier angular frequency $\omega_{\text{rf}}$:
/// $\vec{B}_{\text{eff}} = (B_{1,\text{amp}} \cos\phi,\; B_{1,\text{amp}} \sin\phi,\; B_0 - \omega_{\text{rf}} / \gamma)$.
#[verified_engine::verified]
pub fn rotating_frame_effective_field(
    b0: f64,
    omega_rf: f64,
    b1_amp: f64,
    phase_rad: f64,
) -> Vector3<f64> {
    let delta_bz = b0 - omega_rf / proton::GYROMAGNETIC_RATIO;
    Vector3::new(b1_amp * phase_rad.cos(), b1_amp * phase_rad.sin(), delta_bz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_bloch_integrator_parity_time_and_derives() {
        let dt = 1e-4;
        let b_field = Vector3::new(1e-5, 0.5e-5, 2e-5);
        let methods = [
            IntegrationMethod::Euler,
            IntegrationMethod::Heun,
            IntegrationMethod::RungeKutta4,
        ];
        let mut finals = Vec::new();

        for method in methods {
            let init_m = Vector3::new(0.0, 0.0, 1.0);
            let mut inherent = BlochSimulator::new(init_m, 1.0).with_integration_method(method);
            inherent.set_relaxation(1.0, 0.1);
            let mut via_trait = inherent;
            inherent.set_b_field(b_field);
            via_trait.set_b_field(b_field);
            assert_eq!(inherent, via_trait);
            assert_eq!(inherent.integration_method(), method);

            for _ in 0..5 {
                inherent.step(dt);
                <BlochSimulator as TimeStepper<Vector3<f64>>>::step(&mut via_trait, dt);
            }
            assert_eq!(inherent.magnetization, via_trait.magnetization);
            assert_eq!(inherent.time().to_bits(), via_trait.time().to_bits());
            assert_eq!(inherent, via_trait);

            let mut fresh = BlochSimulator::new(init_m, 1.0).with_integration_method(method);
            fresh.set_relaxation(1.0, 0.1);
            fresh.set_b_field(b_field);
            fresh.magnetization = inherent.magnetization;
            fresh.set_time(inherent.time());
            assert_eq!(inherent, fresh);
            assert!(!format!("{inherent:?}").is_empty());
            finals.push(inherent.magnetization);
        }

        assert_ne!(finals[0], finals[1]);
        assert_ne!(finals[1], finals[2]);
    }
}
