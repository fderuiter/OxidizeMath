//! Classical Dynamics Simulator using Bloch Equations.

use super::proton;
use math_commons::error::MathError;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{
    IntegrationMethod, OdeSystem, PreallocatedOdeSolver, Solver, TimeStepper,
};

#[derive(Debug, Clone, Copy)]
struct BlochDynamics {
    m0: f64,
    t1: f64,
    t2: f64,
    b_field: Vector3<f64>,
}

impl OdeSystem<Vector3<f64>> for BlochDynamics {
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

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &Vector3<f64>, out: &mut Vector3<f64>) {
        *out = self.derivative(t, state);
    }
}

/// Classical Dynamics Simulator using Bloch Equations.
#[derive(Debug, Clone)]
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
    /// Active numerical integration scheme honored by `step`.
    pub integration_method: IntegrationMethod,
    t: f64,
    solver: PreallocatedOdeSolver<Vector3<f64>>,
}

impl PartialEq for BlochSimulator {
    #[verified_engine::verified]
    fn eq(&self, other: &Self) -> bool {
        self.magnetization == other.magnetization
            && self.m0 == other.m0
            && self.t1 == other.t1
            && self.t2 == other.t2
            && self.b_field == other.b_field
            && self.integration_method == other.integration_method
            && self.t == other.t
    }
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
        self.t
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        self.t += dt;
    }

    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        let dynamics = self.dynamics();
        self.solver.step_method(
            self.integration_method,
            &dynamics,
            t,
            &mut self.magnetization,
            dt,
        );
    }

    #[verified_engine::verified]
    fn step_with_at<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, t: f64, dt: f64) {
        let dynamics = self.dynamics();
        solver.step(&dynamics, t, &mut self.magnetization, dt);
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let t = self.t;
        self.step_at(t, dt);
        self.t += dt;
    }

    #[verified_engine::verified]
    fn step_with<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, dt: f64) {
        let t = self.t;
        self.step_with_at(solver, t, dt);
        self.t += dt;
    }
}

impl BlochSimulator {
    #[inline]
    #[verified_engine::verified]
    fn dynamics(&self) -> BlochDynamics {
        BlochDynamics {
            m0: self.m0,
            t1: self.t1,
            t2: self.t2,
            b_field: self.b_field,
        }
    }

    /// Creates a new `BlochSimulator` with pre-allocated solver buffers and [`IntegrationMethod::RungeKutta4`].
    #[verified_engine::verified]
    pub fn new(initial_magnetization: Vector3<f64>, m0: f64) -> Self {
        let solver = PreallocatedOdeSolver::new(&initial_magnetization);
        Self {
            magnetization: initial_magnetization,
            m0,
            t1: f64::INFINITY,
            t2: f64::INFINITY,
            b_field: Vector3::zeros(),
            integration_method: IntegrationMethod::RungeKutta4,
            t: 0.0,
            solver,
        }
    }

    /// Creates a validated `BlochSimulator`, rejecting non-finite magnetization, non-finite $M_0$,
    /// or non-positive relaxation constants $T_1, T_2 \le 0$.
    #[verified_engine::verified]
    pub fn try_new(
        initial_magnetization: Vector3<f64>,
        m0: f64,
        t1: f64,
        t2: f64,
    ) -> Result<Self, MathError> {
        if !(initial_magnetization.x.is_finite()
            && initial_magnetization.y.is_finite()
            && initial_magnetization.z.is_finite())
        {
            return Err(MathError::NumericalError {
                reason: "initial_magnetization components must be finite",
            });
        }
        if !m0.is_finite() {
            return Err(MathError::NumericalError {
                reason: "equilibrium magnetization m0 must be finite",
            });
        }
        if t1.is_nan() || t1 <= 0.0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: t1,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        if t2.is_nan() || t2 <= 0.0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: t2,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        let mut sim = Self::new(initial_magnetization, m0);
        sim.set_relaxation(t1, t2);
        Ok(sim)
    }

    /// Configures the numerical [`IntegrationMethod`] in builder style without reallocating buffers.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_integration_method(mut self, method: IntegrationMethod) -> Self {
        self.integration_method = method;
        self
    }

    /// Updates the active [`IntegrationMethod`] in place without reallocating buffers.
    #[verified_engine::verified]
    pub fn set_integration_method(&mut self, method: IntegrationMethod) {
        self.integration_method = method;
    }

    /// Returns the elapsed simulation time $t$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn time(&self) -> f64 {
        self.t
    }

    /// Sets the simulation clock to `t`.
    #[verified_engine::verified]
    pub fn set_time(&mut self, t: f64) {
        self.t = t;
    }

    /// Advances the magnetization vector in place by `dt` using `self.integration_method`.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        <Self as TimeStepper<Vector3<f64>>>::step(self, dt);
    }

    /// Advances the magnetization vector in place by `dt` using a caller-supplied `solver`.
    #[verified_engine::verified]
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
    fn try_new_validates_finite_inputs_and_positive_relaxation_times() {
        assert!(BlochSimulator::try_new(Vector3::new(f64::NAN, 0.0, 1.0), 1.0, 1.0, 0.1).is_err());
        assert!(BlochSimulator::try_new(Vector3::z(), f64::INFINITY, 1.0, 0.1).is_err());
        assert!(BlochSimulator::try_new(Vector3::z(), 1.0, 0.0, 0.1).is_err());
        assert!(BlochSimulator::try_new(Vector3::z(), 1.0, -1.0, 0.1).is_err());
        assert!(BlochSimulator::try_new(Vector3::z(), 1.0, 1.0, 0.0).is_err());
        assert!(BlochSimulator::try_new(Vector3::z(), 1.0, 1.0, f64::NAN).is_err());

        let sim = BlochSimulator::try_new(Vector3::new(1.0, 0.0, 0.0), 1.0, 1.2, 0.08).unwrap();
        assert_eq!(sim.t1, 1.2);
        assert_eq!(sim.t2, 0.08);
        assert_eq!(sim.time(), 0.0);
        assert_eq!(sim.clone(), sim);
        assert!(!format!("{sim:?}").is_empty());
    }

    #[test]
    fn bloch_simulator_tracks_time_and_honors_integration_methods() {
        let m_init = Vector3::new(1.0, 0.0, 0.0);
        let mut sim_inh = BlochSimulator::new(m_init, 1.0)
            .with_integration_method(IntegrationMethod::ImplicitMidpoint);
        sim_inh.set_b_field(Vector3::new(0.0, 0.0, 1e-6));
        let mut sim_tr = sim_inh.clone();

        sim_inh.step(1e-4);
        TimeStepper::step(&mut sim_tr, 1e-4);
        assert_eq!(sim_inh.magnetization, sim_tr.magnetization);
        assert!((sim_inh.time() - 1e-4).abs() < 1e-15);
        assert_eq!(TimeStepper::current_time(&sim_tr), sim_inh.time());
    }
}
