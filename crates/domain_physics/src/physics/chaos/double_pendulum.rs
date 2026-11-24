//! Canonical Hamiltonian Double Pendulum (`ArrayState<4>`).
//!
//! Models two point masses $(m_1, m_2)$ connected by massless rigid rods $(l_1, l_2)$
//! in a uniform gravitational field $g$ using canonical phase-space coordinates
//! $(\theta_1, \theta_2, p_1, p_2)$. Because the $2 \times 2$ mass matrix depends on
//! $\Delta\theta = \theta_1 - \theta_2$, the Hamiltonian is **non-separable** and showcases
//! implicit symplectic integrators such as [`GaussLegendre4`](pure_math::pure_math::analysis::ode::GaussLegendre4).

// [cite:chaos_theory]

use pure_math::pure_math::analysis::ode::{ArrayState, HamiltonianSystem, OdeSystem};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Conservative planar double pendulum with canonical state $[\theta_1, \theta_2, p_1, p_2]$.
///
/// # Example
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::chaos::double_pendulum::DoublePendulum;
/// use math_explorer::pure_math::analysis::ode::{GaussLegendre4, HamiltonianSystem, Solver};
///
/// let pendulum = DoublePendulum::default();
/// let mut state = DoublePendulum::default_chaotic_state();
/// let e0 = pendulum.hamiltonian(&state);
/// let mut solver = GaussLegendre4::new(&state);
/// for i in 0..200 {
///     solver.step(&pendulum, i as f64 * 0.02, &mut state, 0.02);
/// }
/// assert!(pendulum.relative_energy_error(&state, e0) < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DoublePendulum {
    /// Upper bob mass $m_1$ (kg).
    pub m1: f64,
    /// Lower bob mass $m_2$ (kg).
    pub m2: f64,
    /// Upper rod length $l_1$ (m).
    pub l1: f64,
    /// Lower rod length $l_2$ (m).
    pub l2: f64,
    /// Gravitational acceleration $g$ ($\text{m/s}^2$).
    pub g: f64,
}

impl Default for DoublePendulum {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
        }
    }
}

impl DoublePendulum {
    /// Constructs a canonical double pendulum with explicit physical parameters.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(m1: f64, m2: f64, l1: f64, l2: f64, g: f64) -> Self {
        Self { m1, m2, l1, l2, g }
    }

    /// High-energy chaotic benchmark state $[\theta_1, \theta_2, p_1, p_2] = [2.16, 2.16, 0, 0]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn default_chaotic_state() -> ArrayState<4> {
        ArrayState([2.16, 2.16, 0.0, 0.0])
    }

    /// Computes canonical angular velocities $(\dot{\theta}_1, \dot{\theta}_2) = \nabla_p H(q, p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn angular_velocities(&self, state: &ArrayState<4>) -> (f64, f64) {
        let th1 = state.0[0];
        let th2 = state.0[1];
        let p1 = state.0[2];
        let p2 = state.0[3];
        let delta = th1 - th2;
        let (sd, cd) = delta.sin_cos();
        let eff_mass = self.m1 + self.m2 * sd * sd;
        let dth1 = (self.l2 * p1 - self.l1 * p2 * cd) / (self.l1 * self.l1 * self.l2 * eff_mass);
        let dth2 = ((self.m1 + self.m2) * self.l1 * p2 - self.m2 * self.l2 * p1 * cd)
            / (self.m2 * self.l1 * self.l2 * self.l2 * eff_mass);
        (dth1, dth2)
    }

    /// Computes Cartesian bob positions $((x_1, y_1), (x_2, y_2))$ relative to the pivot $(0, 0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn cartesian_positions(&self, state: &ArrayState<4>) -> ([f64; 2], [f64; 2]) {
        let (s1, c1) = state.0[0].sin_cos();
        let (s2, c2) = state.0[1].sin_cos();
        let x1 = self.l1 * s1;
        let y1 = -self.l1 * c1;
        let x2 = x1 + self.l2 * s2;
        let y2 = y1 - self.l2 * c2;
        ([x1, y1], [x2, y2])
    }

    /// Evaluates non-negative gravitational potential $V(\theta_1, \theta_2)$ referenced to $V(0, 0) = 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn potential_energy(&self, theta1: f64, theta2: f64) -> f64 {
        (self.m1 + self.m2) * self.g * self.l1 * (1.0 - theta1.cos())
            + self.m2 * self.g * self.l2 * (1.0 - theta2.cos())
    }

    /// Reconstructs a canonical state $[0, \theta_2, p_1, p_2]$ on the $\theta_1 = 0, \dot{\theta}_1 \ge 0$
    /// Poincaré surface of section for a prescribed total energy $E$.
    #[must_use]
    #[verified_engine::verified]
    pub fn seed_on_poincare_section(
        &self,
        energy: f64,
        theta2: f64,
        p2: f64,
    ) -> Option<ArrayState<4>> {
        if !energy.is_finite() || !theta2.is_finite() || !p2.is_finite() {
            return None;
        }
        let t_req = energy - self.potential_energy(0.0, theta2);
        if t_req < 0.0 {
            return None;
        }
        let (s2, c2) = theta2.sin_cos();
        let eff_mass = self.m1 + self.m2 * s2 * s2;
        let a = self.m2 * self.l2 * self.l2;
        let b = -2.0 * self.m2 * self.l1 * self.l2 * c2 * p2;
        let c = (self.m1 + self.m2) * self.l1 * self.l1 * p2 * p2
            - 2.0 * self.m2 * self.l1 * self.l1 * self.l2 * self.l2 * eff_mass * t_req;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 || a <= 0.0 {
            return None;
        }
        let p1 = (-b + disc.sqrt()) / (2.0 * a);
        Some(ArrayState([0.0, theta2, p1, p2]))
    }
}

impl OdeSystem<ArrayState<4>> for DoublePendulum {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ArrayState<4>) -> ArrayState<4> {
        let mut out = ArrayState::zeros();
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<4>, out: &mut ArrayState<4>) {
        let th1 = state.0[0];
        let th2 = state.0[1];
        let (dth1, dth2) = self.angular_velocities(state);
        let coupling = self.m2 * self.l1 * self.l2 * dth1 * dth2 * (th1 - th2).sin();
        out.0[0] = dth1;
        out.0[1] = dth2;
        out.0[2] = -coupling - (self.m1 + self.m2) * self.g * self.l1 * th1.sin();
        out.0[3] = coupling - self.m2 * self.g * self.l2 * th2.sin();
    }
}

impl HamiltonianSystem<ArrayState<4>> for DoublePendulum {
    #[verified_engine::verified]
    fn hamiltonian(&self, state: &ArrayState<4>) -> f64 {
        let th1 = state.0[0];
        let th2 = state.0[1];
        let p1 = state.0[2];
        let p2 = state.0[3];
        let (sd, cd) = (th1 - th2).sin_cos();
        let denom =
            2.0 * self.m2 * self.l1 * self.l1 * self.l2 * self.l2 * (self.m1 + self.m2 * sd * sd);
        let numer = self.m2 * self.l2 * self.l2 * p1 * p1
            + (self.m1 + self.m2) * self.l1 * self.l1 * p2 * p2
            - 2.0 * self.m2 * self.l1 * self.l2 * p1 * p2 * cd;
        numer / denom + self.potential_energy(th1, th2)
    }
}

impl TheoryDescribable for DoublePendulum {
    fn theory_description(&self) -> String {
        format!(
            "Canonical Hamiltonian double pendulum (m1={:.2}, m2={:.2}, l1={:.2}, l2={:.2}, g={:.2})",
            self.m1, self.m2, self.l1, self.l2, self.g
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Shinbrot, T., Grebogi, C., Wisdom, J., & Yorke, J. A. (1992). Chaos in a double pendulum. American Journal of Physics, 60(6), 491-499.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Non-separable canonical Hamiltonian double pendulum".to_string(),
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pure_math::pure_math::analysis::ode::{GaussLegendre4, RungeKutta4, Solver};

    #[test]
    #[verified_engine::verified]
    fn test_double_pendulum_gauss_legendre4_beats_rk4_over_10000_steps() {
        let pendulum = DoublePendulum::default();
        let initial = DoublePendulum::default_chaotic_state();
        let e0 = pendulum.hamiltonian(&initial);
        let dt = 0.02;

        let mut s_gl4 = initial;
        let mut s_rk4 = initial;
        let mut gl4 = GaussLegendre4::new(&initial);
        let mut rk4 = RungeKutta4::new(&initial);

        let mut max_gl4_pct = 0.0_f64;
        for step in 0..10_000 {
            let t = step as f64 * dt;
            gl4.step(&pendulum, t, &mut s_gl4, dt);
            rk4.step(&pendulum, t, &mut s_rk4, dt);
            let step_gl4_pct = pendulum.relative_energy_error(&s_gl4, e0) * 100.0;
            if step_gl4_pct > max_gl4_pct {
                max_gl4_pct = step_gl4_pct;
            }
        }

        let rel_gl4_pct = pendulum.relative_energy_error(&s_gl4, e0) * 100.0;
        let rel_rk4_pct = pendulum.relative_energy_error(&s_rk4, e0) * 100.0;

        assert!(
            max_gl4_pct < 0.001 && rel_gl4_pct < 0.001,
            "GaussLegendre4 peak ({max_gl4_pct:.6}%) and final ({rel_gl4_pct:.6}%) error must stay < 0.001%"
        );
        assert!(
            rel_rk4_pct > 3.0,
            "RungeKutta4 expected secular energy loss (>3%), got {rel_rk4_pct:.4}%"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_double_pendulum_poincare_seed_energy_shell() {
        let pendulum = DoublePendulum::default();
        let target_energy = 15.0;
        let seeded = pendulum
            .seed_on_poincare_section(target_energy, 0.35, 0.8)
            .expect("valid state inside energy shell");
        assert!(
            (pendulum.hamiltonian(&seeded) - target_energy).abs() < 1e-12,
            "Seeded state must lie on exact energy shell"
        );
        let (dth1, _) = pendulum.angular_velocities(&seeded);
        assert!(dth1 >= 0.0, "Positive crossing requires dtheta1 >= 0");
        assert!(pendulum.seed_on_poincare_section(1.0, 2.8, 10.0).is_none());
    }
}
