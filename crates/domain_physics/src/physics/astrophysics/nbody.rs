//! Planar 3-Body Gravitational Dynamics & Figure-8 Choreography (`ArrayState<12>`).
//!
//! Implements canonical Newtonian 3-body gravitation in the plane with generalized
//! coordinates $q = (x_1, y_1, x_2, y_2, x_3, y_3)$ and conjugate momenta
//! $p = (p_{x1}, p_{y1}, p_{x2}, p_{y2}, p_{x3}, p_{y3})$, featuring the periodic
//! zero-angular-momentum Figure-8 choreography discovered by Moore (1993) and proved
//! via variational minimization by Chenciner & Montgomery (2000).

// [cite:graph_parameters_rust]

use pure_math::pure_math::analysis::ode::{
    ArrayState, HamiltonianSystem, OdeSystem, SeparableHamiltonian,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Fundamental period $T$ of the unit-mass ($G = 1, m_i = 1$) Figure-8 choreography (Simó, 2002).
pub const FIGURE_EIGHT_PERIOD: f64 = 6.325_913_98;

const PAIR_OFFSETS: [(usize, usize, usize, usize); 3] = [(0, 1, 2, 3), (0, 1, 4, 5), (2, 3, 4, 5)];
const BODY_MASS_PAIR: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Canonical planar 3-body gravitational Hamiltonian system over `ArrayState<12>`.
///
/// # Example
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::astrophysics::nbody::{FigureEightThreeBody, FIGURE_EIGHT_PERIOD};
/// use math_explorer::pure_math::analysis::ode::{ArrayState, HamiltonianSystem, Yoshida4};
///
/// let sys = FigureEightThreeBody::default();
/// let initial = FigureEightThreeBody::figure_eight_initial_state();
/// let mut state = initial;
/// let e0 = sys.hamiltonian(&state);
/// let mut solver = Yoshida4::new(&ArrayState::zeros());
/// let dt = FIGURE_EIGHT_PERIOD / 1_000.0;
/// for _ in 0..1_000 {
///     solver.step_separable(&sys, &mut state, dt);
/// }
/// assert!(state.max_abs_diff(&initial) < 1e-4);
/// assert!(sys.relative_energy_error(&state, e0) < 1e-8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FigureEightThreeBody {
    /// Masses $[m_1, m_2, m_3]$ of the three bodies.
    pub masses: [f64; 3],
    /// Gravitational constant $G$.
    pub g: f64,
    /// Optional Plummer softening length squared $\epsilon^2 \ge 0$ (`0.0` for exact Newtonian gravity).
    pub softening_sq: f64,
}

impl Default for FigureEightThreeBody {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            masses: [1.0, 1.0, 1.0],
            g: 1.0,
            softening_sq: 0.0,
        }
    }
}

impl FigureEightThreeBody {
    /// Returns the high-precision Chenciner-Montgomery / Simó initial state `[q; p]` (`ArrayState<12>`).
    #[must_use]
    #[verified_engine::verified]
    pub fn figure_eight_initial_state() -> ArrayState<12> {
        let x1 = -0.970_004_36;
        let y1 = 0.243_087_53;
        let vx3 = -0.932_407_37;
        let vy3 = -0.864_731_46;
        let vx1 = -0.5 * vx3;
        let vy1 = -0.5 * vy3;
        ArrayState([x1, y1, -x1, -y1, 0.0, 0.0, vx1, vy1, vx1, vy1, vx3, vy3])
    }

    /// Extracts 2D Cartesian coordinates `[[x1, y1], [x2, y2], [x3, y3]]` from `state`.
    #[must_use]
    #[verified_engine::verified]
    pub fn body_positions(state: &ArrayState<12>) -> [[f64; 2]; 3] {
        [
            [state.0[0], state.0[1]],
            [state.0[2], state.0[3]],
            [state.0[4], state.0[5]],
        ]
    }

    /// Computes conserved total linear momentum $[P_x, P_y] = \sum_{i=1}^3 \mathbf{p}_i$.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_linear_momentum(&self, state: &ArrayState<12>) -> [f64; 2] {
        [
            state.0[6] + state.0[8] + state.0[10],
            state.0[7] + state.0[9] + state.0[11],
        ]
    }

    /// Computes conserved planar angular momentum $L_z = \sum_{i=1}^3 (x_i p_{yi} - y_i p_{xi})$.
    #[must_use]
    #[verified_engine::verified]
    pub fn angular_momentum_z(&self, state: &ArrayState<12>) -> f64 {
        (state.0[0] * state.0[7] - state.0[1] * state.0[6])
            + (state.0[2] * state.0[9] - state.0[3] * state.0[8])
            + (state.0[4] * state.0[11] - state.0[5] * state.0[10])
    }

    /// Constructs a zero-center-of-mass, zero-linear-momentum perturbed Figure-8 orbit from
    /// Poincaré coordinates `(x1, px1)` on the $y_3 = 0$ section.
    #[must_use]
    #[verified_engine::verified]
    pub fn seed_on_poincare_section(&self, x1: f64, px1: f64) -> Option<ArrayState<12>> {
        if !x1.is_finite() || !px1.is_finite() || x1.abs() < 0.2 || x1.abs() > 2.0 {
            return None;
        }
        let mut s = Self::figure_eight_initial_state();
        s.0[0] = x1;
        s.0[2] = -x1;
        s.0[4] = 0.0;
        s.0[6] = px1;
        s.0[8] = px1;
        s.0[10] = -2.0 * px1;
        Some(s)
    }
}

impl OdeSystem<ArrayState<12>> for FigureEightThreeBody {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ArrayState<12>) -> ArrayState<12> {
        let mut out = ArrayState::zeros();
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<12>, out: &mut ArrayState<12>) {
        let mut q = ArrayState::<6>::zeros();
        let mut p = ArrayState::<6>::zeros();
        let mut dq = ArrayState::<6>::zeros();
        let mut dp = ArrayState::<6>::zeros();
        self.split_canonical(state, &mut q, &mut p);
        self.grad_kinetic_in_place(&p, &mut dq);
        self.neg_grad_potential_in_place(&q, &mut dp);
        self.join_canonical(&dq, &dp, out);
    }
}

impl HamiltonianSystem<ArrayState<12>> for FigureEightThreeBody {
    #[verified_engine::verified]
    fn hamiltonian(&self, state: &ArrayState<12>) -> f64 {
        let mut q = ArrayState::<6>::zeros();
        let mut p = ArrayState::<6>::zeros();
        self.split_canonical(state, &mut q, &mut p);
        self.kinetic_energy(&p) + self.potential_energy(&q)
    }
}

impl SeparableHamiltonian<ArrayState<12>> for FigureEightThreeBody {
    type Coord = ArrayState<6>;

    #[verified_engine::verified]
    fn split_canonical(&self, state: &ArrayState<12>, q: &mut Self::Coord, p: &mut Self::Coord) {
        q.0.copy_from_slice(&state.0[0..6]);
        p.0.copy_from_slice(&state.0[6..12]);
    }

    #[verified_engine::verified]
    fn join_canonical(&self, q: &Self::Coord, p: &Self::Coord, state: &mut ArrayState<12>) {
        state.0[0..6].copy_from_slice(&q.0);
        state.0[6..12].copy_from_slice(&p.0);
    }

    #[verified_engine::verified]
    fn kinetic_energy(&self, p: &Self::Coord) -> f64 {
        let k1 = (p.0[0] * p.0[0] + p.0[1] * p.0[1]) / (2.0 * self.masses[0]);
        let k2 = (p.0[2] * p.0[2] + p.0[3] * p.0[3]) / (2.0 * self.masses[1]);
        let k3 = (p.0[4] * p.0[4] + p.0[5] * p.0[5]) / (2.0 * self.masses[2]);
        k1 + k2 + k3
    }

    #[verified_engine::verified]
    fn potential_energy(&self, q: &Self::Coord) -> f64 {
        let mut v = 0.0;
        for (pair_idx, &(ix, iy, jx, jy)) in PAIR_OFFSETS.iter().enumerate() {
            let (mi, mj) = BODY_MASS_PAIR[pair_idx];
            let dx = q.0[ix] - q.0[jx];
            let dy = q.0[iy] - q.0[jy];
            let r = (dx * dx + dy * dy + self.softening_sq).max(1e-24).sqrt();
            v -= self.g * self.masses[mi] * self.masses[mj] / r;
        }
        v
    }

    #[verified_engine::verified]
    fn grad_kinetic_in_place(&self, p: &Self::Coord, dq_dt: &mut Self::Coord) {
        dq_dt.0[0] = p.0[0] / self.masses[0];
        dq_dt.0[1] = p.0[1] / self.masses[0];
        dq_dt.0[2] = p.0[2] / self.masses[1];
        dq_dt.0[3] = p.0[3] / self.masses[1];
        dq_dt.0[4] = p.0[4] / self.masses[2];
        dq_dt.0[5] = p.0[5] / self.masses[2];
    }

    #[verified_engine::verified]
    fn neg_grad_potential_in_place(&self, q: &Self::Coord, dp_dt: &mut Self::Coord) {
        dp_dt.0 = [0.0; 6];
        for (pair_idx, &(ix, iy, jx, jy)) in PAIR_OFFSETS.iter().enumerate() {
            let (mi, mj) = BODY_MASS_PAIR[pair_idx];
            let dx = q.0[jx] - q.0[ix];
            let dy = q.0[jy] - q.0[iy];
            let r2 = (dx * dx + dy * dy + self.softening_sq).max(1e-24);
            let inv_r3 = 1.0 / (r2 * r2.sqrt());
            let scale = self.g * self.masses[mi] * self.masses[mj] * inv_r3;
            let fx = scale * dx;
            let fy = scale * dy;
            dp_dt.0[ix] += fx;
            dp_dt.0[iy] += fy;
            dp_dt.0[jx] -= fx;
            dp_dt.0[jy] -= fy;
        }
    }
}

impl TheoryDescribable for FigureEightThreeBody {
    fn theory_description(&self) -> String {
        format!(
            "Planar 3-body Figure-8 choreography (G={:.2}, period T={:.6})",
            self.g, FIGURE_EIGHT_PERIOD
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Chenciner, A., & Montgomery, R. (2000). A remarkable periodic solution of the three-body problem in the case of equal masses. Annals of Mathematics, 152(3), 881-901.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Planar 3-body Figure-8 choreography".to_string(),
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pure_math::pure_math::analysis::ode::{RungeKutta4, Solver, Yoshida4};

    #[test]
    #[verified_engine::verified]
    fn test_yoshida4_preserves_figure_eight_periodicity_and_invariants() {
        let sys = FigureEightThreeBody::default();
        let initial = FigureEightThreeBody::figure_eight_initial_state();
        let e0 = sys.hamiltonian(&initial);
        let mut state = initial;
        let mut solver = Yoshida4::new(&ArrayState::<6>::zeros());

        let steps = 2_000;
        let dt = FIGURE_EIGHT_PERIOD / steps as f64;
        for _ in 0..steps {
            solver.step_separable(&sys, &mut state, dt);
        }

        let closure_err = state.max_abs_diff(&initial);
        let rel_e_err = sys.relative_energy_error(&state, e0);
        let lin_p = sys.total_linear_momentum(&state);
        let ang_l = sys.angular_momentum_z(&state);

        assert!(
            closure_err < 1e-5,
            "Yoshida4 Figure-8 periodic closure error too large: {closure_err}"
        );
        assert!(
            rel_e_err < 1e-9,
            "Yoshida4 Figure-8 energy drift too large: {rel_e_err}"
        );
        assert!(lin_p[0].abs() < 1e-13 && lin_p[1].abs() < 1e-13);
        assert!(ang_l.abs() < 1e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn test_figure_eight_yoshida4_outperforms_rk4_over_multiple_periods() {
        let sys = FigureEightThreeBody::default();
        let initial = FigureEightThreeBody::figure_eight_initial_state();
        let e0 = sys.hamiltonian(&initial);
        let dt = 0.02;

        let mut s_yoshida = initial;
        let mut s_rk4 = initial;
        let mut yoshida = Yoshida4::new(&ArrayState::<6>::zeros());
        let mut rk4 = RungeKutta4::new(&initial);

        for i in 0..3_000 {
            yoshida.step_separable(&sys, &mut s_yoshida, dt);
            rk4.step(&sys, i as f64 * dt, &mut s_rk4, dt);
        }

        let err_y = sys.relative_energy_error(&s_yoshida, e0);
        let err_rk = sys.relative_energy_error(&s_rk4, e0);
        assert!(err_y < 1e-5, "Yoshida4 bounded drift: {err_y}");
        assert!(
            err_y * 20.0 < err_rk,
            "Yoshida4 ({err_y}) must beat RK4 ({err_rk})"
        );
    }
}
