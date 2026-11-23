//! Canonical Hénon-Heiles Galactic Potential (`ArrayState<4>`).
//!
//! Models stellar motion in an axisymmetric galactic potential with cubic anharmonic
//! coupling (Hénon & Heiles, 1964):
//! $$ H(x, y, p_x, p_y) = \frac{1}{2}(p_x^2 + p_y^2) + \frac{1}{2}(x^2 + y^2) + \lambda\left(x^2 y - \frac{1}{3}y^3\right) $$
//! Because $H(q, p) = T(p) + V(q)$ is separable, explicit symplectic integrators such as
//! [`StormerVerlet`](pure_math::pure_math::analysis::ode::StormerVerlet) and
//! [`Yoshida4`](pure_math::pure_math::analysis::ode::Yoshida4) preserve bounded shadow Hamiltonians,
//! revealing crisp KAM stability islands surrounded by chaotic seas on the $x = 0, p_x > 0$
//! Poincaré surface of section.

// [cite:chaos_theory]

use pure_math::pure_math::analysis::ode::{
    ArrayState, HamiltonianSystem, OdeSystem, SeparableHamiltonian,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Canonical Hénon-Heiles Hamiltonian system over `[x, y, px, py]`.
///
/// # Example
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::chaos::henon_heiles::HenonHeiles;
/// use math_explorer::pure_math::analysis::ode::{ArrayState, HamiltonianSystem, Yoshida4};
///
/// let hh = HenonHeiles::default();
/// let mut state = hh.seed_on_poincare_section(0.125, 0.1, 0.0).unwrap();
/// let e0 = hh.hamiltonian(&state);
/// let mut solver = Yoshida4::new(&ArrayState([0.0, 0.0]));
/// for _ in 0..1_000 {
///     solver.step_separable(&hh, &mut state, 0.02);
/// }
/// assert!(hh.relative_energy_error(&state, e0) < 1e-7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HenonHeiles {
    /// Cubic anharmonicity coupling parameter $\lambda$ (standard value `1.0`).
    pub lambda: f64,
}

impl Default for HenonHeiles {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { lambda: 1.0 }
    }
}

impl HenonHeiles {
    /// Constructs a Hénon-Heiles system with coupling constant `lambda`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Saddle escape energy $E_{\text{esc}} = \frac{1}{6 \lambda^2}$ above which Hill's region opens.
    #[must_use]
    #[verified_engine::verified]
    pub fn escape_energy(&self) -> f64 {
        let l_sq = self.lambda * self.lambda;
        if l_sq > 1e-15 {
            1.0 / (6.0 * l_sq)
        } else {
            f64::INFINITY
        }
    }

    /// Evaluates potential energy $V(x, y) = \frac{1}{2}(x^2 + y^2) + \lambda(x^2 y - \frac{1}{3}y^3)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn potential_xy(&self, x: f64, y: f64) -> f64 {
        0.5 * (x * x + y * y) + self.lambda * (x * x * y - (y * y * y) / 3.0)
    }

    /// Maximum admissible $|p_y|$ on the $x = 0$ Poincaré section at coordinate `y` and energy `E`.
    #[must_use]
    #[verified_engine::verified]
    pub fn max_py_at_section(&self, energy: f64, y: f64) -> Option<f64> {
        let rem = 2.0 * (energy - self.potential_xy(0.0, y));
        (rem >= 0.0 && rem.is_finite()).then(|| rem.sqrt())
    }

    /// Seeds an exact canonical state `[0.0, y, px, py]` with $p_x > 0$ on the constant-energy shell $H = E$.
    #[must_use]
    #[verified_engine::verified]
    pub fn seed_on_poincare_section(&self, energy: f64, y: f64, py: f64) -> Option<ArrayState<4>> {
        if !energy.is_finite() || !y.is_finite() || !py.is_finite() {
            return None;
        }
        let px_sq = 2.0 * (energy - self.potential_xy(0.0, y)) - py * py;
        if px_sq <= 1e-14 {
            return None;
        }
        Some(ArrayState([0.0, y, px_sq.sqrt(), py]))
    }
}

impl OdeSystem<ArrayState<4>> for HenonHeiles {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ArrayState<4>) -> ArrayState<4> {
        let mut out = ArrayState::zeros();
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<4>, out: &mut ArrayState<4>) {
        let x = state.0[0];
        let y = state.0[1];
        out.0[0] = state.0[2];
        out.0[1] = state.0[3];
        out.0[2] = -(x + 2.0 * self.lambda * x * y);
        out.0[3] = -(y + self.lambda * (x * x - y * y));
    }
}

impl HamiltonianSystem<ArrayState<4>> for HenonHeiles {
    #[verified_engine::verified]
    fn hamiltonian(&self, state: &ArrayState<4>) -> f64 {
        let x = state.0[0];
        let y = state.0[1];
        let px = state.0[2];
        let py = state.0[3];
        0.5 * (px * px + py * py) + self.potential_xy(x, y)
    }
}

impl SeparableHamiltonian<ArrayState<4>> for HenonHeiles {
    type Coord = ArrayState<2>;

    #[verified_engine::verified]
    fn split_canonical(&self, state: &ArrayState<4>, q: &mut Self::Coord, p: &mut Self::Coord) {
        q.0[0] = state.0[0];
        q.0[1] = state.0[1];
        p.0[0] = state.0[2];
        p.0[1] = state.0[3];
    }

    #[verified_engine::verified]
    fn join_canonical(&self, q: &Self::Coord, p: &Self::Coord, state: &mut ArrayState<4>) {
        state.0[0] = q.0[0];
        state.0[1] = q.0[1];
        state.0[2] = p.0[0];
        state.0[3] = p.0[1];
    }

    #[verified_engine::verified]
    fn kinetic_energy(&self, p: &Self::Coord) -> f64 {
        0.5 * (p.0[0] * p.0[0] + p.0[1] * p.0[1])
    }

    #[verified_engine::verified]
    fn potential_energy(&self, q: &Self::Coord) -> f64 {
        self.potential_xy(q.0[0], q.0[1])
    }

    #[verified_engine::verified]
    fn grad_kinetic_in_place(&self, p: &Self::Coord, dq_dt: &mut Self::Coord) {
        dq_dt.0[0] = p.0[0];
        dq_dt.0[1] = p.0[1];
    }

    #[verified_engine::verified]
    fn neg_grad_potential_in_place(&self, q: &Self::Coord, dp_dt: &mut Self::Coord) {
        let x = q.0[0];
        let y = q.0[1];
        dp_dt.0[0] = -(x + 2.0 * self.lambda * x * y);
        dp_dt.0[1] = -(y + self.lambda * (x * x - y * y));
    }
}

impl TheoryDescribable for HenonHeiles {
    fn theory_description(&self) -> String {
        format!(
            "Hénon-Heiles canonical galactic Hamiltonian (lambda={:.3}, escape energy={:.4})",
            self.lambda,
            self.escape_energy()
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Hénon, M., & Heiles, C. (1964). The applicability of the third integral of motion: Some numerical experiments. The Astronomical Journal, 69, 73-79.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Separable Hénon-Heiles Hamiltonian system".to_string(),
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pure_math::pure_math::analysis::ode::{StormerVerlet, Yoshida4};

    #[test]
    #[verified_engine::verified]
    fn test_henon_heiles_yoshida4_and_verlet_energy_invariants() {
        let hh = HenonHeiles::default();
        assert!((hh.escape_energy() - 1.0 / 6.0).abs() < 1e-14);

        for energy in [1.0 / 12.0, 0.125] {
            let initial = hh
                .seed_on_poincare_section(energy, 0.12, 0.05)
                .expect("valid Hénon-Heiles seed");
            assert!((hh.hamiltonian(&initial) - energy).abs() < 1e-14);

            let mut s_yoshida = initial;
            let mut s_verlet = initial;
            let mut yoshida = Yoshida4::new(&ArrayState([0.0, 0.0]));
            let mut verlet = StormerVerlet::new(&ArrayState([0.0, 0.0]));

            for _ in 0..5_000 {
                yoshida.step_separable(&hh, &mut s_yoshida, 0.02);
                verlet.step_separable(&hh, &mut s_verlet, 0.02);
            }

            let err_y = hh.relative_energy_error(&s_yoshida, energy);
            let err_v = hh.relative_energy_error(&s_verlet, energy);
            assert!(
                err_y < 1e-7,
                "Yoshida4 Hénon-Heiles drift too large: {err_y}"
            );
            assert!(err_v < 5e-4, "Verlet Hénon-Heiles drift too large: {err_v}");
            assert!(err_y * 20.0 < err_v);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_henon_heiles_hill_boundary_rejection() {
        let hh = HenonHeiles::default();
        assert!(hh.seed_on_poincare_section(0.125, 0.0, 0.0).is_some());
        assert!(hh.seed_on_poincare_section(0.125, 0.9, 0.0).is_none());
        assert!(hh.max_py_at_section(0.125, 0.9).is_none());
    }
}
