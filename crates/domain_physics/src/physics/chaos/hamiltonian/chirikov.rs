//! Zero-allocation Chirikov-Taylor Standard Map (`ArrayState<2>`).
//!
//! Models a periodically kicked planar rotor on the canonical 2-torus
//! $(\theta, p) \in [0, 2\pi) \times [0, 2\pi)$:
//! $$ p_{n+1} = p_n + K \sin(\theta_n) \pmod{2\pi}, \qquad \theta_{n+1} = \theta_n + p_{n+1} \pmod{2\pi}. $$
//! Its tangent map has unit determinant $\det DF(\theta, p) = 1$ everywhere, preserving
//! canonical phase-space area while displaying Kolmogorov-Arnold-Moser (KAM) invariant circles,
//! Poincaré-Birkhoff island chains, and global stochastic transport above Greene's threshold
//! $K_c \approx 0.971635406$.

// [cite:chaos_theory]

use pure_math::pure_math::analysis::ode::ArrayState;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;
use std::f64::consts::TAU;

/// Greene's residue-criterion critical stochasticity threshold $K_c$ at which the last
/// golden-mean rotational KAM invariant circle breaks down (Greene, 1979).
pub const CRITICAL_K_GREENE: f64 = 0.971_635_406;

/// Area-preserving Chirikov-Taylor standard map on $[0, 2\pi) \times [0, 2\pi)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChirikovStandardMap {
    /// Non-negative kick strength (stochasticity parameter) $K \ge 0$.
    pub k: f64,
}

impl Default for ChirikovStandardMap {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            k: CRITICAL_K_GREENE,
        }
    }
}

impl ChirikovStandardMap {
    /// Constructs a Chirikov standard map with kick strength `k`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(k: f64) -> Self {
        Self { k }
    }

    /// Validates and wraps a canonical torus coordinate `(\theta, p)` inside $[0, 2\pi] \times [0, 2\pi]$.
    ///
    /// Returns `None` if `theta` or `p` is non-finite or lies outside the fundamental square $[0, 2\pi]^2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn seed_on_torus(theta: f64, p: f64) -> Option<ArrayState<2>> {
        if !theta.is_finite()
            || !p.is_finite()
            || !(0.0..=TAU).contains(&theta)
            || !(0.0..=TAU).contains(&p)
        {
            return None;
        }
        Some(ArrayState([theta.rem_euclid(TAU), p.rem_euclid(TAU)]))
    }

    /// Advances `[theta, p]` in-place by one symplectic kick-drift iteration on $[0, 2\pi)^2$.
    #[verified_engine::verified]
    pub fn step_in_place(&self, state: &mut ArrayState<2>) {
        let theta = state.0[0].rem_euclid(TAU);
        let p_next = (state.0[1] + self.k * theta.sin()).rem_euclid(TAU);
        let theta_next = (theta + p_next).rem_euclid(TAU);
        state.0[0] = theta_next;
        state.0[1] = p_next;
    }

    /// Iterates `initial` into caller-supplied buffer `out` without heap allocation, returning written count.
    #[verified_engine::verified]
    pub fn step_n_into(&self, initial: ArrayState<2>, out: &mut [[f64; 2]]) -> usize {
        if !initial.is_finite() || !self.k.is_finite() {
            return 0;
        }
        let mut cur = ArrayState([initial.0[0].rem_euclid(TAU), initial.0[1].rem_euclid(TAU)]);
        for slot in out.iter_mut() {
            self.step_in_place(&mut cur);
            *slot = cur.0;
        }
        out.len()
    }

    /// Evaluates the $2 \times 2$ symplectic Jacobian matrix $\frac{\partial(\theta_{n+1}, p_{n+1})}{\partial(\theta_n, p_n)}$ at angle `theta`.
    #[must_use]
    #[verified_engine::verified]
    pub fn jacobian(&self, theta: f64) -> [[f64; 2]; 2] {
        let kc = self.k * theta.cos();
        [[1.0 + kc, 1.0], [kc, 1.0]]
    }

    /// Evaluates $\det DF(\theta, p) = (1 + K\cos\theta) - K\cos\theta = 1$, verifying symplectic area preservation.
    #[must_use]
    #[verified_engine::verified]
    pub fn jacobian_determinant(&self, theta: f64) -> f64 {
        let j = self.jacobian(theta);
        j[0][0] * j[1][1] - j[0][1] * j[1][0]
    }

    /// Computes the maximal finite-time Lyapunov exponent $\lambda_{\max}$ over `steps` iterations
    /// using zero-allocation tangent-vector renormalization.
    #[must_use]
    #[verified_engine::verified]
    pub fn finite_time_lyapunov(&self, initial: ArrayState<2>, steps: usize) -> f64 {
        if steps == 0 || !initial.is_finite() || !self.k.is_finite() {
            return 0.0;
        }
        let mut state = ArrayState([initial.0[0].rem_euclid(TAU), initial.0[1].rem_euclid(TAU)]);
        let mut v0 = 1.0_f64;
        let mut v1 = 0.0_f64;
        let mut sum_log = 0.0_f64;

        for _ in 0..steps {
            let j = self.jacobian(state.0[0]);
            let nv0 = j[0][0] * v0 + j[0][1] * v1;
            let nv1 = j[1][0] * v0 + j[1][1] * v1;
            let norm = nv0.hypot(nv1).max(1e-30);
            sum_log += norm.ln();
            v0 = nv0 / norm;
            v1 = nv1 / norm;
            self.step_in_place(&mut state);
        }
        sum_log / (steps as f64)
    }
}

impl TheoryDescribable for ChirikovStandardMap {
    fn theory_description(&self) -> String {
        format!(
            "Chirikov-Taylor area-preserving standard map (K={:.4}, Greene critical Kc={:.6})",
            self.k, CRITICAL_K_GREENE
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Chirikov, B. V. (1979). A universal instability of many-dimensional oscillator systems. Physics Reports, 52(5), 263-379; Greene, J. M. (1979). A method for determining a stochastic transition. Journal of Mathematical Physics, 20(6), 1183-1201.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Area-preserving 2D kicked-rotor symplectic standard map".to_string(),
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_chirikov_area_preservation_and_kam_vs_chaotic_lyapunov() {
        let regular_map = ChirikovStandardMap::new(0.35);
        let chaotic_map = ChirikovStandardMap::new(2.6);

        let seed = ChirikovStandardMap::seed_on_torus(0.2, 1.9).expect("valid torus seed");
        assert!(ChirikovStandardMap::seed_on_torus(-0.1, 1.0).is_none());
        assert!(ChirikovStandardMap::seed_on_torus(1.0, TAU + 0.1).is_none());

        let mut buf = [[0.0_f64; 2]; 256];
        assert_eq!(chaotic_map.step_n_into(seed, &mut buf), 256);
        for &[th, p] in &buf {
            assert!((0.0..TAU).contains(&th) && (0.0..TAU).contains(&p));
            assert!((chaotic_map.jacobian_determinant(th) - 1.0).abs() < 1e-14);
        }

        let lam_reg = regular_map.finite_time_lyapunov(seed, 2_000);
        let lam_chaos = chaotic_map.finite_time_lyapunov(ArrayState([0.05, 0.05]), 2_000);
        assert!(
            lam_reg.abs() < 0.02,
            "Subcritical KAM torus should have near-zero Lyapunov exponent, got {lam_reg}"
        );
        assert!(
            lam_chaos > 0.35,
            "Supercritical chaotic sea should have positive Lyapunov exponent, got {lam_chaos}"
        );
    }
}
