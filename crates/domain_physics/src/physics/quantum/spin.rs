use crate::physics::quantum::types::{QuantumOperator, QuantumState};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::f64::consts::{PI, TAU};

/// Returns the Pauli matrix \sigma_x.
///
/// \sigma_x = [[0, 1], [1, 0]]
#[verified_engine::verified]
pub fn sigma_x() -> QuantumOperator {
    let zero = Complex::new(0.0, 0.0);
    let one = Complex::new(1.0, 0.0);
    let elements = vec![zero, one, one, zero];
    QuantumOperator::new(DMatrix::from_vec(2, 2, elements))
}

/// Returns the Pauli matrix \sigma_y.
///
/// \sigma_y = [[0, -i], [i, 0]]
#[verified_engine::verified]
pub fn sigma_y() -> QuantumOperator {
    let zero = Complex::new(0.0, 0.0);
    let i = Complex::new(0.0, 1.0);
    // Note: DMatrix::from_vec fills column by column.
    // Matrix is [[0, -i], [i, 0]]
    // Col 1: 0, i
    // Col 2: -i, 0
    let elements = vec![zero, i, -i, zero];
    QuantumOperator::new(DMatrix::from_vec(2, 2, elements))
}

/// Returns the Pauli matrix \sigma_z.
///
/// \sigma_z = [[1, 0], [0, -1]]
#[verified_engine::verified]
pub fn sigma_z() -> QuantumOperator {
    let zero = Complex::new(0.0, 0.0);
    let one = Complex::new(1.0, 0.0);
    let neg_one = Complex::new(-1.0, 0.0);
    // Col 1: 1, 0
    // Col 2: 0, -1
    let elements = vec![one, zero, zero, neg_one];
    QuantumOperator::new(DMatrix::from_vec(2, 2, elements))
}

/// Updates a 2-component [`QuantumState`] in place to the normalized SU(2) Bloch state
/// $|\psi(\theta, \phi)\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$
/// without heap allocation when `state.vector.len() == 2`.
///
/// Polar angle `theta` is clamped to $[0, \pi]$ (defaulting non-finite inputs to $0$) and
/// azimuthal angle `phi` is wrapped to $[-\pi, \pi)$.
#[verified_engine::verified]
pub fn set_spin_bloch_angles_in_place(state: &mut QuantumState, theta: f64, phi: f64) {
    let safe_theta = if theta.is_finite() {
        theta.clamp(0.0, PI)
    } else {
        0.0
    };
    let safe_phi = if phi.is_finite() {
        (phi + PI).rem_euclid(TAU) - PI
    } else {
        0.0
    };
    let (sin_half, cos_half) = (0.5 * safe_theta).sin_cos();
    let (sin_phi, cos_phi) = safe_phi.sin_cos();
    let c0 = Complex::new(cos_half, 0.0);
    let c1 = Complex::new(sin_half * cos_phi, sin_half * sin_phi);
    if state.vector.len() == 2 {
        state.vector[0] = c0;
        state.vector[1] = c1;
    } else {
        state.vector = DVector::from_vec(vec![c0, c1]);
    }
}

/// Constructs a normalized spin-1/2 [`QuantumState`] from Bloch polar angle `theta` ($\in [0, \pi]$)
/// and azimuthal angle `phi` (radians):
/// $|\psi(\theta, \phi)\rangle = \cos(\theta/2)|0\rangle + e^{i\phi}\sin(\theta/2)|1\rangle$.
#[verified_engine::verified]
#[must_use]
pub fn spin_from_bloch_angles(theta: f64, phi: f64) -> QuantumState {
    let mut state = QuantumState::spin_zero();
    set_spin_bloch_angles_in_place(&mut state, theta, phi);
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_spin_bloch_angles_match_pauli_expectations_and_zero_alloc() {
        let sx = sigma_x();
        let sy = sigma_y();
        let sz = sigma_z();

        for &(theta, phi) in &[
            (0.0, 0.0),
            (PI, 0.0),
            (0.5 * PI, 0.0),
            (0.5 * PI, 0.5 * PI),
            (1.1, -0.75),
            (2.3, 2.6),
        ] {
            let psi = spin_from_bloch_angles(theta, phi);
            assert!((psi.norm() - 1.0).abs() < 1e-12);
            let ex = sx.expectation_value(&psi).re;
            let ey = sy.expectation_value(&psi).re;
            let ez = sz.expectation_value(&psi).re;
            assert!((ex - theta.sin() * phi.cos()).abs() < 1e-12);
            assert!((ey - theta.sin() * phi.sin()).abs() < 1e-12);
            assert!((ez - theta.cos()).abs() < 1e-12);
        }

        let mut scratch = QuantumState::spin_zero();
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (final_norm, _) = VerifiedEngine::run_verified(|| {
            for k in 0..1_000usize {
                let th = ((k % 180) as f64) * (PI / 180.0);
                let ph = ((k as f64) * 0.07) - PI;
                set_spin_bloch_angles_in_place(&mut scratch, th, ph);
            }
            scratch.vector[0].norm_sqr() + scratch.vector[1].norm_sqr()
        });
        assert!((final_norm - 1.0).abs() < 1e-12);

        set_spin_bloch_angles_in_place(&mut scratch, f64::NAN, f64::INFINITY);
        assert!((sz.expectation_value(&scratch).re - 1.0).abs() < 1e-12);
    }
}
