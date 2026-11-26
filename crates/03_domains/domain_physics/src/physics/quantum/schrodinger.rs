//! # Schrödinger Equation
//!
//! This module provides tools for solving the time-dependent Schrödinger equation,
//! which describes how the quantum state of a physical system changes over time.
//!
//! $$ i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = \hat{H}|\psi(t)\rangle $$
//!
//! For a time-independent Hamiltonian $\hat{H}$, the solution is given by the unitary time-evolution operator:
//!
//! $$ |\psi(t)\rangle = e^{-i\hat{H}t/\hbar} |\psi(0)\rangle $$
//!
//! ## Usage
//!
//! The core function `evolve_state` computes the state at time $t$ given an initial state and a Hamiltonian.

use crate::physics::quantum::types::{QuantumOperator, QuantumState};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

#[verified_engine::verified]
fn write_closed_form_2x2_hermitian(
    matrix: &DMatrix<Complex<f64>>,
    omega: f64,
    out: &mut DMatrix<Complex<f64>>,
) -> bool {
    let h00 = matrix[(0, 0)];
    let h01 = matrix[(0, 1)];
    let h10 = matrix[(1, 0)];
    let h11 = matrix[(1, 1)];
    let herm_tol = 1e-14;
    if h00.im.abs() > herm_tol
        || h11.im.abs() > herm_tol
        || (h01.re - h10.re).abs() > herm_tol
        || (h01.im + h10.im).abs() > herm_tol
    {
        return false;
    }
    let b = Complex::new(0.5 * (h01.re + h10.re), 0.5 * (h01.im - h10.im));
    let mu = 0.5 * (h00.re + h11.re);
    let delta = 0.5 * (h00.re - h11.re);
    let gap = delta.hypot(b.norm());
    let (sin_mu, cos_mu) = (omega * mu).sin_cos();
    let phase = Complex::new(cos_mu, sin_mu);
    let theta = omega * gap;
    let (sin_t, cos_t) = theta.sin_cos();
    let kappa = if gap <= f64::MIN_POSITIVE || theta.abs() < 1e-8 {
        omega * (1.0 - (theta * theta) / 6.0)
    } else {
        sin_t / gap
    };
    out[(0, 0)] = phase * Complex::new(cos_t, kappa * delta);
    out[(1, 1)] = phase * Complex::new(cos_t, -kappa * delta);
    out[(0, 1)] = phase * Complex::new(-kappa * b.im, kappa * b.re);
    out[(1, 0)] = phase * Complex::new(kappa * b.im, kappa * b.re);
    true
}

/// Computes the exact closed-form unitary propagator $U(t) = e^{-i\hat{H}t/\hbar}$ in place for any
/// finite $2 \times 2$ Hermitian [`QuantumOperator`], performing zero heap allocations when
/// `out.matrix` is already $2 \times 2$.
///
/// Returns `false` if `hamiltonian.matrix` is not $2 \times 2$ Hermitian or if `t` / `h_bar` is
/// non-finite or `h_bar == 0.0` (filling `out.matrix` with `NaN` on non-finite inputs).
#[verified_engine::verified]
pub fn time_evolution_operator_2x2_into(
    hamiltonian: &QuantumOperator,
    t: f64,
    h_bar: f64,
    out: &mut QuantumOperator,
) -> bool {
    if hamiltonian.matrix.nrows() != 2 || hamiltonian.matrix.ncols() != 2 {
        return false;
    }
    if out.matrix.nrows() != 2 || out.matrix.ncols() != 2 {
        out.matrix = DMatrix::zeros(2, 2);
    }
    let finite_h = hamiltonian
        .matrix
        .as_slice()
        .iter()
        .all(|c| c.re.is_finite() && c.im.is_finite());
    if !t.is_finite() || !h_bar.is_finite() || h_bar == 0.0 || !finite_h {
        out.matrix.fill(Complex::new(f64::NAN, f64::NAN));
        return false;
    }
    let omega = -t / h_bar;
    if !omega.is_finite() {
        out.matrix.fill(Complex::new(f64::NAN, f64::NAN));
        return false;
    }
    write_closed_form_2x2_hermitian(&hamiltonian.matrix, omega, &mut out.matrix)
}

#[verified_engine::verified]
fn try_extract_real_symmetric(matrix: &DMatrix<Complex<f64>>) -> Option<DMatrix<f64>> {
    let n = matrix.nrows();
    if n == 0 || matrix.ncols() != n {
        return None;
    }
    let src = matrix.as_slice();
    if src.iter().any(|c| !c.re.is_finite() || c.im != 0.0) {
        return None;
    }
    for c in 0..n {
        for r in 0..c {
            if matrix[(r, c)].re != matrix[(c, r)].re {
                return None;
            }
        }
    }
    let real_vec: Vec<f64> = src.iter().map(|c| c.re).collect();
    Some(DMatrix::from_vec(n, n, real_vec))
}

#[verified_engine::verified]
fn accumulate_eigenmode_upper(
    u_slice: &mut [Complex<f64>],
    q_col: &[f64],
    cos_t: f64,
    sin_t: f64,
    n: usize,
) {
    for (c, (&qc, u_col)) in q_col.iter().zip(u_slice.chunks_exact_mut(n)).enumerate() {
        if qc == 0.0 {
            continue;
        }
        let qc_re = cos_t * qc;
        let qc_im = sin_t * qc;
        for (u_entry, &qr) in u_col[..=c].iter_mut().zip(&q_col[..=c]) {
            u_entry.re += qr * qc_re;
            u_entry.im += qr * qc_im;
        }
    }
}

#[verified_engine::verified]
fn real_symmetric_propagator(real_h: DMatrix<f64>, omega: f64) -> Option<QuantumOperator> {
    let n = real_h.nrows();
    let eig = nalgebra::SymmetricEigen::new(real_h);
    let evals = eig.eigenvalues.as_slice();
    let q_slice = eig.eigenvectors.as_slice();
    if evals.iter().any(|v| !v.is_finite()) || q_slice.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut u_matrix = DMatrix::<Complex<f64>>::zeros(n, n);
    let u_slice = u_matrix.as_mut_slice();
    for (&lambda, q_col) in evals.iter().zip(q_slice.chunks_exact(n)) {
        let (sin_t, cos_t) = (omega * lambda).sin_cos();
        accumulate_eigenmode_upper(u_slice, q_col, cos_t, sin_t, n);
    }
    for c in 0..n {
        for r in 0..c {
            u_matrix[(c, r)] = u_matrix[(r, c)];
        }
    }
    Some(QuantumOperator::new(u_matrix))
}

/// Calculates the time evolution operator $U(t) = e^{-i\hat{H}t/\hbar}$.
///
/// This implementation assumes the Hamiltonian $\hat{H}$ is time-independent.
/// The operator is computed via orthogonal spectral decomposition (`SymmetricEigen`) for finite
/// real-symmetric Hamiltonians and Padé complex matrix exponentiation otherwise.
///
/// $$ U(t) = \exp\left(-\frac{i}{\hbar} \hat{H} t\right) $$
///
/// # Arguments
/// * `hamiltonian` - Time-independent square Hamiltonian $\hat{H}$.
/// * `t` - Evolution duration $t$.
/// * `h_bar` - Reduced Planck constant $\hbar$.
///
/// # Examples
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::{time_evolution_operator, sigma_z};
/// use std::f64::consts::PI;
/// use math_explorer::num_complex::Complex;
///
/// let h_hat = sigma_z();
/// let u_matrix = time_evolution_operator(&h_hat, PI, 1.0).matrix;
/// let expected_diag = Complex::new(-1.0, 0.0);
/// assert!((u_matrix[(0, 0)] - expected_diag).norm() < math_explorer::registry::TOLERANCE_STANDARD, "U(0,0) incorrect");
/// assert!((u_matrix[(1, 1)] - expected_diag).norm() < math_explorer::registry::TOLERANCE_STANDARD, "U(1,1) incorrect");
/// assert!(u_matrix[(0, 1)].norm() < math_explorer::registry::TOLERANCE_STANDARD, "U(0,1) should be 0");
/// assert!(u_matrix[(1, 0)].norm() < math_explorer::registry::TOLERANCE_STANDARD, "U(1,0) should be 0");
/// ```
#[verified_engine::verified]
pub fn time_evolution_operator(
    hamiltonian: &QuantumOperator,
    t: f64,
    h_bar: f64,
) -> QuantumOperator {
    let nrows = hamiltonian.matrix.nrows();
    let ncols = hamiltonian.matrix.ncols();
    if nrows == 0 && ncols == 0 {
        return QuantumOperator::new(DMatrix::zeros(0, 0));
    }
    let finite_h = hamiltonian
        .matrix
        .as_slice()
        .iter()
        .all(|c| c.re.is_finite() && c.im.is_finite());
    if !t.is_finite() || !h_bar.is_finite() || h_bar == 0.0 || !finite_h {
        return QuantumOperator::new(DMatrix::from_element(
            nrows,
            ncols,
            Complex::new(f64::NAN, f64::NAN),
        ));
    }
    let omega = -t / h_bar;
    if !omega.is_finite() {
        return QuantumOperator::new(DMatrix::from_element(
            nrows,
            ncols,
            Complex::new(f64::NAN, f64::NAN),
        ));
    }
    if nrows == 2 && ncols == 2 {
        let mut u2 = DMatrix::<Complex<f64>>::zeros(2, 2);
        if write_closed_form_2x2_hermitian(&hamiltonian.matrix, omega, &mut u2) {
            return QuantumOperator::new(u2);
        }
    }
    if let Some(op) = try_extract_real_symmetric(&hamiltonian.matrix)
        .and_then(|real_h| real_symmetric_propagator(real_h, omega))
    {
        return op;
    }

    // Fallback for complex Hermitian or general non-symmetric finite operators
    let factor = Complex::new(0.0, omega);
    let scaled_h = &hamiltonian.matrix * factor;
    QuantumOperator::new(scaled_h.exp())
}

/// Evolves a quantum state over time using the Schrödinger equation.
///
/// Computes the new state vector $|\psi(t)\rangle$ by applying the time evolution operator to the initial state $|\psi(0)\rangle$.
///
/// $$ |\psi(t)\rangle = e^{-i\hat{H}t/\hbar} |\psi(0)\rangle $$
///
/// # Arguments
/// * `state` - The initial quantum state $|\psi(0)\rangle$.
/// * `hamiltonian` - The time-independent Hamiltonian operator $\hat{H}$.
/// * `t` - The time duration $t$.
/// * `h_bar` - The reduced Planck constant $\hbar$.
///
/// # Returns
/// The evolved quantum state $|\psi(t)\rangle$.
///
/// # Example
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::schrodinger::evolve_state;
/// use math_explorer::physics::quantum::types::{QuantumState, QuantumOperator};
/// use math_explorer::nalgebra::{DMatrix, DVector};
/// use math_explorer::num_complex::Complex;
///
/// let h_bar = 1.0;
/// let omega = 2.0;
/// let factor = Complex::new(h_bar * omega / 2.0, 0.0);
/// let hamiltonian_matrix = DMatrix::from_row_slice(2, 2, &[
///     Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
///     Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
/// ]) * factor;
/// let hamiltonian = QuantumOperator::new(hamiltonian_matrix);
/// let initial_state = QuantumState::new(DVector::from_vec(vec![
///     Complex::new(1.0, 0.0),
///     Complex::new(0.0, 0.0),
/// ]));
/// let final_state = evolve_state(&initial_state, &hamiltonian, std::f64::consts::PI / omega, h_bar);
/// assert!((final_state.probability_density()[1] - 1.0).abs() < 1e-4);
/// ```
#[verified_engine::verified]
pub fn evolve_state(
    state: &QuantumState,
    hamiltonian: &QuantumOperator,
    t: f64,
    h_bar: f64,
) -> QuantumState {
    let u = time_evolution_operator(hamiltonian, t, h_bar);
    QuantumState::new(&u.matrix * &state.vector)
}

/// Fallibly evolves `state` in place using `propagator` and `scratch` without heap allocations.
#[verified_engine::verified]
pub fn try_evolve_state_in_place(
    state: &mut QuantumState,
    propagator: &QuantumOperator,
    scratch: &mut DVector<Complex<f64>>,
) -> Result<(), crate::physics::quantum::QuantumError> {
    use crate::physics::quantum::QuantumError;
    let dim = state.vector.len();
    if propagator.matrix.nrows() != propagator.matrix.ncols() {
        return Err(QuantumError::DimensionMismatch {
            context: "Propagator matrix must be square",
            expected: propagator.matrix.nrows(),
            actual: propagator.matrix.ncols(),
        });
    }
    if propagator.matrix.ncols() != dim {
        return Err(QuantumError::DimensionMismatch {
            context: "Propagator dimension must match state vector length",
            expected: dim,
            actual: propagator.matrix.ncols(),
        });
    }
    if scratch.len() != dim {
        return Err(QuantumError::DimensionMismatch {
            context: "Scratch vector length must match state vector length",
            expected: dim,
            actual: scratch.len(),
        });
    }
    propagator.matrix.mul_to(&state.vector, scratch);
    core::mem::swap(&mut state.vector, scratch);
    debug_assert_eq!(state.vector.len(), dim);
    Ok(())
}

/// Evolves a quantum state in place using a precomputed unitary propagator $U(\Delta t)$.
///
/// Multiplies `propagator.matrix` by `state.vector` into `scratch` and swaps the buffers,
/// advancing $|\psi(t + \Delta t)\rangle = U(\Delta t)|\psi(t)\rangle$ with zero heap allocations.
///
/// # Arguments
/// * `state` - Mutable quantum state vector $|\psi\rangle$ to evolve in place.
/// * `propagator` - Precomputed square unitary propagator $U(\Delta t) = e^{-i\hat{H}\Delta t/\hbar}$.
/// * `scratch` - Pre-allocated complex vector with length matching `state.vector.len()`.
///
/// # Panics
/// Panics if `propagator.matrix` is not square, or if its dimensions or `scratch.len()` do not
/// match `state.vector.len()`.
///
/// # Example
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::{
///     evolve_state_in_place, sigma_x, time_evolution_operator, QuantumState,
/// };
/// use math_explorer::nalgebra::DVector;
/// use math_explorer::num_complex::Complex;
///
/// let h_hat = sigma_x();
/// let dt = std::f64::consts::PI / 2.0;
/// let propagator = time_evolution_operator(&h_hat, dt, 1.0);
///
/// let mut state = QuantumState::spin_zero();
/// let mut scratch = DVector::from_element(2, Complex::new(0.0, 0.0));
/// evolve_state_in_place(&mut state, &propagator, &mut scratch);
///
/// assert!((state.probability_density()[1] - 1.0).abs() < 1e-6);
/// ```
#[verified_engine::verified]
pub fn evolve_state_in_place(
    state: &mut QuantumState,
    propagator: &QuantumOperator,
    scratch: &mut DVector<Complex<f64>>,
) {
    try_evolve_state_in_place(state, propagator, scratch)
        .expect("evolve_state_in_place dimension mismatch");
}

/// Computes the total squared norm $\sum_i |\psi_i|^2$ of `state` with zero heap allocations.
#[verified_engine::verified]
pub fn state_norm_squared(state: &QuantumState) -> f64 {
    let mut sum = 0.0;
    for amp in state.vector.iter() {
        sum += amp.norm_sqr();
    }
    sum
}

/// Computes the quantum transmission probability $T = \sum_{i \ge \text{split}} |\psi_i|^2 / \sum_i |\psi_i|^2$
/// to the right of `split_idx` with zero heap allocations.
#[verified_engine::verified]
pub fn transmission_probability(state: &QuantumState, split_idx: usize) -> f64 {
    let total = state_norm_squared(state);
    if total <= 1e-15 || split_idx >= state.vector.len() {
        return 0.0;
    }
    let mut transmitted = 0.0;
    for amp in state.vector.as_slice()[split_idx..].iter() {
        transmitted += amp.norm_sqr();
    }
    (transmitted / total).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::quantum::hamiltonian::construct_1d_hamiltonian;

    #[test]
    #[verified_engine::verified]
    fn test_real_symmetric_spectral_propagator_parity_and_complex_fallback() {
        let n = 24;
        let potential = DVector::from_fn(n, |i, _| {
            let x = -2.0 + i as f64 * 0.18;
            0.4 * x * x + if (7..=10).contains(&i) { 9.5 } else { 0.0 }
        });
        let h_real = construct_1d_hamiltonian(&potential, 0.18, 1.0, 1.0);
        let dt = 0.04;
        let fast_u = time_evolution_operator(&h_real, dt, 1.0);
        let ref_u = (&h_real.matrix * Complex::new(0.0, -dt)).exp();
        assert!((&fast_u.matrix - &ref_u).norm() < 1e-11);

        let identity = DMatrix::<Complex<f64>>::identity(n, n);
        let unitarity_err = (fast_u.matrix.adjoint() * &fast_u.matrix - identity).norm();
        assert!(unitarity_err < 1e-12);

        let mut h_complex = h_real.matrix.clone();
        h_complex[(0, 1)] += Complex::new(0.0, 0.35);
        h_complex[(1, 0)] -= Complex::new(0.0, 0.35);
        let op_complex = QuantumOperator::new(h_complex.clone());
        let u_complex = time_evolution_operator(&op_complex, dt, 1.0);
        let expected_complex = (h_complex * Complex::new(0.0, -dt)).exp();
        assert!((u_complex.matrix - expected_complex).norm() < 1e-12);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod proptest_suite {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

    #[test]
    #[verified_engine::verified]
    fn proptest_quantum_propagator_unitarity_and_group_law() {
        let mut runner = TestRunner::new_with_rng(
            Config::with_cases(64),
            TestRng::deterministic_rng(RngAlgorithm::ChaCha),
        );
        let strategy = (
            -3.0f64..3.0f64,
            -3.0f64..3.0f64,
            -2.0f64..2.0f64,
            -2.0f64..2.0f64,
            0.01f64..1.5f64,
            0.01f64..1.5f64,
        );
        runner
            .run(&strategy, |(e0, e1, v_re, v_im, t1, t2)| {
                let h_mat = DMatrix::from_row_slice(
                    2,
                    2,
                    &[
                        Complex::new(e0, 0.0),
                        Complex::new(v_re, v_im),
                        Complex::new(v_re, -v_im),
                        Complex::new(e1, 0.0),
                    ],
                );
                let h = QuantumOperator::new(h_mat);
                let u1 = time_evolution_operator(&h, t1, 1.0);
                let u2 = time_evolution_operator(&h, t2, 1.0);
                let u12 = time_evolution_operator(&h, t1 + t2, 1.0);
                let id = DMatrix::<Complex<f64>>::identity(2, 2);

                prop_assert!((u1.matrix.adjoint() * &u1.matrix - &id).norm() < 1e-11);
                prop_assert!((&u1.matrix * &u2.matrix - &u12.matrix).norm() < 1e-11);

                let psi0 = QuantumState::new(DVector::from_vec(vec![
                    Complex::new(0.6, -0.2),
                    Complex::new(-0.5, 0.6),
                ]));
                let n0 = state_norm_squared(&psi0);
                let psi1 = evolve_state(&psi0, &h, t1, 1.0);
                prop_assert!((state_norm_squared(&psi1) - n0).abs() < 1e-11);
                Ok(())
            })
            .unwrap();
    }
}
