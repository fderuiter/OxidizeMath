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
fn try_extract_real_symmetric(matrix: &DMatrix<Complex<f64>>) -> Option<DMatrix<f64>> {
    let n = matrix.nrows();
    if n < 4 || matrix.ncols() != n {
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
/// The operator is computed via orthogonal spectral decomposition for finite real-symmetric
/// Hamiltonians ($N \ge 4$) and complex matrix exponentiation otherwise.
///
/// $$ U(t) = \exp\left(-\frac{i}{\hbar} \hat{H} t\right) $$
///
/// # Arguments
/// * `hamiltonian` - The Hamiltonian operator $\hat{H}$ (must be a square matrix).
/// * `t` - The time duration $t$ to evolve the system.
/// * `h_bar` - The reduced Planck constant $\hbar$.
///
/// # Returns
/// The unitary operator U(t).
///
/// # Examples
///
/// Calculate the time evolution operator for a spin-1/2 system with Hamiltonian H = \sigma_z.
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::{time_evolution_operator, sigma_z};
/// use std::f64::consts::PI;
/// use math_explorer::num_complex::Complex;
///
/// // 1. Define Hamiltonian H = sigma_z (Spin-1/2 Z-operator)
/// // \sigma_z = |0><0| - |1><1| = diag(1, -1)
/// let h_hat = sigma_z();
///
/// // 2. Calculate U(t) for t = PI, h_bar = 1.0
/// // U(t) = exp(-i * H * t / h_bar)
/// // U(PI) = exp(-i * diag(1, -1) * PI) = diag(exp(-i*PI), exp(i*PI))
/// //       = diag(-1, -1) = -I
/// let t = PI;
/// let h_bar = 1.0;
/// let u_op = time_evolution_operator(&h_hat, t, h_bar);
///
/// // 3. Verify result is approximately -I
/// let u_matrix = u_op.matrix;
/// let expected_diag = Complex::new(-1.0, 0.0);
///
/// // Check diagonal elements
/// assert!((u_matrix[(0, 0)] - expected_diag).norm() < math_explorer::registry::TOLERANCE_STANDARD, "U(0,0) incorrect");
/// assert!((u_matrix[(1, 1)] - expected_diag).norm() < math_explorer::registry::TOLERANCE_STANDARD, "U(1,1) incorrect");
///
/// // Check off-diagonal elements are zero
/// assert!(u_matrix[(0, 1)].norm() < math_explorer::registry::TOLERANCE_STANDARD, "U(0,1) should be 0");
/// assert!(u_matrix[(1, 0)].norm() < math_explorer::registry::TOLERANCE_STANDARD, "U(1,0) should be 0");
/// ```
#[verified_engine::verified]
pub fn time_evolution_operator(
    hamiltonian: &QuantumOperator,
    t: f64,
    h_bar: f64,
) -> QuantumOperator {
    let omega = -t / h_bar;
    if let Some(op) = omega
        .is_finite()
        .then(|| try_extract_real_symmetric(&hamiltonian.matrix))
        .flatten()
        .and_then(|real_h| real_symmetric_propagator(real_h, omega))
    {
        return op;
    }

    // Fallback for small, complex Hermitian, non-symmetric, or non-finite operators
    let i = Complex::new(0.0, 1.0);
    let factor = -i * t / h_bar;
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
///
/// Simulating a 2-level system (Qubit) undergoing Rabi oscillations.
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::quantum::schrodinger::evolve_state;
/// use math_explorer::physics::quantum::types::{QuantumState, QuantumOperator};
/// use math_explorer::nalgebra::{DMatrix, DVector};
/// use math_explorer::num_complex::Complex;
///
/// // 1. Define the Hamiltonian for a spin in a magnetic field (Pauli X)
/// // H = \hbar * omega * sigma_x / 2
/// let h_bar = 1.0;
/// let omega = 2.0;
/// let factor = Complex::new(h_bar * omega / 2.0, 0.0);
/// let hamiltonian_matrix = DMatrix::from_row_slice(2, 2, &[
///     Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), // 0  1
///     Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), // 1  0
/// ]) * factor;
/// let hamiltonian = QuantumOperator::new(hamiltonian_matrix);
///
/// // 2. Initialize state in |0> (Up)
/// let initial_vector = DVector::from_vec(vec![
///     Complex::new(1.0, 0.0),
///     Complex::new(0.0, 0.0)
/// ]);
/// let initial_state = QuantumState::new(initial_vector);
///
/// // 3. Evolve for time t = pi / omega (Should flip to |1>)
/// let t = std::f64::consts::PI / omega;
/// let final_state = evolve_state(&initial_state, &hamiltonian, t, h_bar);
///
/// // 4. Check probability of being in |1> (Down)
/// let prob_down = final_state.probability_density()[1];
/// assert!((prob_down - 1.0).abs() < 1e-4);
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
    let dim = state.vector.len();
    assert_eq!(
        propagator.matrix.nrows(),
        propagator.matrix.ncols(),
        "Propagator matrix must be square"
    );
    assert_eq!(
        propagator.matrix.ncols(),
        dim,
        "Propagator dimension must match state vector length"
    );
    assert_eq!(
        scratch.len(),
        dim,
        "Scratch vector length must match state vector length"
    );
    propagator.matrix.mul_to(&state.vector, scratch);
    std::mem::swap(&mut state.vector, scratch);
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
