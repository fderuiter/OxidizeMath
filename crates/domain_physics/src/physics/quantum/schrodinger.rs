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
use nalgebra::DVector;
use num_complex::Complex;

/// Calculates the time evolution operator $U(t) = e^{-i\hat{H}t/\hbar}$.
///
/// This implementation assumes the Hamiltonian $\hat{H}$ is time-independent.
/// The operator is computed using matrix exponentiation.
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
    // U(t) = exp(-i * H * t / h_bar)
    let i = Complex::new(0.0, 1.0);
    // Avoid division by zero if h_bar is 0 (though physical h_bar is non-zero)
    let factor = -i * t / h_bar;

    // Scale the Hamiltonian matrix by -it/hbar
    let scaled_h = &hamiltonian.matrix * factor;

    // Compute the matrix exponential: e^A
    // nalgebra's exp() is for square matrices.
    let u_matrix = scaled_h.exp();

    QuantumOperator::new(u_matrix)
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
