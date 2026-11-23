//! Integration tests for test quantum.
use approx::assert_relative_eq;
use domain_physics::physics::quantum::{
    QuantumOperator, QuantumState, construct_1d_hamiltonian, dft_operator, evolve_state,
    evolve_state_in_place, gaussian_wavepacket, sigma_x, sigma_y, sigma_z, time_evolution_operator,
};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_normalization() {
    let c1 = Complex::new(3.0, 0.0);
    let c2 = Complex::new(4.0, 0.0);
    let vec = DVector::from_vec(vec![c1, c2]);
    let state = QuantumState::new(vec);

    let norm = state.norm();
    assert_relative_eq!(
        norm,
        5.0,
        epsilon = math_commons::registry::TOLERANCE_STANDARD
    );

    let normalized_state = state.normalize();
    assert_relative_eq!(
        normalized_state.norm(),
        1.0,
        epsilon = math_commons::registry::TOLERANCE_STANDARD
    );
}

#[test]
#[verified_engine::verified]
fn test_inner_product() {
    // |0> = [1, 0]
    // |1> = [0, 1]
    let zero = QuantumState::new(DVector::from_vec(vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
    ]));
    let one = QuantumState::new(DVector::from_vec(vec![
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    ]));

    let prod = zero.inner_product(&one);
    assert_relative_eq!(
        prod.re,
        0.0,
        epsilon = math_commons::registry::TOLERANCE_STANDARD
    );
    assert_relative_eq!(
        prod.im,
        0.0,
        epsilon = math_commons::registry::TOLERANCE_STANDARD
    );

    let prod_self = zero.inner_product(&zero);
    assert_relative_eq!(
        prod_self.re,
        1.0,
        epsilon = math_commons::registry::TOLERANCE_STANDARD
    );
}

#[test]
#[verified_engine::verified]
fn test_pauli_commutators() {
    let sx = sigma_x();
    let sy = sigma_y();
    let sz = sigma_z();

    // [Sx, Sy] = i 2 Sz
    let comm_xy = sx.commutator(&sy);
    let expected = &sz.matrix * Complex::new(0.0, 2.0);

    let diff = comm_xy.matrix - expected;
    assert!(diff.norm() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_time_evolution() {
    // Hamiltonian H = w * Sz. w = 1.
    // Time t. U(t) = exp(-i Sz t / h_bar).
    // Sz = diag(1, -1).
    // U(t) = diag(e^{-it}, e^{it}) (assuming h_bar=1)

    let sz = sigma_z();
    let t = std::f64::consts::PI; // t = pi
    let h_bar = 1.0;

    let u = domain_physics::physics::quantum::time_evolution_operator(&sz, t, h_bar);

    // Expected: diag(exp(-i*pi), exp(i*pi)) = diag(-1, -1) = -I
    let result = u.matrix;
    let expected = DMatrix::from_diagonal(&DVector::from_vec(vec![
        Complex::new(-1.0, 0.0),
        Complex::new(-1.0, 0.0),
    ]));

    // Note: e^{-i*pi} = -1. e^{i*pi} = -1.
    // Allow small error due to float arithmetic
    assert!((result - expected).norm() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_dft_unitarity() {
    let n = 4;
    let dft = dft_operator(n);

    // Check if DFT is unitary: U^dagger U = I
    let adjoint = QuantumOperator::new(dft.matrix.adjoint());
    let identity_approx = &adjoint.matrix * &dft.matrix;

    let identity = DMatrix::identity(n, n);
    let i_complex = identity.map(|x| Complex::new(x, 0.0));

    assert!((identity_approx - i_complex).norm() < math_commons::registry::TOLERANCE_STANDARD);
}

#[test]
#[verified_engine::verified]
fn test_evolve_and_normalize_in_place_zero_allocations_and_parity() {
    let n = 16;
    let dx = 0.25;
    let x_grid: Vec<f64> = (0..n).map(|i| -2.0 + i as f64 * dx).collect();
    let potential = DVector::from_fn(n, |i, _| 0.5 * x_grid[i] * x_grid[i]);
    let hamiltonian = construct_1d_hamiltonian(&potential, dx, 1.0, 1.0);
    let dt = 0.03;
    let propagator = time_evolution_operator(&hamiltonian, dt, 1.0);

    let mut psi_in_place = gaussian_wavepacket(&x_grid, -0.5, 2.0, 0.5);
    let mut psi_ref = psi_in_place.clone();
    let mut scratch = DVector::from_element(n, Complex::new(0.0, 0.0));

    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..5 {
            evolve_state_in_place(&mut psi_in_place, &propagator, &mut scratch);
            psi_in_place.normalize_in_place();
        }
    });
    assert!(metrics.function_calls >= 10);

    for _ in 0..5 {
        psi_ref = evolve_state(&psi_ref, &hamiltonian, dt, 1.0).normalize();
    }
    assert!((&psi_in_place.vector - &psi_ref.vector).norm() < 1e-12);
    assert_relative_eq!(
        psi_in_place.norm(),
        1.0,
        epsilon = math_commons::registry::TOLERANCE_STANDARD
    );
}

#[test]
#[verified_engine::verified]
fn test_normalize_in_place_zero_vector_and_exact_parity() {
    let mut zero_state = QuantumState::new(DVector::from_element(3, Complex::new(0.0, 0.0)));
    let ((), _) = VerifiedEngine::run_verified(|| {
        zero_state.normalize_in_place();
    });
    assert_eq!(zero_state.norm(), 0.0);

    let mut state = QuantumState::new(DVector::from_vec(vec![
        Complex::new(3.0, -1.5),
        Complex::new(-2.0, 4.0),
    ]));
    let expected = state.normalize();
    let ((), _) = VerifiedEngine::run_verified(|| {
        state.normalize_in_place();
    });
    assert_eq!(state.vector, expected.vector);
}

#[test]
#[should_panic(expected = "Propagator matrix must be square")]
fn test_evolve_state_in_place_rejects_non_square_operator() {
    let mut state = QuantumState::spin_zero();
    let non_square = QuantumOperator::new(DMatrix::zeros(2, 3));
    let mut scratch = DVector::from_element(2, Complex::new(0.0, 0.0));
    evolve_state_in_place(&mut state, &non_square, &mut scratch);
}

#[test]
#[should_panic(expected = "Scratch vector length must match state vector length")]
fn test_evolve_state_in_place_rejects_mismatched_scratch() {
    let mut state = QuantumState::spin_zero();
    let propagator = time_evolution_operator(&sigma_x(), 0.1, 1.0);
    let mut bad_scratch = DVector::from_element(3, Complex::new(0.0, 0.0));
    evolve_state_in_place(&mut state, &propagator, &mut bad_scratch);
}
