use super::*;
use crate::physics::quantum::hamiltonian::{construct_1d_hamiltonian, gaussian_wavepacket};
use crate::physics::quantum::schrodinger::evolve_state;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_crank_nicolson_unitarity_and_dense_parity() {
    let n = 48;
    let dx = 0.15;
    let x_grid: Vec<f64> = (0..n).map(|i| -3.5 + i as f64 * dx).collect();
    let potential = DVector::from_fn(n, |i, _| 0.5 * x_grid[i] * x_grid[i]);
    let tri_h = TridiagonalHamiltonian1D::from_potential(&potential, dx, 1.0, 1.0);
    let dense_h = construct_1d_hamiltonian(&potential, dx, 1.0, 1.0);

    let psi0 = gaussian_wavepacket(&x_grid, -0.8, 2.2, 0.55);
    let e_tri = tri_h.expectation_value(&psi0);
    let e_dense = dense_h.expectation_value(&psi0).re;
    assert!((e_tri - e_dense).abs() < 1e-12);

    let dt = 0.02;
    let substeps = 4;
    let mut cn_prop = CrankNicolsonPropagator1D::new(&tri_h, dt / substeps as f64, 1.0);
    let mut cn_solver =
        CrankNicolson1D::new(potential.as_slice(), dx, dt / substeps as f64, 1.0, 1.0);
    assert!(cn_solver.is_valid());
    assert_eq!(cn_solver.workspace().capacity(), n);

    let mut psi_prop = psi0.clone();
    let mut psi_solver = psi0.clone();
    let (ok_steps, _) = VerifiedEngine::run_verified(|| {
        let mut all_ok = true;
        for _ in 0..10 {
            cn_prop.evolve_substeps_in_place(&mut psi_prop, substeps);
            all_ok &= cn_solver.step_substeps_in_place(&mut psi_solver, substeps);
        }
        all_ok
    });
    assert!(ok_steps);
    assert!((psi_prop.norm() - 1.0).abs() < 1e-13);
    assert!((psi_solver.norm() - 1.0).abs() < 1e-13);
    assert!((&psi_prop.vector - &psi_solver.vector).norm() < 1e-14);

    let psi_exact = evolve_state(&psi0, &dense_h, 10.0 * dt, 1.0);
    assert!((&psi_solver.vector - &psi_exact.vector).norm() < 5e-3);
}

#[test]
#[verified_engine::verified]
fn test_empty_single_point_and_invalid_dt_dx_edge_cases() {
    let empty_v = DVector::<f64>::zeros(0);
    let empty_h = TridiagonalHamiltonian1D::from_potential(&empty_v, 0.1, 1.0, 1.0);
    assert!(empty_h.is_empty());
    let mut empty_prop = CrankNicolsonPropagator1D::new(&empty_h, 0.01, 1.0);
    assert!(empty_prop.is_empty());
    let mut empty_state = QuantumState::new(DVector::zeros(0));
    empty_prop.evolve_in_place(&mut empty_state);
    assert_eq!(empty_state.vector.len(), 0);

    let mut empty_solver = CrankNicolson1D::new(&[], 0.1, 0.01, 1.0, 1.0);
    assert!(empty_solver.is_empty() && empty_solver.is_valid());
    assert!(empty_solver.step_in_place(&mut empty_state));

    let one_h = TridiagonalHamiltonian1D::new(vec![2.5], vec![]);
    let mut one_prop = CrankNicolsonPropagator1D::new(&one_h, 0.05, 1.0);
    let mut one_state = QuantumState::new(DVector::from_vec(vec![Complex::new(0.6, -0.8)]));
    one_prop.evolve_in_place(&mut one_state);
    assert!((one_state.norm() - 1.0).abs() < 1e-14);

    let mut one_solver = CrankNicolson1D::new(&[1.5], 0.5, 0.05, 1.0, 1.0);
    let mut one_solver_state = QuantumState::new(DVector::from_vec(vec![Complex::new(0.6, -0.8)]));
    assert!(one_solver.step_in_place(&mut one_solver_state));
    assert!((one_solver_state.norm() - 1.0).abs() < 1e-14);

    // Reject non-positive / non-finite dx and dt before division
    assert!(!one_solver.configure(&[1.5], 0.0, 0.05, 1.0, 1.0));
    assert!(!one_solver.is_valid());
    assert!(!one_solver.step_in_place(&mut one_solver_state));
    assert!(!one_solver.configure(&[1.5], -0.2, 0.05, 1.0, 1.0));
    assert!(!one_solver.configure(&[1.5], f64::NAN, 0.05, 1.0, 1.0));
    assert!(!one_solver.configure(&[1.5], 0.5, 0.0, 1.0, 1.0));
    assert!(!one_solver.configure(&[1.5], 0.5, -0.01, 1.0, 1.0));
    assert!(!one_solver.configure(&[1.5], 0.5, f64::INFINITY, 1.0, 1.0));

    one_prop.configure_dt(&one_h, -0.01, 1.0);
    one_prop.evolve_in_place(&mut one_state);
    assert!(!one_state.vector[0].re.is_finite());
}
