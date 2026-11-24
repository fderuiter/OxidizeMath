#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use domain_applied::applied::game_theory::evolutionary::MatrixPayoff;
use nalgebra::DMatrix;
use verified_engine::engine::VerifiedEngine;

fn sample_wheel_csr() -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
    let adj: [&[usize]; 5] = [
        &[1, 2, 3, 4],
        &[0, 2, 4],
        &[0, 1, 3],
        &[0, 2, 4],
        &[0, 1, 3],
    ];
    let mut row_ptr = Vec::with_capacity(6);
    let mut col_idx = Vec::new();
    row_ptr.push(0);
    for nbrs in adj {
        col_idx.extend_from_slice(nbrs);
        row_ptr.push(col_idx.len());
    }
    let weights = vec![1.0; col_idx.len()];
    (5, row_ptr, col_idx, weights)
}

#[test]
fn test_graph_spin_system_csr_validation_and_ferromagnetic_alignment() {
    let (n, row_ptr, col_idx, weights) = sample_wheel_csr();
    let mut sys = GraphSpinSystem::try_from_csr(n, row_ptr, col_idx, weights, Some(123)).unwrap();
    assert_eq!(sys.num_edges(), 8);
    sys.evolve(10_000, 0.25 / KB, 1.0, 0.25);
    assert!(sys.normalized_magnetization() > 0.6);
    assert!(sys.total_energy(1.0, 0.25).is_finite());

    assert!(GraphSpinSystem::try_from_csr(2, vec![0, 1], vec![0], vec![1.0], Some(1)).is_err());
    assert!(GraphSpinSystem::try_from_csr(2, vec![0, 1, 1], vec![5], vec![1.0], Some(1)).is_err());
}

#[test]
fn test_graph_spin_system_zero_alloc_1000_steps_under_verified_engine() {
    let (n, row_ptr, col_idx, weights) = sample_wheel_csr();
    let mut sys = GraphSpinSystem::try_from_csr(n, row_ptr, col_idx, weights, Some(42)).unwrap();
    let payoff = MatrixPayoff::new(DMatrix::from_row_slice(2, 2, &[1.0, -0.15, 1.2, 0.0])).unwrap();

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((mag, coop), metrics) = VerifiedEngine::run_verified(|| {
        sys.evolve(1_000, 1.2 / KB, 1.0, 0.0);
        sys.evolve_fermi_game(1_000, 0.1, &payoff);
        (sys.magnetization(), sys.cooperator_fraction())
    });
    assert!(mag.abs() <= 5);
    assert!((0.0..=1.0).contains(&coop));
    assert!(metrics.function_calls > 0);
}
