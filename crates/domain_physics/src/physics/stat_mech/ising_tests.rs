use super::*;
use crate::physics::stat_mech::KB;
use domain_applied::applied::game_theory::evolutionary::ReplicatorDynamics;
use nalgebra::{DMatrix, DVector};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;
static VERIFIED_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn seeded_droplet(r: i32, seed: u64) -> SpinLattice {
    let mut lat = SpinLattice::new(30, 30, Some(seed));
    for y in 0..30 {
        for x in 0..30 {
            let (dx, dy) = (x as i32 - 15, y as i32 - 15);
            lat.set(x, y, if dx * dx + dy * dy <= r * r { 1 } else { -1 });
        }
    }
    lat
}

#[test]
#[verified_engine::verified]
fn test_ising_disordered() {
    // High temperature limit. T >> J/k_B.
    // Magnetization should be near 0 on average.
    let j_val = 1.0;
    let h_val = 0.0;
    let t_high = 100.0 * j_val / KB;

    let width = 20;
    let height = 20;
    let mut lattice = SpinLattice::new(width, height, None);

    let steps = 10_000;
    for _ in 0..steps {
        lattice.metropolis_step(t_high, j_val, h_val);
    }

    let m = lattice.magnetization();
    let max_m = (width * height) as f64;
    let avg_m_per_spin = m as f64 / max_m;

    assert!(
        avg_m_per_spin.abs() < 0.3,
        "High T Ising should be disordered (M ~ 0). Got {}",
        avg_m_per_spin
    );
}

#[test]
#[verified_engine::verified]
fn test_ising_pinned_spins_stay_fixed_with_zero_allocations() {
    let j_val = 1.0;
    let h_val = -2.0;
    let t_high = 50.0 * j_val / KB;
    let mut lattice = SpinLattice::new(10, 10, Some(7));
    for x in 0..10 {
        lattice.set(x, 0, 1);
        lattice.set_pinned(x, 0, true);
        lattice.set(x, 9, -1);
        lattice.set_pinned(x, 9, true);
    }

    let _guard = VERIFIED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (_, metrics) = VerifiedEngine::run_verified(|| {
        lattice.evolve(20_000, t_high, j_val, h_val);
        for _ in 0..500 {
            lattice.metropolis_step(t_high, j_val, h_val);
        }
    });
    drop(_guard);
    assert!(metrics.function_calls > 0);
    for x in 0..10 {
        assert_eq!(lattice.get(x, 0), 1);
        assert!(lattice.is_pinned(x, 0));
        assert_eq!(lattice.get(x, 9), -1);
        assert!(lattice.is_pinned(x, 9));
    }

    lattice.clear_pinned();
    assert!(lattice.pinned().iter().all(|&p| !p));
}

#[test]
#[verified_engine::verified]
fn test_fermi_game_zero_allocations_under_verified_allocator() {
    let mut lattice = SpinLattice::new(100, 100, Some(7));
    let payoff = MatrixPayoff::new(DMatrix::from_row_slice(2, 2, &[1.0, -0.1, 1.35, 0.0])).unwrap();

    let _guard = VERIFIED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        lattice.evolve_fermi_game(10_000, 0.0, &payoff);
    });
    drop(_guard);
    assert!(metrics.function_calls >= 2);
    assert!(lattice.magnetization().abs() <= 10_000);
}

#[test]
#[verified_engine::verified]
fn test_prisoners_dilemma_droplet_survival_vs_replicator_collapse() {
    let pd_mat = DMatrix::from_row_slice(2, 2, &[1.0, -0.1, 1.25, 0.0]);
    let payoff = MatrixPayoff::new(pd_mat.clone()).unwrap();

    let mut isolated_lat = seeded_droplet(0, 1234);
    isolated_lat.evolve_fermi_game(30 * 30 * 20, 0.08, &payoff);
    assert_eq!(isolated_lat.magnetization(), -900);

    let mut droplet_lat = seeded_droplet(5, 1234);
    let init_coop = (droplet_lat.magnetization() as f64 / 900.0 + 1.0) * 0.5;
    droplet_lat.evolve_fermi_game(30 * 30 * 50, 0.08, &payoff);
    let final_coop = (droplet_lat.magnetization() as f64 / 900.0 + 1.0) * 0.5;
    assert!(
        final_coop > 0.7,
        "Droplet must expand ({init_coop:.3} -> {final_coop:.3})"
    );

    let ode = ReplicatorDynamics::new(pd_mat).unwrap();
    let traj = ode.simulate(
        DVector::from_vec(vec![init_coop, 1.0 - init_coop]),
        40.0,
        0.1,
    );
    assert!(traj.last().unwrap().1[0] < 1e-2);
}

#[test]
#[verified_engine::verified]
fn test_fermi_game_temperature_phase_transition() {
    let payoff = MatrixPayoff::new(DMatrix::from_row_slice(2, 2, &[1.0, -0.1, 1.25, 0.0])).unwrap();
    let mut cold = seeded_droplet(5, 99);
    cold.evolve_fermi_game(30 * 30 * 60, 0.08, &payoff);
    let cold_coop = (cold.magnetization() as f64 / 900.0 + 1.0) * 0.5;

    let mut hot = seeded_droplet(5, 99);
    hot.evolve_fermi_game(30 * 30 * 60, 0.9, &payoff);
    let hot_coop = (hot.magnetization() as f64 / 900.0 + 1.0) * 0.5;
    assert!(
        cold_coop > hot_coop + 0.3,
        "Expected phase transition: {cold_coop:.3} vs {hot_coop:.3}"
    );
}
