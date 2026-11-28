use super::*;
use crate::physics::stat_mech::KB;
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::analysis::evolutionary::ReplicatorDynamics;
use verified_engine::engine::VerifiedEngine;

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
    let payoff = MatrixPayoff::new(DMatrix::from_row_slice(2, 2, &[1.5, 0.5, 0.5, 0.0])).unwrap();
    let mut lattice = SpinLattice::new(10, 10, Some(7));
    for x in 0..10 {
        lattice.set(x, 0, 1);
        lattice.set_pinned(x, 0, true);
        lattice.set(x, 9, -1);
        lattice.set_pinned(x, 9, true);
    }

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (_, metrics) = VerifiedEngine::run_verified(|| {
        lattice.evolve(20_000, t_high, j_val, h_val);
        lattice.evolve_fermi_game_with_mutation(5_000, 0.1, &payoff, 0.05);
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

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((m_unif, m_stag), metrics) = VerifiedEngine::run_verified(|| {
        lattice.evolve_fermi_game(5_000, 0.0, &payoff);
        lattice.evolve_fermi_game_with_mutation(5_000, 0.08, &payoff, 0.02);
        (lattice.magnetization(), lattice.staggered_magnetization())
    });
    drop(_guard);
    assert!(metrics.function_calls >= 4);
    assert!(m_unif.abs() <= 10_000);
    assert!(m_stag.abs() <= 10_000);
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
#[test]
#[verified_engine::verified]
fn test_fermi_mutation_unfreezes_all_defector_lattice() {
    let harmony = MatrixPayoff::new(DMatrix::from_row_slice(2, 2, &[1.5, 0.5, 0.5, 0.0])).unwrap();
    let mut absorbed = SpinLattice::new(24, 24, Some(77));
    for y in 0..24 {
        for x in 0..24 {
            absorbed.set(x, y, -1);
        }
    }
    assert_eq!(absorbed.magnetization(), -576);

    absorbed.evolve_fermi_game(5_000, 0.1, &harmony);
    assert_eq!(absorbed.magnetization(), -576);

    absorbed.evolve_fermi_game_with_mutation(24 * 24 * 25, 0.1, &harmony, 0.02);
    let coop_frac = (absorbed.magnetization() as f64 / 576.0 + 1.0) * 0.5;
    assert!(
        coop_frac > 0.85,
        "Mutation must unfreeze absorbed -1 lattice in Harmony game, got {coop_frac:.3}"
    );
}

#[test]
#[verified_engine::verified]
fn test_antiferromagnetic_staggered_magnetization() {
    let mut lattice = SpinLattice::new(16, 16, Some(314));
    for y in 0..16 {
        for x in 0..16 {
            lattice.set(x, y, if (x + y) % 2 == 0 { 1 } else { -1 });
        }
    }
    assert_eq!(lattice.magnetization(), 0);
    assert_eq!(lattice.staggered_magnetization(), 256);

    let mut random_af = SpinLattice::new(12, 12, Some(2026));
    random_af.evolve(250_000, 0.8 / KB, -1.0, 0.0);
    let stag_ratio = (random_af.staggered_magnetization() as f64).abs() / 144.0;
    assert!(
        stag_ratio > 0.7,
        "Antiferromagnetic J = -1 lattice must develop checkerboard order, got {stag_ratio:.3}"
    );
}
