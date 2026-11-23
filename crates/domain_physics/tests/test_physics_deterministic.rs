//! Cross-platform deterministic tests for `domain_physics` Ising, Sandpile, and diffusion dynamics.

use domain_physics::physics::stat_mech::KB;
use domain_physics::physics::stat_mech::dynamics::{
    estimate_diffusion_coefficient_with_rng, random_walk_1d_with_rng,
};
use domain_physics::physics::stat_mech::ising::SpinLattice;
use domain_physics::physics::stat_mech::sandpile::SandpileGrid;
use oxidize_core::rng::{OxidizeRng, TrajectoryDigest};

const PINNED_ISING_DIGEST: &str =
    "e63dbae7fa7038e58f0ca053966e5aa65c8fd152163d885578142296c7dd6c11";
const PINNED_SANDPILE_DIGEST: &str =
    "328491378d9378e7b744ddd9373d3905eb6484476b43811320d3897f2131b2d1";
const PINNED_DIFFUSION_DIGEST: &str =
    "c71b89d5250d9fad829471d188d4d213a29fc1a5837e61a67d40273ab3834c60";

#[test]
#[verified_engine::verified]
fn test_ising_metropolis_pinned_digest() {
    let temp = 1.8 / KB;
    let run = |seed: u64| -> (Vec<i8>, String) {
        let mut lattice = SpinLattice::new(8, 8, Some(seed));
        for _ in 0..256 {
            lattice.metropolis_step(temp, 1.0, 0.05);
        }
        let mut digest = TrajectoryDigest::new();
        for &s in lattice.spins() {
            digest.update_i64(i64::from(s));
        }
        digest.update_i64(lattice.magnetization());
        digest.update_f64(lattice.hamiltonian(1.0, 0.05));
        (lattice.spins().to_vec(), digest.finish_hex())
    };

    let (s1, h1) = run(42);
    let (s2, h2) = run(42);
    let (s3, h3) = run(99);
    assert_eq!(s1, s2);
    assert_eq!(h1, h2);
    assert_ne!(s1, s3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_ISING_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_sandpile_random_grain_avalanches_pinned_digest() {
    let run = |seed: u64| -> (Vec<u32>, String) {
        let mut grid = SandpileGrid::new(7, 7, Some(seed)).unwrap();
        grid.fill(3);
        let mut digest = TrajectoryDigest::new();
        for _ in 0..20 {
            let (x, y) = grid.drop_random_grain();
            let stats = grid.stabilize_wavefront();
            digest.update_usize(x);
            digest.update_usize(y);
            digest.update_u64(stats.topples);
            digest.update_usize(stats.area);
        }
        for &h in grid.heights() {
            digest.update_u32(h);
        }
        (grid.heights().to_vec(), digest.finish_hex())
    };

    let (g1, h1) = run(42);
    let (g2, h2) = run(42);
    let (g3, h3) = run(99);
    assert_eq!(g1, g2);
    assert_eq!(h1, h2);
    assert_ne!(g1, g3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_SANDPILE_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_random_walk_diffusion_pinned_digest() {
    let run = |seed: u64| -> ([f64; 2], String) {
        let mut rng = OxidizeRng::new(seed);
        let walk = random_walk_1d_with_rng(128, &mut rng);
        let diff = estimate_diffusion_coefficient_with_rng(64, 50, &mut rng);
        let mut digest = TrajectoryDigest::new();
        digest.update_f64(walk);
        digest.update_f64(diff);
        ([walk, diff], digest.finish_hex())
    };

    let (v1, h1) = run(42);
    let (v2, h2) = run(42);
    let (v3, h3) = run(99);
    assert_eq!(v1, v2);
    assert_eq!(h1, h2);
    assert_ne!(v1, v3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_DIFFUSION_DIGEST);
}
