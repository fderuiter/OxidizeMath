//! Cross-platform deterministic auction revenue simulation tests with pinned SHA-256 digest.

use domain_applied::applied::game_theory::mechanism_design::simulate_optimal_revenue_with_rng;
use oxidize_core::rng::{OxidizeRng, TrajectoryDigest};
use statrs::distribution::Uniform;

const PINNED_REVENUE_DIGEST: &str =
    "58d7c48f86fc8188e73730ea4191c72594c2c2d8e08ab0bb7ac22fe6b823b890";

#[test]
#[verified_engine::verified]
fn test_revenue_simulation_deterministic() {
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let n_bidders = 2;
    let n_simulations = 100;

    let run_revenues = |seed: u64| -> (Vec<f64>, String) {
        let mut rng = OxidizeRng::new(seed);
        let mut digest = TrajectoryDigest::new();
        let mut revs = Vec::with_capacity(8);
        for extra_bidders in 0..8 {
            let rev = simulate_optimal_revenue_with_rng(
                &dist,
                n_bidders + (extra_bidders % 3),
                n_simulations,
                &mut rng,
            );
            digest.update_f64(rev);
            revs.push(rev);
        }
        (revs, digest.finish_hex())
    };

    let (rev1, hex1) = run_revenues(12345);
    let (rev2, hex2) = run_revenues(12345);
    let (rev3, hex3) = run_revenues(67890);

    assert_eq!(
        rev1, rev2,
        "Simulations with same seed must return identical revenue"
    );
    assert_eq!(hex1, hex2);
    assert_ne!(rev1, rev3, "Simulations with different seeds should vary");
    assert_ne!(hex1, hex3);
    assert_eq!(hex1, PINNED_REVENUE_DIGEST);
}
