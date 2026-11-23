//! Integration tests for test game theory deterministic.
use domain_applied::applied::game_theory::mechanism_design::simulate_optimal_revenue_with_rng;
use statrs::distribution::Uniform;

#[test]
#[verified_engine::verified]
fn test_revenue_simulation_deterministic() {
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let n_bidders = 2;
    let n_simulations = 100;

    // Seed 1
    let mut rng1 = oxidize_core::rng::OxidizeRng::new(12345);
    let rev1 = simulate_optimal_revenue_with_rng(&dist, n_bidders, n_simulations, &mut rng1);

    // Seed 2
    let mut rng2 = oxidize_core::rng::OxidizeRng::new(12345);
    let rev2 = simulate_optimal_revenue_with_rng(&dist, n_bidders, n_simulations, &mut rng2);

    // Seed 3
    let mut rng3 = oxidize_core::rng::OxidizeRng::new(67890);
    let rev3 = simulate_optimal_revenue_with_rng(&dist, n_bidders, n_simulations, &mut rng3);

    assert_eq!(
        rev1, rev2,
        "Simulations with same seed must return identical revenue"
    );
    assert_ne!(rev1, rev3, "Simulations with different seeds should vary");
}
