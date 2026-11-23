//! Integration tests for test clinical trials deterministic.
use domain_applied::applied::clinical_trials::design::{
    AllocationStrategy, BlockRandomizer, Group, SimpleRandomizer,
};

#[test]
#[verified_engine::verified]
fn test_deterministic_simple_randomization() {
    let mut rng1 = oxidize_core::rng::OxidizeRng::default();
    let mut rng2 = oxidize_core::rng::OxidizeRng::default();
    let strategy = SimpleRandomizer;

    let assignments1 = strategy.assign(&mut rng1, 100).unwrap();
    let assignments2 = strategy.assign(&mut rng2, 100).unwrap();

    assert_eq!(assignments1, assignments2);
}

#[test]
#[verified_engine::verified]
fn test_deterministic_block_randomization() {
    let mut rng1 = oxidize_core::rng::OxidizeRng::new(12345);
    let mut rng2 = oxidize_core::rng::OxidizeRng::new(12345);
    let strategy = BlockRandomizer::new(4).unwrap();

    let assignments1 = strategy.assign(&mut rng1, 20).unwrap();
    let assignments2 = strategy.assign(&mut rng2, 20).unwrap();

    assert_eq!(assignments1, assignments2);

    // Check balance
    let treatment_count = assignments1
        .iter()
        .filter(|&&g| g == Group::Treatment)
        .count();
    let control_count = assignments1
        .iter()
        .filter(|&&g| g == Group::Control)
        .count();
    assert_eq!(treatment_count, 10);
    assert_eq!(control_count, 10);
}
