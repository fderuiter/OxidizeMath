//! Cross-platform deterministic clinical trial randomization and bivariate Weibull suites with pinned digests.

use domain_applied::applied::clinical_trials::design::{
    AllocationStrategy, BlockRandomizer, Group, SimpleRandomizer,
};
use domain_applied::applied::win_ratio::simulation::BivariateWeibullModel;
use oxidize_core::rng::{OxidizeRng, TrajectoryDigest};

const PINNED_SIMPLE_DIGEST: &str =
    "e7d7fb1437f42bc237efdb9c379d5649e647f384722077c3c1e9d62c7f5fdd61";
const PINNED_BLOCK_DIGEST: &str =
    "ddb61185a848268483b2145a19b9d4e72b93459b1c6f05866559a7d4566bcbfc";
const PINNED_WEIBULL_DIGEST: &str =
    "a95f6d641594699557714ead734fb5aaef66002f9b4546e4652b0304c0e60c74";

fn digest_groups(groups: &[Group]) -> String {
    let mut digest = TrajectoryDigest::new();
    for &g in groups {
        let tag = match g {
            Group::Control => 0_u8,
            Group::Treatment => 1_u8,
        };
        digest.update_bytes(&[tag]);
    }
    digest.finish_hex()
}

#[test]
#[verified_engine::verified]
fn test_deterministic_simple_randomization() {
    let strategy = SimpleRandomizer;
    let mut rng1 = OxidizeRng::new(42);
    let mut rng2 = OxidizeRng::new(42);
    let mut rng_other = OxidizeRng::new(43);

    let assignments1 = strategy.assign(&mut rng1, 100).unwrap();
    let assignments2 = strategy.assign(&mut rng2, 100).unwrap();
    let assignments_other = strategy.assign(&mut rng_other, 100).unwrap();

    assert_eq!(assignments1, assignments2);
    assert_ne!(assignments1, assignments_other);
    assert_eq!(digest_groups(&assignments1), PINNED_SIMPLE_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_deterministic_block_randomization() {
    let strategy = BlockRandomizer::new(4).unwrap();
    let mut rng1 = OxidizeRng::new(12345);
    let mut rng2 = OxidizeRng::new(12345);
    let mut rng_other = OxidizeRng::new(54321);

    let assignments1 = strategy.assign(&mut rng1, 20).unwrap();
    let assignments2 = strategy.assign(&mut rng2, 20).unwrap();
    let assignments_other = strategy.assign(&mut rng_other, 20).unwrap();

    assert_eq!(assignments1, assignments2);
    assert_ne!(assignments1, assignments_other);

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
    assert_eq!(digest_groups(&assignments1), PINNED_BLOCK_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_deterministic_bivariate_weibull_sampling() {
    let model = BivariateWeibullModel::new(0.18, 0.32, 0.72);
    let sample_cohort = |seed: u64| -> (Vec<[f64; 2]>, String) {
        let mut rng = OxidizeRng::new(seed);
        let mut digest = TrajectoryDigest::new();
        let mut cohort = Vec::with_capacity(24);
        for _ in 0..24 {
            let (fatal_t, nonfatal_t) = model.try_sample_event_times(&mut rng).unwrap();
            assert!(fatal_t > 0.0 && fatal_t.is_finite());
            assert!(nonfatal_t > 0.0 && nonfatal_t.is_finite());
            digest.update_f64(fatal_t);
            digest.update_f64(nonfatal_t);
            cohort.push([fatal_t, nonfatal_t]);
        }
        (cohort, digest.finish_hex())
    };

    let (cohort_a, hex_a) = sample_cohort(0xC11A_2026);
    let (cohort_b, hex_b) = sample_cohort(0xC11A_2026);
    let (cohort_c, hex_c) = sample_cohort(0xC11A_2027);
    assert_eq!(cohort_a, cohort_b);
    assert_eq!(hex_a, hex_b);
    assert_ne!(cohort_a, cohort_c);
    assert_ne!(hex_a, hex_c);
    assert_eq!(hex_a, PINNED_WEIBULL_DIGEST);
}
