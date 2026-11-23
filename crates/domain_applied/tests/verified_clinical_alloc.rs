//! Zero-allocation runtime verification and statistical tests for `evaluate_tiers` and `log_rank_test`.

use domain_applied::applied::clinical_trials::design::{
    BlockRandomizer, Group, Patient, StratifiedRandomizer,
};
use domain_applied::applied::clinical_trials::survival_analysis::{
    Observation, log_rank_test, log_rank_test_into,
};
use domain_applied::applied::clinical_trials::types::{ClinicalTrialError, SurvivalTime};
use domain_applied::applied::win_ratio::pair_comparison::{
    HigherIsBetter, LowerIsBetter, MatchedPairing, ThresholdComparator, TierOutcomeCounts,
    UnmatchedPairing, WinRatioAnalysis,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn sample_survival_arms() -> (Vec<Observation>, Vec<Observation>) {
    let obs = |t: f64, ev: bool| Observation {
        time: SurvivalTime::new(t).expect("finite non-negative time"),
        event_occurred: ev,
    };
    let treatment = vec![
        obs(6.0, true),
        obs(10.0, false),
        obs(13.0, true),
        obs(16.0, false),
        obs(22.0, true),
        obs(25.0, false),
    ];
    let control = vec![
        obs(1.0, true),
        obs(2.0, true),
        obs(3.0, true),
        obs(4.0, true),
        obs(5.0, true),
        obs(8.0, true),
    ];
    (treatment, control)
}

#[test]
fn test_tier_eval_and_log_rank_zero_allocations_under_verified_allocator() {
    let analysis = WinRatioAnalysis::new()
        .add_strategy(Box::new(HigherIsBetter))
        .add_strategy(Box::new(LowerIsBetter))
        .add_strategy(Box::new(ThresholdComparator::new(5.0)));

    let trt = vec![
        vec![400.0, 1.0, 75.0],
        vec![300.0, 0.0, 68.0],
        vec![250.0, 2.0, 62.0],
    ];
    let ctrl = vec![
        vec![300.0, 2.0, 55.0],
        vec![250.0, 2.0, 50.0],
        vec![180.0, 4.0, 40.0],
    ];
    let mut unmatched_counts = vec![TierOutcomeCounts::default(); 3];
    let mut matched_counts = vec![TierOutcomeCounts::default(); 3];

    let (surv_trt, surv_ctrl) = sample_survival_arms();
    let mut scratch1 = Vec::with_capacity(surv_trt.len());
    let mut scratch2 = Vec::with_capacity(surv_ctrl.len());

    // Keep all VerifiedEngine::run_verified assertions in one test to avoid MEMORY_LOCK races.
    let (lr, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..8 {
            analysis
                .evaluate_tiers_into(&trt, &ctrl, &UnmatchedPairing, &mut unmatched_counts)
                .expect("unmatched tier evaluation");
            analysis
                .evaluate_tiers_into(&trt, &ctrl, &MatchedPairing, &mut matched_counts)
                .expect("matched tier evaluation");
        }
        log_rank_test_into(&surv_trt, &surv_ctrl, &mut scratch1, &mut scratch2)
            .expect("log-rank test")
    });

    assert!(metrics.function_calls > 0);
    assert_eq!(
        unmatched_counts[0].wins + unmatched_counts[0].losses + unmatched_counts[0].ties,
        9
    );
    assert_eq!(unmatched_counts[0], TierOutcomeCounts::new(6, 1, 2));
    assert_eq!(unmatched_counts[1], TierOutcomeCounts::new(1, 0, 1));
    assert_eq!(unmatched_counts[2], TierOutcomeCounts::new(1, 0, 0));
    assert_eq!(unmatched_counts[0].decided(), 7);
    assert_eq!(matched_counts[0], TierOutcomeCounts::new(3, 0, 0));
    assert!(lr.statistic > 6.0);
    assert!(lr.p_value < 0.05);
}

#[test]
#[verified_engine::verified]
fn test_evaluate_tiers_fallible_errors_and_unequal_matched_pairs() {
    let empty_analysis: WinRatioAnalysis<f64> = WinRatioAnalysis::new();
    let g1 = vec![vec![10.0, 1.0], vec![20.0, 0.0]];
    let g2 = vec![vec![8.0, 2.0]];
    assert!(
        empty_analysis
            .evaluate_tiers(&g1, &g2, &UnmatchedPairing)
            .is_err()
    );

    let analysis = WinRatioAnalysis::new()
        .add_strategy(Box::new(HigherIsBetter))
        .add_strategy(Box::new(LowerIsBetter));

    let err = analysis
        .evaluate_tiers(&g1, &g2, &MatchedPairing)
        .expect_err("unequal matched pairs must return Err without panicking");
    assert!(matches!(err, ClinicalTrialError::InvalidData(_)));

    let empty_group: Vec<Vec<f64>> = Vec::new();
    let empty_err = analysis
        .evaluate_tiers(&empty_group, &g2, &UnmatchedPairing)
        .expect_err("empty group must fail");
    assert!(matches!(
        empty_err,
        ClinicalTrialError::InsufficientSampleSize { .. }
    ));

    let mut short_buf = [TierOutcomeCounts::default(); 1];
    assert!(
        analysis
            .evaluate_tiers_into(&g1, &g2, &UnmatchedPairing, &mut short_buf)
            .is_err()
    );
}

#[test]
#[verified_engine::verified]
fn test_log_rank_identical_arms_and_censored_only_error() {
    let (trt, ctrl) = sample_survival_arms();
    let diff_res = log_rank_test(&trt, &ctrl).expect("distinct arms log-rank");
    assert!(diff_res.observed_group1 < diff_res.expected_group1);
    assert!(diff_res.p_value < 0.05);

    let same_res = log_rank_test(&ctrl, &ctrl).expect("identical arms log-rank");
    assert!(same_res.statistic.abs() < 1e-12);
    assert!((same_res.p_value - 1.0).abs() < 1e-12);

    let censored_only = vec![Observation {
        time: SurvivalTime::new(5.0).unwrap(),
        event_occurred: false,
    }];
    assert!(log_rank_test(&censored_only, &censored_only).is_err());
    assert!(log_rank_test(&[], &censored_only).is_err());
}

#[test]
#[verified_engine::verified]
fn test_stratified_randomizer_seeded_determinism() {
    let patients: Vec<Patient> = (0..12)
        .map(|i| Patient {
            id: format!("SUBJ-{:02}", i + 1),
            stratum: if i % 2 == 0 {
                "Site-A".to_string()
            } else {
                "Site-B".to_string()
            },
        })
        .collect();
    let strat = StratifiedRandomizer::new(BlockRandomizer::new(2).unwrap());
    let mut rng1 = oxidize_core::rng::OxidizeRng::new(2026);
    let mut rng2 = oxidize_core::rng::OxidizeRng::new(2026);
    let map1 = strat.assign_stratified(&mut rng1, &patients).unwrap();
    let map2 = strat.assign_stratified(&mut rng2, &patients).unwrap();
    assert_eq!(map1, map2);
    let trt_a = patients
        .iter()
        .filter(|p| p.stratum == "Site-A" && map1.get(&p.id) == Some(&Group::Treatment))
        .count();
    assert_eq!(trt_a, 3);
}
