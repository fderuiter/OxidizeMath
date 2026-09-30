#![allow(missing_docs)]
use domain_applied::applied::win_ratio::pair_comparison::{
    ComparisonResult, HigherIsBetter, LowerIsBetter, ThresholdComparator, UnmatchedPairing,
    WinRatioAnalysis,
};

#[test]
#[verified_engine::verified]
fn test_mixed_strategies() {
    // Scenario:
    // Outcome 1: Days to Death (Higher is Better - longer survival)
    // Outcome 2: Hospitalization Count (Lower is Better - fewer events)
    // Outcome 3: Quality of Life (Higher is Better)

    let analysis = WinRatioAnalysis::new()
        .add_strategy(Box::new(HigherIsBetter)) // Death
        .add_strategy(Box::new(LowerIsBetter)) // Hosp
        .add_strategy(Box::new(HigherIsBetter)); // QoL

    // Subject A: Lived 100 days, 2 hosp, 50 QoL
    let subj_a = vec![100.0, 2.0, 50.0];

    // Subject B: Lived 100 days, 5 hosp, 60 QoL
    let subj_b = vec![100.0, 5.0, 60.0];

    // Comparison:
    // 1. Death: 100 vs 100 -> Tie
    // 2. Hosp: 2 vs 5 -> Lower is Better -> 2 is better -> Win for A

    assert_eq!(
        analysis.compare_subjects(&subj_a, &subj_b),
        ComparisonResult::Win
    );

    // Subject C: Lived 50 days, 0 hosp, 90 QoL
    let subj_c = vec![50.0, 0.0, 90.0];

    // Comparison A vs C:
    // 1. Death: 100 vs 50 -> Higher is Better -> Win for A
    assert_eq!(
        analysis.compare_subjects(&subj_a, &subj_c),
        ComparisonResult::Win
    );
}

#[test]
#[verified_engine::verified]
fn test_threshold_strategy() {
    // Scenario:
    // Outcome 1: Biomarker Level (Higher is Better, but difference < 5.0 is a Tie)

    let analysis = WinRatioAnalysis::new().add_strategy(Box::new(ThresholdComparator::new(5.0)));

    let subj_a = vec![100.0];
    let subj_b = vec![104.0]; // Diff is 4.0 -> Tie
    let subj_c = vec![106.0]; // Diff is 6.0 -> Win for C (106 > 100)

    assert_eq!(
        analysis.compare_subjects(&subj_a, &subj_b),
        ComparisonResult::Tie
    );
    assert_eq!(
        analysis.compare_subjects(&subj_b, &subj_a),
        ComparisonResult::Tie
    );

    assert_eq!(
        analysis.compare_subjects(&subj_c, &subj_a),
        ComparisonResult::Win
    );
    assert_eq!(
        analysis.compare_subjects(&subj_a, &subj_c),
        ComparisonResult::Loss
    );
}

#[test]
#[verified_engine::verified]
fn test_backward_compatibility() {
    let group1 = vec![vec![10.0]];
    let group2 = vec![vec![5.0]];

    let analysis = WinRatioAnalysis::new().add_strategy(Box::new(HigherIsBetter));
    let (wins, losses) = analysis.evaluate_pairs(&group1, &group2, &UnmatchedPairing);
    assert_eq!(wins, 1);
    assert_eq!(losses, 0);
}
