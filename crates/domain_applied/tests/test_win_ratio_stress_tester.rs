//! Integration and statistical calibration tests for composite win-ratio U-statistics,
//! right-censored comparators, deterministic Weibull sampling, and zero-allocation sweeps.

use domain_applied::applied::clinical_trials::sample_size::{
    calculate_power_means, calculate_power_proportions, calculate_sample_size_means,
    calculate_sample_size_proportions,
};
use domain_applied::applied::win_ratio::censoring::{CensoredOutcome, RightCensoredComparator};
use domain_applied::applied::win_ratio::pair_comparison::{
    ComparisonResult, HigherIsBetter, OutcomeComparator, UnmatchedPairing, WinRatioAnalysis,
};
use domain_applied::applied::win_ratio::power_sweep::{
    PowerSweepConfig, PowerSweepWorkspace, run_power_sweep_into, run_weibull_power_sweep,
    simulate_weibull_trial_into,
};
use domain_applied::applied::win_ratio::simulation::{
    BivariateWeibullModel, SimulationParams, win_ratio_parameter,
};
use domain_applied::applied::win_ratio::u_statistic::UnmatchedScratch;
use oxidize_core::rng::OxidizeRng;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn unmatched_u_statistic_calibrates_type_i_error_across_2000_null_trials() {
    let params = SimulationParams::new(0.12, 0.30, 0.70, 1.0);
    let mut rng = OxidizeRng::new(0x5EED_2026_u64);
    let mut workspace = PowerSweepWorkspace::with_capacity(50, 1);

    let trials = 2_000usize;
    let mut rejections = 0usize;
    for _ in 0..trials {
        let (summary_opt, _tiers) =
            simulate_weibull_trial_into(&params, 50, 5.0, 0.05, 0.05, &mut rng, &mut workspace)
                .expect("null simulation trial must succeed");
        let summary = summary_opt.expect("50-per-arm trial must produce non-degenerate summary");
        if summary.p_value < 0.05 {
            rejections += 1;
        }
    }

    let empirical_alpha = (rejections as f64) / (trials as f64);
    assert!(
        (0.035..=0.065).contains(&empirical_alpha),
        "Empirical Type I error {empirical_alpha:.4} ({rejections}/{trials}) fell outside [0.035, 0.065]"
    );
}

#[test]
fn right_censored_comparator_handles_observed_and_indeterminate_pairs() {
    let cmp = RightCensoredComparator::new();
    let early_event = CensoredOutcome::new(2.0, true).unwrap();
    let late_event = CensoredOutcome::new(5.0, true).unwrap();
    let early_censored = CensoredOutcome::new(1.5, false).unwrap();
    let late_censored = CensoredOutcome::new(6.0, false).unwrap();

    // Control event at t=2 before treatment follow-up t=6 -> Treatment Win
    assert_eq!(
        cmp.compare(&late_censored, &early_event),
        ComparisonResult::Win
    );
    // Treatment event at t=2 before control event at t=5 -> Treatment Loss
    assert_eq!(
        cmp.compare(&early_event, &late_event),
        ComparisonResult::Loss
    );
    // Control censored at t=1.5 before treatment event at t=2.0 -> Indeterminate Tie!
    assert_eq!(
        cmp.compare(&early_event, &early_censored),
        ComparisonResult::Tie
    );
    // Both censored -> Indeterminate Tie
    assert_eq!(
        cmp.compare(&early_censored, &late_censored),
        ComparisonResult::Tie
    );

    // Horizon c=4.0 censors both t=5 event and t=6 censored -> Tie
    let horizon_cmp = RightCensoredComparator::with_horizon(4.0).unwrap();
    assert_eq!(
        horizon_cmp.compare(&late_censored, &late_event),
        ComparisonResult::Tie
    );
}

#[test]
fn hierarchical_tier_fallthrough_passes_censored_tier1_pairs_to_tier2() {
    let analysis = WinRatioAnalysis::new()
        .add_strategy(Box::new(RightCensoredComparator::new()))
        .add_strategy(Box::new(RightCensoredComparator::new()));

    // Subject Trt 0: censored on fatal at t=3.0, no nonfatal event through t=3.0
    // Subject Trt 1: survived past t=8.0, nonfatal event at t=6.0
    let trt = vec![
        vec![
            CensoredOutcome::new(3.0, false).unwrap(),
            CensoredOutcome::new(3.0, false).unwrap(),
        ],
        vec![
            CensoredOutcome::new(8.0, false).unwrap(),
            CensoredOutcome::new(6.0, true).unwrap(),
        ],
    ];
    // Subject Ctrl 0: fatal event at t=4.0 (indeterminate vs Trt 0 who censored at 3.0, Win for Trt 1!),
    //                 nonfatal event at t=1.5 (Win for both Trt 0 and Trt 1 at Tier 2 if Tier 1 tied!)
    // Subject Ctrl 1: censored at t=7.0, nonfatal event at t=2.0
    let ctrl = vec![
        vec![
            CensoredOutcome::new(4.0, true).unwrap(),
            CensoredOutcome::new(1.5, true).unwrap(),
        ],
        vec![
            CensoredOutcome::new(7.0, false).unwrap(),
            CensoredOutcome::new(2.0, true).unwrap(),
        ],
    ];

    let (summary, tiers) = analysis
        .evaluate_unmatched(&trt, &ctrl, 0.05)
        .expect("evaluation succeeds");
    assert_eq!(summary.total_pairs, 4);
    // (Trt0, Ctrl0): Tier1 Tie (3.0 cens vs 4.0 event), Tier2 Win (3.0 > 1.5 event)
    // (Trt0, Ctrl1): Tier1 Tie (both cens), Tier2 Win (3.0 > 2.0 event)
    // (Trt1, Ctrl0): Tier1 Win (8.0 > 4.0 event)
    // (Trt1, Ctrl1): Tier1 Tie (both cens), Tier2 Win (6.0 > 2.0 event)
    assert_eq!(tiers[0].wins, 1);
    assert_eq!(tiers[0].losses, 0);
    assert_eq!(tiers[0].ties_remaining, 3);
    assert_eq!(tiers[1].wins, 3);
    assert_eq!(tiers[1].losses, 0);
    assert_eq!(tiers[1].ties_remaining, 0);
    assert_eq!(summary.wins, 4);
    assert_eq!(summary.losses, 0);
}

#[test]
fn weibull_inverse_cdf_matches_joint_survival_and_uncensored_win_ratio() {
    let model = BivariateWeibullModel::new(0.25, 0.45, 0.65);
    let mut rng = OxidizeRng::new(123_456_789);
    let n = 8_000usize;
    let t_probe = 1.8;
    let x_probe = 1.1;
    let mut joint_survivors = 0usize;
    for _ in 0..n {
        let (t, x) = model.try_sample_event_times(&mut rng).unwrap();
        if t > t_probe && x > x_probe {
            joint_survivors += 1;
        }
    }
    let empirical_s = (joint_survivors as f64) / (n as f64);
    let exact_s = model.joint_survival(t_probe, x_probe);
    assert!(
        (empirical_s - exact_s).abs() < 0.02,
        "Empirical joint survival {empirical_s:.4} deviated from exact {exact_s:.4}"
    );

    // Uncensored win ratio with wide horizon should closely match PR_W = theta^(-alpha)
    let params = SimulationParams::new(0.25, 0.45, 0.75, 0.60);
    let cfg = PowerSweepConfig {
        n_min: 120,
        n_max: 120,
        n_steps: 1,
        replicates: 80,
        censor_horizon: 500.0,
        dropout_rate: 0.0,
        sig_alpha: 0.05,
    };
    let report = run_weibull_power_sweep(&params, &cfg, 99_001).unwrap();
    let expected_wr = win_ratio_parameter(&params);
    let observed_wr = report.points[0].mean_win_ratio;
    assert!(
        (observed_wr - expected_wr).abs() / expected_wr < 0.08,
        "Uncensored mean WR {observed_wr:.3} deviated from theoretical {expected_wr:.3}"
    );
}

#[test]
fn evaluate_unmatched_into_and_power_sweep_into_perform_zero_heap_allocations() {
    let analysis = WinRatioAnalysis::new()
        .add_strategy(Box::new(HigherIsBetter))
        .add_strategy(Box::new(HigherIsBetter));
    let g1 = vec![[10.0, 4.0], [8.0, 5.0], [6.0, 9.0], [12.0, 3.0]];
    let g2 = vec![[5.0, 3.0], [8.0, 2.0], [7.0, 6.0], [4.0, 1.0]];
    let g1_vecs: Vec<Vec<f64>> = g1.iter().map(|r| r.to_vec()).collect();
    let g2_vecs: Vec<Vec<f64>> = g2.iter().map(|r| r.to_vec()).collect();
    let (expected_w, expected_l) = analysis.evaluate_pairs(&g1_vecs, &g2_vecs, &UnmatchedPairing);

    let mut scratch = UnmatchedScratch::with_capacity(4, 4, 2);
    let (summary_opt, _) = VerifiedEngine::run_verified(|| {
        analysis.evaluate_unmatched_into(&g1, &g2, 0.05, &mut scratch)
    });
    let summary = summary_opt.expect("valid summary");
    assert_eq!(summary.wins, expected_w as usize);
    assert_eq!(summary.losses, expected_l as usize);

    let params = SimulationParams::new(0.15, 0.35, 0.70, 0.65);
    let cfg = PowerSweepConfig {
        n_min: 20,
        n_max: 60,
        n_steps: 3,
        replicates: 15,
        censor_horizon: 4.0,
        dropout_rate: 0.05,
        sig_alpha: 0.05,
    };
    let mut workspace = PowerSweepWorkspace::with_capacity(60, 3);
    let (res, _) =
        VerifiedEngine::run_verified(|| run_power_sweep_into(&params, &cfg, 42, &mut workspace));
    let (steps, rep_summary, _) = res.expect("zero-alloc sweep must succeed");
    assert_eq!(steps, 3);
    assert_eq!(rep_summary.n_treatment, 60);
}

#[test]
fn power_sweep_is_deterministic_and_validates_bounds() {
    let params = SimulationParams::new(0.15, 0.35, 0.75, 0.65);
    let cfg = PowerSweepConfig::default();
    let r1 = run_weibull_power_sweep(&params, &cfg, 777).unwrap();
    let r2 = run_weibull_power_sweep(&params, &cfg, 777).unwrap();
    assert_eq!(r1, r2);

    let mut bad_cfg = cfg;
    bad_cfg.n_max = 501;
    assert!(run_weibull_power_sweep(&params, &bad_cfg, 1).is_err());

    let mut bad_rep = cfg;
    bad_rep.replicates = 501;
    assert!(run_weibull_power_sweep(&params, &bad_rep, 1).is_err());
}

#[test]
fn analytical_power_curves_invert_sample_size_formulas() {
    let n_means = calculate_sample_size_means(0.05, 0.80, 0.5, 1.0).unwrap();
    let p_at_n = calculate_power_means(0.05, n_means, 0.5, 1.0).unwrap();
    let p_below_n = calculate_power_means(0.05, n_means - 1, 0.5, 1.0).unwrap();
    assert!(p_at_n >= 0.80 && p_below_n < 0.80);

    let n_prop = calculate_sample_size_proportions(0.05, 0.80, 0.25, 0.45).unwrap();
    let pp_at_n = calculate_power_proportions(0.05, n_prop, 0.25, 0.45).unwrap();
    let pp_below_n = calculate_power_proportions(0.05, n_prop - 1, 0.25, 0.45).unwrap();
    assert!(pp_at_n >= 0.80 && pp_below_n < 0.80);
}
