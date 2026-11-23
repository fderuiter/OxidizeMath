#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_bankroll_growth_reset_defaults() {
    let mut tool = BankrollGrowthTool {
        initial_bankroll: 5000.0,
        probability: 0.70,
        odds: 3.0,
        num_bets: 500,
        ..Default::default()
    };
    tool.recalculate();

    tool.reset_defaults();

    assert_eq!(tool.initial_bankroll, 1000.0);
    assert_eq!(tool.probability, 0.55);
    assert_eq!(tool.odds, 2.0);
    assert_eq!(tool.num_bets, 100);
    assert_eq!(tool.full_kelly_points.len(), 101);
    assert!(tool.error_msg.is_none());
}

#[test]
fn test_bankroll_growth_borrowed_slice_points() {
    let tool = BankrollGrowthTool::default();
    let slice: &[PlotPoint] = &tool.full_kelly_points[..];
    let plot_points: egui_plot::PlotPoints<'_> = slice.into();
    assert!(matches!(plot_points, egui_plot::PlotPoints::Borrowed(_)));
    assert_eq!(plot_points.points().len(), tool.full_kelly_points.len());
}

#[test]
fn test_even_money_net_odds_and_net_profit_multiplier_settlement() {
    let mut tool = BankrollGrowthTool {
        initial_bankroll: 1000.0,
        probability: 0.60,
        odds: 1.0, // Even money net odds (b = 1.0 -> decimal 2.0, f* = 0.20)
        num_bets: 10,
        ..Default::default()
    };
    tool.recalculate();
    assert!(
        tool.error_msg.is_none(),
        "Even-money net odds b=1.0 must succeed via Odds::from_net_odds"
    );

    let o = Odds::from_net_odds(2.0).unwrap();
    let frac = BankrollFraction::new(0.25).unwrap();
    let settled_win = settle_bet(1000.0, &frac, o.net_profit_multiplier(), true);
    assert!((settled_win - 1500.0).abs() < 1e-9);
    let settled_loss = settle_bet(1000.0, &frac, o.net_profit_multiplier(), false);
    assert!((settled_loss - 750.0).abs() < 1e-9);
}

#[test]
fn test_seed_counter_reruns_produce_distinct_trajectories() {
    let mut tool = BankrollGrowthTool::default();
    let run_0: Vec<[f64; 2]> = tool
        .full_kelly_points
        .iter()
        .map(|pt| [pt.x, pt.y])
        .collect();
    assert_eq!(tool.seed_counter, 0);

    tool.rerun_simulation();
    assert_eq!(tool.seed_counter, 1);
    let run_1: Vec<[f64; 2]> = tool
        .full_kelly_points
        .iter()
        .map(|pt| [pt.x, pt.y])
        .collect();
    assert_ne!(run_0, run_1, "Incrementing seed_counter must vary paths");
}

#[test]
fn test_f_030_overbetting_divergence_callout_and_zero_allocation() {
    let mut lab = BankrollEnsembleLab::new();
    lab.custom_fraction = 0.30;
    let p = UnitInterval::new(0.55).unwrap();
    let o = Odds::from_net_odds(1.0).unwrap();

    // Warm up once, then verify zero heap allocations under VerifiedAllocator.
    lab.evaluate_in_place(100.0, &p, &o, 200, 0);
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            lab.evaluate_in_place(100.0, &p, &o, 200, 1);
        });
    }
    lab.sync_callouts(200);

    assert!(lab.is_ruin_zone);
    assert!(lab.last_growth_rate < 0.0);
    assert!((lab.last_median_terminal - 3.9137).abs() < 1e-3);
    assert!((lab.last_expected_terminal - 36_935.58).abs() < 0.05);
    assert!(
        lab.callout_manager
            .callouts()
            .iter()
            .any(|c| c.tag.as_deref() == Some("ruin_warning")),
        "Ruin warning callout must be set when f > 2 * f*"
    );
}

#[test]
fn test_full_kelly_50pct_drawdown_risk_and_half_kelly_reduction() {
    let mut lab = BankrollEnsembleLab::new();
    let p = UnitInterval::new(0.55).unwrap();
    let o = Odds::from_net_odds(1.0).unwrap(); // even money (decimal 2.0), f* = 0.10
    lab.custom_fraction = 0.10;
    lab.barrier_floor_ratio = 0.50;

    lab.evaluate_in_place(100.0, &p, &o, 200, 0);
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            lab.evaluate_in_place(100.0, &p, &o, 200, 0);
        });
    }

    assert!(!lab.is_ruin_zone);
    assert!(lab.last_growth_rate > 0.0);
    let fk_barrier = lab.last_barrier_risk.expect("Full Kelly barrier risk");
    assert!(
        (fk_barrier.lundberg_barrier_bound - 0.50).abs() < 1e-6,
        "Full Kelly asymptotic 50%-drawdown bound must equal 50%, got {}",
        fk_barrier.lundberg_barrier_bound
    );
    let fk_rep = lab
        .last_solvency_report
        .expect("Full Kelly solvency report");
    assert!(
        (fk_rep.ruin_probability - 0.38).abs() < 0.04,
        "Expected ~38% 200-bet 50%-drawdown rate at Full Kelly, got {}",
        fk_rep.ruin_probability
    );
    assert!(
        fk_rep.wilson_95.lower <= fk_rep.ruin_probability
            && fk_rep.ruin_probability <= fk_rep.wilson_95.upper
    );

    // Half Kelly (f = 0.05) slashes asymptotic 50%-drawdown risk to ~12.5%
    lab.custom_fraction = 0.05;
    lab.evaluate_in_place(100.0, &p, &o, 200, 0);
    let hk_barrier = lab.last_barrier_risk.expect("Half Kelly barrier risk");
    assert!((hk_barrier.lundberg_barrier_bound - 0.1246).abs() < 1e-3);
    let hk_rep = lab
        .last_solvency_report
        .expect("Half Kelly solvency report");
    assert!(hk_rep.ruin_probability < 0.12);
}

#[test]
fn test_bankroll_state_backward_compatible_json_and_barrier_roundtrip() {
    let legacy = r#"{"initial_bankroll":2500.0,"probability":0.6,"odds":2.5,"num_bets":50}"#;
    let mut tool = BankrollGrowthTool::default();
    tool.try_load_state(legacy)
        .expect("legacy 4-field JSON must deserialize cleanly");
    assert_eq!(tool.save_state().as_deref(), Some(legacy));
    assert!((tool.ensemble_lab.barrier_floor_ratio - 0.50).abs() < 1e-12);

    tool.mode = BankrollLabMode::VolatilityDrag;
    tool.ensemble_lab.custom_fraction = 0.15;
    tool.ensemble_lab.barrier_floor_ratio = 0.65;
    let saved = tool.save_state().expect("saved custom barrier state");
    let mut restored = BankrollGrowthTool::default();
    restored
        .try_load_state(&saved)
        .expect("custom barrier state must round-trip");
    assert!((restored.ensemble_lab.barrier_floor_ratio - 0.65).abs() < 1e-12);
    assert!(restored
        .try_load_state(
            r#"{"initial_bankroll":1000.0,"probability":0.55,"odds":2.0,"num_bets":100,"barrier_floor_ratio":1.2}"#
        )
        .is_err());
}

// [cite:clinical_trials_statistics]
