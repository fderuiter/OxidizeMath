#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_bankroll_growth_reset_defaults() {
    let mut tool = BankrollGrowthTool {
        initial_bankroll: 5000.0,
        probability: 0.70,
        odds: 3.0,
        num_bets: 500,
        stake_multiplier: 2.4,
        ..Default::default()
    };
    tool.recalculate();
    tool.reset_defaults();

    assert_eq!(tool.initial_bankroll, 1000.0);
    assert_eq!(tool.probability, 0.55);
    assert_eq!(tool.odds, 2.0);
    assert_eq!(tool.num_bets, 100);
    assert_eq!(tool.stake_multiplier, 1.0);
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
fn test_net_profit_multiplier_rerun_and_overbet_growth() {
    let mut w = 1000.0;
    let odds = Odds::new(2.0).expect("valid decimal odds");
    step_bankroll(&mut w, 0.2, odds.net_profit_multiplier(), true);
    assert!(
        (w - 1200.0).abs() < 1e-9,
        "even-money win credits 1x stake, not 2x"
    );

    let mut tool = BankrollGrowthTool::default();
    tool.try_load_state(
        r#"{"initial_bankroll":1000.0,"probability":0.6,"odds":2.0,"num_bets":120,"stake_multiplier":2.5}"#,
    )
    .expect("overbet state loads");
    let obs = tool.observables();
    let ev = obs.iter().find(|(k, _)| *k == "expected_value").unwrap().1;
    let g = obs
        .iter()
        .find(|(k, _)| *k == "expected_growth_rate")
        .unwrap()
        .1;
    assert!(
        ev > 0.15 && g < -0.02,
        "above 2f*, EV > 0 yet log-growth g < 0"
    );

    let before: Vec<f64> = tool.full_kelly_points.iter().map(|p| p.y).collect();
    tool.rerun_simulation();
    let after: Vec<f64> = tool.full_kelly_points.iter().map(|p| p.y).collect();
    assert_ne!(
        before, after,
        "rerun_simulation must advance deterministic RNG seed"
    );
}
