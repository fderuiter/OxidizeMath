#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

#[test]
fn test_bet_size_growth_curve_and_plot_exports() {
    let tool = BetSizeCalculatorTool::default();
    let caps = tool.export_capabilities();
    assert!(caps.dataset && caps.figure);

    let datasets = tool.export_datasets();
    assert!(!datasets.is_empty());
    assert_eq!(datasets[0].0, "Logarithmic Growth Rate g(f)");
    assert!(datasets[0].1.len() >= 100);

    let prims = tool.export_plot_primitives();
    assert!(!prims.is_empty());

    // At default p=55%, decimal_odds=2.0 (b=1.0), f* = 0.10 maximizes g(f).
    let f_star = tool.optimal_f_star.expect("positive edge");
    assert!((f_star - 0.10).abs() < 1e-9);
    assert!(tool.optimal_growth_rate > 0.004);
}

#[test]
fn test_bet_size_state_roundtrip_recomputes_curve() {
    let mut tool = BetSizeCalculatorTool::default();
    tool.try_load_state(
        r#"{"probability_input":60.0,"odds_format":"Decimal","decimal_odds":2.0,"american_odds":100.0,"fractional_numerator":1.0,"fractional_denominator":1.0,"bankroll":2500.0}"#,
    )
    .expect("loads state");
    assert!((tool.optimal_f_star.unwrap() - 0.20).abs() < 1e-9);
    let saved = tool.save_state().expect("serializes state");
    let mut restored = BetSizeCalculatorTool::default();
    restored.try_load_state(&saved).expect("round-trips");
    assert_eq!(restored.save_state(), Some(saved));
}

// [cite:modular_polynomials_review]
