#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use scientific_metadata::theory::TheoryDescribable;

#[test]
fn test_four_format_synchronization_via_odds() {
    let mut tool = BetSizeCalculatorTool {
        odds_format: OddsFormat::American,
        american_odds: -200.0,
        ..Default::default()
    };
    let odds = tool.resolve_active_odds().unwrap();
    tool.sync_from_odds(odds, OddsFormat::American);
    assert!((tool.decimal_odds - 1.5).abs() < 1e-9);
    assert_eq!(tool.fractional_numerator, 1.0);
    assert_eq!(tool.fractional_denominator, 2.0);
    assert!((tool.implied_probability_input - (200.0 / 3.0)).abs() < 1e-6);

    // Sync from ImpliedProbability 25% (decimal 4.0, American +300, fractional 3/1)
    tool.odds_format = OddsFormat::ImpliedProbability;
    tool.implied_probability_input = 25.0;
    let odds2 = tool.resolve_active_odds().unwrap();
    tool.sync_from_odds(odds2, OddsFormat::ImpliedProbability);
    assert!((tool.decimal_odds - 4.0).abs() < 1e-9);
    assert!((tool.american_odds - 300.0).abs() < 1e-9);
    assert_eq!(tool.fractional_numerator, 3.0);
    assert_eq!(tool.fractional_denominator, 1.0);
}

#[test]
fn test_american_dead_zone_skip_across_all_invalid_steps() {
    for step in -99..=99 {
        let up = skip_american_dead_zone(-100.0, f64::from(step));
        assert_eq!(up, 100.0);
        assert!(Odds::from_american(up).is_ok());

        let down = skip_american_dead_zone(100.0, f64::from(step));
        assert_eq!(down, -100.0);
        assert!(Odds::from_american(down).is_ok());
    }
    assert_eq!(skip_american_dead_zone(-101.0, -100.0), -100.0);
    assert_eq!(skip_american_dead_zone(101.0, 100.0), 100.0);
}

#[test]
fn test_try_load_state_rejects_dead_zone_and_roundtrips_implied_probability() {
    let mut tool = BetSizeCalculatorTool::default();
    let dead_zone_json = r#"{"probability_input":55.0,"odds_format":"American","decimal_odds":2.0,"american_odds":50.0,"fractional_numerator":1.0,"fractional_denominator":1.0,"bankroll":1000.0}"#;
    assert!(tool.try_load_state(dead_zone_json).is_err());

    let zero_american_json = r#"{"probability_input":55.0,"odds_format":"American","decimal_odds":2.0,"american_odds":0.0,"fractional_numerator":1.0,"fractional_denominator":1.0,"bankroll":1000.0}"#;
    assert!(tool.try_load_state(zero_american_json).is_err());

    let neg_100_json = r#"{"probability_input":55.0,"odds_format":"American","decimal_odds":2.0,"american_odds":-100.0,"fractional_numerator":1.0,"fractional_denominator":1.0,"bankroll":1000.0}"#;
    assert!(tool.try_load_state(neg_100_json).is_ok());
    assert!(tool.has_finite_parameters());

    let imp_json = r#"{"probability_input":60.0,"odds_format":"ImpliedProbability","decimal_odds":2.5,"american_odds":150.0,"fractional_numerator":3.0,"fractional_denominator":2.0,"implied_probability_input":40.0,"bankroll":1500.0}"#;
    assert!(tool.try_load_state(imp_json).is_ok());
    let saved = tool.save_state().unwrap();
    let expected: serde_json::Value = serde_json::from_str(imp_json).unwrap();
    let actual: serde_json::Value = serde_json::from_str(&saved).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn test_theory_description_reports_live_net_multiplier_b() {
    let mut tool = BetSizeCalculatorTool {
        probability_input: 55.0,
        odds_format: OddsFormat::American,
        decimal_odds: 5.0, // Stale decimal field
        american_odds: -200.0,
        ..Default::default()
    };
    let desc = tool.theory_description();
    assert!(
        desc.contains("net odds $b = 0.50$"),
        "Unexpected theory_description: {desc}"
    );

    tool.odds_format = OddsFormat::Decimal;
    tool.decimal_odds = 2.0;
    assert!(tool.theory_description().contains("net odds $b = 1.00$"));
}

#[test]
fn test_dataset_only_csv_and_json_exports() {
    let tool = BetSizeCalculatorTool::default();
    let caps = tool.export_capabilities();
    assert!(caps.dataset);
    assert!(!caps.figure);
    assert!(caps.dataset_disabled_reason.is_empty());
    assert!(!caps.figure_disabled_reason.is_empty());

    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 3);
    let csv = egui_plot::export::format_csv(&datasets);
    assert!(csv.contains("Kelly Bet Fraction"));
    assert!(csv.contains("Kelly Bet Amount ($)"));
    let json = egui_plot::export::format_json(&datasets);
    assert!(json.contains("\"Kelly Bet Fraction\""));
    assert!(json.contains("\"Odds & Edge Summary\""));
}

// [cite:modular_polynomials_review]
