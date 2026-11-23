#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::export::{format_kelly_rows_csv, format_kelly_rows_markdown};
use super::*;
use crate::framework::InteractiveTool;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    push_expected_value, push_kelly_fraction, quarter_line_expected_value,
    quarter_line_kelly_fraction, solve_drawdown_constrained_slate, solve_mutually_exclusive_market,
    solve_simultaneous_slate, MarketOutcome, Odds, PushWager, QuarterLineWager, SlateWager,
    UnitInterval,
};
use scientific_metadata::theory::TheoryDescribable;

#[test]
fn legacy_bet_size_json_roundtrips_identically() {
    let legacy = r#"{"probability_input":60.0,"odds_format":"American","decimal_odds":2.2,"american_odds":120.0,"fractional_numerator":6.0,"fractional_denominator":5.0,"bankroll":2000.0}"#;
    let mut tool = BetSizeCalculatorTool::default();
    tool.try_load_state(legacy)
        .expect("legacy state must parse");
    let saved = tool.save_state().expect("save_state must succeed");
    assert_eq!(saved, legacy);
}

#[test]
fn extended_multi_state_bet_size_json_roundtrips() {
    let mut tool = BetSizeCalculatorTool::default();
    tool.load_push_spread_preset();
    tool.drawdown_alpha = 0.75;
    tool.drawdown_beta = 0.05;
    let saved = tool.save_state().expect("save extended state");

    let mut restored = BetSizeCalculatorTool::default();
    restored
        .try_load_state(&saved)
        .expect("extended state must load");
    assert_eq!(restored.mode, BetSizingMode::PushSpread);
    assert!((restored.american_odds - (-110.0)).abs() < 1e-12);
    assert!((restored.drawdown_alpha - 0.75).abs() < 1e-12);
    assert!((restored.drawdown_beta - 0.05).abs() < 1e-12);
}

#[test]
fn canonical_single_wager_presets_match_exact_solver_contracts() {
    let mut tool = BetSizeCalculatorTool::default();

    tool.load_push_spread_preset();
    let odds = tool.active_odds().unwrap();
    let pw = PushWager::from_win_and_push(
        tool.probability_input / 100.0,
        tool.push_probability_pct / 100.0,
        odds,
    )
    .unwrap();
    assert!((push_expected_value(&pw) - 3.0 / 55.0).abs() < 1e-12);
    assert!((push_kelly_fraction(&pw).unwrap().value() - 1.0 / 15.0).abs() < 1e-12);
    assert!(tool.calc_error.is_none());
    assert!(tool.growth_curve_points.len() > 50);

    tool.load_quarter_line_preset();
    let qw = QuarterLineWager::new(
        tool.ql_full_win_pct / 100.0,
        tool.ql_half_win_pct / 100.0,
        tool.ql_push_pct / 100.0,
        tool.ql_half_loss_pct / 100.0,
        tool.ql_full_loss_pct / 100.0,
        tool.active_odds().unwrap(),
    )
    .unwrap();
    assert!((quarter_line_expected_value(&qw) - 0.16 / 11.0).abs() < 1e-12);
    assert!((quarter_line_kelly_fraction(&qw).unwrap().value() - 0.0175166500852366).abs() < 1e-9);
    assert!(tool.calc_error.is_none());
}

#[test]
fn canonical_portfolio_presets_match_exact_solver_contracts() {
    let mut tool = BetSizeCalculatorTool::default();

    tool.load_three_way_market_preset();
    let outcomes = [
        MarketOutcome::new(UnitInterval::new(0.50).unwrap(), Odds::new(2.20).unwrap()),
        MarketOutcome::new(UnitInterval::new(0.28).unwrap(), Odds::new(4.00).unwrap()),
        MarketOutcome::new(UnitInterval::new(0.22).unwrap(), Odds::new(3.20).unwrap()),
    ];
    let alloc = solve_mutually_exclusive_market(&outcomes, &mut tool.market_scratch).unwrap();
    assert!((alloc.stakes()[0] - 21.0 / 130.0).abs() < 1e-12);
    assert!((alloc.stakes()[1] - 61.0 / 650.0).abs() < 1e-12);
    assert_eq!(alloc.stakes()[2], 0.0);
    assert_eq!(tool.strategy_rows.len(), 3);

    tool.load_ten_game_slate_preset();
    let sw = SlateWager::binary(
        UnitInterval::new(0.58).unwrap(),
        Odds::from_net_odds(1.0).unwrap(),
    );
    let wagers = [sw; 10];
    let sol = solve_simultaneous_slate(&wagers, &mut tool.slate_scratch).unwrap();
    assert!((sol.total_fraction() - 0.996724797394549).abs() < 1e-8);
    let dc = tool.active_drawdown_constraint().unwrap();
    let rc_sol = solve_drawdown_constrained_slate(&wagers, &dc, &mut tool.slate_scratch).unwrap();
    assert!((rc_sol.total_fraction() - 0.4148978997813092).abs() < 1e-8);
    assert!(tool.growth_curve_points.len() > 50);
}

#[test]
fn headless_egui_renders_all_five_modes_without_panic() {
    let ctx = egui::Context::default();
    let mut tool = BetSizeCalculatorTool::default();
    for mode in BetSizingMode::ALL {
        tool.mode = mode;
        tool.recalculate();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                tool.show_inside(ui);
            });
        });
    }
}

#[test]
fn test_bet_size_default_populates_growth_curve_and_kelly_markers() {
    let tool = BetSizeCalculatorTool::default();
    let caps = tool.export_capabilities();
    assert!(caps.figure && caps.dataset);
    assert!(tool.calc_error.is_none());
    assert!(tool.growth_curve_points.len() > 50);
    assert_eq!(tool.quarter_marker.len(), 1);
    assert_eq!(tool.half_marker.len(), 1);
    assert_eq!(tool.full_marker.len(), 1);
    assert_eq!(tool.drawdown_marker.len(), 1);

    // Full Kelly maximizes g(f): g(0.25 f*) < g(0.50 f*) < g(1.00 f*)
    let q_pt = tool.quarter_marker[0];
    let h_pt = tool.half_marker[0];
    let f_pt = tool.full_marker[0];
    assert!(q_pt.x < h_pt.x && h_pt.x < f_pt.x);
    assert!(q_pt.y > 0.0 && q_pt.y < h_pt.y && h_pt.y < f_pt.y);

    // Verify zero-allocation borrowed PlotPoints slice
    let slice: &[PlotPoint] = &tool.growth_curve_points[..];
    let plot_pts: egui_plot::PlotPoints<'_> = slice.into();
    assert!(matches!(plot_pts, egui_plot::PlotPoints::Borrowed(_)));

    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 5);
    assert!(egui_plot::export::format_csv(&datasets).contains("Expected Growth Rate g(f)"));
    assert!(egui_plot::export::format_json(&datasets).contains("\"Full Kelly (1.00 f*)\""));
}

#[test]
fn test_bet_size_csv_and_markdown_exports_escape_formula_injection() {
    let tool = BetSizeCalculatorTool::default();
    let csv = tool.export_sizing_csv();
    let md = tool.export_sizing_markdown();
    assert!(csv.contains("Quarter Kelly"));
    assert!(csv.contains("Half Kelly"));
    assert!(csv.contains("Full Kelly"));
    assert!(csv.contains("Drawdown-Constrained Kelly"));
    assert!(md.contains("| Quarter Kelly |"));
    assert!(md.contains("| Full Kelly |"));

    let injected = vec![KellyStrategyRow {
        label: "=CMD|' /C calc'!A0".to_string(),
        fraction: 0.1,
        bet_amount: 100.0,
        growth_rate: 0.005,
        ev_per_unit: 0.10,
    }];
    let safe_csv = format_kelly_rows_csv(&injected);
    assert!(
        safe_csv.contains("\"'=CMD|' /C calc'!A0\""),
        "Formula prefix must be neutralized with leading quote: {safe_csv}"
    );
    let safe_md = format_kelly_rows_markdown(&injected);
    assert!(safe_md.contains("\\|"));
}

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

// [cite:clinical_trials_statistics]
// [cite:modular_polynomials_review]
