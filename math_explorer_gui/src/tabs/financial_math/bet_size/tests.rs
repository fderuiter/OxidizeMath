#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::export::{format_kelly_rows_csv, format_kelly_rows_markdown};
use super::{skip_american_dead_zone, BetSizeCalculatorTool, KellyStrategyRow, OddsFormat};
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::Odds;
use math_explorer::pure_math::statistics::validation::{devig_two_way, single_bet_clv};
use scientific_metadata::theory::TheoryDescribable;

#[test]
fn test_bet_size_growth_curve_and_plot_exports() {
    let tool = BetSizeCalculatorTool::default();
    let caps = tool.export_capabilities();
    assert!(caps.dataset && caps.figure);

    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 5);
    assert_eq!(datasets[1].0, "Expected Growth Rate g(f)");
    assert!(datasets[1].1.len() >= 100);

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

    // Full Kelly maximizes g(f): g(0.25 f*) < g(0.50 f*) < g(1.00 f*)
    let q_pt = tool.quarter_marker[0];
    let h_pt = tool.half_marker[0];
    let f_pt = tool.full_marker[0];
    assert!(q_pt.x < h_pt.x && h_pt.x < f_pt.x);
    assert!(q_pt.y > 0.0 && q_pt.y < h_pt.y && h_pt.y < f_pt.y);

    // Full Kelly 50%-drawdown Lundberg bound equals 50%
    let full_row = tool
        .strategy_rows
        .iter()
        .find(|r| r.label == "Full Kelly")
        .expect("Full Kelly row");
    assert!((full_row.barrier_bound.expect("barrier bound") - 0.50).abs() < 1e-6);

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
    assert!(md.contains("| Quarter Kelly |"));
    assert!(md.contains("| Full Kelly |"));

    let injected = vec![KellyStrategyRow {
        label: "=CMD|' /C calc'!A0".to_string(),
        fraction: 0.1,
        bet_amount: 100.0,
        growth_rate: 0.005,
        ev_per_unit: 0.10,
        barrier_bound: Some(0.50),
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
fn test_bet_size_legacy_json_roundtrip_and_clv_state_persistence() {
    let legacy = r#"{"probability_input":60.0,"odds_format":"American","decimal_odds":2.2,"american_odds":120.0,"fractional_numerator":6.0,"fractional_denominator":5.0,"bankroll":2000.0}"#;
    let mut tool = BetSizeCalculatorTool::default();
    tool.try_load_state(legacy)
        .expect("legacy BetSizeState JSON must load");
    assert_eq!(
        tool.save_state().as_deref(),
        Some(legacy),
        "legacy BetSizeState JSON must re-serialize identically when optional fields are default"
    );

    tool.closing_line_enabled = true;
    tool.closing_for_decimal = 1.85;
    tool.closing_against_decimal = 2.05;
    tool.drawdown_floor_pct = 60.0;
    let extended = tool.save_state().expect("extended state JSON");

    let mut restored = BetSizeCalculatorTool::default();
    restored
        .try_load_state(&extended)
        .expect("extended BetSizeState JSON must round-trip");
    assert!(restored.closing_line_enabled);
    assert!((restored.closing_for_decimal - 1.85).abs() < 1e-12);
    assert!((restored.drawdown_floor_pct - 60.0).abs() < 1e-12);
    assert!(restored.clv_audit.is_some());
}

#[test]
fn test_bet_size_verified_solvency_clv_and_headless_render() {
    let mut tool = BetSizeCalculatorTool {
        probability_input: 55.0,
        odds_format: OddsFormat::Decimal,
        decimal_odds: 2.10,
        closing_line_enabled: true,
        closing_for_decimal: 1.91,
        closing_against_decimal: 1.91,
        ..Default::default()
    };
    tool.recalculate();

    let placed = tool.resolve_active_odds().unwrap();
    let (p_close, _) = devig_two_way(
        &Odds::new(tool.closing_for_decimal).unwrap(),
        &Odds::new(tool.closing_against_decimal).unwrap(),
    )
    .unwrap();
    let clv = single_bet_clv(&placed, &p_close).unwrap();
    assert!((clv.ev_pct - 5.0).abs() < 1e-9);
    assert!(tool.lundberg_summary.is_some());
    assert!(tool.sample_size_95.is_some());

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            tool.show_inside(ui);
        });
    });
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

// [cite:modular_polynomials_review]
