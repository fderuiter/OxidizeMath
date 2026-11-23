#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use math_explorer::pure_math::statistics::validation::{devig_two_way, single_bet_clv};

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

    let placed = tool.current_placed_odds().unwrap();
    let (p_close, _) = devig_two_way(
        &Odds::new(tool.closing_for_decimal).unwrap(),
        &Odds::new(tool.closing_against_decimal).unwrap(),
    )
    .unwrap();
    let clv = single_bet_clv(&placed, &p_close).unwrap();
    assert!((clv.ev_pct - 5.0).abs() < 1e-9);

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            tool.show_inside(ui);
        });
    });
}

// [cite:modular_polynomials_review]
