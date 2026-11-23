#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::export::{format_protocol_csv, format_protocol_markdown};
use super::*;

#[test]
fn test_means_and_proportions_power_curves_monotone_and_exportable() {
    let mut tool = SampleSizeCalculatorTool::default();
    assert!(tool.export_capabilities().figure && tool.export_capabilities().dataset);
    assert_eq!(tool.result, Ok(63));
    assert_eq!(tool.power_curve.len(), 64);
    assert_eq!(tool.protocol_rows.len(), POWER_MILESTONES.len());
    let achieved_63 = tool
        .evaluate_power_at(63)
        .expect("analytical power at N=63");
    assert!((0.80..0.82).contains(&achieved_63));

    // Analytical power 1 - beta must increase strictly with per-arm sample size N
    for pair in tool.power_curve.windows(2) {
        assert!(
            pair[1][0] > pair[0][0] && pair[1][1] > pair[0][1],
            "Analytical power curve must be strictly increasing in N: {pair:?}"
        );
    }

    tool.mode = CalculationMode::Proportions;
    tool.p1 = 0.30;
    tool.p2 = 0.50;
    tool.power = 0.90;
    tool.recalculate();
    let req_props = *tool.result.as_ref().expect("valid proportions sample size");
    assert!(req_props > 50);
    assert_eq!(tool.export_datasets().len(), 3);

    let csv = format_protocol_csv(&tool);
    assert!(csv.starts_with(
        "mode,alpha,effect_param_1,effect_param_2,target_power,n_per_arm,total_subjects\n"
    ));
    assert!(csv.contains("Proportions,0.0500,0.3000,0.5000,0.9000"));

    let md = format_protocol_markdown(&tool);
    assert!(md.contains("Compare Proportions"));
    assert!(md.contains("| 90% (0.90) |"));
}

#[test]
fn test_sample_size_preset_and_bounds_validation() {
    let mut tool = SampleSizeCalculatorTool::default();
    tool.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset valid");
    assert!((tool.power - 0.85).abs() < 1e-12);
    let saved = tool.save_state().expect("serializable state");
    let mut restored = SampleSizeCalculatorTool::default();
    restored.try_load_state(&saved).expect("roundtrip state");
    assert_eq!(restored.result, tool.result);

    let equal_props = r#"{"mode":"Proportions","alpha":0.05,"power":0.8,"delta":0.5,"sigma":1.0,"p1":0.4,"p2":0.4}"#;
    assert!(tool.try_load_state(equal_props).is_err());
}

#[test]
fn test_legacy_roundtrip_power_curves_and_hypothesis_cards() {
    let mut tool = SampleSizeCalculatorTool::default();
    let legacy = r#"{"mode":"Proportions","alpha":0.01,"power":0.9,"delta":0.4,"sigma":1.2,"p1":0.3,"p2":0.5}"#;
    tool.try_load_state(legacy)
        .expect("legacy proportions state loads");
    assert_eq!(tool.save_state().as_deref(), Some(legacy));

    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 3);
    assert!(datasets.iter().all(|(_, pts)| pts.len() >= 20));
    let req_n = *tool.result.as_ref().expect("valid sample size");
    assert!(tool.evaluate_power_at(req_n).is_some_and(|p| p >= 0.9));

    let t_res = tool.cards_eval.t_test.as_ref().expect("t-test evaluated");
    assert!(t_res.statistic > 0.0 && t_res.is_significant);
    let chi_res = tool
        .cards_eval
        .chi_square
        .as_ref()
        .expect("chi-square evaluated");
    assert!(chi_res.statistic > 0.0);
    let rm = tool
        .cards_eval
        .risk_metrics
        .as_ref()
        .expect("risk metrics evaluated");
    assert!(rm.relative_risk < 1.0 && rm.odds_ratio < 1.0);
}

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_wasm_sample_size_power_curve_ui_and_exports() {
        let mut tool = SampleSizeCalculatorTool::default();
        let ctx = egui::Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(1024.0, 768.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(raw_input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                tool.show_inside(ui);
            });
        });
        assert!(!tool.export_datasets().is_empty());
        assert!(format_protocol_markdown(&tool).contains("Protocol Power Milestones"));
    }
}

// [cite:clinical_trials_statistics]
