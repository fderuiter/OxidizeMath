#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{CopulaPlotView, CopulaRiskTool, CorrelationStructureMode, STARTER_PRESET_JSON};
use crate::framework::InteractiveTool;

#[test]
fn test_copula_risk_tool_2_to_8_legs_and_psd_rejection() {
    let mut tool = CopulaRiskTool::default();
    assert!(tool.export_capabilities().figure && tool.export_capabilities().dataset);
    for legs in 2..=8 {
        tool.resize_legs(legs);
        tool.equicorrelation = 0.25;
        tool.recalculate();
        assert!(
            tool.calc_error.is_none(),
            "unexpected error for {legs} legs: {:?}",
            tool.calc_error
        );
        let jp = tool.joint_probability.expect("joint prob");
        assert!(jp > tool.naive_probability);
        assert!(!tool.sweep_copula_points.is_empty());
        assert!(!tool.contour_curves.is_empty());
    }

    // Indefinite 3x3 custom matrix must be rejected cleanly with error message
    tool.resize_legs(3);
    tool.mode = CorrelationStructureMode::PairwiseMatrix;
    tool.pairwise_upper = vec![0.92, 0.92, -0.85];
    tool.recalculate();
    assert!(tool.joint_probability.is_none());
    assert!(tool
        .calc_error
        .as_deref()
        .unwrap_or("")
        .contains("positive definite"));
}

#[test]
fn test_copula_risk_csv_json_pricing_and_preset_roundtrip() {
    let mut tool = CopulaRiskTool::default();
    tool.leg_labels[0] = "=HYPERLINK(\"http://evil\")".to_string();
    tool.recalculate();

    let csv = tool.export_pricing_csv();
    assert!(csv.contains("\"'=HYPERLINK"));
    assert!(csv.contains("ActiveParlay"));

    let json = tool.export_pricing_json().expect("valid JSON sheet");
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["num_legs"], 3);
    assert!(parsed["correlation_sweep_schedule"]
        .as_array()
        .map(|a| !a.is_empty())
        .unwrap_or(false));

    tool.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset loads");
    assert_eq!(tool.num_legs, 4);
    tool.plot_view = CopulaPlotView::BivariateContours;
    let saved = tool.save_state().expect("serializes state");
    let mut restored = CopulaRiskTool::default();
    restored.try_load_state(&saved).expect("restores state");
    assert_eq!(restored.num_legs, 4);
    assert_eq!(restored.plot_view, CopulaPlotView::BivariateContours);
}

// [cite:clinical_trials_statistics]
