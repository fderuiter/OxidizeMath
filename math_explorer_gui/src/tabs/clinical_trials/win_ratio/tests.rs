#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::cohort_io::{load_and_analyze_cohort, SAMPLE_COHORT_CSV};
use super::{ChartMode, WinRatioStressTesterTool, STARTER_PRESET_JSON};
use crate::framework::InteractiveTool;
use crate::url_state::{
    encode_share_fragment, parse_share_location, validate_url_preset_json,
    MAX_DECODED_URL_STATE_BYTES,
};

#[test]
fn starter_preset_and_compact_url_state_omit_uploaded_cohort_rows() {
    let mut tool = WinRatioStressTesterTool::default();
    tool.apply_cohort_content("sample.csv", SAMPLE_COHORT_CSV)
        .expect("sample cohort loads");
    assert!(tool.uploaded_cohort.is_some());

    let saved = tool.save_state().expect("state serializes");
    assert!(
        saved.len() < 400 && saved.len() < MAX_DECODED_URL_STATE_BYTES,
        "State JSON length {} must remain compact",
        saved.len()
    );
    assert!(!saved.contains("sample.csv"));
    assert!(!saved.contains("fatal_time"));

    let envelope = tool.export_preset_json().expect("preset envelope exports");
    validate_url_preset_json(&envelope).expect("preset fits within URL_JSON_BOUNDS");
    let frag = encode_share_fragment(
        "clinical_trials",
        Some("WinRatioStressTesterTool"),
        Some(&envelope),
    )
    .expect("URL fragment encodes");
    let parsed = parse_share_location(&frag, "").unwrap().unwrap();
    assert_eq!(parsed.tab, "clinical_trials");
    assert_eq!(parsed.tool.as_deref(), Some("WinRatioStressTesterTool"));

    let mut restored = WinRatioStressTesterTool::default();
    restored.import_preset_json(STARTER_PRESET_JSON).unwrap();
    assert_eq!(restored.chart_mode, ChartMode::TierWaterfall);
    assert_eq!(restored.seed, 4242);
}

#[test]
fn local_csv_and_json_cohort_import_computes_wr_hr_and_risk_metrics() {
    let csv_analysis =
        load_and_analyze_cohort("cardio.csv", SAMPLE_COHORT_CSV, 0.05, None).unwrap();
    assert_eq!(csv_analysis.treatment.len(), 8);
    assert_eq!(csv_analysis.control.len(), 8);
    assert!(csv_analysis.summary.win_ratio > 1.5);
    assert!(csv_analysis.summary.ci_low > 0.0);
    assert!(csv_analysis.fatal_hazard_ratio.unwrap() < 1.0);
    let rm = csv_analysis.risk_metrics.unwrap();
    assert!(rm.relative_risk.is_finite() && rm.odds_ratio.is_finite());

    let json_payload = r#"{
        "treatment": [
            {"fatal_time": 6.0, "fatal_event": 0, "nonfatal_time": 5.2, "nonfatal_event": 0},
            {"fatal_time": 5.5, "fatal_event": 0, "nonfatal_time": 3.8, "nonfatal_event": 1},
            {"fatal_time": 4.9, "fatal_event": 1, "nonfatal_time": 2.5, "nonfatal_event": 1}
        ],
        "control": [
            {"fatal_time": 1.8, "fatal_event": 1, "nonfatal_time": 0.9, "nonfatal_event": 1},
            {"fatal_time": 2.4, "fatal_event": 1, "nonfatal_time": 1.4, "nonfatal_event": 1},
            {"fatal_time": 5.0, "fatal_event": 0, "nonfatal_time": 2.0, "nonfatal_event": 1}
        ]
    }"#;
    let json_analysis = load_and_analyze_cohort("cardio.json", json_payload, 0.05, Some(5.0))
        .expect("json cohort loads");
    assert_eq!(json_analysis.summary.total_pairs, 9);
    assert!(json_analysis.summary.win_ratio > 1.0);
}

#[test]
fn malformed_non_finite_negative_and_oversized_cohorts_are_rejected() {
    for bad_csv in [
        "",
        "arm,fatal_time,fatal_event,nonfatal_time,nonfatal_event\ntreatment,NaN,1,2.0,0\ntreatment,3.0,0,2.0,0\ncontrol,1.0,1,0.5,1\ncontrol,2.0,1,1.0,1",
        "arm,fatal_time,fatal_event,nonfatal_time,nonfatal_event\ntreatment,inf,1,2.0,0\ntreatment,3.0,0,2.0,0\ncontrol,1.0,1,0.5,1\ncontrol,2.0,1,1.0,1",
        "arm,fatal_time,fatal_event,nonfatal_time,nonfatal_event\ntreatment,-1.0,1,2.0,0\ntreatment,3.0,0,2.0,0\ncontrol,1.0,1,0.5,1\ncontrol,2.0,1,1.0,1",
        "arm,fatal_time,fatal_event,nonfatal_time,nonfatal_event\ntreatment,2.0,9,2.0,0\ntreatment,3.0,0,2.0,0\ncontrol,1.0,1,0.5,1\ncontrol,2.0,1,1.0,1",
        "arm,fatal_time,fatal_event,nonfatal_time,nonfatal_event\ntreatment,2.0,1,1.0,1\ntreatment,3.0,0,2.0,0",
    ] {
        assert!(
            load_and_analyze_cohort("bad.csv", bad_csv, 0.05, None).is_err(),
            "Expected cohort rejection for: {bad_csv}"
        );
    }

    let mut oversized = String::from("arm,fatal_time,fatal_event,nonfatal_time,nonfatal_event\n");
    for _ in 0..501 {
        oversized.push_str("treatment,3.0,0,2.0,1\n");
    }
    oversized.push_str("control,1.0,1,0.5,1\ncontrol,2.0,1,1.0,1\n");
    assert!(load_and_analyze_cohort("oversized.csv", &oversized, 0.05, None).is_err());
}

#[test]
fn protocol_exports_and_plot_primitives_are_non_empty_across_modes() {
    let mut tool = WinRatioStressTesterTool::default();
    tool.apply_cohort_content("sample.csv", SAMPLE_COHORT_CSV)
        .unwrap();

    let csv_out = tool.build_protocol_csv();
    assert!(csv_out.contains("tier_waterfall") && csv_out.contains("power_by_n"));

    let json_out = tool.build_protocol_json();
    let parsed: serde_json::Value = serde_json::from_str(&json_out).unwrap();
    assert_eq!(parsed["tool"], "WinRatioStressTesterTool");
    assert!(parsed["tier_waterfall"].as_array().unwrap().len() == 2);
    assert!(!parsed["power_by_n"].as_array().unwrap().is_empty());

    for mode in [
        ChartMode::PowerCurve,
        ChartMode::TierWaterfall,
        ChartMode::CohortSurvival,
    ] {
        tool.chart_mode = mode;
        assert!(!tool.export_datasets().is_empty());
        assert!(!tool.export_plot_primitives().is_empty());
    }
}
