#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::url_state::validate_url_preset_json;

#[test]
fn test_default_workbench_evaluates_tiers_and_exports_datasets() {
    let tool = WinRatioWorkbenchTool::default();
    assert!(tool.error_message.is_none());
    let eval = tool.cohort_eval.as_ref().expect("default cohort evaluated");
    assert_eq!(eval.tiers.len(), 3);
    assert_eq!(eval.n_treatment, 6);
    assert_eq!(eval.n_control, 6);
    assert!(eval.total_wins > eval.total_losses);
    assert!(eval.sample_wr > 1.0);
    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 3);
}

#[test]
fn test_csv_and_tsv_with_utf8_bom_and_threshold_comparator() {
    let tsv = "\u{feff}arm\tsurvival\thosp\tqol\nTreatment\t500\t1\t12.0\nTreatment\t400\t1\t9.0\nControl\t500\t1\t4.0\nControl\t300\t2\t1.0";
    let cohort = parse_win_ratio_cohort(tsv).expect("BOM TSV should parse");
    assert_eq!(cohort.tier_count, 3);
    let specs = vec![
        TierSpec::new("Survival", TierComparatorKind::HigherIsBetter, 0.0),
        TierSpec::new("Hosp", TierComparatorKind::LowerIsBetter, 0.0),
        TierSpec::new("QoL", TierComparatorKind::Threshold, 6.0),
    ];
    let eval = evaluate_cohort(&cohort, &specs, PairingMode::Matched).expect("matched 2x2");
    assert_eq!(eval.tiers[0].ties, 1);
    assert_eq!(eval.tiers[2].wins, 1);
    let csv_out = format_waterfall_csv(&eval, &specs);
    assert!(csv_out.contains("Threshold Margin"));
    let json_out = format_waterfall_json(&eval, &specs);
    assert!(json_out.contains("\"total_wins\""));
}

#[test]
fn test_rejects_malformed_rows_non_finite_values_and_unequal_matched_pairs() {
    assert!(parse_win_ratio_cohort("Treatment,NaN\nControl,10.0").is_err());
    assert!(parse_win_ratio_cohort("Treatment,inf\nControl,10.0").is_err());
    assert!(parse_win_ratio_cohort("UnknownArm,12.0\nControl,10.0").is_err());
    assert!(parse_win_ratio_cohort("Treatment,12.0,1.0\nControl,10.0").is_err());

    let unequal = parse_win_ratio_cohort("Treatment,100\nTreatment,90\nControl,80").unwrap();
    let specs = vec![TierSpec::new("T1", TierComparatorKind::HigherIsBetter, 0.0)];
    let err = evaluate_cohort(&unequal, &specs, PairingMode::Matched).unwrap_err();
    assert!(err.contains("equal length"));
}

#[test]
fn test_enforces_max_iterations_and_url_preset_bounds() {
    let cap = math_commons::registry::MAX_ITERATIONS;
    let mut rows = String::with_capacity((cap + 2) * 14);
    rows.push_str("Control,10.0\n");
    for i in 0..cap {
        use std::fmt::Write;
        let _ = writeln!(&mut rows, "Treatment,{}.0", i + 1);
    }
    let err = parse_win_ratio_cohort(&rows).unwrap_err();
    assert!(err.contains(&cap.to_string()));

    let mut tool = WinRatioWorkbenchTool::default();
    tool.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset valid");
    let exported = tool.export_preset_json().expect("preset export succeeds");
    validate_url_preset_json(&exported).expect("default preset fits URL_JSON_BOUNDS");

    let oversized = "x".repeat(ToolStateBounds::DEFAULT.max_bytes + 16);
    assert!(tool.try_load_state(&oversized).is_err());
}

#[test]
fn test_weibull_simulator_view_and_dataset_export() {
    let mut tool = WinRatioWorkbenchTool {
        active_view: WinRatioView::WeibullSimulator,
        ..Default::default()
    };
    tool.recalculate();
    let sim = tool.weibull_eval.as_ref().expect("weibull evaluated");
    assert!(sim.uncensored_wr > 1.0);
    assert!(sim.prob_wr.is_finite() && sim.prob_wr > 1.0);
    let ds = tool.export_datasets();
    assert_eq!(ds.len(), 3);
    assert!(ds.iter().all(|(_, pts)| !pts.is_empty()));
}
