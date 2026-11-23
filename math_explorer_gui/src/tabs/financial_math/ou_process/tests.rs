#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{OuProcessTool, STARTER_PRESET_JSON};
use crate::framework::InteractiveTool;

#[test]
fn test_ou_process_bom_csv_calibration_and_rejections() {
    let mut tool = OuProcessTool::default();
    assert!(tool.export_capabilities().figure && tool.export_capabilities().dataset);
    assert!(!tool.mean_points.is_empty());
    assert_eq!(tool.p05_points.len(), tool.p95_points.len());

    // UTF-8 BOM + header 2-column CSV calibration
    let bom_csv =
        "\u{feff}time,metric\n0.0,0.72\n0.5,0.64\n1.0,0.58\n1.5,0.55\n2.0,0.51\n2.5,0.49\n";
    let count = tool.import_csv_series(bom_csv).expect("BOM CSV imports");
    assert_eq!(count, 6);
    assert!((tool.sample_dt - 0.5).abs() < 1e-9);
    assert!(tool.calc_error.is_none());
    assert!(tool.theta > 0.0 && tool.sigma >= 0.0);

    // Reject too-short and malformed CSV inputs cleanly
    assert!(tool.import_csv_series("0.5\n0.6\n").is_err());
    assert!(tool
        .import_csv_series("t,x\n0.0,0.5\n1.0,not_a_float\n2.0,0.4\n")
        .is_err());
    assert!(tool.import_csv_series("0.5\nNaN\n0.4\n").is_err());
}

#[test]
fn test_ou_process_snippets_exports_and_url_warning() {
    let mut tool = OuProcessTool::default();
    assert!(tool.rust_snippet().contains("EulerMaruyama::new"));
    assert!(tool.python_snippet().contains("np.sqrt(dt)"));
    assert!(tool.latex_snippet().contains("dW_t"));

    let csv = tool.export_bands_csv();
    assert!(csv.starts_with("Step,Time,Mean,P05,P25,P75,P95\n"));
    let json = tool.export_bands_json().expect("valid JSON");
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed["bands"].as_array().map(Vec::len),
        Some(tool.n_steps + 1)
    );

    tool.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset loads");
    assert!(tool.url_share_warning().is_none());

    // Large imported CSV triggers URL-share-bound warning
    let large_csv: String = (0..1_200)
        .map(|i| format!("{:.4}", 0.5 + 0.1 * ((i as f64) * 0.17).sin()))
        .collect::<Vec<_>>()
        .join("\n");
    tool.import_csv_series(&large_csv)
        .expect("large CSV imports");
    assert!(tool.url_share_warning().is_some());
}

// [cite:clinical_trials_statistics]
