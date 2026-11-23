#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::export::{
    format_markdown_summary, format_risk_set_csv, format_risk_table_csv, render_survival_svg,
};
use super::*;
use egui_plot::{HLine, Line};

#[test]
fn test_reject_infinite_and_non_finite_presets() {
    let inf_preset = r#"{"metadata":{"schema_version":"1.0","title":"KM","domain":"clinical_trials","timestamp":"2026-11-19T00:00:00Z"},"config":{"input_text":"inf, 1"}}"#;
    let mut tool = SurvivalAnalysisTool::default();
    let baseline_len = tool.observations.len();
    assert!(tool.import_preset_json(inf_preset).is_err());
    assert_eq!(tool.observations.len(), baseline_len);

    for invalid_input in ["-inf, 1", "Infinity, 0", "NaN, 1", "-1.0, 1"] {
        let payload = format!(
            r#"{{"metadata":{{"schema_version":"1.0","title":"KM","domain":"clinical_trials","timestamp":"2026-11-19T00:00:00Z"}},"config":{{"input_text":"{invalid_input}"}}}}"#
        );
        assert!(
            tool.import_preset_json(&payload).is_err(),
            "Expected error for preset input '{invalid_input}'"
        );
        assert_eq!(tool.observations.len(), baseline_len);
    }
}

#[test]
fn test_observation_cap_enforced_in_recalculate_and_load_state() {
    let max_obs = math_commons::registry::MAX_ITERATIONS;
    let at_cap_input: String = (0..max_obs)
        .map(|idx| format!("{}.0, 1", idx + 1))
        .collect::<Vec<_>>()
        .join("\n");

    let mut tool = SurvivalAnalysisTool::default();
    let state_ok = serde_json::to_string(&SurvivalState {
        input_text: at_cap_input,
        control_text: String::new(),
    })
    .unwrap();
    assert!(tool.try_load_state(&state_ok).is_ok());
    assert_eq!(tool.observations.len(), max_obs);
    assert!(tool.error_message.is_none());

    let over_cap_input: String = (0..=max_obs)
        .map(|idx| format!("{}.0, 1", idx + 1))
        .collect::<Vec<_>>()
        .join("\n");
    let state_over = serde_json::to_string(&SurvivalState {
        input_text: over_cap_input.clone(),
        control_text: String::new(),
    })
    .unwrap();
    let err = tool.try_load_state(&state_over).unwrap_err();
    assert!(err.contains(&max_obs.to_string()));
    assert_eq!(tool.observations.len(), max_obs);

    tool.input_text = over_cap_input;
    tool.recalculate();
    assert!(tool.error_message.is_some());
    assert!(tool.observations.is_empty());
    assert!(tool.curve.is_empty());
}

#[test]
fn test_starter_preset_preserves_survival_probabilities() {
    let mut tool = SurvivalAnalysisTool::default();
    tool.import_preset_json(STARTER_PRESET_JSON).unwrap();
    assert_eq!(tool.observations.len(), 10);
    assert_eq!(tool.curve.len(), 10);
    assert!((tool.curve[0].survival_probability - 0.9).abs() < 1e-12);
    assert!((tool.curve[1].survival_probability - 0.8).abs() < 1e-12);
    assert!((tool.curve[2].survival_probability - 0.8).abs() < 1e-12);
}

#[test]
fn test_two_arm_kaplan_meier_estimates_hazard_ratio() {
    let mut tool = SurvivalAnalysisTool::default();
    let state = serde_json::to_string(&SurvivalState {
        input_text: "10.0, 1\n12.0, 0\n15.0, 1".to_string(),
        control_text: "4.0, 1\n5.0, 1\n6.0, 1".to_string(),
    })
    .unwrap();
    tool.try_load_state(&state).unwrap();
    assert_eq!(tool.arms.len(), 2);
    assert_eq!(tool.arms[0].name, "Treatment");
    assert_eq!(tool.arms[1].name, "Control");
    assert_eq!(tool.arms[0].observations.len(), 3);
    assert_eq!(tool.arms[1].observations.len(), 3);
    assert!(!tool.arms[1].curve.is_empty());
    let hr = tool.hazard_ratio.as_ref().unwrap().as_ref().unwrap();
    assert!(*hr < 1.0 && hr.is_finite());

    let inline_prefix = serde_json::to_string(&SurvivalState {
        input_text: "Treatment, 10.0, 1\nTreatment, 14.0, 0\nControl, 4.0, 1\nControl, 5.0, 1"
            .to_string(),
        control_text: String::new(),
    })
    .unwrap();
    tool.try_load_state(&inline_prefix).unwrap();
    assert_eq!(tool.arms.len(), 2);
    assert_eq!(tool.observations.len(), 2);
    assert_eq!(tool.control_observations.len(), 2);
}

#[test]
fn test_two_arm_tsv_bom_log_rank_hazard_ratio_and_risk_table_exports() {
    let bom_tsv = format!("\u{feff}{}", TWO_ARM_SAMPLE_TEXT.replace(',', "\t"));
    let state = serde_json::to_string(&SurvivalState {
        input_text: bom_tsv,
        control_text: String::new(),
    })
    .unwrap();
    let mut tool = SurvivalAnalysisTool::default();
    tool.try_load_state(&state)
        .expect("two-arm TSV state valid");
    assert_eq!(tool.arms.len(), 2);
    let hr = tool.hazard_ratio.as_ref().unwrap().as_ref().unwrap();
    assert!(*hr < 1.0 && hr.is_finite());
    let lr = tool.log_rank.as_ref().unwrap().as_ref().unwrap();
    assert!(lr.statistic > 0.0 && (0.0..=1.0).contains(&lr.p_value));

    let csv = format_risk_table_csv(&tool.arms);
    assert!(csv.starts_with("arm,time,survival_probability,n_at_risk,n_events,n_censored\n"));
    assert!(csv.contains("Treatment,"));
    assert!(csv.contains("Control,"));
    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 8);
}

#[test]
fn test_median_survival_and_half_probability_step_crossing() {
    let mut tool = SurvivalAnalysisTool::default();
    tool.import_preset_json(STARTER_PRESET_JSON).unwrap();
    let median = tool
        .median_survival_time()
        .expect("median survival reached");
    assert!((median - 18.0).abs() < 1e-9);

    let step_pts = tool.build_step_points();
    let items: Vec<Box<dyn egui_plot::PlotItem>> = vec![
        Box::new(Line::new("S(t) (Kaplan-Meier)", step_pts)),
        Box::new(HLine::new("S(t) = 0.50", 0.50)),
    ];
    let tf = egui_plot::PlotTransform::new(
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(800.0, 600.0)),
        egui_plot::PlotBounds::from_min_max([0.0, 0.0], [35.0, 1.05]),
        true,
    );
    let pois = egui_plot::detect_critical_points(&items, &tf);
    let crossing = pois
        .iter()
        .find(|p| p.kind == egui_plot::CriticalPointKind::Intersection)
        .expect("S(t) = 0.50 step intersection");
    assert!((crossing.point[0] - median).abs() < 1e-9);
    assert!((crossing.point[1] - 0.50).abs() < 1e-9);
}

#[test]
fn test_multi_arm_csv_and_tsv_import_hazard_ratio_and_exports() {
    let mut tool = SurvivalAnalysisTool::default();
    let tsv = "arm\ttime\tevent\nControl\t4.0\t1\nControl\t6.0\t1\nControl\t9.0\t0\nTreatment\t8.0\t1\nTreatment\t14.0\t0\nTreatment\t20.0\t0";
    let (n_arms, n_obs) = tool.import_cohort_str(tsv).expect("valid multi-arm TSV");
    assert_eq!(n_arms, 2);
    assert_eq!(n_obs, 6);
    assert_eq!(tool.arms[0].name, "Treatment");
    assert_eq!(tool.arms[1].name, "Control");

    let hr = tool
        .hazard_ratio
        .as_ref()
        .expect("two-arm HR present")
        .as_ref()
        .copied()
        .expect("valid HR");
    // Treatment: 1 event / 42.0 time; Control: 2 events / 19.0 time -> HR = (1/42)/(2/19) < 1.0
    assert!(hr > 0.0 && hr < 1.0, "Expected protective HR < 1, got {hr}");

    let csv = format_risk_set_csv(&tool);
    assert!(csv.starts_with("arm,time,n_at_risk,n_events,n_censored,survival_probability\n"));
    assert!(csv.contains("Treatment,8.0000,3,1,0"));
    assert!(csv.contains("Control,4.0000,3,1,0"));

    let md = format_markdown_summary(&tool);
    assert!(md.contains("Incidence-Density Hazard Ratio (Treatment vs Control)"));
    assert!(md.contains("| Treatment |"));

    let svg = render_survival_svg(&tool).expect("SVG export succeeds");
    assert!(svg.contains("<svg") && svg.len() > 250);
    assert_eq!(tool.export_datasets().len(), 8);
}

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_wasm_survival_multi_arm_render_and_exports() {
        let mut tool = SurvivalAnalysisTool::default();
        tool.import_cohort_str(TWO_ARM_EXAMPLE_CSV)
            .expect("two-arm CSV import");
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
        assert!(render_survival_svg(&tool).unwrap().contains("<svg"));
        assert!(format_risk_set_csv(&tool).contains("Treatment"));
    }
}

// [cite:clinical_trials_statistics]
