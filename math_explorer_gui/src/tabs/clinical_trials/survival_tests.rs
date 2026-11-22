#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

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
fn test_two_arm_tsv_bom_log_rank_hazard_ratio_and_risk_table_exports() {
    let bom_tsv = format!("\u{feff}{}", TWO_ARM_SAMPLE_TEXT.replace(',', "\t"));
    let state = serde_json::to_string(&SurvivalState {
        input_text: bom_tsv,
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
    let median = tool.median_survival_time().expect("median survival reached");
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
