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
    assert_eq!(tool.observations.len(), 3);
    assert_eq!(tool.control_observations.len(), 3);
    assert!(!tool.control_curve.is_empty());
    let hr = tool.hazard_ratio.as_ref().unwrap().as_ref().unwrap();
    assert!(*hr < 1.0 && hr.is_finite());
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
