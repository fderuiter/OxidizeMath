use super::*;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
struct DummyConfig {
    pub value: f64,
    pub name: String,
}

#[test]
fn test_roundtrip_preset_envelope_and_bom_stripping() {
    let config = DummyConfig {
        value: 42.5,
        name: "Test Simulation".to_string(),
    };

    let json = serialize_preset_envelope("Dummy Tool", "test_domain", &config).unwrap();
    assert!(json.contains("test_domain"));
    assert!(json.contains("42.5"));

    let restored: DummyConfig =
        deserialize_preset_envelope(&json, "test_domain", "Dummy Tool").unwrap();
    assert_eq!(restored, config);

    let bom_json = format!("\u{feff}{json}");
    let bom_restored: DummyConfig =
        deserialize_preset_envelope(&bom_json, "test_domain", "Dummy Tool").unwrap();
    assert_eq!(bom_restored, config);
}

#[test]
fn test_corrupted_json_and_empty_domain_graceful_failure() {
    let bad_schema = deserialize_preset_envelope::<DummyConfig>(
        "INVALID JSON CONTENT {",
        "test_domain",
        "Dummy Tool",
    );
    assert!(bad_schema
        .unwrap_err()
        .contains("Invalid JSON preset schema"));

    let valid_json = serialize_preset_envelope(
        "Dummy Tool",
        "test_domain",
        DummyConfig {
            value: 10.0,
            name: "Original".to_string(),
        },
    )
    .unwrap();
    let unsupported =
        deserialize_preset_envelope::<DummyConfig>(&valid_json, "", "Unsupported Tool");
    assert_eq!(
        unsupported.unwrap_err(),
        "Preset import not supported for Unsupported Tool"
    );
}

#[test]
fn test_domain_mismatch_failure() {
    let wrong_domain_json = r#"{
        "metadata": {
            "schema_version": "1.0",
            "title": "Wrong Domain",
            "domain": "other_domain",
            "timestamp": "2026-10-01T00:00:00Z"
        },
        "config": {
            "value": 99.0,
            "name": "Changed"
        }
    }"#;

    let result =
        deserialize_preset_envelope::<DummyConfig>(wrong_domain_json, "test_domain", "Dummy Tool");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Domain mismatch"));
}

#[test]
fn test_replicator_matrix_validation() {
    let valid = ReplicatorPresetConfig {
        payoff_matrix: vec![vec![-1.0, 2.0], vec![0.0, 1.0]],
        initial_population: vec![0.5, 0.5],
        time_horizon: 20.0,
        dt: 0.05,
        strategy_names: vec!["A".to_string(), "B".to_string()],
        symbolic_payoffs: None,
        symbolic_params: std::collections::BTreeMap::new(),
    };
    assert!(valid.validate().is_ok());

    let non_square = ReplicatorPresetConfig {
        payoff_matrix: vec![vec![-1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0]],
        initial_population: vec![0.5, 0.5],
        time_horizon: 20.0,
        dt: 0.05,
        strategy_names: vec!["A".to_string(), "B".to_string()],
        symbolic_payoffs: None,
        symbolic_params: std::collections::BTreeMap::new(),
    };
    assert!(non_square.validate().is_err());

    let strategy_mismatch = ReplicatorPresetConfig {
        payoff_matrix: vec![vec![-1.0, 2.0], vec![0.0, 1.0]],
        initial_population: vec![0.5, 0.5],
        time_horizon: 20.0,
        dt: 0.05,
        strategy_names: vec!["A".to_string()],
        symbolic_payoffs: None,
        symbolic_params: std::collections::BTreeMap::new(),
    };
    assert!(strategy_mismatch.validate().is_err());

    let mut params = std::collections::BTreeMap::new();
    params.insert("v".to_string(), 2.0);
    params.insert("c".to_string(), 4.0);
    let valid_symbolic = ReplicatorPresetConfig {
        payoff_matrix: vec![vec![-1.0, 2.0], vec![0.0, 1.0]],
        initial_population: vec![0.1, 0.9],
        time_horizon: 20.0,
        dt: 0.05,
        strategy_names: vec!["Hawk".to_string(), "Dove".to_string()],
        symbolic_payoffs: Some(vec![
            vec!["(v - c) / 2".to_string(), "v".to_string()],
            vec!["0".to_string(), "v / 2".to_string()],
        ]),
        symbolic_params: params,
    };
    assert!(valid_symbolic.validate().is_ok());
}

#[test]
fn test_checked_ode_step_count_bounds_and_step_caps() {
    assert_eq!(checked_ode_step_count(0.001, 100.0), Some(100_000));
    assert_eq!(checked_ode_step_count(0.0005, 100.0), Some(MAX_ODE_STEPS));
    assert_eq!(checked_ode_step_count(1e-300, 10.0), None);
    assert_eq!(checked_ode_step_count(0.0, 10.0), None);
    assert_eq!(checked_ode_step_count(-0.05, 10.0), None);
    assert_eq!(checked_ode_step_count(0.05, -10.0), None);
    assert_eq!(checked_ode_step_count(f64::NAN, 10.0), None);
    assert_eq!(checked_ode_step_count(0.05, f64::INFINITY), None);
}

#[test]
fn test_morphogenesis_preset_config_roundtrip() {
    let config = MorphogenesisPresetConfig {
        a: 0.14,
        b: 0.86,
        d_u: 1.0,
        d_v: 50.0,
        dt: 0.02,
        width: 80,
        height: 80,
    };

    let json = serialize_preset_envelope("Stripes Test", "biology", &config).unwrap();
    let deserialized: MorphogenesisPresetConfig =
        deserialize_preset_envelope(&json, "biology", "Morphogenesis").unwrap();
    assert_eq!(deserialized, config);
}

#[test]
fn test_preset_dialog_poll_pending_import_export_and_cancel_cleanup() {
    let mut state = DummyConfig {
        value: 1.0,
        name: "Initial".to_string(),
    };
    let exported = serialize_preset_envelope(
        "Dummy Tool",
        "test_domain",
        DummyConfig {
            value: 77.25,
            name: "Browser Imported".to_string(),
        },
    )
    .unwrap();

    let mut dialog = PresetDialogState {
        import_rx: None,
        export_rx: None,
        toast_error: Some("Previous warning".to_string()),
        toast_info: None,
    };

    // Simulate export completion over channel
    let (exp_tx, exp_rx) = channel();
    dialog.export_rx = Some(exp_rx);
    exp_tx
        .send(Ok("test_domain_preset.json".to_string()))
        .unwrap();
    dialog.poll_export();
    assert!(dialog.toast_error.is_none());
    assert_eq!(
        dialog.toast_info.as_deref(),
        Some("Exported preset to test_domain_preset.json")
    );

    let (tx, rx) = channel();
    dialog.import_rx = Some(rx);
    tx.send(Ok(("custom_preset.json".to_string(), exported)))
        .unwrap();

    assert!(dialog.poll_pending_import(|json_str| {
        state = deserialize_preset_envelope(json_str, "test_domain", "Dummy Tool")?;
        Ok(())
    }));
    assert_eq!(state.value, 77.25);
    assert_eq!(state.name, "Browser Imported");
    assert!(dialog.toast_error.is_none());
    assert_eq!(
        dialog.toast_info.as_deref(),
        Some("Loaded preset from custom_preset.json")
    );
    assert!(dialog.import_rx.is_none());

    // Simulate user canceling the browser file picker (sender dropped)
    let (cancel_tx, cancel_rx) = channel::<Result<(String, String), String>>();
    dialog.import_rx = Some(cancel_rx);
    drop(cancel_tx);

    assert!(!dialog.poll_pending_import(|_| Ok(())));
    assert!(dialog.import_rx.is_none());
    assert!(dialog.toast_error.is_none());
}

#[test]
fn test_preset_dialog_poll_pending_import_with_interactive_tool_and_read_error() {
    let mut framework = crate::framework::SimulationFramework::new("chaos");
    let tool = framework
        .active_tool
        .as_mut()
        .expect("chaos domain should have an active tool");

    let valid_json = tool.export_preset_json().unwrap();
    let mut dialog = PresetDialogState::default();
    let (tx, rx) = channel();
    dialog.import_rx = Some(rx);
    tx.send(Ok(("lorenz_preset.json".to_string(), valid_json)))
        .unwrap();

    assert!(dialog.poll_pending_import(|json_str| tool.import_preset_json(json_str)));
    assert!(dialog.toast_error.is_none());
    assert_eq!(
        dialog.toast_info.as_deref(),
        Some("Loaded preset from lorenz_preset.json")
    );
    assert!(dialog.import_rx.is_none());

    // Simulate FileReader error over channel
    let (err_tx, err_rx) = channel();
    dialog.import_rx = Some(err_rx);
    err_tx
        .send(Err("Browser failed to read selected file".to_string()))
        .unwrap();
    assert!(!dialog.poll_pending_import(|json_str| tool.import_preset_json(json_str)));
    assert_eq!(
        dialog.toast_error.as_deref(),
        Some("Browser failed to read selected file")
    );
    assert!(dialog.toast_info.is_none());

    // Simulate cancel on poll_pending_import
    let (cancel_tx, cancel_rx) = channel::<Result<(String, String), String>>();
    dialog.import_rx = Some(cancel_rx);
    drop(cancel_tx);
    assert!(!dialog.poll_pending_import(|json_str| tool.import_preset_json(json_str)));
    assert!(dialog.import_rx.is_none());
}

#[test]
fn test_preset_dialog_invalid_json_and_export_error_toasts() {
    let mut state = DummyConfig {
        value: 12.5,
        name: "Baseline".to_string(),
    };
    let mut dialog = PresetDialogState::default();

    // Invalid JSON over import channel preserves tool state and sets toast_error
    let (bad_tx, bad_rx) = channel();
    dialog.import_rx = Some(bad_rx);
    bad_tx
        .send(Ok(("corrupt.json".to_string(), "{ malformed".to_string())))
        .unwrap();
    assert!(!dialog.poll_pending_import(|json_str| {
        state = deserialize_preset_envelope(json_str, "test_domain", "Dummy Tool")?;
        Ok(())
    }));
    assert_eq!(state.value, 12.5);
    assert_eq!(state.name, "Baseline");
    assert!(dialog
        .toast_error
        .as_deref()
        .is_some_and(|m| m.contains("Invalid JSON preset schema")));
    assert!(dialog.toast_info.is_none());

    // Direct export serialization failure records error toast immediately
    dialog.export_preset_result(Err("Preset export failed".to_string()), "dummy");
    assert_eq!(dialog.toast_error.as_deref(), Some("Preset export failed"));

    // Export error over channel sets toast_error, and export cancel clears export_rx quietly
    let (exp_err_tx, exp_err_rx) = channel();
    dialog.export_rx = Some(exp_err_rx);
    exp_err_tx
        .send(Err("Browser window is unavailable".to_string()))
        .unwrap();
    dialog.poll_export();
    assert_eq!(
        dialog.toast_error.as_deref(),
        Some("Browser window is unavailable")
    );

    dialog.toast_error = None;
    let (exp_cancel_tx, exp_cancel_rx) = channel::<Result<String, String>>();
    dialog.export_rx = Some(exp_cancel_rx);
    drop(exp_cancel_tx);
    dialog.poll_export();
    assert!(dialog.export_rx.is_none());
    assert!(dialog.toast_error.is_none());
}

#[test]
fn test_poll_pending_import_keeps_receivers_while_channels_empty() {
    let mut dialog = PresetDialogState::default();
    let (_imp_tx, imp_rx) = channel::<Result<(String, String), String>>();
    let (_exp_tx, exp_rx) = channel::<Result<String, String>>();
    dialog.import_rx = Some(imp_rx);
    dialog.export_rx = Some(exp_rx);

    assert!(!dialog.poll_pending_import(|_| Ok(())));
    assert!(dialog.import_rx.is_some());
    assert!(dialog.export_rx.is_some());
    assert!(dialog.toast_info.is_none());
    assert!(dialog.toast_error.is_none());
}

#[test]
fn test_preset_filename_sanitization_matches_web_io_contract() {
    assert_eq!(
        crate::web_io::sanitize_download_filename("/nested/dir/stripes.json", "preset.json"),
        "stripes.json"
    );
    assert_eq!(
        crate::web_io::sanitize_download_filename("C:\\fakepath\\chaos_preset.json", "preset.json"),
        "chaos_preset.json"
    );
    assert_eq!(
        crate::web_io::sanitize_download_filename("..", "biology_preset.json"),
        "biology_preset.json"
    );
}

#[test]
fn test_morphogenesis_tab_shares_poll_pending_import_lifecycle() {
    let mut tab = crate::tabs::morphogenesis::MorphogenesisTab::default();
    let exported = tab.to_json_string().unwrap();

    let mut dialog = PresetDialogState::default();
    let (tx, rx) = channel();
    dialog.import_rx = Some(rx);
    tx.send(Ok(("morphogenesis_preset.json".to_string(), exported)))
        .unwrap();

    assert!(dialog.poll_pending_import(|json_str| tab.from_json_str(json_str)));
    assert_eq!(
        dialog.toast_info.as_deref(),
        Some("Loaded preset from morphogenesis_preset.json")
    );
    assert!(dialog.toast_error.is_none());
}
