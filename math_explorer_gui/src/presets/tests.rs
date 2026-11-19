use super::*;

struct DummyTool {
    pub value: f64,
    pub name: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
struct DummyConfig {
    pub value: f64,
    pub name: String,
}

impl InteractivePreset for DummyTool {
    type Config = DummyConfig;

    fn domain(&self) -> &'static str {
        "test_domain"
    }

    fn title(&self) -> String {
        "Dummy Tool".to_string()
    }

    fn export_config(&self) -> Self::Config {
        DummyConfig {
            value: self.value,
            name: self.name.clone(),
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        self.value = config.value;
        self.name = config.name;
        Ok(())
    }
}

#[test]
fn test_roundtrip_dummy_preset() {
    let mut tool = DummyTool {
        value: 42.5,
        name: "Test Simulation".to_string(),
    };

    let json = tool.to_json_string().unwrap();
    assert!(json.contains("test_domain"));
    assert!(json.contains("42.5"));

    tool.value = 0.0;
    tool.name = "Cleared".to_string();

    tool.from_json_str(&json).unwrap();
    assert_eq!(tool.value, 42.5);
    assert_eq!(tool.name, "Test Simulation");
}

#[test]
fn test_corrupted_json_graceful_failure() {
    let mut tool = DummyTool {
        value: 10.0,
        name: "Original".to_string(),
    };

    let result = tool.from_json_str("INVALID JSON CONTENT {");
    assert!(result.is_err());
    // Value should remain unaltered
    assert_eq!(tool.value, 10.0);
    assert_eq!(tool.name, "Original");
}

#[test]
fn test_domain_mismatch_failure() {
    let mut tool = DummyTool {
        value: 10.0,
        name: "Original".to_string(),
    };

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

    let result = tool.from_json_str(wrong_domain_json);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Domain mismatch"));
    assert_eq!(tool.value, 10.0);
}

#[test]
fn test_replicator_matrix_validation() {
    let valid = ReplicatorPresetConfig {
        payoff_matrix: vec![vec![-1.0, 2.0], vec![0.0, 1.0]],
        initial_population: vec![0.5, 0.5],
        time_horizon: 20.0,
        dt: 0.05,
        strategy_names: vec!["A".to_string(), "B".to_string()],
    };
    assert!(valid.validate().is_ok());

    let non_square = ReplicatorPresetConfig {
        payoff_matrix: vec![vec![-1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0]],
        initial_population: vec![0.5, 0.5],
        time_horizon: 20.0,
        dt: 0.05,
        strategy_names: vec!["A".to_string(), "B".to_string()],
    };
    assert!(non_square.validate().is_err());

    let strategy_mismatch = ReplicatorPresetConfig {
        payoff_matrix: vec![vec![-1.0, 2.0], vec![0.0, 1.0]],
        initial_population: vec![0.5, 0.5],
        time_horizon: 20.0,
        dt: 0.05,
        strategy_names: vec!["A".to_string()],
    };
    assert!(strategy_mismatch.validate().is_err());
}

#[test]
fn test_ode_preset_config_roundtrip() {
    let config = OdePresetConfig {
        preset: OdePreset::HarmonicOscillator,
        dt: 0.01,
        total_time: 15.0,
        param_k: 2.5,
        param_r: 1.0,
        param_cap_k: 10.0,
        ic_y0: 1.5,
        ic_v0: -0.5,
        custom_expr: String::new(),
    };

    let wrapper = PresetWrapper {
        metadata: PresetMetadata {
            schema_version: "1.0".to_string(),
            title: "Harmonic Test".to_string(),
            domain: "analysis".to_string(),
            timestamp: "2026-10-01T00:00:00Z".to_string(),
        },
        config: config.clone(),
    };

    let json = serde_json::to_string_pretty(&wrapper).unwrap();
    let deserialized: PresetWrapper<OdePresetConfig> = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.config, config);
    assert_eq!(deserialized.metadata.domain, "analysis");
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

    let wrapper = PresetWrapper {
        metadata: PresetMetadata {
            schema_version: "1.0".to_string(),
            title: "Stripes Test".to_string(),
            domain: "biology".to_string(),
            timestamp: "2026-10-01T00:00:00Z".to_string(),
        },
        config: config.clone(),
    };

    let json = serde_json::to_string_pretty(&wrapper).unwrap();
    let deserialized: PresetWrapper<MorphogenesisPresetConfig> =
        serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.config, config);
    assert_eq!(deserialized.metadata.domain, "biology");
}

#[test]
fn test_preset_dialog_poll_import_export_and_cancel_cleanup() {
    let mut tool = DummyTool {
        value: 1.0,
        name: "Initial".to_string(),
    };
    let exported = DummyTool {
        value: 77.25,
        name: "Browser Imported".to_string(),
    }
    .to_json_string()
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

    assert!(dialog.poll_import(&mut tool));
    assert_eq!(tool.value, 77.25);
    assert_eq!(tool.name, "Browser Imported");
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

    assert!(!dialog.poll_import(&mut tool));
    assert!(dialog.import_rx.is_none());
    assert!(dialog.toast_error.is_none());
}

#[test]
fn test_preset_dialog_poll_import_tool_and_read_error() {
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

    assert!(dialog.poll_import_tool(tool.as_mut()));
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
    assert!(!dialog.poll_import_tool(tool.as_mut()));
    assert_eq!(
        dialog.toast_error.as_deref(),
        Some("Browser failed to read selected file")
    );
    assert!(dialog.toast_info.is_none());

    // Simulate cancel on poll_import_tool
    let (cancel_tx, cancel_rx) = channel::<Result<(String, String), String>>();
    dialog.import_rx = Some(cancel_rx);
    drop(cancel_tx);
    assert!(!dialog.poll_import_tool(tool.as_mut()));
    assert!(dialog.import_rx.is_none());
}

#[test]
fn test_preset_dialog_invalid_json_and_export_error_toasts() {
    let mut tool = DummyTool {
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
    assert!(!dialog.poll_import(&mut tool));
    assert_eq!(tool.value, 12.5);
    assert_eq!(tool.name, "Baseline");
    assert!(dialog
        .toast_error
        .as_deref()
        .is_some_and(|m| m.contains("Invalid JSON preset schema")));
    assert!(dialog.toast_info.is_none());

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
