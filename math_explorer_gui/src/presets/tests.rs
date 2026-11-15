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
