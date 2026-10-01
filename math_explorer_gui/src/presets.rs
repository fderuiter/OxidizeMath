//! Presets module providing uniform Serde JSON export and import traits
//! and configuration models for Math Explorer simulation tools.

use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Receiver};

/// Metadata included in exported preset JSON files.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PresetMetadata {
    /// Schema version for compatibility checking.
    pub schema_version: String,
    /// Title of the preset configuration.
    pub title: String,
    /// Domain category of the preset (e.g. "analysis", "biology", "game_theory").
    pub domain: String,
    /// ISO 8601 UTC timestamp of export.
    pub timestamp: String,
}

/// Generic JSON envelope wrapping preset metadata and configuration parameters.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PresetWrapper<T> {
    /// Metadata header.
    pub metadata: PresetMetadata,
    /// Tool-specific configuration struct.
    pub config: T,
}

/// ODE System preset type enum.
#[derive(Serialize, Deserialize, PartialEq, Clone, Copy, Debug)]
pub enum OdePreset {
    /// Exponential growth/decay model.
    Exponential,
    /// Harmonic oscillator spring model.
    HarmonicOscillator,
    /// Logistic population growth model.
    LogisticGrowth,
}

impl OdePreset {
    /// User-friendly display name of the ODE preset.
    pub fn name(&self) -> &'static str {
        match self {
            OdePreset::Exponential => "y' = k*y (Exponential)",
            OdePreset::HarmonicOscillator => "y'' = -k*y (Harmonic Oscillator)",
            OdePreset::LogisticGrowth => "y' = r*y*(1 - y/K) (Logistic Growth)",
        }
    }
}

/// Configuration struct for ODE Solver simulation parameters.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OdePresetConfig {
    /// System equation type.
    pub preset: OdePreset,
    /// Integration time step size.
    pub dt: f64,
    /// Total simulation time horizon.
    pub total_time: f64,
    /// Parameter k (rate or spring constant).
    pub param_k: f64,
    /// Parameter r (growth rate).
    pub param_r: f64,
    /// Carrying capacity K.
    pub param_cap_k: f64,
    /// Initial condition y(0).
    pub ic_y0: f64,
    /// Initial condition y'(0).
    pub ic_v0: f64,
}

/// Configuration struct for Turing Morphogenesis simulation parameters.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MorphogenesisPresetConfig {
    /// Reaction kinetic parameter a.
    pub a: f64,
    /// Reaction kinetic parameter b.
    pub b: f64,
    /// Diffusion coefficient for activator (u).
    pub d_u: f64,
    /// Diffusion coefficient for inhibitor (v).
    pub d_v: f64,
    /// Simulation time step size.
    pub dt: f64,
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
}

/// Configuration struct for Replicator Dynamics evolutionary game parameters.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ReplicatorPresetConfig {
    /// Payoff matrix (rows x cols).
    pub payoff_matrix: Vec<Vec<f64>>,
    /// Initial population proportions per strategy.
    pub initial_population: Vec<f64>,
    /// Total simulation duration.
    pub time_horizon: f64,
    /// Integration time step size.
    pub dt: f64,
    /// Descriptive strategy names.
    pub strategy_names: Vec<String>,
}

impl ReplicatorPresetConfig {
    /// Validates matrix dimensions and strategy counts.
    pub fn validate(&self) -> Result<(), String> {
        let rows = self.payoff_matrix.len();
        if rows == 0 {
            return Err("Payoff matrix must not be empty".to_string());
        }
        for (i, row) in self.payoff_matrix.iter().enumerate() {
            if row.len() != rows {
                return Err(format!(
                    "Payoff matrix must be square (row {} has length {}, expected {})",
                    i,
                    row.len(),
                    rows
                ));
            }
        }
        if self.strategy_names.len() != rows {
            return Err(format!(
                "Strategy names count ({}) does not match payoff matrix dimensions ({}x{})",
                self.strategy_names.len(),
                rows,
                rows
            ));
        }
        if self.initial_population.len() != rows {
            return Err(format!(
                "Initial population count ({}) does not match strategy count ({})",
                self.initial_population.len(),
                rows
            ));
        }
        Ok(())
    }
}

/// Trait providing uniform JSON export and import capabilities for interactive simulation tools.
pub trait InteractivePreset: Sized {
    /// Serializable parameter configuration struct.
    type Config: Serialize + for<'de> Deserialize<'de>;

    /// Domain category string (e.g. "analysis", "biology", "game_theory").
    fn domain(&self) -> &'static str;

    /// Human-readable title of the current preset/simulation state.
    fn title(&self) -> String;

    /// Export active UI/simulation state into the configuration struct.
    fn export_config(&self) -> Self::Config;

    /// Apply imported configuration parameters to the simulation state.
    fn apply_config(&mut self, config: Self::Config) -> Result<(), String>;

    /// Serializes active simulation state to a formatted JSON string with metadata envelope.
    fn to_json_string(&self) -> Result<String, String> {
        let envelope = PresetWrapper {
            metadata: PresetMetadata {
                schema_version: "1.0".to_string(),
                title: self.title(),
                domain: self.domain().to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            },
            config: self.export_config(),
        };
        serde_json::to_string_pretty(&envelope).map_err(|e| format!("Serialization error: {}", e))
    }

    /// Parses and validates a JSON preset string, applying configuration if valid.
    fn from_json_str(&mut self, json_str: &str) -> Result<(), String> {
        let envelope: PresetWrapper<Self::Config> = serde_json::from_str(json_str)
            .map_err(|e| format!("Invalid JSON preset schema: {}", e))?;
        if envelope.metadata.domain != self.domain() {
            return Err(format!(
                "Domain mismatch: preset is for domain '{}', but active tool is '{}'",
                envelope.metadata.domain,
                self.domain()
            ));
        }
        self.apply_config(envelope.config)
    }
}

/// UI helper state for non-blocking file operations and user warnings/toasts.
#[derive(Default)]
pub struct PresetDialogState {
    /// Pending channel receiver for imported file JSON string content.
    import_rx: Option<Receiver<Result<String, String>>>,
    /// Inline toast error message to show user on file import failure.
    pub toast_error: Option<String>,
}

impl PresetDialogState {
    /// Spawns background thread for file pick dialog and returns imported JSON content over channel.
    pub fn trigger_import(&mut self) {
        let (tx, rx) = channel();
        self.import_rx = Some(rx);

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON Preset", &["json"])
                .pick_file()
            {
                match std::fs::read_to_string(path) {
                    Ok(content) => {
                        let _ = tx.send(Ok(content));
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("Failed to read file: {}", e)));
                    }
                }
            }
        });
    }

    /// Spawns background thread for file save dialog to write exported JSON string.
    pub fn trigger_export(&self, json_content: String, default_filename: &str) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let filename = default_filename.to_string();
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .set_file_name(&filename)
                    .add_filter("JSON Preset", &["json"])
                    .save_file()
                {
                    let _ = std::fs::write(path, json_content);
                }
            });
        }
    }

    /// Polls pending import channel and applies imported preset JSON to the target tool if ready.
    pub fn poll_import<T: InteractivePreset>(&mut self, tool: &mut T) -> bool {
        let mut loaded = false;
        if let Some(rx) = &self.import_rx {
            if let Ok(res) = rx.try_recv() {
                self.import_rx = None;
                match res {
                    Ok(json_str) => match tool.from_json_str(&json_str) {
                        Ok(()) => {
                            self.toast_error = None;
                            loaded = true;
                        }
                        Err(err) => {
                            self.toast_error = Some(err);
                        }
                    },
                    Err(err) => {
                        self.toast_error = Some(err);
                    }
                }
            }
        }
        loaded
    }
}

/// Renders Export/Import buttons and any active inline toast message in the egui UI.
pub fn render_preset_buttons<T: InteractivePreset>(
    dialog_state: &mut PresetDialogState,
    ui: &mut eframe::egui::Ui,
    tool: &mut T,
) -> bool {
    ui.separator();
    ui.heading("Preset Import / Export");
    let mut changed = false;

    ui.horizontal(|ui| {
        if ui.button("📥 Import Preset JSON").clicked() {
            dialog_state.trigger_import();
        }
        if ui.button("📤 Export Preset JSON").clicked() {
            if let Ok(json_str) = tool.to_json_string() {
                let filename = format!("{}_preset.json", tool.domain());
                dialog_state.trigger_export(json_str, &filename);
            }
        }
    });

    if dialog_state.poll_import(tool) {
        changed = true;
    }

    if let Some(ref err) = dialog_state.toast_error {
        ui.add_space(4.0);
        ui.colored_label(eframe::egui::Color32::RED, format!("⚠️ {}", err));
        if ui.small_button("Dismiss Warning").clicked() {
            dialog_state.toast_error = None;
        }
    }

    changed
}

#[cfg(test)]
mod tests {
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
}
