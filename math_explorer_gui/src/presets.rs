//! Presets module providing uniform Serde JSON export and import traits
//! and configuration models for Math Explorer simulation tools.

use crate::framework::InteractiveTool;
use federated_registry::{global_registry, Severity, TelemetryEvent};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Receiver, TryRecvError};

// theory_verification!

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
    /// Custom dynamic user equation.
    Custom,
}

impl OdePreset {
    /// User-friendly display name of the ODE preset.
    pub fn name(&self) -> &'static str {
        match self {
            OdePreset::Exponential => "y' = k*y (Exponential)",
            OdePreset::HarmonicOscillator => "y'' = -k*y (Harmonic Oscillator)",
            OdePreset::LogisticGrowth => "y' = r*y*(1 - y/K) (Logistic Growth)",
            OdePreset::Custom => "y' = f(t, y) (Custom Equation)",
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
    /// Custom user equation expression.
    #[serde(default)]
    pub custom_expr: String,
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
    #[allow(clippy::wrong_self_convention)]
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
    /// Pending channel receiver for imported file `(filename, json_content)`.
    import_rx: Option<Receiver<Result<(String, String), String>>>,
    /// Pending channel receiver for exported preset `filename`.
    export_rx: Option<Receiver<Result<String, String>>>,
    /// Inline toast error message to show user on file import/export failure.
    pub toast_error: Option<String>,
    /// Inline toast info message to show user on file import/export success.
    pub toast_info: Option<String>,
}

impl PresetDialogState {
    /// Spawns background thread or browser file picker dialog and returns imported JSON content over channel.
    pub fn trigger_import(&mut self) {
        let (tx, rx) = channel();
        self.import_rx = Some(rx);

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("JSON Preset", &["json"])
                    .pick_file()
                {
                    let name = crate::web_io::sanitize_download_filename(
                        &path.to_string_lossy(),
                        "preset.json",
                    );
                    match std::fs::read_to_string(path) {
                        Ok(content) => {
                            let _ = tx.send(Ok((name, content)));
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Failed to read file: {}", e)));
                        }
                    }
                }
            });
        }

        #[cfg(target_arch = "wasm32")]
        {
            crate::web_io::open_browser_file_picker(".json,application/json", move |outcome| {
                if let Some(res) = outcome {
                    let _ = tx.send(res);
                }
            });
        }
    }

    /// Spawns file save dialog or triggers browser download to export preset JSON string.
    pub fn trigger_export(&mut self, json_content: String, default_filename: &str) {
        let filename = crate::web_io::sanitize_download_filename(default_filename, "preset.json");
        let (tx, rx) = channel();
        self.export_rx = Some(rx);

        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                if let Some(path) = rfd::FileDialog::new()
                    .set_file_name(&filename)
                    .add_filter("JSON Preset", &["json"])
                    .save_file()
                {
                    let saved_name = crate::web_io::sanitize_download_filename(
                        &path.to_string_lossy(),
                        &filename,
                    );
                    match std::fs::write(path, json_content) {
                        Ok(()) => {
                            let _ = tx.send(Ok(saved_name));
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Failed to write file: {}", e)));
                        }
                    }
                }
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            let res = crate::web_io::download_browser_file(&filename, json_content.as_bytes())
                .map(|()| filename);
            let _ = tx.send(res);
            crate::web_io::request_browser_repaint();
        }
    }

    fn record_toast_info(&mut self, msg: String) {
        self.toast_error = None;
        crate::accessibility::announce_status(&msg);
        self.toast_info = Some(msg);
    }

    fn record_import_error(&mut self, err: String) {
        self.toast_info = None;
        global_registry().emit(TelemetryEvent {
            source: "PresetSerialization".to_string(),
            severity: Severity::Warning,
            message: err.clone(),
            metadata: std::collections::HashMap::new(),
            thread_name: None,
        });
        crate::accessibility::announce_status_with_priority(&err, "assertive");
        self.toast_error = Some(err);
    }

    /// Polls pending export channel and updates toast state when an export completes.
    pub fn poll_export(&mut self) {
        if let Some(rx) = &self.export_rx {
            match rx.try_recv() {
                Ok(Ok(name)) => {
                    self.export_rx = None;
                    self.record_toast_info(format!("Exported preset to {name}"));
                }
                Ok(Err(err)) => {
                    self.export_rx = None;
                    self.record_import_error(err);
                }
                Err(TryRecvError::Disconnected) => {
                    self.export_rx = None;
                }
                Err(TryRecvError::Empty) => {}
            }
        }
    }

    /// Polls pending import channel and applies imported preset JSON to the target tool if ready.
    pub fn poll_import<T: InteractivePreset>(&mut self, tool: &mut T) -> bool {
        self.poll_export();
        let mut loaded = false;
        if let Some(rx) = &self.import_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.import_rx = None;
                    match res
                        .and_then(|(name, json_str)| tool.from_json_str(&json_str).map(|()| name))
                    {
                        Ok(name) => {
                            self.record_toast_info(format!("Loaded preset from {name}"));
                            loaded = true;
                        }
                        Err(err) => self.record_import_error(err),
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    self.import_rx = None;
                }
                Err(TryRecvError::Empty) => {}
            }
        }
        loaded
    }

    /// Polls pending import channel and applies imported preset JSON to an InteractiveTool if ready.
    pub fn poll_import_tool(&mut self, tool: &mut dyn InteractiveTool) -> bool {
        self.poll_export();
        let mut loaded = false;
        if let Some(rx) = &self.import_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.import_rx = None;
                    match res.and_then(|(name, json_str)| {
                        tool.import_preset_json(&json_str).map(|()| name)
                    }) {
                        Ok(name) => {
                            self.record_toast_info(format!("Loaded preset from {name}"));
                            loaded = true;
                        }
                        Err(err) => self.record_import_error(err),
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    self.import_rx = None;
                }
                Err(TryRecvError::Empty) => {}
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

    if let Some(ref msg) = dialog_state.toast_info {
        ui.add_space(4.0);
        ui.colored_label(eframe::egui::Color32::LIGHT_GREEN, format!("ℹ️ {}", msg));
        if ui.small_button("Dismiss").clicked() {
            dialog_state.toast_info = None;
        }
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
#[path = "presets/tests.rs"]
mod tests;
