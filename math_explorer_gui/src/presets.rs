//! Presets module providing shared Serde JSON envelope helpers, dialog state,
//! and configuration models for Math Explorer simulation tools.

use federated_registry::Severity;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Receiver, TryRecvError};

// theory_verification!

/// Metadata included in exported preset JSON files.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct PresetWrapper<T> {
    /// Metadata header.
    pub metadata: PresetMetadata,
    /// Tool-specific configuration struct.
    pub config: T,
}

/// ODE System preset type enum.
#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
pub enum OdePreset {
    /// Exponential growth/decay model.
    Exponential,
    /// Harmonic oscillator spring model.
    HarmonicOscillator,
    /// Logistic population growth model.
    LogisticGrowth,
    /// Coupled 2x2 linear system x' = A * x with Poincaré trace-determinant classification.
    LinearSystem2D,
    /// Coupled 2D planar custom dynamic equations x' = f(t, x, y), y' = g(t, x, y).
    PlanarCustom,
    /// Custom dynamic scalar user equation y' = f(t, y).
    Custom,
}

impl OdePreset {
    /// All selectable ODE presets in sidebar order.
    pub const ALL: [Self; 6] = [
        Self::Exponential,
        Self::HarmonicOscillator,
        Self::LogisticGrowth,
        Self::LinearSystem2D,
        Self::PlanarCustom,
        Self::Custom,
    ];

    /// User-friendly display name of the ODE preset.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            OdePreset::Exponential => "y' = k*y (Exponential)",
            OdePreset::HarmonicOscillator => "y'' = -k*y (Harmonic Oscillator)",
            OdePreset::LogisticGrowth => "y' = r*y*(1 - y/K) (Logistic Growth)",
            OdePreset::LinearSystem2D => "x' = A*x (2x2 Linear / Poincaré)",
            OdePreset::PlanarCustom => "x' = f(t,x,y), y' = g(t,x,y) (2D Planar)",
            OdePreset::Custom => "y' = f(t, y) (Custom Equation)",
        }
    }

    /// Returns `true` when this preset evolves a 2D state vector and supports phase-plane portraits.
    #[must_use]
    pub fn is_2d(self) -> bool {
        matches!(
            self,
            Self::HarmonicOscillator | Self::LinearSystem2D | Self::PlanarCustom
        )
    }

    /// Returns `(first_time_label, second_time_label, phase_x_label, phase_y_label)`.
    #[must_use]
    pub fn state_labels(self) -> (&'static str, &'static str, &'static str, &'static str) {
        match self {
            Self::HarmonicOscillator => ("y(t)", "y'(t)", "y", "y'"),
            Self::LinearSystem2D | Self::PlanarCustom => ("x(t)", "y(t)", "x", "y"),
            Self::Exponential | Self::LogisticGrowth | Self::Custom => ("y(t)", "y'(t)", "y", "y'"),
        }
    }
}

/// Maximum allowed integration step count for ODE solvers.
pub const MAX_ODE_STEPS: usize = 200_000;

/// Computes a bounded ODE integration step count when `dt` and `total_time` are finite,
/// strictly positive, and require at most [`MAX_ODE_STEPS`] steps.
#[must_use]
pub fn checked_ode_step_count(dt: f64, total_time: f64) -> Option<usize> {
    if !dt.is_finite() || dt <= 0.0 || !total_time.is_finite() || total_time <= 0.0 {
        return None;
    }
    let steps = (total_time / dt).ceil();
    if !steps.is_finite() || !(0.0..=MAX_ODE_STEPS as f64).contains(&steps) {
        return None;
    }
    Some(steps as usize)
}

/// Configuration struct for Turing Morphogenesis simulation parameters.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

/// Packs `config` into a `"1.0"` [`PresetWrapper`] metadata envelope and serializes it as pretty JSON.
pub fn serialize_preset_envelope<T: Serialize>(
    title: &str,
    domain: &str,
    config: T,
) -> Result<String, String> {
    let envelope = PresetWrapper {
        metadata: PresetMetadata {
            schema_version: "1.0".to_string(),
            title: title.to_string(),
            domain: domain.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        },
        config,
    };
    serde_json::to_string_pretty(&envelope).map_err(|e| format!("Serialization error: {e}"))
}

/// Packs `config` into a deterministic, compact `"1.0"` [`PresetWrapper`] JSON envelope for URL state sharing.
pub fn serialize_compact_preset_envelope<T: Serialize>(
    title: &str,
    domain: &str,
    config: T,
) -> Result<String, String> {
    let envelope = PresetWrapper {
        metadata: PresetMetadata {
            schema_version: "1.0".to_string(),
            title: title.to_string(),
            domain: domain.to_string(),
            timestamp: "1970-01-01T00:00:00Z".to_string(),
        },
        config,
    };
    serde_json::to_string(&envelope).map_err(|e| format!("Serialization error: {e}"))
}

/// Strips any UTF-8 BOM, enforces `bounds.for_preset_envelope()`, parses a [`PresetWrapper`]
/// JSON envelope, and validates `metadata.schema_version` and `metadata.domain`.
pub fn deserialize_preset_envelope_bounded<T: for<'de> Deserialize<'de>>(
    json_str: &str,
    expected_domain: &str,
    target_name: &str,
    bounds: crate::framework::ToolStateBounds,
) -> Result<T, String> {
    let cleaned = egui_plot::export::strip_utf8_bom(json_str);
    bounds.for_preset_envelope().validate_json(cleaned)?;
    let envelope: PresetWrapper<T> =
        serde_json::from_str(cleaned).map_err(|e| format!("Invalid JSON preset schema: {e}"))?;
    if envelope.metadata.schema_version != "1.0" {
        return Err(format!(
            "Unsupported preset schema version '{}': expected '1.0'",
            envelope.metadata.schema_version
        ));
    }
    if expected_domain.is_empty() {
        return Err(format!("Preset import not supported for {target_name}"));
    }
    if envelope.metadata.domain != expected_domain {
        return Err(format!(
            "Domain mismatch: preset is for domain '{}', but active tool is '{}'",
            envelope.metadata.domain, expected_domain
        ));
    }
    Ok(envelope.config)
}

/// Strips any UTF-8 BOM, enforces default preset envelope bounds, parses a [`PresetWrapper`]
/// JSON envelope, and validates `metadata.domain`.
pub fn deserialize_preset_envelope<T: for<'de> Deserialize<'de>>(
    json_str: &str,
    expected_domain: &str,
    target_name: &str,
) -> Result<T, String> {
    deserialize_preset_envelope_bounded(
        json_str,
        expected_domain,
        target_name,
        crate::framework::ToolStateBounds::DEFAULT,
    )
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
        crate::web_io::spawn_open_file_dialog(
            tx,
            crate::web_io::OpenDialogConfig {
                filter_name: "JSON Preset",
                extensions: &["json"],
                accept: ".json,application/json",
                read_error_prefix: "Failed to read file",
            },
            |path| {
                crate::web_io::sanitize_download_filename(&path.to_string_lossy(), "preset.json")
            },
        );
    }

    /// Spawns file save dialog or triggers browser download to export preset JSON string.
    pub fn trigger_export(&mut self, json_content: String, default_filename: &str) {
        let (tx, rx) = channel();
        self.export_rx = Some(rx);
        crate::web_io::spawn_save_as_file_dialog(
            tx,
            crate::web_io::SaveAsDialogConfig {
                filter_name: "JSON Preset",
                extensions: &["json"],
                fallback_filename: "preset.json",
                write_error_prefix: "Failed to write file",
            },
            default_filename,
            json_content,
            |path, default_name| {
                crate::web_io::sanitize_download_filename(&path.to_string_lossy(), default_name)
            },
        );
    }

    pub(crate) fn record_toast_info(&mut self, msg: String) {
        self.toast_error = None;
        crate::accessibility::announce_status(&msg);
        self.toast_info = Some(msg);
    }

    /// Records a preset deserialization or import error into toast state, telemetry, and accessibility status.
    pub(crate) fn record_import_error(&mut self, err: String) {
        self.toast_info = None;
        crate::telemetry::emit_telemetry(
            "PresetSerialization",
            Severity::Warning,
            err.clone(),
            std::collections::HashMap::new(),
        );
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

    /// Triggers a preset file export from serialized JSON or records any serialization error.
    pub fn export_preset_result(&mut self, result: Result<String, String>, file_prefix: &str) {
        match result {
            Ok(json_str) => {
                let filename = format!("{file_prefix}_preset.json");
                self.trigger_export(json_str, &filename);
            }
            Err(err) => self.record_import_error(err),
        }
    }

    /// Polls pending export and import channels, applying imported preset JSON with `apply_json` if ready.
    pub fn poll_pending_import<F>(&mut self, apply_json: F) -> bool
    where
        F: FnOnce(&str) -> Result<(), String>,
    {
        self.poll_export();
        let mut loaded = false;
        if let Some(rx) = &self.import_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.import_rx = None;
                    match res.and_then(|(name, json_str)| apply_json(&json_str).map(|()| name)) {
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
pub fn render_preset_buttons<T, E, I>(
    dialog_state: &mut PresetDialogState,
    ui: &mut eframe::egui::Ui,
    target: &mut T,
    file_prefix: &str,
    export_json: E,
    import_json: I,
) -> bool
where
    E: FnOnce(&T) -> Result<String, String>,
    I: FnOnce(&mut T, &str) -> Result<(), String>,
{
    ui.separator();
    ui.heading("Preset Import / Export");
    let mut changed = false;

    ui.horizontal(|ui| {
        if ui.button("📥 Import Preset JSON").clicked() {
            dialog_state.trigger_import();
        }
        if ui.button("📤 Export Preset JSON").clicked() {
            dialog_state.export_preset_result(export_json(target), file_prefix);
        }
    });

    if dialog_state.poll_pending_import(|json_str| import_json(target, json_str)) {
        changed = true;
    }

    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    if let Some(ref msg) = dialog_state.toast_info {
        ui.add_space(4.0);
        ui.colored_label(palette.success, format!("ℹ️ {}", msg));
        if ui.small_button("Dismiss").clicked() {
            dialog_state.toast_info = None;
        }
    }

    if let Some(ref err) = dialog_state.toast_error {
        ui.add_space(4.0);
        ui.colored_label(palette.error, format!("⚠️ {}", err));
        if ui.small_button("Dismiss Warning").clicked() {
            dialog_state.toast_error = None;
        }
    }

    changed
}

#[cfg(test)]
#[path = "presets/tests.rs"]
mod tests;
