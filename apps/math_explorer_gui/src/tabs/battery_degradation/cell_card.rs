//! Shared cell chemistry choices, downloadable JSON cell cards, CSV sanitization,
//! and non-blocking `web_io` dialog wrappers for battery degradation tools.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:battery_degradation]

use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, OpenDialogConfig, SaveAsDialogConfig,
};
use math_explorer::engineering::battery_degradation::{
    Capacity, ChemistryPreset, DepthOfDischarge, ElectrochemicalAgingModel,
};
use std::sync::mpsc::{channel, Receiver, TryRecvError};

/// Serde-serializable mirror of [`ChemistryPreset`] for URL state, presets, and JSON cell cards.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ChemistryChoice {
    /// Nickel-rich NMC811 / graphite high-energy cell.
    #[default]
    Nmc811,
    /// Olivine LFP / graphite long-cycle-life cell.
    Lfp,
    /// Nickel-Cobalt-Aluminum / graphite-Si high-power cell.
    Nca,
    /// Lithium titanate spinel anode fast-charge cell.
    Lto,
}

impl ChemistryChoice {
    /// Ordered list of all chemistry choices for UI combo boxes.
    pub const ALL: [Self; 4] = [Self::Nmc811, Self::Lfp, Self::Nca, Self::Lto];

    /// Converts this UI choice to [`ChemistryPreset`].
    #[must_use]
    pub const fn to_preset(self) -> ChemistryPreset {
        match self {
            Self::Nmc811 => ChemistryPreset::Nmc811,
            Self::Lfp => ChemistryPreset::Lfp,
            Self::Nca => ChemistryPreset::Nca,
            Self::Lto => ChemistryPreset::Lto,
        }
    }

    /// Converts a domain [`ChemistryPreset`] to [`ChemistryChoice`].
    #[must_use]
    pub const fn from_preset(preset: ChemistryPreset) -> Self {
        match preset {
            ChemistryPreset::Nmc811 => Self::Nmc811,
            ChemistryPreset::Lfp => Self::Lfp,
            ChemistryPreset::Nca => Self::Nca,
            ChemistryPreset::Lto => Self::Lto,
        }
    }

    /// Short display label (`"NMC811"`, `"LFP"`, `"NCA"`, `"LTO"`).
    #[must_use]
    pub const fn label(self) -> &'static str {
        self.to_preset().label()
    }

    /// Predicate used by `#[serde(skip_serializing_if = ...)]` to omit default `NMC811`.
    #[must_use]
    pub const fn is_default(&self) -> bool {
        matches!(self, Self::Nmc811)
    }
}

/// Downloadable calibrated `.json` cell specification card shared across battery tools.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BatteryCellCard {
    /// Schema revision identifier.
    pub schema_version: String,
    /// Originating interactive calculator name.
    pub tool: String,
    /// Base chemistry family.
    pub chemistry: ChemistryChoice,
    /// Cathode/anode chemistry summary.
    pub chemistry_description: String,
    /// DoD power-law prefactor $\alpha$.
    pub alpha: f64,
    /// DoD power-law exponent $\beta$.
    pub beta: f64,
    /// Stretched cycling exponent $z \in [0.1, 2.0]$.
    pub exponent_z: f64,
    /// Arrhenius activation energy $E_a$ ($\text{J/mol}$).
    pub activation_energy_j_mol: f64,
    /// C-rate stress exponent $\gamma_c$.
    pub c_rate_exponent: f64,
    /// Reference $25\,^\circ\text{C}$ SEI calendar fade rate ($\text{day}^{-1/2}$).
    pub sei_rate_per_sqrt_day: f64,
    /// Operating temperature ($^\circ\text{C}$).
    pub temperature_c: f64,
    /// Operating C-rate (`1/h`).
    pub c_rate: f64,
    /// Reference depth of discharge (`%`).
    pub dod_pct: f64,
    /// Arrhenius thermal acceleration factor relative to $25\,^\circ\text{C}$.
    pub arrhenius_factor: f64,
    /// Projected cycling life to `80%` SOH (cycles).
    pub projected_n80_cycles: f64,
    /// Projected cycling life to `70%` SOH (cycles).
    pub projected_n70_cycles: f64,
    /// Optional Levenberg-Marquardt calibration RMSE.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_rmse: Option<f64>,
    /// Optional number of lab `(cycles, capacity)` points used in calibration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_samples: Option<usize>,
}

impl BatteryCellCard {
    /// Builds a [`BatteryCellCard`] snapshot from an [`ElectrochemicalAgingModel`] and DoD.
    #[must_use]
    pub fn from_model(
        tool: &str,
        model: &ElectrochemicalAgingModel,
        dod_pct: f64,
        calibration: Option<(f64, usize)>,
    ) -> Self {
        let dod = DepthOfDischarge::new_clamped(dod_pct.clamp(0.1, 100.0));
        let n80 = model
            .cycles_to_capacity(Capacity::new_clamped(0.80), dod)
            .as_f64();
        let n70 = model.n70(dod).as_f64();
        let preset = model.preset();
        Self {
            schema_version: "1.0".to_string(),
            tool: tool.to_string(),
            chemistry: ChemistryChoice::from_preset(preset),
            chemistry_description: preset.description().to_string(),
            alpha: model.alpha(),
            beta: model.beta(),
            exponent_z: model.exponent_z(),
            activation_energy_j_mol: model.activation_energy(),
            c_rate_exponent: model.c_rate_exponent(),
            sei_rate_per_sqrt_day: model.sei_rate(),
            temperature_c: model.temperature_c(),
            c_rate: model.c_rate(),
            dod_pct,
            arrhenius_factor: model.arrhenius_factor(),
            projected_n80_cycles: n80,
            projected_n70_cycles: n70,
            calibration_rmse: calibration.map(|(rmse, _)| rmse),
            calibration_samples: calibration.map(|(_, count)| count),
        }
    }

    /// Serializes this cell card as formatted UTF-8 JSON.
    pub fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize JSON cell card: {e}"))
    }
}

/// Guards exported CSV text cells against spreadsheet formula injection (`=`, `+`, `-`, `@`).
#[must_use]
pub fn sanitize_csv_text_cell(value: &str) -> String {
    let trimmed = value.trim_start();
    let needs_prefix = matches!(
        trimmed.chars().next(),
        Some('=' | '+' | '-' | '@' | '\t' | '\r')
    );
    let safe = if needs_prefix {
        format!("'{value}")
    } else {
        value.to_string()
    };
    if safe.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", safe.replace('"', "\"\""))
    } else {
        safe
    }
}

/// Non-blocking channel state for client-side `.csv` open and `.json`/`.csv` save dialogs.
#[derive(Default)]
pub struct BatteryDialogIo {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
}

impl BatteryDialogIo {
    /// Launches a `.csv` file picker for lab `(cycles, capacity)` cycling logs.
    pub fn trigger_csv_open(&mut self) {
        let (tx, rx) = channel();
        let cfg = OpenDialogConfig {
            filter_name: "Battery Cycling Log (*.csv)",
            extensions: &["csv", "txt"],
            accept: ".csv,text/csv,text/plain",
            read_error_prefix: "Failed to read battery cycling CSV",
        };
        spawn_open_file_dialog(tx, cfg, |path| {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "cycling_log.csv".to_string())
        });
        self.open_rx = Some(rx);
    }

    /// Launches a `.json` save-as dialog / browser download for a [`BatteryCellCard`].
    pub fn trigger_cell_card_save(
        &mut self,
        filename: &str,
        card: &BatteryCellCard,
    ) -> Result<(), String> {
        let payload = card.to_pretty_json()?;
        let (tx, rx) = channel();
        let cfg = SaveAsDialogConfig {
            filter_name: "Battery Cell Card (*.json)",
            extensions: &["json"],
            fallback_filename: "battery_cell_card.json",
            write_error_prefix: "Failed to save battery cell card JSON",
        };
        spawn_save_as_file_dialog(tx, cfg, filename, payload, |_, name| {
            format!("Saved JSON cell card to {name}")
        });
        self.save_rx = Some(rx);
        Ok(())
    }

    /// Launches a `.csv` save-as dialog / browser download for a warranty SOH schedule.
    pub fn trigger_schedule_csv_save(&mut self, filename: &str, csv_payload: String) {
        let (tx, rx) = channel();
        let cfg = SaveAsDialogConfig {
            filter_name: "Warranty SOH Schedule (*.csv)",
            extensions: &["csv"],
            fallback_filename: "battery_warranty_schedule.csv",
            write_error_prefix: "Failed to save warranty schedule CSV",
        };
        spawn_save_as_file_dialog(tx, cfg, filename, csv_payload, |_, name| {
            format!("Saved yearly SOH warranty schedule to {name}")
        });
        self.save_rx = Some(rx);
    }

    /// Polls any completed CSV open operation without blocking the GUI frame.
    pub fn poll_open(&mut self) -> Option<Result<(String, String), String>> {
        let rx = self.open_rx.as_ref()?;
        match rx.try_recv() {
            Ok(res) => {
                self.open_rx = None;
                Some(res)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.open_rx = None;
                None
            }
        }
    }

    /// Polls any completed JSON/CSV save operation without blocking the GUI frame.
    pub fn poll_save(&mut self) -> Option<Result<String, String>> {
        let rx = self.save_rx.as_ref()?;
        match rx.try_recv() {
            Ok(res) => {
                self.save_rx = None;
                Some(res)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.save_rx = None;
                None
            }
        }
    }
}
