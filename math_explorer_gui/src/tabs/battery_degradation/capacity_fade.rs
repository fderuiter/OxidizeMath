//! Coupled Li-ion capacity fade calculator with Arrhenius thermal kinetics,
//! C-rate stress, stretched exponent $z$, and live Levenberg-Marquardt CSV calibration.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:battery_degradation]

pub mod io;
#[cfg(test)]
mod tests;
mod ui;

use super::capacity_fade_atlas::{CapacityFadeCartographer, CAPACITY_FADE_ATLAS_AXIS_PAIRS};
use super::cell_card::{BatteryCellCard, BatteryDialogIo, ChemistryChoice};
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::framework::InteractiveTool;
use eframe::egui;
use io::CalibratedLabSummary;
use math_explorer::applied::battery_degradation::{
    Capacity, ChemistryPreset, Cycles, DepthOfDischarge, ElectrochemicalAgingModel, MAX_C_RATE,
    MAX_EXPONENT_Z, MAX_TEMPERATURE_C, MIN_C_RATE, MIN_EXPONENT_Z, MIN_TEMPERATURE_C,
};

const CURVE_SAMPLE_STEPS: usize = 500;

fn default_c_rate() -> f64 {
    1.0
}

fn is_default_c_rate(val: &f64) -> bool {
    (*val - 1.0).abs() < 1e-12
}

pub struct CapacityFadeTool {
    pub(crate) dod: f64,
    pub(crate) temperature: f64,
    pub(crate) cycles_to_simulate: f64,
    pub(crate) chemistry: ChemistryChoice,
    pub(crate) c_rate: f64,
    pub(crate) exponent_z: f64,
    pub(crate) calibrated_alpha: Option<f64>,
    pub(crate) show_residuals: bool,
    pub(crate) lab_samples: Vec<[f64; 2]>,
    pub(crate) last_fit: Option<CalibratedLabSummary>,
    pub(crate) io_status: Option<(bool, String)>,
    pub(crate) dialogs: BatteryDialogIo,
}

impl Default for CapacityFadeTool {
    fn default() -> Self {
        let preset = ChemistryPreset::Nmc811;
        Self {
            dod: 80.0,
            temperature: 25.0,
            cycles_to_simulate: 2000.0,
            chemistry: ChemistryChoice::Nmc811,
            c_rate: 1.0,
            exponent_z: preset.default_exponent_z(),
            calibrated_alpha: None,
            show_residuals: true,
            lab_samples: Vec::new(),
            last_fit: None,
            io_status: None,
            dialogs: BatteryDialogIo::default(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CapacityFadeState {
    dod: f64,
    temperature: f64,
    cycles_to_simulate: f64,
    #[serde(default, skip_serializing_if = "ChemistryChoice::is_default")]
    chemistry: ChemistryChoice,
    #[serde(default = "default_c_rate", skip_serializing_if = "is_default_c_rate")]
    c_rate: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exponent_z: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    calibrated_alpha: Option<f64>,
}

impl InteractiveTool for CapacityFadeTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Capacity Fade"
    }
    fn domain(&self) -> &'static str {
        "battery_degradation"
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.dod.to_bits().hash(&mut h);
        self.temperature.to_bits().hash(&mut h);
        self.cycles_to_simulate.to_bits().hash(&mut h);
        self.chemistry.hash(&mut h);
        self.c_rate.to_bits().hash(&mut h);
        self.exponent_z.to_bits().hash(&mut h);
        self.calibrated_alpha.map(f64::to_bits).hash(&mut h);
        h.finish().max(1)
    }

    fn atlas_axis_pairs(&self) -> &'static [(&'static str, &'static str)] {
        CAPACITY_FADE_ATLAS_AXIS_PAIRS
    }

    fn active_atlas_coords(&self, _axis_keys: (&'static str, &'static str)) -> (f64, f64) {
        (self.dod, self.cycles_to_simulate)
    }

    fn atlas_context_signature(&self, _axis_keys: (&'static str, &'static str)) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.temperature.to_bits().hash(&mut h);
        self.chemistry.hash(&mut h);
        self.c_rate.to_bits().hash(&mut h);
        self.exponent_z.to_bits().hash(&mut h);
        self.calibrated_alpha.map(f64::to_bits).hash(&mut h);
        h.finish()
    }

    fn build_atlas_cartographer(
        &self,
        axis_keys: (&'static str, &'static str),
    ) -> Option<std::sync::Arc<dyn RegimeCartographer>> {
        CapacityFadeCartographer::new(self, axis_keys)
            .map(|c| std::sync::Arc::new(c) as std::sync::Arc<dyn RegimeCartographer>)
    }

    fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        self.apply_capacity_fade_teleport(ev);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn save_state(&self) -> Option<String> {
        let default_z = self.chemistry.to_preset().default_exponent_z();
        let custom_z = if (self.exponent_z - default_z).abs() > 1e-9 {
            Some(self.exponent_z)
        } else {
            None
        };
        serde_json::to_string(&CapacityFadeState {
            dod: self.dod,
            temperature: self.temperature,
            cycles_to_simulate: self.cycles_to_simulate,
            chemistry: self.chemistry,
            c_rate: self.c_rate,
            exponent_z: custom_z,
            calibrated_alpha: self.calibrated_alpha,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: CapacityFadeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid capacity fade state: {e}"))?;
        validate_capacity_fade_state(&s)?;
        let preset = s.chemistry.to_preset();
        self.dod = s.dod;
        self.temperature = s.temperature;
        self.cycles_to_simulate = s.cycles_to_simulate;
        self.chemistry = s.chemistry;
        self.c_rate = s.c_rate;
        self.exponent_z = s.exponent_z.unwrap_or_else(|| preset.default_exponent_z());
        self.calibrated_alpha = s.calibrated_alpha;
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = vec![("Capacity".to_string(), self.capacity_curve_points())];
        if !self.lab_samples.is_empty() {
            out.push(("Measured SOH".to_string(), self.lab_samples.clone()));
            out.push((
                "Residuals (Measured - Model)".to_string(),
                self.residual_points(),
            ));
        }
        out
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.clamp_operating_bounds();
        self.poll_dialog_channels();
        self.render_controls_panel(ui);
        self.render_projection_panel(ui);
    }
}

fn validate_capacity_fade_state(s: &CapacityFadeState) -> Result<(), String> {
    let basic_valid = s.dod.is_finite()
        && (1.0..=100.0).contains(&s.dod)
        && s.temperature.is_finite()
        && (MIN_TEMPERATURE_C..=MAX_TEMPERATURE_C).contains(&s.temperature)
        && s.cycles_to_simulate.is_finite()
        && (100.0..=10_000.0).contains(&s.cycles_to_simulate)
        && s.c_rate.is_finite()
        && (MIN_C_RATE..=MAX_C_RATE).contains(&s.c_rate);
    if !basic_valid {
        return Err("Capacity fade parameters out of bounds".into());
    }
    if let Some(z) = s.exponent_z {
        if !z.is_finite() || !(MIN_EXPONENT_Z..=MAX_EXPONENT_Z).contains(&z) {
            return Err("Stretched exponent z out of bounds [0.1, 2.0]".into());
        }
    }
    if let Some(alpha) = s.calibrated_alpha {
        if !alpha.is_finite() || !(1.0..=1.0e9).contains(&alpha) {
            return Err("Calibrated prefactor alpha out of bounds".into());
        }
    }
    Ok(())
}

impl CapacityFadeTool {
    pub(crate) fn clamp_operating_bounds(&mut self) {
        self.dod = self.dod.clamp(1.0, 100.0);
        self.temperature = self.temperature.clamp(MIN_TEMPERATURE_C, MAX_TEMPERATURE_C);
        self.c_rate = self.c_rate.clamp(MIN_C_RATE, MAX_C_RATE);
        self.exponent_z = self.exponent_z.clamp(MIN_EXPONENT_Z, MAX_EXPONENT_Z);
        self.cycles_to_simulate = self.cycles_to_simulate.clamp(100.0, 10_000.0);
    }

    #[must_use]
    pub fn active_model(&self) -> ElectrochemicalAgingModel {
        let preset = self.chemistry.to_preset();
        let alpha = self
            .calibrated_alpha
            .unwrap_or_else(|| preset.default_alpha());
        ElectrochemicalAgingModel::new(
            preset,
            alpha,
            preset.default_beta(),
            self.exponent_z,
            self.temperature,
            self.c_rate,
        )
    }

    pub(crate) fn apply_chemistry_preset(&mut self, choice: ChemistryChoice) {
        self.chemistry = choice;
        self.exponent_z = choice.to_preset().default_exponent_z();
        self.calibrated_alpha = None;
        self.last_fit = None;
    }

    pub(crate) fn capacity_curve_points(&self) -> Vec<[f64; 2]> {
        let model = self.active_model();
        let dod_opt = DepthOfDischarge::new(self.dod).ok();
        (0..=CURVE_SAMPLE_STEPS)
            .map(|i| {
                let cycle = (self.cycles_to_simulate / CURVE_SAMPLE_STEPS as f64) * (i as f64);
                match (Cycles::new(cycle).ok(), dod_opt) {
                    (Some(c), Some(d)) if self.dod > 0.0 => [cycle, model.capacity(c, d).as_f64()],
                    _ => [cycle, 1.0],
                }
            })
            .collect()
    }

    pub(crate) fn residual_points(&self) -> Vec<[f64; 2]> {
        let model = self.active_model();
        let dod = DepthOfDischarge::new_clamped(self.dod.max(0.1));
        self.lab_samples
            .iter()
            .map(|&[cyc, measured]| {
                let pred = model.capacity(Cycles::new_clamped(cyc), dod).as_f64();
                [cyc, measured - pred]
            })
            .collect()
    }

    pub(crate) fn projected_life_cycles_coupled(&self) -> (f64, f64) {
        if self.dod <= 0.0 {
            return (0.0, 0.0);
        }
        let model = self.active_model();
        if let Ok(d) = DepthOfDischarge::new(self.dod) {
            let n80 = model
                .cycles_to_capacity(Capacity::new_clamped(0.8), d)
                .as_f64();
            (n80, model.n70(d).as_f64())
        } else {
            (0.0, 0.0)
        }
    }

    #[cfg(test)]
    pub(crate) fn projected_life_cycles(dod_percent: f64) -> (f64, f64) {
        if dod_percent <= 0.0 {
            return (0.0, 0.0);
        }
        let model = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Nmc811);
        if let Ok(d) = DepthOfDischarge::new(dod_percent) {
            let n80 = model
                .cycles_to_capacity(Capacity::new_clamped(0.8), d)
                .as_f64();
            (n80, model.n70(d).as_f64())
        } else {
            (0.0, 0.0)
        }
    }

    #[must_use]
    pub fn build_cell_card(&self) -> BatteryCellCard {
        let cal = self.last_fit.as_ref().map(|f| (f.rmse, f.sample_count));
        BatteryCellCard::from_model("CapacityFadeTool", &self.active_model(), self.dod, cal)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CapacityFadeTool",
        display_name: "Capacity Fade",
        domain: "battery_degradation",
        tags: &["lithium-ion", "battery", "sei", "arrhenius", "power-law", "degradation", "soh"],
        build: || Box::new(CapacityFadeTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CapacityFadeTool {
    fn theory_description(&self) -> String {
        format!(
            "Coupled {} capacity fade model: $C(N) = C_0 (1 - 0.3 (N / N_{{70}}(D, T, I_c))^z)$ with Arrhenius acceleration $\\exp(-E_a / (R T))$ at {:.1}% DoD, {:.1}°C, {:.2}C, and $z = {:.2}$.",
            self.chemistry.label(),
            self.dod,
            self.temperature,
            self.c_rate,
            self.exponent_z
        )
    }
    fn phonetic_description(&self) -> String {
        "Coupled electrochemical battery capacity fade and Arrhenius aging theory".into()
    }
    fn theory_citation(&self) -> String {
        "Cycle-life model for graphite-LiFePO4 cells (Wang et al., 2011); Capacity Fade Model for Li-ion Batteries (Smith et al., 2011)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }

    fn theory_parameters(
        &self,
    ) -> std::collections::HashMap<String, scientific_metadata::theory::ParameterConstraint> {
        use scientific_metadata::theory::ParameterConstraint;
        let c = |min, max, step, default| ParameterConstraint {
            min,
            max,
            step,
            default: Some(default),
        };
        std::collections::HashMap::from([
            ("dod".to_string(), c(1.0, 100.0, 1.0, 80.0)),
            ("temperature".to_string(), c(-20.0, 60.0, 1.0, 25.0)),
            ("c_rate".to_string(), c(0.05, 10.0, 0.05, 1.0)),
            (
                "exponent_z".to_string(),
                c(
                    0.1,
                    2.0,
                    0.05,
                    self.chemistry.to_preset().default_exponent_z(),
                ),
            ),
            (
                "cycles_to_simulate".to_string(),
                c(100.0, 10_000.0, 50.0, 2000.0),
            ),
        ])
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Capacity Fade",
    "domain": "battery_degradation",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "dod": 80.0,
    "temperature": 25.0,
    "cycles_to_simulate": 2500.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "capacity_fade_starter",
        title: "Capacity Fade",
        description: "Coupled Arrhenius and stretched-exponent lithium-ion capacity retention with live CSV calibration.",
        category: "Electrochemistry",
        icon: "🔋",
        domain: "battery_degradation",
        tool_name: Some("CapacityFadeTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
