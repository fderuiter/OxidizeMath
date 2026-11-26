//! Coupled battery lifetime and warranty estimator with `1..=8` multi-segment duty
//! missions, `egui_plot` SOH trajectories, JSON cell cards, and yearly warranty CSV schedules.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:battery_degradation]

pub mod duty;
pub mod export;
#[cfg(test)]
mod tests;
mod ui;

use super::cell_card::{BatteryCellCard, BatteryDialogIo, ChemistryChoice};
use crate::framework::{ExportCapabilities, InteractiveTool};
use duty::{validate_duty_segments, DutyPhaseKind, DutySegmentSpec};
use eframe::egui;
use math_explorer::applied::battery_degradation::{
    estimate_duty_lifetime, evaluate_duty_point, Capacity, ChemistryPreset, DutyLifetimeSummary,
    DutySegment, ElectrochemicalAgingModel, DAYS_PER_YEAR, MAX_C_RATE, MAX_DUTY_SEGMENTS,
    MAX_EXPONENT_Z, MAX_TEMPERATURE_C, MIN_C_RATE, MIN_EXPONENT_Z, MIN_TEMPERATURE_C,
};

const PLOT_SAMPLES: usize = 240;

fn default_temp_c() -> f64 {
    25.0
}

fn is_default_temp_c(v: &f64) -> bool {
    (*v - 25.0).abs() < 1e-12
}

fn default_c_rate() -> f64 {
    1.0
}

fn is_default_c_rate(v: &f64) -> bool {
    (*v - 1.0).abs() < 1e-12
}

pub struct LifetimeEstimatorTool {
    pub(crate) target_capacity: f64,
    pub(crate) dod: f64,
    pub(crate) cycles_per_day: f64,
    pub(crate) chemistry: ChemistryChoice,
    pub(crate) temperature: f64,
    pub(crate) c_rate: f64,
    pub(crate) exponent_z: f64,
    pub(crate) duty_segments: Vec<DutySegmentSpec>,
    pub(crate) io_status: Option<(bool, String)>,
    pub(crate) dialogs: BatteryDialogIo,
}

impl Default for LifetimeEstimatorTool {
    fn default() -> Self {
        let preset = ChemistryPreset::Nmc811;
        Self {
            target_capacity: 80.0,
            dod: 50.0,
            cycles_per_day: 1.0,
            chemistry: ChemistryChoice::Nmc811,
            temperature: 25.0,
            c_rate: 1.0,
            exponent_z: preset.default_exponent_z(),
            duty_segments: Vec::new(),
            io_status: None,
            dialogs: BatteryDialogIo::default(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LifetimeEstimatorState {
    target_capacity: f64,
    dod: f64,
    cycles_per_day: f64,
    #[serde(default, skip_serializing_if = "ChemistryChoice::is_default")]
    chemistry: ChemistryChoice,
    #[serde(default = "default_temp_c", skip_serializing_if = "is_default_temp_c")]
    temperature: f64,
    #[serde(default = "default_c_rate", skip_serializing_if = "is_default_c_rate")]
    c_rate: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exponent_z: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    duty_segments: Vec<DutySegmentSpec>,
}

impl InteractiveTool for LifetimeEstimatorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Lifetime Estimator"
    }
    fn domain(&self) -> &'static str {
        "battery_degradation"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn save_state(&self) -> Option<String> {
        let default_z = self.chemistry.to_preset().default_exponent_z();
        let custom_z = if (self.exponent_z - default_z).abs() > 1e-9 {
            Some(self.exponent_z)
        } else {
            None
        };
        serde_json::to_string(&LifetimeEstimatorState {
            target_capacity: self.target_capacity,
            dod: self.dod,
            cycles_per_day: self.cycles_per_day,
            chemistry: self.chemistry,
            temperature: self.temperature,
            c_rate: self.c_rate,
            exponent_z: custom_z,
            duty_segments: self
                .duty_segments
                .iter()
                .take(MAX_DUTY_SEGMENTS)
                .cloned()
                .collect(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: LifetimeEstimatorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid lifetime estimator state: {e}"))?;
        validate_lifetime_state(&s)?;
        let preset = s.chemistry.to_preset();
        self.target_capacity = s.target_capacity;
        self.dod = s.dod;
        self.cycles_per_day = s.cycles_per_day;
        self.chemistry = s.chemistry;
        self.temperature = s.temperature;
        self.c_rate = s.c_rate;
        self.exponent_z = s.exponent_z.unwrap_or_else(|| preset.default_exponent_z());
        self.duty_segments = s.duty_segments;
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (coupled, cycling_only) = self.soh_trajectory_curves();
        vec![
            ("Coupled SOH".to_string(), coupled),
            ("Cycling-Only SOH".to_string(), cycling_only),
        ]
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.clamp_bounds();
        self.poll_save_channel();
        self.render_controls_panel(ui);
        self.render_lifetime_plot_panel(ui);
    }
}

fn validate_lifetime_state(s: &LifetimeEstimatorState) -> Result<(), String> {
    let scalar_ok = s.target_capacity.is_finite()
        && (10.0..=99.0).contains(&s.target_capacity)
        && s.dod.is_finite()
        && (0.0..=100.0).contains(&s.dod)
        && s.cycles_per_day.is_finite()
        && (0.0..=10.0).contains(&s.cycles_per_day)
        && s.temperature.is_finite()
        && (MIN_TEMPERATURE_C..=MAX_TEMPERATURE_C).contains(&s.temperature)
        && s.c_rate.is_finite()
        && (MIN_C_RATE..=MAX_C_RATE).contains(&s.c_rate);
    if !scalar_ok {
        return Err("Lifetime estimator parameters out of bounds".into());
    }
    if let Some(z) = s.exponent_z {
        if !z.is_finite() || !(MIN_EXPONENT_Z..=MAX_EXPONENT_Z).contains(&z) {
            return Err("Stretched exponent z out of bounds [0.1, 2.0]".into());
        }
    }
    validate_duty_segments(&s.duty_segments)
}

impl LifetimeEstimatorTool {
    pub(crate) fn clamp_bounds(&mut self) {
        self.target_capacity = self.target_capacity.clamp(10.0, 99.0);
        self.dod = self.dod.clamp(0.0, 100.0);
        self.cycles_per_day = self.cycles_per_day.clamp(0.0, 10.0);
        self.temperature = self.temperature.clamp(MIN_TEMPERATURE_C, MAX_TEMPERATURE_C);
        self.c_rate = self.c_rate.clamp(MIN_C_RATE, MAX_C_RATE);
        self.exponent_z = self.exponent_z.clamp(MIN_EXPONENT_Z, MAX_EXPONENT_Z);
        self.duty_segments.truncate(MAX_DUTY_SEGMENTS);
        for seg in &mut self.duty_segments {
            seg.clamp_in_place();
        }
    }

    #[must_use]
    pub fn active_model(&self) -> ElectrochemicalAgingModel {
        let preset = self.chemistry.to_preset();
        ElectrochemicalAgingModel::new(
            preset,
            preset.default_alpha(),
            preset.default_beta(),
            self.exponent_z,
            self.temperature,
            self.c_rate,
        )
    }

    #[must_use]
    pub fn effective_duty_segments(&self) -> Vec<DutySegment> {
        if self.duty_segments.is_empty() {
            vec![DutySegment::new(
                1.0,
                self.dod,
                self.cycles_per_day,
                self.temperature,
                self.c_rate,
            )]
        } else {
            self.duty_segments
                .iter()
                .take(MAX_DUTY_SEGMENTS)
                .map(DutySegmentSpec::to_domain)
                .collect()
        }
    }

    #[must_use]
    pub fn lifetime_summary(&self) -> DutyLifetimeSummary {
        let model = self.active_model();
        let segs = self.effective_duty_segments();
        let target = Capacity::new_clamped((self.target_capacity / 100.0).clamp(0.10, 0.99));
        estimate_duty_lifetime(&model, &segs, target)
    }

    pub(crate) fn soh_trajectory_curves(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let model = self.active_model();
        let segs = self.effective_duty_segments();
        let summary = self.lifetime_summary();
        let horizon_years = (summary.years_to_70 * 1.15).clamp(1.0, 60.0);
        let mut coupled = Vec::with_capacity(PLOT_SAMPLES + 1);
        let mut cyc_only = Vec::with_capacity(PLOT_SAMPLES + 1);
        for i in 0..=PLOT_SAMPLES {
            let yr = horizon_years * (i as f64) / (PLOT_SAMPLES as f64);
            let pt = evaluate_duty_point(&model, &segs, yr * DAYS_PER_YEAR);
            coupled.push([yr, pt.soh]);
            cyc_only.push([yr, pt.cycling_only_soh]);
        }
        (coupled, cyc_only)
    }

    #[must_use]
    pub fn build_cell_card(&self) -> BatteryCellCard {
        BatteryCellCard::from_model(
            "LifetimeEstimatorTool",
            &self.active_model(),
            self.dod,
            None,
        )
    }

    pub(crate) fn add_duty_segment(&mut self, kind: DutyPhaseKind) {
        if self.duty_segments.is_empty() {
            self.duty_segments.push(DutySegmentSpec {
                kind: DutyPhaseKind::Commute,
                days: 5.0,
                dod: self.dod,
                cycles: self.cycles_per_day,
                temp_c: self.temperature,
                c_rate: self.c_rate,
            });
        }
        if self.duty_segments.len() < MAX_DUTY_SEGMENTS {
            self.duty_segments.push(DutySegmentSpec::from_kind(kind));
        }
    }

    fn poll_save_channel(&mut self) {
        if let Some(res) = self.dialogs.poll_save() {
            match res {
                Ok(msg) => self.io_status = Some((false, msg)),
                Err(err) => self.io_status = Some((true, err)),
            }
        }
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LifetimeEstimatorTool",
        display_name: "Lifetime Estimator",
        domain: "battery_degradation",
        tags: &["battery", "cycle life", "end-of-life", "depth of discharge", "wohler", "longevity"],
        build: || Box::new(LifetimeEstimatorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for LifetimeEstimatorTool {
    fn theory_description(&self) -> String {
        format!(
            "Coupled {} battery lifetime and warranty solver integrating stretched cycling fade $0.3 (N / N_{{70}}(D, T, I_c))^z$ and parabolic $\\sqrt{{t}}$ SEI calendar growth toward {:.1}% target SOH.",
            self.chemistry.label(),
            self.target_capacity
        )
    }
    fn phonetic_description(&self) -> String {
        "Coupled battery duty-cycle lifetime and parabolic SEI calendar warranty theory".into()
    }
    fn theory_citation(&self) -> String {
        "Cycle-life model for graphite-LiFePO4 cells (Wang et al., 2011); Capacity Fade Model for Li-ion Batteries (Smith et al., 2011)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Battery Lifetime Estimator",
    "domain": "battery_degradation",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "target_capacity": 80.0,
    "dod": 60.0,
    "cycles_per_day": 1.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lifetime_estimator_starter",
        title: "Battery Lifetime Estimator",
        description: "Coupled multi-segment duty-cycle and calendar SEI battery warranty lifetime calculator.",
        category: "Battery Systems",
        icon: "⏳",
        domain: "battery_degradation",
        tool_name: Some("LifetimeEstimatorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
