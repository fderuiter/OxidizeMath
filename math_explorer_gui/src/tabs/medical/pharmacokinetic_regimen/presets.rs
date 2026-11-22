//! `ToolMetadata`, `StarterPresetMetadata` cards, and `TheoryDescribable` implementation.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::collections::HashMap;
use std::sync::OnceLock;

use scientific_metadata::theory::TheoryDescribable;

use super::PharmacokineticRegimenTool;
use super::state::RegimenState;

fn preset_loading_dose_json() -> &'static str {
    static JSON: OnceLock<String> = OnceLock::new();
    JSON.get_or_init(|| {
        crate::presets::serialize_compact_preset_envelope(
            "2D Loading Dose vs Maintenance Window",
            "medical",
            RegimenState::loading_dose_scenario(),
        )
        .expect("valid loading dose preset envelope")
    })
}

fn preset_missed_dose_json() -> &'static str {
    static JSON: OnceLock<String> = OnceLock::new();
    JSON.get_or_init(|| {
        crate::presets::serialize_compact_preset_envelope(
            "Missed 12h Dose: 1.5D Catch-Up Recovery",
            "medical",
            RegimenState::missed_dose_catchup_scenario(),
        )
        .expect("valid missed-dose preset envelope")
    })
}

fn preset_chiral_xr_json() -> &'static str {
    static JSON: OnceLock<String> = OnceLock::new();
    JSON.get_or_init(|| {
        crate::presets::serialize_compact_preset_envelope(
            "Chiral Biphasic XR Regimen",
            "medical",
            RegimenState::chiral_xr_scenario(),
        )
        .expect("valid chiral XR preset envelope")
    })
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PharmacokineticRegimenTool",
        display_name: "Pharmacokinetic Regimen",
        domain: "medical",
        tags: &["pharmacokinetics", "adme", "regimen", "therapeutic-window", "dosing"],
        build: || Box::new(PharmacokineticRegimenTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "pk_loading_dose_window",
        title: "2D Loading Dose vs Maintenance Window",
        description: "Compares a 2D (200 mg) loading dose + 100 mg q12h maintenance schedule against uniform maintenance, raising 48 h Time-in-Therapeutic-Range from 66.49% to 86.89% with 0% toxic overshoot.",
        category: "Pharmacokinetics",
        icon: "💊",
        domain: "medical",
        tool_name: Some("PharmacokineticRegimenTool"),
        preset_generator: preset_loading_dose_json,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "pk_missed_dose_recovery",
        title: "Missed 12h Dose: 1.5D Catch-Up Recovery",
        description: "Models a missed 12 h dose recovered with a 150 mg (1.5D) catch-up dose at 18 h, restoring therapeutic concentrations safely (0% toxic time) unlike an uncalibrated 2D double dose.",
        category: "Pharmacokinetics",
        icon: "⏱",
        domain: "medical",
        tool_name: Some("PharmacokineticRegimenTool"),
        preset_generator: preset_missed_dose_json,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "pk_chiral_xr_biphasic",
        title: "Chiral Biphasic XR Regimen",
        description: "Simulates a 3:1 (d:l) chiral extended-release two-pulse capsule with an afternoon immediate-release booster, overlaying individual isomer plasma curves.",
        category: "Pharmacokinetics",
        icon: "🧬",
        domain: "medical",
        tool_name: Some("PharmacokineticRegimenTool"),
        preset_generator: preset_chiral_xr_json,
    }
}

impl TheoryDescribable for PharmacokineticRegimenTool {
    fn theory_description(&self) -> String {
        format!(
            "Linear one-compartment ADME multi-dose superposition sums event-scaled Bateman responses $C(t) = \\sum_{{i: t \\ge t_i}} C_{{D_i}}(t - t_i)$, where $C_D(\\Delta t) = \\frac{{F D k_a}}{{V_d (k_a - k_e)}}(e^{{-k_e \\Delta t}} - e^{{-k_a \\Delta t}})$ for $k_a = {:.2}\\,\\text{{h}}^{{-1}}$, $k_e = {:.3}\\,\\text{{h}}^{{-1}}$, $V_d = {:.1}\\,\\text{{L}}$, and therapeutic window $[\\text{{MEC}}, \\text{{MTC}}] = [{:.2}, {:.2}]\\,\\text{{mg/L}}$.Steady-state accumulation follows $R_{{\\text{{ac}}}} = (1 - e^{{-k_e \\tau}})^{{-1}}$ and trapezoidal exposure $\\text{{AUC}}_{{0-\\tau}} = \\int_{{t - \\tau}}^t C(u)\\,du$.",
            self.state.ka, self.state.ke, self.state.v, self.state.mec, self.state.mtc
        )
    }

    fn phonetic_description(&self) -> String {
        "Multi-dose pharmacokinetic Bateman superposition, absorption calibration, and therapeutic window exposure analysis".to_string()
    }

    fn theory_citation(&self) -> String {
        "Rowland & Tozer, Clinical Pharmacokinetics and Pharmacodynamics (4th ed., 2011); Gibaldi & Perrier, Pharmacokinetics (2nd ed., 1982)".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "maintenance_dose_mg".into(),
            "Reference maintenance dose D (mg) for interval comparisons".into(),
        );
        map.insert(
            "ka".into(),
            "First-order gastrointestinal absorption rate constant ka (1/h)".into(),
        );
        map.insert(
            "ke".into(),
            "First-order systemic elimination rate constant ke (1/h)".into(),
        );
        map.insert(
            "v".into(),
            "Apparent volume of distribution Vd (L)".into(),
        );
        map.insert(
            "mec".into(),
            "Minimum Effective Concentration (mg/L) lower bound of therapeutic window".into(),
        );
        map.insert(
            "mtc".into(),
            "Minimum Toxic Concentration (mg/L) upper bound of therapeutic window".into(),
        );
        map
    }
}
