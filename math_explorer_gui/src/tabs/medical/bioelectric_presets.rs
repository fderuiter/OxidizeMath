#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::bioelectric_leads::BioelectricLeadTool;
use crate::starter_gallery::StarterPresetMetadata;
use math_explorer::physics::medical::bioelectric::{
    BioelectricPresetKind, LeadPosition, LeadReadoutMode,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub(super) const MAP_DIM: usize = 96;

/// Validated state schema for [`BioelectricLeadTool`] in `MedicalTab`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BioelectricLeadState {
    /// Default stand-off elevation for moved leads (`0.5..=40.0`).
    pub lead_height_z: f64,
    /// Kinematic phase increment per frame (`0.01..=0.5`).
    pub phase_rate: f64,
    /// Shared electrophysiology wavefront preset.
    pub preset: BioelectricPresetKind,
    /// Extracellular readout derivation mode.
    pub readout_mode: LeadReadoutMode,
    /// Three movable virtual lead positions.
    pub leads: [LeadPosition; 3],
    /// Whether WCAG high-contrast colormap is active.
    pub high_contrast: bool,
    /// Whether catheter ablation line connects central scar to boundary.
    pub ablated_to_edge: bool,
    /// Whether kinematic wavefront animation is active.
    pub running: bool,
}

impl Default for BioelectricLeadState {
    fn default() -> Self {
        let spec = BioelectricPresetKind::S1S2SpiralInduction.spec(MAP_DIM, MAP_DIM);
        Self {
            lead_height_z: 8.0,
            phase_rate: 0.08,
            preset: spec.kind,
            readout_mode: spec.readout_mode,
            leads: spec.leads,
            high_contrast: false,
            ablated_to_edge: false,
            running: false,
        }
    }
}

impl BioelectricLeadState {
    pub(super) fn validate(&self) -> Result<(), String> {
        if !self.lead_height_z.is_finite() || !(0.5..=40.0).contains(&self.lead_height_z) {
            return Err("lead_height_z must be within [0.5, 40.0]".to_string());
        }
        if !self.phase_rate.is_finite() || !(0.01..=0.5).contains(&self.phase_rate) {
            return Err("phase_rate must be within [0.01, 0.5]".to_string());
        }
        for l in &self.leads {
            let ok = l.x.is_finite()
                && l.y.is_finite()
                && l.z.is_finite()
                && (0.0..=96.0).contains(&l.x)
                && (0.0..=96.0).contains(&l.y)
                && (0.5..=64.0).contains(&l.z);
            if !ok {
                return Err("LeadPosition out of bounds".to_string());
            }
        }
        Ok(())
    }
}

const PRESET_MED_DUAL_EDGE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Dual-Edge Wavefront Collision",
    "domain": "medical",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "lead_height_z": 8.0,
    "phase_rate": 0.08,
    "preset": "DualEdgeCollision",
    "readout_mode": "UnipolarLfp",
    "leads": [
      { "x": 21.12, "y": 48.0, "z": 8.0 },
      { "x": 48.0, "y": 48.0, "z": 8.0 },
      { "x": 74.88, "y": 48.0, "z": 8.0 }
    ],
    "high_contrast": false,
    "ablated_to_edge": false,
    "running": true
  }
}"#;

const PRESET_MED_S1_S2_SPIRAL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "S1-S2 Spiral Arrhythmia",
    "domain": "medical",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "lead_height_z": 8.0,
    "phase_rate": 0.09,
    "preset": "S1S2SpiralInduction",
    "readout_mode": "BipolarPseudoEcg",
    "leads": [
      { "x": 17.28, "y": 19.2, "z": 8.0 },
      { "x": 78.72, "y": 19.2, "z": 8.0 },
      { "x": 48.0, "y": 82.56, "z": 8.0 }
    ],
    "high_contrast": false,
    "ablated_to_edge": false,
    "running": true
  }
}"#;

const PRESET_MED_SCAR_ABLATION: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Scar-Pinned Rotor & Ablation",
    "domain": "medical",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "lead_height_z": 7.5,
    "phase_rate": 0.09,
    "preset": "ScarPinningAblation",
    "readout_mode": "BipolarPseudoEcg",
    "leads": [
      { "x": 23.04, "y": 24.96, "z": 7.5 },
      { "x": 72.96, "y": 24.96, "z": 7.5 },
      { "x": 48.0, "y": 78.72, "z": 7.5 }
    ],
    "high_contrast": false,
    "ablated_to_edge": false,
    "running": true
  }
}"#;

const PRESET_MED_DEFIBRILLATION: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Whole-Sheet Defibrillation",
    "domain": "medical",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "lead_height_z": 8.0,
    "phase_rate": 0.08,
    "preset": "WholeSheetDefibrillation",
    "readout_mode": "BipolarPseudoEcg",
    "leads": [
      { "x": 17.28, "y": 19.2, "z": 8.0 },
      { "x": 78.72, "y": 19.2, "z": 8.0 },
      { "x": 48.0, "y": 82.56, "z": 8.0 }
    ],
    "high_contrast": true,
    "ablated_to_edge": false,
    "running": true
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "medical_bioelectric_dual_edge",
        title: "Dual-Edge Wavefront Collision",
        description: "3-lead extracellular LFP dipole signatures of colliding planar wavefronts.",
        category: "Medical Physics",
        icon: "🌊",
        domain: "medical",
        tool_name: Some("BioelectricLeadTool"),
        preset_generator: || PRESET_MED_DUAL_EDGE,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "medical_bioelectric_s1_s2_spiral",
        title: "S1-S2 Spiral Arrhythmia",
        description: "Einthoven 3-lead pseudo-ECG mapping of re-entrant spiral tachycardia.",
        category: "Medical Physics",
        icon: "🌀",
        domain: "medical",
        tool_name: Some("BioelectricLeadTool"),
        preset_generator: || PRESET_MED_S1_S2_SPIRAL,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "medical_bioelectric_scar_ablation",
        title: "Scar-Pinned Rotor & Ablation",
        description: "Optimize 3-lead placement and catheter ablation across ischemic conduction-block scars.",
        category: "Medical Physics",
        icon: "🫀",
        domain: "medical",
        tool_name: Some("BioelectricLeadTool"),
        preset_generator: || PRESET_MED_SCAR_ABLATION,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "medical_bioelectric_defibrillation",
        title: "Whole-Sheet Defibrillation",
        description: "Whole-sheet defibrillation reset across Einthoven 3-lead extracellular electrodes.",
        category: "Medical Physics",
        icon: "⚡",
        domain: "medical",
        tool_name: Some("BioelectricLeadTool"),
        preset_generator: || PRESET_MED_DEFIBRILLATION,
    }
}

impl scientific_metadata::theory::TheoryDescribable for BioelectricLeadTool {
    fn theory_description(&self) -> String {
        "Volume-conductor 3-lead dipole projection Φ_k ∝ ∑(∇V · d_k)/‖d_k‖³ evaluating Einthoven pseudo-ECG and local field potentials around excitable wavefronts and ischemic scars.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Extracellular potential Phi k is proportional to the spatial sum of transmembrane voltage gradient dot lead vector d k divided by norm d k cubed.".to_string()
    }

    fn theory_citation(&self) -> String {
        "Plonsey, R., & Barr, R. C. (2007). Bioelectricity: A Quantitative Approach (3rd ed.). Springer.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}
