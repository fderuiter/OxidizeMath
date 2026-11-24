#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::bioelectric_leads::BioelectricLeadTool;
use crate::app::palette::SemanticPalette;
use crate::starter_gallery::StarterPresetMetadata;
use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke};
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

    pub(super) fn clamp_in_place(&mut self) {
        self.lead_height_z = if self.lead_height_z.is_finite() {
            self.lead_height_z.clamp(0.5, 40.0)
        } else {
            8.0
        };
        self.phase_rate = if self.phase_rate.is_finite() {
            self.phase_rate.clamp(0.01, 0.5)
        } else {
            0.08
        };
        for l in &mut self.leads {
            l.x = if l.x.is_finite() {
                l.x.clamp(0.0, 96.0)
            } else {
                48.0
            };
            l.y = if l.y.is_finite() {
                l.y.clamp(0.0, 96.0)
            } else {
                48.0
            };
            l.z = if l.z.is_finite() {
                l.z.clamp(0.5, 64.0)
            } else {
                8.0
            };
        }
    }
}

const PRESET_MED_DUAL_EDGE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Dual-Edge Wavefront Collision",
    "domain": "medical",
    "timestamp": "2026-11-23T00:00:00Z"
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
    "timestamp": "2026-11-23T00:00:00Z"
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
    "timestamp": "2026-11-23T00:00:00Z"
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
    "timestamp": "2026-11-23T00:00:00Z"
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

impl BioelectricLeadTool {
    pub(super) fn paint_trace_strip(
        &self,
        painter: &egui::Painter,
        strip: Rect,
        palette: &SemanticPalette,
        bg: Color32,
    ) {
        if strip.height() <= 20.0 || strip.width() <= 40.0 {
            return;
        }
        painter.rect_filled(strip, 4.0_f32, bg);
        painter.rect_stroke(
            strip,
            4.0_f32,
            Stroke::new(1.0_f32, palette.graph_edge),
            egui::StrokeKind::Inside,
        );
        let names = self.lead_channel_names();
        let peak = self
            .history
            .iter()
            .flat_map(|(_, v)| v.iter().map(|x| x.abs()))
            .fold(0.25_f64, f64::max)
            * 1.15;
        let lane_h = strip.height() / 3.0;
        let n_pts = self.history.len();
        for (k, &col) in palette.series.iter().take(3).enumerate() {
            let y_top = strip.top() + k as f32 * lane_h;
            let y_mid = y_top + lane_h * 0.5;
            if n_pts >= 2 {
                let pts: Vec<Pos2> = self
                    .history
                    .iter()
                    .enumerate()
                    .map(|(idx, &(_, v))| {
                        let ux = idx as f32 / (n_pts - 1) as f32;
                        let norm = (v[k] / peak).clamp(-1.0, 1.0) as f32;
                        egui::pos2(
                            strip.left() + 6.0 + ux * (strip.width() - 12.0),
                            y_mid - norm * (lane_h * 0.40),
                        )
                    })
                    .collect();
                painter.add(egui::Shape::line(pts, Stroke::new(1.8_f32, col)));
            }
            let latest = self.history.back().map_or(0.0, |(_, v)| v[k]);
            painter.text(
                egui::pos2(strip.left() + 8.0, y_top + 3.0),
                egui::Align2::LEFT_TOP,
                format!("{}: {:+.3}", names[k], latest),
                FontId::monospace(11.0),
                col,
            );
        }
    }

    pub(super) fn paint_lead_markers(
        &self,
        painter: &egui::Painter,
        map_rect: Rect,
        palette: &SemanticPalette,
        bg: Color32,
    ) {
        let tags = ["L1", "L2", "L3"];
        for (i, lead) in self.cfg.leads.iter().enumerate() {
            let p = egui::pos2(
                map_rect.left() + (lead.x as f32 / 96.0) * map_rect.width(),
                map_rect.top() + (lead.y as f32 / 96.0) * map_rect.height(),
            );
            let rad = if self.active_lead == i {
                8.0_f32
            } else {
                6.5_f32
            };
            painter.circle_filled(p, rad, bg);
            painter.circle_stroke(p, rad, Stroke::new(2.0_f32, palette.series[i]));
            let anchor = p + egui::vec2(9.0, -7.0);
            painter.rect_filled(
                Rect::from_min_size(anchor - egui::vec2(2.0, 7.0), egui::vec2(20.0, 14.0)),
                3.0_f32,
                bg,
            );
            painter.text(
                anchor,
                egui::Align2::LEFT_CENTER,
                tags[i],
                FontId::proportional(11.0),
                palette.node_stroke,
            );
        }
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

    fn maturity_tier(&self) -> scientific_metadata::theory::MaturityTier {
        scientific_metadata::theory::MaturityTier::Certified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::{ExportCapabilities, InteractiveTool};
    use eframe::egui::{self, Pos2, RawInput, Rect};

    #[test]
    fn test_bioelectric_lead_tool_exports_focus_and_validation() {
        let mut tool = BioelectricLeadTool::default();
        assert_eq!(
            tool.theory().maturity_tier(),
            scientific_metadata::theory::MaturityTier::Certified
        );
        assert_eq!(tool.export_capabilities(), ExportCapabilities::plot());
        let bipolar_ds = tool.export_datasets();
        assert_eq!(bipolar_ds.len(), 3);
        assert_eq!(bipolar_ds[0].0, "Lead I (LA-RA)");

        tool.cfg.readout_mode = LeadReadoutMode::UnipolarLfp;
        tool.rebuild_field_and_sample();
        let unipolar_ds = tool.export_datasets();
        assert_eq!(unipolar_ds[0].0, "LFP Lead 1");

        assert_eq!(tool.export_raster_image().unwrap().size, [MAP_DIM, MAP_DIM]);
        assert!(tool.focused_canvas_id().is_none());

        let narrow = BioelectricLeadTool::map_subrect(Rect::from_min_size(
            Pos2::ZERO,
            egui::vec2(320.0, 800.0),
        ));
        assert!((narrow.width() - narrow.height()).abs() < 1e-4);

        tool.cfg.lead_height_z = f64::NAN;
        tool.cfg.leads[1].x = 250.0;
        assert!(!tool.has_finite_parameters());
        assert!(tool.clamp_parameters());
        assert!(tool.has_finite_parameters());
        assert_eq!(tool.cfg.leads[1].x, 96.0);

        let ctx = egui::Context::default();
        let screen = Rect::from_min_size(Pos2::ZERO, egui::vec2(1200.0, 800.0));
        let _ = ctx.run(
            RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ctx| tool.show(ctx),
        );
        let saved = tool.save_state().unwrap();
        assert!(tool.try_load_state(&saved).is_ok());
        assert!(tool.try_load_state(r#"{"unknown": 1}"#).is_err());
    }
}
