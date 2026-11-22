#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::starter_gallery::StarterPresetMetadata;

/// Starter preset JSON for Dual-Edge Wavefront Collision.
pub const PRESET_EXCITABLE_DUAL_EDGE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Dual-Edge Wavefront Collision",
    "domain": "neuroscience",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "a": 0.7,
    "b": 0.8,
    "epsilon": 0.06,
    "i_ext": 0.0,
    "d_u": 0.5,
    "dt": 0.15,
    "steps_per_frame": 4,
    "brush_radius": 3.5,
    "brush_current": 2.2,
    "clamp_voltage": 1.6,
    "propagation_mode": "FitzHughNagumo2D",
    "brush_mode": "InjectCurrent",
    "colormap": "Perceptual",
    "readout_bipolar": false,
    "active_lead": 0,
    "leads": [
      { "x": 21.12, "y": 48.0, "z": 8.0 },
      { "x": 48.0, "y": 48.0, "z": 8.0 },
      { "x": 74.88, "y": 48.0, "z": 8.0 }
    ],
    "scenario": "DualEdgeCollision",
    "ablated_to_edge": false,
    "running": true
  }
}"#;

/// Starter preset JSON for S1-S2 Spiral Arrhythmia Induction.
pub const PRESET_EXCITABLE_S1_S2_SPIRAL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "S1-S2 Spiral Arrhythmia",
    "domain": "neuroscience",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "a": 0.7,
    "b": 0.8,
    "epsilon": 0.06,
    "i_ext": 0.0,
    "d_u": 0.5,
    "dt": 0.16,
    "steps_per_frame": 5,
    "brush_radius": 3.5,
    "brush_current": 2.2,
    "clamp_voltage": 1.6,
    "propagation_mode": "FitzHughNagumo2D",
    "brush_mode": "InjectCurrent",
    "colormap": "Perceptual",
    "readout_bipolar": true,
    "active_lead": 0,
    "leads": [
      { "x": 17.28, "y": 19.2, "z": 8.0 },
      { "x": 78.72, "y": 19.2, "z": 8.0 },
      { "x": 48.0, "y": 82.56, "z": 8.0 }
    ],
    "scenario": "S1S2SpiralInduction",
    "ablated_to_edge": false,
    "running": true
  }
}"#;

/// Starter preset JSON for Scar-Pinned Rotor & Catheter Ablation.
pub const PRESET_EXCITABLE_SCAR_PINNING: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Scar-Pinned Rotor & Ablation",
    "domain": "neuroscience",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "a": 0.7,
    "b": 0.8,
    "epsilon": 0.06,
    "i_ext": 0.0,
    "d_u": 0.5,
    "dt": 0.16,
    "steps_per_frame": 5,
    "brush_radius": 3.0,
    "brush_current": 2.2,
    "clamp_voltage": 1.6,
    "propagation_mode": "FitzHughNagumo2D",
    "brush_mode": "PaintScar",
    "colormap": "Perceptual",
    "readout_bipolar": true,
    "active_lead": 0,
    "leads": [
      { "x": 23.04, "y": 24.96, "z": 7.5 },
      { "x": 72.96, "y": 24.96, "z": 7.5 },
      { "x": 48.0, "y": 78.72, "z": 7.5 }
    ],
    "scenario": "ScarPinnedRotor",
    "ablated_to_edge": false,
    "running": true
  }
}"#;

/// Starter preset JSON for Whole-Sheet Defibrillation Reset.
pub const PRESET_EXCITABLE_DEFIBRILLATION: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Whole-Sheet Defibrillation",
    "domain": "neuroscience",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "a": 0.7,
    "b": 0.8,
    "epsilon": 0.06,
    "i_ext": 0.0,
    "d_u": 0.5,
    "dt": 0.15,
    "steps_per_frame": 4,
    "brush_radius": 4.0,
    "brush_current": 2.5,
    "clamp_voltage": 1.6,
    "propagation_mode": "FitzHughNagumo2D",
    "brush_mode": "InjectCurrent",
    "colormap": "HighContrast",
    "readout_bipolar": true,
    "active_lead": 0,
    "leads": [
      { "x": 17.28, "y": 19.2, "z": 8.0 },
      { "x": 78.72, "y": 19.2, "z": 8.0 },
      { "x": 48.0, "y": 82.56, "z": 8.0 }
    ],
    "scenario": "WholeSheetDefibrillation",
    "ablated_to_edge": false,
    "running": true
  }
}"#;

inventory::submit! {
    StarterPresetMetadata {
        id: "excitable_dual_edge_collision",
        title: "Dual-Edge Wavefront Collision",
        description: "Opposing 96×96 FitzHugh-Nagumo planar waves annihilate head-on via refractory recovery tails.",
        category: "Neuroscience",
        icon: "🌊",
        domain: "neuroscience",
        tool_name: Some("ExcitableSheetTool"),
        preset_generator: || PRESET_EXCITABLE_DUAL_EDGE,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "excitable_s1_s2_spiral",
        title: "S1-S2 Spiral Arrhythmia",
        description: "Planar S1 wave and timed vulnerable-tail S2 stimulus induce a self-sustaining spiral rotor.",
        category: "Neuroscience",
        icon: "🌀",
        domain: "neuroscience",
        tool_name: Some("ExcitableSheetTool"),
        preset_generator: || PRESET_EXCITABLE_S1_S2_SPIRAL,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "excitable_scar_pinning",
        title: "Scar-Pinned Rotor & Ablation",
        description: "Zero-flux ischemic scar anchors tachycardia until an ablation line connects to the boundary.",
        category: "Neuroscience",
        icon: "🫀",
        domain: "neuroscience",
        tool_name: Some("ExcitableSheetTool"),
        preset_generator: || PRESET_EXCITABLE_SCAR_PINNING,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "excitable_whole_sheet_defib",
        title: "Whole-Sheet Defibrillation",
        description: "Synchronous whole-sheet defibrillation shock restores resting membrane potential across 3 leads.",
        category: "Neuroscience",
        icon: "⚡",
        domain: "neuroscience",
        tool_name: Some("ExcitableSheetTool"),
        preset_generator: || PRESET_EXCITABLE_DEFIBRILLATION,
    }
}
