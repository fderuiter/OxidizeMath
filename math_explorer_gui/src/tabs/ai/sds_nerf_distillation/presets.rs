#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    SdsNeRfDistillationTool, SdsOptimizerKind, SdsPromptPreset, CFG_MAX, CFG_MIN, LR_MAX, LR_MIN,
    MAX_RAY_GRID, MAX_RAY_SAMPLES, MIN_RAY_GRID, MIN_RAY_SAMPLES,
};
use crate::framework::{Camera3D, InteractiveTool};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SdsDistillationState {
    #[serde(default)]
    pub prompt_preset: SdsPromptPreset,
    #[serde(default)]
    pub optimizer_kind: SdsOptimizerKind,
    pub guidance_scale: f64,
    pub learning_rate: f64,
    pub ray_grid_size: usize,
    pub samples_per_ray: usize,
    #[serde(default)]
    pub camera: Option<Camera3D>,
}

impl SdsNeRfDistillationTool {
    pub(super) fn encode_state_json(&self) -> Option<String> {
        serde_json::to_string(&SdsDistillationState {
            prompt_preset: self.prompt_preset,
            optimizer_kind: self.optimizer_kind,
            guidance_scale: self.guidance_scale,
            learning_rate: self.learning_rate,
            ray_grid_size: self.ray_grid_size,
            samples_per_ray: self.samples_per_ray,
            camera: Some(self.camera),
        })
        .ok()
    }

    pub(super) fn decode_state_json(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: SdsDistillationState =
            serde_json::from_str(state).map_err(|e| format!("Invalid SDS NeRF state: {e}"))?;
        if !parsed.guidance_scale.is_finite()
            || !(CFG_MIN..=CFG_MAX).contains(&parsed.guidance_scale)
        {
            return Err("guidance_scale out of bounds".into());
        }
        if !parsed.learning_rate.is_finite() || !(LR_MIN..=LR_MAX).contains(&parsed.learning_rate) {
            return Err("learning_rate out of bounds".into());
        }
        if !(MIN_RAY_GRID..=MAX_RAY_GRID).contains(&parsed.ray_grid_size) {
            return Err("ray_grid_size must be within [8, 32]".into());
        }
        if !(MIN_RAY_SAMPLES..=MAX_RAY_SAMPLES).contains(&parsed.samples_per_ray) {
            return Err("samples_per_ray must be within [6, 24]".into());
        }
        let cam = Camera3D::from_preset_fields(
            parsed.camera,
            None,
            None,
            None,
            Some(Camera3D::new(0.45, 0.55, 1.0).with_perspective_distance(3.2)),
        )?;
        self.prompt_preset = parsed.prompt_preset;
        self.optimizer_kind = parsed.optimizer_kind;
        self.guidance_scale = parsed.guidance_scale;
        self.learning_rate = parsed.learning_rate;
        self.ray_grid_size = parsed.ray_grid_size;
        self.samples_per_ray = parsed.samples_per_ray;
        self.camera.apply_pose(cam);
        self.is_playing = false;
        self.reset_distillation();
        Ok(())
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SdsNeRfDistillationTool",
        display_name: "SDS NeRF Distillation",
        domain: "ai",
        tags: &["ai", "nerf", "sds", "diffusion", "3d"],
        build: || Box::new(SdsNeRfDistillationTool::default()),
    }
}

const SDS_TORUS_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "SDS Amber Torus Distillation",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "prompt_preset": "amber_torus",
    "optimizer_kind": "adam",
    "guidance_scale": 7.5,
    "learning_rate": 0.15,
    "ray_grid_size": 20,
    "samples_per_ray": 14,
    "camera": {
      "pitch": 0.45,
      "yaw": 0.6,
      "zoom": 1.05
    }
  }
}"#;

const SDS_OCTAHEDRON_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "SDS Crystal Octahedron Momentum",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "prompt_preset": "cyan_octahedron",
    "optimizer_kind": "sgd",
    "guidance_scale": 8.0,
    "learning_rate": 0.28,
    "ray_grid_size": 20,
    "samples_per_ray": 14,
    "camera": {
      "pitch": 0.52,
      "yaw": -0.45,
      "zoom": 1.1
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sds_amber_torus_adam",
        title: "SDS Amber Torus Distillation",
        description: "Text-to-3D Score Distillation Sampling sculpting a volumetric voxel NeRF torus with Classifier-Free Guidance and Adam.",
        category: "Neural Rendering",
        icon: "🍩",
        domain: "ai",
        tool_name: Some("SdsNeRfDistillationTool"),
        preset_generator: || SDS_TORUS_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sds_crystal_octahedron_sgd",
        title: "SDS Crystal Octahedron Momentum",
        description: "Momentum SGD volumetric score distillation carving a crystalline octahedron radiance field from 2D diffusion score residuals.",
        category: "Neural Rendering",
        icon: "💎",
        domain: "ai",
        tool_name: Some("SdsNeRfDistillationTool"),
        preset_generator: || SDS_OCTAHEDRON_PRESET_JSON,
    }
}
