#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "gaussian_splatting_scene.rs"]
mod scene;
#[path = "gaussian_splatting_ui.rs"]
mod ui;

#[path = "gaussian_splatting_export.rs"]
mod export;

#[cfg(test)]
use crate::framework::ProjectionMode;
use crate::framework::{Camera3D, ExportCapabilities, InteractiveTool, Scene3DBuilder};
use eframe::egui;
use math_explorer::ai::gaussian_splatting::{Gaussian2D, Scene};
use nalgebra::Point3;
#[cfg(test)]
use scene::evaluate_density_counts_with_thresholds;
use scene::{
    build_preset_scene_and_gradients, evaluate_scene_density_counts_with_thresholds,
    project_single_splat, rasterize_splats_into, VolumetricSplatPreset, CAMERA_DISTANCE,
    DEFAULT_GRAD_THRESHOLD, DEFAULT_OPACITY_THRESHOLD, ELLIPSE_POINTS, GRAD_THRESHOLD_MAX,
    GRAD_THRESHOLD_MIN, OPACITY_THRESHOLD_MAX, OPACITY_THRESHOLD_MIN, VIEWPORT_HALF_EXTENT,
};
use scientific_metadata::theory::TheoryDescribable;

pub use scene::DensityActionCounts;
#[cfg(test)]
pub(crate) use scene::SPLAT_GRID_SIZE;
#[cfg(not(test))]
use scene::SPLAT_GRID_SIZE;

type RenderCacheKey = (u64, u64, u64, u64);

const fn default_grad_threshold() -> f64 {
    DEFAULT_GRAD_THRESHOLD
}

const fn default_opacity_threshold() -> f64 {
    DEFAULT_OPACITY_THRESHOLD
}

/// Interactive 3D Gaussian Splatting radiance field viewer built on `Plot3DViewport`.
pub struct GaussianSplattingTool {
    /// Uniform multiplier applied to 3D Gaussian principal semi-axes.
    pub scale: f64,
    /// Positional gradient threshold (`tau_pos`) separating converged splats from densification.
    pub grad_threshold: f64,
    /// Minimum opacity threshold below which nearly transparent splats are pruned.
    pub opacity_threshold: f64,
    /// Shared 3D orbit, pan, projection, and zoom camera state.
    pub camera: Camera3D,
    /// Active volumetric scene preset (canonical 8-splat or dynamic orbital/Turing volumes).
    pub scene_preset: VolumetricSplatPreset,
    pub(super) frustum_half_extents: (f64, f64),
    pub(super) scene: Scene,
    pub(super) grad_magnitudes: Vec<f64>,
    pub(super) projected_splats: Vec<Gaussian2D>,
    pub(super) ellipse_rings: Vec<([[f64; 2]; ELLIPSE_POINTS], egui::Color32)>,
    pub(super) ellipse_rings_3d: Vec<([Point3<f64>; ELLIPSE_POINTS], egui::Color32)>,
    pub(super) splat_centers: Vec<[f64; 2]>,
    pub(super) world_means: Vec<Point3<f64>>,
    pub(super) scene_builder: Scene3DBuilder,
    pub(super) pixel_buffer: Vec<egui::Color32>,
    pub(super) texture: Option<egui::TextureHandle>,
    cached_key: Option<RenderCacheKey>,
    pub(super) cached_counts: DensityActionCounts,
}

impl Default for GaussianSplattingTool {
    fn default() -> Self {
        let scale = 1.0;
        let grad_threshold = DEFAULT_GRAD_THRESHOLD;
        let opacity_threshold = DEFAULT_OPACITY_THRESHOLD;
        let scene_preset = VolumetricSplatPreset::Canonical;
        let (scene, grad_magnitudes) = build_preset_scene_and_gradients(scene_preset);
        let count = scene.gaussians.len();
        let cached_counts = evaluate_scene_density_counts_with_thresholds(
            &scene,
            &grad_magnitudes,
            scale,
            grad_threshold,
            opacity_threshold,
        );
        let camera = Camera3D::new(0.5, 0.5, 1.0).with_perspective_distance(CAMERA_DISTANCE);
        let world_means = scene.gaussians.iter().map(|g| g.mean).collect();
        Self {
            scale,
            grad_threshold,
            opacity_threshold,
            camera,
            scene_preset,
            frustum_half_extents: (VIEWPORT_HALF_EXTENT, VIEWPORT_HALF_EXTENT),
            scene,
            grad_magnitudes,
            projected_splats: Vec::with_capacity(count),
            ellipse_rings: Vec::with_capacity(count),
            ellipse_rings_3d: Vec::with_capacity(count),
            splat_centers: Vec::with_capacity(count),
            world_means,
            scene_builder: Scene3DBuilder::new(camera),
            pixel_buffer: vec![egui::Color32::TRANSPARENT; SPLAT_GRID_SIZE * SPLAT_GRID_SIZE],
            texture: None,
            cached_key: None,
            cached_counts,
        }
    }
}

impl GaussianSplattingTool {
    /// Evaluates `determine_density_action` across the canonical 3D scene for a given `scale`.
    #[cfg(test)]
    pub fn compute_density_counts_for_scale(scale: f64) -> DensityActionCounts {
        evaluate_density_counts_with_thresholds(
            scale,
            DEFAULT_GRAD_THRESHOLD,
            DEFAULT_OPACITY_THRESHOLD,
        )
    }

    /// Evaluates `determine_density_action` for `scale`, `grad_threshold`, and `opacity_threshold`.
    #[cfg(test)]
    #[must_use]
    pub fn compute_density_counts(
        scale: f64,
        grad_threshold: f64,
        opacity_threshold: f64,
    ) -> DensityActionCounts {
        evaluate_density_counts_with_thresholds(scale, grad_threshold, opacity_threshold)
    }

    pub(super) fn sync_density_counts(&mut self) {
        self.cached_counts = evaluate_scene_density_counts_with_thresholds(
            &self.scene,
            &self.grad_magnitudes,
            self.scale,
            self.grad_threshold,
            self.opacity_threshold,
        );
    }

    /// Switches the active volumetric `Gaussian3D` scene preset and refreshes density metrics.
    pub fn set_scene_preset(&mut self, preset: VolumetricSplatPreset) {
        if self.scene_preset == preset && !self.scene.gaussians.is_empty() {
            return;
        }
        self.scene_preset = preset;
        let (scene, grads) = build_preset_scene_and_gradients(preset);
        self.world_means = scene.gaussians.iter().map(|g| g.mean).collect();
        self.scene = scene;
        self.grad_magnitudes = grads;
        self.sync_density_counts();
        self.cached_key = None;
    }

    /// Projects a world `Point3<f64>` using the active splat & reference-axis camera projection.
    #[must_use]
    pub fn project_splat_point(&self, world: Point3<f64>) -> [f64; 2] {
        self.camera.project(&[world.x, world.y, world.z])
    }

    /// Renders the projected 3D Gaussian radiance field at the current `scale`, `camera`, and frustum.
    #[cfg(test)]
    pub fn render_splat_image(&self) -> egui::ColorImage {
        let mut projected = Vec::with_capacity(self.scene.gaussians.len());
        let mut rings = Vec::new();
        let mut rings_3d = Vec::new();
        let mut centers = Vec::new();
        self.project_into(&mut projected, &mut rings, &mut rings_3d, &mut centers);
        let mut pixels = vec![egui::Color32::TRANSPARENT; SPLAT_GRID_SIZE * SPLAT_GRID_SIZE];
        let (x_half, y_half) = self.frustum_half_extents;
        rasterize_splats_into(&projected, x_half, y_half, &mut pixels);
        egui::ColorImage {
            size: [SPLAT_GRID_SIZE, SPLAT_GRID_SIZE],
            source_size: egui::vec2(SPLAT_GRID_SIZE as f32, SPLAT_GRID_SIZE as f32),
            pixels,
        }
    }

    fn render_key(&self) -> RenderCacheKey {
        use std::hash::{Hash, Hasher};
        let mut cam_hasher = std::collections::hash_map::DefaultHasher::new();
        self.camera.hash_state(&mut cam_hasher);
        self.scene_preset.hash(&mut cam_hasher);
        let (x_half, y_half) = self.frustum_half_extents;
        (
            self.scale.to_bits(),
            cam_hasher.finish(),
            x_half.to_bits(),
            y_half.to_bits(),
        )
    }

    pub(super) fn refresh_projection_and_raster(&mut self, ctx: &egui::Context) {
        self.sync_density_counts();
        let key = self.render_key();
        if self.cached_key == Some(key) && self.texture.is_some() {
            return;
        }
        let mut splats = std::mem::take(&mut self.projected_splats);
        let mut rings = std::mem::take(&mut self.ellipse_rings);
        let mut rings_3d = std::mem::take(&mut self.ellipse_rings_3d);
        let mut centers = std::mem::take(&mut self.splat_centers);
        self.project_into(&mut splats, &mut rings, &mut rings_3d, &mut centers);
        let (x_half, y_half) = self.frustum_half_extents;
        rasterize_splats_into(&splats, x_half, y_half, &mut self.pixel_buffer);
        self.projected_splats = splats;
        self.ellipse_rings = rings;
        self.ellipse_rings_3d = rings_3d;
        self.splat_centers = centers;

        let image = egui::ColorImage {
            size: [SPLAT_GRID_SIZE, SPLAT_GRID_SIZE],
            source_size: egui::vec2(SPLAT_GRID_SIZE as f32, SPLAT_GRID_SIZE as f32),
            pixels: self.pixel_buffer.clone(),
        };
        if let Some(tex) = &mut self.texture {
            tex.set(image, egui::TextureOptions::LINEAR);
        } else {
            self.texture = Some(ctx.load_texture(
                "gaussian_splatting_radiance_field",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
        self.cached_key = Some(key);
    }

    fn project_into(
        &self,
        out_splats: &mut Vec<Gaussian2D>,
        out_rings: &mut Vec<([[f64; 2]; ELLIPSE_POINTS], egui::Color32)>,
        out_rings_3d: &mut Vec<([Point3<f64>; ELLIPSE_POINTS], egui::Color32)>,
        out_centers: &mut Vec<[f64; 2]>,
    ) {
        out_splats.clear();
        out_rings.clear();
        out_rings_3d.clear();
        out_centers.clear();

        let scale_factor = if self.scale.is_finite() {
            self.scale.clamp(0.1, 10.0)
        } else {
            1.0
        };
        let dist = self.camera.perspective_distance();
        let view_matrix = self.camera.view_matrix(dist);
        for base in &self.scene.gaussians {
            let want_ring = out_rings.len() < 48;
            if let Some((splat_2d, maybe_ring)) =
                project_single_splat(self, base, scale_factor, dist, &view_matrix, want_ring)
            {
                out_centers.push([splat_2d.mean.x, splat_2d.mean.y]);
                if let Some((ring_2d, ring_3d, color)) = maybe_ring {
                    out_rings.push((ring_2d, color));
                    out_rings_3d.push((ring_3d, color));
                }
                out_splats.push(splat_2d);
            }
        }
        out_splats.sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
    }
}

#[derive(serde::Serialize)]
struct GaussianSplattingSaveState {
    scale: f64,
    grad_threshold: f64,
    opacity_threshold: f64,
    camera: Camera3D,
    #[serde(skip_serializing_if = "VolumetricSplatPreset::is_canonical")]
    scene_preset: VolumetricSplatPreset,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GaussianSplattingState {
    scale: f64,
    #[serde(default = "default_grad_threshold")]
    grad_threshold: f64,
    #[serde(default = "default_opacity_threshold")]
    opacity_threshold: f64,
    #[serde(default)]
    scene_preset: VolumetricSplatPreset,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    pitch: Option<f32>,
    #[serde(default)]
    yaw: Option<f32>,
    #[serde(default)]
    zoom: Option<f32>,
}

impl InteractiveTool for GaussianSplattingTool {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Gaussian Splatting"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::splat_ply_only()
    }
    fn export_splat_ply_3d(&self) -> Option<Result<Vec<u8>, String>> {
        Some(self.serialize_splats_ply_binary())
    }
    fn domain(&self) -> &'static str {
        "ai"
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.scale.to_bits().hash(&mut h);
        self.grad_threshold.to_bits().hash(&mut h);
        self.opacity_threshold.to_bits().hash(&mut h);
        self.scene_preset.hash(&mut h);
        self.camera.hash_state(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.scale.is_finite()
            && (0.1..=10.0).contains(&self.scale)
            && self.grad_threshold.is_finite()
            && (GRAD_THRESHOLD_MIN..=GRAD_THRESHOLD_MAX).contains(&self.grad_threshold)
            && self.opacity_threshold.is_finite()
            && (OPACITY_THRESHOLD_MIN..=OPACITY_THRESHOLD_MAX).contains(&self.opacity_threshold)
            && self.camera.is_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.scale = if self.scale.is_finite() {
            self.scale.clamp(0.1, 10.0)
        } else {
            1.0
        };
        self.grad_threshold = if self.grad_threshold.is_finite() {
            self.grad_threshold
                .clamp(GRAD_THRESHOLD_MIN, GRAD_THRESHOLD_MAX)
        } else {
            DEFAULT_GRAD_THRESHOLD
        };
        self.opacity_threshold = if self.opacity_threshold.is_finite() {
            self.opacity_threshold
                .clamp(OPACITY_THRESHOLD_MIN, OPACITY_THRESHOLD_MAX)
        } else {
            DEFAULT_OPACITY_THRESHOLD
        };
        if !self.camera.is_valid() {
            self.camera.reset();
        }
        self.sync_density_counts();
        self.cached_key = None;
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&GaussianSplattingSaveState {
            scale: self.scale,
            grad_threshold: self.grad_threshold,
            opacity_threshold: self.opacity_threshold,
            camera: self.camera,
            scene_preset: self.scene_preset,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: GaussianSplattingState =
            serde_json::from_str(state).map_err(|e| format!("Invalid splatting state: {e}"))?;
        if !parsed.scale.is_finite() || !(0.1..=10.0).contains(&parsed.scale) {
            return Err("Gaussian splat scale must be finite within [0.1, 10.0]".into());
        }
        if !parsed.grad_threshold.is_finite()
            || !(GRAD_THRESHOLD_MIN..=GRAD_THRESHOLD_MAX).contains(&parsed.grad_threshold)
        {
            return Err("Gradient threshold out of valid bounds".into());
        }
        if !parsed.opacity_threshold.is_finite()
            || !(OPACITY_THRESHOLD_MIN..=OPACITY_THRESHOLD_MAX).contains(&parsed.opacity_threshold)
        {
            return Err("Opacity threshold out of valid bounds".into());
        }
        let camera = Camera3D::from_preset_fields(
            parsed.camera,
            parsed.pitch,
            parsed.yaw,
            parsed.zoom,
            Some(Camera3D::default()),
        )?;
        self.scale = parsed.scale;
        self.grad_threshold = parsed.grad_threshold;
        self.opacity_threshold = parsed.opacity_threshold;
        self.camera.apply_pose(camera);
        self.set_scene_preset(parsed.scene_preset);
        self.sync_density_counts();
        self.cached_key = None;
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.draw_side_panel(parent_ui);
        self.draw_viewport(parent_ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "Gaussian Splatting",
        display_name: "Gaussian Splatting",
        domain: "ai",
        tags: &["ai", "graphics", "3d"],
        build: || Box::new(GaussianSplattingTool::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Gaussian Splatting",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "scale": 1.5,
    "camera": {
      "pitch": 0.45,
      "yaw": 0.65,
      "zoom": 1.15
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gaussian_splatting_starter",
        title: "Gaussian Splatting",
        description: "Interactive 3D Gaussian radiance field splatting with EWA covariance projection and alpha blending.",
        category: "Neural Rendering",
        icon: "🎨",
        domain: "ai",
        tool_name: Some("Gaussian Splatting"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:gaussian_splatting]

#[cfg(test)]
#[path = "gaussian_splatting_tests.rs"]
mod tests;
