#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleTheoryHover;
use crate::framework::{Camera3D, InteractiveTool, Plot3DViewport};
use eframe::egui;
use egui_plot::{Line, PlotImage, PlotPoint, PlotPoints, Points};
use math_explorer::ai::gaussian_splatting::{
    projection, rendering, Gaussian2D, Gaussian3D, Scene,
};
use nalgebra::{Matrix2, Point2, Point3, UnitQuaternion, Vector3};

const RASTER_DIM: usize = 96;
const VIEWPORT_HALF_EXTENT: f64 = 3.0;
const CAMERA_DISTANCE: f64 = 8.0;
const ELLIPSE_SEGMENTS: usize = 24;
const ELLIPSE_POINTS: usize = ELLIPSE_SEGMENTS + 1;
const ELLIPSE_SIGMA: f64 = 1.5;

type RenderCacheKey = (u64, u64, u64, u64);
type SplatSpec = ([f64; 3], [f64; 3], [f64; 3], f64, [f64; 3]);

pub struct GaussianSplattingTool {
    pub scale: f64,
    pub camera: Camera3D,
    scene: Scene,
    projected_splats: Vec<Gaussian2D>,
    ellipse_rings: Vec<([[f64; 2]; ELLIPSE_POINTS], egui::Color32)>,
    splat_centers: Vec<[f64; 2]>,
    pixel_buffer: Vec<egui::Color32>,
    texture: Option<egui::TextureHandle>,
    cached_key: Option<RenderCacheKey>,
}

impl Default for GaussianSplattingTool {
    fn default() -> Self {
        let scene = build_default_scene();
        let count = scene.gaussians.len();
        Self {
            scale: 1.0,
            camera: Camera3D::new(0.5, 0.5, 1.0),
            scene,
            projected_splats: Vec::with_capacity(count),
            ellipse_rings: Vec::with_capacity(count),
            splat_centers: Vec::with_capacity(count),
            pixel_buffer: vec![egui::Color32::TRANSPARENT; RASTER_DIM * RASTER_DIM],
            texture: None,
            cached_key: None,
        }
    }
}

fn build_default_scene() -> Scene {
    let specs: [SplatSpec; 7] = [
        ([0.0, 0.0, 0.0], [0.42, 0.30, 0.26], [0.2, 0.4, 0.1], 0.90, [1.0, 0.92, 0.45]),
        ([1.25, 0.15, 0.25], [0.48, 0.20, 0.22], [0.0, 0.35, 0.6], 0.86, [1.0, 0.34, 0.28]),
        ([-1.15, -0.25, 0.35], [0.44, 0.22, 0.28], [0.5, -0.2, 0.4], 0.84, [0.25, 0.86, 1.0]),
        ([0.20, 1.20, -0.30], [0.24, 0.50, 0.22], [-0.4, 0.2, 0.3], 0.85, [0.36, 1.0, 0.54]),
        ([-0.25, -1.15, -0.20], [0.32, 0.44, 0.20], [0.3, 0.6, -0.3], 0.82, [1.0, 0.52, 0.84]),
        ([0.45, -0.35, 1.15], [0.26, 0.24, 0.48], [0.6, -0.4, 0.2], 0.86, [0.76, 0.44, 1.0]),
        ([-0.40, 0.45, -1.10], [0.36, 0.26, 0.42], [-0.3, 0.5, -0.5], 0.80, [1.0, 0.68, 0.22]),
    ];
    let gaussians = specs
        .into_iter()
        .map(|(m, s, euler, opacity, c)| Gaussian3D {
            mean: Point3::new(m[0], m[1], m[2]),
            scale: Vector3::new(s[0], s[1], s[2]),
            rotation: UnitQuaternion::from_euler_angles(euler[0], euler[1], euler[2]),
            opacity,
            color: Vector3::new(c[0], c[1], c[2]),
        })
        .collect();
    Scene { gaussians }
}

fn covariance_ellipse_ring(mean: Point2<f64>, cov2d: &Matrix2<f64>) -> [[f64; 2]; ELLIPSE_POINTS] {
    let s_xx = cov2d[(0, 0)];
    let s_xy = cov2d[(0, 1)];
    let s_yy = cov2d[(1, 1)];
    let half_trace = 0.5 * (s_xx + s_yy);
    let half_diff = 0.5 * (s_xx - s_yy);
    let disc = (half_diff * half_diff + s_xy * s_xy).max(0.0).sqrt();
    let major = ELLIPSE_SIGMA * (half_trace + disc).max(1e-6).sqrt();
    let minor = ELLIPSE_SIGMA * (half_trace - disc).max(1e-6).sqrt();
    let (sin_p, cos_p) = (0.5 * (2.0 * s_xy).atan2(s_xx - s_yy)).sin_cos();

    let mut ring = [[0.0_f64; 2]; ELLIPSE_POINTS];
    for (idx, slot) in ring.iter_mut().enumerate() {
        let theta = (idx as f64) * std::f64::consts::TAU / (ELLIPSE_SEGMENTS as f64);
        let (sin_t, cos_t) = theta.sin_cos();
        let lx = major * cos_t;
        let ly = minor * sin_t;
        *slot = [
            mean.x + lx * cos_p - ly * sin_p,
            mean.y + lx * sin_p + ly * cos_p,
        ];
    }
    ring
}

fn splat_stroke_color(rgb: &Vector3<f64>) -> egui::Color32 {
    let r = (rgb.x as f32).clamp(0.0, 1.0);
    let g = (rgb.y as f32).clamp(0.0, 1.0);
    let b = (rgb.z as f32).clamp(0.0, 1.0);
    egui::Color32::from(egui::Rgba::from_rgba_premultiplied(
        r * 0.85,
        g * 0.85,
        b * 0.85,
        0.85,
    ))
}

impl GaussianSplattingTool {
    fn render_key(&self) -> RenderCacheKey {
        (
            self.scale.to_bits(),
            self.camera.pitch.to_bits() as u64,
            self.camera.yaw.to_bits() as u64,
            self.camera.zoom.to_bits() as u64,
        )
    }

    fn refresh_projection_and_raster(&mut self, ctx: &egui::Context) {
        let key = self.render_key();
        if self.cached_key == Some(key) && self.texture.is_some() {
            return;
        }
        self.project_scene_gaussians();
        self.rasterize_splat_field();
        let image = egui::ColorImage {
            size: [RASTER_DIM, RASTER_DIM],
            source_size: egui::vec2(RASTER_DIM as f32, RASTER_DIM as f32),
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

    fn project_scene_gaussians(&mut self) {
        self.projected_splats.clear();
        self.ellipse_rings.clear();
        self.splat_centers.clear();

        let scale_factor = self.scale.clamp(0.1, 10.0);
        let view_matrix = self.camera.view_matrix(CAMERA_DISTANCE);
        let focal = CAMERA_DISTANCE * (self.camera.zoom as f64);

        for base in &self.scene.gaussians {
            let scaled = Gaussian3D {
                mean: base.mean,
                scale: base.scale * scale_factor,
                rotation: base.rotation,
                opacity: base.opacity,
                color: base.color,
            };
            let [tx, ty, vz] = self
                .camera
                .rotate_to_camera_space(&[scaled.mean.x, scaled.mean.y, scaled.mean.z]);
            let tz = (CAMERA_DISTANCE - vz).max(0.25);
            let cam_pos = Vector3::new(tx, ty, tz);
            let jacobian = projection::compute_jacobian(&cam_pos, focal, focal);
            let cov3d = scaled.compute_covariance();
            let cov2d = projection::project_covariance(&cov3d, &view_matrix, &jacobian)
                + Matrix2::from_diagonal_element(1e-4);
            let Some(inv_cov) = cov2d.try_inverse() else {
                continue;
            };
            let mean2d = Point2::new(focal * tx / tz, focal * ty / tz);
            self.ellipse_rings.push((
                covariance_ellipse_ring(mean2d, &cov2d),
                splat_stroke_color(&scaled.color),
            ));
            self.splat_centers.push([mean2d.x, mean2d.y]);
            self.projected_splats.push(Gaussian2D {
                mean: mean2d,
                conic: -0.5 * inv_cov,
                opacity: scaled.opacity,
                color: scaled.color,
                depth: tz,
            });
        }

        self.projected_splats
            .sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
    }

    fn rasterize_splat_field(&mut self) {
        let span = 2.0 * VIEWPORT_HALF_EXTENT;
        let inv_dim = 1.0 / (RASTER_DIM as f64);
        let (rows, _) = self.pixel_buffer.as_chunks_mut::<RASTER_DIM>();
        for (row_idx, row_pixels) in rows.iter_mut().enumerate() {
            let y = VIEWPORT_HALF_EXTENT - span * ((row_idx as f64 + 0.5) * inv_dim);
            for (col_idx, pixel) in row_pixels.iter_mut().enumerate() {
                let x = -VIEWPORT_HALF_EXTENT + span * ((col_idx as f64 + 0.5) * inv_dim);
                let rgb =
                    rendering::blend_gaussians(&self.projected_splats, &Point2::new(x, y));
                let r = (rgb.x as f32).clamp(0.0, 1.0);
                let g = (rgb.y as f32).clamp(0.0, 1.0);
                let b = (rgb.z as f32).clamp(0.0, 1.0);
                let alpha = r.max(g).max(b);
                *pixel = egui::Color32::from(egui::Rgba::from_rgba_premultiplied(r, g, b, alpha));
            }
        }
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "gaussian_splatting_controls");
        egui::SidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Gaussian Splatting");
            ui.separator();
            ui.add(egui::Slider::new(&mut self.scale, 0.1..=10.0).text("Scale"));

            ui.separator();
            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));

            ui.separator();
            ui.label(format!("Active 3D Splats: {}", self.scene.gaussians.len()));
            if let Some(front) = self.projected_splats.first() {
                ui.label(format!("Nearest Depth: {:.2}", front.depth));
            }
        });
    }

    fn draw_viewport(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.refresh_projection_and_raster(ui.ctx());
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let origin = self.camera.project(&[0.0, 0.0, 0.0]);
            let axes = [
                (self.camera.project(&[1.6, 0.0, 0.0]), palette.axis_x, "X Axis"),
                (self.camera.project(&[0.0, 1.6, 0.0]), palette.axis_y, "Y Axis"),
                (self.camera.project(&[0.0, 0.0, 1.6]), palette.axis_z, "Z Axis"),
            ];
            let tex_id = self.texture.as_ref().map(egui::TextureHandle::id);
            let extent_size = (2.0 * VIEWPORT_HALF_EXTENT) as f32;
            let mut camera = self.camera;

            let response = Plot3DViewport::new("gaussian_splatting_plot", VIEWPORT_HALF_EXTENT)
                .show(ui, &mut camera, |plot_ui| {
                    if let Some(tid) = tex_id {
                        plot_ui.image(PlotImage::new(
                            "Radiance Field",
                            tid,
                            PlotPoint::new(0.0, 0.0),
                            egui::vec2(extent_size, extent_size),
                        ));
                    }
                    for (end, color, label) in axes {
                        plot_ui.line(
                            Line::new(label, PlotPoints::new(vec![origin, end]))
                                .color(color)
                                .width(1.5_f32),
                        );
                    }
                    for (idx, (ring, color)) in self.ellipse_rings.iter().enumerate() {
                        plot_ui.line(
                            Line::new(
                                format!("Splat {idx} (1.5σ)"),
                                PlotPoints::new(ring.to_vec()),
                            )
                            .color(*color)
                            .width(1.25_f32),
                        );
                    }
                    plot_ui.points(
                        Points::new("Splat Means", self.splat_centers.clone())
                            .radius(3.0_f32)
                            .color(palette.accent),
                    );
                })
                .response;

            self.camera = camera;
            response.accessible_theory_hover(self);
        });
    }
}

#[derive(serde::Serialize)]
struct GaussianSplattingSaveState {
    scale: f64,
    camera: Camera3D,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GaussianSplattingState {
    scale: f64,
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
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Gaussian Splatting"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }
    fn domain(&self) -> &'static str {
        "ai"
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.scale.to_bits().hash(&mut h);
        self.camera.pitch.to_bits().hash(&mut h);
        self.camera.yaw.to_bits().hash(&mut h);
        self.camera.zoom.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.scale.is_finite() && (0.1..=10.0).contains(&self.scale) && self.camera.is_valid()
    }

    fn clamp_parameters(&mut self) -> bool {
        self.scale = if self.scale.is_finite() {
            self.scale.clamp(0.1, 10.0)
        } else {
            1.0
        };
        if !self.camera.is_valid() {
            self.camera.reset();
        }
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&GaussianSplattingSaveState {
            scale: self.scale,
            camera: self.camera,
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
        let camera = Camera3D::from_preset_fields(
            parsed.camera,
            parsed.pitch,
            parsed.yaw,
            parsed.zoom,
            Some(Camera3D::default()),
        )?;
        self.scale = parsed.scale;
        self.camera.apply_pose(camera);
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

impl scientific_metadata::theory::TheoryDescribable for GaussianSplattingTool {
    fn theory_description(&self) -> String {
        "3D Gaussian Splatting is a rasterization-based technique for real-time rendering of radiance fields, projecting anisotropic 3D Gaussians via EWA covariance Jacobians and front-to-back alpha blending.".into()
    }
    fn phonetic_description(&self) -> String {
        "3D Gaussian Splatting is a rasterization-based technique for real-time rendering of radiance fields, using explicit 3D Gaussians to represent scenes instead of implicit neural representations.".into()
    }
    fn theory_citation(&self) -> String {
        "Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics.".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "scale".into(),
            "Uniform multiplier applied to the 3D Gaussian covariance principal semi-axes.".into(),
        );
        map
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
