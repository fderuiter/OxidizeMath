use crate::accessibility::AccessibleTheoryHover;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use math_explorer::ai::gaussian_splatting::{
    optimization::{determine_density_action, DensityAction},
    projection::{compute_jacobian, project_covariance},
    rendering::blend_gaussians,
    Gaussian2D, Gaussian3D,
};
use nalgebra::{Matrix4, Point2, Point3, UnitQuaternion, Vector3};
use scientific_metadata::theory::TheoryDescribable;

const SPLAT_GRID_SIZE: usize = 96;
const FOCAL_LENGTH_PX: f64 = 64.0;
const GRAD_THRESHOLD: f64 = 0.10;
const SCALE_THRESHOLD: f64 = 0.45;
const OPACITY_THRESHOLD: f64 = 0.05;

struct SplatSpec {
    position: [f64; 3],
    base_scale: [f64; 3],
    axis_angle: [f64; 3],
    opacity: f64,
    color: [f64; 3],
    grad_magnitude: f64,
}

const CANONICAL_SCENE: [SplatSpec; 8] = [
    SplatSpec {
        position: [-0.65, 0.45, 3.4],
        base_scale: [0.28, 0.16, 0.20],
        axis_angle: [0.0, 0.0, 0.45],
        opacity: 0.90,
        color: [0.96, 0.32, 0.28],
        grad_magnitude: 0.24,
    },
    SplatSpec {
        position: [0.60, 0.42, 3.6],
        base_scale: [0.18, 0.30, 0.22],
        axis_angle: [0.2, -0.1, -0.55],
        opacity: 0.86,
        color: [0.22, 0.82, 0.56],
        grad_magnitude: 0.19,
    },
    SplatSpec {
        position: [0.0, -0.15, 3.1],
        base_scale: [0.36, 0.24, 0.26],
        axis_angle: [0.15, 0.25, 0.10],
        opacity: 0.82,
        color: [0.28, 0.58, 0.98],
        grad_magnitude: 0.31,
    },
    SplatSpec {
        position: [-0.48, -0.58, 4.0],
        base_scale: [0.58, 0.32, 0.28],
        axis_angle: [-0.2, 0.1, 0.35],
        opacity: 0.78,
        color: [0.95, 0.76, 0.22],
        grad_magnitude: 0.16,
    },
    SplatSpec {
        position: [0.55, -0.52, 4.2],
        base_scale: [0.25, 0.22, 0.20],
        axis_angle: [0.0, 0.2, -0.30],
        opacity: 0.80,
        color: [0.78, 0.36, 0.94],
        grad_magnitude: 0.06,
    },
    SplatSpec {
        position: [0.02, 0.68, 4.5],
        base_scale: [0.34, 0.20, 0.24],
        axis_angle: [0.1, -0.15, 0.05],
        opacity: 0.74,
        color: [0.20, 0.86, 0.90],
        grad_magnitude: 0.04,
    },
    SplatSpec {
        position: [-0.15, 0.08, 4.9],
        base_scale: [0.65, 0.52, 0.40],
        axis_angle: [0.05, 0.1, -0.20],
        opacity: 0.52,
        color: [0.92, 0.46, 0.68],
        grad_magnitude: 0.07,
    },
    SplatSpec {
        position: [0.85, 0.82, 5.4],
        base_scale: [0.22, 0.18, 0.18],
        axis_angle: [0.0, 0.0, 0.0],
        opacity: 0.02,
        color: [0.50, 0.55, 0.65],
        grad_magnitude: 0.14,
    },
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DensityActionCounts {
    pub split: usize,
    pub clone: usize,
    pub keep: usize,
    pub prune: usize,
}

#[cfg_attr(any(), verified(opt_out = "gui_tool"))]
pub struct GaussianSplattingTool {
    pub scale: f64,
    texture: Option<egui::TextureHandle>,
    cached_scale_bits: Option<u64>,
    cached_counts: DensityActionCounts,
}

impl Default for GaussianSplattingTool {
    fn default() -> Self {
        let scale = 1.0;
        Self {
            scale,
            texture: None,
            cached_scale_bits: None,
            cached_counts: Self::compute_density_counts_for_scale(scale),
        }
    }
}

impl GaussianSplattingTool {
    pub fn compute_density_counts_for_scale(scale: f64) -> DensityActionCounts {
        let safe_scale = if scale.is_finite() {
            scale.clamp(0.1, 10.0)
        } else {
            1.0
        };
        let mut counts = DensityActionCounts::default();
        for spec in &CANONICAL_SCENE {
            let max_base = spec.base_scale[0]
                .max(spec.base_scale[1])
                .max(spec.base_scale[2]);
            let action = determine_density_action(
                spec.grad_magnitude,
                max_base * safe_scale,
                spec.opacity,
                GRAD_THRESHOLD,
                SCALE_THRESHOLD,
                OPACITY_THRESHOLD,
            );
            match action {
                DensityAction::Split => counts.split += 1,
                DensityAction::Clone => counts.clone += 1,
                DensityAction::Keep => counts.keep += 1,
                DensityAction::Prune => counts.prune += 1,
            }
        }
        counts
    }

    fn project_scene(&self) -> Vec<Gaussian2D> {
        let safe_scale = if self.scale.is_finite() {
            self.scale.clamp(0.1, 10.0)
        } else {
            1.0
        };
        let view_matrix = Matrix4::identity();
        let center = SPLAT_GRID_SIZE as f64 * 0.5;
        let mut projected = Vec::with_capacity(CANONICAL_SCENE.len());

        for spec in &CANONICAL_SCENE {
            let g3d = Gaussian3D {
                mean: Point3::new(spec.position[0], spec.position[1], spec.position[2]),
                scale: Vector3::new(
                    spec.base_scale[0] * safe_scale,
                    spec.base_scale[1] * safe_scale,
                    spec.base_scale[2] * safe_scale,
                ),
                rotation: UnitQuaternion::new(Vector3::new(
                    spec.axis_angle[0],
                    spec.axis_angle[1],
                    spec.axis_angle[2],
                )),
                opacity: spec.opacity,
                color: Vector3::new(spec.color[0], spec.color[1], spec.color[2]),
            };
            let t = g3d.mean.coords;
            let cov3d = g3d.compute_covariance();
            let jacobian = compute_jacobian(&t, FOCAL_LENGTH_PX, FOCAL_LENGTH_PX);
            let mut cov2d = project_covariance(&cov3d, &view_matrix, &jacobian);
            cov2d[(0, 0)] += 0.3;
            cov2d[(1, 1)] += 0.3;
            let Some(inv_cov) = cov2d.try_inverse() else {
                continue;
            };
            projected.push(Gaussian2D {
                mean: Point2::new(
                    center + FOCAL_LENGTH_PX * t.x / t.z,
                    center - FOCAL_LENGTH_PX * t.y / t.z,
                ),
                conic: -0.5 * inv_cov,
                opacity: g3d.opacity,
                color: g3d.color,
                depth: t.z,
            });
        }
        projected.sort_by(|a, b| a.depth.total_cmp(&b.depth));
        projected
    }

    pub fn render_splat_image(&self) -> egui::ColorImage {
        let splats = self.project_scene();
        let mut pixels =
            vec![egui::Color32::from_rgb(12, 15, 22); SPLAT_GRID_SIZE * SPLAT_GRID_SIZE];

        for (row_idx, row) in pixels.as_chunks_mut::<SPLAT_GRID_SIZE>().0.iter_mut().enumerate() {
            let py = row_idx as f64 + 0.5;
            for (col_idx, px) in row.iter_mut().enumerate() {
                let pt = Point2::new(col_idx as f64 + 0.5, py);
                let c = blend_gaussians(&splats, &pt);
                let r = (c.x.clamp(0.0, 1.0) * 243.0 + 12.0).round() as u8;
                let g = (c.y.clamp(0.0, 1.0) * 240.0 + 15.0).round() as u8;
                let b = (c.z.clamp(0.0, 1.0) * 233.0 + 22.0).round() as u8;
                *px = egui::Color32::from_rgb(r, g, b);
            }
        }
        egui::ColorImage::new([SPLAT_GRID_SIZE, SPLAT_GRID_SIZE], pixels)
    }

    fn ensure_cached_texture(&mut self, ctx: &egui::Context) {
        let bits = self.scale.to_bits();
        if self.texture.is_none() || self.cached_scale_bits != Some(bits) {
            let image = self.render_splat_image();
            self.texture = Some(ctx.load_texture(
                "gaussian_splatting_raster",
                image,
                egui::TextureOptions::LINEAR,
            ));
            self.cached_counts = Self::compute_density_counts_for_scale(self.scale);
            self.cached_scale_bits = Some(bits);
        }
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "gaussian_splatting_controls");
        egui::SidePanel::right(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Controls");
            ui.separator();
            if ui
                .add(egui::Slider::new(&mut self.scale, 0.1..=10.0).text("Scale"))
                .changed()
            {
                self.cached_counts = Self::compute_density_counts_for_scale(self.scale);
            }

            ui.separator();
            ui.heading("Adaptive Density Control");
            ui.label(format!("Split (large grad & scale): {}", self.cached_counts.split));
            ui.label(format!("Clone (large grad, small scale): {}", self.cached_counts.clone));
            ui.label(format!("Keep (converged): {}", self.cached_counts.keep));
            ui.label(format!("Prune (opacity < {:.2}): {}", OPACITY_THRESHOLD, self.cached_counts.prune));
            ui.separator();
            ui.small(format!(
                "Raster grid: {SPLAT_GRID_SIZE}×{SPLAT_GRID_SIZE} px ({} Gaussians)",
                CANONICAL_SCENE.len()
            ));
        });
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GaussianSplattingState {
    scale: f64,
}

impl InteractiveTool for GaussianSplattingTool {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Gaussian Splatting"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::none()
    }
    fn domain(&self) -> &'static str {
        "ai"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&GaussianSplattingState { scale: self.scale }).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: GaussianSplattingState =
            serde_json::from_str(state).map_err(|e| format!("Invalid splatting state: {e}"))?;
        if !parsed.scale.is_finite() || !(0.1..=10.0).contains(&parsed.scale) {
            return Err("Gaussian splat scale must be finite within [0.1, 10.0]".into());
        }
        self.scale = parsed.scale;
        self.cached_counts = Self::compute_density_counts_for_scale(self.scale);
        self.cached_scale_bits = None;
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        self.draw_side_panel(parent_ui);
        self.ensure_cached_texture(&ctx);

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("3D Gaussian Splatting Visualization");
            if let Some(texture) = &self.texture {
                let avail = ui.available_size();
                let side = avail.x.min(avail.y).clamp(96.0, 420.0);
                let alt_text = self.phonetic_description();
                let response = ui
                    .add(
                        egui::Image::new((texture.id(), egui::vec2(side, side)))
                            .alt_text(alt_text)
                            .sense(egui::Sense::focusable_noninteractive()),
                    )
                    .accessible_theory_hover(self);
                crate::accessibility::paint_canvas_focus_ring(ui, &response);
            }
        });
    }
}

impl TheoryDescribable for GaussianSplattingTool {
    fn theory_description(&self) -> String {
        format!(
            "3D Gaussian Splatting projects anisotropic 3D Gaussians (scale multiplier {:.2}) via Jacobian covariance $\\Sigma' = J W \\Sigma W^T J^T$ and front-to-back alpha compositing $C = \\sum_i c_i \\alpha_i \\prod_{{j < i}} (1 - \\alpha_j)$.",
            self.scale
        )
    }
    fn phonetic_description(&self) -> String {
        format!(
            "3D Gaussian Splatting radiance field rasterization at scale {:.2} with covariance projection and front-to-back alpha blending.",
            self.scale
        )
    }
    fn theory_citation(&self) -> String {
        "Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics.".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "scale".into(),
            "Multiplier applied to 3D Gaussian principal axis scales prior to EWA covariance projection.".into(),
        );
        map
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "Gaussian Splatting",
        display_name: "Gaussian Splatting",
        domain: "ai",
        tags: &["ai", "graphics"],
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
    "scale": 1.5
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gaussian_splatting_starter",
        title: "Gaussian Splatting",
        description: "3D Gaussian radiance field scale parameter control and theoretical model overview.",
        category: "Neural Rendering",
        icon: "🎨",
        domain: "ai",
        tool_name: Some("Gaussian Splatting"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_splatting_image_and_density_respond_to_scale() {
        let small = GaussianSplattingTool {
            scale: 0.3,
            ..Default::default()
        };
        let large = GaussianSplattingTool {
            scale: 2.5,
            ..Default::default()
        };

        let img_small = small.render_splat_image();
        let img_large = large.render_splat_image();
        assert_eq!(img_small.size, [SPLAT_GRID_SIZE, SPLAT_GRID_SIZE]);
        assert_ne!(img_small.pixels, img_large.pixels);

        let counts_small = GaussianSplattingTool::compute_density_counts_for_scale(0.3);
        let counts_large = GaussianSplattingTool::compute_density_counts_for_scale(2.5);
        assert!(counts_small.clone > counts_large.clone);
        assert!(counts_large.split > counts_small.split);
        assert_eq!(counts_small.prune, 1);
        assert_eq!(counts_small.keep, 3);
    }
}
