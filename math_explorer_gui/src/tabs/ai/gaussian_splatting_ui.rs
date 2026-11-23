#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::scene::{
    GRAD_THRESHOLD_MAX, GRAD_THRESHOLD_MIN, OPACITY_THRESHOLD_MAX, OPACITY_THRESHOLD_MIN,
    SPLAT_GRID_SIZE, VIEWPORT_HALF_EXTENT,
};
use super::GaussianSplattingTool;
use crate::accessibility::AccessibleTheoryHover;
use crate::framework::Plot3DViewport;
use crate::reflective_ui::{AnimatedSlider, SweepMode};
use eframe::egui;
use egui_plot::{LineStyle, MarkerShape, PlotImage, PlotPoint};

impl GaussianSplattingTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "gaussian_splatting_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Gaussian Splatting");
            ui.separator();
            let mut selected_preset = self.scene_preset;
            egui::ComboBox::from_label("Scene Preset")
                .selected_text(selected_preset.label())
                .show_ui(ui, |ui| {
                    for preset in crate::volumetric_scene::presets::VolumetricSplatPreset::ALL {
                        ui.selectable_value(&mut selected_preset, preset, preset.label());
                    }
                });
            if selected_preset != self.scene_preset {
                self.set_scene_preset(selected_preset);
            }
            let prev_scale = self.scale;
            if ui
                .add(
                    AnimatedSlider::new(&mut self.scale, 0.1..=10.0)
                        .hard_bounds(0.1..=10.0)
                        .speed(1.8)
                        .sweep_mode(SweepMode::Oscillate)
                        .text("Scale")
                        .id_salt("gaussian_splatting_scale"),
                )
                .changed()
            {
                self.scale = if self.scale.is_finite() {
                    self.scale.clamp(0.1, 10.0)
                } else {
                    prev_scale
                };
                self.sync_density_counts();
            }

            ui.separator();
            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));

            ui.separator();
            self.draw_density_controls(ui);
        });
    }

    fn draw_threshold_sliders(&mut self, ui: &mut egui::Ui) {
        let prev_grad = self.grad_threshold;
        if ui
            .add(
                AnimatedSlider::new(
                    &mut self.grad_threshold,
                    GRAD_THRESHOLD_MIN..=GRAD_THRESHOLD_MAX,
                )
                .hard_bounds(GRAD_THRESHOLD_MIN..=GRAD_THRESHOLD_MAX)
                .text("Gradient Threshold")
                .id_salt("gaussian_splatting_grad_threshold"),
            )
            .changed()
        {
            self.grad_threshold = if self.grad_threshold.is_finite() {
                self.grad_threshold
                    .clamp(GRAD_THRESHOLD_MIN, GRAD_THRESHOLD_MAX)
            } else {
                prev_grad
            };
            self.sync_density_counts();
        }

        let prev_opacity = self.opacity_threshold;
        if ui
            .add(
                AnimatedSlider::new(
                    &mut self.opacity_threshold,
                    OPACITY_THRESHOLD_MIN..=OPACITY_THRESHOLD_MAX,
                )
                .hard_bounds(OPACITY_THRESHOLD_MIN..=OPACITY_THRESHOLD_MAX)
                .text("Opacity Threshold")
                .id_salt("gaussian_splatting_opacity_threshold"),
            )
            .changed()
        {
            self.opacity_threshold = if self.opacity_threshold.is_finite() {
                self.opacity_threshold
                    .clamp(OPACITY_THRESHOLD_MIN, OPACITY_THRESHOLD_MAX)
            } else {
                prev_opacity
            };
            self.sync_density_counts();
        }
    }

    fn draw_density_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Adaptive Density Control");
        self.draw_threshold_sliders(ui);
        ui.label(format!(
            "Split (large grad & scale): {}",
            self.cached_counts.split
        ));
        ui.label(format!(
            "Clone (large grad, small scale): {}",
            self.cached_counts.clone
        ));
        ui.label(format!("Keep (converged): {}", self.cached_counts.keep));
        ui.label(format!(
            "Prune (opacity < {:.3}): {}",
            self.opacity_threshold, self.cached_counts.prune
        ));
        ui.separator();
        ui.small(format!(
            "Raster grid: {SPLAT_GRID_SIZE}×{SPLAT_GRID_SIZE} px ({} Gaussians)",
            self.scene.gaussians.len()
        ));
    }

    pub(super) fn draw_viewport(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let avail = ui.available_size_before_wrap();
            let (_plot_h, x_half, y_half) =
                Plot3DViewport::compute_half_extents(avail, VIEWPORT_HALF_EXTENT, true);
            self.frustum_half_extents = (x_half, y_half);
            self.refresh_projection_and_raster(ui.ctx());

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let dark_mode = ui.visuals().dark_mode;
            let tex_id = self.texture.as_ref().map(egui::TextureHandle::id);
            let img_size = egui::vec2((2.0 * x_half) as f32, (2.0 * y_half) as f32);
            let mut camera = self.camera;

            self.scene_builder.begin(camera);
            self.scene_builder.add_axes(1.4, dark_mode);
            for (ring_3d, color) in &self.ellipse_rings_3d {
                self.scene_builder.add_polyline(
                    "",
                    ring_3d,
                    egui::Stroke::new(1.25_f32, *color),
                    LineStyle::Solid,
                    false,
                );
            }
            self.scene_builder.add_points(
                "Splat Means",
                &self.world_means,
                3.0_f32,
                palette.accent,
                MarkerShape::Circle,
                true,
            );

            let response = Plot3DViewport::new("gaussian_splatting_plot", VIEWPORT_HALF_EXTENT)
                .show(ui, &mut camera, |plot_ui| {
                    if let Some(tid) = tex_id {
                        plot_ui.image(PlotImage::new(
                            "Radiance Field",
                            tid,
                            PlotPoint::new(0.0, 0.0),
                            img_size,
                        ));
                    }
                    self.scene_builder.finish(plot_ui);
                })
                .response;

            self.camera = camera;
            response.accessible_theory_hover(self);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for GaussianSplattingTool {
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
        map.insert(
            "grad_threshold".into(),
            "Positional view-space gradient threshold separating converged splats from split or clone densification.".into(),
        );
        map.insert(
            "opacity_threshold".into(),
            "Minimum opacity threshold below which nearly transparent Gaussians are pruned.".into(),
        );
        map
    }
}
