#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::scene::{OPACITY_THRESHOLD, SPLAT_GRID_SIZE, VIEWPORT_HALF_EXTENT};
use super::GaussianSplattingTool;
use crate::accessibility::AccessibleTheoryHover;
use crate::framework::Plot3DViewport;
use eframe::egui;
use egui_plot::{LineStyle, MarkerShape, PlotImage, PlotPoint};

impl GaussianSplattingTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "gaussian_splatting_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Gaussian Splatting");
            ui.separator();
            let prev_scale = self.scale;
            if ui
                .add(egui::Slider::new(&mut self.scale, 0.1..=10.0).text("Scale"))
                .changed()
            {
                self.scale = if self.scale.is_finite() {
                    self.scale.clamp(0.1, 10.0)
                } else {
                    prev_scale
                };
                self.cached_counts = Self::compute_density_counts_for_scale(self.scale);
            }

            ui.separator();
            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));

            ui.separator();
            ui.heading("Adaptive Density Control");
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
                "Prune (opacity < {OPACITY_THRESHOLD:.2}): {}",
                self.cached_counts.prune
            ));
            ui.separator();
            ui.small(format!(
                "Raster grid: {SPLAT_GRID_SIZE}×{SPLAT_GRID_SIZE} px ({} Gaussians)",
                self.scene.gaussians.len()
            ));
        });
    }

    pub(super) fn draw_viewport(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.refresh_projection_and_raster(ui.ctx());
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let dark_mode = ui.visuals().dark_mode;
            let tex_id = self.texture.as_ref().map(egui::TextureHandle::id);
            let extent_size = (2.0 * VIEWPORT_HALF_EXTENT) as f32;
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

            let img_center = PlotPoint::new(f64::from(camera.pan_x), f64::from(camera.pan_y));
            let response = Plot3DViewport::new("gaussian_splatting_plot", VIEWPORT_HALF_EXTENT)
                .show(ui, &mut camera, |plot_ui| {
                    if let Some(tid) = tex_id {
                        plot_ui.image(PlotImage::new(
                            "Radiance Field",
                            tid,
                            img_center,
                            egui::vec2(extent_size, extent_size),
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
