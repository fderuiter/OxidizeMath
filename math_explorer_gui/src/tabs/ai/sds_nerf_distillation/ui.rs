#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    SdsNeRfDistillationTool, SdsOptimizerKind, SdsPromptPreset, CFG_MAX, CFG_MIN, LR_MAX, LR_MIN,
    MAX_RAY_GRID, MAX_RAY_SAMPLES, MIN_RAY_GRID, MIN_RAY_SAMPLES, VIEWPORT_EXTENT,
};
use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport, Plot3DViewport};
use eframe::egui;
use egui_plot::{LineStyle, MarkerShape, PlotImage, PlotPoint};
use math_explorer::nalgebra::Point3;

impl SdsNeRfDistillationTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "sds_nerf_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("SDS NeRF Distillation");
            ui.separator();

            if let Some(action) = PlaybackTransport::in_thread(
                "sds_nerf_playback",
                "SDS NeRF Distillation",
                self.is_playing,
            )
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => self.is_playing = true,
                    PlaybackAction::Pause => self.is_playing = false,
                    PlaybackAction::Step => self.step_distillation_once(),
                    PlaybackAction::Reset => {
                        self.is_playing = false;
                        self.reset_distillation();
                    }
                }
            }

            ui.separator();
            self.draw_hyperparameter_controls(ui);
            ui.separator();
            egui::CollapsingHeader::new("3D Camera")
                .default_open(false)
                .show(ui, |ui| self.camera.ui(ui));

            ui.separator();
            ui.label(format!(
                "Step: {} | Target MSE: {:.4} (Init: {:.4})",
                self.steps_completed, self.current_mse, self.initial_mse
            ));
            ui.label(format!(
                "RMS SDS Score Residual: {:.4}",
                self.last_grad_norm
            ));

            let grid = self.ray_grid_size.clamp(MIN_RAY_GRID, MAX_RAY_GRID);
            ui.horizontal(|ui| {
                self.render_tile
                    .show(ui, "NeRF RGB Render", grid, &self.render_pixels);
                self.grad_tile
                    .show(ui, "SDS Score Gradient", grid, &self.grad_pixels);
            });
        });
    }

    fn draw_hyperparameter_controls(&mut self, ui: &mut egui::Ui) {
        let mut next_prompt = self.prompt_preset;
        egui::ComboBox::from_label("Text Prompt")
            .selected_text(next_prompt.label())
            .show_ui(ui, |ui| {
                for p in SdsPromptPreset::ALL {
                    ui.selectable_value(&mut next_prompt, p, p.label());
                }
            });
        let mut reset_needed = next_prompt != self.prompt_preset;
        self.prompt_preset = next_prompt;

        let mut next_opt = self.optimizer_kind;
        egui::ComboBox::from_label("Optimizer")
            .selected_text(next_opt.label())
            .show_ui(ui, |ui| {
                for o in SdsOptimizerKind::ALL {
                    ui.selectable_value(&mut next_opt, o, o.label());
                }
            });
        if next_opt != self.optimizer_kind {
            self.optimizer_kind = next_opt;
            self.rebuild_optimizers();
        }

        ui.add(egui::Slider::new(&mut self.guidance_scale, CFG_MIN..=CFG_MAX).text("CFG Guidance"));
        if ui
            .add(egui::Slider::new(&mut self.learning_rate, LR_MIN..=LR_MAX).text("Learning Rate"))
            .changed()
        {
            self.rebuild_optimizers();
        }
        reset_needed |= ui
            .add(
                egui::Slider::new(&mut self.ray_grid_size, MIN_RAY_GRID..=MAX_RAY_GRID)
                    .text("Ray Grid Pixels"),
            )
            .changed();
        reset_needed |= ui
            .add(
                egui::Slider::new(&mut self.samples_per_ray, MIN_RAY_SAMPLES..=MAX_RAY_SAMPLES)
                    .text("Samples per Ray"),
            )
            .changed();
        self.clamp_parameters();
        if reset_needed {
            self.is_playing = false;
            self.reset_distillation();
        }
    }

    pub(super) fn draw_viewport_3d(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let grid = self.ray_grid_size.clamp(MIN_RAY_GRID, MAX_RAY_GRID);
            if (self.backdrop_dirty || self.backdrop_tex.is_none())
                && self.render_pixels.len() >= grid * grid
            {
                let img = egui::ColorImage {
                    size: [grid, grid],
                    pixels: self.render_pixels[..grid * grid].to_vec(),
                    ..Default::default()
                };
                if let Some(tex) = &mut self.backdrop_tex {
                    tex.set(img, egui::TextureOptions::LINEAR);
                } else {
                    self.backdrop_tex = Some(ui.ctx().load_texture(
                        "sds_nerf_viewport_backdrop",
                        img,
                        egui::TextureOptions::LINEAR,
                    ));
                }
                self.backdrop_dirty = false;
            }

            let avail = ui.available_size_before_wrap();
            let (_h, x_half, y_half) =
                Plot3DViewport::compute_half_extents(avail, VIEWPORT_EXTENT, true);
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let dark_mode = ui.visuals().dark_mode;
            let tex_id = self.backdrop_tex.as_ref().map(egui::TextureHandle::id);
            let img_size = egui::vec2((1.65 * x_half) as f32, (1.65 * y_half) as f32);

            let mut camera = self.camera;
            self.scene_builder.begin(camera);
            self.scene_builder.add_axes(1.25, dark_mode);
            self.populate_voxel_and_frustum_geometry(&palette);

            let response = Plot3DViewport::new("sds_nerf_distillation_viewport", VIEWPORT_EXTENT)
                .show(ui, &mut camera, |plot_ui| {
                    if let Some(tid) = tex_id {
                        plot_ui.image(PlotImage::new(
                            "Distilled NeRF Projection",
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

    fn populate_voxel_and_frustum_geometry(&mut self, palette: &SemanticPalette) {
        let voxels = self.nerf.active_voxels(0.18);
        let mut dense_pts = Vec::new();
        let mut mid_pts = Vec::new();
        for v in &voxels {
            let pt = Point3::new(v.position.x, v.position.y, v.position.z);
            if v.density >= 0.48 {
                dense_pts.push(pt);
            } else {
                mid_pts.push(pt);
            }
        }
        if !mid_pts.is_empty() {
            self.scene_builder.add_points(
                "Shell Voxels",
                &mid_pts,
                3.5_f32,
                palette.muted_text,
                MarkerShape::Circle,
                false,
            );
        }
        if !dense_pts.is_empty() {
            let core_color = match self.prompt_preset {
                SdsPromptPreset::AmberTorus => palette.series[1],
                SdsPromptPreset::CyanOctahedron => palette.info,
                SdsPromptPreset::GoldSphere => palette.accent,
                SdsPromptPreset::TwinLobes => palette.series[4],
            };
            self.scene_builder.add_points(
                "Distilled Volume Core",
                &dense_pts,
                5.5_f32,
                core_color,
                MarkerShape::Diamond,
                true,
            );
        }
        self.add_camera_frustum_wireframe(palette);
    }

    fn add_camera_frustum_wireframe(&mut self, palette: &SemanticPalette) {
        let apex = Point3::new(0.0, 0.0, 1.55);
        let corners = [
            Point3::new(-0.85, -0.85, 0.85),
            Point3::new(0.85, -0.85, 0.85),
            Point3::new(0.85, 0.85, 0.85),
            Point3::new(-0.85, 0.85, 0.85),
            Point3::new(-0.85, -0.85, 0.85),
        ];
        self.scene_builder.add_polyline(
            "Camera Frustum",
            &corners,
            egui::Stroke::new(1.2_f32, palette.accent),
            LineStyle::Dashed { length: 5.0 },
            false,
        );
        for c in &corners[..4] {
            self.scene_builder.add_polyline(
                "",
                &[apex, *c],
                egui::Stroke::new(1.0_f32, palette.accent),
                LineStyle::Dotted { spacing: 4.0 },
                false,
            );
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for SdsNeRfDistillationTool {
    fn theory_description(&self) -> String {
        format!(
            "Score Distillation Sampling optimizes 3D radiance parameters $\\theta$ by injecting forward diffusion noise $z_t = \\sqrt{{\\bar{{\\alpha}}_t}} x(\\theta) + \\sqrt{{1 - \\bar{{\\alpha}}_t}}\\epsilon$, applying Classifier-Free Guidance ($s = {:.1}$), and backpropagating $\\nabla_\\theta \\mathcal{{L}}_{{\\mathrm{{SDS}}}} = w(t)(\\hat{{\\epsilon}}_\\phi(z_t; y, t) - \\epsilon)\\frac{{\\partial x}}{{\\partial \\theta}}$ without differentiating through the diffusion network.",
            self.guidance_scale
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "Score Distillation Sampling 3D volumetric NeRF studio with guidance scale {:.1}, {} by {} ray grid, and {} optimizer.",
            self.guidance_scale,
            self.ray_grid_size,
            self.ray_grid_size,
            self.optimizer_kind.label()
        )
    }

    fn theory_citation(&self) -> String {
        "Poole, B., Jain, A., Barron, J. T., & Mildenhall, B. (2022). DreamFusion: Text-to-3D using 2D Diffusion; Mildenhall, B. et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "guidance_scale".into(),
            "Classifier-Free Guidance factor amplifying prompt-conditioned score directions relative to the unconditional prior.".into(),
        );
        map.insert(
            "ray_grid_size".into(),
            "Square camera ray-bundle resolution capped at 32 by 32 pixels for real-time WebAssembly frame rates.".into(),
        );
        map.insert(
            "samples_per_ray".into(),
            "Stratified volumetric quadrature points per ray capped at 24 samples.".into(),
        );
        map
    }
}
