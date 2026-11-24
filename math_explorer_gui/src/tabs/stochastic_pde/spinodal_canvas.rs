//! Interactive 2D phase-field torus canvas with conserved droplet brushing, keyboard stamping,
//! AccessKit `Role::GraphicsDocument`, and WCAG 2.2 AA colormaps for [`SpinodalDecompositionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::hash::{Hash, Hasher};

use eframe::egui;
use eframe::egui::ColorImage;
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use scientific_metadata::theory::TheoryDescribable;

use super::presets::SpinodalToolState;
use super::spinodal_tool::{PhaseFieldModelChoice, SpinodalBrushAction, SpinodalDecompositionTool};
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, register_canvas_accessibility,
};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};

impl SpinodalDecompositionTool {
    pub(crate) fn refresh_pixels_in_place(&mut self) {
        let w = self.solver.width();
        let h = self.solver.height();
        let total = w * h;
        if self.pixels.len() != total {
            self.pixels.resize(total, egui::Color32::BLACK);
        }
        let eq_scale = (self.critical_temperature - self.temperature)
            .max(0.15)
            .sqrt()
            .max(0.5);

        for (dst, &phi) in self.pixels.iter_mut().zip(self.solver.field().iter()) {
            let norm = if phi.is_finite() {
                (0.5 + 0.5 * (phi / eq_scale)).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let rgb = if self.high_contrast {
                heatmap_color_high_contrast(norm)
            } else {
                heatmap_color(norm)
            };
            *dst = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        }
        self.texture_dirty = true;
    }

    fn sync_gpu_texture(&mut self, ctx: &egui::Context) {
        if !self.texture_dirty && self.texture.is_some() {
            return;
        }
        let image = ColorImage {
            size: [self.solver.width(), self.solver.height()],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let opts = egui::TextureOptions::NEAREST;
        if let Some(tex) = &mut self.texture {
            tex.set(image, opts);
        } else {
            self.texture = Some(ctx.load_texture("spinodal_phase_field_2d", image, opts));
        }
        self.texture_dirty = false;
    }

    pub(crate) fn draw_phase_field_canvas(&mut self, ui: &mut egui::Ui, target_side: f32) {
        let ctx = ui.ctx().clone();
        self.sync_gpu_texture(&ctx);

        let side = target_side.clamp(160.0, 520.0);
        let (response, painter) =
            ui.allocate_painter(egui::vec2(side, side), egui::Sense::click_and_drag());

        if let Some(tex) = &self.texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(tex.id(), response.rect, uv, egui::Color32::WHITE);
        }

        self.handle_droplet_pointer(ui, &response);
        self.handle_droplet_keyboard(ui, &response);
        self.paint_droplet_reticle(ui, &response, &painter);

        let sum = self.solver.telemetry().latest_summary();
        let summary = format!(
            "{} on {}x{} torus: <phi> = {:.5}, F = {:.2}, k_max = {:.3}, cursor ({}, {})",
            self.model.label(),
            self.solver.width(),
            self.solver.height(),
            self.solver.mean_concentration(),
            self.solver.free_energy(),
            sum.k_max,
            self.cursor[0],
            self.cursor[1],
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Spinodal Decomposition Phase-Field Canvas",
            Some(&summary),
        );
        paint_canvas_focus_ring(ui, &response);
        lock_canvas_arrow_keys(ui, &response);
        self.focused_canvas = response.has_focus().then_some(response.id);
    }

    fn handle_droplet_pointer(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.clicked() || response.dragged() {
            response.request_focus();
        }
        let Some(pos) = response.interact_pointer_pos() else {
            return;
        };
        if !response.rect.contains(pos) {
            return;
        }
        let primary = response.clicked_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Primary);
        let secondary = response.clicked_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Secondary);
        if !primary && !secondary {
            return;
        }

        let rx = ((pos.x - response.rect.min.x) / response.rect.width()).clamp(0.0, 0.999_9);
        let ry = ((pos.y - response.rect.min.y) / response.rect.height()).clamp(0.0, 0.999_9);
        let gx = (rx * (self.solver.width() as f32)) as usize;
        let gy = (ry * (self.solver.height() as f32)) as usize;
        self.cursor = [gx, gy];

        let invert = secondary
            || ui.input(|i| i.modifiers.shift)
            || self.brush_action == SpinodalBrushAction::EnrichNegative;
        let delta = if invert { -0.45 } else { 0.45 };
        let conserve = self.model == PhaseFieldModelChoice::CahnHilliard;
        self.solver
            .paint_droplet(gx, gy, self.brush_radius, delta, conserve);
        let _ = self.solver.refresh_telemetry();
        self.refresh_pixels_in_place();
    }

    fn handle_droplet_keyboard(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if !response.has_focus() {
            return;
        }
        let w = self.solver.width();
        let h = self.solver.height();
        let (dx, dy, stamp_pos, stamp_neg) = ui.input(|i| {
            let step = if i.modifiers.shift { 4isize } else { 1isize };
            let dx = if i.key_pressed(egui::Key::ArrowRight) {
                step
            } else if i.key_pressed(egui::Key::ArrowLeft) {
                -step
            } else {
                0
            };
            let dy = if i.key_pressed(egui::Key::ArrowDown) {
                step
            } else if i.key_pressed(egui::Key::ArrowUp) {
                -step
            } else {
                0
            };
            let pos_hit = i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Equals);
            let neg_hit = i.key_pressed(egui::Key::Minus) || i.key_pressed(egui::Key::Backspace);
            (dx, dy, pos_hit, neg_hit)
        });

        if dx != 0 {
            self.cursor[0] = (self.cursor[0] as isize + dx).rem_euclid(w as isize) as usize;
        }
        if dy != 0 {
            self.cursor[1] = (self.cursor[1] as isize + dy).rem_euclid(h as isize) as usize;
        }
        if stamp_pos || stamp_neg {
            let delta = if stamp_neg { -0.55 } else { 0.55 };
            let conserve = self.model == PhaseFieldModelChoice::CahnHilliard;
            self.solver.paint_droplet(
                self.cursor[0],
                self.cursor[1],
                self.brush_radius,
                delta,
                conserve,
            );
            let _ = self.solver.refresh_telemetry();
            self.refresh_pixels_in_place();
        }
    }

    fn paint_droplet_reticle(
        &self,
        ui: &egui::Ui,
        response: &egui::Response,
        painter: &egui::Painter,
    ) {
        if !response.has_focus() && !response.hovered() {
            return;
        }
        let w = self.solver.width().max(1) as f32;
        let h = self.solver.height().max(1) as f32;
        let cx = response.rect.min.x + ((self.cursor[0] as f32 + 0.5) / w) * response.rect.width();
        let cy = response.rect.min.y + ((self.cursor[1] as f32 + 0.5) / h) * response.rect.height();
        let rad = ((self.brush_radius as f32) / w) * response.rect.width();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let color = if self.brush_action == SpinodalBrushAction::EnrichNegative {
            palette.warning
        } else {
            palette.accent
        };
        painter.circle_stroke(
            egui::pos2(cx, cy),
            rad.max(4.0_f32),
            egui::Stroke::new(1.5_f32, color),
        );
    }
}

impl InteractiveTool for SpinodalDecompositionTool {
    fn name(&self) -> &'static str {
        "Spinodal Decomposition"
    }

    fn domain(&self) -> &'static str {
        "stochastic_pde"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        let w = self.solver.width();
        let h = self.solver.height();
        (w > 0 && h > 0 && self.pixels.len() == w * h).then(|| ColorImage {
            size: [w, h],
            pixels: self.pixels.clone(),
            ..Default::default()
        })
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    fn step_simulation(&mut self) {
        self.step_once();
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_workspace(ui);
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.model.hash(&mut h);
        self.temperature.to_bits().hash(&mut h);
        self.critical_temperature.to_bits().hash(&mut h);
        self.mobility.to_bits().hash(&mut h);
        self.kappa.to_bits().hash(&mut h);
        self.noise_sigma.to_bits().hash(&mut h);
        self.mean_phi.to_bits().hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        self.grid_size.hash(&mut h);
        self.steps_per_frame.hash(&mut h);
        self.rng_seed.hash(&mut h);
        self.plot_mode.hash(&mut h);
        self.brush_action.hash(&mut h);
        self.brush_radius.hash(&mut h);
        self.playing.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.snapshot_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: SpinodalToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Spinodal state JSON: {e}"))?;
        parsed.validate()?;
        self.model = parsed.model;
        self.temperature = parsed.temperature;
        self.critical_temperature = parsed.critical_temperature;
        self.mobility = parsed.mobility;
        self.kappa = parsed.kappa;
        self.noise_sigma = parsed.noise_sigma;
        self.mean_phi = parsed.mean_phi;
        self.dt = parsed.dt;
        self.grid_size = parsed.grid_size;
        self.steps_per_frame = parsed.steps_per_frame;
        self.rng_seed = parsed.rng_seed;
        self.plot_mode = parsed.plot_mode;
        self.brush_action = parsed.brush_action;
        self.brush_radius = parsed.brush_radius;
        self.playing = parsed.playing;
        self.reseed_spinodal_state();
        Ok(())
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        let sum = self.solver.telemetry().latest_summary();
        vec![
            ("t", self.solver.time()),
            ("mean_phi", self.solver.mean_concentration()),
            ("free_energy", self.solver.free_energy()),
            ("k_max", sum.k_max),
            ("k_first_moment", sum.k_first_moment),
        ]
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SpinodalDecompositionTool",
        display_name: "Spinodal Decomposition",
        domain: "stochastic_pde",
        tags: &[
            "cahn-hilliard",
            "allen-cahn",
            "spinodal",
            "phase-field",
            "spde",
            "biharmonic",
            "structure-factor",
            "coarsening",
        ],
        build: || Box::new(SpinodalDecompositionTool::default()),
    }
}
