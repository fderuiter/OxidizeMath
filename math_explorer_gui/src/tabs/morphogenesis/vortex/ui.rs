#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, Peclet/Damkohler badges, brush painting, and accessible canvas rendering.

use super::render::render_vortex_field_into;
use super::state::{
    VortexBrushMode, VortexDisplayMode, VortexKineticsKind, VortexPresetTag,
};
use super::VortexMorphogenesisTool;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::{SemanticPalette, ThemeMode};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui::{self, ColorImage};
use math_explorer::biology::morphogenesis::{GrayScottKinetics, SchnakenbergKinetics};
use math_explorer::vortex_morphogenesis::{
    MorphogenKinetics, VELOCITY_SAFETY_CAP, VortexRegimePreset,
};

impl VortexMorphogenesisTool {
    pub(super) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "vortex_morphogenesis_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Vortex Morphogenesis");
                ui.separator();
                self.draw_recovery_banner(ui);
                self.draw_transport_bar(ui);
                self.draw_regime_badges(ui);
                ui.separator();
                self.draw_regime_presets(ui);
                ui.separator();
                self.draw_brush_controls(ui);
                ui.separator();
                self.draw_kinetics_controls(ui);
                ui.separator();
                self.draw_fluid_transport_sliders(ui);
                ui.separator();
                self.draw_cross_tab_link(ui);
            });
        });
    }

    fn draw_recovery_banner(&mut self, ui: &mut egui::Ui) {
        if !FiniteStateGuard::diverged(self) {
            return;
        }
        let palette = SemanticPalette::from_visuals(ui.visuals());
        egui::Frame::group(ui.style())
            .fill(palette.error_banner_bg)
            .stroke(egui::Stroke::new(1.5_f32, palette.error_banner_stroke))
            .show(ui, |ui| {
                ui.colored_label(
                    palette.error,
                    "⚠ Numerical divergence halted coupled stepping",
                );
                if let Some(report) = &self.diagnostic_report {
                    ui.small(&report.message);
                }
                ui.horizontal_wrapped(|ui| {
                    if ui.button("⏪ Rollback Snapshot").clicked() {
                        self.rollback_to_last_valid();
                    }
                    if ui.button("½ Halve Δt").clicked() {
                        FiniteStateGuard::auto_scale_step_size(self);
                    }
                    if ui.button("↺ Reset Preset").clicked() {
                        FiniteStateGuard::reset_parameters(self);
                    }
                });
            });
        ui.separator();
    }

    fn draw_transport_bar(&mut self, ui: &mut egui::Ui) {
        let is_playing = !self.paused && !FiniteStateGuard::diverged(self);
        let Some(action) = PlaybackTransport::in_thread(
            "vortex_morphogenesis_transport",
            "Vortex Morphogenesis",
            is_playing,
        )
        .show(ui) else {
            return;
        };
        match action {
            PlaybackAction::Play => {
                self.paused = false;
                self.bump_revision();
            }
            PlaybackAction::Pause => {
                self.paused = true;
                self.bump_revision();
            }
            PlaybackAction::Step => self.step_single_tick(),
            PlaybackAction::Reset => self.reset_active_regime(),
        }
    }

    fn draw_regime_badges(&self, ui: &mut egui::Ui) {
        let metrics = self.engine.metrics();
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let pe_color = if metrics.peclet < 1.0 {
            palette.info
        } else if metrics.peclet <= 18.0 {
            palette.success
        } else {
            palette.warning
        };
        let da_color = if metrics.damkohler < 1.0 {
            palette.warning
        } else {
            palette.success
        };
        ui.add_space(4.0);
        egui::Frame::group(ui.style())
            .fill(palette.surface_fill)
            .show(ui, |ui| {
                ui.colored_label(
                    pe_color,
                    format!("Pe = {:.2} · {}", metrics.peclet, metrics.peclet_label()),
                )
                .accessible_hover_text(
                    "Peclet number Pe = U L / D_slow measuring advective vs diffusive transport",
                );
                ui.colored_label(
                    da_color,
                    format!("Da = {:.2} · {}", metrics.damkohler, metrics.damkohler_label()),
                )
                .accessible_hover_text(
                    "Damkohler number Da = k_react L / U measuring reaction persistence vs washout",
                );
                ui.small(format!(
                    "|u|_max = {:.3} (≤ {:.2}) · CFL sub-steps: {}",
                    metrics.max_speed, VELOCITY_SAFETY_CAP, metrics.substeps
                ));
            });
    }

    fn draw_regime_presets(&mut self, ui: &mut egui::Ui) {
        ui.label("Flow-Morphogenesis Regimes:");
        for preset in VortexRegimePreset::ALL {
            let selected = self.preset_tag == VortexPresetTag::from_regime(preset);
            if ui
                .selectable_label(selected, preset.title())
                .accessible_hover_text(preset.summary())
                .clicked()
            {
                self.select_regime_preset(preset);
                announce_live_status(
                    ui.ctx(),
                    format!("Loaded {} regime preset", preset.title()),
                );
            }
        }
    }

    fn draw_brush_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Interactive Canvas Brush:");
        ui.horizontal_wrapped(|ui| {
            for (mode, icon_label) in [
                (VortexBrushMode::Obstacle, "🧱 Obstacle"),
                (VortexBrushMode::Emitter, "🧪 Emitter"),
                (VortexBrushMode::Erase, "🧹 Erase"),
            ] {
                if ui
                    .selectable_label(self.brush_mode == mode, icon_label)
                    .accessible_hover_text(format!("Switch canvas brush to {}", mode.label()))
                    .clicked()
                {
                    self.brush_mode = mode;
                    self.bump_revision();
                }
            }
        });
        let mut radius = self.brush_radius;
        if ui
            .add(egui::Slider::new(&mut radius, 1..=8).text("Brush Radius"))
            .changed()
        {
            self.brush_radius = radius;
            self.bump_revision();
        }
        if ui
            .button("Clear Obstacles & Emitters")
            .accessible_hover_text("Remove all solid obstacles and chemical emitters from channel")
            .clicked()
        {
            self.engine.clear_geometry();
            self.preset_tag = VortexPresetTag::Custom;
            self.dirty = true;
            self.bump_revision();
        }
    }

    fn draw_kinetics_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Reaction Kinetics:");
        let current_kind = self.kinetics_kind();
        ui.horizontal(|ui| {
            if ui
                .selectable_label(current_kind == VortexKineticsKind::GrayScott, "Gray-Scott")
                .clicked()
                && current_kind != VortexKineticsKind::GrayScott
            {
                self.set_kinetics_kind(VortexKineticsKind::GrayScott);
            }
            if ui
                .selectable_label(
                    current_kind == VortexKineticsKind::Schnakenberg,
                    "Schnakenberg",
                )
                .clicked()
                && current_kind != VortexKineticsKind::Schnakenberg
            {
                self.set_kinetics_kind(VortexKineticsKind::Schnakenberg);
            }
        });

        let mut changed = false;
        match current_kind {
            VortexKineticsKind::GrayScott => {
                changed |= ui
                    .add(egui::Slider::new(&mut self.gs_params.0, 0.010..=0.090).text("Feed (f)"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.gs_params.1, 0.030..=0.080).text("Kill (k)"))
                    .changed();
                if changed {
                    self.engine.config.kinetics = MorphogenKinetics::GrayScott(
                        GrayScottKinetics::new(self.gs_params.0, self.gs_params.1),
                    );
                }
            }
            VortexKineticsKind::Schnakenberg => {
                changed |= ui
                    .add(egui::Slider::new(&mut self.sn_params.0, 0.01..=0.50).text("Source (a)"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(&mut self.sn_params.1, 0.20..=1.80).text("Source (b)"))
                    .changed();
                if changed {
                    self.engine.config.kinetics = MorphogenKinetics::Schnakenberg(
                        SchnakenbergKinetics::new(self.sn_params.0, self.sn_params.1),
                    );
                }
            }
        }
        if changed {
            self.preset_tag = VortexPresetTag::Custom;
            self.engine.transpose_lbm_to_row_major();
            self.bump_revision();
        }
    }

    fn draw_fluid_transport_sliders(&mut self, ui: &mut egui::Ui) {
        ui.label("Fluid & Transport Coupling:");
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.engine.config.inlet_ux,
                    -VELOCITY_SAFETY_CAP..=VELOCITY_SAFETY_CAP,
                )
                .text("Inlet Speed (u_in)"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.engine.config.tau, 0.52..=1.40).text("Relaxation (τ)"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.engine.config.advection_coupling, 0.0..=1.5)
                    .text("Advection Coupling"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.engine.config.d_u, 0.01..=1.0).text("Diffusion D_u"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.engine.config.d_v, 0.01..=2.5).text("Diffusion D_v"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.engine.config.dt, 0.05..=4.0).text("Step Size Δt"))
            .changed();

        if changed {
            self.preset_tag = VortexPresetTag::Custom;
            self.engine.transpose_lbm_to_row_major();
            self.dirty = true;
            self.bump_revision();
        }

        egui::ComboBox::from_label("Layer")
            .selected_text(self.display_mode.label())
            .show_ui(ui, |ui| {
                for mode in VortexDisplayMode::ALL {
                    if ui
                        .selectable_value(&mut self.display_mode, mode, mode.label())
                        .changed()
                    {
                        self.dirty = true;
                        self.bump_revision();
                    }
                }
            });
    }

    fn draw_cross_tab_link(&self, ui: &mut egui::Ui) {
        if ui
            .button("🌊 Inspect Pure Fluid Wake in Fluid Dynamics →")
            .accessible_hover_text(
                "Switch to Lattice Boltzmann (Demo) in the Fluid Dynamics tab",
            )
            .clicked()
        {
            crate::app::request_cross_tab_navigation(
                ui.ctx(),
                "Fluid Dynamics",
                Some("LatticeBoltzmannTool"),
            );
        }
    }

    pub(super) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let high_contrast =
            ThemeMode::from_visuals(ui.visuals()) == ThemeMode::HighContrast;
        if self.last_high_contrast != Some(high_contrast) {
            self.last_high_contrast = Some(high_contrast);
            self.dirty = true;
        }
        self.sync_texture(&ctx, high_contrast);

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let avail = ui.available_size().max(egui::vec2(120.0, 80.0));
            let aspect = (self.engine.config.width as f32) / (self.engine.config.height as f32);
            let target_w = avail.x.min(avail.y * aspect);
            let target_h = (target_w / aspect).min(avail.y);
            let (response, painter) =
                ui.allocate_painter(egui::vec2(target_w, target_h), egui::Sense::click_and_drag());
            self.last_canvas_id = Some(response.id);

            self.handle_canvas_input(ui, &response);

            if let Some(tex) = &self.texture {
                painter.image(
                    tex.id(),
                    response.rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }
            self.paint_cursor_overlay(ui, &response, &painter);

            let metrics = self.engine.metrics();
            let value_summary = format!(
                "{}, Pe = {:.2} ({}), Da = {:.2} ({}), brush = {}",
                self.display_mode.label(),
                metrics.peclet,
                metrics.peclet_label(),
                metrics.damkohler,
                metrics.damkohler_label(),
                self.brush_mode.label()
            );
            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::Canvas,
                "Coupled Vortex Morphogenesis Field",
                Some(&value_summary),
            );
            lock_canvas_arrow_keys(ui, &response);
            paint_canvas_focus_ring(ui, &response);
        });
    }

    fn sync_texture(&mut self, ctx: &egui::Context, high_contrast: bool) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        render_vortex_field_into(
            &self.engine,
            self.display_mode,
            high_contrast,
            &mut self.pixels,
        );
        let image = ColorImage {
            size: [self.engine.config.width, self.engine.config.height],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        if let Some(tex) = &mut self.texture {
            tex.set(image, egui::TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture(
                "vortex_morphogenesis_tex",
                image,
                egui::TextureOptions::NEAREST,
            ));
        }
        self.dirty = false;
    }

    fn handle_canvas_input(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.clicked() || response.dragged() {
            if let Some((gx, gy)) = response
                .interact_pointer_pos()
                .and_then(|pos| self.screen_to_grid(response.rect, pos))
            {
                self.cursor_cell = (gx, gy);
                self.apply_brush_at(gx, gy);
            }
        }

        if response.has_focus() {
            let mut moved = false;
            ui.input(|i| {
                if i.key_pressed(egui::Key::ArrowLeft) {
                    self.cursor_cell.0 = self.cursor_cell.0.saturating_sub(1);
                    moved = true;
                }
                if i.key_pressed(egui::Key::ArrowRight) {
                    self.cursor_cell.0 =
                        (self.cursor_cell.0 + 1).min(self.engine.config.width.saturating_sub(1));
                    moved = true;
                }
                if i.key_pressed(egui::Key::ArrowUp) {
                    self.cursor_cell.1 =
                        (self.cursor_cell.1 + 1).min(self.engine.config.height.saturating_sub(1));
                    moved = true;
                }
                if i.key_pressed(egui::Key::ArrowDown) {
                    self.cursor_cell.1 = self.cursor_cell.1.saturating_sub(1);
                    moved = true;
                }
                if i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space) {
                    let (gx, gy) = self.cursor_cell;
                    self.apply_brush_at(gx, gy);
                }
            });
            if moved {
                ui.ctx().request_repaint();
            }
        }
    }

    fn paint_cursor_overlay(
        &self,
        ui: &egui::Ui,
        response: &egui::Response,
        painter: &egui::Painter,
    ) {
        let cell_w = response.rect.width() / (self.engine.config.width as f32);
        let cell_h = response.rect.height() / (self.engine.config.height as f32);
        let target_cell = response
            .hover_pos()
            .and_then(|pos| self.screen_to_grid(response.rect, pos))
            .or_else(|| response.has_focus().then_some(self.cursor_cell));
        if let Some((gx, gy)) = target_cell {
            let py = self.engine.config.height.saturating_sub(1).saturating_sub(gy);
            let center = egui::pos2(
                response.rect.min.x + (gx as f32 + 0.5) * cell_w,
                response.rect.min.y + (py as f32 + 0.5) * cell_h,
            );
            let palette = SemanticPalette::from_visuals(ui.visuals());
            painter.circle_stroke(
                center,
                (self.brush_radius as f32) * cell_w.max(1.0),
                egui::Stroke::new(1.5_f32, palette.accent),
            );
        }
    }
}
