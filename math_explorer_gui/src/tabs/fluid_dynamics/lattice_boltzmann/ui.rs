//! Side-panel controls, aerodynamic telemetry HUD, and interactive sculpting canvas
//! for [`super::LatticeBoltzmannTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::render_lbm_color_image;
use super::state::{
    bluff_plate_obstacle_grid, cylinder_obstacle_grid, screen_pos_to_cell,
    streamlined_airfoil_obstacle_grid, LbmBrushMode, LbmTelemetry, LbmViewMode, AREA_BUDGET_MAX,
    AREA_BUDGET_MIN, BRUSH_RADIUS_MAX, BRUSH_RADIUS_MIN, INFLOW_SPEED_MAX, INFLOW_SPEED_MIN,
    LBM_CELL_COUNT, LBM_HEIGHT, LBM_INLET_GUARD_COLS, LBM_WIDTH, STEPS_PER_FRAME_MAX,
    STEPS_PER_FRAME_MIN, VISCOSITY_MAX, VISCOSITY_MIN,
};
use super::LatticeBoltzmannTool;
use crate::accessibility::{
    lock_canvas_arrow_keys, paint_canvas_focus_ring, AccessibleHoverText, AccessibleTheoryHover,
};
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::{SimCommand, StateSnapshot};
use crate::framework::{BrushMode, BrushStroke, PlaybackAction, PlaybackTransport};
use eframe::egui;
use scientific_metadata::theory::TheoryDescribable;

impl LatticeBoltzmannTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "lbm_wind_tunnel_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .default_width(250.0)
            .show_inside(parent_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Wind Tunnel Controls");
                    ui.separator();
                    self.draw_fault_and_transport(ui, ctx);
                    ui.separator();
                    self.draw_telemetry_hud(ui);
                    ui.separator();
                    self.draw_flow_controls(ui, ctx);
                    ui.separator();
                    self.draw_sculpt_controls(ui, ctx);
                });
            });
    }

    fn draw_fault_and_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(WorkerFaultAction::Reset | WorkerFaultAction::ResetParameters) =
            render_worker_fault_banner(
                ui,
                self.worker.state(),
                self.worker.diagnostic_report(),
                true,
            )
        {
            self.reset_parameters_and_flow();
            ctx.request_repaint();
            ui.separator();
        }
        if let Some(action) = PlaybackTransport::from_worker_state(
            "lbm_wind_tunnel_transport",
            "Lattice Boltzmann Wind Tunnel",
            self.worker.state(),
        )
        .can_reset(true)
        .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    let _ = self.worker.send_command(SimCommand::Start);
                }
                PlaybackAction::Pause => {
                    let _ = self.worker.send_command(SimCommand::Pause);
                }
                PlaybackAction::Step => {
                    let _ = self.worker.send_command(SimCommand::Step(1));
                }
                PlaybackAction::Reset => {
                    self.reset_flow_field();
                    ctx.request_repaint();
                }
            }
        }
    }

    fn draw_telemetry_hud(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        egui::Frame::new()
            .fill(palette.surface_fill)
            .stroke(egui::Stroke::new(1.0_f32, palette.graph_edge))
            .corner_radius(egui::CornerRadius::same(6))
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("Aerodynamic Telemetry").strong());
                egui::Grid::new("lbm_telemetry_grid")
                    .num_columns(2)
                    .spacing([12.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Drag Coeff (C_D):");
                        ui.monospace(format!("{:+.3}", self.telemetry.cd));
                        ui.end_row();
                        ui.label("Lift Coeff (C_L):");
                        ui.monospace(format!("{:+.3}", self.telemetry.cl));
                        ui.end_row();
                        ui.label("Reynolds (Re):");
                        ui.monospace(format!("{:.1}", self.telemetry.reynolds));
                        ui.end_row();
                        ui.label("Solid Cells:");
                        ui.monospace(format!(
                            "{} / {}",
                            self.telemetry.solid_cells, self.area_budget
                        ));
                        ui.end_row();
                        ui.label("Frontal Height (D):");
                        ui.monospace(format!("{} lu", self.telemetry.frontal_height));
                        ui.end_row();
                    });
            });
    }

    fn draw_flow_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label(egui::RichText::new("Flow & Field View").strong());
        ui.horizontal(|ui| {
            let speed_clicked = ui
                .radio_value(&mut self.view_mode, LbmViewMode::Speed, "Speed |u|")
                .clicked();
            let vort_clicked = ui
                .radio_value(&mut self.view_mode, LbmViewMode::Vorticity, "Vorticity ωz")
                .clicked();
            if speed_clicked || vort_clicked {
                self.push_shared_config();
                self.dirty = true;
                ctx.request_repaint();
            }
        });

        let flow_unlocked = self.active_challenge.is_none();
        if ui
            .add_enabled(
                flow_unlocked,
                egui::Slider::new(&mut self.viscosity, VISCOSITY_MIN..=VISCOSITY_MAX)
                    .logarithmic(true)
                    .text("Viscosity (ν)"),
            )
            .changed()
        {
            self.push_shared_config();
            self.refresh_idle_telemetry();
        }
        if ui
            .add_enabled(
                flow_unlocked,
                egui::Slider::new(&mut self.inflow_speed, INFLOW_SPEED_MIN..=INFLOW_SPEED_MAX)
                    .text("Inflow Speed (U)"),
            )
            .changed()
        {
            self.push_shared_config();
            self.refresh_idle_telemetry();
            self.dirty = true;
        }
        if ui
            .add(
                egui::Slider::new(
                    &mut self.steps_per_frame,
                    STEPS_PER_FRAME_MIN..=STEPS_PER_FRAME_MAX,
                )
                .text("Steps / Frame"),
            )
            .changed()
        {
            self.push_shared_config();
        }
    }

    fn draw_sculpt_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label(egui::RichText::new("Obstacle Sculpting").strong());
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.brush_mode, LbmBrushMode::Solid, "Solid");
            ui.radio_value(&mut self.brush_mode, LbmBrushMode::Erase, "Erase");
            ui.radio_value(&mut self.brush_mode, LbmBrushMode::VelocityJet, "Jet");
        });
        ui.add(
            egui::Slider::new(&mut self.brush_radius, BRUSH_RADIUS_MIN..=BRUSH_RADIUS_MAX)
                .text("Brush Radius"),
        );
        if ui
            .add(
                egui::Slider::new(&mut self.area_budget, AREA_BUDGET_MIN..=AREA_BUDGET_MAX)
                    .text("Area Budget"),
            )
            .changed()
        {
            self.enforce_current_budget();
            ctx.request_repaint();
        }
        self.draw_profile_buttons(ui, ctx);
        ui.small("Left-drag sculpts/jets • Right-drag erases • Inlet x < 2 protected");
    }

    fn draw_profile_buttons(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        fn preset_btn(ui: &mut egui::Ui, label: &str, hover: &str) -> bool {
            ui.button(label).accessible_hover_text(hover).clicked()
        }
        ui.label("Tunnel Profiles:");
        ui.horizontal_wrapped(|ui| {
            if preset_btn(ui, "● Cylinder", "Load circular cylinder vortex-street benchmark") {
                self.apply_obstacle_preset(cylinder_obstacle_grid());
                ctx.request_repaint();
            }
            if preset_btn(ui, "▮ Bluff Plate", "Load high-drag flat bluff plate obstacle") {
                self.apply_obstacle_preset(bluff_plate_obstacle_grid());
                ctx.request_repaint();
            }
            if preset_btn(ui, "➤ Min-Drag Airfoil", "Load streamlined airfoil benchmark") {
                self.apply_obstacle_preset(streamlined_airfoil_obstacle_grid());
                ctx.request_repaint();
            }
            if preset_btn(ui, "✈ NACA 2412", "Stamp cambered NACA 2412 airfoil at 8° AoA") {
                self.apply_naca_2412_preset(8.0);
                ctx.request_repaint();
            }
            if preset_btn(ui, "⏳ Venturi", "Stamp converging-diverging Venturi nozzle") {
                self.apply_venturi_preset();
                ctx.request_repaint();
            }
            for stamp in [
                super::super::lbm_fairing::FairingStamp::TeardropFairing,
                super::super::lbm_fairing::FairingStamp::SymmetricWedge,
            ] {
                if preset_btn(
                    ui,
                    &format!("♦ {}", stamp.label()),
                    "Stamp fairing geometry around protected R_core = 4",
                ) {
                    self.apply_fairing_stamp(stamp);
                    ctx.request_repaint();
                }
            }
            if preset_btn(ui, "✕ Clear", "Remove all solid obstacles from the wind tunnel") {
                self.apply_obstacle_preset(vec![false; LBM_CELL_COUNT]);
                ctx.request_repaint();
            }
        });
    }

    pub(super) fn draw_central_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let avail = ui.available_size();
            if avail.x <= 4.0 || avail.y <= 4.0 {
                return;
            }
            self.ensure_texture(ctx);
            let aspect = (LBM_WIDTH as f32) / (LBM_HEIGHT as f32);
            let fit_w = avail.x.min(avail.y * aspect).max(1.0);
            let fit_h = (fit_w / aspect).max(1.0);
            let fitted = egui::vec2(fit_w, fit_h);

            if let Some(texture) = self.texture.clone() {
                let alt_label = format!(
                    "{} — C_D {:+.3}, C_L {:+.3}, Re {:.1}, {} of {} solid cells",
                    self.theory_description(),
                    self.telemetry.cd,
                    self.telemetry.cl,
                    self.telemetry.reynolds,
                    self.telemetry.solid_cells,
                    self.area_budget
                );
                let response = ui
                    .add(
                        egui::Image::new(&texture)
                            .maintain_aspect_ratio(false)
                            .fit_to_exact_size(fitted)
                            .alt_text(alt_label)
                            .sense(egui::Sense::click_and_drag()),
                    )
                    .accessible_theory_hover(self);
                ui.painter().rect_stroke(
                    response.rect,
                    2.0_f32,
                    ui.visuals().widgets.noninteractive.bg_stroke,
                    egui::StrokeKind::Inside,
                );
                paint_canvas_focus_ring(ui, &response);
                lock_canvas_arrow_keys(ui, &response);
                self.focused_canvas = response.has_focus().then_some(response.id);
                self.handle_canvas_sculpting(ui, ctx, &response);
                self.paint_canvas_hud_overlay(ui, response.rect);
            }
        });
    }

    fn handle_canvas_sculpting(
        &mut self,
        ui: &egui::Ui,
        ctx: &egui::Context,
        response: &egui::Response,
    ) {
        let (primary_down, secondary_down, interact_pos) = ui.input(|i| {
            (
                i.pointer.primary_down(),
                i.pointer.secondary_down(),
                i.pointer.interact_pos(),
            )
        });
        if !(primary_down || secondary_down) {
            self.last_drag_cell = None;
        } else if response.hovered() || response.dragged() || response.clicked() {
            if let Some((gx, gy)) = interact_pos
                .filter(|p| response.rect.contains(*p))
                .and_then(|pos| screen_pos_to_cell(pos, response.rect))
            {
                let (x0, y0) = self.last_drag_cell.unwrap_or((gx, gy));
                self.last_drag_cell = Some((gx, gy));
                self.cursor_cell = (gx, gy);
                let mode = if secondary_down {
                    BrushMode::Erase
                } else {
                    match self.brush_mode {
                        LbmBrushMode::Solid => BrushMode::Solid,
                        LbmBrushMode::Erase => BrushMode::Erase,
                        LbmBrushMode::VelocityJet => BrushMode::VelocityJet,
                    }
                };
                let stroke = BrushStroke {
                    start: [x0 as f64 + 0.5, y0 as f64 + 0.5],
                    end: [gx as f64 + 0.5, gy as f64 + 0.5],
                    radius: f64::from(self.brush_radius),
                    mode,
                    drag_vector: [gx as f64 - x0 as f64, gy as f64 - y0 as f64],
                };
                self.apply_brush_stroke(&stroke);
                ctx.request_repaint();
            }
        }
        if response.has_focus() {
            self.handle_keyboard_sculpting(ui, ctx);
        }
    }

    fn handle_keyboard_sculpting(&mut self, ui: &egui::Ui, ctx: &egui::Context) {
        let (mut cx, mut cy) = self.cursor_cell;
        let mut moved = false;
        ui.input(|i| {
            if i.key_pressed(egui::Key::ArrowLeft) {
                cx = cx.saturating_sub(1).max(LBM_INLET_GUARD_COLS);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                cx = (cx + 1).min(LBM_WIDTH - 1);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                cy = cy.saturating_sub(1);
                moved = true;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                cy = (cy + 1).min(LBM_HEIGHT - 1);
                moved = true;
            }
            if i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter) {
                let paint = self.brush_mode == LbmBrushMode::Solid;
                self.sculpt_at_cell(cx, cy, paint);
                ctx.request_repaint();
            } else if i.key_pressed(egui::Key::Backspace) || i.key_pressed(egui::Key::Delete) {
                self.sculpt_at_cell(cx, cy, false);
                ctx.request_repaint();
            }
        });
        if moved {
            self.cursor_cell = (cx, cy);
            ctx.request_repaint();
        }
    }

    fn paint_canvas_hud_overlay(&self, ui: &egui::Ui, rect: egui::Rect) {
        if rect.width() < 180.0 || rect.height() < 60.0 {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let badge_text = format!(
            "CD {:+.3}  |  CL {:+.3}  |  Re {:.1}  |  Solid {}/{}",
            self.telemetry.cd,
            self.telemetry.cl,
            self.telemetry.reynolds,
            self.telemetry.solid_cells,
            self.area_budget
        );
        let text_color = ui.visuals().text_color();
        let galley = ui.painter().layout_no_wrap(
            badge_text,
            egui::FontId::monospace(11.5),
            text_color,
        );
        let pad = egui::vec2(8.0, 4.0);
        let badge_rect = egui::Rect::from_min_size(
            egui::pos2(
                rect.right() - galley.size().x - pad.x * 2.0 - 8.0,
                rect.top() + 8.0,
            ),
            galley.size() + pad * 2.0,
        );
        ui.painter().rect(
            badge_rect,
            4.0_f32,
            palette.surface_fill.gamma_multiply(0.88),
            egui::Stroke::new(1.0_f32, palette.graph_edge),
            egui::StrokeKind::Inside,
        );
        ui.painter()
            .galley(badge_rect.min + pad, galley, text_color);
    }

    pub(crate) fn ensure_texture(&mut self, ctx: &egui::Context) {
        if self.texture.is_some() && !self.dirty {
            return;
        }
        let image = render_lbm_color_image(
            &self.build_snapshot_solver(),
            self.inflow_speed,
            self.view_mode,
            self.high_contrast,
        );
        if let Some(tex) = &mut self.texture {
            tex.set(image, egui::TextureOptions::NEAREST);
        } else {
            self.texture = Some(ctx.load_texture(
                "lbm_wind_tunnel_texture",
                image,
                egui::TextureOptions::NEAREST,
            ));
        }
        self.dirty = false;
    }

    pub(super) fn sync_from_worker(&mut self, ctx: &egui::Context) {
        if let Some(snapshot) = self.worker.update() {
            Self::upload_worker_snapshot(
                &mut self.telemetry,
                &mut self.texture,
                &mut self.dirty,
                ctx,
                snapshot,
            );
        } else if self.dirty {
            self.ensure_texture(ctx);
        }
        if self.worker.is_running() {
            ctx.request_repaint();
        }
    }

    fn upload_worker_snapshot(
        telemetry: &mut LbmTelemetry,
        texture_slot: &mut Option<egui::TextureHandle>,
        dirty: &mut bool,
        ctx: &egui::Context,
        snapshot: &StateSnapshot,
    ) {
        if let Some(telem) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<LbmTelemetry>())
            .copied()
        {
            *telemetry = telem;
        }
        if let Ok(pixels) = snapshot.pixels.try_read() {
            if pixels.len() == snapshot.width * snapshot.height && !pixels.is_empty() {
                let image =
                    egui::ColorImage::new([snapshot.width, snapshot.height], pixels.clone());
                if let Some(tex) = texture_slot {
                    tex.set(image, egui::TextureOptions::NEAREST);
                } else {
                    *texture_slot = Some(ctx.load_texture(
                        "lbm_wind_tunnel_texture",
                        image,
                        egui::TextureOptions::NEAREST,
                    ));
                }
                *dirty = false;
            }
        }
    }
}
