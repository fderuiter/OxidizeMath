//! RLE soundboard & refractive medium sculpting via [`SpatialBrushKit`] for [`super::ChladniWaveTankTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use eframe::egui::{self, EventFilter, Key, Modifiers, Pos2, Rect, Shape};
use egui_plot::focus_ring_stroke;
use math_explorer::math_kernel::types::flatten_2d_index;

pub(crate) use super::sculpt_codec::*;
use super::{ChladniWaveTankTool, GRID_SIZE};
use crate::accessibility::announce_live_status;
use crate::framework::{
    rasterize_disk_stroke, BrushInteractionMode, BrushMode, BrushStroke, InteractionContext,
    InteractiveTool, SpatialBrushKit,
};

pub(crate) fn default_chladni_brush_kit() -> SpatialBrushKit {
    let mut kit = SpatialBrushKit::new(BrushMode::PotentialBarrier, 3.0)
        .with_interaction_mode(BrushInteractionMode::Pan)
        .with_radius_bounds(1.0, 12.0, 1.0)
        .with_cursor_step(3.0, 3.0);
    kit.cursor_pos = [64.0, 64.0];
    kit
}

fn stroke_cell_role(mode: BrushMode, dist_sq: i32, r_sq: i32) -> AcousticCellRole {
    match mode {
        BrushMode::Solid | BrushMode::ZeroFluxWall => AcousticCellRole::RigidClamp,
        BrushMode::PotentialBarrier => AcousticCellRole::CarbonBrace,
        BrushMode::VelocityJet | BrushMode::PotentialStep => AcousticCellRole::SpruceRib,
        BrushMode::PotentialWell => {
            if r_sq <= 1 || dist_sq * 5 <= r_sq {
                AcousticCellRole::DeepSlowLens
            } else if dist_sq * 5 <= r_sq * 3 {
                AcousticCellRole::MediumSlowLens
            } else {
                AcousticCellRole::MildSlowLens
            }
        }
        _ => AcousticCellRole::Baseline,
    }
}

pub(crate) fn rasterize_stroke_onto_mask(
    sculpt_mask: &mut [u8],
    width: usize,
    height: usize,
    stroke: &BrushStroke,
) {
    let (x0, y0) = SpatialBrushKit::plot_to_grid_y_up(stroke.start, width, height);
    let (x1, y1) = SpatialBrushKit::plot_to_grid_y_up(stroke.end, width, height);
    let r = (stroke.radius.round() as i32).clamp(0, 20);
    let r_sq = (r * r).max(1);

    if stroke.mode == BrushMode::PotentialWell && r >= 2 {
        for (cx, cy) in crate::framework::bresenham_line(x0, y0, x1, y1) {
            for y in (cy - r)..=(cy + r) {
                if y < 0 || y >= height as i32 {
                    continue;
                }
                for x in (cx - r)..=(cx + r) {
                    let d2 = (x - cx).pow(2) + (y - cy).pow(2);
                    if x >= 0 && x < width as i32 && d2 <= r_sq {
                        let role = stroke_cell_role(stroke.mode, d2, r_sq);
                        sculpt_mask[flatten_2d_index(x as usize, y as usize, width)] = role as u8;
                    }
                }
            }
        }
    } else {
        let code = stroke_cell_role(stroke.mode, 0, r_sq) as u8;
        rasterize_disk_stroke((x0, y0), (x1, y1), r, (width, height), |gx, gy| {
            sculpt_mask[flatten_2d_index(gx, gy, width)] = code;
        });
    }
}

impl ChladniWaveTankTool {
    pub(crate) fn apply_sculpt_stroke(&mut self, stroke: BrushStroke) {
        rasterize_stroke_onto_mask(&mut self.sculpt_mask, GRID_SIZE, GRID_SIZE, &stroke);
        self.state.sculpt_rle = encode_bounded_sculpt_rle(&self.sculpt_mask, GRID_SIZE, GRID_SIZE);
        self.rebuild_masks();
        self.solver.reset_detector();
        self.analysis_dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    pub(crate) fn clear_sculpt_mask(&mut self) {
        self.sculpt_mask.fill(0);
        self.state.sculpt_rle.clear();
        self.rebuild_masks();
        self.solver.reset_detector();
        self.analysis_dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    pub(super) fn draw_sculpt_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Canvas Interaction & Luthiery Brush:");
        ui.horizontal_wrapped(|ui| {
            let is_driver = !self.brush_kit.is_sculpting() && !self.pickup_drag_mode;
            if ui.selectable_label(is_driver, "🎯 Driver").clicked() {
                self.brush_kit.interaction_mode = BrushInteractionMode::Pan;
                self.pickup_drag_mode = false;
            }
            let is_pickup = !self.brush_kit.is_sculpting() && self.pickup_drag_mode;
            if ui.selectable_label(is_pickup, "🎙 Pickup").clicked() {
                self.brush_kit.interaction_mode = BrushInteractionMode::Pan;
                self.pickup_drag_mode = true;
            }
            if ui
                .selectable_label(self.brush_kit.is_sculpting(), "🖌 Sculpt")
                .clicked()
            {
                self.brush_kit.interaction_mode = BrushInteractionMode::Sculpt;
                self.pickup_drag_mode = false;
            }
        });

        if self.brush_kit.is_sculpting() {
            let role_label = match self.brush_kit.mode {
                BrushMode::Solid | BrushMode::ZeroFluxWall => "Clamp",
                BrushMode::PotentialBarrier => "Spruce Rib",
                BrushMode::VelocityJet | BrushMode::PotentialStep => "Fast Channel",
                BrushMode::PotentialWell => "Slow GRIN Lens",
                _ => "Erase",
            };
            egui::ComboBox::from_id_salt("chladni_brush_role")
                .selected_text(format!("Brush: {role_label}"))
                .show_ui(ui, |ui| {
                    for (mode, label) in [
                        (BrushMode::Solid, "Clamp"),
                        (BrushMode::PotentialBarrier, "Spruce Rib"),
                        (BrushMode::VelocityJet, "Fast Channel"),
                        (BrushMode::PotentialWell, "Slow GRIN Lens"),
                        (BrushMode::Erase, "Erase"),
                    ] {
                        ui.selectable_value(&mut self.brush_kit.mode, mode, label);
                    }
                });
            ui.add(egui::Slider::new(&mut self.brush_kit.radius, 1.0..=12.0).text("Brush R"));
        }
        if !self.state.sculpt_rle.is_empty() && ui.small_button("Clear Sculpt").clicked() {
            self.clear_sculpt_mask();
        }
    }

    pub(super) fn handle_sculpt_canvas_input(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        rect: Rect,
    ) {
        let ctx = ui.ctx().clone();
        if response.has_focus() {
            ctx.memory_mut(|m| {
                m.set_focus_lock_filter(
                    response.id,
                    EventFilter {
                        tab: false,
                        horizontal_arrows: true,
                        vertical_arrows: true,
                        escape: false,
                    },
                );
            });
            self.sample_sculpt_keyboard(&ctx, response);
        }
        self.sample_sculpt_pointer(&ctx, response, rect);
    }

    fn sample_sculpt_keyboard(&mut self, ctx: &egui::Context, response: &egui::Response) {
        let (stamped, pulsed, mut msg) = ctx.input_mut(|i| self.consume_sculpt_keys(i));
        if stamped.is_some() || pulsed || msg.is_some() {
            ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
        }
        if let Some(stroke) = stamped {
            let ic = InteractionContext::from_brush_stroke(response, stroke);
            self.on_brush(&ic);
            msg = Some(format!(
                "Stamped {} at ({:.0}, {:.0})",
                stroke.mode.label(),
                stroke.end[0],
                stroke.end[1]
            ));
        }
        if pulsed {
            self.trigger_mallet_pulse(self.state.driver_x, self.state.driver_y, 0.045, 1.0, 1.0);
            msg = Some("Injected Gaussian wave pulse".to_string());
        }
        if let Some(text) = msg {
            announce_live_status(ctx, text);
            ctx.request_repaint();
        }
    }

    fn consume_sculpt_keys(
        &mut self,
        i: &mut egui::InputState,
    ) -> (Option<BrushStroke>, bool, Option<String>) {
        if i.modifiers.ctrl || i.modifiers.command || i.modifiers.alt {
            return (None, false, None);
        }
        let max_c = (GRID_SIZE - 1) as f64;
        let step = self.brush_kit.cursor_step;
        let mut msg = None;
        let mut moved = false;
        for (key, dx, dy) in [
            (Key::ArrowLeft, -step[0], 0.0),
            (Key::ArrowRight, step[0], 0.0),
            (Key::ArrowUp, 0.0, step[1]),
            (Key::ArrowDown, 0.0, -step[1]),
        ] {
            if i.consume_key(Modifiers::SHIFT, key) {
                self.brush_kit.cursor_pos[0] += 3.0 * dx;
                self.brush_kit.cursor_pos[1] += 3.0 * dy;
                moved = true;
            } else if i.consume_key(Modifiers::NONE, key) {
                self.brush_kit.cursor_pos[0] += dx;
                self.brush_kit.cursor_pos[1] += dy;
                moved = true;
            }
        }
        if moved {
            self.brush_kit.cursor_pos[0] = self.brush_kit.cursor_pos[0].clamp(0.0, max_c);
            self.brush_kit.cursor_pos[1] = self.brush_kit.cursor_pos[1].clamp(0.0, max_c);
            msg = Some(format!(
                "Brush reticle at ({:.0}, {:.0}), mode {}",
                self.brush_kit.cursor_pos[0],
                self.brush_kit.cursor_pos[1],
                self.brush_kit.mode.label()
            ));
        }
        for (key, delta) in [(Key::OpenBracket, -1.0), (Key::CloseBracket, 1.0)] {
            if i.consume_key(Modifiers::NONE, key) {
                self.brush_kit.radius = (self.brush_kit.radius + delta)
                    .clamp(self.brush_kit.min_radius, self.brush_kit.max_radius);
                msg = Some(format!("Brush radius {:.1}", self.brush_kit.radius));
            }
        }
        let stamped = i
            .consume_key(Modifiers::NONE, Key::Space)
            .then_some(BrushStroke {
                start: self.brush_kit.cursor_pos,
                end: self.brush_kit.cursor_pos,
                radius: self.brush_kit.radius,
                mode: self.brush_kit.mode,
                drag_vector: [0.0, 0.0],
            });
        let pulsed = i.consume_key(Modifiers::NONE, Key::Enter);
        (stamped, pulsed, msg)
    }

    fn sample_sculpt_pointer(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        rect: Rect,
    ) {
        let (primary, secondary) =
            ctx.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down()));
        let active_mode = if primary {
            Some(self.brush_kit.mode)
        } else if secondary {
            Some(BrushMode::Erase)
        } else {
            None
        };
        let Some(mode) = active_mode else {
            self.last_brush_grid_pos = None;
            return;
        };
        if !(response.hovered() || response.dragged())
            || rect.width() <= 1.0
            || rect.height() <= 1.0
        {
            self.last_brush_grid_pos = None;
            return;
        }
        if let Some(pos) = response.interact_pointer_pos() {
            let gx = (((pos.x - rect.left()) / rect.width()) as f64 * (GRID_SIZE as f64))
                .clamp(0.0, (GRID_SIZE - 1) as f64);
            let gy = ((1.0 - (pos.y - rect.top()) / rect.height()) as f64 * (GRID_SIZE as f64))
                .clamp(0.0, (GRID_SIZE - 1) as f64);
            let curr = [gx, gy];
            let prev = self.last_brush_grid_pos.unwrap_or(curr);
            self.brush_kit.cursor_pos = curr;
            self.last_brush_grid_pos = Some(curr);
            let stroke = BrushStroke {
                start: prev,
                end: curr,
                radius: self.brush_kit.radius,
                mode,
                drag_vector: [curr[0] - prev[0], curr[1] - prev[1]],
            };
            let ic = InteractionContext::from_brush_stroke(response, stroke);
            self.on_brush(&ic);
        }
    }

    pub(super) fn paint_sculpt_reticle(
        &self,
        ui: &egui::Ui,
        painter: &egui::Painter,
        response: &egui::Response,
        rect: Rect,
    ) {
        if !self.brush_kit.is_sculpting() || !(response.has_focus() || response.hovered()) {
            return;
        }
        let stroke = focus_ring_stroke(ui.visuals(), ui.visuals().extreme_bg_color);
        let inv = 1.0 / (GRID_SIZE as f32);
        let cx = rect.left() + ((self.brush_kit.cursor_pos[0] as f32) + 0.5) * inv * rect.width();
        let cy =
            rect.bottom() - ((self.brush_kit.cursor_pos[1] as f32) + 0.5) * inv * rect.height();
        let center = Pos2::new(cx, cy);
        let px_radius = ((self.brush_kit.radius as f32) * inv * rect.width()).max(4.0);
        let clipped = painter.with_clip_rect(rect);
        clipped.add(Shape::circle_stroke(center, px_radius, stroke));
        let arm = 4.5_f32;
        clipped.line_segment([Pos2::new(cx - arm, cy), Pos2::new(cx + arm, cy)], stroke);
        clipped.line_segment([Pos2::new(cx, cy - arm), Pos2::new(cx, cy + arm)], stroke);
    }
}
