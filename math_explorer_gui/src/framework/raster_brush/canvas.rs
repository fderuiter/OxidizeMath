#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::bresenham::{cell_screen_rect, compute_aspect_locked_rect, map_pointer_to_cell};
use super::state::{RasterBrushState, RasterBrushStroke};
use super::stencils::{generate_brush_cells, BrushStencil};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleHoverText,
};
use crate::app::SemanticPalette;
use eframe::egui::{self, Color32, Key, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Vec2};

/// Reusable aspect-locked 2D raster sculpting widget with Bresenham strokes and WCAG 2.2 AA reticle.
pub struct RasterBrushCanvas<'a> {
    id_salt: &'a str,
    grid_w: usize,
    grid_h: usize,
    flip_y: bool,
    role: egui::accesskit::Role,
    label: &'a str,
    value_summary: Option<&'a str>,
    max_canvas_size: Option<Vec2>,
}

impl<'a> RasterBrushCanvas<'a> {
    /// Creates a new raster brush canvas for a `grid_w x grid_h` discrete grid.
    #[must_use]
    pub fn new(id_salt: &'a str, grid_w: usize, grid_h: usize, label: &'a str) -> Self {
        Self {
            id_salt,
            grid_w,
            grid_h,
            flip_y: false,
            role: egui::accesskit::Role::Image,
            label,
            value_summary: None,
            max_canvas_size: None,
        }
    }

    /// Configures bottom-up Y-axis mapping (`screen top -> grid_h - 1`).
    #[must_use]
    pub const fn flip_y(mut self, flip_y: bool) -> Self {
        self.flip_y = flip_y;
        self
    }

    /// Overrides the `AccessKit` semantic role (defaults to `Role::Image`).
    #[must_use]
    pub const fn role(mut self, role: egui::accesskit::Role) -> Self {
        self.role = role;
        self
    }

    /// Attaches a live dynamic value summary exposed to screen readers alongside reticle position.
    #[must_use]
    pub const fn value_summary(mut self, summary: &'a str) -> Self {
        self.value_summary = Some(summary);
        self
    }

    /// Caps the maximum visual size while preserving aspect lock inside the available UI region.
    #[must_use]
    pub const fn max_canvas_size(mut self, size: Vec2) -> Self {
        self.max_canvas_size = Some(size);
        self
    }

    /// Renders a texture-backed 2D simulation grid and returns `(Response, Option<RasterBrushStroke>)`.
    pub fn show_texture(
        self,
        ui: &mut egui::Ui,
        texture: &egui::TextureHandle,
        state: &mut RasterBrushState,
    ) -> (Response, Option<RasterBrushStroke>) {
        let tex_id = texture.id();
        self.show_custom(ui, state, |painter, rect, _palette| {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex_id, rect, uv, Color32::WHITE);
        })
    }

    /// Renders a custom-painted 2D grid and processes pointer + keyboard brush interactions.
    pub fn show_custom<F>(
        self,
        ui: &mut egui::Ui,
        state: &mut RasterBrushState,
        paint_grid: F,
    ) -> (Response, Option<RasterBrushStroke>)
    where
        F: FnOnce(&egui::Painter, Rect, &SemanticPalette),
    {
        state.clamp_to_grid(self.grid_w, self.grid_h);
        let avail = ui.available_rect_before_wrap();
        let bounded_avail = if let Some(cap) = self.max_canvas_size {
            let w = avail.width().min(cap.x).max(1.0);
            let h = avail.height().min(cap.y).max(1.0);
            Rect::from_min_size(avail.min, Vec2::new(w, h))
        } else {
            avail
        };
        let target_rect = compute_aspect_locked_rect(bounded_avail, self.grid_w, self.grid_h);
        let sense = Sense::click_and_drag().union(Sense::focusable_noninteractive());
        let id = ui.make_persistent_id(self.id_salt);
        let raw_space_pressed = ui.memory(|m| m.has_focus(id))
            && ui.input(|i| {
                i.events.iter().any(|e| {
                    matches!(
                        e,
                        egui::Event::Key {
                            key: Key::Space,
                            pressed: true,
                            repeat: false,
                            ..
                        }
                    )
                })
            });
        let response = ui.interact(target_rect, id, sense);
        ui.advance_cursor_after_rect(target_rect);

        let palette = SemanticPalette::from_visuals(ui.visuals());
        let painter = ui.painter_at(target_rect);
        paint_grid(&painter, target_rect, &palette);
        painter.rect_stroke(
            target_rect,
            0.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            StrokeKind::Inside,
        );

        let a11y_value = self.compose_a11y_value(state);
        register_canvas_accessibility(ui, &response, self.role, self.label, Some(&a11y_value));
        let response = response.accessible_hover_text(self.label);
        lock_canvas_arrow_keys(ui, &response);

        let pointer_stroke = self.handle_pointer(ui, &response, target_rect, state);
        let key_stroke = self.handle_keyboard(ui, &response, raw_space_pressed, state);
        self.paint_overlays(&painter, target_rect, &response, state, &palette);
        paint_canvas_focus_ring(ui, &response);

        state.focused_id = response.has_focus().then_some(response.id);
        (response, pointer_stroke.or(key_stroke))
    }

    fn compose_a11y_value(&self, state: &RasterBrushState) -> String {
        let base = format!(
            "Reticle ({}, {}), Stencil {}, Radius {}",
            state.reticle.0,
            state.reticle.1,
            state.stencil.label(),
            state.radius
        );
        match self.value_summary {
            Some(extra) => format!("{extra}; {base}"),
            None => base,
        }
    }

    fn handle_pointer(
        &self,
        ui: &egui::Ui,
        response: &Response,
        rect: Rect,
        state: &mut RasterBrushState,
    ) -> Option<RasterBrushStroke> {
        let (primary_down, secondary_down, interact_pos) = ui.input(|i| {
            (
                i.pointer.primary_down(),
                i.pointer.secondary_down(),
                i.pointer.interact_pos(),
            )
        });
        let active_btn = primary_down || secondary_down;
        let pointer_cell = interact_pos
            .and_then(|pos| map_pointer_to_cell(rect, pos, self.grid_w, self.grid_h, self.flip_y));

        if state.stencil == BrushStencil::Line {
            return self.handle_line_pointer(
                response,
                active_btn,
                secondary_down,
                pointer_cell,
                state,
            );
        }

        let interacting = (response.is_pointer_button_down_on()
            || response.dragged()
            || response.clicked()
            || response.secondary_clicked())
            && pointer_cell.is_some();

        if interacting {
            let cell = pointer_cell?;
            let secondary = secondary_down || response.secondary_clicked();
            state.reticle = cell;
            let was_dragging = state.drag_prev.is_some();
            if !state.stencil.is_continuous_drag() && was_dragging {
                state.drag_prev = Some(cell);
                return None;
            }
            let from = state.drag_prev.unwrap_or(cell);
            state.drag_prev = if active_btn { Some(cell) } else { None };
            state.drag_secondary = secondary;
            let cells = generate_brush_cells(
                state.stencil,
                state.symmetry,
                from,
                cell,
                state.radius,
                self.grid_w,
                self.grid_h,
            );
            return Some(RasterBrushStroke {
                cells,
                secondary,
                via_keyboard: false,
                focus_cell: cell,
            });
        }

        if !active_btn {
            state.drag_prev = None;
        }
        None
    }

    fn handle_line_pointer(
        &self,
        response: &Response,
        active_btn: bool,
        secondary_down: bool,
        pointer_cell: Option<(usize, usize)>,
        state: &mut RasterBrushState,
    ) -> Option<RasterBrushStroke> {
        if (response.drag_started() || response.is_pointer_button_down_on()) && active_btn {
            if let Some(cell) = pointer_cell {
                state.reticle = cell;
                if state.line_anchor.is_none() {
                    state.line_anchor = Some(cell);
                    state.drag_secondary = secondary_down;
                }
            }
            return None;
        }
        if (response.drag_stopped() || response.clicked() || response.secondary_clicked())
            && (state.line_anchor.is_some() || pointer_cell.is_some())
        {
            let end_cell = pointer_cell.unwrap_or(state.reticle);
            let start_cell = state.line_anchor.take().unwrap_or(end_cell);
            let secondary = state.drag_secondary || response.secondary_clicked();
            state.drag_secondary = false;
            state.reticle = end_cell;
            let cells = generate_brush_cells(
                BrushStencil::Line,
                state.symmetry,
                start_cell,
                end_cell,
                state.radius,
                self.grid_w,
                self.grid_h,
            );
            return Some(RasterBrushStroke {
                cells,
                secondary,
                via_keyboard: false,
                focus_cell: end_cell,
            });
        }
        None
    }

    fn handle_keyboard(
        &self,
        ui: &egui::Ui,
        response: &Response,
        raw_space_pressed: bool,
        state: &mut RasterBrushState,
    ) -> Option<RasterBrushStroke> {
        if !response.has_focus() || self.grid_w == 0 || self.grid_h == 0 {
            return None;
        }
        let step = ui.input(|i| if i.modifiers.shift { 5usize } else { 1usize });
        let mut moved = false;
        ui.input(|i| {
            if i.key_pressed(Key::ArrowLeft) {
                state.reticle.0 = state.reticle.0.saturating_sub(step);
                moved = true;
            }
            if i.key_pressed(Key::ArrowRight) {
                state.reticle.0 = (state.reticle.0 + step).min(self.grid_w - 1);
                moved = true;
            }
            if i.key_pressed(Key::ArrowUp) {
                state.reticle.1 = if self.flip_y {
                    (state.reticle.1 + step).min(self.grid_h - 1)
                } else {
                    state.reticle.1.saturating_sub(step)
                };
                moved = true;
            }
            if i.key_pressed(Key::ArrowDown) {
                state.reticle.1 = if self.flip_y {
                    state.reticle.1.saturating_sub(step)
                } else {
                    (state.reticle.1 + step).min(self.grid_h - 1)
                };
                moved = true;
            }
        });
        if moved {
            announce_live_status(
                ui.ctx(),
                format!(
                    "Reticle moved to ({}, {})",
                    state.reticle.0, state.reticle.1
                ),
            );
            ui.ctx().request_repaint();
        }
        self.handle_keyboard_stamp(ui, raw_space_pressed, state)
    }

    fn handle_keyboard_stamp(
        &self,
        ui: &egui::Ui,
        raw_space_pressed: bool,
        state: &mut RasterBrushState,
    ) -> Option<RasterBrushStroke> {
        let space = raw_space_pressed || ui.input(|i| i.key_pressed(Key::Space));
        let erase = ui.input(|i| i.key_pressed(Key::Backspace) || i.key_pressed(Key::Delete));
        if !space && !erase {
            return None;
        }
        if erase && state.line_anchor.take().is_some() {
            announce_live_status(ui.ctx(), "Cancelled line anchor");
            return None;
        }
        let from = if state.stencil == BrushStencil::Line && space {
            match state.line_anchor.take() {
                Some(anchor) => anchor,
                None => {
                    state.line_anchor = Some(state.reticle);
                    announce_live_status(
                        ui.ctx(),
                        format!(
                            "Line anchor set at ({}, {}); move reticle and press Space to complete",
                            state.reticle.0, state.reticle.1
                        ),
                    );
                    return None;
                }
            }
        } else {
            state.reticle
        };

        let cells = generate_brush_cells(
            state.stencil,
            state.symmetry,
            from,
            state.reticle,
            state.radius,
            self.grid_w,
            self.grid_h,
        );
        let action_verb = if erase { "Cleared" } else { "Applied" };
        announce_live_status(
            ui.ctx(),
            format!(
                "{action_verb} {} at ({}, {}) ({} cells)",
                state.stencil.label(),
                state.reticle.0,
                state.reticle.1,
                cells.len()
            ),
        );
        Some(RasterBrushStroke {
            cells,
            secondary: erase,
            via_keyboard: true,
            focus_cell: state.reticle,
        })
    }

    fn paint_overlays(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        response: &Response,
        state: &RasterBrushState,
        palette: &SemanticPalette,
    ) {
        if let Some((ax, ay)) = state.line_anchor {
            let start_c =
                cell_screen_rect(rect, ax, ay, self.grid_w, self.grid_h, self.flip_y).center();
            let hover_cell = response
                .hover_pos()
                .and_then(|p| map_pointer_to_cell(rect, p, self.grid_w, self.grid_h, self.flip_y))
                .unwrap_or(state.reticle);
            let end_c = cell_screen_rect(
                rect,
                hover_cell.0,
                hover_cell.1,
                self.grid_w,
                self.grid_h,
                self.flip_y,
            )
            .center();
            painter.line_segment([start_c, end_c], Stroke::new(2.0_f32, palette.accent));
        }

        if response.has_focus() {
            let cell_r = cell_screen_rect(
                rect,
                state.reticle.0,
                state.reticle.1,
                self.grid_w,
                self.grid_h,
                self.flip_y,
            );
            let min_box = Vec2::new(cell_r.width().max(8.0), cell_r.height().max(8.0));
            let reticle_rect = Rect::from_center_size(cell_r.center(), min_box);
            painter.rect_stroke(
                reticle_rect,
                1.0,
                Stroke::new(3.0_f32, palette.surface_fill),
                StrokeKind::Outside,
            );
            painter.rect_stroke(
                reticle_rect,
                1.0,
                Stroke::new(2.0_f32, palette.warning),
                StrokeKind::Middle,
            );
        }
    }
}
