use super::Camera3D;
use crate::accessibility::AccessibleHoverText;
use eframe::egui;

const GIZMO_TOOLBAR_CLEARANCE_PX: f32 = 30.0_f32;
const GIZMO_MIN_TARGET_PX: f32 = 24.0_f32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum GizmoAxis {
    X,
    Y,
    Z,
    Iso,
}

impl GizmoAxis {
    pub(super) const fn short_label(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
            Self::Iso => "ISO",
        }
    }

    pub(super) const fn a11y_label(self) -> &'static str {
        match self {
            Self::X => "Orient camera to X",
            Self::Y => "Orient camera to Y",
            Self::Z => "Orient camera to Z",
            Self::Iso => "Orient camera to ISO",
        }
    }

    pub(super) const fn tooltip_text(self) -> &'static str {
        match self {
            Self::X => "Orient camera to X-axis (YZ plane)",
            Self::Y => "Orient camera to Y-axis (XZ plane)",
            Self::Z => "Orient camera to Z-axis (XY plane)",
            Self::Iso => "Orient camera to isometric 3D view",
        }
    }

    pub(super) fn apply_snap(self, camera: &mut Camera3D) {
        match self {
            Self::X => camera.snap_to_yz(),
            Self::Y => camera.snap_to_xz(),
            Self::Z => camera.snap_to_xy(),
            Self::Iso => camera.snap_to_iso(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GizmoHandle {
    pub(super) axis: GizmoAxis,
    pub(super) color: egui::Color32,
    pub(super) screen_pos: egui::Pos2,
    pub(super) button_rect: egui::Rect,
    pub(super) depth: f32,
}

fn build_gizmo_handle(
    center: egui::Pos2,
    half_btn: f32,
    btn_extent: f32,
    axis: GizmoAxis,
    color: egui::Color32,
    offset: (f32, f32, f32),
) -> GizmoHandle {
    let (dx, dy, depth) = offset;
    let screen_pos = egui::pos2(center.x + dx, center.y - dy);
    let min_pos = egui::pos2(
        (screen_pos.x - half_btn).round(),
        (screen_pos.y - half_btn).round(),
    );
    GizmoHandle {
        axis,
        color,
        screen_pos,
        button_rect: egui::Rect::from_min_size(min_pos, egui::Vec2::splat(btn_extent)),
        depth,
    }
}

impl Camera3D {
    pub(super) fn gizmo_geometry(
        &self,
        rect: egui::Rect,
        palette: &crate::app::SemanticPalette,
    ) -> (egui::Pos2, f32, f32, [GizmoHandle; 4]) {
        let min_dim = rect.width().min(rect.height());
        let scale = (min_dim / 160.0_f32).clamp(0.5_f32, 1.0_f32);
        let axis_length = 30.0_f32 * scale;
        let knob_radius = 8.0_f32 * scale;
        let btn_extent = (2.0_f32 * (knob_radius + 4.0_f32))
            .ceil()
            .max(GIZMO_MIN_TARGET_PX);
        let half_btn = 0.5_f32 * btn_extent;
        let pad = (45.0_f32 * scale).max(axis_length + half_btn + 2.0_f32);
        let toolbar_offset = if rect.height() >= 140.0_f32 {
            GIZMO_TOOLBAR_CLEARANCE_PX
        } else {
            0.0_f32
        };

        let center_x = (rect.max.x - pad).max(rect.min.x + half_btn);
        let max_cy = (rect.max.y - pad).max(rect.min.y + half_btn);
        let center_y = (rect.min.y + toolbar_offset + pad).min(max_cy);
        let center = egui::pos2(center_x, center_y);

        let (px_x, py_x, pz_x) = self.project_unit(&[1.0, 0.0, 0.0]);
        let (px_y, py_y, pz_y) = self.project_unit(&[0.0, 1.0, 0.0]);
        let (px_z, py_z, pz_z) = self.project_unit(&[0.0, 0.0, 1.0]);
        let mut handles = [
            build_gizmo_handle(
                center,
                half_btn,
                btn_extent,
                GizmoAxis::Iso,
                palette.surface_fill,
                (0.0, 0.0, -10.0_f32),
            ),
            build_gizmo_handle(
                center,
                half_btn,
                btn_extent,
                GizmoAxis::X,
                palette.axis_x,
                (px_x * axis_length, py_x * axis_length, pz_x),
            ),
            build_gizmo_handle(
                center,
                half_btn,
                btn_extent,
                GizmoAxis::Y,
                palette.axis_y,
                (px_y * axis_length, py_y * axis_length, pz_y),
            ),
            build_gizmo_handle(
                center,
                half_btn,
                btn_extent,
                GizmoAxis::Z,
                palette.axis_z,
                (px_z * axis_length, py_z * axis_length, pz_z),
            ),
        ];
        handles.sort_by(|a, b| a.depth.total_cmp(&b.depth));

        (center, axis_length, knob_radius, handles)
    }

    pub(super) fn is_inside_gizmo(&self, pos: egui::Pos2, rect: egui::Rect) -> bool {
        let palette = crate::app::ThemeMode::Dark.palette();
        let (center, axis_length, knob_radius, handles) = self.gizmo_geometry(rect, &palette);
        let bg_radius = axis_length + knob_radius + 4.0_f32;
        pos.distance(center) <= bg_radius || handles.iter().any(|h| h.button_rect.contains(pos))
    }

    pub(super) fn pick_frontmost_gizmo_axis(
        handles: &[GizmoHandle; 4],
        pos: egui::Pos2,
        knob_radius: f32,
    ) -> Option<GizmoAxis> {
        let hit_radius = knob_radius + 4.0_f32;
        if let Some(handle) = handles
            .iter()
            .rev()
            .find(|h| pos.distance(h.screen_pos) <= hit_radius)
        {
            return Some(handle.axis);
        }
        handles
            .iter()
            .rev()
            .find(|h| h.button_rect.contains(pos))
            .map(|h| h.axis)
    }

    fn allocate_gizmo_button(
        ui: &egui::Ui,
        base_id: egui::Id,
        handle: &GizmoHandle,
    ) -> egui::Response {
        let enabled = ui.is_enabled();
        let btn_id = base_id.with(("camera_gizmo_btn", handle.axis));
        let sense = egui::Sense::click() | egui::Sense::focusable_noninteractive();
        let resp = ui.interact(handle.button_rect, btn_id, sense);
        let label = handle.axis.a11y_label();
        resp.widget_info(move || {
            egui::WidgetInfo::labeled(egui::WidgetType::Button, enabled, label)
        });
        ui.ctx().accesskit_node_builder(resp.id, |builder| {
            builder.set_role(egui::accesskit::Role::Button);
            builder.set_label(label);
            if !enabled {
                builder.set_disabled();
            }
        });
        resp.accessible_hover_text(handle.axis.tooltip_text())
    }

    fn interact_gizmo_buttons(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        handles: &[GizmoHandle; 4],
        knob_radius: f32,
    ) -> (bool, Option<GizmoAxis>, [bool; 4]) {
        let pointer_clicked = ui.input(|i| i.pointer.any_click());
        let interact_pos = ui.input(|i| i.pointer.interact_pos());
        let hover_pos = ui.input(|i| i.pointer.hover_pos());
        let mut focused = [false; 4];
        let mut focused_axis = None;
        let mut any_hovered = false;
        let mut any_clicked = false;
        let mut last_clicked_axis = None;
        let mut kbd_clicked_axis = None;

        for (idx, handle) in handles.iter().enumerate() {
            let btn_resp = Self::allocate_gizmo_button(ui, response.id, handle);
            if btn_resp.has_focus() {
                focused[idx] = true;
                focused_axis = Some(handle.axis);
            }
            any_hovered |= btn_resp.hovered();
            if btn_resp.clicked() {
                any_clicked = true;
                last_clicked_axis = Some(handle.axis);
                if btn_resp.has_focus() && !pointer_clicked {
                    kbd_clicked_axis = Some(handle.axis);
                }
            }
        }

        let clicked_axis = if any_clicked {
            kbd_clicked_axis.or_else(|| {
                interact_pos
                    .and_then(|pos| Self::pick_frontmost_gizmo_axis(handles, pos, knob_radius))
                    .or(last_clicked_axis)
            })
        } else {
            None
        };
        if let Some(axis) = clicked_axis {
            axis.apply_snap(self);
        }

        let hovered_axis = if any_hovered {
            hover_pos
                .and_then(|pos| Self::pick_frontmost_gizmo_axis(handles, pos, knob_radius))
                .or(focused_axis)
        } else {
            focused_axis
        };
        (clicked_axis.is_some(), hovered_axis, focused)
    }

    fn render_gizmo_painter(
        &self,
        ui: &egui::Ui,
        metrics: (egui::Pos2, f32, f32),
        handles: &[GizmoHandle; 4],
        hovered_axis: Option<GizmoAxis>,
        focused: &[bool; 4],
    ) {
        let (center, axis_length, knob_radius) = metrics;
        let painter = ui.painter();
        let visuals = ui.visuals();
        let palette = crate::app::SemanticPalette::from_visuals(visuals);
        let scale = axis_length / 30.0_f32;
        let bg_radius = axis_length + knob_radius + 4.0_f32;
        let wf = visuals.window_fill();
        let backdrop_fill = egui::Color32::from_rgba_unmultiplied(wf.r(), wf.g(), wf.b(), 180);

        painter.circle_filled(center, bg_radius, backdrop_fill);
        painter.circle_stroke(
            center,
            bg_radius,
            egui::Stroke::new(1.0_f32, palette.graph_edge),
        );

        for (idx, handle) in handles.iter().enumerate() {
            let is_active = hovered_axis == Some(handle.axis) || focused[idx];
            let current_radius = if is_active {
                knob_radius * 1.25_f32
            } else {
                knob_radius
            };

            if handle.axis != GizmoAxis::Iso {
                painter.line_segment(
                    [center, handle.screen_pos],
                    egui::Stroke::new(2.5_f32 * scale, handle.color),
                );
            }

            painter.circle_filled(handle.screen_pos, current_radius, handle.color);
            painter.circle_stroke(
                handle.screen_pos,
                current_radius,
                egui::Stroke::new(
                    if is_active { 2.0_f32 } else { 1.0_f32 },
                    palette.node_stroke,
                ),
            );

            let font_size = if handle.axis == GizmoAxis::Iso {
                8.0_f32 * scale
            } else {
                10.0_f32 * scale
            };
            painter.text(
                handle.screen_pos,
                egui::Align2::CENTER_CENTER,
                handle.axis.short_label(),
                egui::FontId::proportional(font_size),
                crate::app::contrasting_text_color(handle.color),
            );

            if focused[idx] {
                painter.rect_stroke(
                    handle.button_rect,
                    4.0_f32,
                    egui::Stroke::new(2.0_f32, palette.accent),
                    egui::StrokeKind::Inside,
                );
            }
        }
    }

    /// Renders on-canvas interactive orientation gizmo and returns true if activated.
    pub fn draw_gizmo_overlay(&mut self, response: &egui::Response, ui: &egui::Ui) -> bool {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let (center, axis_length, knob_radius, handles) =
            self.gizmo_geometry(response.rect, &palette);
        let (handled_click, hovered_axis, focused) =
            self.interact_gizmo_buttons(response, ui, &handles, knob_radius);
        self.render_gizmo_painter(
            ui,
            (center, axis_length, knob_radius),
            &handles,
            hovered_axis,
            &focused,
        );
        handled_click
    }
}
