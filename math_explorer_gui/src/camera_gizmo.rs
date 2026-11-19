use super::Camera3D;
use eframe::egui;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GizmoAxis {
    X,
    Y,
    Z,
    Iso,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GizmoHandle {
    pub(super) axis: GizmoAxis,
    pub(super) label: &'static str,
    pub(super) color: egui::Color32,
    pub(super) screen_pos: egui::Pos2,
    pub(super) depth: f32,
}

impl Camera3D {
    pub(super) fn gizmo_geometry(
        &self,
        rect: egui::Rect,
        palette: &crate::app::SemanticPalette,
    ) -> (egui::Pos2, f32, f32, [GizmoHandle; 4]) {
        let min_dim = rect.width().min(rect.height());
        let scale = (min_dim / 160.0_f32).clamp(0.5_f32, 1.0_f32);
        let margin = 45.0_f32 * scale;

        let center_x = (rect.max.x - margin).max(rect.min.x + margin);
        let center_y = (rect.min.y + margin).min(rect.max.y - margin);
        let center = egui::pos2(center_x, center_y);

        let axis_length = 30.0_f32 * scale;
        let knob_radius = 8.0_f32 * scale;

        let (px_x, py_x, pz_x) = self.project_unit(&[1.0, 0.0, 0.0]);
        let (px_y, py_y, pz_y) = self.project_unit(&[0.0, 1.0, 0.0]);
        let (px_z, py_z, pz_z) = self.project_unit(&[0.0, 0.0, 1.0]);

        let pos_x = egui::pos2(center.x + px_x * axis_length, center.y - py_x * axis_length);
        let pos_y = egui::pos2(center.x + px_y * axis_length, center.y - py_y * axis_length);
        let pos_z = egui::pos2(center.x + px_z * axis_length, center.y - py_z * axis_length);

        let handles = [
            GizmoHandle {
                axis: GizmoAxis::Iso,
                label: "ISO",
                color: palette.surface_fill,
                screen_pos: center,
                depth: -10.0_f32,
            },
            GizmoHandle {
                axis: GizmoAxis::X,
                label: "X",
                color: palette.axis_x,
                screen_pos: pos_x,
                depth: pz_x,
            },
            GizmoHandle {
                axis: GizmoAxis::Y,
                label: "Y",
                color: palette.axis_y,
                screen_pos: pos_y,
                depth: pz_y,
            },
            GizmoHandle {
                axis: GizmoAxis::Z,
                label: "Z",
                color: palette.axis_z,
                screen_pos: pos_z,
                depth: pz_z,
            },
        ];

        (center, axis_length, knob_radius, handles)
    }

    pub(super) fn is_inside_gizmo(&self, pos: egui::Pos2, rect: egui::Rect) -> bool {
        let palette = crate::app::ThemeMode::Dark.palette();
        let (center, axis_length, knob_radius, _) = self.gizmo_geometry(rect, &palette);
        let bg_radius = axis_length + knob_radius + 4.0_f32;
        pos.distance(center) <= bg_radius
    }

    fn check_gizmo_click(
        &mut self,
        handles: &[GizmoHandle],
        pointer_pos: Option<egui::Pos2>,
        is_clicked: bool,
        knob_radius: f32,
    ) -> bool {
        let mut hovered_axis = None;
        if let Some(pos) = pointer_pos {
            let hit_threshold = knob_radius + 4.0_f32;
            for handle in handles {
                if pos.distance(handle.screen_pos) <= hit_threshold {
                    hovered_axis = Some(handle.axis);
                    break;
                }
            }
        }

        if is_clicked {
            if let Some(axis) = hovered_axis {
                match axis {
                    GizmoAxis::X => self.snap_to_yz(),
                    GizmoAxis::Y => self.snap_to_xz(),
                    GizmoAxis::Z => self.snap_to_xy(),
                    GizmoAxis::Iso => self.snap_to_iso(),
                }
                return true;
            }
        }
        false
    }

    fn render_gizmo_painter(
        &self,
        ui: &egui::Ui,
        center: egui::Pos2,
        axis_length: f32,
        knob_radius: f32,
        mut handles: Vec<GizmoHandle>,
        hovered_axis: Option<GizmoAxis>,
    ) {
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

        handles.sort_by(|a, b| {
            a.depth
                .partial_cmp(&b.depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for handle in &handles {
            let is_hovered = hovered_axis == Some(handle.axis);
            let current_radius = if is_hovered {
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
                    if is_hovered { 2.0_f32 } else { 1.0_f32 },
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
                handle.label,
                egui::FontId::proportional(font_size),
                crate::app::contrasting_text_color(handle.color),
            );
        }
    }

    /// Renders on-canvas interactive orientation gizmo and returns true if clicked.
    pub fn draw_gizmo_overlay(&mut self, response: &egui::Response, ui: &egui::Ui) -> bool {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let (center, axis_length, knob_radius, handles) =
            self.gizmo_geometry(response.rect, &palette);
        let pointer_pos = ui.input(|i| i.pointer.interact_pos());
        let is_clicked = ui.input(|i| i.pointer.primary_clicked());

        let hovered_axis = pointer_pos.and_then(|pos| {
            let hit_threshold = knob_radius + 4.0_f32;
            handles
                .iter()
                .find(|h| pos.distance(h.screen_pos) <= hit_threshold)
                .map(|h| h.axis)
        });

        let handled_click = self.check_gizmo_click(&handles, pointer_pos, is_clicked, knob_radius);
        self.render_gizmo_painter(
            ui,
            center,
            axis_length,
            knob_radius,
            handles.to_vec(),
            hovered_axis,
        );

        handled_click
    }
}
