use eframe::egui;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GizmoAxis {
    X,
    Y,
    Z,
    Iso,
}

#[derive(Clone, Copy, Debug)]
struct GizmoHandle {
    axis: GizmoAxis,
    label: &'static str,
    color: egui::Color32,
    screen_pos: egui::Pos2,
    depth: f32,
}

#[allow(missing_docs)]
pub struct CoordinateMapper {
    #[allow(missing_docs)]
    pub screen_rect: egui::Rect,
    #[allow(missing_docs)]
    pub sim_rect: egui::Rect,
}

#[derive(Clone, Copy, Debug)]
#[allow(missing_docs)]
pub struct Camera3D {
    #[allow(missing_docs)]
    pub pitch: f32,
    #[allow(missing_docs)]
    pub yaw: f32,
    #[allow(missing_docs)]
    pub zoom: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            pitch: 0.5_f32,
            yaw: 0.5_f32,
            zoom: 1.0_f32,
        }
    }
}

impl Camera3D {
    #[allow(missing_docs)]
    pub fn new(pitch: f32, yaw: f32, zoom: f32) -> Self {
        Self { pitch, yaw, zoom }
    }

    /// Snaps camera orientation to Top view (XY plane).
    pub fn snap_to_xy(&mut self) {
        self.pitch = 0.0_f32;
        self.yaw = 0.0_f32;
    }

    /// Snaps camera orientation to Front view (XZ plane).
    pub fn snap_to_xz(&mut self) {
        self.pitch = -std::f32::consts::FRAC_PI_2;
        self.yaw = 0.0_f32;
    }

    /// Snaps camera orientation to Side view (YZ plane).
    pub fn snap_to_yz(&mut self) {
        self.pitch = -std::f32::consts::FRAC_PI_2;
        self.yaw = -std::f32::consts::FRAC_PI_2;
    }

    /// Snaps camera orientation to Isometric view.
    pub fn snap_to_iso(&mut self) {
        self.pitch = 0.5_f32;
        self.yaw = 0.5_f32;
    }

    #[allow(missing_docs)]
    pub fn handle_interaction(&mut self, response: &egui::Response, ui: &egui::Ui) {
        let gizmo_handled = self.draw_gizmo_overlay(response, ui);

        let multi_touch = ui.input(|i| i.multi_touch());

        if !gizmo_handled {
            if let Some(touch) = multi_touch {
                if touch.zoom_delta != 1.0_f32 {
                    let zoom_factor = 1.0_f32 + (touch.zoom_delta - 1.0_f32) * 0.5_f32;
                    self.zoom *= zoom_factor;
                    self.zoom = self.zoom.clamp(0.01_f32, 100.0_f32);
                }
                if touch.translation_delta != egui::Vec2::ZERO {
                    self.yaw -= touch.translation_delta.x * 0.01_f32;
                    self.pitch -= touch.translation_delta.y * 0.01_f32;
                }
            } else if response.dragged() {
                let press_inside_gizmo = ui.input(|i| {
                    i.pointer
                        .press_origin()
                        .is_some_and(|pos| self.is_inside_gizmo(pos, response.rect))
                });
                if !press_inside_gizmo {
                    let delta = response.drag_delta();
                    self.yaw -= delta.x * 0.01_f32;
                    self.pitch -= delta.y * 0.01_f32;
                }
            }
        }

        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0_f32 && response.hovered() && multi_touch.is_none() {
            self.zoom *= 1.0_f32 + (scroll * 0.001_f32);
            self.zoom = self.zoom.clamp(0.01_f32, 100.0_f32);
        }

        if response.has_focus() {
            self.handle_keyboard_interaction(ui);
        }
    }

    fn project_unit(&self, point: &[f64; 3]) -> (f32, f32, f32) {
        let cy = (self.yaw as f64).cos();
        let sy = (self.yaw as f64).sin();
        let cp = (self.pitch as f64).cos();
        let sp = (self.pitch as f64).sin();

        let x1 = point[0] * cy - point[1] * sy;
        let y1 = point[0] * sy + point[1] * cy;
        let z1 = point[2];

        let x2 = x1;
        let y2 = y1 * cp - z1 * sp;
        let z2 = y1 * sp + z1 * cp;

        (x2 as f32, y2 as f32, z2 as f32)
    }

    fn gizmo_geometry(&self, rect: egui::Rect) -> (egui::Pos2, f32, f32, [GizmoHandle; 4]) {
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
                color: egui::Color32::from_rgb(180, 180, 180),
                screen_pos: center,
                depth: -10.0_f32,
            },
            GizmoHandle {
                axis: GizmoAxis::X,
                label: "X",
                color: egui::Color32::from_rgb(235, 65, 65),
                screen_pos: pos_x,
                depth: pz_x,
            },
            GizmoHandle {
                axis: GizmoAxis::Y,
                label: "Y",
                color: egui::Color32::from_rgb(65, 200, 65),
                screen_pos: pos_y,
                depth: pz_y,
            },
            GizmoHandle {
                axis: GizmoAxis::Z,
                label: "Z",
                color: egui::Color32::from_rgb(65, 130, 245),
                screen_pos: pos_z,
                depth: pz_z,
            },
        ];

        (center, axis_length, knob_radius, handles)
    }

    fn is_inside_gizmo(&self, pos: egui::Pos2, rect: egui::Rect) -> bool {
        let (center, axis_length, knob_radius, _) = self.gizmo_geometry(rect);
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

        if let Some(axis) = hovered_axis {
            if is_clicked {
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
        let scale = axis_length / 30.0_f32;
        let bg_radius = axis_length + knob_radius + 4.0_f32;

        painter.circle_filled(center, bg_radius, egui::Color32::from_black_alpha(35));
        painter.circle_stroke(
            center,
            bg_radius,
            egui::Stroke::new(1.0_f32, egui::Color32::from_white_alpha(25)),
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
                    egui::Color32::WHITE,
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
                egui::Color32::WHITE,
            );
        }
    }

    /// Renders on-canvas interactive orientation gizmo and returns true if clicked.
    pub fn draw_gizmo_overlay(&mut self, response: &egui::Response, ui: &egui::Ui) -> bool {
        let (center, axis_length, knob_radius, handles) = self.gizmo_geometry(response.rect);
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

    fn handle_keyboard_interaction(&mut self, ui: &egui::Ui) {
        let mut yaw_delta = 0.0_f32;
        let mut pitch_delta = 0.0_f32;
        let mut zoom_factor = 1.0_f32;
        if ui.input(|i| i.key_down(egui::Key::A)) {
            yaw_delta += 0.05_f32;
        }
        if ui.input(|i| i.key_down(egui::Key::D)) {
            yaw_delta -= 0.05_f32;
        }
        if ui.input(|i| i.key_down(egui::Key::W)) {
            pitch_delta += 0.05_f32;
        }
        if ui.input(|i| i.key_down(egui::Key::S)) {
            pitch_delta -= 0.05_f32;
        }
        if ui.input(|i| i.key_down(egui::Key::Q)) {
            zoom_factor *= 1.05_f32;
        }
        if ui.input(|i| i.key_down(egui::Key::E)) {
            zoom_factor /= 1.05_f32;
        }

        self.yaw -= yaw_delta;
        self.pitch -= pitch_delta;
        self.zoom *= zoom_factor;
        self.zoom = self.zoom.clamp(0.01_f32, 100.0_f32);

        self.register_camera_commands(ui);
    }

    fn register_camera_commands(&self, ui: &egui::Ui) {
        ui.ctx().data_mut(|d| {
            let mut registry = d
                .get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
                .unwrap_or_default();

            let commands = [
                (egui::Key::W, "Pitch Up", "Rotate camera pitch upwards"),
                (egui::Key::S, "Pitch Down", "Rotate camera pitch downwards"),
                (egui::Key::A, "Yaw Left", "Rotate camera yaw left"),
                (egui::Key::D, "Yaw Right", "Rotate camera yaw right"),
                (egui::Key::Q, "Zoom Out", "Zoom camera out"),
                (egui::Key::E, "Zoom In", "Zoom camera in"),
            ];

            for (key, name, desc) in commands {
                if !registry
                    .commands
                    .iter()
                    .any(|c| c.name == name && c.context == "Camera 3D")
                {
                    registry
                        .commands
                        .push(egui_plot::commands::CommandMetadata {
                            name: name.to_string(),
                            description: desc.to_string(),
                            trigger: egui_plot::commands::CommandTrigger::Key(key),
                            desktop_only: true,
                            context: "Camera 3D".to_string(),
                        });
                }
            }
            d.insert_temp(egui::Id::new("CMD_REGISTRY"), registry);
        });
    }

    #[allow(missing_docs)]
    pub fn project(&self, point: &[f64; 3]) -> [f64; 2] {
        let cy = (self.yaw as f64).cos();
        let sy = (self.yaw as f64).sin();
        let cp = (self.pitch as f64).cos();
        let sp = (self.pitch as f64).sin();

        // Apply yaw (rotate around Z)
        let x1 = point[0] * cy - point[1] * sy;
        let y1 = point[0] * sy + point[1] * cy;
        let z1 = point[2];

        // Apply pitch (rotate around X)
        let x2 = x1;
        let y2 = y1 * cp - z1 * sp;

        [x2 * (self.zoom as f64), y2 * (self.zoom as f64)]
    }

    #[allow(missing_docs)]
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Yaw");
        ui.drag_angle(&mut self.yaw);

        ui.label("Pitch");
        ui.drag_angle(&mut self.pitch);

        ui.add(egui::Slider::new(&mut self.zoom, 0.1..=5.0).text("Zoom"));
    }
}

impl CoordinateMapper {
    #[allow(missing_docs)]
    pub fn new(screen_rect: egui::Rect, sim_rect: egui::Rect) -> Self {
        Self {
            screen_rect,
            sim_rect,
        }
    }

    #[allow(missing_docs)]
    pub fn screen_to_sim(&self, pos: egui::Pos2) -> egui::Pos2 {
        let x_norm = (pos.x - self.screen_rect.min.x) / self.screen_rect.width();
        let y_norm = (pos.y - self.screen_rect.min.y) / self.screen_rect.height();

        egui::Pos2::new(
            self.sim_rect.min.x + x_norm * self.sim_rect.width(),
            self.sim_rect.min.y + y_norm * self.sim_rect.height(),
        )
    }

    #[allow(missing_docs)]
    pub fn sim_to_screen(&self, pos: egui::Pos2) -> egui::Pos2 {
        let x_norm = (pos.x - self.sim_rect.min.x) / self.sim_rect.width();
        let y_norm = (pos.y - self.sim_rect.min.y) / self.sim_rect.height();

        egui::Pos2::new(
            self.screen_rect.min.x + x_norm * self.screen_rect.width(),
            self.screen_rect.min.y + y_norm * self.screen_rect.height(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_snapping() {
        let mut camera = Camera3D::default();

        camera.snap_to_xy();
        assert_eq!(camera.pitch, 0.0_f32);
        assert_eq!(camera.yaw, 0.0_f32);

        camera.snap_to_xz();
        assert_eq!(camera.pitch, -std::f32::consts::FRAC_PI_2);
        assert_eq!(camera.yaw, 0.0_f32);

        camera.snap_to_yz();
        assert_eq!(camera.pitch, -std::f32::consts::FRAC_PI_2);
        assert_eq!(camera.yaw, -std::f32::consts::FRAC_PI_2);

        camera.snap_to_iso();
        assert_eq!(camera.pitch, 0.5_f32);
        assert_eq!(camera.yaw, 0.5_f32);
    }
}
