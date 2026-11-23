use eframe::egui;

#[path = "camera_gizmo.rs"]
mod gizmo;

const fn default_angle() -> f32 {
    0.5_f32
}

const fn default_zoom() -> f32 {
    1.0_f32
}

/// Minimum shared zoom scale across all 3D camera controls and gestures.
pub const CAMERA_3D_ZOOM_MIN: f32 = 0.01_f32;

/// Maximum shared zoom scale across all 3D camera controls and gestures.
pub const CAMERA_3D_ZOOM_MAX: f32 = 100.0_f32;

/// Maximum finite camera angle magnitude (radians) safe from `f32::to_degrees()` overflow.
pub const CAMERA_3D_MAX_ANGLE_RAD: f32 = 1_000.0_f32 * std::f32::consts::TAU;

/// Canonical canvas helper text displayed below interactive 3D viewports.
pub const CAMERA_3D_HELPER_TEXT: &str =
    "Drag canvas or use WASD to orbit, scroll or Q/E to zoom, double-click or press R to reset view.";

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Camera3D {
    #[allow(missing_docs)]
    pub pitch: f32,
    #[allow(missing_docs)]
    pub yaw: f32,
    #[allow(missing_docs)]
    pub zoom: f32,
    #[serde(skip, default = "default_angle")]
    reset_pitch: f32,
    #[serde(skip, default = "default_angle")]
    reset_yaw: f32,
    #[serde(skip, default = "default_zoom")]
    reset_zoom: f32,
}

impl PartialEq for Camera3D {
    fn eq(&self, other: &Self) -> bool {
        self.pitch == other.pitch && self.yaw == other.yaw && self.zoom == other.zoom
    }
}

impl Default for Camera3D {
    fn default() -> Self {
        Self::new(default_angle(), default_angle(), default_zoom())
    }
}

impl Camera3D {
    #[allow(missing_docs)]
    pub fn new(pitch: f32, yaw: f32, zoom: f32) -> Self {
        Self {
            pitch,
            yaw,
            zoom,
            reset_pitch: pitch,
            reset_yaw: yaw,
            reset_zoom: zoom,
        }
    }

    /// Returns true if pitch, yaw, and zoom are finite and within safe numeric bounds.
    pub fn is_valid(&self) -> bool {
        self.pitch.is_finite()
            && self.yaw.is_finite()
            && self.zoom.is_finite()
            && self.pitch.abs() <= CAMERA_3D_MAX_ANGLE_RAD
            && self.yaw.abs() <= CAMERA_3D_MAX_ANGLE_RAD
            && (CAMERA_3D_ZOOM_MIN..=CAMERA_3D_ZOOM_MAX).contains(&self.zoom)
    }

    /// Applies orientation and zoom from `pose` while preserving this tool's default reset pose.
    pub fn apply_pose(&mut self, pose: Self) {
        self.pitch = pose.pitch;
        self.yaw = pose.yaw;
        self.zoom = pose.zoom;
    }

    /// Restores camera pitch, yaw, and zoom to this instance's configured default pose.
    pub fn reset(&mut self) {
        self.pitch = self.reset_pitch;
        self.yaw = self.reset_yaw;
        self.zoom = self.reset_zoom;
    }

    /// Resolves either a nested `camera` object or legacy `pitch`/`yaw`/`zoom` preset fields.
    pub fn from_preset_fields(
        camera: Option<Self>,
        pitch: Option<f32>,
        yaw: Option<f32>,
        zoom: Option<f32>,
        fallback: Option<Self>,
    ) -> Result<Self, String> {
        let has_legacy = pitch.is_some() || yaw.is_some() || zoom.is_some();
        if let Some(cam) = camera {
            if has_legacy {
                return Err("Conflicting camera and legacy pitch/yaw/zoom fields".to_string());
            }
            if !cam.is_valid() {
                return Err("Camera parameters out of valid bounds".to_string());
            }
            return Ok(cam);
        }
        match (pitch, yaw, zoom) {
            (Some(p), Some(y), Some(z)) => {
                let cam = Self::new(p, y, z);
                if !cam.is_valid() {
                    return Err("Camera parameters out of valid bounds".to_string());
                }
                Ok(cam)
            }
            (None, None, None) => {
                fallback.ok_or_else(|| "Missing camera configuration in preset state".to_string())
            }
            _ => Err("Incomplete legacy camera fields: pitch, yaw, and zoom required".to_string()),
        }
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
        self.handle_interaction_with_plot(response, ui, None);
    }

    pub(crate) fn handle_interaction_with_plot(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        plot_id: Option<egui::Id>,
    ) {
        let prev = *self;
        let gizmo_handled = self.draw_gizmo_overlay(response, ui);
        let is_dragging_callout = ui.ctx().data(|d| {
            let by_resp = d
                .get_temp::<bool>(response.id.with("dragging_callout"))
                .unwrap_or(false);
            let by_plot = plot_id.is_some_and(|pid| {
                d.get_temp::<bool>(pid.with("dragging_callout"))
                    .unwrap_or(false)
            });
            by_resp || by_plot
        });
        let multi_touch = ui.input(|i| i.multi_touch());

        if !gizmo_handled && !is_dragging_callout {
            self.apply_canvas_pointer(response, ui, multi_touch);
        }

        self.apply_pointer_zoom(response, ui, multi_touch.is_some());
        self.register_camera_commands(ui);
        if response.has_focus() {
            self.handle_keyboard_interaction(ui);
        }
        if *self != prev {
            ui.ctx().request_repaint();
        }
    }

    fn apply_canvas_pointer(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        multi_touch: Option<egui::MultiTouchInfo>,
    ) {
        let press_inside_gizmo = ui.input(|i| {
            i.pointer
                .press_origin()
                .is_some_and(|pos| self.is_inside_gizmo(pos, response.rect))
        });
        if !press_inside_gizmo
            && (response.clicked() || response.dragged_by(egui::PointerButton::Primary))
        {
            response.request_focus();
        }
        if !press_inside_gizmo && response.double_clicked() {
            self.reset();
        } else if let Some(touch) = multi_touch {
            if touch.zoom_delta != 1.0_f32 {
                let zoom_factor = 1.0_f32 + (touch.zoom_delta - 1.0_f32) * 0.5_f32;
                self.zoom = (self.zoom * zoom_factor).clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX);
            }
            if touch.translation_delta != egui::Vec2::ZERO {
                self.yaw = (self.yaw - touch.translation_delta.x * 0.01_f32)
                    .clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
                self.pitch = (self.pitch - touch.translation_delta.y * 0.01_f32)
                    .clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
            }
        } else if !press_inside_gizmo && response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.yaw = (self.yaw - delta.x * 0.01_f32)
                .clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
            self.pitch = (self.pitch - delta.y * 0.01_f32)
                .clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
        }
    }

    fn apply_pointer_zoom(&mut self, response: &egui::Response, ui: &egui::Ui, has_touch: bool) {
        if !response.hovered() || has_touch {
            return;
        }
        let pinch_zoom = ui.input(|i| i.zoom_delta());
        if (pinch_zoom - 1.0_f32).abs() > f32::EPSILON {
            self.zoom = (self.zoom * pinch_zoom).clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX);
            return;
        }
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0_f32 {
            let factor = (1.0_f32 + scroll * 0.001_f32).max(0.1_f32);
            self.zoom = (self.zoom * factor).clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX);
        }
    }

    /// Rotates a 3D world point `[x, y, z]` into viewer-aligned `[right, up, viewer_z]` space.
    pub fn rotate_to_camera_space(&self, point: &[f64; 3]) -> [f64; 3] {
        let (sy, cy) = (self.yaw as f64).sin_cos();
        let (sp, cp) = (self.pitch as f64).sin_cos();

        let x1 = point[0] * cy - point[1] * sy;
        let y1 = point[0] * sy + point[1] * cy;
        let z1 = point[2];

        let y2 = y1 * cp - z1 * sp;
        let z2 = y1 * sp + z1 * cp;

        [x1, y2, z2]
    }

    /// Builds a 4x4 pinhole world-to-camera matrix (`[right, up, depth]` with `depth = camera_distance - viewer_z`).
    pub fn view_matrix(&self, camera_distance: f64) -> nalgebra::Matrix4<f64> {
        let c0 = self.rotate_to_camera_space(&[1.0, 0.0, 0.0]);
        let c1 = self.rotate_to_camera_space(&[0.0, 1.0, 0.0]);
        let c2 = self.rotate_to_camera_space(&[0.0, 0.0, 1.0]);
        nalgebra::Matrix4::new(
            c0[0],
            c1[0],
            c2[0],
            0.0,
            c0[1],
            c1[1],
            c2[1],
            0.0,
            -c0[2],
            -c1[2],
            -c2[2],
            camera_distance,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    pub(super) fn project_unit(&self, point: &[f64; 3]) -> (f32, f32, f32) {
        let [x2, y2, z2] = self.rotate_to_camera_space(point);
        (x2 as f32, y2 as f32, z2 as f32)
    }

    fn handle_keyboard_interaction(&mut self, ui: &egui::Ui) {
        let reset_pressed = ui.input(|i| {
            !i.modifiers.command
                && !i.modifiers.ctrl
                && !i.modifiers.alt
                && i.key_pressed(egui::Key::R)
        });
        if reset_pressed {
            self.reset();
            return;
        }

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
            zoom_factor /= 1.05_f32;
        }
        if ui.input(|i| i.key_down(egui::Key::E)) {
            zoom_factor *= 1.05_f32;
        }

        self.yaw = (self.yaw - yaw_delta).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
        self.pitch =
            (self.pitch - pitch_delta).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
        self.zoom = (self.zoom * zoom_factor).clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX);
    }

    fn register_camera_commands(&self, ui: &egui::Ui) {
        let commands = [
            (egui::Key::W, "Pitch Up", "Rotate camera pitch upwards"),
            (egui::Key::S, "Pitch Down", "Rotate camera pitch downwards"),
            (egui::Key::A, "Yaw Left", "Rotate camera yaw left"),
            (egui::Key::D, "Yaw Right", "Rotate camera yaw right"),
            (egui::Key::Q, "Zoom Out", "Zoom camera out"),
            (egui::Key::E, "Zoom In", "Zoom camera in"),
            (
                egui::Key::R,
                "Reset View",
                "Reset camera orientation and zoom",
            ),
        ];

        let ctx = ui.ctx();
        for (key, name, desc) in commands {
            egui_plot::commands::CommandRegistryData::register(
                ctx,
                name,
                desc,
                egui_plot::commands::CommandTrigger::Key(key),
                true,
                "Camera 3D",
            );
        }
    }

    #[allow(missing_docs)]
    pub fn project(&self, point: &[f64; 3]) -> [f64; 2] {
        let [x2, y2, _] = self.rotate_to_camera_space(point);
        let z = self.zoom as f64;
        [x2 * z, y2 * z]
    }

    #[allow(missing_docs)]
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let prev = *self;
        ui.label("Yaw");
        ui.drag_angle(&mut self.yaw);
        self.yaw = if self.yaw.is_finite() {
            self.yaw
                .clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD)
        } else {
            prev.yaw
        };

        ui.label("Pitch");
        ui.drag_angle(&mut self.pitch);
        self.pitch = if self.pitch.is_finite() {
            self.pitch
                .clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD)
        } else {
            prev.pitch
        };

        ui.add(
            egui::Slider::new(&mut self.zoom, CAMERA_3D_ZOOM_MIN..=CAMERA_3D_ZOOM_MAX)
                .logarithmic(true)
                .text("Zoom"),
        );
        self.zoom = if self.zoom.is_finite() {
            self.zoom.clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX)
        } else {
            prev.zoom
        };

        if ui.button("↻ Reset View").clicked() {
            self.reset();
        }
        if *self != prev {
            ui.ctx().request_repaint();
        }
    }
}

/// Shared 3D viewport wrapper around `egui_plot::Plot` that disables 2D pan/zoom gestures,
/// locks symmetric viewport bounds so `Camera3D::zoom` never gets cancelled by auto-bounds,
/// and delegates canvas orbit, zoom, gizmo, and reset gestures to `Camera3D`.
pub struct Plot3DViewport<'a> {
    id_source: egui::Id,
    title: Option<String>,
    half_extent: f64,
    helper_text: Option<&'a str>,
}

impl<'a> Plot3DViewport<'a> {
    /// Creates a 3D viewport with locked symmetric half-extent in projected plot units.
    pub fn new(id_source: impl egui_plot::PlotIdSource, half_extent: f64) -> Self {
        let (id_source, title) = id_source.into_plot_id_and_title();
        Self {
            id_source,
            title,
            half_extent: half_extent.max(0.1),
            helper_text: Some(CAMERA_3D_HELPER_TEXT),
        }
    }

    /// Overrides the helper caption displayed beneath the 3D canvas.
    pub fn helper_text(mut self, text: &'a str) -> Self {
        self.helper_text = Some(text);
        self
    }

    /// Renders the locked 3D plot, processes `Camera3D` gestures, and draws optional helper copy.
    pub fn show<'p, R>(
        self,
        ui: &mut egui::Ui,
        camera: &mut Camera3D,
        build_fn: impl FnOnce(&mut egui_plot::PlotUi<'p>) -> R,
    ) -> egui_plot::PlotResponse<R> {
        let avail = ui.available_size_before_wrap();
        let label_reserve = if self.helper_text.is_some() {
            24.0_f32
        } else {
            0.0_f32
        };
        let plot_height = (avail.y - label_reserve).max(64.0_f32);
        let aspect = if plot_height > 0.0_f32 {
            (avail.x / plot_height).max(1.0_f32) as f64
        } else {
            1.0
        };
        let x_half = self.half_extent * aspect;
        let y_half = self.half_extent;

        let mut captured_plot_id = None;
        let mut plot = egui_plot::Plot::new(self.id_source)
            .height(plot_height)
            .data_aspect(1.0)
            .allow_2d_analysis(false)
            .show_x(false)
            .show_y(false)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_axis_zoom_drag(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .show_navigation_controls(false)
            .default_x_bounds(-x_half, x_half)
            .default_y_bounds(-y_half, y_half);
        if let Some(title) = self.title {
            plot = plot.title(title);
        }
        let plot_response = plot.show(ui, |plot_ui| {
            plot_ui.set_plot_bounds_x(-x_half..=x_half);
            plot_ui.set_plot_bounds_y(-y_half..=y_half);
            captured_plot_id = Some(plot_ui.plot_id());
            build_fn(plot_ui)
        });

        camera.handle_interaction_with_plot(&plot_response.response, ui, captured_plot_id);

        if let Some(text) = self.helper_text {
            ui.label(text);
        }

        plot_response
    }
}

#[cfg(test)]
#[path = "camera_tests.rs"]
mod tests;
