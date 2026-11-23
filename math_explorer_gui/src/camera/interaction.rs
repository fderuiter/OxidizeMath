#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Pointer, touch, keyboard, and side-panel UI interactions for [`super::Camera3D`].

use eframe::egui;

use super::{
    Camera3D, ProjectionMode, CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_PAN, CAMERA_3D_ZOOM_MAX,
    CAMERA_3D_ZOOM_MIN,
};

const ORBIT_RAD_PER_SEC: f32 = 3.0_f32;
const PAN_PX_PER_SEC: f32 = 720.0_f32;
const ZOOM_BASE_PER_FRAME_60HZ: f32 = 1.05_f32;
const DEFAULT_FRAME_DT: f32 = 1.0_f32 / 60.0_f32;

const TRACKED_CAMERA_KEYS: [egui::Key; 12] = [
    egui::Key::W,
    egui::Key::A,
    egui::Key::S,
    egui::Key::D,
    egui::Key::Q,
    egui::Key::E,
    egui::Key::R,
    egui::Key::P,
    egui::Key::Num1,
    egui::Key::Num2,
    egui::Key::Num3,
    egui::Key::Num4,
];

fn camera_key_bit(key: egui::Key) -> Option<u16> {
    TRACKED_CAMERA_KEYS
        .iter()
        .position(|&k| k == key)
        .map(|idx| 1_u16 << idx)
}

const fn has_shortcut_modifier(mods: egui::Modifiers) -> bool {
    mods.ctrl || mods.command || mods.mac_cmd || mods.alt
}

fn effective_stable_dt(ui: &egui::Ui) -> f32 {
    let raw_dt = ui.input(|i| i.stable_dt);
    if raw_dt.is_finite() && raw_dt > 0.0_f32 {
        raw_dt.clamp(1.0e-4_f32, 0.1_f32)
    } else {
        DEFAULT_FRAME_DT
    }
}

fn update_suppressed_key_mask(ui: &egui::Ui, scope_id: egui::Id) -> (u16, bool, bool) {
    let mask_id = scope_id.with("camera_suppressed_mod_keys");
    let prev_mask = ui.ctx().data(|d| d.get_temp::<u16>(mask_id)).unwrap_or(0);
    let (next_mask, blocked_frame, shift_active) = ui.input(|i| {
        let mut mask = prev_mask;
        let state_blocked = has_shortcut_modifier(i.modifiers);
        let mut event_blocked = false;
        let mut event_shift = false;
        for ev in &i.events {
            if let egui::Event::Key {
                key,
                pressed,
                modifiers,
                ..
            } = ev
            {
                let key_blocked = state_blocked || has_shortcut_modifier(*modifiers);
                if *pressed && key_blocked {
                    event_blocked = true;
                }
                if *pressed && modifiers.shift && !key_blocked {
                    event_shift = true;
                }
                if let Some(bit) = camera_key_bit(*key) {
                    if *pressed && key_blocked {
                        mask |= bit;
                    } else {
                        mask &= !bit;
                    }
                }
            }
        }
        for (idx, &k) in TRACKED_CAMERA_KEYS.iter().enumerate() {
            if !i.key_down(k) {
                mask &= !(1_u16 << idx);
            }
        }
        (
            mask,
            state_blocked || event_blocked,
            i.modifiers.shift || event_shift,
        )
    });
    ui.ctx().data_mut(|d| d.insert_temp(mask_id, next_mask));
    (next_mask, blocked_frame, shift_active)
}

fn unmodified_key_down(i: &egui::InputState, key: egui::Key, suppressed_mask: u16) -> bool {
    let suppressed = camera_key_bit(key).is_some_and(|bit| (suppressed_mask & bit) != 0);
    !suppressed && i.key_down(key)
}

fn unmodified_key_pressed(i: &egui::InputState, key: egui::Key, suppressed_mask: u16) -> bool {
    let suppressed = camera_key_bit(key).is_some_and(|bit| (suppressed_mask & bit) != 0);
    !suppressed && i.key_pressed(key)
}

impl Camera3D {
    /// Handles canvas pointer, multi-touch, gizmo, and keyboard interactions for standalone rects.
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
        let is_dragging_callout = Self::is_callout_drag_active(response, ui, plot_id);
        let multi_touch = ui.input(|i| i.multi_touch());

        if !gizmo_handled && !is_dragging_callout {
            self.apply_canvas_pointer(response, ui, multi_touch, plot_id.is_none());
        }

        self.apply_pointer_zoom(response, ui, multi_touch.is_some());
        self.register_camera_commands(ui);
        self.apply_dispatched_camera_commands(ui.ctx());
        if response.has_focus() {
            self.handle_keyboard_interaction(ui, response.id);
        }
        if *self != prev {
            ui.ctx().request_repaint();
        }
    }

    fn apply_dispatched_camera_commands(&mut self, ctx: &egui::Context) {
        use egui_plot::commands::CommandRegistryData;
        if !CommandRegistryData::has_pending_dispatch(ctx, "Camera 3D") {
            return;
        }
        let step_px = 12.0_f32;
        if CommandRegistryData::take_dispatched(ctx, "Reset View", "Camera 3D") {
            self.reset();
        } else if CommandRegistryData::take_dispatched(ctx, "Toggle Projection", "Camera 3D") {
            self.projection = self.projection.toggle();
        } else if CommandRegistryData::take_dispatched(ctx, "Zoom In", "Camera 3D") {
            self.zoom = (self.zoom * 1.15_f32).clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX);
        } else if CommandRegistryData::take_dispatched(ctx, "Zoom Out", "Camera 3D") {
            self.zoom = (self.zoom / 1.15_f32).clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX);
        } else if CommandRegistryData::take_dispatched(ctx, "Pitch Up", "Camera 3D") {
            self.pitch =
                (self.pitch - 0.1_f32).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
        } else if CommandRegistryData::take_dispatched(ctx, "Pitch Down", "Camera 3D") {
            self.pitch =
                (self.pitch + 0.1_f32).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
        } else if CommandRegistryData::take_dispatched(ctx, "Yaw Left", "Camera 3D") {
            self.yaw =
                (self.yaw - 0.1_f32).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
        } else if CommandRegistryData::take_dispatched(ctx, "Yaw Right", "Camera 3D") {
            self.yaw =
                (self.yaw + 0.1_f32).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
        } else if CommandRegistryData::take_dispatched(ctx, "Pan Up", "Camera 3D") {
            self.pan_by_pixels(egui::vec2(0.0, -step_px));
        } else if CommandRegistryData::take_dispatched(ctx, "Pan Down", "Camera 3D") {
            self.pan_by_pixels(egui::vec2(0.0, step_px));
        } else if CommandRegistryData::take_dispatched(ctx, "Pan Left", "Camera 3D") {
            self.pan_by_pixels(egui::vec2(-step_px, 0.0));
        } else if CommandRegistryData::take_dispatched(ctx, "Pan Right", "Camera 3D") {
            self.pan_by_pixels(egui::vec2(step_px, 0.0));
        }
    }

    fn is_callout_drag_active(
        response: &egui::Response,
        ui: &egui::Ui,
        plot_id: Option<egui::Id>,
    ) -> bool {
        ui.ctx().data(|d| {
            let by_resp = d
                .get_temp::<bool>(response.id.with("dragging_callout"))
                .unwrap_or(false);
            let by_plot = plot_id.is_some_and(|pid| {
                d.get_temp::<bool>(pid.with("dragging_callout"))
                    .unwrap_or(false)
            });
            by_resp || by_plot
        })
    }

    fn pan_by_pixels(&mut self, delta: egui::Vec2) {
        let scale = if self.pan_units_per_px.is_finite() && self.pan_units_per_px > 0.0_f32 {
            self.pan_units_per_px
        } else {
            0.02_f32
        };
        self.pan_x = (self.pan_x + delta.x * scale).clamp(-CAMERA_3D_MAX_PAN, CAMERA_3D_MAX_PAN);
        self.pan_y = (self.pan_y - delta.y * scale).clamp(-CAMERA_3D_MAX_PAN, CAMERA_3D_MAX_PAN);
    }

    fn apply_canvas_pointer(
        &mut self,
        response: &egui::Response,
        ui: &egui::Ui,
        multi_touch: Option<egui::MultiTouchInfo>,
        allow_secondary_pan: bool,
    ) {
        let press_inside_gizmo = ui.input(|i| {
            i.pointer
                .press_origin()
                .is_some_and(|pos| self.is_inside_gizmo(pos, response.rect))
        });
        let primary_drag = response.dragged_by(egui::PointerButton::Primary);
        let aux_pan_drag = response.dragged_by(egui::PointerButton::Middle)
            || (allow_secondary_pan && response.dragged_by(egui::PointerButton::Secondary));
        if !press_inside_gizmo && (response.clicked() || primary_drag || aux_pan_drag) {
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
                self.pan_by_pixels(touch.translation_delta);
            }
        } else if !press_inside_gizmo && aux_pan_drag {
            self.pan_by_pixels(response.drag_delta());
        } else if !press_inside_gizmo && primary_drag {
            let delta = response.drag_delta();
            if ui.input(|i| i.modifiers.shift) {
                self.pan_by_pixels(delta);
            } else {
                self.yaw = (self.yaw - delta.x * 0.01_f32)
                    .clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
                self.pitch = (self.pitch - delta.y * 0.01_f32)
                    .clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
            }
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

    fn handle_keyboard_interaction(&mut self, ui: &egui::Ui, scope_id: egui::Id) {
        let (suppressed_mask, blocked_frame, shift_active) =
            update_suppressed_key_mask(ui, scope_id);
        if blocked_frame {
            return;
        }
        let dt = effective_stable_dt(ui);
        if !shift_active && self.apply_discrete_view_keys(ui, suppressed_mask) {
            return;
        }
        if shift_active {
            self.apply_shift_wasd_pan(ui, dt, suppressed_mask);
        } else {
            self.apply_wasd_orbit(ui, dt, suppressed_mask);
        }
        let step_zoom = ZOOM_BASE_PER_FRAME_60HZ.powf(60.0_f32 * dt);
        let mut zoom_factor = 1.0_f32;
        if ui.input(|i| unmodified_key_down(i, egui::Key::Q, suppressed_mask)) {
            zoom_factor /= step_zoom;
        }
        if ui.input(|i| unmodified_key_down(i, egui::Key::E, suppressed_mask)) {
            zoom_factor *= step_zoom;
        }
        self.zoom = (self.zoom * zoom_factor).clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX);
    }

    fn apply_discrete_view_keys(&mut self, ui: &egui::Ui, suppressed_mask: u16) -> bool {
        if ui.input(|i| unmodified_key_pressed(i, egui::Key::R, suppressed_mask)) {
            self.reset();
            return true;
        }
        if ui.input(|i| unmodified_key_pressed(i, egui::Key::P, suppressed_mask)) {
            self.projection = self.projection.toggle();
        }
        if ui.input(|i| unmodified_key_pressed(i, egui::Key::Num1, suppressed_mask)) {
            self.snap_to_yz();
        } else if ui.input(|i| unmodified_key_pressed(i, egui::Key::Num2, suppressed_mask)) {
            self.snap_to_xz();
        } else if ui.input(|i| unmodified_key_pressed(i, egui::Key::Num3, suppressed_mask)) {
            self.snap_to_xy();
        } else if ui.input(|i| unmodified_key_pressed(i, egui::Key::Num4, suppressed_mask)) {
            self.snap_to_iso();
        }
        false
    }

    fn apply_shift_wasd_pan(&mut self, ui: &egui::Ui, dt: f32, suppressed_mask: u16) {
        let step_px = PAN_PX_PER_SEC * dt;
        let delta = ui.input(|i| {
            let mut d = egui::Vec2::ZERO;
            if unmodified_key_down(i, egui::Key::A, suppressed_mask) {
                d.x -= step_px;
            }
            if unmodified_key_down(i, egui::Key::D, suppressed_mask) {
                d.x += step_px;
            }
            if unmodified_key_down(i, egui::Key::W, suppressed_mask) {
                d.y -= step_px;
            }
            if unmodified_key_down(i, egui::Key::S, suppressed_mask) {
                d.y += step_px;
            }
            d
        });
        if delta != egui::Vec2::ZERO {
            self.pan_by_pixels(delta);
        }
    }

    fn apply_wasd_orbit(&mut self, ui: &egui::Ui, dt: f32, suppressed_mask: u16) {
        let step_rad = ORBIT_RAD_PER_SEC * dt;
        let (yaw_delta, pitch_delta) = ui.input(|i| {
            let mut dyaw = 0.0_f32;
            let mut dpitch = 0.0_f32;
            if unmodified_key_down(i, egui::Key::A, suppressed_mask) {
                dyaw += step_rad;
            }
            if unmodified_key_down(i, egui::Key::D, suppressed_mask) {
                dyaw -= step_rad;
            }
            if unmodified_key_down(i, egui::Key::W, suppressed_mask) {
                dpitch += step_rad;
            }
            if unmodified_key_down(i, egui::Key::S, suppressed_mask) {
                dpitch -= step_rad;
            }
            (dyaw, dpitch)
        });
        self.yaw = (self.yaw - yaw_delta).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
        self.pitch =
            (self.pitch - pitch_delta).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
    }

    fn register_camera_commands(&self, ui: &egui::Ui) {
        use egui_plot::commands::{CommandRegistryData, CommandTrigger};
        let key_commands = [
            (egui::Key::W, "Pitch Up", "Rotate camera pitch upwards"),
            (egui::Key::S, "Pitch Down", "Rotate camera pitch downwards"),
            (egui::Key::A, "Yaw Left", "Rotate camera yaw left"),
            (egui::Key::D, "Yaw Right", "Rotate camera yaw right"),
            (egui::Key::Q, "Zoom Out", "Zoom camera out"),
            (egui::Key::E, "Zoom In", "Zoom camera in"),
            (
                egui::Key::Num1,
                "Snap to X-Axis View",
                "Orient camera along X-axis (YZ plane)",
            ),
            (
                egui::Key::Num2,
                "Snap to Y-Axis View",
                "Orient camera along Y-axis (XZ plane)",
            ),
            (
                egui::Key::Num3,
                "Snap to Z-Axis View",
                "Orient camera along Z-axis (XY plane)",
            ),
            (
                egui::Key::Num4,
                "Snap to Isometric View",
                "Orient camera to isometric 3D pose",
            ),
            (
                egui::Key::P,
                "Toggle Projection",
                "Switch between parallel and perspective 3D projection",
            ),
            (
                egui::Key::R,
                "Reset View",
                "Reset camera orientation, pan, projection, and zoom",
            ),
        ];
        let ctx = ui.ctx();
        for (key, name, desc) in key_commands {
            CommandRegistryData::register(
                ctx,
                name,
                desc,
                CommandTrigger::Key(key),
                true,
                "Camera 3D",
            );
        }
        Self::register_pan_commands(ctx);
    }

    fn register_pan_commands(ctx: &egui::Context) {
        use egui_plot::commands::{CommandRegistryData, CommandTrigger};
        let pan_commands = [
            (egui::Key::W, "Pan Up", "Pan 3D viewport upwards"),
            (egui::Key::S, "Pan Down", "Pan 3D viewport downwards"),
            (egui::Key::A, "Pan Left", "Pan 3D viewport left"),
            (egui::Key::D, "Pan Right", "Pan 3D viewport right"),
        ];
        for (key, name, desc) in pan_commands {
            CommandRegistryData::register(
                ctx,
                name,
                desc,
                CommandTrigger::Shortcut(egui::Modifiers::SHIFT, key),
                true,
                "Camera 3D",
            );
        }
    }

    /// Renders side-panel controls for yaw, pitch, zoom, projection mode, and view reset.
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

        ui.horizontal(|ui| {
            ui.label("Projection:");
            ui.selectable_value(
                &mut self.projection,
                ProjectionMode::Parallel,
                ProjectionMode::Parallel.label(),
            );
            ui.selectable_value(
                &mut self.projection,
                ProjectionMode::Perspective,
                ProjectionMode::Perspective.label(),
            );
        });

        if ui.button("↻ Reset View").clicked() {
            self.reset();
        }
        if *self != prev {
            ui.ctx().request_repaint();
        }
    }
}
