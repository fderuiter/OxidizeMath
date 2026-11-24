#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Command-palette registration and dispatch for [`super::Camera3D`].

use eframe::egui;

use super::{Camera3D, CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_ZOOM_MAX, CAMERA_3D_ZOOM_MIN};

impl Camera3D {
    pub(super) fn apply_dispatched_camera_commands(&mut self, ctx: &egui::Context) {
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

    pub(super) fn register_camera_commands(&self, ui: &egui::Ui) {
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
}
