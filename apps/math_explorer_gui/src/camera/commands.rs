#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Command-palette registration and dispatch for [`super::Camera3D`].

use eframe::egui;

use super::{Camera3D, CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_ZOOM_MAX, CAMERA_3D_ZOOM_MIN};

struct SnapDispatchSpec {
    cmd: &'static str,
    apply: fn(&mut Camera3D),
    label: &'static str,
}

impl Camera3D {
    pub(super) fn apply_dispatched_camera_commands(&mut self, ctx: &egui::Context) {
        use egui_plot::commands::CommandRegistryData;
        if !CommandRegistryData::has_pending_dispatch(ctx, "Camera 3D") {
            return;
        }
        if self.apply_dispatched_view_or_snap(ctx) {
            return;
        }
        for name in [
            "Zoom In",
            "Zoom Out",
            "Pitch Up",
            "Pitch Down",
            "Yaw Left",
            "Yaw Right",
            "Pan Up",
            "Pan Down",
            "Pan Left",
            "Pan Right",
        ] {
            if CommandRegistryData::take_dispatched(ctx, name, "Camera 3D") {
                self.apply_named_camera_step(name);
                self.announce_command_step(ctx, name);
                return;
            }
        }
    }

    fn apply_dispatched_view_or_snap(&mut self, ctx: &egui::Context) -> bool {
        use egui_plot::commands::CommandRegistryData;
        if CommandRegistryData::take_dispatched(ctx, "Reset View", "Camera 3D") {
            self.reset();
            self.announce_reset(ctx);
            return true;
        }
        if CommandRegistryData::take_dispatched(ctx, "Toggle Projection", "Camera 3D") {
            self.projection = self.projection.toggle();
            self.announce_projection(ctx);
            return true;
        }
        let snaps = [
            SnapDispatchSpec {
                cmd: "Snap to X-Axis View",
                apply: Self::snap_to_yz,
                label: "X-axis (YZ plane)",
            },
            SnapDispatchSpec {
                cmd: "Snap to Y-Axis View",
                apply: Self::snap_to_xz,
                label: "Y-axis (XZ plane)",
            },
            SnapDispatchSpec {
                cmd: "Snap to Z-Axis View",
                apply: Self::snap_to_xy,
                label: "Z-axis (XY plane)",
            },
            SnapDispatchSpec {
                cmd: "Snap to Isometric View",
                apply: Self::snap_to_iso,
                label: "Isometric",
            },
        ];
        for spec in snaps {
            if CommandRegistryData::take_dispatched(ctx, spec.cmd, "Camera 3D") {
                (spec.apply)(self);
                self.announce_view_snap(ctx, spec.label);
                return true;
            }
        }
        false
    }

    fn apply_named_camera_step(&mut self, name: &str) {
        let step_px = 12.0_f32;
        match name {
            "Zoom In" => {
                self.zoom = (self.zoom * 1.15_f32).clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX);
            }
            "Zoom Out" => {
                self.zoom = (self.zoom / 1.15_f32).clamp(CAMERA_3D_ZOOM_MIN, CAMERA_3D_ZOOM_MAX);
            }
            "Pitch Up" => {
                self.pitch =
                    (self.pitch - 0.1_f32).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
            }
            "Pitch Down" => {
                self.pitch =
                    (self.pitch + 0.1_f32).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
            }
            "Yaw Left" => {
                self.yaw =
                    (self.yaw - 0.1_f32).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
            }
            "Yaw Right" => {
                self.yaw =
                    (self.yaw + 0.1_f32).clamp(-CAMERA_3D_MAX_ANGLE_RAD, CAMERA_3D_MAX_ANGLE_RAD);
            }
            "Pan Up" => self.pan_by_pixels(egui::vec2(0.0, -step_px)),
            "Pan Down" => self.pan_by_pixels(egui::vec2(0.0, step_px)),
            "Pan Left" => self.pan_by_pixels(egui::vec2(-step_px, 0.0)),
            "Pan Right" => self.pan_by_pixels(egui::vec2(step_px, 0.0)),
            _ => {}
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
