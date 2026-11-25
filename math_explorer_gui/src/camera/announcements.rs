//! Polite live-region status announcements and `AccessKit` orientation values for [`super::Camera3D`].

use eframe::egui;

use super::{gizmo::GizmoAxis, Camera3D};
use crate::accessibility::announce_live_status;

impl Camera3D {
    /// Formats 3D camera orientation (`Pitch`, `Yaw`, `Zoom`, `Projection`) and optional tool
    /// status summary for the `Role::GraphicsDocument` `AccessKit` `value` attribute.
    #[must_use]
    pub(crate) fn orientation_value_text(&self, status: Option<&str>) -> String {
        let base = format!(
            "3D View · Pitch {:.1}° · Yaw {:.1}° · Zoom {:.2}× · {}",
            self.pitch.to_degrees(),
            self.yaw.to_degrees(),
            self.zoom,
            self.projection.label()
        );
        match status.map(str::trim).filter(|s| !s.is_empty()) {
            Some(extra) => format!("{base} · {extra}"),
            None => base,
        }
    }

    pub(super) fn announce_gizmo_snap(&self, ctx: &egui::Context, axis: GizmoAxis) {
        let view_name = match axis {
            GizmoAxis::X => "X-axis (YZ plane)",
            GizmoAxis::Y => "Y-axis (XZ plane)",
            GizmoAxis::Z => "Z-axis (XY plane)",
            GizmoAxis::Iso => "Isometric",
        };
        self.announce_view_snap(ctx, view_name);
    }

    pub(super) fn announce_view_snap(&self, ctx: &egui::Context, view_name: &str) {
        announce_live_status(
            ctx,
            format!(
                "Camera snapped to {view_name} view (Pitch {:.1}°, Yaw {:.1}°)",
                self.pitch.to_degrees(),
                self.yaw.to_degrees()
            ),
        );
    }

    pub(super) fn announce_projection(&self, ctx: &egui::Context) {
        announce_live_status(
            ctx,
            format!("Camera projection switched to {}", self.projection.label()),
        );
    }

    pub(super) fn announce_reset(&self, ctx: &egui::Context) {
        announce_live_status(
            ctx,
            format!(
                "Camera view reset (Pitch {:.1}°, Yaw {:.1}°, Zoom {:.2}×, {})",
                self.pitch.to_degrees(),
                self.yaw.to_degrees(),
                self.zoom,
                self.projection.label()
            ),
        );
    }

    pub(super) fn announce_command_step(&self, ctx: &egui::Context, action: &str) {
        announce_live_status(
            ctx,
            format!(
                "{action}: Pitch {:.1}°, Yaw {:.1}°, Zoom {:.2}×",
                self.pitch.to_degrees(),
                self.yaw.to_degrees(),
                self.zoom
            ),
        );
    }
}
