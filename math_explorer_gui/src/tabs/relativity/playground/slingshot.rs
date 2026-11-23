#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Tactile pointer-drag slingshot and WCAG 2.2 AA keyboard reticle launch gestures.

use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Line, PlotPoints, PlotUi, Points};

use super::{GeodesicInteractionMode, GeodesicPlaygroundTool, SpacetimePreset};

/// Transient pointer drag and keyboard reticle state for `Slingshot` and `PhotonFan` launches.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SlingshotDragState {
    /// Active drag origin in plot/world coordinates `[x, y]`.
    pub drag_origin: Option<[f64; 2]>,
    /// Current drag tip in plot/world coordinates `[x, y]`.
    pub drag_tip: Option<[f64; 2]>,
    /// Keyboard reticle impact offset and radial bias adjustments.
    pub reticle_active: bool,
}

impl GeodesicPlaygroundTool {
    /// Returns `true` when `Camera3D` rotation must be locked via `dragging_callout`.
    pub const fn locks_camera_orbit(&self) -> bool {
        !matches!(self.state.interaction_mode, GeodesicInteractionMode::Orbit)
    }

    /// Processes pointer slingshot drags and draws aim vectors inside `Plot3DViewport`.
    pub(super) fn interact_3d_slingshot(
        &mut self,
        plot_ui: &mut PlotUi<'_>,
        palette: &SemanticPalette,
    ) -> bool {
        let pid = plot_ui.plot_id();
        let lock_orbit = self.locks_camera_orbit();
        plot_ui.ctx().data_mut(|d| {
            d.insert_temp(pid.with("dragging_callout"), lock_orbit);
        });
        if !lock_orbit {
            self.slingshot.drag_origin = None;
            self.slingshot.drag_tip = None;
            return false;
        }

        let resp = plot_ui.response();
        let ptr_coord = plot_ui.pointer_coordinate().map(|p| [p.x, p.y]);
        let mut launched = false;

        if resp.drag_started_by(egui::PointerButton::Primary)
            || resp.clicked_by(egui::PointerButton::Primary)
        {
            if let Some(pt) = ptr_coord {
                self.slingshot.drag_origin = Some(pt);
                self.slingshot.drag_tip = Some(pt);
            }
        }
        if resp.dragged_by(egui::PointerButton::Primary) {
            if let Some(pt) = ptr_coord {
                if self.slingshot.drag_origin.is_none() {
                    self.slingshot.drag_origin = Some(pt);
                }
                self.slingshot.drag_tip = Some(pt);
                if let Some(origin) = self.slingshot.drag_origin {
                    self.apply_aim_vector(origin, pt);
                    launched = true;
                }
            }
        }
        if resp.drag_stopped() {
            if let (Some(origin), Some(tip)) = (self.slingshot.drag_origin, self.slingshot.drag_tip)
            {
                self.apply_aim_vector(origin, tip);
                launched = true;
            }
            self.slingshot.drag_origin = None;
            self.slingshot.drag_tip = None;
        }

        self.draw_slingshot_reticle_3d(plot_ui, palette);
        launched
    }

    /// Processes `Arrow`, `Space`, and `Enter` keyboard launch gestures on a focused viewport.
    pub fn handle_keyboard_slingshot(&mut self, ctx: &egui::Context, has_focus: bool) -> bool {
        if !has_focus {
            return false;
        }
        let (dx, dy, fire) = read_keyboard_slingshot_delta(ctx);
        let mut changed = false;
        if dx != 0.0 || dy != 0.0 {
            if self.state.interaction_mode == GeodesicInteractionMode::Orbit {
                self.state.interaction_mode = GeodesicInteractionMode::Slingshot;
            }
            self.state.impact_or_momentum =
                (self.state.impact_or_momentum + 0.30 * dx).clamp(-14.0, 14.0);
            self.state.radial_bias = (self.state.radial_bias + 0.06 * dy).clamp(-0.98, 0.98);
            self.slingshot.reticle_active = true;
            self.clamp_state_in_place();
            self.rebuild_trajectory(260);
            self.state.scrub_index = self.samples.len().saturating_sub(1);
            self.revision = self.revision.wrapping_add(1);
            crate::accessibility::announce_live_status(
                ctx,
                format!(
                    "Geodesic reticle impact {:.2}, radial bias {:.2}",
                    self.state.impact_or_momentum, self.state.radial_bias
                ),
            );
            changed = true;
        }
        if fire {
            if self.state.interaction_mode == GeodesicInteractionMode::Orbit {
                self.state.interaction_mode = GeodesicInteractionMode::PhotonFan;
            }
            self.clamp_state_in_place();
            self.rebuild_trajectory(360);
            self.state.scrub_index = self.samples.len().saturating_sub(1);
            self.state.is_playing = true;
            self.revision = self.revision.wrapping_add(1);
            crate::accessibility::announce_live_status(
                ctx,
                format!(
                    "Launched {} ({} rays)",
                    self.state.interaction_mode.label(),
                    self.active_ray_count.max(1)
                ),
            );
            changed = true;
        }
        changed
    }

    /// Applies a 2D normalized drag `(origin_uv -> tip_uv)` on the lensing canvas to launch rays.
    pub(super) fn apply_lensing_canvas_drag(&mut self, origin_uv: [f64; 2], tip_uv: [f64; 2]) {
        let fov_b = 6.0 * self.state.scale_radius.max(1.0);
        let bx = origin_uv[0] * fov_b;
        let by = origin_uv[1] * fov_b;
        let b_signed = if bx >= 0.0 {
            bx.hypot(by)
        } else {
            -bx.hypot(by)
        };
        let drag_dx = tip_uv[0] - origin_uv[0];
        let drag_dy = tip_uv[1] - origin_uv[1];
        self.state.impact_or_momentum = (b_signed + 4.0 * drag_dx).clamp(-14.0, 14.0);
        self.state.radial_bias = (-0.85 + 0.6 * drag_dy).clamp(-0.98, 0.98);
        self.clamp_state_in_place();
        self.rebuild_trajectory(300);
        self.state.scrub_index = self.samples.len().saturating_sub(1);
        self.revision = self.revision.wrapping_add(1);
    }

    fn apply_aim_vector(&mut self, origin: [f64; 2], tip: [f64; 2]) {
        let r_proj = origin[0].hypot(origin[1]).clamp(2.4, 17.5);
        let phi = origin[1].atan2(origin[0]);
        let vx = tip[0] - origin[0];
        let vy = tip[1] - origin[1];
        let drag_len = vx.hypot(vy);

        if self.state.spacetime != SpacetimePreset::MorrisThorneWormhole {
            self.state.initial_radius = r_proj.max(1.25 * self.state.scale_radius);
        } else {
            self.state.initial_radius = r_proj.clamp(1.5, 12.0);
        }
        self.state.launch_azimuth = phi;

        if drag_len > 0.08 {
            let (sin_p, cos_p) = phi.sin_cos();
            let v_rad = (vx * cos_p + vy * sin_p) / drag_len;
            let v_tan = (-vx * sin_p + vy * cos_p) / drag_len;
            let speed_scale = (drag_len / 4.0).clamp(0.35, 2.2);
            self.state.radial_bias = v_rad.clamp(-0.98, 0.98);
            self.state.impact_or_momentum =
                (r_proj * v_tan * speed_scale * 0.85).clamp(-14.0, 14.0);
        }
        self.clamp_state_in_place();
        self.rebuild_trajectory(300);
        self.state.scrub_index = self.samples.len().saturating_sub(1);
        self.revision = self.revision.wrapping_add(1);
    }

    fn draw_slingshot_reticle_3d(&self, plot_ui: &mut PlotUi<'_>, palette: &SemanticPalette) {
        if let (Some(origin), Some(tip)) = (self.slingshot.drag_origin, self.slingshot.drag_tip) {
            plot_ui.line(
                Line::new("Slingshot Aim Vector", PlotPoints::new(vec![origin, tip]))
                    .color(palette.warning)
                    .width(2.5_f32),
            );
            plot_ui.points(
                Points::new("Launch Anchor", vec![origin])
                    .radius(6.0_f32)
                    .color(palette.accent),
            );
        } else if let Some(first) = self.samples.first() {
            let anchor = self.state.camera.project(&first.embed_xyz);
            plot_ui.points(
                Points::new("Launch Reticle", vec![anchor])
                    .radius(5.5_f32)
                    .color(palette.warning),
            );
        }
    }
}

fn read_keyboard_slingshot_delta(ctx: &egui::Context) -> (f64, f64, bool) {
    ctx.input_mut(|i| {
        if i.modifiers.ctrl || i.modifiers.command || i.modifiers.alt {
            return (0.0, 0.0, false);
        }
        let mut x = 0.0_f64;
        let mut y = 0.0_f64;
        for (key, dx, dy) in [
            (egui::Key::ArrowRight, 1.0, 0.0),
            (egui::Key::ArrowLeft, -1.0, 0.0),
            (egui::Key::ArrowUp, 0.0, 1.0),
            (egui::Key::ArrowDown, 0.0, -1.0),
        ] {
            if i.consume_key(egui::Modifiers::SHIFT, key) {
                x += 3.0 * dx;
                y += 3.0 * dy;
            } else if i.consume_key(egui::Modifiers::NONE, key) {
                x += dx;
                y += dy;
            }
        }
        let trigger = i.consume_key(egui::Modifiers::NONE, egui::Key::Space)
            || i.consume_key(egui::Modifiers::NONE, egui::Key::Enter);
        (x, y, trigger)
    })
}
