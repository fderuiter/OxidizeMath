//! Direct 3D Bloch sphere handle dragging, WCAG 2.2 AA keyboard steering, and Zeeman
//! sonification controls for [`super::SpinVisualizer`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::f64::consts::{PI, TAU};

use eframe::egui;
use math_explorer::physics::quantum::spin;
use nalgebra::Point3;

use super::SpinVisualizer;
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility, AccessibleTheoryHover,
};
use crate::framework::{PlaybackAction, PlaybackTransport, Plot3DViewport};

/// Hit-test radius in logical pixels for grabbing 3D Bloch sphere tip handles.
pub const BLOCH_HANDLE_HIT_RADIUS_PX: f32 = 24.0;
/// Discrete angular step (`5 deg` in radians) per keyboard arrow press on the Bloch sphere.
pub const BLOCH_KEY_STEP_RAD: f64 = 5.0 * PI / 180.0;
/// Base visual radius of the $\vec{B}$ tip handle outside the unit Bloch sphere.
pub const B_HANDLE_BASE_RADIUS: f64 = 1.18;

/// Selectable 3D vector steered by canvas arrow keys and direct Bloch handle dragging.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlochSteerTarget {
    /// Initial quantum spin state $|\psi(0)\rangle$ on $S^2$.
    #[default]
    InitialState,
    /// External Zeeman magnetic field vector $\vec{B}$.
    MagneticField,
}

pub(super) fn wrap_azimuth_rad(phi: f64) -> f64 {
    if !phi.is_finite() {
        return 0.0;
    }
    let shifted = (phi + PI).rem_euclid(TAU);
    shifted - PI
}

pub(super) fn b_field_visual_tip(b_field: [f64; 3]) -> Option<(Point3<f64>, f64)> {
    let [bx, by, bz] = b_field;
    let b_mag = (bx * bx + by * by + bz * bz).sqrt();
    if !b_mag.is_finite() || b_mag <= 1.0e-6 {
        return None;
    }
    let radius = (B_HANDLE_BASE_RADIUS + 0.04 * b_mag.min(5.0)).clamp(1.18, 1.38);
    let inv = radius / b_mag;
    Some((Point3::new(bx * inv, by * inv, bz * inv), radius))
}

fn plot_to_screen(rect: egui::Rect, bounds: [f64; 4], plot_xy: [f64; 2]) -> egui::Pos2 {
    let [min_x, max_x, min_y, max_y] = bounds;
    let span_x = (max_x - min_x).max(1.0e-6);
    let span_y = (max_y - min_y).max(1.0e-6);
    let u = ((plot_xy[0] - min_x) / span_x) as f32;
    let v = ((max_y - plot_xy[1]) / span_y) as f32;
    egui::pos2(
        rect.left() + u * rect.width(),
        rect.top() + v * rect.height(),
    )
}

fn screen_to_plot(rect: egui::Rect, bounds: [f64; 4], pos: egui::Pos2) -> [f64; 2] {
    let [min_x, max_x, min_y, max_y] = bounds;
    let u = f64::from(((pos.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0));
    let v = f64::from(((pos.y - rect.top()) / rect.height().max(1.0)).clamp(0.0, 1.0));
    [min_x + u * (max_x - min_x), max_y - v * (max_y - min_y)]
}

impl SpinVisualizer {
    /// Returns the active $(\theta, \phi)$ spherical Bloch angles of $|\psi(0)\rangle$.
    #[must_use]
    pub fn current_initial_angles(&self) -> [f64; 2] {
        if let Some(angles) = self.initial_bloch_angles {
            return angles;
        }
        let ex = self.sx.expectation_value(&self.psi).re;
        let ey = self.sy.expectation_value(&self.psi).re;
        let ez = self.sz.expectation_value(&self.psi).re;
        let theta = ez.clamp(-1.0, 1.0).acos();
        let phi = if ex.hypot(ey) > 1.0e-9 {
            ey.atan2(ex)
        } else {
            0.0
        };
        [theta, phi]
    }

    /// Sets the prepared initial Bloch state $|\psi(0)\rangle$ at $(\theta, \phi)$ in place.
    pub fn set_initial_bloch_angles(&mut self, theta: f64, phi: f64) {
        let clamped_theta = if theta.is_finite() {
            theta.clamp(0.0, PI)
        } else {
            0.0
        };
        let wrapped_phi = wrap_azimuth_rad(phi);
        self.initial_bloch_angles = Some([clamped_theta, wrapped_phi]);
        self.time = 0.0;
        spin::set_spin_bloch_angles_in_place(&mut self.psi, clamped_theta, wrapped_phi);
    }

    /// Steers $|\psi(0)\rangle$ directly to a unit vector on $S^2$.
    pub fn steer_initial_state_to_unit(&mut self, dir: [f64; 3]) {
        let prev_phi = self.current_initial_angles()[1];
        let theta = dir[2].clamp(-1.0, 1.0).acos();
        let phi = if dir[0].hypot(dir[1]) > 1.0e-9 {
            dir[1].atan2(dir[0])
        } else {
            prev_phi
        };
        self.set_initial_bloch_angles(theta, phi);
    }

    /// Steers the $\vec{B}$ field direction onto `dir` while preserving field magnitude $|\vec{B}|$.
    pub fn steer_b_field_to_unit(&mut self, dir: [f64; 3]) {
        let [bx, by, bz] = self.b_field;
        let b_mag = (bx * bx + by * by + bz * bz).sqrt();
        let mag = if b_mag.is_finite() && b_mag > 1.0e-6 {
            b_mag
        } else {
            1.0
        };
        self.b_field = [
            (dir[0] * mag).clamp(-5.0, 5.0),
            (dir[1] * mag).clamp(-5.0, 5.0),
            (dir[2] * mag).clamp(-5.0, 5.0),
        ];
    }

    /// Steps either $|\psi(0)\rangle$ or $\vec{B}$ by `(d_theta, d_phi)` radians and formats an ARIA status string.
    pub fn step_steer_angles(
        &mut self,
        target: BlochSteerTarget,
        d_theta: f64,
        d_phi: f64,
    ) -> String {
        match target {
            BlochSteerTarget::InitialState => {
                let [theta, phi] = self.current_initial_angles();
                let next_theta = (theta + d_theta).clamp(0.0, PI);
                let next_phi = wrap_azimuth_rad(phi + d_phi);
                self.set_initial_bloch_angles(next_theta, next_phi);
                format!(
                    "Initial spin state steered to theta {:.0} degrees, phi {:.0} degrees",
                    next_theta.to_degrees(),
                    next_phi.to_degrees()
                )
            }
            BlochSteerTarget::MagneticField => {
                let [bx, by, bz] = self.b_field;
                let mag = (bx * bx + by * by + bz * bz).sqrt().max(0.25);
                let cur_theta = (bz / mag).clamp(-1.0, 1.0).acos();
                let cur_phi = if bx.hypot(by) > 1.0e-9 {
                    by.atan2(bx)
                } else {
                    0.0
                };
                let next_theta = (cur_theta + d_theta).clamp(0.0, PI);
                let next_phi = wrap_azimuth_rad(cur_phi + d_phi);
                let (st, ct) = next_theta.sin_cos();
                let (sp, cp) = next_phi.sin_cos();
                self.steer_b_field_to_unit([st * cp, st * sp, ct]);
                format!(
                    "Magnetic field B steered to [{:.2}, {:.2}, {:.2}]",
                    self.b_field[0], self.b_field[1], self.b_field[2]
                )
            }
        }
    }

    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui, ex: f64, ey: f64, ez: f64) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "spin_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Controls");
            ui.separator();

            ui.group(|ui| {
                ui.add(
                    egui::Slider::new(&mut self.b_field[0], -5.0..=5.0)
                        .text("Magnetic Field B - Bx"),
                );
                ui.add(egui::Slider::new(&mut self.b_field[1], -5.0..=5.0).text("By"));
                ui.add(egui::Slider::new(&mut self.b_field[2], -5.0..=5.0).text("Bz"));
            });

            self.draw_bloch_steering_controls(ui);
            ui.separator();
            self.draw_transport_controls(ui);
            ui.separator();
            ui.heading("View");
            self.camera.ui(ui);

            ui.separator();
            let frame = crate::audio::mappings::zeeman_bloch_frame([ex, ey, ez], self.b_field);
            let is_paused = self.paused && self.active_drag.is_none();
            crate::audio::render_sonification_panel(ui, "spin_viz", &frame, is_paused);

            ui.horizontal_wrapped(|ui| {
                ui.label(format!("Ex: {ex:.3}"));
                ui.label(format!("Ey: {ey:.3}"));
                ui.label(format!("Ez: {ez:.3}"));
            });
        });
    }

    fn draw_bloch_steering_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Steer:");
            ui.selectable_value(
                &mut self.steer_target,
                BlochSteerTarget::InitialState,
                "|ψ(0)⟩",
            );
            ui.selectable_value(
                &mut self.steer_target,
                BlochSteerTarget::MagneticField,
                "B Field",
            );
            if ui.small_button("|0⟩").clicked() {
                self.initial_bloch_angles = None;
                self.reset();
            }
        });
        let [theta, phi] = self.current_initial_angles();
        let mut theta_deg = theta.to_degrees();
        let mut phi_deg = phi.to_degrees();
        ui.horizontal(|ui| {
            ui.label("θ₀");
            let tc = ui
                .add(
                    egui::DragValue::new(&mut theta_deg)
                        .range(0.0..=180.0)
                        .speed(1.0)
                        .suffix("°"),
                )
                .changed();
            ui.label("φ₀");
            let pc = ui
                .add(
                    egui::DragValue::new(&mut phi_deg)
                        .range(-180.0..=180.0)
                        .speed(1.0)
                        .suffix("°"),
                )
                .changed();
            if tc || pc {
                self.set_initial_bloch_angles(theta_deg.to_radians(), phi_deg.to_radians());
            }
        });
    }

    fn draw_transport_controls(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = PlaybackTransport::in_thread(
            "spin_viz_transport",
            "Spin Dynamics (Bloch Sphere)",
            !self.paused,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.paused = false,
                PlaybackAction::Pause => self.paused = true,
                PlaybackAction::Step => self.step_once(),
                PlaybackAction::Reset => self.reset(),
            }
        }
    }

    pub(super) fn draw_bloch_sphere(
        &mut self,
        parent_ui: &mut egui::Ui,
        ex: f64,
        ey: f64,
        ez: f64,
    ) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let dark_mode = ui.visuals().dark_mode;
            self.populate_scene(ex, ey, ez, palette.graph_edge, palette.accent, dark_mode);

            let was_dragging = self.active_drag;
            self.detect_handle_press(ui, [ex, ey, ez]);
            let allow_cam_drag = self.active_drag.is_none();
            let mut camera = self.camera;
            let mut captured_bounds = None;
            let response = Plot3DViewport::new("bloch_sphere", 1.5)
                .allow_camera_drag(allow_cam_drag)
                .show(ui, &mut camera, |plot_ui| {
                    self.scene_builder.finish(plot_ui);
                    let pb = plot_ui.plot_bounds();
                    captured_bounds = Some([pb.min()[0], pb.max()[0], pb.min()[1], pb.max()[1]]);
                })
                .response;
            self.camera = camera;

            if let Some(bounds) = captured_bounds {
                self.last_plot_rect = Some(response.rect);
                self.last_plot_bounds = Some(bounds);
                self.update_active_handle_drag(ui, response.rect, bounds, was_dragging);
            }

            let val_desc = format!(
                "Spin [{ex:.2}, {ey:.2}, {ez:.2}], B [{:.2}, {:.2}, {:.2}]",
                self.b_field[0], self.b_field[1], self.b_field[2]
            );
            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::Canvas,
                "3D Quantum Bloch Sphere",
                Some(&val_desc),
            );
            self.handle_bloch_canvas_keyboard(ui, &response);
            response.accessible_theory_hover(&self.psi);
        });
    }

    fn detect_handle_press(&mut self, ui: &egui::Ui, spin_xyz: [f64; 3]) {
        let (primary_down, press_origin, interact_pos) = ui.input(|i| {
            (
                i.pointer.primary_down(),
                i.pointer.press_origin(),
                i.pointer.interact_pos(),
            )
        });
        if !primary_down {
            self.active_drag = None;
            return;
        }
        if self.active_drag.is_some() {
            return;
        }
        let Some(pos) = press_origin.or(interact_pos) else {
            return;
        };
        let (rect, bounds) = self.estimated_plot_frame(ui);
        if !rect.contains(pos) {
            return;
        }
        let spin_px = plot_to_screen(rect, bounds, self.camera.project(&spin_xyz));
        let d_spin = pos.distance(spin_px);
        let d_b = b_field_visual_tip(self.b_field)
            .map(|(pt, _)| {
                pos.distance(plot_to_screen(
                    rect,
                    bounds,
                    self.camera.project(&[pt.x, pt.y, pt.z]),
                ))
            })
            .unwrap_or(f32::INFINITY);

        if d_spin <= BLOCH_HANDLE_HIT_RADIUS_PX || d_b <= BLOCH_HANDLE_HIT_RADIUS_PX {
            let chosen = if d_b < d_spin {
                BlochSteerTarget::MagneticField
            } else {
                BlochSteerTarget::InitialState
            };
            self.active_drag = Some(chosen);
            self.steer_target = chosen;
        }
    }

    fn estimated_plot_frame(&self, ui: &egui::Ui) -> (egui::Rect, [f64; 4]) {
        if let (Some(rect), Some(bounds)) = (self.last_plot_rect, self.last_plot_bounds) {
            return (rect, bounds);
        }
        let avail = ui.available_size_before_wrap();
        let (plot_h, x_half, y_half) = Plot3DViewport::compute_half_extents(avail, 1.5, true);
        let rect = egui::Rect::from_min_size(ui.cursor().min, egui::vec2(avail.x, plot_h));
        (rect, [-x_half, x_half, -y_half, y_half])
    }

    fn update_active_handle_drag(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        bounds: [f64; 4],
        was_dragging: Option<BlochSteerTarget>,
    ) {
        if let Some(target) = self.active_drag {
            if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                let plot_xy = screen_to_plot(rect, bounds, pos);
                match target {
                    BlochSteerTarget::InitialState => {
                        let dir = self.camera.unproject_to_unit_sphere(plot_xy);
                        self.steer_initial_state_to_unit(dir);
                    }
                    BlochSteerTarget::MagneticField => {
                        let scale =
                            b_field_visual_tip(self.b_field).map_or(B_HANDLE_BASE_RADIUS, |t| t.1);
                        let px = f64::from(self.camera.pan_x);
                        let py = f64::from(self.camera.pan_y);
                        let scaled = [
                            px + (plot_xy[0] - px) / scale,
                            py + (plot_xy[1] - py) / scale,
                        ];
                        let dir = self.camera.unproject_to_unit_sphere(scaled);
                        self.steer_b_field_to_unit(dir);
                    }
                }
                ui.ctx().request_repaint();
            }
        } else if let Some(finished) = was_dragging {
            let msg = match finished {
                BlochSteerTarget::InitialState => {
                    let [th, ph] = self.current_initial_angles();
                    format!(
                        "Initial spin state set to theta {:.0} degrees, phi {:.0} degrees",
                        th.to_degrees(),
                        ph.to_degrees()
                    )
                }
                BlochSteerTarget::MagneticField => format!(
                    "Magnetic field B set to [{:.2}, {:.2}, {:.2}]",
                    self.b_field[0], self.b_field[1], self.b_field[2]
                ),
            };
            announce_live_status(ui.ctx(), msg);
        }
    }

    fn handle_bloch_canvas_keyboard(&mut self, ui: &egui::Ui, response: &egui::Response) {
        lock_canvas_arrow_keys(ui, response);
        paint_canvas_focus_ring(ui, response);
        if !response.has_focus() {
            return;
        }
        let (d_theta, d_phi) = ui.input(|i| {
            let mut dt = 0.0;
            let mut dp = 0.0;
            if i.key_pressed(egui::Key::ArrowUp) {
                dt -= BLOCH_KEY_STEP_RAD;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                dt += BLOCH_KEY_STEP_RAD;
            }
            if i.key_pressed(egui::Key::ArrowLeft) {
                dp -= BLOCH_KEY_STEP_RAD;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                dp += BLOCH_KEY_STEP_RAD;
            }
            (dt, dp)
        });
        if d_theta != 0.0 || d_phi != 0.0 {
            let msg = self.step_steer_angles(self.steer_target, d_theta, d_phi);
            announce_live_status(ui.ctx(), msg);
            ui.ctx().request_repaint();
        }
    }
}
