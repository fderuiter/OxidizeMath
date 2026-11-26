//! Tactile wind-tunnel canvas rendering, singularity/probe drag handles,
//! `PlotBrushController` sculpting, and WCAG 2.2 AA keyboard reticle navigation.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::interact::render_flow_layers;
use super::{ElementType, PotentialFlowTool, FLOW_POS_MAX, FLOW_POS_MIN};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use crate::framework::{BrushMode, PlotBrushController, PlotBrushFrame};
use eframe::egui::{self, Color32};
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotUi, Points};
use math_commons::math_kernel::colormap::ensure_dark_canvas_contrast;

fn contrast_safe(color: Color32, dark: bool) -> Color32 {
    if dark {
        let rgb = ensure_dark_canvas_contrast([color.r(), color.g(), color.b()]);
        Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    } else {
        color
    }
}

impl PotentialFlowTool {
    pub(crate) fn draw_wind_tunnel_canvas(&mut self, parent_ui: &mut egui::Ui) {
        let brush_sculpt = self.brush_kit.is_sculpting();
        let allow_pan = !brush_sculpt
            && self.dragged_element.is_none()
            && !self.dragging_probe
            && !self.canvas_stamp_mode;
        let probe_readout = self.show_probe.then(|| self.evaluate_probe());

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let dark = ui.visuals().dark_mode || self.high_contrast;
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let mut pointer_coord: Option<[f64; 2]> = None;
            let mut brush_frame: Option<PlotBrushFrame> = None;
            let vr = self.view_range.max(1.0);
            let plot_resp = Plot::new("potential_flow_plot")
                .data_aspect(1.0)
                .view_aspect(1.0)
                .allow_drag(allow_pan)
                .allow_zoom(allow_pan)
                .include_x(-vr)
                .include_x(vr)
                .include_y(-vr)
                .include_y(vr)
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    pointer_coord = plot_ui.pointer_coordinate().map(|p| [p.x, p.y]);
                    brush_frame = Some(self.populate_wind_tunnel_plot(plot_ui, &palette, dark, vr));
                });

            self.plot_focused = plot_resp.response.has_focus();
            let (u_inf, gamma, lift, _drag) = self.aerodynamic_summary();
            let a11y_val = format!(
                "{} elements, {} stagnation points, U_inf={u_inf:.1}, Gamma={gamma:.1}, Lift={lift:.1}",
                self.elements.len(),
                self.cached_stagnation.len()
            );
            register_canvas_accessibility(
                ui,
                &plot_resp.response,
                egui::accesskit::Role::GraphicsDocument,
                "Potential flow wind-tunnel canvas",
                Some(&a11y_val),
            );
            if let Some(frame) = brush_frame {
                if !frame.is_dragging {
                    self.active_drag_element = None;
                }
                frame.dispatch(ui, &plot_resp.response, self);
            }
            if !brush_sculpt {
                let ctx = ui.ctx().clone();
                self.handle_canvas_pointer(&ctx, &plot_resp.response, pointer_coord);
                self.handle_canvas_keyboard(ui, &ctx, &plot_resp.response);
            }
            if let Some(readout) = probe_readout {
                Self::draw_probe_overlay(ui, &palette, &readout, self.high_contrast);
            }
        });
    }

    fn populate_wind_tunnel_plot<'a>(
        &'a mut self,
        plot_ui: &mut PlotUi<'a>,
        palette: &SemanticPalette,
        dark: bool,
        vr: f64,
    ) -> PlotBrushFrame {
        let arrows = if self.show_arrows {
            (self.origins.as_slice(), self.tips.as_slice())
        } else {
            (&[][..], &[][..])
        };
        render_flow_layers(
            plot_ui,
            palette,
            self.show_streamlines
                .then_some(self.cached_polylines.as_slice()),
            arrows,
            self.show_stagnation_points
                .then_some(self.cached_stagnation.as_slice()),
        );
        self.render_dye_layer(plot_ui, palette, dark);
        self.render_singularity_handles(plot_ui, palette, dark);
        self.render_focus_and_probe_markers(plot_ui, palette, dark);
        self.callout_manager.render_on_plot(plot_ui);
        PlotBrushController::capture_plot(
            &mut self.brush_kit,
            plot_ui,
            [-vr, -vr],
            [vr, vr],
            Some(BrushMode::Erase),
        )
    }

    fn render_dye_layer(&self, plot_ui: &mut PlotUi<'_>, palette: &SemanticPalette, dark: bool) {
        if !self.show_dye_streaks {
            return;
        }
        let dye_col = contrast_safe(palette.info, dark);
        for (idx, streak) in self.dye_streaks.iter().enumerate() {
            if streak.trail.len() < 2 {
                continue;
            }
            let pts: Vec<[f64; 2]> = streak.trail.iter().copied().collect();
            let name = if idx == 0 { "RK4 Dye Streaks" } else { "" };
            plot_ui.line(Line::new(name, pts).color(dye_col).width(2.1_f32));
        }
        let heads: Vec<[f64; 2]> = self.dye_streaks.iter().map(|s| s.head).collect();
        if !heads.is_empty() {
            plot_ui.points(
                Points::new("Dye Tracers", heads)
                    .shape(MarkerShape::Circle)
                    .radius(3.2_f32)
                    .color(dye_col),
            );
        }
    }

    fn render_focus_and_probe_markers(
        &self,
        plot_ui: &mut PlotUi<'_>,
        palette: &SemanticPalette,
        dark: bool,
    ) {
        let sel_col = contrast_safe(palette.node_stroke, dark);
        if let Some(sel) = self
            .selected_element
            .and_then(|i| self.elements.get(i))
            .filter(|s| s.has_canvas_handle())
        {
            plot_ui.points(
                Points::new("Selected Element", vec![[sel.x, sel.y]])
                    .shape(MarkerShape::Circle)
                    .filled(false)
                    .radius(10.5_f32)
                    .color(sel_col),
            );
        }
        if self.show_probe {
            let p_col = contrast_safe(palette.series_color(3), dark);
            plot_ui.points(
                Points::new("Lagrangian Probe", vec![self.probe_pos])
                    .shape(MarkerShape::Asterisk)
                    .radius(7.5_f32)
                    .color(p_col),
            );
        }
        if self.plot_focused && !self.brush_kit.is_sculpting() {
            plot_ui.points(
                Points::new("Keyboard Reticle", vec![self.focused_reticle])
                    .shape(MarkerShape::Square)
                    .filled(false)
                    .radius(9.0_f32)
                    .color(sel_col),
            );
        }
    }

    fn draw_probe_overlay(
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        readout: &math_explorer::physics::fluid_dynamics::solver::TracerProbeReadout,
        high_contrast: bool,
    ) {
        let rect = ui.max_rect();
        let pos = rect.left_top() + egui::vec2(14.0, 14.0);
        let stroke_w = if high_contrast { 2.0_f32 } else { 1.0_f32 };
        egui::Area::new(ui.make_persistent_id("potential_flow_probe_hud"))
            .fixed_pos(pos)
            .interactable(false)
            .show(ui.ctx(), |ui| {
                egui::Frame::window(ui.style())
                    .stroke(egui::Stroke::new(stroke_w, palette.graph_edge))
                    .show(ui, |ui| {
                        ui.strong("Lagrangian Probe (Euler + Bernoulli)");
                        ui.label(format!(
                            "x = ({:+.2}, {:+.2})  |u| = {:.2} m/s",
                            readout.position.x,
                            readout.position.y,
                            readout.velocity.norm()
                        ));
                        ui.label(format!(
                            "Du/Dt = ({:+.2}, {:+.2}) m/s²",
                            readout.convective_acceleration.x, readout.convective_acceleration.y
                        ));
                        ui.label(format!(
                            "p = {:.1} Pa   Cp = {:+.3}",
                            readout.bernoulli_pressure, readout.pressure_coefficient
                        ));
                    });
            });
    }

    pub(crate) fn hit_test_handle(&self, pos: [f64; 2]) -> Option<usize> {
        let hit_r = (self.view_range * 0.065).max(0.25);
        let mut best: Option<(usize, f64)> = None;
        for (idx, el) in self.elements.iter().enumerate() {
            if !el.has_canvas_handle() {
                continue;
            }
            let d = (pos[0] - el.x).hypot(pos[1] - el.y);
            if d <= hit_r && best.is_none_or(|(_, b)| d < b) {
                best = Some((idx, d));
            }
        }
        best.map(|(idx, _)| idx)
    }

    fn hit_test_probe(&self, pos: [f64; 2]) -> bool {
        let hit_r = (self.view_range * 0.065).max(0.25);
        self.show_probe && (pos[0] - self.probe_pos[0]).hypot(pos[1] - self.probe_pos[1]) <= hit_r
    }

    fn handle_canvas_pointer(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        pointer_coord: Option<[f64; 2]>,
    ) {
        let primary_pressed = ctx.input(|i| i.pointer.primary_pressed());
        let primary_down = ctx.input(|i| i.pointer.primary_down());
        if !primary_down {
            if self.dragged_element.take().is_some() {
                self.reseed_dye_emitters();
            }
            self.dragging_probe = false;
        }
        let Some(pos) = pointer_coord else {
            return;
        };
        let cx = pos[0].clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        let cy = pos[1].clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        if response.hovered() && primary_pressed {
            response.request_focus();
            if self.handle_canvas_press(ctx, cx, cy) {
                return;
            }
        }
        if primary_down {
            self.update_active_pointer_drag(cx, cy);
        }
    }

    fn handle_canvas_press(&mut self, ctx: &egui::Context, cx: f64, cy: f64) -> bool {
        self.focused_reticle = [cx, cy];
        if let Some(idx) = self.hit_test_handle([cx, cy]) {
            self.selected_element = Some(idx);
            self.dragged_element = Some(idx);
            let el = &self.elements[idx];
            announce_live_status(
                ctx,
                format!(
                    "Selected {} at ({:.2}, {:.2})",
                    el.element_type.label(),
                    el.x,
                    el.y
                ),
            );
            return true;
        }
        if self.hit_test_probe([cx, cy]) {
            self.dragging_probe = true;
            announce_live_status(ctx, "Selected Lagrangian probe");
            return true;
        }
        if self.canvas_stamp_mode && self.selected_element_type != ElementType::Uniform {
            self.add_element_at(cx, cy);
            announce_live_status(
                ctx,
                format!(
                    "Stamped {} at ({cx:.2}, {cy:.2})",
                    self.selected_element_type.label()
                ),
            );
            return true;
        }
        false
    }

    fn update_active_pointer_drag(&mut self, cx: f64, cy: f64) {
        if let Some(el) = self
            .dragged_element
            .and_then(|idx| self.elements.get_mut(idx))
        {
            if (el.x - cx).abs() > 1e-6 || (el.y - cy).abs() > 1e-6 {
                el.x = cx;
                el.y = cy;
                self.focused_reticle = [cx, cy];
                self.rebuild_field();
            }
        } else if self.dragging_probe {
            self.probe_pos = [cx, cy];
            self.focused_reticle = [cx, cy];
        }
    }

    pub(crate) fn handle_canvas_keyboard(
        &mut self,
        ui: &egui::Ui,
        ctx: &egui::Context,
        response: &egui::Response,
    ) {
        if !response.has_focus() {
            return;
        }
        lock_canvas_arrow_keys(ui, response);
        let step = (self.view_range * 0.05).max(0.1);
        let (dx, dy, stamp, delete, clear_sel, alt) = ctx.input(|i| {
            let s = if i.modifiers.shift { step * 4.0 } else { step };
            let mut dx = 0.0;
            let mut dy = 0.0;
            if i.key_pressed(egui::Key::ArrowLeft) {
                dx -= s;
            }
            if i.key_pressed(egui::Key::ArrowRight) {
                dx += s;
            }
            if i.key_pressed(egui::Key::ArrowUp) {
                dy += s;
            }
            if i.key_pressed(egui::Key::ArrowDown) {
                dy -= s;
            }
            let stamp = i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space);
            let delete = i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace);
            let esc = i.key_pressed(egui::Key::Escape);
            (dx, dy, stamp, delete, esc, i.modifiers.alt)
        });
        self.apply_keyboard_commands(ctx, [dx, dy], stamp, delete, clear_sel, alt);
    }

    pub(crate) fn apply_keyboard_commands(
        &mut self,
        ctx: &egui::Context,
        delta: [f64; 2],
        stamp: bool,
        delete: bool,
        clear_sel: bool,
        alt_probe: bool,
    ) {
        if clear_sel {
            self.selected_element = None;
            announce_live_status(ctx, "Cleared element selection");
        }
        if delta[0].abs() > 0.0 || delta[1].abs() > 0.0 {
            self.nudge_keyboard_target(ctx, delta, alt_probe);
        }
        if delete {
            if let Some(idx) = self.selected_element.filter(|&i| i < self.elements.len()) {
                let removed = self.elements.remove(idx);
                self.selected_element = None;
                self.rebuild_field();
                self.reseed_dye_emitters();
                announce_live_status(ctx, format!("Removed {}", removed.element_type.label()));
            }
        } else if stamp && self.selected_element_type != ElementType::Uniform {
            let [rx, ry] = self.focused_reticle;
            self.add_element_at(rx, ry);
            announce_live_status(
                ctx,
                format!(
                    "Placed {} at ({rx:.2}, {ry:.2})",
                    self.selected_element_type.label()
                ),
            );
        }
    }

    fn nudge_keyboard_target(&mut self, ctx: &egui::Context, delta: [f64; 2], alt_probe: bool) {
        if alt_probe && self.show_probe {
            self.probe_pos[0] = (self.probe_pos[0] + delta[0]).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
            self.probe_pos[1] = (self.probe_pos[1] + delta[1]).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
            let [px, py] = self.probe_pos;
            announce_live_status(ctx, format!("Probe moved to ({px:.2}, {py:.2})"));
        } else if let Some(el) = self
            .selected_element
            .and_then(|idx| self.elements.get_mut(idx))
            .filter(|e| e.has_canvas_handle())
        {
            el.x = (el.x + delta[0]).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
            el.y = (el.y + delta[1]).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
            let (nx, ny, label) = (el.x, el.y, el.element_type.label());
            self.focused_reticle = [nx, ny];
            self.rebuild_field();
            self.reseed_dye_emitters();
            announce_live_status(ctx, format!("Moved {label} to ({nx:.2}, {ny:.2})"));
        } else {
            self.focused_reticle[0] =
                (self.focused_reticle[0] + delta[0]).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
            self.focused_reticle[1] =
                (self.focused_reticle[1] + delta[1]).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
            let [rx, ry] = self.focused_reticle;
            announce_live_status(ctx, format!("Reticle at ({rx:.2}, {ry:.2})"));
        }
    }
}
