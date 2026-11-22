#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    DragTarget, ElementType, FocusTarget, PotentialFlowTool, FLOW_POS_MAX, FLOW_POS_MIN,
    FLOW_STRENGTH_MAX, FLOW_STRENGTH_MIN,
};
use crate::accessibility::{
    announce_live_status, lock_canvas_arrow_keys, paint_canvas_focus_ring,
    register_canvas_accessibility,
};
use crate::app::SemanticPalette;
use eframe::egui;
use egui_plot::{Arrows, Line, MarkerShape, Plot, PlotPoint, PlotUi, Points};

impl PotentialFlowTool {
    pub(super) fn hit_test_cached_handle(
        &self,
        pos: egui::Pos2,
        radius: f32,
    ) -> Option<DragTarget> {
        self.cached_handles
            .iter()
            .filter_map(|&(target, screen_pos)| {
                let d = (pos - screen_pos).length();
                (d <= radius).then_some((target, d))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(t, _)| t)
    }

    pub(super) fn render_canvas(&mut self, ui: &mut egui::Ui) {
        let hit_radius = if ui.ctx().input(|i| i.any_touches()) {
            22.0
        } else {
            14.0
        };
        let press_hits = ui
            .input(|i| i.pointer.press_origin().or_else(|| i.pointer.latest_pos()))
            .and_then(|p| self.hit_test_cached_handle(p, hit_radius))
            .is_some();
        let hover_hits = ui
            .input(|i| i.pointer.hover_pos())
            .and_then(|p| self.hit_test_cached_handle(p, hit_radius))
            .is_some();
        let allow_drag = !self.stamp_mode && self.active_drag.is_none() && !press_hits;
        let allow_wheel_zoom = !hover_hits && self.active_drag.is_none();

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let r = self.view_range.max(0.5);
            let plot_resp = Plot::new("potential_flow_plot")
                .data_aspect(1.0)
                .view_aspect(1.0)
                .include_x(-r)
                .include_x(r)
                .include_y(-r)
                .include_y(r)
                .allow_drag(allow_drag)
                .allow_zoom(allow_wheel_zoom)
                .allow_scroll(allow_wheel_zoom)
                .show(ui, |plot_ui| {
                    self.refresh_cached_handles(plot_ui);
                    self.handle_canvas_pointer(plot_ui, hit_radius);
                    self.handle_canvas_wheel(plot_ui, hit_radius);
                    if plot_ui.response().has_focus() {
                        self.handle_canvas_keyboard(plot_ui);
                    }
                    let focused = plot_ui.response().has_focus();
                    self.draw_plot_layers(plot_ui, focused);
                    self.refresh_cached_handles(plot_ui);
                });

            self.last_canvas_id = Some(plot_resp.response.id);
            lock_canvas_arrow_keys(ui, &plot_resp.response);
            paint_canvas_focus_ring(ui, &plot_resp.response);
            let summary = self.canvas_accessibility_summary();
            register_canvas_accessibility(
                ui,
                &plot_resp.response,
                egui::accesskit::Role::GraphicsDocument,
                "Potential Flow Wind Tunnel Canvas",
                Some(&summary),
            );
        });
    }

    fn refresh_cached_handles(&mut self, plot_ui: &PlotUi<'_>) {
        self.cached_handles.clear();
        for (idx, el) in self.elements.iter().enumerate() {
            if el.element_type != ElementType::Uniform {
                let sp = plot_ui.screen_from_plot(PlotPoint::new(el.x, el.y));
                self.cached_handles.push((DragTarget::Element(idx), sp));
            }
        }
        let rake_sp = plot_ui.screen_from_plot(PlotPoint::new(self.dye_rake_x, self.dye_rake_y));
        self.cached_handles.push((DragTarget::DyeRake, rake_sp));
    }

    fn handle_canvas_pointer(&mut self, plot_ui: &PlotUi<'_>, hit_radius: f32) {
        let ctx = plot_ui.ctx();
        let (primary_pressed, primary_down, shift_down, latest_pos) = ctx.input(|i| {
            (
                i.pointer.button_pressed(egui::PointerButton::Primary),
                i.pointer.button_down(egui::PointerButton::Primary),
                i.modifiers.shift,
                i.pointer.latest_pos(),
            )
        });
        if !primary_down {
            self.active_drag = None;
        }
        if primary_pressed {
            if let Some(pos) = latest_pos.filter(|p| plot_ui.response().rect.contains(*p)) {
                if let Some(target) = self.hit_test_cached_handle(pos, hit_radius) {
                    self.active_drag = Some(target);
                    if let DragTarget::Element(idx) = target {
                        self.selected_element = Some(idx);
                        self.focus_target = FocusTarget::Element(idx);
                    } else {
                        self.focus_target = FocusTarget::DyeRake;
                    }
                    plot_ui.response().request_focus();
                }
            }
        }
        if let (true, Some(target), Some(pos)) = (primary_down, self.active_drag, latest_pos) {
            let coord = plot_ui.plot_from_screen(pos);
            self.apply_drag_target(target, coord.x, coord.y);
            return;
        }
        if plot_ui.response().clicked()
            && (self.stamp_mode || shift_down)
            && latest_pos
                .and_then(|p| self.hit_test_cached_handle(p, hit_radius))
                .is_none()
        {
            if let Some(idx) = plot_ui
                .pointer_coordinate()
                .and_then(|c| self.stamp_element_at(c.x, c.y))
            {
                let stamped = self.elements[idx].clone();
                self.focus_target = FocusTarget::Element(idx);
                plot_ui.response().request_focus();
                announce_live_status(
                    ctx,
                    format!(
                        "Stamped {} at ({:.1}, {:.1})",
                        stamped.element_type.label(),
                        stamped.x,
                        stamped.y
                    ),
                );
            }
        }
    }

    pub(crate) fn apply_drag_target(&mut self, target: DragTarget, x: f64, y: f64) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        let (cx, cy) = (
            x.clamp(FLOW_POS_MIN, FLOW_POS_MAX),
            y.clamp(FLOW_POS_MIN, FLOW_POS_MAX),
        );
        match target {
            DragTarget::Element(idx) => {
                if let Some(el) = self.elements.get_mut(idx) {
                    if (el.x - cx).abs() > 1e-6 || (el.y - cy).abs() > 1e-6 {
                        el.x = cx;
                        el.y = cy;
                        self.rebuild_field();
                    }
                }
            }
            DragTarget::DyeRake => {
                if (self.dye_rake_x - cx).abs() > 1e-6 || (self.dye_rake_y - cy).abs() > 1e-6 {
                    self.dye_rake_x = cx;
                    self.dye_rake_y = cy;
                    self.reseed_dye_streaks();
                }
            }
        }
    }

    fn handle_canvas_wheel(&mut self, plot_ui: &PlotUi<'_>, hit_radius: f32) {
        let ctx = plot_ui.ctx();
        let (hover_pos, scroll_y) = ctx.input(|i| {
            let sy = if i.raw_scroll_delta.y.abs() > f32::EPSILON {
                i.raw_scroll_delta.y
            } else {
                i.smooth_scroll_delta.y
            };
            (i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()), sy)
        });
        if scroll_y.abs() <= f32::EPSILON {
            return;
        }
        if let Some(DragTarget::Element(idx)) =
            hover_pos.and_then(|p| self.hit_test_cached_handle(p, hit_radius))
        {
            if let Some(el) = self.elements.get_mut(idx) {
                let delta = f64::from(scroll_y.signum()) * 2.5;
                el.strength = (el.strength + delta).clamp(FLOW_STRENGTH_MIN, FLOW_STRENGTH_MAX);
                let strength = el.strength;
                let kind = el.element_type.label();
                self.selected_element = Some(idx);
                self.rebuild_field();
                announce_live_status(ctx, format!("{kind} strength {strength:.1}"));
            }
        }
    }

    fn handle_canvas_keyboard(&mut self, plot_ui: &PlotUi<'_>) {
        let ctx = plot_ui.ctx();
        let step = if ctx.input(|i| i.modifiers.shift) {
            0.1
        } else {
            0.5
        };
        let nudge = ctx.input_mut(|i| {
            let m = if i.modifiers.shift {
                egui::Modifiers::SHIFT
            } else {
                egui::Modifiers::NONE
            };
            if i.consume_key(m, egui::Key::ArrowLeft) {
                Some((-step, 0.0))
            } else if i.consume_key(m, egui::Key::ArrowRight) {
                Some((step, 0.0))
            } else if i.consume_key(m, egui::Key::ArrowUp) {
                Some((0.0, step))
            } else if i.consume_key(m, egui::Key::ArrowDown) {
                Some((0.0, -step))
            } else {
                None
            }
        });
        if let Some((dx, dy)) = nudge {
            ctx.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            let msg = self.nudge_focus_target(dx, dy);
            announce_live_status(ctx, msg);
        }
        self.handle_canvas_action_keys(ctx);
    }

    fn handle_canvas_action_keys(&mut self, ctx: &egui::Context) {
        let (cycle_next, cycle_prev, stamp, del, tune) = ctx.input_mut(|i| {
            let next = i.consume_key(egui::Modifiers::NONE, egui::Key::CloseBracket);
            let prev = i.consume_key(egui::Modifiers::NONE, egui::Key::OpenBracket);
            let stamp = i.consume_key(egui::Modifiers::NONE, egui::Key::Enter);
            let del = i.consume_key(egui::Modifiers::NONE, egui::Key::Delete)
                || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace);
            let tune = if i.consume_key(egui::Modifiers::NONE, egui::Key::Equals)
                || i.consume_key(egui::Modifiers::SHIFT, egui::Key::Plus)
            {
                2.5
            } else if i.consume_key(egui::Modifiers::NONE, egui::Key::Minus) {
                -2.5
            } else {
                0.0
            };
            (next, prev, stamp, del, tune)
        });
        if cycle_next || cycle_prev {
            announce_live_status(ctx, self.cycle_focus_target(cycle_next));
        } else if stamp {
            if let Some(idx) = self.stamp_element_at(self.reticle[0], self.reticle[1]) {
                announce_live_status(
                    ctx,
                    format!(
                        "Stamped {} #{} at ({:.1}, {:.1})",
                        self.selected_element_type.label(),
                        idx + 1,
                        self.reticle[0],
                        self.reticle[1]
                    ),
                );
            }
        } else if del {
            if let Some((idx, removed)) = self
                .focused_or_selected_element()
                .and_then(|idx| self.delete_element(idx).map(|r| (idx, r)))
            {
                announce_live_status(
                    ctx,
                    format!("Deleted {} #{}", removed.element_type.label(), idx + 1),
                );
            }
        } else if tune != 0.0 {
            announce_live_status(ctx, self.tune_focused_strength(tune));
        }
    }

    fn focused_or_selected_element(&self) -> Option<usize> {
        match self.focus_target {
            FocusTarget::Element(idx) => Some(idx),
            _ => self.selected_element,
        }
    }

    pub(super) fn cycle_focus_target(&mut self, forward: bool) -> String {
        let mut order = vec![FocusTarget::Reticle];
        for (idx, el) in self.elements.iter().enumerate() {
            if el.element_type != ElementType::Uniform {
                order.push(FocusTarget::Element(idx));
            }
        }
        order.push(FocusTarget::DyeRake);
        let cur = order
            .iter()
            .position(|&t| t == self.focus_target)
            .unwrap_or(0);
        let next_idx = if forward {
            (cur + 1) % order.len()
        } else {
            (cur + order.len() - 1) % order.len()
        };
        self.focus_target = order[next_idx];
        if let FocusTarget::Element(idx) = self.focus_target {
            self.selected_element = Some(idx);
        }
        self.canvas_accessibility_summary()
    }

    pub(super) fn nudge_focus_target(&mut self, dx: f64, dy: f64) -> String {
        match self.focus_target {
            FocusTarget::Reticle => {
                self.reticle[0] = (self.reticle[0] + dx).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
                self.reticle[1] = (self.reticle[1] + dy).clamp(FLOW_POS_MIN, FLOW_POS_MAX);
                format!(
                    "Reticle at ({:.1}, {:.1})",
                    self.reticle[0], self.reticle[1]
                )
            }
            FocusTarget::Element(idx) => {
                if let Some(el) = self.elements.get(idx) {
                    let (nx, ny) = (el.x + dx, el.y + dy);
                    let kind = el.element_type.label();
                    self.apply_drag_target(DragTarget::Element(idx), nx, ny);
                    let updated = &self.elements[idx];
                    format!(
                        "{kind} #{} moved to ({:.1}, {:.1})",
                        idx + 1,
                        updated.x,
                        updated.y
                    )
                } else {
                    "Reticle ready".to_string()
                }
            }
            FocusTarget::DyeRake => {
                self.apply_drag_target(
                    DragTarget::DyeRake,
                    self.dye_rake_x + dx,
                    self.dye_rake_y + dy,
                );
                format!(
                    "Dye rake moved to ({:.1}, {:.1})",
                    self.dye_rake_x, self.dye_rake_y
                )
            }
        }
    }

    fn tune_focused_strength(&mut self, delta: f64) -> String {
        if let Some(idx) = self.focused_or_selected_element() {
            if let Some(el) = self.elements.get_mut(idx) {
                el.strength = (el.strength + delta).clamp(FLOW_STRENGTH_MIN, FLOW_STRENGTH_MAX);
                let s = el.strength;
                let label = el.element_type.label();
                self.rebuild_field();
                return format!("{label} #{} strength {:.1}", idx + 1, s);
            }
        }
        self.new_element_strength =
            (self.new_element_strength + delta).clamp(FLOW_STRENGTH_MIN, FLOW_STRENGTH_MAX);
        format!("Stamp strength {:.1}", self.new_element_strength)
    }

    fn draw_plot_layers(&self, plot_ui: &mut PlotUi<'_>, canvas_focused: bool) {
        let palette = SemanticPalette::from_visuals(&plot_ui.ctx().style().visuals);
        let r = self.view_range.max(0.5);
        let bounds = (-r, r, -r, r);

        if self.show_streamlines {
            let levels = self.compute_iso_levels();
            let segs =
                self.field
                    .extract_streamline_segments(bounds, self.contour_resolution, &levels);
            let contour_color = palette.series_color(2).gamma_multiply(0.75);
            for (i, chain) in super::presets::chain_segments(&segs)
                .into_iter()
                .enumerate()
            {
                let name = if i == 0 {
                    "Iso-ψ Streamlines".to_string()
                } else {
                    String::new()
                };
                plot_ui.line(Line::new(name, chain).color(contour_color).width(1.4_f32));
            }
        }
        if self.show_vectors {
            let (origins, tips) = self.compute_arrow_field();
            plot_ui
                .arrows(Arrows::new("Vector Field", origins, tips).color(palette.series_color(0)));
        }
        if self.show_dye {
            self.draw_dye_streaks(plot_ui, &palette);
        }
        let stag: Vec<[f64; 2]> = self
            .field
            .detect_stagnation_points(bounds, 40)
            .into_iter()
            .map(|p| [p.x, p.y])
            .collect();
        if !stag.is_empty() {
            plot_ui.points(
                Points::new("Stagnation Points", stag)
                    .shape(MarkerShape::Diamond)
                    .radius(5.5_f32)
                    .color(palette.warning),
            );
        }
        self.draw_handle_markers(plot_ui, &palette, canvas_focused);
    }
}
