#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    ElementType, FocusTarget, PotentialFlowTool, FLOW_ANGLE_MAX, FLOW_ANGLE_MIN, FLOW_POS_MAX,
    FLOW_POS_MIN, FLOW_STRENGTH_MAX, FLOW_STRENGTH_MIN, FLOW_VIEW_MAX, MAX_CONTOUR_LEVELS,
    MAX_DYE_PROBES,
};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::framework::{bounded_drag_value, InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;

impl PotentialFlowTool {
    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "potential_flow_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Potential Flow Controls");
                ui.separator();
                self.draw_playback_and_stamp_bar(ui);
                ui.separator();
                self.draw_preset_buttons(ui);
                ui.separator();
                self.draw_element_inspector(ui);
                ui.separator();
                self.draw_add_element_controls(ui);
                ui.separator();
                self.draw_layer_and_grid_controls(ui);
                ui.separator();
                self.draw_probe_readout(ui);
            });
        });
    }

    fn draw_playback_and_stamp_bar(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = PlaybackTransport::in_thread(
            "potential_flow_transport",
            "Potential Flow",
            self.is_playing,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.is_playing = true,
                PlaybackAction::Pause => self.is_playing = false,
                PlaybackAction::Step => self.advance_dye_step(),
                PlaybackAction::Reset => {
                    self.is_playing = false;
                    self.reseed_dye_streaks();
                }
            }
        }
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.stamp_mode, "🎯 Click-to-Stamp")
                .accessible_hover_text(
                    "Click on the canvas to stamp the selected singularity without panning",
                );
            if ui
                .button("🔄 Clear All")
                .accessible_hover_text("Remove all potential flow elements and reset the field")
                .clicked()
            {
                self.clear();
            }
        });
        ui.small("Canvas keys: [ / ] cycle handles, Arrows nudge, Enter stamps, - / = strength, Delete removes.");
    }

    fn draw_preset_buttons(&mut self, ui: &mut egui::Ui) {
        ui.label("Aerodynamic Wind-Tunnel Presets");
        ui.horizontal_wrapped(|ui| {
            for (label, json) in [
                (
                    "✈ Lifting Cylinder",
                    crate::starter_gallery::PRESET_LIFTING_CYLINDER,
                ),
                (
                    "🥚 Rankine Oval",
                    crate::starter_gallery::PRESET_RANKINE_OVAL,
                ),
                (
                    "🌀 Vortex Street",
                    crate::starter_gallery::PRESET_VORTEX_STREET,
                ),
            ] {
                if ui
                    .button(label)
                    .accessible_hover_text(format!("Load {label} potential flow preset"))
                    .clicked()
                    && self.import_preset_json(json).is_ok()
                {
                    announce_live_status(ui.ctx(), format!("Loaded {label}"));
                }
            }
        });
    }

    fn element_row_label(&self, idx: usize) -> String {
        let el = &self.elements[idx];
        if el.element_type == ElementType::Uniform {
            format!(
                "#{} {} (U={:.1}, α={:.0}°)",
                idx + 1,
                el.element_type.label(),
                el.strength,
                el.angle
            )
        } else {
            format!(
                "#{} {} ({:.1}, {:.1})",
                idx + 1,
                el.element_type.label(),
                el.x,
                el.y
            )
        }
    }

    fn draw_element_inspector(&mut self, ui: &mut egui::Ui) {
        ui.label(format!("Active Singularities ({})", self.elements.len()));
        let mut to_delete = None;
        let mut needs_rebuild = false;
        for idx in 0..self.elements.len() {
            let selected = self.selected_element == Some(idx);
            let summary = self.element_row_label(idx);
            ui.horizontal(|ui| {
                if ui.selectable_label(selected, summary).clicked() {
                    self.selected_element = Some(idx);
                    self.focus_target = FocusTarget::Element(idx);
                }
                if ui
                    .small_button("🗑")
                    .accessible_hover_text(format!("Delete element #{}", idx + 1))
                    .clicked()
                {
                    to_delete = Some(idx);
                }
            });
        }
        if let Some(idx) = to_delete {
            self.delete_element(idx);
            return;
        }
        if let Some(el) = self
            .selected_element
            .and_then(|idx| self.elements.get_mut(idx))
        {
            let kind = el.element_type;
            needs_rebuild |= ui
                .add(
                    egui::Slider::new(&mut el.strength, FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX)
                        .text("Selected Strength"),
                )
                .changed();
            if kind == ElementType::Uniform {
                needs_rebuild |= ui
                    .add(
                        egui::Slider::new(&mut el.angle, FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX)
                            .text("Selected Angle (°)"),
                    )
                    .changed();
            }
        }
        if needs_rebuild {
            self.rebuild_field();
        }
    }

    fn draw_add_element_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Add Element", |ui| {
            for (elem, label) in [
                (ElementType::Uniform, "Uniform Flow"),
                (ElementType::Source, "Source"),
                (ElementType::Sink, "Sink"),
                (ElementType::Vortex, "Vortex"),
                (ElementType::Doublet, "Doublet"),
            ] {
                ui.radio_value(&mut self.selected_element_type, elem, label);
            }

            ui.separator();
            ui.label("Strength / Velocity");
            ui.add(bounded_drag_value(
                &mut self.new_element_strength,
                FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX,
                0.1,
            ));

            if self.selected_element_type == ElementType::Uniform {
                ui.add(
                    egui::Slider::new(&mut self.new_element_angle, FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX)
                        .text("Angle (degrees)"),
                );
            } else {
                ui.label("Position X");
                ui.add(bounded_drag_value(
                    &mut self.new_element_x,
                    FLOW_POS_MIN..=FLOW_POS_MAX,
                    0.1,
                ));
                ui.label("Position Y");
                ui.add(bounded_drag_value(
                    &mut self.new_element_y,
                    FLOW_POS_MIN..=FLOW_POS_MAX,
                    0.1,
                ));
            }

            if ui
                .button("➕ Add Element")
                .accessible_hover_text("Add the selected flow element to the field")
                .clicked()
            {
                self.add_element();
            }
        });
    }

    fn draw_layer_and_grid_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Layers & Resolution");
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.show_streamlines, "Iso-ψ Streamlines");
            ui.checkbox(&mut self.show_vectors, "Vector Field");
            ui.checkbox(&mut self.show_dye, "RK4 Dye Streaks");
        });
        ui.add(
            egui::Slider::new(&mut self.contour_levels, 3..=MAX_CONTOUR_LEVELS)
                .text("Iso-ψ Levels"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.grid_size,
                10..=math_commons::registry::MAX_GRID_SIZE,
            )
            .text("Visualization - Grid Density"),
        );
        let range_changed = ui
            .add(egui::Slider::new(&mut self.view_range, 5.0..=FLOW_VIEW_MAX).text("View Range"))
            .changed();
        let count_changed = ui
            .add(egui::Slider::new(&mut self.dye_count, 1..=MAX_DYE_PROBES).text("Dye Probes"))
            .changed();
        let max_probe = self.dye_count.saturating_sub(1);
        if max_probe > 0 {
            ui.add(
                egui::Slider::new(&mut self.selected_probe, 0..=max_probe).text("Active Probe #"),
            );
        }
        if range_changed || count_changed {
            self.reseed_dye_streaks();
        }
    }

    fn draw_probe_readout(&self, ui: &mut egui::Ui) {
        ui.label("RK4 Dye Probe & Euler Telemetry");
        if let Some(readout) = self.active_probe_readout() {
            ui.monospace(format!(
                "Pos: ({:+.2}, {:+.2}) | |V|: {:.2} m/s",
                readout.position.x,
                readout.position.y,
                readout.velocity.norm()
            ));
            ui.monospace(format!(
                "Conv. Accel (u·∇)u: ({:+.2}, {:+.2}) m/s²",
                readout.convective_acceleration.x, readout.convective_acceleration.y
            ));
            ui.monospace(format!(
                "Bernoulli p: {:.1} Pa | Cp: {:+.3}",
                readout.bernoulli_pressure, readout.pressure_coefficient
            ));
        }
    }

    pub(super) fn draw_dye_streaks(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        use egui_plot::{Line, MarkerShape, Points};
        let streak_color = palette.series_color(1);
        let mut heads = Vec::with_capacity(self.dye_probes.len());
        for (i, probe) in self.dye_probes.iter().enumerate() {
            if probe.trail.len() >= 2 {
                let is_active = i == self.selected_probe;
                let width = if is_active { 2.6_f32 } else { 1.6_f32 };
                let col = if is_active {
                    palette.accent
                } else {
                    streak_color
                };
                plot_ui.line(
                    Line::new(format!("Dye Streak {}", i + 1), probe.trail.clone())
                        .color(col)
                        .width(width),
                );
            }
            heads.push(probe.head);
        }
        if !heads.is_empty() {
            plot_ui.points(
                Points::new("Dye Probes", heads)
                    .shape(MarkerShape::Circle)
                    .radius(3.5_f32)
                    .color(palette.accent),
            );
        }
    }

    pub(super) fn draw_handle_markers(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
        canvas_focused: bool,
    ) {
        use egui_plot::{MarkerShape, Points};
        for (kind, name, shape, color) in [
            (
                ElementType::Source,
                "Sources",
                MarkerShape::Circle,
                palette.success,
            ),
            (
                ElementType::Sink,
                "Sinks",
                MarkerShape::Square,
                palette.error,
            ),
            (
                ElementType::Vortex,
                "Vortices",
                MarkerShape::Asterisk,
                palette.series_color(3),
            ),
            (
                ElementType::Doublet,
                "Doublets",
                MarkerShape::Up,
                palette.series_color(4),
            ),
        ] {
            let pts: Vec<[f64; 2]> = self
                .elements
                .iter()
                .filter(|e| e.element_type == kind)
                .map(|e| [e.x, e.y])
                .collect();
            if !pts.is_empty() {
                plot_ui.points(
                    Points::new(name, pts)
                        .shape(shape)
                        .radius(6.5_f32)
                        .color(color),
                );
            }
        }
        plot_ui.points(
            Points::new("Dye Rake", vec![[self.dye_rake_x, self.dye_rake_y]])
                .shape(MarkerShape::Right)
                .radius(6.5_f32)
                .color(palette.accent),
        );
        if canvas_focused || self.stamp_mode {
            plot_ui.points(
                Points::new("Reticle", vec![self.reticle])
                    .shape(MarkerShape::Cross)
                    .radius(7.5_f32)
                    .color(palette.accent),
            );
        }
    }

    pub(super) fn canvas_accessibility_summary(&self) -> String {
        match self.focus_target {
            FocusTarget::Reticle => format!(
                "Reticle ({:.1}, {:.1}), {} elements, stamp {}",
                self.reticle[0],
                self.reticle[1],
                self.elements.len(),
                self.selected_element_type.label()
            ),
            FocusTarget::Element(idx) => self.elements.get(idx).map_or_else(
                || "Reticle active".to_string(),
                |e| {
                    format!(
                        "{} #{} at ({:.1}, {:.1}), strength {:.1}",
                        e.element_type.label(),
                        idx + 1,
                        e.x,
                        e.y,
                        e.strength
                    )
                },
            ),
            FocusTarget::DyeRake => format!(
                "Dye rake at ({:.1}, {:.1})",
                self.dye_rake_x, self.dye_rake_y
            ),
        }
    }
}
