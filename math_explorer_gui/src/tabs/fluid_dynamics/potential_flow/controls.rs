//! Sidebar controls for presets, flow visualization layers, dye transport,
//! active singularity list, and bounded `DragValue` element editors.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets;
use super::{
    ElementType, PotentialFlowTool, DYE_SPEED_MAX, DYE_SPEED_MIN, FLOW_ANGLE_MAX, FLOW_ANGLE_MIN,
    FLOW_POS_MAX, FLOW_POS_MIN, FLOW_STRENGTH_MAX, FLOW_STRENGTH_MIN, FLOW_VIEW_MAX, FLOW_VIEW_MIN,
};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::framework::{
    bounded_drag_value, scoped_panel_id, PaneSidePanel, PlaybackAction, PlaybackTransport,
};
use eframe::egui;

type QuickPresetEntry = (&'static str, fn() -> Vec<super::SerializedFlowElement>, f64);

impl PotentialFlowTool {
    pub(crate) fn draw_sidebar_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "potential_flow_controls");
        let ctx = parent_ui.ctx().clone();
        let (u_inf, gamma, lift, drag) = self.aerodynamic_summary();
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Potential Flow Controls");
                ui.small(format!(
                    "U∞={u_inf:.1} | Γ={gamma:.1} | L'={lift:.1} | D'={drag:.1}"
                ))
                .accessible_hover_text(
                    "Freestream speed, total circulation, Kutta-Joukowski lift per unit span, and Lagally source-sink drag",
                );
                self.draw_preset_buttons(&ctx, ui);
                ui.separator();
                self.draw_layer_and_dye_controls(&ctx, ui);
                ui.separator();
                self.draw_grid_and_view_sliders(ui);
                self.draw_brush_toolbar(ui);
                ui.separator();
                self.draw_add_element_section(ui);
                ui.separator();
                self.draw_active_elements_section(&ctx, ui);
            });
        });
    }

    fn draw_preset_buttons(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.label("Wind-Tunnel Presets");
        ui.horizontal_wrapped(|ui| {
            let presets_list: [QuickPresetEntry; 4] = [
                (
                    "Symmetric Cylinder",
                    presets::symmetric_cylinder_elements,
                    10.0,
                ),
                (
                    "Kutta-Joukowski Lift",
                    presets::lifting_cylinder_elements,
                    10.0,
                ),
                ("Rankine Oval", presets::rankine_oval_elements, 10.0),
                ("Vortex Street", presets::vortex_street_elements, 12.0),
            ];
            for (label, builder, range) in presets_list {
                if ui
                    .button(label)
                    .accessible_hover_text(format!("Load {label} potential flow preset"))
                    .clicked()
                {
                    self.elements = builder();
                    self.view_range = range;
                    self.selected_element = None;
                    self.rebuild_field();
                    self.reseed_dye_emitters();
                    announce_live_status(ctx, format!("Loaded {label} preset"));
                }
            }
        });
    }

    fn draw_layer_and_dye_controls(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.label("Visualization Layers");
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.show_streamlines, "Show Streamlines (ψ)");
            ui.checkbox(&mut self.show_stagnation_points, "Show Stagnation Points");
            ui.checkbox(&mut self.show_arrows, "Velocity Arrows");
            ui.checkbox(&mut self.show_dye_streaks, "RK4 Dye Streaks");
            ui.checkbox(&mut self.show_probe, "Euler Probe");
        });
        ui.checkbox(&mut self.canvas_stamp_mode, "Click Canvas to Stamp Element")
            .accessible_hover_text(
                "Click empty canvas to stamp the selected singularity; drag handles to reposition",
            );

        ui.add_space(4.0);
        let canvas_sculpt_focused = self.brush_kit.focused_plot_id().is_some();
        let action = PlaybackTransport::in_thread(
            "potential_flow_dye_transport",
            "Potential Flow",
            self.dye_playing,
        )
        .reset_label("↺ Reseed")
        .allow_space_shortcut(!canvas_sculpt_focused)
        .show(ui);
        match action {
            Some(PlaybackAction::Play) => self.dye_playing = true,
            Some(PlaybackAction::Pause) => self.dye_playing = false,
            Some(PlaybackAction::Step) => {
                self.dye_playing = false;
                self.step_dye_tracers(0.03 * self.dye_speed);
            }
            Some(PlaybackAction::Reset) => {
                self.reseed_dye_emitters();
                announce_live_status(ctx, "Reseeded dye tracer emitters");
            }
            Some(PlaybackAction::StepBack | PlaybackAction::ScrubTo(_)) | None => {}
        }
        ui.add(
            egui::Slider::new(&mut self.dye_speed, DYE_SPEED_MIN..=DYE_SPEED_MAX).text("Dye Speed"),
        );
    }

    fn draw_grid_and_view_sliders(&mut self, ui: &mut egui::Ui) {
        let max_grid = crate::presets::MAX_INTERACTIVE_GRID_SIZE;
        let clamped_grid = self.grid_size.clamp(2, max_grid);
        if clamped_grid != self.grid_size {
            self.grid_size = clamped_grid;
            self.grid_revision = self.grid_revision.wrapping_add(1);
        }
        let g_changed = ui
            .add(egui::Slider::new(&mut self.grid_size, 5..=max_grid).text("Arrow Grid"))
            .changed();
        let v_changed = ui
            .add(
                egui::Slider::new(&mut self.view_range, FLOW_VIEW_MIN..=FLOW_VIEW_MAX)
                    .text("View Range"),
            )
            .changed();
        if g_changed || v_changed {
            self.grid_revision = self.grid_revision.wrapping_add(1);
            self.cached_field_key = 0;
            self.ensure_contour_cache();
            if v_changed {
                self.reseed_dye_emitters();
            }
        }
    }

    fn draw_add_element_section(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Add Element", |ui| {
            for (etype, label) in [
                (ElementType::Uniform, "Uniform Flow"),
                (ElementType::Source, "Source (+Q)"),
                (ElementType::Sink, "Sink (-Q)"),
                (ElementType::Vortex, "Vortex (Γ)"),
                (ElementType::Doublet, "Doublet (κ)"),
            ] {
                ui.radio_value(&mut self.selected_element_type, etype, label);
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

    fn draw_active_elements_section(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong(format!("Active Elements ({})", self.elements.len()));
            if ui
                .button("🔄 Clear All")
                .accessible_hover_text("Remove all flow elements")
                .clicked()
            {
                self.clear();
            }
        });
        let mut remove_idx: Option<usize> = None;
        let mut changed = false;
        for idx in 0..self.elements.len() {
            let is_sel = self.selected_element == Some(idx);
            let label = Self::format_element_row_label(idx, &self.elements[idx]);
            ui.horizontal(|ui| {
                if ui.selectable_label(is_sel, label).clicked() {
                    self.selected_element = Some(idx);
                    self.focused_reticle = [self.elements[idx].x, self.elements[idx].y];
                }
                if ui
                    .small_button("✖")
                    .accessible_hover_text("Remove element")
                    .clicked()
                {
                    remove_idx = Some(idx);
                }
            });
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.elements[idx].strength, -500.0..=500.0)
                        .text("Strength"),
                )
                .changed();
        }
        if let Some(idx) = remove_idx {
            let removed = self.elements.remove(idx);
            self.selected_element = None;
            self.rebuild_field();
            self.reseed_dye_emitters();
            announce_live_status(ctx, format!("Removed {}", removed.element_type.label()));
        } else if changed {
            self.rebuild_field();
            self.reseed_dye_emitters();
        }
    }

    fn format_element_row_label(idx: usize, el: &super::SerializedFlowElement) -> String {
        if el.has_canvas_handle() {
            format!(
                "#{} {} ({:+.1}, {:+.1})",
                idx + 1,
                el.element_type.label(),
                el.x,
                el.y
            )
        } else {
            format!(
                "#{} {} ({:.0}°)",
                idx + 1,
                el.element_type.label(),
                el.angle
            )
        }
    }
}
