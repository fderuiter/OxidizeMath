#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, active superposition element list, and accessible individual deletion
//! for [`PotentialFlowTool`].

use super::{
    ElementType, PotentialFlowTool, SerializedFlowElement, FLOW_ANGLE_MAX, FLOW_ANGLE_MIN,
    FLOW_POS_MAX, FLOW_POS_MIN, FLOW_STRENGTH_MAX, FLOW_STRENGTH_MIN, FLOW_VIEW_MAX,
};
use crate::accessibility::{announce_status, AccessibleHoverText};
use crate::framework::bounded_drag_value;
use eframe::egui;

impl SerializedFlowElement {
    /// Human-readable summary of element type and parameters for list rows and screen readers.
    #[must_use]
    pub(crate) fn summary(&self) -> String {
        match self.element_type {
            ElementType::Uniform => {
                format!(
                    "Uniform Flow (U = {:.2}, α = {:.1}°)",
                    self.strength, self.angle
                )
            }
            ElementType::Source => {
                format!(
                    "Source (m = {:.2}, x = {:.2}, y = {:.2})",
                    self.strength, self.x, self.y
                )
            }
            ElementType::Sink => {
                format!(
                    "Sink (m = {:.2}, x = {:.2}, y = {:.2})",
                    self.strength, self.x, self.y
                )
            }
            ElementType::Vortex => {
                format!(
                    "Vortex (Γ = {:.2}, x = {:.2}, y = {:.2})",
                    self.strength, self.x, self.y
                )
            }
            ElementType::Doublet => {
                format!(
                    "Doublet (κ = {:.2}, x = {:.2}, y = {:.2})",
                    self.strength, self.x, self.y
                )
            }
        }
    }
}

impl PotentialFlowTool {
    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "potential_flow_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Potential Flow Controls");
            ui.separator();
            self.draw_add_element_controls(ui);

            ui.separator();
            self.draw_active_elements_list(ui);
            self.draw_active_element_sliders(ui);

            ui.separator();
            if ui
                .button("🔄 Clear All")
                .accessible_hover_text("Remove all potential flow elements and reset the field")
                .clicked()
            {
                self.clear();
            }

            ui.separator();
            ui.checkbox(&mut self.show_streamlines, "Show Streamlines (ψ)");
            ui.checkbox(&mut self.show_stagnation_points, "Show Stagnation Points");
            let max_grid = crate::presets::MAX_INTERACTIVE_GRID_SIZE;
            self.grid_size = self.grid_size.clamp(2, max_grid);
            ui.add(
                egui::Slider::new(&mut self.grid_size, 10..=max_grid)
                    .text("Visualization - Grid Density"),
            );
            ui.add(egui::Slider::new(&mut self.view_range, 5.0..=FLOW_VIEW_MAX).text("View Range"));
            let (u_inf, gamma, lift, drag) = self.aerodynamic_summary();
            ui.separator();
            ui.add(
                egui::Label::new(
                    egui::RichText::new(format!(
                        "U∞={u_inf:.1} | Γ={gamma:.1} | L'={lift:.1} | D'={drag:.1}"
                    ))
                    .small(),
                )
                .wrap(),
            );
        });
    }

    fn draw_active_element_sliders(&mut self, ui: &mut egui::Ui) {
        if self.elements.is_empty() {
            return;
        }
        let has_vortex = self
            .elements
            .iter()
            .any(|el| el.element_type == ElementType::Vortex);
        let mut modified = false;
        egui::CollapsingHeader::new("Tune Element Strengths")
            .default_open(has_vortex)
            .show(ui, |ui| {
                for (idx, el) in self.elements.iter_mut().enumerate() {
                    let label = format!("#{} {}", idx + 1, el.element_type.label());
                    modified |= ui
                        .add(egui::Slider::new(&mut el.strength, -500.0..=500.0).text(label))
                        .changed();
                }
            });
        if modified {
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

    fn draw_active_elements_list(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new(format!("Active Elements ({})", self.elements.len())).strong());
        if self.elements.is_empty() {
            ui.weak("No active flow elements.");
            return;
        }
        let mut remove_idx: Option<usize> = None;
        egui::ScrollArea::vertical()
            .id_salt("potential_flow_active_elements")
            .max_height(280.0)
            .show(ui, |ui| {
                for (idx, elem) in self.elements.iter().enumerate() {
                    let summary = elem.summary();
                    let a11y_label = format!("Remove element {}: {}", idx + 1, summary);
                    ui.push_id(("flow_elem_row", idx), |ui| {
                        ui.horizontal(|ui| {
                            let btn_resp = ui.add(egui::Button::new("✕ Remove").small());
                            let enabled = ui.is_enabled();
                            btn_resp.widget_info({
                                let label = a11y_label.clone();
                                move || {
                                    egui::WidgetInfo::labeled(
                                        egui::WidgetType::Button,
                                        enabled,
                                        label.clone(),
                                    )
                                }
                            });
                            ui.ctx().accesskit_node_builder(btn_resp.id, |builder| {
                                builder.set_role(egui::accesskit::Role::Button);
                                builder.set_label(a11y_label.clone());
                            });
                            if btn_resp.accessible_hover_text(a11y_label).clicked() {
                                remove_idx = Some(idx);
                            }
                            ui.add(egui::Label::new(format!("{}. {}", idx + 1, summary)).wrap());
                        });
                    });
                }
            });
        if let Some(idx) = remove_idx {
            if let Some(removed) = self.remove_element(idx) {
                announce_status(&format!("Removed {}", removed.summary()));
            }
        }
    }
}
