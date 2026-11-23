//! SpatialBrushKit controls and `InteractiveTool::on_brush` handler for [`super::potential_flow::PotentialFlowTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::potential_flow::{
    ElementType, PotentialFlowTool, SerializedFlowElement, FLOW_POS_MAX, FLOW_POS_MIN,
    FLOW_STRENGTH_MAX, FLOW_STRENGTH_MIN,
};
use crate::framework::{BrushInteractionMode, BrushMode, BrushStroke, SpatialBrushKit};
use eframe::egui;

impl PotentialFlowTool {
    pub(super) fn default_flow_brush_kit() -> SpatialBrushKit {
        SpatialBrushKit::new(BrushMode::FlowSource, 1.5)
            .with_interaction_mode(BrushInteractionMode::Pan)
            .with_radius_bounds(0.5, 6.0, 0.5)
            .with_cursor_step(1.0, 1.0)
    }

    pub(super) fn draw_brush_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Canvas Sculpting");
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.brush_kit.interaction_mode,
                BrushInteractionMode::Pan,
                "✋ Pan",
            );
            ui.selectable_value(
                &mut self.brush_kit.interaction_mode,
                BrushInteractionMode::Sculpt,
                "🖌 Sculpt",
            );
        });
        if self.brush_kit.is_sculpting() {
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.brush_kit.mode, BrushMode::FlowSource, "Source");
                ui.selectable_value(&mut self.brush_kit.mode, BrushMode::FlowVortex, "Vortex");
                ui.selectable_value(&mut self.brush_kit.mode, BrushMode::FlowDoublet, "Doublet");
                ui.selectable_value(&mut self.brush_kit.mode, BrushMode::Erase, "Erase");
            });
            ui.add(
                egui::Slider::new(&mut self.brush_kit.radius, 0.5..=6.0).text("Capture Radius"),
            );
        }
    }

    pub(crate) fn apply_flow_brush_stroke(&mut self, stroke: &BrushStroke) {
        let px = stroke.end[0].clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        let py = stroke.end[1].clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        let r_sq = stroke.radius.max(0.5).powi(2);

        if stroke.mode == BrushMode::Erase {
            let before = self.elements.len();
            self.elements.retain(|el| {
                el.element_type == ElementType::Uniform
                    || (el.x - px).powi(2) + (el.y - py).powi(2) > r_sq
            });
            if self.elements.len() != before {
                self.rebuild_field();
            }
            return;
        }

        let target_type = match stroke.mode {
            BrushMode::FlowVortex => ElementType::Vortex,
            BrushMode::FlowDoublet => ElementType::Doublet,
            _ => {
                if self.new_element_strength < 0.0 {
                    ElementType::Sink
                } else {
                    ElementType::Source
                }
            }
        };
        let strength = self
            .new_element_strength
            .abs()
            .max(1.0)
            .clamp(FLOW_STRENGTH_MIN, FLOW_STRENGTH_MAX);

        if let Some(existing) = self.elements.iter_mut().rev().find(|el| {
            el.element_type == target_type && (el.x - stroke.start[0]).powi(2) + (el.y - stroke.start[1]).powi(2) <= r_sq
        }) {
            existing.x = px;
            existing.y = py;
            existing.strength = strength;
        } else if self.elements.len() < crate::framework::MAX_JSON_ARRAY_LEN {
            self.elements.push(SerializedFlowElement {
                element_type: target_type,
                strength,
                x: px,
                y: py,
                angle: 0.0,
            });
        }
        self.new_element_x = px;
        self.new_element_y = py;
        self.rebuild_field();
    }
}
