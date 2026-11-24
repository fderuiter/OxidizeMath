//! SpatialBrushKit controls for [`super::potential_flow::PotentialFlowTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::potential_flow::PotentialFlowTool;
use crate::framework::{BrushInteractionMode, BrushMode, SpatialBrushKit};
use eframe::egui;

impl PotentialFlowTool {
    pub(super) fn default_flow_brush_kit() -> SpatialBrushKit {
        SpatialBrushKit::new(BrushMode::FlowSource, 1.5)
            .with_interaction_mode(BrushInteractionMode::Sculpt)
            .with_radius_bounds(0.5, 6.0, 0.5)
            .with_cursor_step(1.0, 1.0)
    }

    pub(super) fn draw_brush_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Canvas Sculpting");
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.brush_kit.interaction_mode,
                BrushInteractionMode::Sculpt,
                "🖌 Sculpt",
            );
            ui.selectable_value(
                &mut self.brush_kit.interaction_mode,
                BrushInteractionMode::Pan,
                "✋ Pan",
            );
        });
        if self.brush_kit.is_sculpting() {
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.brush_kit.mode, BrushMode::FlowSource, "Source");
                ui.selectable_value(&mut self.brush_kit.mode, BrushMode::FlowSink, "Sink");
                ui.selectable_value(&mut self.brush_kit.mode, BrushMode::FlowVortex, "Vortex");
                ui.selectable_value(&mut self.brush_kit.mode, BrushMode::FlowDoublet, "Doublet");
                ui.selectable_value(&mut self.brush_kit.mode, BrushMode::Erase, "Erase");
            });
            ui.add(egui::Slider::new(&mut self.brush_kit.radius, 0.5..=6.0).text("Capture Radius"));
        }
    }
}
