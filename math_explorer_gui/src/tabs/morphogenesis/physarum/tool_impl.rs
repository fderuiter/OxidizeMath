//! [`FiniteStateGuard`], [`InteractiveTool`], and [`crate::framework::ToolMetadata`]
//! implementations for [`super::PhysarumTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{MAX_SAVED_OBSTACLE_STROKES, default_brush_radius};
use super::{PhysarumPreset, PhysarumTool, PhysarumToolState};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use eframe::egui::ColorImage;
use math_explorer::biology::physarum::{
    MAX_PHYSARUM_AGENTS, MAX_PHYSARUM_GRID, MAX_PHYSARUM_TERMINALS, MIN_PHYSARUM_AGENTS,
    MIN_PHYSARUM_GRID, MIN_PHYSARUM_TERMINALS,
};
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};

impl FiniteStateGuard for PhysarumTool {
    fn is_finite_state(&self) -> bool {
        self.diagnostic.is_none()
            && self.parameters_within_safe_bounds()
            && self.engine.trail().iter().all(|v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic.as_ref()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let next_speed = if self.step_size.is_finite() && self.step_size > 0.0 {
            self.step_size * 0.5
        } else {
            1.0
        };
        self.step_size = next_speed.clamp(0.20, 1.75);
        FiniteStateGuard::clamp_parameters(self);
    }

    fn clamp_parameters(&mut self) {
        let clean = self.current_params().sanitized();
        self.sensor_angle = clean.sensor_angle;
        self.turn_angle = clean.turn_angle;
        self.sensor_distance = clean.sensor_distance;
        self.step_size = clean.step_size;
        self.deposition = clean.deposition;
        self.diffusion_rate = clean.diffusion_rate;
        self.decay_rate = clean.decay_rate;
        self.prune_rate = clean.prune_rate;
        self.beacon_strength = clean.beacon_strength;
        self.grid_size = self.grid_size.clamp(MIN_PHYSARUM_GRID, MAX_PHYSARUM_GRID);
        self.agent_count = self
            .agent_count
            .clamp(MIN_PHYSARUM_AGENTS, MAX_PHYSARUM_AGENTS);
        self.brush_radius = if self.brush_radius.is_finite() {
            self.brush_radius.clamp(1.0, 24.0)
        } else {
            4.0
        };
        self.sanitize_trail_and_collections();
        self.diagnostic = None;
        self.sync_engine_parameters();
        self.refresh_pixels_in_place();
    }

    fn reset_parameters(&mut self) {
        self.playing = false;
        self.brush_radius = default_brush_radius();
        self.apply_preset(PhysarumPreset::SteinerTriangle);
    }
}

impl PhysarumTool {
    fn sanitize_trail_and_collections(&mut self) {
        if self.terminals.len() < MIN_PHYSARUM_TERMINALS {
            self.terminals = self.preset.to_domain().normalized_terminals();
        }
        self.terminals.truncate(MAX_PHYSARUM_TERMINALS);
        for pair in &mut self.terminals {
            pair[0] = if pair[0].is_finite() {
                pair[0].clamp(0.02, 0.98)
            } else {
                0.5
            };
            pair[1] = if pair[1].is_finite() {
                pair[1].clamp(0.02, 0.98)
            } else {
                0.5
            };
        }
        self.obstacle_strokes.truncate(MAX_SAVED_OBSTACLE_STROKES);
        if !self.engine.trail().iter().all(|v| v.is_finite()) {
            for cell in self.engine.trail_mut() {
                *cell = if cell.is_finite() {
                    cell.clamp(0.0, 120.0)
                } else {
                    0.0
                };
            }
        }
    }
}

impl InteractiveTool for PhysarumTool {
    fn name(&self) -> &'static str {
        "Physarum Network Lab"
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        Vec::new()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        let w = self.engine.width();
        let h = self.engine.height();
        (w > 0 && h > 0 && self.pixels.len() == w * h).then(|| ColorImage {
            size: [w, h],
            pixels: self.pixels.clone(),
            ..Default::default()
        })
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_benchmark_datasets()
    }

    fn step_simulation(&mut self) {
        for _ in 0..2 {
            self.step_once();
        }
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_lab(ui);
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.sensor_angle.to_bits().hash(&mut h);
        self.turn_angle.to_bits().hash(&mut h);
        self.sensor_distance.to_bits().hash(&mut h);
        self.step_size.to_bits().hash(&mut h);
        self.deposition.to_bits().hash(&mut h);
        self.diffusion_rate.to_bits().hash(&mut h);
        self.decay_rate.to_bits().hash(&mut h);
        self.prune_rate.to_bits().hash(&mut h);
        self.beacon_strength.to_bits().hash(&mut h);
        self.grid_size.hash(&mut h);
        self.agent_count.hash(&mut h);
        self.preset.hash(&mut h);
        self.terminals.len().hash(&mut h);
        for &[u, v] in &self.terminals {
            u.to_bits().hash(&mut h);
            v.to_bits().hash(&mut h);
        }
        self.obstacle_strokes.len().hash(&mut h);
        self.rng_seed.hash(&mut h);
        self.brush_radius.to_bits().hash(&mut h);
        self.brush_mode.hash(&mut h);
        self.show_skeleton_overlay.hash(&mut h);
        self.playing.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.current_state_snapshot()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: PhysarumToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Physarum state: {e}"))?;
        parsed.validate()?;
        self.sensor_angle = parsed.sensor_angle;
        self.turn_angle = parsed.turn_angle;
        self.sensor_distance = parsed.sensor_distance;
        self.step_size = parsed.step_size;
        self.deposition = parsed.deposition;
        self.diffusion_rate = parsed.diffusion_rate;
        self.decay_rate = parsed.decay_rate;
        self.prune_rate = parsed.prune_rate;
        self.beacon_strength = parsed.beacon_strength;
        self.grid_size = parsed.grid_size;
        self.agent_count = parsed.agent_count;
        self.preset = parsed.preset;
        self.terminals = parsed.terminals;
        self.obstacle_strokes = parsed.obstacle_strokes;
        self.rng_seed = parsed.rng_seed;
        self.brush_radius = parsed.brush_radius;
        self.brush_mode = parsed.brush_mode;
        self.show_skeleton_overlay = parsed.show_skeleton_overlay;
        self.playing = parsed.playing;
        self.diagnostic = None;
        self.rebuild_from_state();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PhysarumTool",
        display_name: "Physarum Network Lab",
        domain: "biology",
        tags: &[
            "physarum",
            "slime-mold",
            "chemotaxis",
            "steiner",
            "dijkstra",
            "maze",
            "tokyo-rail",
        ],
        build: || Box::new(PhysarumTool::default()),
    }
}
