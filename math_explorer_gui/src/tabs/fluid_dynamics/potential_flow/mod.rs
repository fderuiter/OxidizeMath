//! Interactive 2D potential-flow wind tunnel with iso-$\psi$ contours, stagnation
//! markers, `RungeKutta4` dye streaks, and tactile singularity handles.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{
    CalloutManager, CalloutSpec, InteractionContext, InteractiveTool, SpatialBrushKit,
};
use eframe::egui;
use egui_plot::PlotPoint;
#[cfg(test)]
use egui_plot::PlotPoints;
use math_explorer::physics::fluid_dynamics::{
    potential_flow::{Doublet, FlowElement, PotentialFlowField, Source, UniformFlow, Vortex},
    solver::{PotentialTracerSystem, TracerProbeReadout},
    types::FluidProperties,
};
use nalgebra::Vector2;
use std::collections::VecDeque;

mod cache;
mod canvas;
mod controls;
mod interact;
mod presets;
mod shader;
#[cfg(test)]
pub(crate) use shader::MAX_SHADER_SINGULARITIES;

pub(crate) const FLOW_STRENGTH_MIN: f64 = -1_000.0;
pub(crate) const FLOW_STRENGTH_MAX: f64 = 1_000.0;
pub(crate) const FLOW_POS_MIN: f64 = -50.0;
pub(crate) const FLOW_POS_MAX: f64 = 50.0;
pub(crate) const FLOW_ANGLE_MIN: f64 = -180.0;
pub(crate) const FLOW_ANGLE_MAX: f64 = 180.0;
pub(crate) const FLOW_VIEW_MIN: f64 = 0.1;
pub(crate) const FLOW_VIEW_MAX: f64 = 50.0;
pub(crate) const DYE_SPEED_MIN: f64 = 0.1;
pub(crate) const DYE_SPEED_MAX: f64 = 5.0;
pub(crate) const DYE_EMITTER_COUNT: usize = 11;
pub(crate) const DYE_TRAIL_CAP: usize = 64;

#[derive(Clone, Debug)]
pub(crate) struct DyeStreak {
    pub(crate) emitter: [f64; 2],
    pub(crate) head: [f64; 2],
    pub(crate) trail: VecDeque<[f64; 2]>,
}

/// Interactive 2D potential-flow simulation tool.
pub struct PotentialFlowTool {
    pub(crate) field: PotentialFlowField,
    pub(crate) elements: Vec<SerializedFlowElement>,
    pub(crate) grid_size: usize,
    pub(crate) view_range: f64,
    pub(crate) selected_element_type: ElementType,
    pub(crate) new_element_strength: f64,
    pub(crate) new_element_x: f64,
    pub(crate) new_element_y: f64,
    pub(crate) new_element_angle: f64,
    pub(crate) brush_kit: SpatialBrushKit,
    pub(crate) active_drag_element: Option<usize>,
    pub(crate) show_arrows: bool,
    pub(crate) show_streamlines: bool,
    pub(crate) show_stagnation_points: bool,
    pub(crate) show_dye_streaks: bool,
    pub(crate) show_probe: bool,
    pub(crate) dye_playing: bool,
    pub(crate) dye_speed: f64,
    pub(crate) probe_pos: [f64; 2],
    pub(crate) selected_element: Option<usize>,
    pub(crate) dragged_element: Option<usize>,
    pub(crate) dragging_probe: bool,
    pub(crate) canvas_stamp_mode: bool,
    pub(crate) focused_reticle: [f64; 2],
    pub(crate) plot_focused: bool,
    pub(crate) cached_field_key: u64,
    pub(crate) cached_streamlines: Vec<([f64; 2], [f64; 2])>,
    pub(crate) cached_polylines: Vec<Vec<[f64; 2]>>,
    pub(crate) cached_stagnation: Vec<[f64; 2]>,
    pub(crate) dye_streaks: Vec<DyeStreak>,
    pub(crate) high_contrast: bool,
    pub(crate) callout_manager: CalloutManager,
    pub(crate) origins: Vec<PlotPoint>,
    pub(crate) tips: Vec<PlotPoint>,
    pub(crate) grid_revision: u64,
    pub(crate) cached_revision: u64,
    pub(crate) cached_grid_size: usize,
    pub(crate) cached_arrow_view_bits: u64,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) enum ElementType {
    Uniform,
    #[default]
    Source,
    Sink,
    Vortex,
    Doublet,
}

impl ElementType {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Uniform => "Uniform",
            Self::Source => "Source",
            Self::Sink => "Sink",
            Self::Vortex => "Vortex",
            Self::Doublet => "Doublet",
        }
    }
}

#[derive(PartialEq, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SerializedFlowElement {
    pub(crate) element_type: ElementType,
    pub(crate) strength: f64,
    #[serde(default)]
    pub(crate) x: f64,
    #[serde(default)]
    pub(crate) y: f64,
    #[serde(default)]
    pub(crate) angle: f64,
}

impl SerializedFlowElement {
    pub(crate) fn is_valid(&self) -> bool {
        self.strength.is_finite()
            && (FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX).contains(&self.strength)
            && self.x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.x)
            && self.y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.y)
            && self.angle.is_finite()
            && (FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX).contains(&self.angle)
    }

    pub(crate) const fn has_canvas_handle(&self) -> bool {
        !matches!(self.element_type, ElementType::Uniform)
    }
}

impl Default for PotentialFlowTool {
    fn default() -> Self {
        let mut tool = Self {
            field: PotentialFlowField::new(),
            elements: presets::symmetric_cylinder_elements(),
            grid_size: 20,
            view_range: 10.0,
            selected_element_type: ElementType::Source,
            new_element_strength: 10.0,
            new_element_x: 0.0,
            new_element_y: 0.0,
            new_element_angle: 0.0,
            brush_kit: Self::default_flow_brush_kit(),
            active_drag_element: None,
            show_arrows: true,
            show_streamlines: true,
            show_stagnation_points: true,
            show_dye_streaks: true,
            show_probe: true,
            dye_playing: false,
            dye_speed: 1.0,
            probe_pos: [-4.0, 2.2],
            selected_element: None,
            dragged_element: None,
            dragging_probe: false,
            canvas_stamp_mode: false,
            focused_reticle: [0.0, 0.0],
            plot_focused: false,
            cached_field_key: 0,
            cached_streamlines: Vec::new(),
            cached_polylines: Vec::new(),
            cached_stagnation: Vec::new(),
            dye_streaks: Vec::new(),
            high_contrast: false,
            callout_manager: CalloutManager::new(),
            origins: Vec::new(),
            tips: Vec::new(),
            grid_revision: 0,
            cached_revision: 0,
            cached_grid_size: 20,
            cached_arrow_view_bits: cache::canonical_bits(10.0),
        };
        tool.rebuild_field();
        tool.reseed_dye_emitters();
        tool.rebuild_arrows();
        tool
    }
}

impl PotentialFlowTool {
    pub(crate) fn rebuild_field(&mut self) {
        self.field.clear();
        for el in &self.elements {
            match el.element_type {
                ElementType::Uniform => self
                    .field
                    .add(Box::new(UniformFlow::new(el.strength, el.angle))),
                ElementType::Source => {
                    self.field
                        .add(Box::new(Source::new(el.strength, el.x, el.y)))
                }
                ElementType::Sink => {
                    self.field
                        .add(Box::new(Source::new(-el.strength, el.x, el.y)))
                }
                ElementType::Vortex => {
                    self.field
                        .add(Box::new(Vortex::new(el.strength, el.x, el.y)))
                }
                ElementType::Doublet => {
                    self.field
                        .add(Box::new(Doublet::new(el.strength, el.x, el.y)))
                }
            }
        }
        self.cached_field_key = 0;
        self.grid_revision = self.grid_revision.wrapping_add(1);
        self.ensure_contour_cache();
    }

    pub(crate) fn rebuild_arrows(&mut self) {
        self.populate_arrow_buffers();
        self.cached_revision = self.grid_revision;
        self.cached_grid_size = self.grid_size;
        self.cached_arrow_view_bits = cache::canonical_bits(self.view_range);
    }

    pub(crate) fn reconcile_arrows(&mut self) {
        let view_bits = cache::canonical_bits(self.view_range);
        if self.cached_grid_size != self.grid_size || self.cached_arrow_view_bits != view_bits {
            self.grid_revision = self.grid_revision.wrapping_add(1);
        }
        if self.cached_revision != self.grid_revision {
            self.rebuild_arrows();
        }
    }

    pub(crate) fn freestream_speed(&self) -> f64 {
        let u = self.freestream_magnitude();
        if u > 1e-3 {
            u
        } else {
            10.0
        }
    }

    pub(crate) fn evaluate_probe(&self) -> TracerProbeReadout {
        let pos = Vector2::new(self.probe_pos[0], self.probe_pos[1]);
        let u_inf = self.freestream_speed();
        let tracer = PotentialTracerSystem::new(&self.field, pos, u_inf * 8.0 + 40.0);
        let air = FluidProperties::air();
        tracer.evaluate_probe(&air, u_inf, 101_325.0, 0.02)
    }

    pub(crate) fn add_element_at(&mut self, x: f64, y: f64) {
        if self.elements.len() >= crate::framework::MAX_JSON_ARRAY_LEN {
            return;
        }
        let cx = x.clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        let cy = y.clamp(FLOW_POS_MIN, FLOW_POS_MAX);
        self.elements.push(SerializedFlowElement {
            element_type: self.selected_element_type,
            strength: self
                .new_element_strength
                .clamp(FLOW_STRENGTH_MIN, FLOW_STRENGTH_MAX),
            x: cx,
            y: cy,
            angle: self.new_element_angle.clamp(FLOW_ANGLE_MIN, FLOW_ANGLE_MAX),
        });
        self.selected_element = Some(self.elements.len() - 1);
        self.focused_reticle = [cx, cy];
        self.rebuild_field();
        self.reseed_dye_emitters();
    }

    pub(crate) fn add_element(&mut self) {
        self.add_element_at(self.new_element_x, self.new_element_y);
    }

    pub(crate) fn clear(&mut self) {
        self.elements.clear();
        self.selected_element = None;
        self.dragged_element = None;
        self.active_drag_element = None;
        self.field.clear();
        self.origins.clear();
        self.tips.clear();
        self.cached_field_key = 0;
        self.cached_streamlines.clear();
        self.cached_polylines.clear();
        self.cached_stagnation.clear();
        self.grid_revision = self.grid_revision.wrapping_add(1);
        self.reseed_dye_emitters();
    }

    #[cfg(test)]
    pub(crate) fn compute_arrow_field(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        if !self.view_range.is_finite() || self.view_range <= 0.0 || self.grid_size < 2 {
            return (Vec::new(), Vec::new());
        }
        let step = (2.0 * self.view_range) / (self.grid_size as f64);
        if !step.is_finite() || step <= 0.0 {
            return (Vec::new(), Vec::new());
        }
        let min = -self.view_range;
        let max_len = step * 0.8;
        let mut origins = Vec::new();
        let mut tips = Vec::new();
        for i in 0..=self.grid_size {
            let x = min + (i as f64) * step;
            for j in 0..=self.grid_size {
                let y = min + (j as f64) * step;
                let vel = self.field.velocity(x, y);
                let mag = vel.norm();
                if mag.is_finite() && mag > 1e-6 {
                    let scale = (mag * 0.1).min(max_len) / mag;
                    let tip = [x + vel.x * scale, y + vel.y * scale];
                    if tip[0].is_finite() && tip[1].is_finite() {
                        origins.push([x, y]);
                        tips.push(tip);
                    }
                }
            }
        }
        (origins, tips)
    }

    fn populate_arrow_buffers(&mut self) {
        self.origins.clear();
        self.tips.clear();
        if !self.view_range.is_finite()
            || self.view_range <= 0.0
            || !(2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
        {
            return;
        }
        let step = (2.0 * self.view_range) / (self.grid_size as f64);
        if !step.is_finite() || step <= 0.0 {
            return;
        }
        let target_cap = (self.grid_size + 1).pow(2);
        self.origins.reserve(target_cap);
        self.tips.reserve(target_cap);
        if self.elements.is_empty() {
            return;
        }
        let min = -self.view_range;
        let max_len = step * 0.8;
        let cyl = self.inferred_cylinder();
        for ix in 0..=self.grid_size {
            let x = min + (ix as f64) * step;
            for iy in 0..=self.grid_size {
                let y = min + (iy as f64) * step;
                if cyl.is_some_and(|(cx, cy, r)| (x - cx).hypot(y - cy) < r * 0.88) {
                    continue;
                }
                let vel = self.field.velocity(x, y);
                let mag = vel.norm();
                if mag.is_finite() && mag > 1e-6 {
                    let scale = (mag * 0.1).min(max_len) / mag;
                    let (tx, ty) = (x + vel.x * scale, y + vel.y * scale);
                    if tx.is_finite() && ty.is_finite() {
                        self.origins.push(PlotPoint::new(x, y));
                        self.tips.push(PlotPoint::new(tx, ty));
                    }
                }
            }
        }
    }
}

impl InteractiveTool for PotentialFlowTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Potential Flow"
    }

    fn domain(&self) -> &'static str {
        "fluid_dynamics"
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        self.compute_aerodynamic_observables()
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.callout_manager.sync_specs(callouts);
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::plot().with_shader()
    }

    fn export_glsl_shader(
        &self,
        dialect: crate::framework::ShaderDialect,
    ) -> Result<String, String> {
        self.build_glsl_shader(dialect)
    }

    fn has_finite_parameters(&self) -> bool {
        self.check_finite_parameters()
    }

    fn save_state(&self) -> Option<String> {
        self.save_state_json()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_state_json(state)
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn on_click(&mut self, ctx: &InteractionContext) {
        self.handle_flow_click(ctx);
    }

    fn on_drag(&mut self, ctx: &InteractionContext) {
        self.handle_flow_drag(ctx);
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        self.apply_flow_brush_context(ctx);
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.brush_kit.focused_plot_id()
    }

    fn step_simulation(&mut self) {
        self.step_dye_tracers(0.03 * self.dye_speed);
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.high_contrast = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        self.reconcile_arrows();
        self.ensure_contour_cache();
        if self.dye_playing && self.show_dye_streaks {
            let dt = ui
                .ctx()
                .input(|i| i.stable_dt.clamp(1.0 / 120.0, 0.05) as f64)
                * self.dye_speed;
            self.step_dye_tracers(dt);
            ui.ctx().request_repaint();
        }
        self.draw_sidebar_controls(ui);
        self.draw_wind_tunnel_canvas(ui);
    }
}

// [cite:modular_polynomials_review]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PotentialFlowTool",
        display_name: "Potential Flow",
        domain: "fluid_dynamics",
        tags: &["laplace", "irrotational", "incompressible", "streamlines", "vortex"],
        build: || Box::new(PotentialFlowTool::default()),
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;
