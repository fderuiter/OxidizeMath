#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod cache;
mod canvas;
mod controls;
mod presets;
mod shader;
#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

#[cfg(test)]
pub(crate) use shader::MAX_SHADER_SINGULARITIES;

use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::nalgebra::Vector2;
use math_explorer::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, PotentialFlowField, Source, UniformFlow, Vortex,
};
use math_explorer::physics::fluid_dynamics::solver::{PotentialTracerSystem, TracerProbeReadout};
use math_explorer::physics::fluid_dynamics::types::FluidProperties;
use math_explorer::pure_math::analysis::ode::TimeStepper;
use std::f64::consts::PI;

pub(crate) const FLOW_STRENGTH_MIN: f64 = -1_000.0;
pub(crate) const FLOW_STRENGTH_MAX: f64 = 1_000.0;
pub(crate) const FLOW_POS_MIN: f64 = -50.0;
pub(crate) const FLOW_POS_MAX: f64 = 50.0;
pub(crate) const FLOW_ANGLE_MIN: f64 = -180.0;
pub(crate) const FLOW_ANGLE_MAX: f64 = 180.0;
pub(crate) const FLOW_VIEW_MIN: f64 = 0.1;
pub(crate) const FLOW_VIEW_MAX: f64 = 50.0;
pub(crate) const MAX_CONTOUR_LEVELS: usize = 48;
pub(crate) const MAX_DYE_PROBES: usize = 24;
pub(crate) const MAX_DYE_TRAIL: usize = 72;
pub(crate) const DYE_DT: f64 = 0.03;
pub(crate) const DYE_MAX_SPEED: f64 = 80.0;

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
            Self::Uniform => "Uniform Flow",
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DragTarget {
    Element(usize),
    DyeRake,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum FocusTarget {
    #[default]
    Reticle,
    Element(usize),
    DyeRake,
}

#[derive(Clone, Debug)]
pub(crate) struct DyeProbe {
    pub(crate) anchor: [f64; 2],
    pub(crate) head: [f64; 2],
    pub(crate) trail: Vec<[f64; 2]>,
    pub(crate) age: usize,
}

/// Interactive 2D potential-flow wind tunnel with singularity sculpting, iso-$\psi$ contours, and `RungeKutta4` dye probes.
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
    pub(crate) high_contrast: bool,
    pub(crate) show_vectors: bool,
    pub(crate) show_streamlines: bool,
    pub(crate) show_dye: bool,
    pub(crate) contour_levels: usize,
    pub(crate) contour_resolution: usize,
    pub(crate) dye_rake_x: f64,
    pub(crate) dye_rake_y: f64,
    pub(crate) dye_rake_span: f64,
    pub(crate) dye_count: usize,
    pub(crate) stamp_mode: bool,
    pub(crate) is_playing: bool,
    pub(crate) selected_element: Option<usize>,
    pub(crate) selected_probe: usize,
    pub(crate) active_drag: Option<DragTarget>,
    pub(crate) focus_target: FocusTarget,
    pub(crate) reticle: [f64; 2],
    pub(crate) dye_probes: Vec<DyeProbe>,
    pub(crate) cached_handles: Vec<(DragTarget, egui::Pos2)>,
    pub(crate) last_canvas_id: Option<egui::Id>,
    pub(crate) cache: cache::ArrowFieldCache,
}

impl Default for PotentialFlowTool {
    fn default() -> Self {
        let kappa = 2.0 * PI * 10.0 * 2.0 * 2.0;
        let elements = vec![
            SerializedFlowElement {
                element_type: ElementType::Uniform,
                strength: 10.0,
                x: 0.0,
                y: 0.0,
                angle: 0.0,
            },
            SerializedFlowElement {
                element_type: ElementType::Doublet,
                strength: kappa,
                x: 0.0,
                y: 0.0,
                angle: 0.0,
            },
        ];
        let mut tool = Self {
            field: PotentialFlowField::new(),
            elements,
            grid_size: 20,
            view_range: 10.0,
            selected_element_type: ElementType::Source,
            new_element_strength: 25.0,
            new_element_x: 0.0,
            new_element_y: 0.0,
            new_element_angle: 0.0,
            high_contrast: false,
            show_vectors: true,
            show_streamlines: true,
            show_dye: true,
            contour_levels: 17,
            contour_resolution: 48,
            dye_rake_x: -8.0,
            dye_rake_y: 0.0,
            dye_rake_span: 6.5,
            dye_count: 9,
            stamp_mode: false,
            is_playing: false,
            selected_element: Some(1),
            selected_probe: 4,
            active_drag: None,
            focus_target: FocusTarget::Reticle,
            reticle: [0.0, 3.0],
            dye_probes: Vec::new(),
            cached_handles: Vec::new(),
            last_canvas_id: None,
            cache: cache::ArrowFieldCache::default(),
        };
        tool.rebuild_field();
        tool
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

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::plot().with_shader()
    }

    fn export_glsl_shader(
        &self,
        dialect: crate::framework::ShaderDialect,
    ) -> Result<String, String> {
        self.build_glsl_shader(dialect)
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.last_canvas_id
    }

    fn has_finite_parameters(&self) -> bool {
        self.parameters_are_valid()
    }

    fn save_state(&self) -> Option<String> {
        self.save_preset_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.try_load_preset_state(state)
    }

    fn step_simulation(&mut self) {
        self.advance_dye_step();
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_export_datasets()
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.high_contrast = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.is_playing {
            self.advance_dye_step();
            ui.ctx().request_repaint();
        }
        self.draw_controls(ui);
        self.render_canvas(ui);
    }
}

impl PotentialFlowTool {
    pub(crate) fn rebuild_field(&mut self) {
        let mut field = PotentialFlowField::new();
        for el in &self.elements {
            let elem: Box<dyn FlowElement> = match el.element_type {
                ElementType::Uniform => Box::new(UniformFlow::new(el.strength, el.angle)),
                ElementType::Source => Box::new(Source::new(el.strength, el.x, el.y)),
                ElementType::Sink => Box::new(Source::new(-el.strength, el.x, el.y)),
                ElementType::Vortex => Box::new(Vortex::new(el.strength, el.x, el.y)),
                ElementType::Doublet => Box::new(Doublet::new(el.strength, el.x, el.y)),
            };
            field.add(elem);
        }
        self.field = field;
        if self
            .selected_element
            .is_some_and(|idx| idx >= self.elements.len())
        {
            self.selected_element = self.elements.len().checked_sub(1);
        }
        if let FocusTarget::Element(idx) = self.focus_target {
            if idx >= self.elements.len() {
                self.focus_target = FocusTarget::Reticle;
            }
        }
        self.reseed_dye_streaks();
    }

    pub(crate) fn reseed_dye_streaks(&mut self) {
        let count = self.dye_count.clamp(1, MAX_DYE_PROBES);
        self.selected_probe = self.selected_probe.min(count - 1);
        let mut probes = Vec::with_capacity(count);
        for i in 0..count {
            let frac = if count == 1 {
                0.5
            } else {
                (i as f64) / ((count - 1) as f64)
            };
            let anchor = [
                self.dye_rake_x.clamp(FLOW_POS_MIN, FLOW_POS_MAX),
                (self.dye_rake_y + (frac - 0.5) * self.dye_rake_span)
                    .clamp(FLOW_POS_MIN, FLOW_POS_MAX),
            ];
            let mut trail = Vec::with_capacity(32);
            trail.push(anchor);
            let mut tracer = PotentialTracerSystem::new(
                &self.field,
                Vector2::new(anchor[0], anchor[1]),
                DYE_MAX_SPEED,
            );
            let limit = (self.view_range * 1.35).max(1.0);
            for _ in 0..28 {
                tracer.step(DYE_DT);
                let pt = [tracer.position.x, tracer.position.y];
                if !pt[0].is_finite()
                    || !pt[1].is_finite()
                    || pt[0].abs() > limit
                    || pt[1].abs() > limit
                {
                    break;
                }
                trail.push(pt);
            }
            let head = trail.last().copied().unwrap_or(anchor);
            probes.push(DyeProbe {
                anchor,
                head,
                trail,
                age: 0,
            });
        }
        self.dye_probes = probes;
    }

    pub(crate) fn advance_dye_step(&mut self) {
        if self.dye_probes.is_empty() {
            self.reseed_dye_streaks();
        }
        let limit = (self.view_range * 1.35).max(1.0);
        for probe in &mut self.dye_probes {
            let mut tracer = PotentialTracerSystem::new(
                &self.field,
                Vector2::new(probe.head[0], probe.head[1]),
                DYE_MAX_SPEED,
            );
            let prev = tracer.position;
            tracer.step(DYE_DT);
            tracer.step(DYE_DT);
            let moved = (tracer.position - prev).norm();
            let next = [tracer.position.x, tracer.position.y];
            probe.age += 1;
            if !next[0].is_finite()
                || !next[1].is_finite()
                || next[0].abs() > limit
                || next[1].abs() > limit
                || (moved < 1e-5 && probe.age > 8)
                || probe.age > MAX_DYE_TRAIL * 2
            {
                probe.head = probe.anchor;
                probe.trail.clear();
                probe.trail.push(probe.anchor);
                probe.age = 0;
            } else {
                probe.head = next;
                probe.trail.push(next);
                if probe.trail.len() > MAX_DYE_TRAIL {
                    probe.trail.remove(0);
                }
            }
        }
    }

    pub(crate) fn active_probe_readout(&self) -> Option<TracerProbeReadout> {
        let probe = self
            .dye_probes
            .get(self.selected_probe)
            .or_else(|| self.dye_probes.first())?;
        let freestream = self
            .elements
            .iter()
            .filter(|e| e.element_type == ElementType::Uniform)
            .fold(Vector2::zeros(), |acc, e| {
                let rad = e.angle.to_radians();
                acc + Vector2::new(e.strength * rad.cos(), e.strength * rad.sin())
            })
            .norm()
            .max(1.0);
        let air = FluidProperties::air();
        let tracer = PotentialTracerSystem::new(
            &self.field,
            Vector2::new(probe.head[0], probe.head[1]),
            DYE_MAX_SPEED,
        );
        Some(tracer.evaluate_probe(&air, freestream, 101_325.0, DYE_DT))
    }

    pub(crate) fn add_element(&mut self) {
        self.stamp_element_at(self.new_element_x, self.new_element_y);
    }

    pub(crate) fn stamp_element_at(&mut self, x: f64, y: f64) -> Option<usize> {
        if self.elements.len() >= crate::framework::MAX_JSON_ARRAY_LEN {
            return None;
        }
        self.elements.push(SerializedFlowElement {
            element_type: self.selected_element_type,
            strength: self
                .new_element_strength
                .clamp(FLOW_STRENGTH_MIN, FLOW_STRENGTH_MAX),
            x: x.clamp(FLOW_POS_MIN, FLOW_POS_MAX),
            y: y.clamp(FLOW_POS_MIN, FLOW_POS_MAX),
            angle: self.new_element_angle.clamp(FLOW_ANGLE_MIN, FLOW_ANGLE_MAX),
        });
        let idx = self.elements.len() - 1;
        self.selected_element = Some(idx);
        self.rebuild_field();
        Some(idx)
    }

    pub(crate) fn delete_element(&mut self, idx: usize) -> Option<SerializedFlowElement> {
        if idx >= self.elements.len() {
            return None;
        }
        let removed = self.elements.remove(idx);
        self.rebuild_field();
        Some(removed)
    }

    pub(crate) fn clear(&mut self) {
        self.elements.clear();
        self.selected_element = None;
        self.focus_target = FocusTarget::Reticle;
        self.rebuild_field();
    }
}

// [cite:modular_polynomials_review]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PotentialFlowTool",
        display_name: "Potential Flow",
        domain: "fluid_dynamics",
        tags: &["streamlines", "vortex", "aerodynamics", "rk4"],
        build: || Box::new(PotentialFlowTool::default()),
    }
}
