#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::potential_flow_streamlines::{default_show_streamlines, default_streamline_count};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{
    BrushMode, CalloutManager, CalloutSpec, InteractionContext, InteractiveTool,
    PlotBrushController, PlotBrushFrame, SpatialBrushKit,
};
use eframe::egui;
#[cfg(test)]
use egui_plot::PlotPoints;
use egui_plot::{Legend, Plot, PlotPoint};
use math_explorer::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, PotentialFlowField, Source, UniformFlow, Vortex,
};
use std::f64::consts::PI;

#[path = "potential_flow/cache.rs"]
mod cache;
#[path = "potential_flow/interact.rs"]
mod interact;
#[path = "potential_flow/shader.rs"]
mod shader;
#[path = "potential_flow/streamlines.rs"]
mod streamlines;
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
    pub(crate) show_streamlines: bool,
    pub(crate) streamline_count: usize,
    pub(crate) show_stagnation_points: bool,
    pub(crate) cached_polylines: Vec<Vec<[f64; 2]>>,
    pub(crate) cached_stagnation: Vec<[f64; 2]>,
    pub(crate) cached_view_range: f64,
    pub(crate) cached_streamline_count: usize,
    pub(crate) callout_manager: CalloutManager,
    pub(crate) high_contrast: bool,
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
            new_element_strength: 10.0,
            new_element_x: 0.0,
            new_element_y: 0.0,
            new_element_angle: 0.0,
            brush_kit: Self::default_flow_brush_kit(),
            active_drag_element: None,
            show_streamlines: default_show_streamlines(),
            streamline_count: default_streamline_count(),
            show_stagnation_points: true,
            cached_polylines: Vec::new(),
            cached_stagnation: Vec::new(),
            cached_view_range: 10.0,
            cached_streamline_count: default_streamline_count(),
            callout_manager: CalloutManager::new(),
            high_contrast: false,
            origins: Vec::new(),
            tips: Vec::new(),
            grid_revision: 0,
            cached_revision: 0,
            cached_grid_size: 0,
            cached_arrow_view_bits: 0,
        };
        tool.rebuild_field();
        tool.rebuild_arrows();
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

    fn observables(&self) -> Vec<(&'static str, f64)> {
        self.flow_observables()
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
        (2..=crate::presets::MAX_INTERACTIVE_GRID_SIZE).contains(&self.grid_size)
            && (2..=64).contains(&self.streamline_count)
            && self.view_range.is_finite()
            && (FLOW_VIEW_MIN..=FLOW_VIEW_MAX).contains(&self.view_range)
            && self.new_element_strength.is_finite()
            && (FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX).contains(&self.new_element_strength)
            && self.new_element_x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_x)
            && self.new_element_y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_y)
            && self.new_element_angle.is_finite()
            && (FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX).contains(&self.new_element_angle)
            && self.elements.len() <= crate::framework::MAX_JSON_ARRAY_LEN
            && self.elements.iter().all(SerializedFlowElement::is_valid)
    }

    fn save_state(&self) -> Option<String> {
        self.save_state_json()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_state_json(state)
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
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

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.high_contrast = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        self.draw_controls(ui);
        if (self.cached_view_range - self.view_range).abs() > 1e-9
            || self.cached_streamline_count != self.streamline_count
        {
            self.refresh_streamline_cache();
        }
        self.reconcile_arrows();
        let allow_pan = !self.brush_kit.is_sculpting();
        let vr = self.view_range.max(1.0);
        let mut brush_frame: Option<PlotBrushFrame> = None;
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let dark_mode = ui.visuals().dark_mode || self.high_contrast;
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let plot_resp = Plot::new("potential_flow_plot")
                .data_aspect(1.0)
                .view_aspect(1.0)
                .allow_drag(allow_pan)
                .allow_zoom(allow_pan)
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    interact::render_flow_layers(
                        plot_ui,
                        &palette,
                        self.show_streamlines
                            .then_some(self.cached_polylines.as_slice()),
                        (&self.origins, &self.tips),
                        self.show_stagnation_points
                            .then_some(self.cached_stagnation.as_slice()),
                    );
                    self.render_singularity_handles(plot_ui, &palette, dark_mode);
                    self.callout_manager.render_on_plot(plot_ui);
                    brush_frame = Some(PlotBrushController::capture_plot(
                        &mut self.brush_kit,
                        plot_ui,
                        [-vr, -vr],
                        [vr, vr],
                        Some(BrushMode::Erase),
                    ));
                });
            if let Some(frame) = brush_frame {
                if !frame.is_dragging {
                    self.active_drag_element = None;
                }
                frame.dispatch(ui, &plot_resp.response, self);
            }
        });
    }
}

impl PotentialFlowTool {
    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "potential_flow_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Potential Flow Controls");
            ui.separator();
            self.draw_add_element_controls(ui);
            self.draw_brush_toolbar(ui);
            ui.separator();
            self.draw_active_element_sliders(ui);
            if ui
                .button("🔄 Clear All")
                .accessible_hover_text("Remove all potential flow elements and reset the field")
                .clicked()
            {
                self.clear();
            }
            ui.separator();
            ui.checkbox(&mut self.show_streamlines, "Show Streamlines (ψ)");
            if self.show_streamlines {
                ui.add(
                    egui::Slider::new(&mut self.streamline_count, 4..=48)
                        .text("Streamline Density"),
                );
            }
            ui.checkbox(&mut self.show_stagnation_points, "Show Stagnation Points");
            let max_grid = crate::presets::MAX_INTERACTIVE_GRID_SIZE;
            let clamped_grid = self.grid_size.clamp(2, max_grid);
            if clamped_grid != self.grid_size {
                self.grid_size = clamped_grid;
                self.grid_revision = self.grid_revision.wrapping_add(1);
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.grid_size, 10..=max_grid)
                        .text("Visualization - Grid Density"),
                )
                .changed()
            {
                self.grid_revision = self.grid_revision.wrapping_add(1);
            }
            if ui
                .add(
                    egui::Slider::new(&mut self.view_range, 5.0..=FLOW_VIEW_MAX).text("View Range"),
                )
                .changed()
            {
                self.grid_revision = self.grid_revision.wrapping_add(1);
            }
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
        self.refresh_streamline_cache();
        self.grid_revision = self.grid_revision.wrapping_add(1);
    }

    pub(crate) fn add_element(&mut self) {
        self.elements.push(SerializedFlowElement {
            element_type: self.selected_element_type,
            strength: self
                .new_element_strength
                .clamp(FLOW_STRENGTH_MIN, FLOW_STRENGTH_MAX),
            x: self.new_element_x.clamp(FLOW_POS_MIN, FLOW_POS_MAX),
            y: self.new_element_y.clamp(FLOW_POS_MIN, FLOW_POS_MAX),
            angle: self.new_element_angle.clamp(FLOW_ANGLE_MIN, FLOW_ANGLE_MAX),
        });
        self.rebuild_field();
    }

    pub(crate) fn clear(&mut self) {
        self.active_drag_element = None;
        self.elements.clear();
        self.field.clear();
        self.cached_polylines.clear();
        self.cached_stagnation.clear();
        self.origins.clear();
        self.tips.clear();
        self.grid_revision = self.grid_revision.wrapping_add(1);
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
#[path = "potential_flow/tests.rs"]
mod tests;
