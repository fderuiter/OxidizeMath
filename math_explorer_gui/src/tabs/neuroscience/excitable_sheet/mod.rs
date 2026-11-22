#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod canvas;
mod dipole_bridge;
mod interaction;
pub mod presets;
pub mod state;
mod ui;

use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    ExportCapabilities, InteractionContext, InteractiveTool, ToolMetadata, ToolStateBounds,
};
use dipole_bridge::LeadIntegratorBridge;
use eframe::egui;
use math_explorer::biology::morphogenesis::FitzHughNagumoKinetics;
use math_explorer::biology::neuroscience::{
    ExcitableSheet2D, HodgkinHuxleyCable, DEFAULT_SHEET_DIM,
};
use math_explorer::math_kernel::types::flatten_2d_index;
use state::{ExcitableSheetState, ExcitableSheetTheory, PropagationMode};
use std::collections::VecDeque;

const MAX_TRACE_SAMPLES: usize = 360;

/// Interactive 2D FitzHugh-Nagumo excitable sheet and 1D Hodgkin-Huxley cable tool with virtual electrodes.
pub struct ExcitableSheetTool {
    pub(crate) cfg: ExcitableSheetState,
    pub(crate) sheet: ExcitableSheet2D,
    pub(crate) cable: HodgkinHuxleyCable,
    pub(crate) cable_kymograph: Vec<f64>,
    pub(crate) lead_bridge: LeadIntegratorBridge,
    pub(crate) trace_history: VecDeque<(f64, [f64; 3])>,
    pub(crate) cursor_grid: (f64, f64),
    pub(crate) dragging_lead: Option<usize>,
    pub(crate) sheet_texture: Option<egui::TextureHandle>,
    pub(crate) focused_canvas: Option<egui::Id>,
    last_valid_cfg: ExcitableSheetState,
    diagnostic: Option<DiagnosticReport>,
    revision: u64,
    theory: ExcitableSheetTheory,
}

impl Default for ExcitableSheetTool {
    fn default() -> Self {
        let cfg = ExcitableSheetState::default();
        let sheet = ExcitableSheet2D::new_default();
        let cable = HodgkinHuxleyCable::new_default();
        let lead_bridge = LeadIntegratorBridge::new(DEFAULT_SHEET_DIM, DEFAULT_SHEET_DIM, cfg.leads);
        let mut tool = Self {
            last_valid_cfg: cfg.clone(),
            cfg,
            sheet,
            cable,
            cable_kymograph: vec![-65.0; DEFAULT_SHEET_DIM * DEFAULT_SHEET_DIM],
            lead_bridge,
            trace_history: VecDeque::with_capacity(MAX_TRACE_SAMPLES),
            cursor_grid: (48.0, 48.0),
            dragging_lead: None,
            sheet_texture: None,
            focused_canvas: None,
            diagnostic: None,
            revision: 1,
            theory: ExcitableSheetTheory,
        };
        let initial_running = tool.cfg.running;
        tool.apply_scenario(tool.cfg.scenario);
        tool.cfg.running = initial_running;
        for _ in 0..8 {
            tool.step_once_guarded();
        }
        tool.revision = 1;
        tool
    }
}

impl ExcitableSheetTool {
    pub(crate) fn sync_engine_parameters(&mut self) {
        self.sheet.system.kinetics =
            FitzHughNagumoKinetics::new(self.cfg.a, self.cfg.b, self.cfg.epsilon, self.cfg.i_ext);
        self.sheet.set_diffusion_u(self.cfg.d_u);
        self.cable.set_diffusion_v((self.cfg.d_u * 2.4).clamp(0.1, 3.0));
        self.lead_bridge.sync_leads(self.cfg.leads);
    }

    pub(crate) fn trigger_defibrillation(&mut self) {
        self.sheet.defibrillate();
        self.cable.reset_resting();
        self.record_lead_sample();
        self.bump_revision();
    }

    pub(crate) fn clear_trace_history(&mut self) {
        self.trace_history.clear();
    }

    pub(crate) fn step_once_guarded(&mut self) {
        self.last_valid_cfg = self.cfg.clone();
        match self.cfg.propagation_mode {
            PropagationMode::FitzHughNagumo2D => self.sheet.step(self.cfg.dt),
            PropagationMode::HodgkinHuxleyCable1D => {
                let hh_dt = (self.cfg.dt * 0.15).clamp(0.005, 0.025);
                for _ in 0..4 {
                    self.cable.step(hh_dt);
                }
                self.push_cable_kymograph_row();
            }
        }
        if !self.tissue_is_finite() {
            self.diagnostic = Some(DiagnosticReport::new(
                self.sheet.elapsed_time(),
                self.trace_history.len(),
                "Voltage",
                "Non-finite membrane potential detected; rolling back step.",
            ));
            FiniteStateGuard::auto_scale_step_size(self);
            return;
        }
        self.diagnostic = None;
        self.record_lead_sample();
    }

    pub(crate) fn record_lead_sample(&mut self) {
        let (t, sample) = match self.cfg.propagation_mode {
            PropagationMode::FitzHughNagumo2D => (
                self.sheet.elapsed_time(),
                self.lead_bridge.evaluate_sheet(
                    self.sheet.voltage(),
                    self.sheet.scar_mask(),
                    self.cfg.readout_bipolar,
                ),
            ),
            PropagationMode::HodgkinHuxleyCable1D => (
                self.cable.elapsed_time(),
                self.lead_bridge.evaluate_cable(
                    self.cable.voltage(),
                    self.cable.scar_mask(),
                    self.cfg.readout_bipolar,
                ),
            ),
        };
        if self.trace_history.len() >= MAX_TRACE_SAMPLES {
            self.trace_history.pop_front();
        }
        self.trace_history.push_back((t, sample));
    }

    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    fn tissue_is_finite(&self) -> bool {
        self.sheet.voltage().iter().all(|v| v.is_finite())
            && self.sheet.recovery().iter().all(|w| w.is_finite())
            && self.cable.voltage().iter().all(|v| v.is_finite())
    }

    fn push_cable_kymograph_row(&mut self) {
        let n = DEFAULT_SHEET_DIM;
        self.cable_kymograph.copy_within(n..(n * n), 0);
        let last_row = n - 1;
        for (x, &mv) in self.cable.voltage().iter().enumerate().take(n) {
            let idx = flatten_2d_index(x, last_row, n);
            self.cable_kymograph[idx] = mv;
        }
    }
}

impl FiniteStateGuard for ExcitableSheetTool {
    fn is_finite_state(&self) -> bool {
        self.diagnostic.is_none() && self.cfg.validate().is_ok() && self.tissue_is_finite()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic.as_ref()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        self.cfg = self.last_valid_cfg.clone();
        self.cfg.dt = (self.cfg.dt * 0.5).max(0.01);
        self.sync_engine_parameters();
        self.sheet.defibrillate();
        self.cable.reset_resting();
        self.diagnostic = None;
        self.record_lead_sample();
    }

    fn clamp_parameters(&mut self) {
        self.cfg.clamp_in_place();
        self.sync_engine_parameters();
        self.diagnostic = None;
    }

    fn reset_parameters(&mut self) {
        self.cfg = ExcitableSheetState::default();
        self.sync_engine_parameters();
        self.apply_scenario(self.cfg.scenario);
        self.diagnostic = None;
    }
}

impl InteractiveTool for ExcitableSheetTool {
    fn name(&self) -> &'static str {
        "2D Excitable Sheet & Virtual Electrodes"
    }

    fn domain(&self) -> &'static str {
        "neuroscience"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        &self.theory
    }

    fn show_ui(&mut self, ui: &mut egui::Ui) {
        self.render_side_panel(ui);
    }

    fn draw(&mut self, ui: &mut egui::Ui, response: &egui::Response, painter: &egui::Painter) {
        self.draw_canvas(ui, response, painter);
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        self.handle_brush(ctx);
    }

    fn on_click(&mut self, ctx: &InteractionContext) {
        self.handle_click(ctx);
    }

    fn on_hover(&mut self, ctx: &InteractionContext) {
        self.handle_hover(ctx);
    }

    fn on_keyboard(&mut self, ctx: &InteractionContext) {
        self.handle_keyboard(ctx);
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn has_finite_parameters(&self) -> bool {
        self.cfg.validate().is_ok()
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: ExcitableSheetState =
            serde_json::from_str(state).map_err(|e| format!("Invalid ExcitableSheet JSON: {e}"))?;
        parsed.validate()?;
        let scenario_changed = parsed.scenario != self.cfg.scenario;
        self.cfg = parsed;
        self.sync_engine_parameters();
        if scenario_changed {
            let saved = self.cfg.clone();
            self.apply_scenario(saved.scenario);
            self.cfg = saved;
            self.lead_bridge.sync_leads(self.cfg.leads);
        }
        if self.cfg.ablated_to_edge {
            self.sheet.ablate_scar_to_boundary();
        }
        self.record_lead_sample();
        self.bump_revision();
        Ok(())
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_raster_image(&self) -> Option<egui::ColorImage> {
        Some(self.build_export_raster_image())
    }

    fn export_figure_options(&self) -> Option<egui_plot::export::ExportOptions> {
        Some(egui_plot::export::ExportOptions::default())
    }

    fn step_simulation(&mut self) {
        self.step_once_guarded();
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    ToolMetadata {
        id: "ExcitableSheetTool",
        display_name: "2D Excitable Sheet & Virtual Electrodes",
        domain: "neuroscience",
        tags: &["electrophysiology", "fitzhugh-nagumo", "ecg", "spiral-waves"],
        build: || Box::new(ExcitableSheetTool::default()),
    }
}

#[cfg(test)]
mod tests;
