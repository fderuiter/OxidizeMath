#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::InteractiveTool;
use crate::presets::{InteractivePreset, OdePreset, OdePresetConfig, PresetDialogState};
use crate::widgets::{GridDataSource, NumericalDataGrid};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::analysis::ode::{
    OdeModel, OdeSystem, RungeKutta4, TimeStepper, VecState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum OdeViewMode {
    Plot,
    DataGrid,
}

// ----------------------------------------------------------------------------
// ODE System Definitions
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ExponentialOde { k: f64 }

impl OdeSystem<VecState> for ExponentialOde {
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }
    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        out.0[0] = self.k * state.0[0];
    }
}

#[derive(Clone, Debug)]
struct HarmonicOscillatorOde { k: f64 }

impl OdeSystem<VecState> for HarmonicOscillatorOde {
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }
    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        out.0[0] = state.0[1];
        out.0[1] = -self.k * state.0[0];
    }
}

#[derive(Clone, Debug)]
struct LogisticGrowthOde { r: f64, cap_k: f64 }

impl OdeSystem<VecState> for LogisticGrowthOde {
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }
    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        out.0[0] = self.r * state.0[0] * (1.0 - state.0[0] / self.cap_k);
    }
}

// ----------------------------------------------------------------------------
// Tool Implementation
// ----------------------------------------------------------------------------

pub struct OdeSolverTool {
    preset: OdePreset,
    dt: f64,
    total_time: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
    time_series: Vec<f64>,
    y_series: Vec<f64>,
    v_series: Vec<f64>,
    diverged: bool,
    view_mode: OdeViewMode,
    grid_widget: NumericalDataGrid,
    // Dialog & Presets
    dialog_state: PresetDialogState,
}

struct OdeGridAdapter<'a> {
    tool: &'a mut OdeSolverTool,
}

impl<'a> GridDataSource for OdeGridAdapter<'a> {
    fn num_rows(&self) -> usize {
        self.tool.time_series.len()
    }

    fn num_cols(&self) -> usize {
        if self.tool.preset == OdePreset::HarmonicOscillator { 4 } else { 3 }
    }

    fn header(&self, col: usize) -> String {
        match col {
            0 => "Step".to_string(),
            1 => "Time (t)".to_string(),
            2 => "y(t)".to_string(),
            3 => "y'(t)".to_string(),
            _ => String::new(),
        }
    }

    fn cell_value(&self, row: usize, col: usize) -> f64 {
        match col {
            0 => row as f64,
            1 => self.tool.time_series.get(row).copied().unwrap_or(0.0),
            2 => self.tool.y_series.get(row).copied().unwrap_or(0.0),
            3 => self.tool.v_series.get(row).copied().unwrap_or(0.0),
            _ => 0.0,
        }
    }

    fn set_cell_value(&mut self, row: usize, col: usize, val: f64) {
        if row < self.tool.time_series.len() {
            if row == 0 {
                if col == 2 {
                    self.tool.ic_y0 = val;
                    self.tool.recalculate();
                } else if col == 3 {
                    self.tool.ic_v0 = val;
                    self.tool.recalculate();
                }
            } else {
                if col == 2 && row < self.tool.y_series.len() {
                    self.tool.y_series[row] = val;
                } else if col == 3 && row < self.tool.v_series.len() {
                    self.tool.v_series[row] = val;
                }
            }
        }
    }

    fn is_editable(&self, _row: usize, col: usize) -> bool {
        col >= 2
    }
}

impl Default for OdeSolverTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: OdePreset::Exponential,
            dt: 0.05,
            total_time: 10.0,
            param_k: 1.0,
            param_r: 1.0,
            param_cap_k: 10.0,
            ic_y0: 1.0,
            ic_v0: 0.0,
            time_series: Vec::new(),
            y_series: Vec::new(),
            v_series: Vec::new(),
            diverged: false,
            view_mode: OdeViewMode::Plot,
            grid_widget: NumericalDataGrid::new(),
            dialog_state: PresetDialogState::default(),
        };
        tool.recalculate();
        tool
    }
}

impl OdeSolverTool {
    #[must_use]
    #[allow(dead_code)]
    pub fn diverged(&self) -> bool {
        self.diverged
    }

    pub fn reset_parameters(&mut self) {
        match self.preset {
            OdePreset::Exponential => {
                self.param_k = 1.0;
                self.ic_y0 = 1.0;
            }
            OdePreset::HarmonicOscillator => {
                self.param_k = 1.0;
                self.ic_y0 = 1.0;
                self.ic_v0 = 0.0;
            }
            OdePreset::LogisticGrowth => {
                self.param_r = 1.0;
                self.param_cap_k = 10.0;
                self.ic_y0 = 1.0;
            }
        }
        self.dt = 0.05;
        self.total_time = 10.0;
        self.recalculate();
    }

    #[allow(clippy::too_many_lines)]
    fn recalculate(&mut self) {
        self.time_series.clear();
        self.y_series.clear();
        self.v_series.clear();
        self.diverged = false;

        if self.dt <= 0.0 || self.total_time <= 0.0 || !self.dt.is_finite() || !self.total_time.is_finite() {
            return;
        }

        let num_steps = (self.total_time / self.dt).ceil() as usize;

        match self.preset {
            OdePreset::Exponential => {
                let init = VecState(vec![self.ic_y0]);
                let mut model = OdeModel::new(init.clone(), ExponentialOde { k: self.param_k }, RungeKutta4::new(&init));
                for i in 0..=num_steps {
                    let y = model.get_state().0[0];
                    if !y.is_finite() { self.diverged = true; break; }
                    self.time_series.push(i as f64 * self.dt);
                    self.y_series.push(y);
                    model.step(self.dt);
                }
            }
            OdePreset::HarmonicOscillator => {
                let init = VecState(vec![self.ic_y0, self.ic_v0]);
                let mut model = OdeModel::new(init.clone(), HarmonicOscillatorOde { k: self.param_k }, RungeKutta4::new(&init));
                for i in 0..=num_steps {
                    let (y, v) = (model.get_state().0[0], model.get_state().0[1]);
                    if !y.is_finite() || !v.is_finite() { self.diverged = true; break; }
                    self.time_series.push(i as f64 * self.dt);
                    self.y_series.push(y);
                    self.v_series.push(v);
                    model.step(self.dt);
                }
            }
            OdePreset::LogisticGrowth => {
                let init = VecState(vec![self.ic_y0]);
                let mut model = OdeModel::new(init.clone(), LogisticGrowthOde { r: self.param_r, cap_k: self.param_cap_k }, RungeKutta4::new(&init));
                for i in 0..=num_steps {
                    let y = model.get_state().0[0];
                    if !y.is_finite() { self.diverged = true; break; }
                    self.time_series.push(i as f64 * self.dt);
                    self.y_series.push(y);
                    model.step(self.dt);
                }
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct OdeState {
    dt: f64,
    total_time: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
}

impl InteractiveTool for OdeSolverTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str { "ODE Solvers" }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&OdeState {
            dt: self.dt,
            total_time: self.total_time,
            param_k: self.param_k,
            param_r: self.param_r,
            param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0,
            ic_v0: self.ic_v0,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<OdeState>(state) {
            self.dt = s.dt;
            self.total_time = s.total_time;
            self.param_k = s.param_k;
            self.param_r = s.param_r;
            self.param_cap_k = s.param_cap_k;
            self.ic_y0 = s.ic_y0;
            self.ic_v0 = s.ic_v0;
            self.recalculate();
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        let mut changed = false;

        egui::SidePanel::left("ode_controls").show(ctx, |ui| {
            ui.heading("ODE System");
            ui.separator();

            egui::ComboBox::from_id_salt("ode_preset")
                .selected_text(self.preset.name())
                .show_ui(ui, |ui| {
                    if ui.selectable_value(&mut self.preset, OdePreset::Exponential, OdePreset::Exponential.name()).changed()
                        || ui.selectable_value(&mut self.preset, OdePreset::HarmonicOscillator, OdePreset::HarmonicOscillator.name()).changed()
                        || ui.selectable_value(&mut self.preset, OdePreset::LogisticGrowth, OdePreset::LogisticGrowth.name()).changed()
                    {
                        changed = true;
                    }
                });

            ui.add_space(10.0);
            ui.heading("Parameters");
            match self.preset {
                OdePreset::Exponential => {
                    ui.horizontal(|ui| {
                        ui.label("k (Rate):");
                        if ui.add(egui::DragValue::new(&mut self.param_k).speed(0.1)).changed() { changed = true; }
                    });
                }
                OdePreset::HarmonicOscillator => {
                    ui.horizontal(|ui| {
                        ui.label("k (Spring Constant):");
                        if ui.add(egui::DragValue::new(&mut self.param_k).speed(0.1)).changed() { changed = true; }
                    });
                }
                OdePreset::LogisticGrowth => {
                    ui.horizontal(|ui| {
                        ui.label("r (Growth Rate):");
                        if ui.add(egui::DragValue::new(&mut self.param_r).speed(0.1)).changed() { changed = true; }
                    });
                    ui.horizontal(|ui| {
                        ui.label("K (Carrying Capacity):");
                        if ui.add(egui::DragValue::new(&mut self.param_cap_k).speed(0.1)).changed() { changed = true; }
                    });
                }
            }

            ui.add_space(10.0);
            ui.heading("Initial Conditions");
            ui.horizontal(|ui| {
                ui.label("y(0):");
                if ui.add(egui::DragValue::new(&mut self.ic_y0).speed(0.1)).changed() { changed = true; }
            });
            if self.preset == OdePreset::HarmonicOscillator {
                ui.horizontal(|ui| {
                    ui.label("y'(0):");
                    if ui.add(egui::DragValue::new(&mut self.ic_v0).speed(0.1)).changed() { changed = true; }
                });
            }

            ui.add_space(10.0);
            ui.heading("Solver Settings");
            ui.horizontal(|ui| {
                ui.label("dt (Step Size):");
                if ui.add(egui::DragValue::new(&mut self.dt).speed(0.01).range(0.001..=1.0)).changed() { changed = true; }
            });
            ui.horizontal(|ui| {
                ui.label("Total Time:");
                if ui.add(egui::DragValue::new(&mut self.total_time).speed(1.0).range(1.0..=100.0)).changed() { changed = true; }
            });

            let mut dialog_state = std::mem::take(&mut self.dialog_state);
            if crate::presets::render_preset_buttons(&mut dialog_state, ui, self) {
                changed = true;
            }
            self.dialog_state = dialog_state;

            ui.add_space(10.0);
            ui.separator();
            ui.heading("View Mode");
            ui.selectable_value(&mut self.view_mode, OdeViewMode::Plot, "📈 Plot View");
            ui.selectable_value(&mut self.view_mode, OdeViewMode::DataGrid, "🔢 Numerical Data Grid");
        });

        if changed {
            self.recalculate();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.view_mode == OdeViewMode::DataGrid {
                ui.heading("ODE State Vectors Grid");
                ui.label("Double-click state vector cells to audit or overwrite numerical trajectory values.");
                ui.separator();
                let mut grid_widget = self.grid_widget.clone();
                let mut adapter = OdeGridAdapter { tool: self };
                grid_widget.show(ui, &mut adapter);
                self.grid_widget = grid_widget;
            } else {
                ui.heading("Solution Trajectory");

                if self.diverged {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("⚠️ Solver Divergence Warning: Non-finite state detected (NaN or Inf). Integration terminated.")
                                    .color(egui::Color32::RED)
                                    .strong(),
                            );
                            if ui.button("↻ Reset Parameters").clicked() {
                                self.reset_parameters();
                                changed = true;
                            }
                        });
                    });
                }

                let mut plot_points_y = Vec::new();
                let mut plot_points_v = Vec::new();

                for (i, &t) in self.time_series.iter().enumerate() {
                    if let Some(&y) = self.y_series.get(i) {
                        plot_points_y.push([t, y]);
                    }
                    if let Some(&v) = self.v_series.get(i) {
                        plot_points_v.push([t, v]);
                    }
                }

                let line_y = Line::new("y(t)", PlotPoints::new(plot_points_y)).color(egui::Color32::LIGHT_BLUE);

                Plot::new("ode_plot")
                    .view_aspect(2.0)
                    .x_axis_label("Time (t)")
                    .y_axis_label("Value")
                    .legend(egui_plot::Legend::default())
                    .show(ui, |plot_ui| {
                        plot_ui.line(line_y);
                        if self.preset == OdePreset::HarmonicOscillator && !plot_points_v.is_empty() {
                            let line_v = Line::new("y'(t)", PlotPoints::new(plot_points_v))
                                .color(egui::Color32::LIGHT_RED);
                            plot_ui.line(line_v);
                        }
                    });
            }
        });
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "OdeSolverTool",
        domain: "analysis",
        tags: &[],
        build: || Box::new(OdeSolverTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for OdeSolverTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

impl InteractivePreset for OdeSolverTool {
    type Config = OdePresetConfig;

    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn title(&self) -> String {
        format!("ODE Solver - {}", self.preset.name())
    }

    fn export_config(&self) -> Self::Config {
        OdePresetConfig {
            preset: self.preset,
            dt: self.dt,
            total_time: self.total_time,
            param_k: self.param_k,
            param_r: self.param_r,
            param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0,
            ic_v0: self.ic_v0,
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        self.preset = config.preset;
        self.dt = config.dt;
        self.total_time = config.total_time;
        self.param_k = config.param_k;
        self.param_r = config.param_r;
        self.param_cap_k = config.param_cap_k;
        self.ic_y0 = config.ic_y0;
        self.ic_v0 = config.ic_v0;
        self.recalculate();
        Ok(())
    }
}

#[cfg(test)]
#[path = "ode_tests.rs"]
mod ode_tests;
