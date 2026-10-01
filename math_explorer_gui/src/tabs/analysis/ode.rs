use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::analysis::ode::{
    OdeModel, OdeSystem, RungeKutta4, TimeStepper, VecState,
};

#[derive(PartialEq, Clone, Copy, Debug)]
enum OdePreset {
    Exponential,
    HarmonicOscillator,
    LogisticGrowth,
}

impl OdePreset {
    fn name(&self) -> &'static str {
        match self {
            OdePreset::Exponential => "y' = k*y (Exponential)",
            OdePreset::HarmonicOscillator => "y'' = -k*y (Harmonic Oscillator)",
            OdePreset::LogisticGrowth => "y' = r*y*(1 - y/K) (Logistic Growth)",
        }
    }
}

// ----------------------------------------------------------------------------
// ODE System Definitions
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ExponentialOde {
    k: f64,
}

impl OdeSystem<VecState> for ExponentialOde {
    fn derivative(&self, _t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(_t, state, &mut out);
        out
    }

    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        out.0[0] = self.k * state.0[0];
    }
}

#[derive(Clone, Debug)]
struct HarmonicOscillatorOde {
    k: f64,
}

impl OdeSystem<VecState> for HarmonicOscillatorOde {
    fn derivative(&self, _t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(_t, state, &mut out);
        out
    }

    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        // state[0] is position (y), state[1] is velocity (y')
        out.0[0] = state.0[1];
        out.0[1] = -self.k * state.0[0];
    }
}

#[derive(Clone, Debug)]
struct LogisticGrowthOde {
    r: f64,
    cap_k: f64, // K
}

impl OdeSystem<VecState> for LogisticGrowthOde {
    fn derivative(&self, _t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(_t, state, &mut out);
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
    // Parameters
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    // Initial Conditions
    ic_y0: f64,
    ic_v0: f64,
    // Results
    time_series: Vec<f64>,
    y_series: Vec<f64>,
    v_series: Vec<f64>,
    diverged: bool,
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
        };
        tool.recalculate();
        tool
    }
}

impl OdeSolverTool {
    fn reset(&mut self) {
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
                let init_state = VecState(vec![self.ic_y0]);
                let dynamics = ExponentialOde { k: self.param_k };
                let solver = RungeKutta4::new(&init_state);
                let mut model = OdeModel::new(init_state, dynamics, solver);

                for i in 0..=num_steps {
                    let state = model.get_state();
                    if !state.0.iter().all(|v| v.is_finite()) {
                        self.diverged = true;
                        break;
                    }
                    let t = i as f64 * self.dt;
                    self.time_series.push(t);
                    self.y_series.push(state.0[0]);
                    model.step(self.dt);
                }
            }
            OdePreset::HarmonicOscillator => {
                let init_state = VecState(vec![self.ic_y0, self.ic_v0]);
                let dynamics = HarmonicOscillatorOde { k: self.param_k };
                let solver = RungeKutta4::new(&init_state);
                let mut model = OdeModel::new(init_state, dynamics, solver);

                for i in 0..=num_steps {
                    let state = model.get_state();
                    if !state.0.iter().all(|v| v.is_finite()) {
                        self.diverged = true;
                        break;
                    }
                    let t = i as f64 * self.dt;
                    self.time_series.push(t);
                    self.y_series.push(state.0[0]);
                    self.v_series.push(state.0[1]);
                    model.step(self.dt);
                }
            }
            OdePreset::LogisticGrowth => {
                let init_state = VecState(vec![self.ic_y0]);
                let dynamics = LogisticGrowthOde {
                    r: self.param_r,
                    cap_k: self.param_cap_k,
                };
                let solver = RungeKutta4::new(&init_state);
                let mut model = OdeModel::new(init_state, dynamics, solver);

                for i in 0..=num_steps {
                    let state = model.get_state();
                    if !state.0.iter().all(|v| v.is_finite()) {
                        self.diverged = true;
                        break;
                    }
                    let t = i as f64 * self.dt;
                    self.time_series.push(t);
                    self.y_series.push(state.0[0]);
                    model.step(self.dt);
                }
            }
        }
    }
}

impl InteractiveTool for OdeSolverTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "ODE Solvers"
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
                    if ui
                        .selectable_value(
                            &mut self.preset,
                            OdePreset::Exponential,
                            OdePreset::Exponential.name(),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    if ui
                        .selectable_value(
                            &mut self.preset,
                            OdePreset::HarmonicOscillator,
                            OdePreset::HarmonicOscillator.name(),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    if ui
                        .selectable_value(
                            &mut self.preset,
                            OdePreset::LogisticGrowth,
                            OdePreset::LogisticGrowth.name(),
                        )
                        .changed()
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
                        if ui
                            .add(egui::DragValue::new(&mut self.param_k).speed(0.1))
                            .changed()
                        {
                            changed = true;
                        }
                    });
                }
                OdePreset::HarmonicOscillator => {
                    ui.horizontal(|ui| {
                        ui.label("k (Spring Constant):");
                        if ui
                            .add(egui::DragValue::new(&mut self.param_k).speed(0.1))
                            .changed()
                        {
                            changed = true;
                        }
                    });
                }
                OdePreset::LogisticGrowth => {
                    ui.horizontal(|ui| {
                        ui.label("r (Growth Rate):");
                        if ui
                            .add(egui::DragValue::new(&mut self.param_r).speed(0.1))
                            .changed()
                        {
                            changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("K (Carrying Capacity):");
                        if ui
                            .add(egui::DragValue::new(&mut self.param_cap_k).speed(0.1))
                            .changed()
                        {
                            changed = true;
                        }
                    });
                }
            }

            ui.add_space(10.0);
            ui.heading("Initial Conditions");
            ui.horizontal(|ui| {
                ui.label("y(0):");
                if ui
                    .add(egui::DragValue::new(&mut self.ic_y0).speed(0.1))
                    .changed()
                {
                    changed = true;
                }
            });
            if self.preset == OdePreset::HarmonicOscillator {
                ui.horizontal(|ui| {
                    ui.label("y'(0):");
                    if ui
                        .add(egui::DragValue::new(&mut self.ic_v0).speed(0.1))
                        .changed()
                    {
                        changed = true;
                    }
                });
            }

            ui.add_space(10.0);
            ui.heading("Solver Settings");
            ui.horizontal(|ui| {
                ui.label("dt (Step Size):");
                if ui
                    .add(
                        egui::DragValue::new(&mut self.dt)
                            .speed(0.01)
                            .range(0.001..=1.0),
                    )
                    .changed()
                {
                    changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Total Time:");
                if ui
                    .add(
                        egui::DragValue::new(&mut self.total_time)
                            .speed(1.0)
                            .range(1.0..=100.0),
                    )
                    .changed()
                {
                    changed = true;
                }
            });
        });

        if changed {
            self.recalculate();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Solution Trajectory");

            if self.diverged {
                egui::Frame::NONE
                    .fill(egui::Color32::from_rgb(80, 20, 20))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::RED))
                    .inner_margin(8.0)
                    .corner_radius(4.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(
                                    "⚠ Warning: Solver diverged (non-finite state detected). Integration halted.",
                                )
                                .color(egui::Color32::YELLOW)
                                .strong(),
                            );
                            if ui.button("↻ Reset Parameters").clicked() {
                                self.reset();
                            }
                        });
                    });
                ui.add_space(8.0);
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

            let line_y =
                Line::new("y(t)", PlotPoints::new(plot_points_y)).color(egui::Color32::LIGHT_BLUE);

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
        });
    }
}

// [cite:graph_parameters_rust]


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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ode_divergence_detection() {
        let mut tool = OdeSolverTool {
            preset: OdePreset::Exponential,
            param_k: 1000.0,
            dt: 1.0,
            total_time: 100.0,
            ..Default::default()
        };
        tool.recalculate();

        assert!(tool.diverged, "Tool should detect divergence and set diverged flag");
        assert!(
            tool.y_series.iter().all(|y| y.is_finite()),
            "y_series must not contain non-finite values"
        );
        assert!(
            tool.v_series.iter().all(|v| v.is_finite()),
            "v_series must not contain non-finite values"
        );

        tool.reset();
        assert!(!tool.diverged, "Reset should clear diverged flag");
        assert!(!tool.y_series.is_empty(), "y_series should be populated after reset");
    }
}
