#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::{SimCommand, StateSnapshot};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use scientific_metadata::theory::ParameterConstraint;
use math_explorer::physics::quantum::{
    construct_1d_hamiltonian, evolve_state, gaussian_wavepacket, QuantumOperator, QuantumState,
};
use nalgebra::DVector;
use num_complex::Complex;
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

pub struct WaveData {
    pub time: f64,
    pub x_axis: Vec<f64>,
    pub prob_density: Vec<f64>,
    pub potential: Vec<f64>,
}

pub struct WaveUnified {
    pub(crate) psi: QuantumState,
    pub(crate) hamiltonian: QuantumOperator,
    pub(crate) potential: DVector<f64>,
    pub(crate) x_axis: Vec<f64>,
    pub(crate) time: f64,
    pub(crate) step_count: usize,
    pub(crate) n_points: usize,
    pub(crate) x_min: f64,
    pub(crate) x_max: f64,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<crate::async_sim::recovery::DiagnosticReport>,
    pub(crate) last_valid_prob_density: RefCell<Vec<f64>>,
}

impl UnifiedModel for WaveUnified {
    fn domain() -> &'static str {
        "quantum"
    }

    fn new(params: &HashMap<String, f64>) -> Self {
        let n_points = 200;
        let x_min = -5.0;
        let x_max = 5.0;

        let potential_type = *params.get("potential_type").unwrap_or(&0.0);
        let mut sim = Self {
            psi: QuantumState::new(DVector::from_element(n_points, Complex::new(0.0, 0.0))),
            hamiltonian: construct_1d_hamiltonian(&DVector::zeros(n_points), 1.0, 1.0, 1.0),
            potential: DVector::zeros(n_points),
            x_axis: Vec::new(),
            time: 0.0,
            step_count: 0,
            n_points,
            x_min,
            x_max,
            diverged: false,
            diagnostic_report: None,
            last_valid_prob_density: RefCell::new(Vec::new()),
        };
        sim.init_system(potential_type);
        sim
    }

    fn step(&mut self, _params: &HashMap<String, f64>) {
        if self.diverged {
            return;
        }
        self.step_count += 1;
        let dt = 0.05;

        // Check if current wave amplitudes or potential are finite before evolution
        let current_finite = self.psi.vector.iter().all(|c| c.re.is_finite() && c.im.is_finite());
        if !current_finite {
            self.trigger_divergence(dt, "psi(x)");
            return;
        }

        let next_psi = evolve_state(&self.psi, &self.hamiltonian, dt, 1.0);
        let norm = next_psi.norm();

        // Evaluate f64::is_finite() during every numerical step
        let is_finite = norm.is_finite()
            && norm > 0.0
            && next_psi.vector.iter().all(|c| c.re.is_finite() && c.im.is_finite());

        if !is_finite {
            self.trigger_divergence(dt, "psi(x)");
            return;
        }

        self.psi = next_psi.normalize();
        self.time += dt;
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let prob_density: Vec<f64> = if self.diverged && !self.last_valid_prob_density.borrow().is_empty() {
            self.last_valid_prob_density.borrow().clone()
        } else {
            let p: Vec<f64> = self.psi.probability_density().iter().copied().collect();
            let all_finite = p.iter().all(|val| val.is_finite());
            if all_finite {
                *self.last_valid_prob_density.borrow_mut() = p.clone();
                p
            } else if !self.last_valid_prob_density.borrow().is_empty() {
                self.last_valid_prob_density.borrow().clone()
            } else {
                vec![0.0; self.n_points]
            }
        };

        let structured_data = Box::new(WaveData {
            time: self.time,
            x_axis: self.x_axis.clone(),
            prob_density,
            potential: self.potential.iter().copied().collect(),
        }) as Box<dyn Any + Send>;

        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: Vec::new(),
            structured_data: Some(structured_data),
        }
    }

    fn process_command(&mut self, cmd: SimCommand, params: &HashMap<String, f64>) {
        match cmd {
            SimCommand::Reset => {
                let potential_type = *params.get("potential_type").unwrap_or(&0.0);
                self.init_system(potential_type);
                self.diverged = false;
                self.diagnostic_report = None;
            }
            SimCommand::Custom(ref action) if action == "ClampParameters" => {
                let potential_type = params.get("potential_type").copied().unwrap_or(0.0).clamp(0.0, 1.0);
                self.init_system(potential_type);
                self.diverged = false;
                self.diagnostic_report = None;
            }
            _ => {}
        }
    }

    fn diverged(&self) -> bool {
        self.diverged
    }

    fn diagnostic_report(&self) -> Option<crate::async_sim::recovery::DiagnosticReport> {
        self.diagnostic_report.clone()
    }

    fn default_parameters() -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("potential_type".to_string(), 0.0);
        map
    }

    fn clamp_parameters(params: &mut HashMap<String, f64>) {
        if let Some(p) = params.get_mut("potential_type") {
            *p = if !p.is_finite() { 0.0 } else { p.clamp(0.0, 1.0) };
        }
    }

    fn tool_source() -> &'static str {
        "WaveSimulator"
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        // 0.0 for Infinite Well, 1.0 for Harmonic Oscillator
        map.insert("potential_type".to_string(), ParameterConstraint { min: 0.0, max: 1.0, step: 1.0 });
        map
    }

    fn name() -> &'static str {
        "Wave Simulator"
    }

    fn custom_central_panel(ui: &mut egui::Ui, snapshot_opt: Option<&StateSnapshot>) -> bool {
        if let Some(snap) = snapshot_opt {
            if let Some(any_data) = &snap.structured_data {
                if let Some(wave_data) = any_data.downcast_ref::<WaveData>() {
                    ui.label(format!("Time: {:.2}", wave_data.time));
                    ui.label("White: |ψ(x)|² (Probability)");
                    ui.colored_label(
                        egui::Color32::from_rgb(100, 100, 255),
                        "Blue: V(x) (Potential)",
                    );

                    let mut psi_points = Vec::new();
                    let mut v_points = Vec::new();

                    for i in 0..wave_data.x_axis.len() {
                        let x = wave_data.x_axis[i];
                        let p = wave_data.prob_density[i];
                        let v = wave_data.potential[i];
                        psi_points.push([x, p]);
                        v_points.push([x, v * 0.2]); // Scale potential for display
                    }

                    Plot::new("quantum_plot")
                        .legend(Legend::default())
                        .x_axis_label("Position (x)")
                        .y_axis_label("Probability Density / Potential")
                        .view_aspect(2.0)
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new("Probability |ψ|²", PlotPoints::new(psi_points))
                                    .color(egui::Color32::WHITE)
                                    .width(2.0_f32),
                            );
                            plot_ui.line(
                                Line::new("Potential V(x) (scaled)", PlotPoints::new(v_points))
                                    .color(egui::Color32::from_rgb(100, 100, 255)),
                            );
                        });
                    return true;
                }
            }
        }
        false
    }
}

impl WaveUnified {
    fn init_system(&mut self, potential_type: f64) {
        let dx = (self.x_max - self.x_min) / (self.n_points as f64 - 1.0);
        self.x_axis = (0..self.n_points)
            .map(|i| self.x_min + i as f64 * dx)
            .collect();

        self.potential = DVector::zeros(self.n_points);
        for (i, &x) in self.x_axis.iter().enumerate() {
            let v = if potential_type < 0.5 {
                0.0 // InfiniteWell
            } else {
                0.5 * x * x // HarmonicOscillator
            };
            self.potential[i] = v;
        }

        self.hamiltonian = construct_1d_hamiltonian(&self.potential, dx, 1.0, 1.0);

        let x0 = -2.0;
        let k0 = 5.0;
        let sigma = 0.5;

        self.psi = gaussian_wavepacket(&self.x_axis, x0, k0, sigma);
        self.time = 0.0;
    }

    fn trigger_divergence(&mut self, dt: f64, var_name: &str) {
        self.diverged = true;
        let report = crate::async_sim::recovery::DiagnosticReport::new(
            self.time,
            self.step_count,
            var_name,
            format!(
                "Quantum wave packet dispersion failure: non-finite wave amplitude at t={:.4}s (step {})",
                self.time, self.step_count
            ),
        );
        crate::async_sim::recovery::emit_divergence_telemetry(
            "WaveSimulator",
            self.time,
            self.step_count,
            var_name,
            "Wave Simulator",
            dt,
        );
        self.diagnostic_report = Some(report);
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "WaveSimulator",
        domain: "quantum",
        tags: &[],
        build: || Box::new(UnifiedSimTool::<WaveUnified>::new()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for WaveUnified {
    fn theory_description(&self) -> String {
        "Quantum wave simulation [cite:quantum_mechanics]".to_string()
    }
    fn phonetic_description(&self) -> String {
        "Quantum wave simulation".to_string()
    }
    fn theory_citation(&self) -> String {
        "[cite:quantum_mechanics]".to_string()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}
