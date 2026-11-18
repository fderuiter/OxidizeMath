// [cite:quantum_mechanics]
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::{SimCommand, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use scientific_metadata::theory::ParameterConstraint;
use math_explorer::physics::quantum::{
    construct_1d_hamiltonian, evolve_state, gaussian_wavepacket, QuantumOperator, QuantumState,
};
use nalgebra::DVector;
use num_complex::Complex;
use std::any::Any;
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
    hamiltonian: QuantumOperator,
    potential: DVector<f64>,
    x_axis: Vec<f64>,
    time: f64,
    n_points: usize,
    x_min: f64,
    x_max: f64,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) step_count: usize,
}

impl FiniteStateGuard for WaveUnified {
    fn is_finite_state(&self) -> bool {
        if self.diverged {
            return false;
        }
        self.psi.vector.iter().all(|c| c.re.is_finite() && c.im.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic_report.as_ref()
    }

    fn diverged(&self) -> bool {
        self.diverged || !FiniteStateGuard::is_finite_state(self)
    }

    fn reset_parameters(&mut self) {
        self.diverged = false;
        self.diagnostic_report = None;
        self.step_count = 0;
    }
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
            n_points,
            x_min,
            x_max,
            diverged: false,
            diagnostic_report: None,
            step_count: 0,
        };
        sim.init_system(potential_type);
        sim
    }

    fn step(&mut self, params: &HashMap<String, f64>) {
        if self.diverged {
            return;
        }
        let dt = *params.get("dt").unwrap_or(&0.05);
        if !dt.is_finite() || dt <= 0.0 {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.time,
                self.step_count,
                "dt",
                format!("Invalid non-finite time step dt: {}", dt),
            ));
            return;
        }
        self.step_count += 1;
        self.psi = evolve_state(&self.psi, &self.hamiltonian, dt, 1.0);
        self.time += dt;

        if !self.psi.vector.iter().all(|c| c.re.is_finite() && c.im.is_finite()) {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.time,
                self.step_count,
                "psi",
                "Non-finite wave packet amplitude detected during time evolution.",
            ));
            return;
        }

        self.psi = self.psi.normalize();

        if !self.psi.vector.iter().all(|c| c.re.is_finite() && c.im.is_finite()) {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.time,
                self.step_count,
                "psi_norm",
                "Wave function normalization diverged: non-finite norm encountered.",
            ));
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let prob_density = self.psi.probability_density();
        let safe_prob: Vec<f64> = prob_density
            .iter()
            .map(|&p| if p.is_finite() { p } else { 0.0 })
            .collect();
        let safe_potential: Vec<f64> = self.potential
            .iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect();

        let structured_data = Box::new(WaveData {
            time: self.time,
            x_axis: self.x_axis.clone(),
            prob_density: safe_prob,
            potential: safe_potential,
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
        if let SimCommand::Reset = cmd {
            self.diverged = false;
            self.diagnostic_report = None;
            self.step_count = 0;
            let potential_type = *params.get("potential_type").unwrap_or(&0.0);
            self.init_system(potential_type);
        }
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        // 0.0 for Infinite Well, 1.0 for Harmonic Oscillator
        map.insert("potential_type".to_string(), ParameterConstraint { min: 0.0, max: 1.0, step: 1.0 });
        map.insert("dt".to_string(), ParameterConstraint { min: 0.001, max: 0.1, step: 0.005 });
        map
    }

    fn parameter_label(name: &str) -> String {
        match name {
            "potential_type" => "Potential Profile".to_string(),
            "dt" => "Time Step (Δt)".to_string(),
            other => other.to_string(),
        }
    }

    fn parameter_choices(name: &str) -> &'static [(&'static str, f64)] {
        match name {
            "potential_type" => &[
                ("Infinite Potential Well", 0.0),
                ("Harmonic Oscillator", 1.0),
            ],
            _ => &[],
        }
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
        crate::async_sim::ui::render_empty_plot_guidance(
            ui,
            "Wavepacket ready — click ▶ Run in the side panel to evolve ψ(x, t).",
        );
        true
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
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "WaveSimulator",
        display_name: "Wave Simulator",
        domain: "quantum",
        tags: &[],
        build: || Box::new(UnifiedSimTool::<WaveUnified>::new()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for WaveUnified {
    fn theory_description(&self) -> String {
        "Time-dependent 1D Schrödinger wave packet evolution $i\\hbar \\partial_t \\psi(x,t) = [-\\frac{\\hbar^2}{2m}\\partial_x^2 + V(x)]\\psi(x,t)$ for infinite well and harmonic oscillator potentials.".to_string()
    }
    fn phonetic_description(&self) -> String {
        "Time-dependent one-dimensional Schrodinger wave packet simulation".to_string()
    }
    fn theory_citation(&self) -> String {
        "Schrödinger, E. (1926). An Undulatory Theory of the Mechanics of Atoms and Molecules. Physical Review, 28(6), 1049-1070; Griffiths, D. J., & Schroeter, D. F. (2018). Introduction to Quantum Mechanics (3rd ed.). Cambridge University Press.".to_string()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "potential_type".to_string(),
            "Selects between an infinite square well (V = 0 inside walls) and a quadratic harmonic potential V(x) = 0.5 x².".to_string(),
        );
        map.insert(
            "dt".to_string(),
            "Integration time step Δt controlling unitary wave-packet propagation per tick.".to_string(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Schr\u00f6dinger Wave Packet",
    "domain": "quantum",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "potential_type": 1.0,
    "dt": 0.05
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "wave_sim_starter",
        title: "Schrödinger Wave Packet",
        description: "iℏ·∂ψ/∂t = -ℏ²/(2m)∇²ψ + V·ψ: 1D Gaussian wave packet evolution in infinite-well and harmonic oscillator potentials.",
        category: "Wave Mechanics",
        icon: "🌊",
        domain: "quantum",
        tool_name: Some("WaveSimulator"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
