// [cite:quantum_mechanics]
use super::wave_sim_sculpt::{
    apply_discrete_brush_to_potential, apply_stroke_to_potential_slice, build_base_potential,
    detect_barrier_split_index, draw_wave_brush_controls, record_wave_stroke_sparse,
    render_wave_central_panel, stamp_double_barrier_into,
};
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::async_sim::{SculptStatePayload, SimCommand, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    apply_sparse_potential, encode_sparse_potential, BrushMode, BrushStroke, SpatialBrushKit,
};
use eframe::egui;
use math_explorer::physics::quantum::{
    construct_1d_hamiltonian, construct_1d_hamiltonian_into, evolve_state_in_place,
    gaussian_wavepacket, state_norm_squared, time_evolution_operator, transmission_probability,
    QuantumOperator, QuantumState,
};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use scientific_metadata::theory::ParameterConstraint;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub(crate) const WAVE_N_POINTS: usize = 200;
pub(crate) const WAVE_X_MIN: f64 = -5.0;
pub(crate) const WAVE_X_MAX: f64 = 5.0;

pub struct WaveData {
    pub time: f64,
    pub x_axis: Vec<f64>,
    pub prob_density: Vec<f64>,
    pub potential: Vec<f64>,
    pub transmission_prob: f64,
    pub norm_sq: f64,
    pub barrier_split_x: f64,
}

pub struct WaveUnified {
    pub(crate) psi: QuantumState,
    pub(crate) hamiltonian: QuantumOperator,
    pub(crate) propagator: QuantumOperator,
    pub(crate) cached_dt: Option<f64>,
    pub(crate) scratch: DVector<Complex<f64>>,
    pub(crate) potential: DVector<f64>,
    pub(crate) base_potential: Vec<f64>,
    pub(crate) x_axis: Vec<f64>,
    pub(crate) time: f64,
    pub(crate) n_points: usize,
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
        self.psi
            .vector
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite())
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
        let potential_type = *params.get("potential_type").unwrap_or(&0.0);
        let mut sim = Self {
            psi: QuantumState::new(DVector::from_element(
                WAVE_N_POINTS,
                Complex::new(0.0, 0.0),
            )),
            hamiltonian: QuantumOperator::new(DMatrix::zeros(WAVE_N_POINTS, WAVE_N_POINTS)),
            propagator: QuantumOperator::new(DMatrix::zeros(WAVE_N_POINTS, WAVE_N_POINTS)),
            cached_dt: None,
            scratch: DVector::from_element(WAVE_N_POINTS, Complex::new(0.0, 0.0)),
            potential: DVector::zeros(WAVE_N_POINTS),
            base_potential: vec![0.0; WAVE_N_POINTS],
            x_axis: Vec::new(),
            time: 0.0,
            n_points: WAVE_N_POINTS,
            x_min: WAVE_X_MIN,
            x_max: WAVE_X_MAX,
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
                format!("Invalid non-finite time step dt: {dt}"),
            ));
            return;
        }
        self.refresh_propagator(dt);
        self.step_count += 1;
        evolve_state_in_place(&mut self.psi, &self.propagator, &mut self.scratch);
        self.time += dt;

        if !self
            .psi
            .vector
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite())
        {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.time,
                self.step_count,
                "psi",
                "Non-finite wave packet amplitude detected during time evolution.",
            ));
            return;
        }

        self.psi.normalize_in_place();

        if !self
            .psi
            .vector
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite())
        {
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
        let safe_potential: Vec<f64> = self
            .potential
            .iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect();
        let split_idx = detect_barrier_split_index(&safe_potential, &self.base_potential);
        let split_x = self.x_axis.get(split_idx).copied().unwrap_or(0.0);
        let trans = transmission_probability(&self.psi, split_idx);
        let norm_sq = state_norm_squared(&self.psi);

        let structured_data = Box::new(WaveData {
            time: self.time,
            x_axis: self.x_axis.clone(),
            prob_density: safe_prob,
            potential: safe_potential,
            transmission_prob: trans,
            norm_sq,
            barrier_split_x: split_x,
        }) as Box<dyn Any + Send>;

        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(RwLock::new(Vec::new())),
            custom_data: Vec::new(),
            structured_data: Some(structured_data),
        }
    }

    fn process_command(&mut self, cmd: SimCommand, params: &HashMap<String, f64>) {
        let dt = *params.get("dt").unwrap_or(&0.05);
        match cmd {
            SimCommand::Reset => {
                self.diverged = false;
                self.diagnostic_report = None;
                self.step_count = 0;
                let potential_type = *params.get("potential_type").unwrap_or(&0.0);
                self.init_system(potential_type);
            }
            SimCommand::ApplyBrush {
                cx,
                cy: _,
                r,
                is_obstacle,
            } => {
                apply_discrete_brush_to_potential(
                    self.potential.as_mut_slice(),
                    &self.base_potential,
                    cx,
                    r,
                    is_obstacle,
                );
                self.rebuild_operators(dt);
            }
            SimCommand::ApplyBrushStroke(ref stroke) => {
                apply_stroke_to_potential_slice(
                    self.potential.as_mut_slice(),
                    &self.base_potential,
                    self.x_min,
                    self.x_max,
                    stroke,
                );
                self.rebuild_operators(dt);
            }
            SimCommand::ClearObstacles => {
                self.potential.as_mut_slice().copy_from_slice(&self.base_potential);
                self.rebuild_operators(dt);
            }
            SimCommand::Custom(ref s) if s == "Clear Potential Edits" => {
                self.potential.as_mut_slice().copy_from_slice(&self.base_potential);
                self.rebuild_operators(dt);
            }
            SimCommand::Custom(ref s) if s == "stamp_double_barrier" => {
                stamp_double_barrier_into(self.potential.as_mut_slice(), &self.x_axis);
                self.rebuild_operators(dt);
            }
            SimCommand::Custom(ref s) if s == "relaunch_wavepacket" => {
                self.psi = gaussian_wavepacket(&self.x_axis, -2.0, 5.0, 0.5);
                self.time = 0.0;
                self.step_count = 0;
            }
            _ => {}
        }
    }

    fn custom_actions() -> Vec<&'static str> {
        vec!["Clear Potential Edits"]
    }

    fn supports_spatial_brush() -> bool {
        true
    }

    fn default_brush_kit() -> SpatialBrushKit {
        SpatialBrushKit::new(BrushMode::PotentialBarrier, 0.25)
            .with_radius_bounds(0.10, 1.20, 0.05)
            .with_cursor_step(0.20, 0.30)
    }

    fn draw_brush_controls(
        ui: &mut egui::Ui,
        brush_kit: &mut SpatialBrushKit,
        params: &Arc<RwLock<HashMap<String, f64>>>,
        sculpt_state: &Arc<RwLock<SculptStatePayload>>,
        snapshot: Option<&StateSnapshot>,
    ) -> Vec<SimCommand> {
        draw_wave_brush_controls(ui, brush_kit, params, sculpt_state, snapshot)
    }

    fn record_ui_brush_stroke(
        stroke: &BrushStroke,
        params: &HashMap<String, f64>,
        sculpt_state: &mut SculptStatePayload,
    ) {
        record_wave_stroke_sparse(stroke, params, sculpt_state);
    }

    fn apply_sculpt_payload(&mut self, payload: &SculptStatePayload, params: &HashMap<String, f64>) {
        self.potential.as_mut_slice().copy_from_slice(&self.base_potential);
        if !payload.sparse_potential.is_empty()
            && apply_sparse_potential(self.potential.as_mut_slice(), &payload.sparse_potential)
                .is_ok()
        {
            let dt = *params.get("dt").unwrap_or(&0.05);
            self.rebuild_operators(dt);
        }
    }

    fn export_sculpt_payload(&self, _params: &HashMap<String, f64>) -> SculptStatePayload {
        SculptStatePayload {
            rle_mask: Vec::new(),
            sparse_potential: encode_sparse_potential(
                &self.base_potential,
                self.potential.as_slice(),
                1e-6,
            ),
        }
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        map.insert(
            "potential_type".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 1.0,
            },
        );
        map.insert(
            "dt".to_string(),
            ParameterConstraint {
                min: 0.001,
                max: 0.1,
                step: 0.005,
            },
        );
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
        let mut kit = Self::default_brush_kit();
        render_wave_central_panel(ui, snapshot_opt, &mut kit).0
    }

    fn custom_central_panel_with_brush(
        ui: &mut egui::Ui,
        snapshot: Option<&StateSnapshot>,
        brush_kit: &mut SpatialBrushKit,
    ) -> (bool, Vec<BrushStroke>) {
        render_wave_central_panel(ui, snapshot, brush_kit)
    }
}

impl WaveUnified {
    pub(crate) fn dx(&self) -> f64 {
        (self.x_max - self.x_min) / (self.n_points.max(2) as f64 - 1.0)
    }

    pub(crate) fn refresh_propagator(&mut self, dt: f64) {
        if self.cached_dt != Some(dt) {
            self.propagator = time_evolution_operator(&self.hamiltonian, dt, 1.0);
            self.cached_dt = Some(dt);
        }
    }

    /// Rebuilds the 1D finite-difference Hamiltonian in place and invalidates the cached propagator $U(\Delta t)$.
    pub(crate) fn rebuild_operators(&mut self, _dt: f64) {
        construct_1d_hamiltonian_into(&self.potential, self.dx(), 1.0, 1.0, &mut self.hamiltonian);
        self.cached_dt = None;
    }

    pub(crate) fn init_system(&mut self, potential_type: f64) {
        let dx = self.dx();
        self.x_axis = (0..self.n_points)
            .map(|i| self.x_min + i as f64 * dx)
            .collect();
        self.base_potential = build_base_potential(potential_type, &self.x_axis);
        self.potential = DVector::from_vec(self.base_potential.clone());

        self.hamiltonian = construct_1d_hamiltonian(&self.potential, dx, 1.0, 1.0);
        self.propagator = QuantumOperator::new(DMatrix::zeros(self.n_points, self.n_points));
        self.cached_dt = None;
        self.scratch = DVector::from_element(self.n_points, Complex::new(0.0, 0.0));
        self.psi = gaussian_wavepacket(&self.x_axis, -2.0, 5.0, 0.5);
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
        "Time-dependent 1D Schrödinger wave packet evolution $i\\hbar \\partial_t \\psi(x,t) = [-\\frac{\\hbar^2}{2m}\\partial_x^2 + V(x)]\\psi(x,t)$ for infinite well, harmonic oscillator, and sculpted resonant-tunneling potentials.".to_string()
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
            "Selects baseline potential profile V(x) (infinite square well, harmonic oscillator, single/double finite barrier, or potential step).".to_string(),
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
    "dt": 0.02
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
