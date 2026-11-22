#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:quantum_mechanics]

mod canvas;
mod presets;
#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

use crate::async_sim::unified::{UnifiedCanvasContext, UnifiedModel, UnifiedSimTool};
use crate::async_sim::{SimCommand, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::url_state::CUSTOM_POTENTIAL_POINTS;
use eframe::egui;
use math_explorer::physics::quantum::{
    MAX_SCULPT_POTENTIAL, MIN_SCULPT_POTENTIAL, PotentialPreset1D,
    apply_bresenham_potential_stroke, build_1d_potential_preset, construct_1d_hamiltonian,
    evolve_state_in_place, gaussian_wavepacket, partition_region_probabilities,
    time_evolution_operator, QuantumOperator, QuantumState,
};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use scientific_metadata::theory::ParameterConstraint;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) const GRID_POINTS: usize = CUSTOM_POTENTIAL_POINTS;
pub(crate) const X_MIN: f64 = -5.0;
pub(crate) const X_MAX: f64 = 5.0;
pub(crate) const DEFAULT_X0: f64 = -2.0;
pub(crate) const DEFAULT_K0: f64 = 5.0;
pub(crate) const DEFAULT_SIGMA: f64 = 0.5;
pub(crate) const DEFAULT_DT: f64 = 0.05;
pub(crate) const POTENTIAL_DISPLAY_SCALE: f64 = 0.01;
pub(crate) const CUSTOM_POTENTIAL_KEY: &str = "custom_potential";

#[allow(missing_docs)]
#[derive(Clone)]
pub struct WaveData {
    pub time: f64,
    pub x_axis: Vec<f64>,
    pub prob_density: Vec<f64>,
    pub potential: Vec<f64>,
    pub p_transmitted: f64,
    pub p_reflected: f64,
    pub barrier_idx: usize,
}

#[allow(missing_docs)]
pub struct WaveUnified {
    pub(crate) psi: QuantumState,
    pub(crate) hamiltonian: QuantumOperator,
    pub(crate) propagator: QuantumOperator,
    pub(crate) cached_dt: Option<f64>,
    pub(crate) hamiltonian_dirty: bool,
    pub(crate) propagator_rebuilds: usize,
    pub(crate) scratch: DVector<Complex<f64>>,
    pub(crate) potential: DVector<f64>,
    pub(crate) x_axis: Vec<f64>,
    pub(crate) time: f64,
    pub(crate) n_points: usize,
    pub(crate) x_min: f64,
    pub(crate) x_max: f64,
    pub(crate) x0: f64,
    pub(crate) k0: f64,
    pub(crate) sigma: f64,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) step_count: usize,
}

pub(crate) fn clamp_wavepacket_params(x0: f64, k0: f64, sigma: f64) -> (f64, f64, f64) {
    let sx0 = if x0.is_finite() {
        x0.clamp(-4.5, 4.5)
    } else {
        DEFAULT_X0
    };
    let sk0 = if k0.is_finite() {
        k0.clamp(-15.0, 15.0)
    } else {
        DEFAULT_K0
    };
    let ssigma = if sigma.is_finite() {
        sigma.clamp(0.2, 2.0)
    } else {
        DEFAULT_SIGMA
    };
    (sx0, sk0, ssigma)
}

pub(crate) fn detect_barrier_index(potential: &DVector<f64>) -> usize {
    let n = potential.len();
    if n < 3 {
        return 0;
    }
    let edge_ref = potential[0].max(potential[n - 1]);
    let mut max_interior = f64::NEG_INFINITY;
    for i in 1..(n - 1) {
        let v = potential[i];
        if v.is_finite() && v > max_interior {
            max_interior = v;
        }
    }
    if max_interior > edge_ref + 0.5 {
        for i in (1..(n - 1)).rev() {
            if potential[i].is_finite() && potential[i] >= max_interior - 0.25 {
                return i;
            }
        }
    }
    n / 2
}

impl FiniteStateGuard for WaveUnified {
    fn is_finite_state(&self) -> bool {
        !self.diverged
            && self
                .psi
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

    fn name() -> &'static str {
        "Wave Simulator"
    }

    fn new(params: &HashMap<String, f64>) -> Self {
        let potential_type = params.get("potential_type").copied().unwrap_or(0.0);
        let x0 = params.get("x0").copied().unwrap_or(DEFAULT_X0);
        let k0 = params.get("k0").copied().unwrap_or(DEFAULT_K0);
        let sigma = params.get("sigma").copied().unwrap_or(DEFAULT_SIGMA);
        let mut sim = Self {
            psi: QuantumState::new(DVector::from_element(GRID_POINTS, Complex::new(0.0, 0.0))),
            hamiltonian: QuantumOperator::new(DMatrix::zeros(GRID_POINTS, GRID_POINTS)),
            propagator: QuantumOperator::new(DMatrix::zeros(GRID_POINTS, GRID_POINTS)),
            cached_dt: None,
            hamiltonian_dirty: false,
            propagator_rebuilds: 0,
            scratch: DVector::from_element(GRID_POINTS, Complex::new(0.0, 0.0)),
            potential: DVector::zeros(GRID_POINTS),
            x_axis: Vec::new(),
            time: 0.0,
            n_points: GRID_POINTS,
            x_min: X_MIN,
            x_max: X_MAX,
            x0: DEFAULT_X0,
            k0: DEFAULT_K0,
            sigma: DEFAULT_SIGMA,
            diverged: false,
            diagnostic_report: None,
            step_count: 0,
        };
        sim.init_system_with_packet(potential_type, x0, k0, sigma);
        sim
    }

    fn step(&mut self, params: &HashMap<String, f64>) {
        if self.diverged {
            return;
        }
        let dt = params.get("dt").copied().unwrap_or(DEFAULT_DT);
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
        let barrier_idx = detect_barrier_index(&self.potential);
        let (p_transmitted, p_reflected) =
            partition_region_probabilities(&self.psi, barrier_idx);

        let structured_data = Box::new(WaveData {
            time: self.time,
            x_axis: self.x_axis.clone(),
            prob_density: safe_prob,
            potential: safe_potential,
            p_transmitted,
            p_reflected,
            barrier_idx,
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
        self.handle_sim_command(cmd, params);
    }

    fn parameters() -> HashMap<String, ParameterConstraint> {
        presets::wave_parameters()
    }

    fn default_parameter(name: &str, constraint: &ParameterConstraint) -> f64 {
        presets::wave_default_parameter(name, constraint)
    }

    fn is_wavepacket_parameter(name: &str) -> bool {
        matches!(name, "x0" | "k0" | "sigma")
    }

    fn clears_custom_series_on_action(action: &str) -> bool {
        action == "Reset Potential"
    }

    fn custom_series_schema() -> &'static [(&'static str, usize, f64, f64)] {
        &[(
            CUSTOM_POTENTIAL_KEY,
            CUSTOM_POTENTIAL_POINTS,
            MIN_SCULPT_POTENTIAL,
            MAX_SCULPT_POTENTIAL,
        )]
    }

    fn apply_custom_series(
        &mut self,
        series: &HashMap<String, Vec<f64>>,
        _params: &HashMap<String, f64>,
    ) {
        if let Some(values) = series.get(CUSTOM_POTENTIAL_KEY) {
            if values.len() == self.n_points && values.iter().all(|v| v.is_finite()) {
                for (i, &v) in values.iter().enumerate() {
                    self.potential[i] = v.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
                }
                self.hamiltonian =
                    construct_1d_hamiltonian(&self.potential, self.grid_dx(), 1.0, 1.0);
                self.hamiltonian_dirty = false;
                self.cached_dt = None;
            }
        }
    }

    fn parameter_label(name: &str) -> String {
        presets::wave_parameter_label(name)
    }

    fn parameter_choices(name: &str) -> &'static [(&'static str, f64)] {
        presets::wave_parameter_choices(name)
    }

    fn presets() -> Vec<(&'static str, HashMap<String, f64>)> {
        presets::wave_sidebar_presets()
    }

    fn custom_actions() -> Vec<&'static str> {
        vec!["Reset Potential"]
    }

    fn custom_central_panel(ui: &mut egui::Ui, snapshot_opt: Option<&StateSnapshot>) -> bool {
        canvas::render_readonly_wave_panel(ui, snapshot_opt)
    }

    fn custom_central_panel_interactive(
        ui: &mut egui::Ui,
        ctx: &mut UnifiedCanvasContext<'_>,
    ) -> bool {
        canvas::render_interactive_wave_panel(ui, ctx)
    }
}

impl WaveUnified {
    pub(crate) fn grid_dx(&self) -> f64 {
        if self.n_points > 1 {
            (self.x_max - self.x_min) / (self.n_points as f64 - 1.0)
        } else {
            1.0
        }
    }

    pub(crate) fn refresh_propagator(&mut self, dt: f64) {
        if self.hamiltonian_dirty {
            self.hamiltonian = construct_1d_hamiltonian(&self.potential, self.grid_dx(), 1.0, 1.0);
            self.hamiltonian_dirty = false;
            self.cached_dt = None;
        }
        if self.cached_dt != Some(dt) {
            self.propagator = time_evolution_operator(&self.hamiltonian, dt, 1.0);
            self.cached_dt = Some(dt);
            self.propagator_rebuilds = self.propagator_rebuilds.saturating_add(1);
        }
    }

    #[cfg(test)]
    pub(crate) fn init_system(&mut self, potential_type: f64) {
        self.init_system_with_packet(potential_type, self.x0, self.k0, self.sigma);
    }

    pub(crate) fn init_system_with_packet(
        &mut self,
        potential_type: f64,
        x0: f64,
        k0: f64,
        sigma: f64,
    ) {
        let dx = self.grid_dx();
        self.x_axis = (0..self.n_points)
            .map(|i| self.x_min + i as f64 * dx)
            .collect();
        self.potential =
            build_1d_potential_preset(PotentialPreset1D::from_code(potential_type), &self.x_axis);
        self.hamiltonian = construct_1d_hamiltonian(&self.potential, dx, 1.0, 1.0);
        self.propagator = QuantumOperator::new(DMatrix::zeros(self.n_points, self.n_points));
        self.cached_dt = None;
        self.hamiltonian_dirty = false;
        self.scratch = DVector::from_element(self.n_points, Complex::new(0.0, 0.0));

        let (sx0, sk0, ssigma) = clamp_wavepacket_params(x0, k0, sigma);
        self.x0 = sx0;
        self.k0 = sk0;
        self.sigma = ssigma;
        self.psi = gaussian_wavepacket(&self.x_axis, sx0, sk0, ssigma);
        self.time = 0.0;
    }

    fn apply_brush_stroke(&mut self, cx: i32, cy: i32, r: i32, is_obstacle: bool) {
        if self.n_points == 0 {
            return;
        }
        let radius = r.max(0);
        let last = (self.n_points - 1) as i32;
        let i0 = (cx - radius).clamp(0, last) as usize;
        let i1 = (cx + radius).clamp(0, last) as usize;
        let v = if !is_obstacle {
            0.0
        } else if cy == 0 {
            14.0
        } else {
            (cy as f64).clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL)
        };
        apply_bresenham_potential_stroke(&mut self.potential, i0, v, i1, v);
        self.hamiltonian_dirty = true;
        self.cached_dt = None;
    }

    fn handle_sim_command(&mut self, cmd: SimCommand, params: &HashMap<String, f64>) {
        match cmd {
            SimCommand::Reset => {
                self.diverged = false;
                self.diagnostic_report = None;
                self.step_count = 0;
                let potential_type = params.get("potential_type").copied().unwrap_or(0.0);
                let x0 = params.get("x0").copied().unwrap_or(DEFAULT_X0);
                let k0 = params.get("k0").copied().unwrap_or(DEFAULT_K0);
                let sigma = params.get("sigma").copied().unwrap_or(DEFAULT_SIGMA);
                self.init_system_with_packet(potential_type, x0, k0, sigma);
            }
            SimCommand::SculptStroke1D { i0, v0, i1, v1 } if v0.is_finite() && v1.is_finite() => {
                apply_bresenham_potential_stroke(&mut self.potential, i0, v0, i1, v1);
                self.hamiltonian_dirty = true;
                self.cached_dt = None;
            }
            SimCommand::ApplyBrush {
                cx,
                cy,
                r,
                is_obstacle,
            } => self.apply_brush_stroke(cx, cy, r, is_obstacle),
            SimCommand::SetWavepacket { x0, k0, sigma } => {
                let (sx0, sk0, ssigma) = clamp_wavepacket_params(x0, k0, sigma);
                self.x0 = sx0;
                self.k0 = sk0;
                self.sigma = ssigma;
                self.psi = gaussian_wavepacket(&self.x_axis, sx0, sk0, ssigma);
                self.time = 0.0;
                self.step_count = 0;
                self.diverged = false;
                self.diagnostic_report = None;
            }
            SimCommand::ClearObstacles => self.restore_preset_potential(params),
            SimCommand::Custom(ref action) if action == "Reset Potential" => {
                self.restore_preset_potential(params);
            }
            _ => {}
        }
    }

    fn restore_preset_potential(&mut self, params: &HashMap<String, f64>) {
        let potential_type = params.get("potential_type").copied().unwrap_or(0.0);
        self.potential =
            build_1d_potential_preset(PotentialPreset1D::from_code(potential_type), &self.x_axis);
        self.hamiltonian_dirty = true;
        self.cached_dt = None;
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "WaveSimulator",
        display_name: "Wave Simulator",
        domain: "quantum",
        tags: &["tunneling", "sculpting", "wavepacket"],
        build: || Box::new(UnifiedSimTool::<WaveUnified>::new()),
    }
}
