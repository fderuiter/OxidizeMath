#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::observables::{compute_tunneling_from_psi, ScatterLatchState};
use super::presets::{
    barrier_interval_for_preset, clamp_wavepacket_params, detect_barrier_index,
    evaluate_potential_grid, DEFAULT_DT, GRID_POINTS, GRID_X_MAX, GRID_X_MIN,
};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_explorer::physics::quantum::{
    apply_bresenham_potential_stroke, construct_1d_hamiltonian, evolve_state_in_place,
    gaussian_wavepacket, partition_region_probabilities, time_evolution_operator, QuantumOperator,
    QuantumState, MAX_SCULPT_POTENTIAL, MIN_SCULPT_POTENTIAL,
};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use std::any::Any;
use std::sync::{Arc, RwLock};

pub use super::observables::{
    compute_tunneling_observables, TunnelingObservables, WaveData, WaveSimConfig,
};

/// Background worker runner evolving the 1D time-dependent Schrödinger equation with zero per-step allocations.
pub struct WaveRunner {
    config: Arc<RwLock<WaveSimConfig>>,
    pub(crate) psi: QuantumState,
    pub(crate) hamiltonian: QuantumOperator,
    pub(crate) propagator: QuantumOperator,
    pub(crate) cached_dt: Option<f64>,
    pub(crate) hamiltonian_dirty: bool,
    pub(crate) propagator_rebuilds: usize,
    pub(crate) scratch: DVector<Complex<f64>>,
    pub(crate) potential: DVector<f64>,
    pub(crate) x_axis: Vec<f64>,
    pub(crate) x_left: f64,
    pub(crate) x_right: f64,
    pub(crate) barrier_idx: usize,
    pub(crate) p_transmitted: f64,
    pub(crate) p_reflected: f64,
    pub(crate) x0: f64,
    pub(crate) k0: f64,
    pub(crate) sigma: f64,
    pub(crate) kinetic_energy: f64,
    pub(crate) time: f64,
    pub(crate) n_points: usize,
    pub(crate) x_min: f64,
    pub(crate) x_max: f64,
    steps_per_frame: usize,
    observables: TunnelingObservables,
    latch: ScatterLatchState,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) step_count: usize,
    prev_mode_phases: [f64; 8],
    sonification: crate::audio::SonificationFrame,
}

impl WaveRunner {
    /// Creates and initializes a new 1D Schrödinger runner from shared configuration.
    #[must_use]
    pub fn new(config: Arc<RwLock<WaveSimConfig>>) -> Self {
        let cfg = config.read().unwrap_or_else(|e| e.into_inner()).clone();
        let mut runner = Self {
            config,
            psi: QuantumState::new(DVector::from_element(GRID_POINTS, Complex::new(0.0, 0.0))),
            hamiltonian: QuantumOperator::new(DMatrix::zeros(GRID_POINTS, GRID_POINTS)),
            propagator: QuantumOperator::new(DMatrix::zeros(GRID_POINTS, GRID_POINTS)),
            cached_dt: None,
            hamiltonian_dirty: false,
            propagator_rebuilds: 0,
            scratch: DVector::from_element(GRID_POINTS, Complex::new(0.0, 0.0)),
            potential: DVector::zeros(GRID_POINTS),
            x_axis: Vec::new(),
            x_left: -0.25,
            x_right: 0.25,
            barrier_idx: GRID_POINTS / 2,
            p_transmitted: 0.0,
            p_reflected: 1.0,
            x0: cfg.x0,
            k0: cfg.k0,
            sigma: cfg.sigma,
            kinetic_energy: 0.5 * cfg.k0 * cfg.k0,
            time: 0.0,
            n_points: GRID_POINTS,
            x_min: GRID_X_MIN,
            x_max: GRID_X_MAX,
            steps_per_frame: cfg.steps_per_frame.max(1),
            observables: TunnelingObservables::default(),
            latch: ScatterLatchState::default(),
            diverged: false,
            diagnostic_report: None,
            step_count: 0,
            prev_mode_phases: [0.0; 8],
            sonification: crate::audio::SonificationFrame::silent(),
        };
        runner.init_from_config(&cfg);
        runner
    }

    pub(crate) fn grid_dx(&self) -> f64 {
        if self.n_points > 1 {
            (self.x_max - self.x_min) / (self.n_points as f64 - 1.0)
        } else {
            1.0
        }
    }

    /// Rebuilds the spatial grid, clamped potential, Hamiltonian, and initial Gaussian wavepacket.
    pub fn init_from_config(&mut self, cfg: &WaveSimConfig) {
        let n = self.n_points.max(2);
        let dx = (self.x_max - self.x_min) / (n as f64 - 1.0);
        self.x_axis = (0..n).map(|i| self.x_min + i as f64 * dx).collect();
        (self.x_left, self.x_right) = barrier_interval_for_preset(cfg.preset, cfg.width, dx);
        let expr = &cfg.custom_expr;
        self.potential =
            evaluate_potential_grid(cfg.preset, &self.x_axis, cfg.v0, cfg.width, dx, expr);
        if let Some(ref sculpted) = cfg.custom_potential {
            if sculpted.len() == n {
                for (i, &v) in sculpted.iter().enumerate() {
                    if v.is_finite() {
                        self.potential[i] = v.clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL);
                    }
                }
            }
        }
        self.barrier_idx = detect_barrier_index(&self.potential);
        self.hamiltonian = construct_1d_hamiltonian(&self.potential, dx, 1.0, 1.0);
        self.propagator = QuantumOperator::new(DMatrix::zeros(n, n));
        self.cached_dt = None;
        self.hamiltonian_dirty = false;
        self.scratch = DVector::from_element(n, Complex::new(0.0, 0.0));

        let (sx0, sk0, ssigma) = clamp_wavepacket_params(cfg.x0, cfg.k0, cfg.sigma);
        self.x0 = sx0;
        self.k0 = sk0;
        self.sigma = ssigma;
        self.kinetic_energy = 0.5 * sk0 * sk0;
        self.steps_per_frame = cfg.steps_per_frame.max(1);
        self.psi = gaussian_wavepacket(&self.x_axis, sx0, sk0, ssigma);
        self.time = 0.0;
        self.latch = ScatterLatchState::default();
        self.prev_mode_phases = [0.0; 8];
        self.update_observables_in_place();
        self.refresh_sonification(cfg.dt);
    }

    fn refresh_sonification(&mut self, dt: f64) {
        let modes = self.psi.project_spatial_modes::<8>();
        self.sonification = crate::audio::mappings::quantum_wavepacket_frame(
            &modes,
            &mut self.prev_mode_phases,
            dt.max(1.0e-4),
        );
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

    /// Updates Born R/B/T integrals, two-region partition, and `T_scatter` latch with zero heap allocations.
    fn update_observables_in_place(&mut self) {
        let obs = compute_tunneling_from_psi(&self.x_axis, &self.psi, self.x_left, self.x_right);
        let prev = self.observables;
        self.observables = obs;
        let (pt, pr) = partition_region_probabilities(&self.psi, self.barrier_idx);
        self.p_transmitted = pt;
        self.p_reflected = pr;
        let classical_past_barrier = (self.x0 + self.k0 * self.time) > (self.x_right + self.sigma);
        self.latch
            .update(self.time, classical_past_barrier, prev, obs);
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
            f64::from(cy).clamp(MIN_SCULPT_POTENTIAL, MAX_SCULPT_POTENTIAL)
        };
        apply_bresenham_potential_stroke(&mut self.potential, i0, v, i1, v);
        self.barrier_idx = detect_barrier_index(&self.potential);
        self.hamiltonian_dirty = true;
        self.cached_dt = None;
        self.update_observables_in_place();
    }

    fn restore_base_potential(&mut self) {
        let cfg = self
            .config
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let dx = self.grid_dx();
        self.potential = evaluate_potential_grid(
            cfg.preset,
            &self.x_axis,
            cfg.v0,
            cfg.width,
            dx,
            &cfg.custom_expr,
        );
        self.barrier_idx = detect_barrier_index(&self.potential);
        self.hamiltonian_dirty = true;
        self.cached_dt = None;
        self.update_observables_in_place();
    }

    /// Advances the quantum state by an explicit `dt` step.
    pub fn step_with_dt(&mut self, dt: f64) {
        if self.diverged {
            return;
        }
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
            self.sonification = crate::audio::SonificationFrame::silent();
            return;
        }
        self.update_observables_in_place();
        self.refresh_sonification(dt);
    }

    /// Builds a structured [`WaveData`] snapshot integrating [`QuantumState::probability_density`].
    #[must_use]
    pub fn build_wave_data(&self) -> WaveData {
        let prob_density = self.psi.probability_density();
        let obs =
            compute_tunneling_observables(&self.x_axis, &prob_density, self.x_left, self.x_right);
        let (p_transmitted, p_reflected) =
            partition_region_probabilities(&self.psi, self.barrier_idx);
        let safe_prob: Vec<f64> = prob_density
            .iter()
            .map(|&p| if p.is_finite() { p } else { 0.0 })
            .collect();
        let safe_potential: Vec<f64> = self
            .potential
            .iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect();

        WaveData {
            time: self.time,
            x_axis: self.x_axis.clone(),
            prob_density: safe_prob,
            potential: safe_potential,
            reflection: obs.reflection,
            barrier_prob: obs.barrier_prob,
            transmission: obs.transmission,
            right_wall_prob: obs.right_wall_prob,
            latched_transmission: self.latch.latched_transmission,
            x_left: self.x_left,
            x_right: self.x_right,
            kinetic_energy: self.kinetic_energy,
            p_transmitted,
            p_reflected,
            barrier_idx: self.barrier_idx,
        }
    }

    /// Current latched pre-wall transmission readout $T_{\text{scatter}}$, if latched.
    #[cfg(test)]
    #[must_use]
    pub fn latched_transmission(&self) -> Option<f64> {
        self.latch.latched_transmission
    }
}

impl FiniteStateGuard for WaveRunner {
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

impl SimulationRunner for WaveRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                self.diverged = false;
                self.diagnostic_report = None;
                self.step_count = 0;
                let cfg = self
                    .config
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                self.init_from_config(&cfg);
            }
            SimCommand::SetSpeed(speed) => {
                self.steps_per_frame = speed.max(1);
            }
            SimCommand::SculptStroke1D { i0, v0, i1, v1 } if v0.is_finite() && v1.is_finite() => {
                apply_bresenham_potential_stroke(&mut self.potential, i0, v0, i1, v1);
                self.barrier_idx = detect_barrier_index(&self.potential);
                self.hamiltonian_dirty = true;
                self.cached_dt = None;
                self.update_observables_in_place();
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
                self.kinetic_energy = 0.5 * sk0 * sk0;
                self.psi = gaussian_wavepacket(&self.x_axis, sx0, sk0, ssigma);
                self.time = 0.0;
                self.step_count = 0;
                self.diverged = false;
                self.diagnostic_report = None;
                self.latch = ScatterLatchState::default();
                self.prev_mode_phases = [0.0; 8];
                self.update_observables_in_place();
                let dt = self.config.read().map_or(DEFAULT_DT, |c| c.dt);
                self.refresh_sonification(dt);
            }
            SimCommand::ClearObstacles => self.restore_base_potential(),
            SimCommand::Custom(ref action) if action == "Reset Potential" => {
                self.restore_base_potential();
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        let dt = self.config.read().map_or(DEFAULT_DT, |c| c.dt);
        self.step_with_dt(dt);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let sonification = if self.diverged {
            crate::audio::SonificationFrame::silent()
        } else {
            self.sonification
        };
        let structured_data = Box::new(self.build_wave_data()) as Box<dyn Any + Send>;
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(RwLock::new(Vec::new())),
            custom_data: Vec::new(),
            structured_data: Some(structured_data),
            sonification,
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        self.steps_per_frame
    }

    fn is_finite_state(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn diagnostic_report(&self) -> Option<DiagnosticReport> {
        self.diagnostic_report.clone()
    }
}
