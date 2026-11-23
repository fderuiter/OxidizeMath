#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{
    barrier_interval_for_preset, compute_tunneling_observables_slice, default_compiled_expr,
    evaluate_potential_grid, WavePotentialPreset, DEFAULT_DT, DEFAULT_K0, DEFAULT_SIGMA,
    DEFAULT_STEPS_PER_FRAME, DEFAULT_V0, DEFAULT_WIDTH, DEFAULT_X0, GRID_POINTS, GRID_X_MAX,
    GRID_X_MIN, SIGMA_MAX, SIGMA_MIN,
};
use crate::async_sim::{SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use math_commons::ast::Expr;
use math_explorer::physics::quantum::{
    gaussian_wavepacket_into, state_norm_squared, CrankNicolson1D, QuantumState,
};
use math_explorer::registry::MAX_GRID_SIZE;
use nalgebra::DVector;
use num_complex::Complex;
use std::any::Any;
use std::sync::{Arc, RwLock};

/// Number of unitary Crank-Nicolson substeps executed per `WaveRunner` step.
pub const CN_SUBSTEPS: usize = 4;

pub use super::presets::{TunnelingObservables, BARRIER_TRANSIT_THRESHOLD, WALL_ARRIVAL_THRESHOLD};

/// Shared parameter bundle synchronized between [`super::WaveSimulatorTool`] and [`WaveRunner`].
#[derive(Clone, Debug)]
pub struct WaveSimConfig {
    /// Selected potential profile or custom AST mode.
    pub preset: WavePotentialPreset,
    /// Initial Gaussian packet center $x_0$.
    pub x0: f64,
    /// Initial wave number $k_0$ ($E_0 = k_0^2 / 2$).
    pub k0: f64,
    /// Initial Gaussian spatial width $\sigma$.
    pub sigma: f64,
    /// Potential barrier scale $V_0$.
    pub v0: f64,
    /// Potential barrier width $w$.
    pub width: f64,
    /// Unitary propagation time step $\Delta t$.
    pub dt: f64,
    /// Worker steps executed per visual frame.
    pub steps_per_frame: usize,
    /// Last valid compiled AST expression for `V(x)`.
    pub custom_expr: Expr,
    /// Optional sparse `(index, value)` sculpted potential overrides.
    pub sparse_potential: Vec<(u16, f64)>,
}

impl Default for WaveSimConfig {
    fn default() -> Self {
        Self {
            preset: WavePotentialPreset::InfiniteWell,
            x0: DEFAULT_X0,
            k0: DEFAULT_K0,
            sigma: DEFAULT_SIGMA,
            v0: DEFAULT_V0,
            width: DEFAULT_WIDTH,
            dt: DEFAULT_DT,
            steps_per_frame: DEFAULT_STEPS_PER_FRAME,
            custom_expr: default_compiled_expr(),
            sparse_potential: Vec::new(),
        }
    }
}

/// Structured snapshot payload emitted by [`WaveRunner`] to the UI thread.
#[derive(Clone, Debug)]
pub struct WaveData {
    /// Current simulation time $t$.
    pub time: f64,
    /// Spatial coordinate grid $x_i \in [-5, 5]$.
    pub x_axis: Vec<f64>,
    /// Sanitized Born probability density $|\psi(x_i, t)|^2$.
    pub prob_density: Vec<f64>,
    /// Clamped physical potential $V(x_i) \in [-500, 500]$.
    pub potential: Vec<f64>,
    /// Live reflection probability $R(t)$.
    pub reflection: f64,
    /// Live barrier interior probability $B(t)$.
    pub barrier_prob: f64,
    /// Live transmission probability $T(t)$.
    pub transmission: f64,
    /// Right-wall boundary probability $\sum_{x > 4.25} |\psi|^2$.
    #[cfg(test)]
    pub right_wall_prob: f64,
    /// Latched post-scattering, pre-wall transmission readout $T_{\text{scatter}}$.
    pub latched_transmission: Option<f64>,
    /// Whether the wavepacket has reached either Dirichlet guard wall (`|x| > 4.25`).
    pub wall_arrived: bool,
    /// Whether `latched_transmission` was captured from a quasi-bound resonance at pre-wall arrival.
    pub quasi_bound_latched: bool,
    /// Left edge of the barrier integration interval.
    pub x_left: f64,
    /// Right edge of the barrier integration interval.
    pub x_right: f64,
    /// Incident wavepacket kinetic energy $E_0 = k_0^2 / 2$.
    pub kinetic_energy: f64,
    /// Unitary state norm squared $\|\psi\|^2$.
    pub norm_sq: f64,
}

/// Background worker runner evolving the 1D time-dependent Schrödinger equation with zero per-step allocations.
pub struct WaveRunner {
    pub(crate) config: Arc<RwLock<WaveSimConfig>>,
    pub(crate) psi: QuantumState,
    pub(crate) solver: CrankNicolson1D,
    pub(crate) cached_dt: Option<f64>,
    pub(crate) propagator_rebuilds: Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) prob_scratch: Vec<f64>,
    pub(crate) potential: DVector<f64>,
    pub(crate) base_potential: Vec<f64>,
    pub(crate) x_axis: Vec<f64>,
    pub(crate) x_left: f64,
    pub(crate) x_right: f64,
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
    pub(crate) entered_barrier: bool,
    pub(crate) wall_arrived: bool,
    pub(crate) quasi_bound_latched: bool,
    pub(crate) latched_transmission: Option<f64>,
    pub(crate) diverged: bool,
    pub(crate) diagnostic_report: Option<DiagnosticReport>,
    pub(crate) step_count: usize,
    pub(crate) prev_mode_phases: [f64; 8],
    sonification: crate::audio::SonificationFrame,
}

impl WaveRunner {
    /// Creates and initializes a new 1D Schrödinger runner from shared configuration.
    #[cfg(test)]
    #[must_use]
    pub fn new(config: Arc<RwLock<WaveSimConfig>>) -> Self {
        Self::new_with_counter(config, Arc::new(std::sync::atomic::AtomicUsize::new(0)))
    }

    /// Creates a runner sharing an external `propagator_rebuilds` counter.
    #[must_use]
    pub fn new_with_counter(
        config: Arc<RwLock<WaveSimConfig>>,
        propagator_rebuilds: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        let cfg = config.read().unwrap_or_else(|e| e.into_inner()).clone();
        let max_grid = MAX_GRID_SIZE.max(GRID_POINTS);
        let mut base_potential = Vec::with_capacity(max_grid);
        base_potential.resize(GRID_POINTS, 0.0);
        let mut runner = Self {
            config,
            psi: QuantumState::new(DVector::from_element(GRID_POINTS, Complex::new(0.0, 0.0))),
            solver: CrankNicolson1D::with_capacity(max_grid),
            cached_dt: None,
            propagator_rebuilds,
            prob_scratch: vec![0.0; max_grid],
            potential: DVector::zeros(GRID_POINTS),
            base_potential,
            x_axis: Vec::with_capacity(max_grid),
            x_left: -0.25,
            x_right: 0.25,
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
            entered_barrier: false,
            wall_arrived: false,
            quasi_bound_latched: false,
            latched_transmission: None,
            diverged: false,
            diagnostic_report: None,
            step_count: 0,
            prev_mode_phases: [0.0; 8],
            sonification: crate::audio::SonificationFrame::silent(),
        };
        runner.init_from_config(&cfg);
        runner
    }

    pub(crate) fn record_divergence(&mut self, param: &'static str, message: impl Into<String>) {
        self.diverged = true;
        self.sonification = crate::audio::SonificationFrame::silent();
        self.diagnostic_report = Some(DiagnosticReport::new(
            self.time,
            self.step_count,
            param,
            message.into(),
        ));
    }

    /// Rebuilds the spatial grid, clamped potential, Hamiltonian, and initial Gaussian wavepacket.
    pub fn init_from_config(&mut self, cfg: &WaveSimConfig) {
        let n = self.n_points;
        let span = self.x_max - self.x_min;
        if n < 2 || !span.is_finite() || span <= 0.0 {
            self.record_divergence(
                "dx",
                format!("Invalid spatial grid (n={n}, span={span}): requires n >= 2 and dx > 0."),
            );
            return;
        }
        let dx = span / (n as f64 - 1.0);
        self.x_axis.resize(n, 0.0);
        self.x_axis
            .iter_mut()
            .enumerate()
            .for_each(|(i, s)| *s = self.x_min + i as f64 * dx);
        (self.x_left, self.x_right) = barrier_interval_for_preset(cfg.preset, cfg.width, dx);
        self.potential = evaluate_potential_grid(
            cfg.preset,
            &self.x_axis,
            cfg.v0,
            cfg.width,
            dx,
            &cfg.custom_expr,
        );
        self.sync_base_and_sparse_potential(&cfg.sparse_potential);
        if !self
            .solver
            .update_potential(self.potential.as_slice(), dx, 1.0, 1.0)
        {
            self.record_divergence("dx", format!("Invalid spatial step dx: {dx}"));
            return;
        }
        self.cached_dt = None;
        if n > self.prob_scratch.len() {
            self.prob_scratch.resize(n, 0.0);
        }
        let finite_or = |v: f64, d: f64| if v.is_finite() { v } else { d };
        (self.x0, self.k0) = (finite_or(cfg.x0, DEFAULT_X0), finite_or(cfg.k0, DEFAULT_K0));
        self.sigma = if cfg.sigma.is_finite() && cfg.sigma > 0.0 {
            cfg.sigma.clamp(SIGMA_MIN, SIGMA_MAX)
        } else {
            DEFAULT_SIGMA
        };
        self.kinetic_energy = 0.5 * self.k0 * self.k0;
        self.steps_per_frame = cfg.steps_per_frame.max(1);
        gaussian_wavepacket_into(&self.x_axis, self.x0, self.k0, self.sigma, &mut self.psi);
        self.time = 0.0;
        self.entered_barrier = false;
        self.wall_arrived = false;
        self.quasi_bound_latched = false;
        self.latched_transmission = None;
        self.prev_mode_phases = [0.0; 8];
        self.update_observables_in_place();
        self.refresh_sonification(cfg.dt);
    }

    pub(crate) fn refresh_sonification(&mut self, dt: f64) {
        let modes = self.psi.project_spatial_modes::<8>();
        self.sonification = crate::audio::mappings::quantum_wavepacket_frame(
            &modes,
            &mut self.prev_mode_phases,
            dt.max(1.0e-4),
        );
    }

    pub(crate) fn refresh_propagator(&mut self, dt: f64) -> bool {
        if self.cached_dt != Some(dt) {
            if !dt.is_finite() || dt <= 0.0 {
                self.solver.invalidate();
                self.cached_dt = None;
                return false;
            }
            let sub_dt = dt / (CN_SUBSTEPS as f64);
            if !self.solver.update_dt(sub_dt, 1.0) {
                self.cached_dt = None;
                return false;
            }
            self.cached_dt = Some(dt);
            self.propagator_rebuilds
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.solver.is_valid()
    }

    /// Updates Born R/B/T integrals and pre-wall `T_scatter` latch in-place with zero heap allocations.
    pub(crate) fn update_observables_in_place(&mut self) {
        let n = self.psi.vector.len();
        if n <= self.prob_scratch.len() {
            self.psi
                .probability_density_in_place(&mut self.prob_scratch[..n]);
            let obs = compute_tunneling_observables_slice(
                &self.x_axis,
                &self.prob_scratch[..n],
                self.x_left,
                self.x_right,
            );
            let prev = self.observables;
            self.observables = obs;
            self.update_scatter_latch(prev, obs);
        }
    }

    fn update_scatter_latch(&mut self, prev: TunnelingObservables, obs: TunnelingObservables) {
        if self.time <= 0.0 {
            return;
        }
        let classical_past_barrier = (self.x0 + self.k0 * self.time) > (self.x_right + self.sigma);
        if obs.barrier_prob >= BARRIER_TRANSIT_THRESHOLD || classical_past_barrier {
            self.entered_barrier = true;
        }
        if self.entered_barrier && !self.wall_arrived {
            if obs.barrier_prob < BARRIER_TRANSIT_THRESHOLD {
                if obs.right_wall_prob < WALL_ARRIVAL_THRESHOLD {
                    self.latched_transmission = Some(obs.transmission);
                    self.quasi_bound_latched = false;
                } else if self.latched_transmission.is_none()
                    && prev.right_wall_prob < WALL_ARRIVAL_THRESHOLD
                {
                    let alpha_b = if prev.barrier_prob < BARRIER_TRANSIT_THRESHOLD {
                        0.0
                    } else {
                        let denom = (prev.barrier_prob - obs.barrier_prob).max(f64::EPSILON);
                        ((prev.barrier_prob - BARRIER_TRANSIT_THRESHOLD) / denom).clamp(0.0, 1.0)
                    };
                    let denom_w = (obs.right_wall_prob - prev.right_wall_prob).max(f64::EPSILON);
                    let alpha_w =
                        ((WALL_ARRIVAL_THRESHOLD - prev.right_wall_prob) / denom_w).clamp(0.0, 1.0);
                    if alpha_b <= alpha_w {
                        let t_pre =
                            prev.transmission + alpha_w * (obs.transmission - prev.transmission);
                        self.latched_transmission = Some(t_pre.clamp(0.0, 1.0));
                        self.quasi_bound_latched = false;
                    }
                }
            } else if self.latched_transmission.is_none()
                && obs.right_wall_prob >= WALL_ARRIVAL_THRESHOLD
            {
                let alpha_w = if prev.right_wall_prob < WALL_ARRIVAL_THRESHOLD {
                    let denom_w = (obs.right_wall_prob - prev.right_wall_prob).max(f64::EPSILON);
                    ((WALL_ARRIVAL_THRESHOLD - prev.right_wall_prob) / denom_w).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let t_pre = prev.transmission + alpha_w * (obs.transmission - prev.transmission);
                self.latched_transmission = Some(t_pre.clamp(0.0, 1.0));
                self.quasi_bound_latched = true;
            }
        }
        if obs.right_wall_prob >= WALL_ARRIVAL_THRESHOLD
            || (self.latched_transmission.is_some() && obs.left_wall_prob >= WALL_ARRIVAL_THRESHOLD)
        {
            self.wall_arrived = true;
        }
    }

    /// Advances the quantum state by an explicit `dt` step.
    pub fn step_with_dt(&mut self, dt: f64) {
        if self.diverged {
            return;
        }
        if !dt.is_finite() || dt <= 0.0 {
            self.record_divergence("dt", format!("Invalid non-finite time step dt: {dt}"));
            return;
        }
        if !self.refresh_propagator(dt) {
            self.record_divergence(
                "solver",
                format!("Failed to factor Crank-Nicolson tridiagonal system for dt: {dt}"),
            );
            return;
        }
        self.step_count += 1;
        let stepped = self
            .solver
            .step_substeps_in_place(&mut self.psi, CN_SUBSTEPS);
        self.time += dt;
        let psi_finite = |s: &QuantumState| {
            s.vector
                .iter()
                .all(|c| c.re.is_finite() && c.im.is_finite())
        };
        if !stepped || !psi_finite(&self.psi) {
            self.record_divergence(
                "psi",
                "Non-finite wave packet amplitude detected during time evolution.",
            );
            return;
        }
        self.psi.normalize_in_place();
        if !psi_finite(&self.psi) {
            self.record_divergence(
                "psi_norm",
                "Wave function normalization diverged: non-finite norm encountered.",
            );
            return;
        }
        self.update_observables_in_place();
        self.refresh_sonification(dt);
    }

    /// Builds a structured [`WaveData`] snapshot integrating [`QuantumState::probability_density_in_place`].
    #[must_use]
    pub fn build_wave_data(&self) -> WaveData {
        let mut safe_prob = vec![0.0; self.psi.vector.len()];
        self.psi.probability_density_in_place(&mut safe_prob);
        let obs = compute_tunneling_observables_slice(
            &self.x_axis,
            &safe_prob,
            self.x_left,
            self.x_right,
        );
        for p in &mut safe_prob {
            *p = if p.is_finite() { *p } else { 0.0 };
        }
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
            #[cfg(test)]
            right_wall_prob: obs.right_wall_prob,
            latched_transmission: self.latched_transmission,
            wall_arrived: self.wall_arrived,
            quasi_bound_latched: self.quasi_bound_latched,
            x_left: self.x_left,
            x_right: self.x_right,
            kinetic_energy: self.kinetic_energy,
            norm_sq: state_norm_squared(&self.psi),
        }
    }

    /// Current latched pre-wall transmission readout $T_{\text{scatter}}$, if latched.
    #[cfg(test)]
    #[must_use]
    pub fn latched_transmission(&self) -> Option<f64> {
        self.latched_transmission
    }
}

impl SimulationRunner for WaveRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::Reset => {
                self.reset_parameters();
                let cfg = self
                    .config
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                self.init_from_config(&cfg);
            }
            SimCommand::SetSpeed(speed) => self.steps_per_frame = speed.max(1),
            other => self.handle_sculpt_command(other),
        }
    }

    fn step(&mut self) {
        let dt = self.config.read().map_or(DEFAULT_DT, |c| c.dt);
        self.step_with_dt(dt);
    }

    fn get_snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(RwLock::new(Vec::new())),
            custom_data: Vec::new(),
            structured_data: Some(Box::new(self.build_wave_data()) as Box<dyn Any + Send>),
            sonification: if self.diverged {
                crate::audio::SonificationFrame::silent()
            } else {
                self.sonification
            },
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
