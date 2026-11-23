//! Preallocated zero-allocation respiratory, clutter, phase, Kalman, and LINAC gating pipeline.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::spectra::{build_fmcw_config, compute_range_spectra, SpectralSnapshot};
use super::state::RadarGatingState;
use math_explorer::physics::medical::optical_motion::simulation::cosine_respiratory_curve;
use math_explorer::physics::medical::radar_gating::surface::weighted_average_height;
use math_explorer::physics::medical::radar_gating::{
    BeamState, BiQuadraticSurface, ConstantVelocityModel, EllipticalFilter, GatingLogic,
    PhaseUnwrapper, TrackingFilter,
};
use nalgebra::Point3;
use num_complex::Complex;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use std::f64::consts::PI;

/// Fixed time-series ring length preallocated at startup for zero-allocation stepping.
pub const HISTORY_LEN: usize = 200;
/// Slow-time control-loop step duration in seconds (`40 Hz` gating rate).
pub const SIM_DT: f64 = 0.025;
/// Coherent slow-time chirp substeps per control frame to prevent 60 GHz phase aliasing on coughs.
const CHIRP_SUBSTEPS: usize = 8;

/// Stack-resident 2-state Constant-Velocity Kalman filter matching `TrackingFilter` algebra.
#[derive(Debug, Clone, Copy)]
pub struct FastCvKalman {
    /// State vector `[amplitude_m, velocity_m_s]`.
    pub state: [f64; 2],
    /// State error covariance `[[p00, p01], [p10, p11]]`.
    pub cov: [[f64; 2]; 2],
    /// Discretization interval in seconds.
    pub dt: f64,
    /// Process noise variance $Q$.
    pub process_var: f64,
    /// Measurement noise variance $R$.
    pub meas_var: f64,
}

impl FastCvKalman {
    /// Initializes stack state directly from a verified `TrackingFilter` instance.
    pub fn from_tracking_filter(
        tf: &TrackingFilter,
        dt: f64,
        process_var: f64,
        meas_var: f64,
    ) -> Self {
        let p = tf.covariance();
        Self {
            state: [tf.amplitude(), tf.velocity()],
            cov: [[p[(0, 0)], p[(0, 1)]], [p[(1, 0)], p[(1, 1)]]],
            dt,
            process_var,
            meas_var,
        }
    }

    /// Executes the exact `ConstantVelocityModel` predict step on stack scalars.
    pub fn predict(&mut self) {
        let dt = self.dt;
        self.state[0] += dt * self.state[1];
        let p00 = self.cov[0][0];
        let p01 = self.cov[0][1];
        let p10 = self.cov[1][0];
        let p11 = self.cov[1][1];
        self.cov[0][0] = p00 + dt * (p10 + p01) + dt * dt * p11 + self.process_var;
        self.cov[0][1] = p01 + dt * p11;
        self.cov[1][0] = p10 + dt * p11;
        self.cov[1][1] = p11 + self.process_var;
    }

    /// Executes the exact scalar position measurement update (`H = [1, 0]`) on stack scalars.
    pub fn update(&mut self, z_m: f64) {
        let s = self.cov[0][0] + self.meas_var;
        if !s.is_finite() || s.abs() <= 1.0e-18 {
            return;
        }
        let y = z_m - self.state[0];
        let k0 = self.cov[0][0] / s;
        let k1 = self.cov[1][0] / s;
        self.state[0] += k0 * y;
        self.state[1] += k1 * y;
        let p00 = self.cov[0][0];
        let p01 = self.cov[0][1];
        let p10 = self.cov[1][0];
        let p11 = self.cov[1][1];
        self.cov[0][0] = (1.0 - k0) * p00;
        self.cov[0][1] = (1.0 - k0) * p01;
        self.cov[1][0] = p10 - k1 * p00;
        self.cov[1][1] = p11 - k1 * p01;
    }
}

/// Preallocated simulation engine holding spectral snapshots and zero-allocation time-series buffers.
pub struct RadarPipeline {
    /// Cached range-domain spectra updated when RF or spatial parameters change.
    pub spectra: SpectralSnapshot,
    /// Verified `EllipticalFilter` pre-warmed beyond `window_size + 1` capacity.
    pub clutter_filter: EllipticalFilter,
    /// Verified `PhaseUnwrapper` converting IQ slow-time phase into displacement.
    pub phase_unwrapper: PhaseUnwrapper,
    /// Fitted `BiQuadraticSurface` smoothing thoracic 3x3 patch sparkle noise.
    pub chest_surface: Option<BiQuadraticSurface>,
    /// Spatial ROI weighted average offset computed from the fitted thoracic surface.
    pub surface_wah_m: f64,
    /// Verified `TrackingFilter` seeded on initialization/reset.
    pub tracking_filter: TrackingFilter,
    /// Zero-allocation mirror of `TrackingFilter` stepped during locked playback.
    pub fast_kalman: FastCvKalman,
    /// Verified Schmitt-trigger `GatingLogic` state machine.
    pub gating: GatingLogic,
    /// Deterministic PRNG for continuous slow-time IQ noise.
    pub rng: OxidizeRng,
    /// Current simulation clock in seconds.
    pub sim_time_s: f64,
    /// Remaining duration of an injected cough transient in seconds.
    pub cough_timer_s: f64,
    /// Extra static clutter reflector toggle injected by the user.
    pub extra_clutter_active: bool,
    /// Latched sensor-dropout `NaN` fault feeding `GatingLogic::evaluate`.
    pub nan_fault_active: bool,
    /// Total operational frames evaluated since reset.
    pub total_frames: u64,
    /// Total frames spent in `BeamState::BeamActive`.
    pub active_frames: u64,
    /// Preallocated unwrapped + surface-smoothed displacement trace (`[t_s, disp_mm]`).
    pub measured_trace_mm: Vec<[f64; 2]>,
    /// Preallocated latency-compensated Kalman prediction trace (`[t_s, pred_mm]`).
    pub kalman_trace_mm: Vec<[f64; 2]>,
    /// Preallocated cumulative LINAC duty-cycle trace (`[t_s, duty_pct]`).
    pub duty_cycle_trace_pct: Vec<[f64; 2]>,
    /// Preallocated `BeamActive` window indicator (`[t_s, 0_or_1]`).
    pub beam_active_trace: Vec<[f64; 2]>,
    /// Preallocated `BeamHold` window indicator (`[t_s, 0_or_1]`).
    pub beam_hold_trace: Vec<[f64; 2]>,
    /// Preallocated `SafetyInterlock` window indicator (`[t_s, 0_or_1]`).
    pub interlock_trace: Vec<[f64; 2]>,
    /// Baseline offset aligning the unwrapped phase origin with true respiratory mean.
    phase_origin_m: f64,
    /// Cached wavelength in meters for slow-time IQ synthesis.
    wavelength_m: f64,
    /// Duration of the most recent step execution in microseconds.
    pub last_step_us: f64,
}

impl RadarPipeline {
    /// Allocates all ring buffers up front and warms up filters from `state`.
    pub fn new(state: &RadarGatingState) -> Self {
        let fmcw = build_fmcw_config(state);
        let wavelength_m = fmcw.wavelength();
        let model = ConstantVelocityModel::new(state.kalman_process_var, state.kalman_meas_var);
        let tracking_filter = TrackingFilter::new(0.0, SIM_DT, model);
        let fast_kalman = FastCvKalman::from_tracking_filter(
            &tracking_filter,
            SIM_DT,
            state.kalman_process_var,
            state.kalman_meas_var,
        );
        let gating = GatingLogic::new(
            state.gating_threshold_mm * 1.0e-3,
            state.gating_hysteresis_mm * 1.0e-3,
            state.system_latency_ms * 1.0e-3,
        );
        let mut pipeline = Self {
            spectra: SpectralSnapshot::default(),
            clutter_filter: EllipticalFilter::new(state.clutter_window as f64),
            phase_unwrapper: PhaseUnwrapper::new(wavelength_m),
            chest_surface: None,
            surface_wah_m: 0.0,
            tracking_filter,
            fast_kalman,
            gating,
            rng: OxidizeRng::new(state.rng_seed),
            sim_time_s: 0.0,
            cough_timer_s: 0.0,
            extra_clutter_active: false,
            nan_fault_active: false,
            total_frames: 0,
            active_frames: 0,
            measured_trace_mm: vec![[0.0, 0.0]; HISTORY_LEN],
            kalman_trace_mm: vec![[0.0, 0.0]; HISTORY_LEN],
            duty_cycle_trace_pct: vec![[0.0, 0.0]; HISTORY_LEN],
            beam_active_trace: vec![[0.0, 0.0]; HISTORY_LEN],
            beam_hold_trace: vec![[0.0, 0.0]; HISTORY_LEN],
            interlock_trace: vec![[0.0, 0.0]; HISTORY_LEN],
            phase_origin_m: 0.0,
            wavelength_m,
            last_step_us: 0.0,
        };
        pipeline.rebuild_all(state);
        pipeline
    }

    /// Recomputes spectra, fits `BiQuadraticSurface`, warms up `EllipticalFilter`, and seeds history.
    pub fn rebuild_all(&mut self, state: &RadarGatingState) {
        self.spectra = compute_range_spectra(state);
        self.rebuild_time_domain(state);
    }

    /// Recomputes only the spatial range spectra (`Coarse FFT`, `CZT`, `MUSIC`).
    pub fn refresh_spectra_only(&mut self, state: &RadarGatingState) {
        self.spectra = compute_range_spectra(state);
        let fmcw = build_fmcw_config(state);
        self.wavelength_m = fmcw.wavelength();
    }

    /// Rebuilds the thoracic surface fit, Kalman filter, clutter window, and armed gating history.
    pub fn rebuild_time_domain(&mut self, state: &RadarGatingState) {
        let fmcw = build_fmcw_config(state);
        self.wavelength_m = fmcw.wavelength();
        self.fit_thoracic_patch(state.rng_seed);

        self.clutter_filter = EllipticalFilter::new(state.clutter_window as f64);
        self.rng = OxidizeRng::new(state.rng_seed ^ 0xA5A5_1234);
        self.sim_time_s = 0.0;
        self.cough_timer_s = 0.0;
        self.nan_fault_active = false;
        self.total_frames = 0;
        self.active_frames = 0;

        self.warm_clutter_capacity(state);
        self.phase_unwrapper = PhaseUnwrapper::new(self.wavelength_m);
        let z0 = respiratory_displacement_m(0.0, state, 0.0);
        let iq0 = self.sample_slow_time_iq(z0, state);
        let _ = self.phase_unwrapper.process(iq0);
        self.phase_origin_m = z0;

        let model = ConstantVelocityModel::new(state.kalman_process_var, state.kalman_meas_var);
        self.tracking_filter = TrackingFilter::new(z0, SIM_DT, model);
        self.gating = GatingLogic::new(
            state.gating_threshold_mm * 1.0e-3,
            state.gating_hysteresis_mm * 1.0e-3,
            state.system_latency_ms * 1.0e-3,
        );
        let _ = self.gating.arm();

        for _ in 0..HISTORY_LEN {
            self.seed_step_with_tracking_filter(state);
        }
        self.fast_kalman = FastCvKalman::from_tracking_filter(
            &self.tracking_filter,
            SIM_DT,
            state.kalman_process_var,
            state.kalman_meas_var,
        );
    }

    fn fit_thoracic_patch(&mut self, seed: u64) {
        let mut patch_rng = OxidizeRng::new(seed ^ 0xBEEF_CAFE);
        let coords = [-0.04_f64, 0.0, 0.04];
        let mut pts = Vec::with_capacity(9);
        for &px in &coords {
            for &py in &coords {
                let sparkle = (rng_unit_f64(&mut patch_rng) - 0.5) * 2.0e-4;
                let pz = -0.012 * px * px - 0.008 * py * py + sparkle;
                pts.push(Point3::new(px, py, pz));
            }
        }
        self.surface_wah_m = weighted_average_height(&pts);
        self.chest_surface = BiQuadraticSurface::fit(&pts);
    }

    fn warm_clutter_capacity(&mut self, state: &RadarGatingState) {
        let warm_steps = (state.clutter_window + 8) * CHIRP_SUBSTEPS;
        let dt_sub = SIM_DT / CHIRP_SUBSTEPS as f64;
        for idx in 0..warm_steps {
            let t = idx as f64 * dt_sub;
            let z = respiratory_displacement_m(t, state, 0.0);
            let _ = self.sample_slow_time_iq(z, state);
        }
    }

    fn sample_slow_time_iq(&mut self, disp_m: f64, state: &RadarGatingState) -> Complex<f64> {
        let r_total = state.primary_target.range_m() + disp_m;
        let phase = (4.0 * PI * r_total) / self.wavelength_m;
        let clutter_mag = state.static_clutter_gain + if self.extra_clutter_active { 1.25 } else { 0.0 };
        let static_clutter = Complex::new(clutter_mag * 0.8, -clutter_mag * 0.6);
        let n_re = (rng_unit_f64(&mut self.rng) - 0.5) * 2.0 * state.noise_sigma * 0.25;
        let n_im = (rng_unit_f64(&mut self.rng) - 0.5) * 2.0 * state.noise_sigma * 0.25;
        let raw_iq = Complex::new(phase.cos(), phase.sin()) * state.primary_target.rcs
            + static_clutter
            + Complex::new(n_re, n_im);
        self.clutter_filter.filter(raw_iq)
    }

    fn advance_unwrapped_measurement(&mut self, state: &RadarGatingState) -> f64 {
        let dt_sub = SIM_DT / CHIRP_SUBSTEPS as f64;
        let mut unwrapped_m = 0.0;
        for _ in 0..CHIRP_SUBSTEPS {
            self.sim_time_s += dt_sub;
            if self.cough_timer_s > 0.0 {
                self.cough_timer_s = (self.cough_timer_s - dt_sub).max(0.0);
            }
            let z_true = respiratory_displacement_m(self.sim_time_s, state, self.cough_timer_s);
            let iq = self.sample_slow_time_iq(z_true, state);
            unwrapped_m = self.phase_unwrapper.process(iq) + self.phase_origin_m;
        }
        let surface_corr = self
            .chest_surface
            .as_ref()
            .map_or(0.0, |surf| surf.evaluate(0.0, 0.0) - self.surface_wah_m);
        unwrapped_m + 0.15 * surface_corr
    }

    fn seed_step_with_tracking_filter(&mut self, state: &RadarGatingState) {
        let measured_m = self.advance_unwrapped_measurement(state);
        self.tracking_filter.predict();
        self.tracking_filter.update(measured_m);
        let pos = self.tracking_filter.amplitude();
        let vel = self.tracking_filter.velocity();
        let beam_state = self.gating.evaluate(pos, vel);
        self.record_history_point(measured_m, pos, vel, state.system_latency_ms, beam_state);
    }

    /// Advances one control step with **zero dynamic memory allocations** under `VerifiedAllocator`.
    pub fn step_zero_alloc(&mut self, state: &RadarGatingState) -> BeamState {
        let start = crate::web_io::PlatformInstant::now();
        let measured_m = self.advance_unwrapped_measurement(state);
        self.fast_kalman.predict();
        if !self.nan_fault_active {
            self.fast_kalman.update(measured_m);
        }
        let pos = self.fast_kalman.state[0];
        let vel = self.fast_kalman.state[1];
        let eval_pos = if self.nan_fault_active { f64::NAN } else { pos };
        let beam_state = self.gating.evaluate(eval_pos, vel);
        self.record_history_point(measured_m, pos, vel, state.system_latency_ms, beam_state);
        self.last_step_us = start.elapsed().as_secs_f64() * 1.0e6;
        beam_state
    }

    fn record_history_point(
        &mut self,
        measured_m: f64,
        pos_m: f64,
        vel_m_s: f64,
        latency_ms: f64,
        beam_state: BeamState,
    ) {
        self.total_frames = self.total_frames.saturating_add(1);
        if beam_state == BeamState::BeamActive {
            self.active_frames = self.active_frames.saturating_add(1);
        }
        let duty_pct = self.duty_cycle_percent();
        let pred_mm = (pos_m + vel_m_s * (latency_ms * 1.0e-3)) * 1.0e3;
        let t = self.sim_time_s;

        push_ring(&mut self.measured_trace_mm, [t, measured_m * 1.0e3]);
        push_ring(&mut self.kalman_trace_mm, [t, pred_mm]);
        push_ring(&mut self.duty_cycle_trace_pct, [t, duty_pct]);
        push_ring(
            &mut self.beam_active_trace,
            [t, if beam_state == BeamState::BeamActive { 1.0 } else { 0.0 }],
        );
        push_ring(
            &mut self.beam_hold_trace,
            [t, if beam_state == BeamState::BeamHold { 1.0 } else { 0.0 }],
        );
        push_ring(
            &mut self.interlock_trace,
            [
                t,
                if beam_state == BeamState::SafetyInterlock {
                    1.0
                } else {
                    0.0
                },
            ],
        );
    }

    /// Current cumulative LINAC beam-on duty cycle in `[0.0, 100.0]` percent.
    pub fn duty_cycle_percent(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            100.0 * (self.active_frames as f64) / (self.total_frames as f64)
        }
    }

    /// Injects a deterministic `+10 mm` cough impulse lasting `0.45 s`.
    pub fn inject_cough_spike(&mut self) {
        self.cough_timer_s = 0.45;
    }

    /// Injects a sensor-loss `NaN` measurement fault to test `BeamState::SafetyInterlock`.
    pub fn inject_nan_fault(&mut self) {
        self.nan_fault_active = true;
        let _ = self.gating.evaluate(f64::NAN, self.fast_kalman.state[1]);
    }

    /// Clears a latched `SafetyInterlock`, returning `GatingLogic` to `BeamState::Standby`.
    pub fn reset_interlock_to_standby(&mut self) -> BeamState {
        self.nan_fault_active = false;
        let _ = self.gating.reset_interlock();
        self.gating.state()
    }

    /// Arms `GatingLogic` from `Standby` into `BeamState::Armed`.
    pub fn arm_controller(&mut self) -> BeamState {
        let _ = self.gating.arm();
        self.gating.state()
    }

    /// Disarms `GatingLogic` into `BeamState::Standby` when not interlocked.
    pub fn disarm_controller(&mut self) -> BeamState {
        let _ = self.gating.disarm();
        self.gating.state()
    }
}

fn respiratory_displacement_m(t_s: f64, state: &RadarGatingState, cough_timer_s: f64) -> f64 {
    let amp_m = state.breath_amplitude_mm * 1.0e-3;
    let tau_param = 3.0 * state.breath_period_s.max(0.5);
    let base = cosine_respiratory_curve(t_s, amp_m, tau_param);
    let cough = if cough_timer_s > 0.0 {
        let progress = 1.0 - (cough_timer_s / 0.45).clamp(0.0, 1.0);
        0.010 * (PI * progress).sin().powi(2)
    } else {
        0.0
    };
    base + cough
}

fn rng_unit_f64(rng: &mut OxidizeRng) -> f64 {
    ((rng.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn push_ring(buf: &mut [[f64; 2]], point: [f64; 2]) {
    let len = buf.len();
    if len > 1 {
        buf.copy_within(1..len, 0);
    }
    if len > 0 {
        buf[len - 1] = point;
    }
}

// [cite:mmwave_radiotherapy_setup]
