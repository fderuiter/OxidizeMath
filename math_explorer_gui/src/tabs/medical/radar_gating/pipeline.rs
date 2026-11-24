//! Preallocated zero-allocation respiratory, clutter, phase, Kalman, and LINAC gating pipeline.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub use super::challenge::FastCvKalman;
use super::challenge::{
    should_trigger_scheduled_cough, step_physical_beam_delay, GEOGRAPHIC_MISS_CEILING_MM,
};
use super::spectra::{build_fmcw_config, compute_range_spectra, SpectralSnapshot};
use super::state::RadarGatingState;
use crate::tabs::medical::link_bus::GatingTelemetry;
use math_explorer::physics::medical::evaluation::{
    pearson_correlation, root_mean_square_error, spatial_accuracy_error,
};
use math_explorer::physics::medical::optical_motion::simulation::cosine_respiratory_curve;
use math_explorer::physics::medical::radar_gating::surface::weighted_average_height;
use math_explorer::physics::medical::radar_gating::{
    BeamState, BiQuadraticSurface, ConstantVelocityModel, EllipticalFilter, GatingLogic,
    PhaseUnwrapper, TrackingFilter,
};
use nalgebra::Point3;
use num_complex::Complex;
use oxidize_core::rng::{sample_closed_open, OxidizeRng};
use std::f64::consts::PI;

/// Fixed time-series ring length preallocated at startup for zero-allocation stepping.
pub const HISTORY_LEN: usize = 200;
/// Preallocated physical beam-delay ring capacity (`24 * 25 ms = 600 ms` max latency).
pub const BEAM_DELAY_RING_LEN: usize = 24;
/// Slow-time control-loop step duration in seconds (`40 Hz` gating rate).
pub const SIM_DT: f64 = 0.025;
/// Coherent slow-time chirp substeps per control frame to prevent 60 GHz phase aliasing on coughs.
const CHIRP_SUBSTEPS: usize = 8;

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
    /// Total frames where the delayed physical LINAC beam actively delivered dose.
    pub physical_active_frames: u64,
    /// Frames where delayed physical beam was ON while `z_true > -1.20 mm`.
    pub geographic_miss_frames: u64,
    /// Whether periodic `+10 mm` challenge cough transients are enabled.
    pub scheduled_cough_enabled: bool,
    /// Fixed 24-slot delay ring buffer modeling `system_latency_ms` physical beam lag.
    pub beam_delay_ring: [bool; BEAM_DELAY_RING_LEN],
    /// Write cursor into `beam_delay_ring`.
    pub beam_delay_cursor: usize,
    /// Preallocated ground-truth respiratory + cough chest trace (`[t_s, z_true_mm]`).
    pub true_trace_mm: Vec<[f64; 2]>,
    /// Preallocated delayed physical beam-on window trace (`[t_s, 0_or_1]`).
    pub physical_beam_trace: Vec<[f64; 2]>,
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
    true_disp_ring_mm: Box<[f64; HISTORY_LEN]>,
    predicted_disp_ring_mm: Box<[f64; HISTORY_LEN]>,
    /// Verified `[pearson_r, rmse_mm, spatial_err_pct]` tracking summary.
    pub eval_metrics: [f64; 3],
    /// Latest `[true_mm, measured_mm, predicted_mm]` respiratory displacements.
    pub latest_disp_mm: [f64; 3],
    phase_origin_m: f64,
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
            physical_active_frames: 0,
            geographic_miss_frames: 0,
            scheduled_cough_enabled: false,
            beam_delay_ring: [false; BEAM_DELAY_RING_LEN],
            beam_delay_cursor: 0,
            true_trace_mm: vec![[0.0, 0.0]; HISTORY_LEN],
            physical_beam_trace: vec![[0.0, 0.0]; HISTORY_LEN],
            measured_trace_mm: vec![[0.0, 0.0]; HISTORY_LEN],
            kalman_trace_mm: vec![[0.0, 0.0]; HISTORY_LEN],
            duty_cycle_trace_pct: vec![[0.0, 0.0]; HISTORY_LEN],
            beam_active_trace: vec![[0.0, 0.0]; HISTORY_LEN],
            beam_hold_trace: vec![[0.0, 0.0]; HISTORY_LEN],
            interlock_trace: vec![[0.0, 0.0]; HISTORY_LEN],
            true_disp_ring_mm: Box::new([0.0; HISTORY_LEN]),
            predicted_disp_ring_mm: Box::new([0.0; HISTORY_LEN]),
            eval_metrics: [1.0, 0.0, 0.0],
            latest_disp_mm: [0.0; 3],
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
        self.rng = OxidizeRng::new(state.rng_seed).with_stream(2);
        self.sim_time_s = 0.0;
        self.cough_timer_s = 0.0;
        self.nan_fault_active = false;
        self.total_frames = 0;
        self.active_frames = 0;
        self.physical_active_frames = 0;
        self.geographic_miss_frames = 0;
        self.beam_delay_ring = [false; BEAM_DELAY_RING_LEN];
        self.beam_delay_cursor = 0;

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
        let mut patch_rng = OxidizeRng::new(seed).with_stream(3);
        let coords = [-0.04_f64, 0.0, 0.04];
        let mut pts = Vec::with_capacity(9);
        for &px in &coords {
            for &py in &coords {
                let sparkle = (sample_closed_open::<f64, _>(&mut patch_rng) - 0.5) * 2.0e-4;
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
            let t = -((warm_steps - idx) as f64) * dt_sub;
            let z = respiratory_displacement_m(t, state, 0.0);
            let _ = self.sample_slow_time_iq(z, state);
        }
    }

    fn sample_slow_time_iq(&mut self, disp_m: f64, state: &RadarGatingState) -> Complex<f64> {
        let r_total = state.primary_target.range_m() + disp_m;
        let phase = (4.0 * PI * r_total) / self.wavelength_m;
        let clutter_mag =
            state.static_clutter_gain + if self.extra_clutter_active { 1.25 } else { 0.0 };
        let static_clutter = Complex::new(clutter_mag * 0.8, -clutter_mag * 0.6);
        let n_re =
            (sample_closed_open::<f64, _>(&mut self.rng) - 0.5) * 2.0 * state.noise_sigma * 0.25;
        let n_im =
            (sample_closed_open::<f64, _>(&mut self.rng) - 0.5) * 2.0 * state.noise_sigma * 0.25;
        let raw_iq = Complex::new(phase.cos(), phase.sin()) * state.primary_target.rcs
            + static_clutter
            + Complex::new(n_re, n_im);
        self.clutter_filter.filter(raw_iq)
    }

    fn advance_unwrapped_measurement(&mut self, state: &RadarGatingState) -> (f64, f64) {
        let dt_sub = SIM_DT / CHIRP_SUBSTEPS as f64;
        let mut unwrapped_m = 0.0;
        let mut z_true = 0.0;
        for _ in 0..CHIRP_SUBSTEPS {
            let prev_t = self.sim_time_s;
            self.sim_time_s += dt_sub;
            if self.scheduled_cough_enabled
                && should_trigger_scheduled_cough(prev_t, self.sim_time_s, state.breath_period_s)
            {
                self.cough_timer_s = 0.45;
            } else if self.cough_timer_s > 0.0 {
                self.cough_timer_s = (self.cough_timer_s - dt_sub).max(0.0);
            }
            z_true = respiratory_displacement_m(self.sim_time_s, state, self.cough_timer_s);
            let iq = self.sample_slow_time_iq(z_true, state);
            unwrapped_m = self.phase_unwrapper.process(iq) + self.phase_origin_m;
        }
        let surface_corr = self
            .chest_surface
            .as_ref()
            .map_or(0.0, |surf| surf.evaluate(0.0, 0.0) - self.surface_wah_m);
        (unwrapped_m + 0.15 * surface_corr, z_true)
    }

    fn seed_step_with_tracking_filter(&mut self, state: &RadarGatingState) {
        let (measured_m, z_true_m) = self.advance_unwrapped_measurement(state);
        self.tracking_filter.predict();
        self.tracking_filter.update(measured_m);
        let pos = self.tracking_filter.amplitude();
        let vel = self.tracking_filter.velocity();
        let beam_state = self.gating.evaluate(pos, vel);
        self.record_history_point(z_true_m, measured_m, pos, vel, state, beam_state);
    }

    /// Advances one control step with **zero dynamic memory allocations** under `VerifiedAllocator`.
    pub fn step_zero_alloc(&mut self, state: &RadarGatingState) -> BeamState {
        let start = crate::web_io::PlatformInstant::now();
        let (measured_m, z_true_m) = self.advance_unwrapped_measurement(state);
        self.fast_kalman.predict();
        if !self.nan_fault_active {
            self.fast_kalman.update(measured_m);
        }
        let pos = self.fast_kalman.state[0];
        let vel = self.fast_kalman.state[1];
        let eval_pos = if self.nan_fault_active { f64::NAN } else { pos };
        let beam_state = self.gating.evaluate(eval_pos, vel);
        self.record_history_point(z_true_m, measured_m, pos, vel, state, beam_state);
        self.last_step_us = start.elapsed().as_secs_f64() * 1.0e6;
        beam_state
    }

    fn record_history_point(
        &mut self,
        true_m: f64,
        measured_m: f64,
        pos_m: f64,
        vel_m_s: f64,
        state: &RadarGatingState,
        beam_state: BeamState,
    ) {
        let latency_ms = state.system_latency_ms;
        self.total_frames = self.total_frames.saturating_add(1);
        let cmd_active = beam_state == BeamState::BeamActive;
        if cmd_active {
            self.active_frames = self.active_frames.saturating_add(1);
        }
        let physical_active = step_physical_beam_delay(
            &mut self.beam_delay_ring,
            &mut self.beam_delay_cursor,
            latency_ms,
            cmd_active,
        );
        if physical_active {
            self.physical_active_frames = self.physical_active_frames.saturating_add(1);
        }
        let true_mm = true_m * 1.0e3;
        if physical_active && true_mm > GEOGRAPHIC_MISS_CEILING_MM {
            self.geographic_miss_frames = self.geographic_miss_frames.saturating_add(1);
        }
        let pred_mm = (pos_m + vel_m_s * (latency_ms * 1.0e-3)) * 1.0e3;
        let meas_mm = measured_m * 1.0e3;
        self.push_trace_rings(
            self.sim_time_s,
            true_mm,
            meas_mm,
            pred_mm,
            physical_active,
            beam_state,
        );
        self.update_evaluation_metrics(true_mm, meas_mm, pos_m * 1.0e3, pred_mm, state);
    }

    fn push_trace_rings(
        &mut self,
        t: f64,
        true_mm: f64,
        meas_mm: f64,
        pred_mm: f64,
        physical_active: bool,
        beam_state: BeamState,
    ) {
        let duty_pct = self.duty_cycle_percent();
        let flag = |b: bool| [t, f64::from(u8::from(b))];
        push_ring(&mut self.true_trace_mm, [t, true_mm]);
        push_ring(&mut self.physical_beam_trace, flag(physical_active));
        push_ring(&mut self.measured_trace_mm, [t, meas_mm]);
        push_ring(&mut self.kalman_trace_mm, [t, pred_mm]);
        push_ring(&mut self.duty_cycle_trace_pct, [t, duty_pct]);
        push_ring(
            &mut self.beam_active_trace,
            flag(beam_state == BeamState::BeamActive),
        );
        push_ring(
            &mut self.beam_hold_trace,
            flag(beam_state == BeamState::BeamHold),
        );
        push_ring(
            &mut self.interlock_trace,
            flag(beam_state == BeamState::SafetyInterlock),
        );
    }

    fn update_evaluation_metrics(
        &mut self,
        true_mm: f64,
        meas_mm: f64,
        kalman_pos_mm: f64,
        pred_mm: f64,
        state: &RadarGatingState,
    ) {
        self.latest_disp_mm = [true_mm, meas_mm, pred_mm];
        push_ring(&mut *self.true_disp_ring_mm, true_mm);
        push_ring(&mut *self.predicted_disp_ring_mm, kalman_pos_mm);
        let r = pearson_correlation(&*self.predicted_disp_ring_mm, &*self.true_disp_ring_mm)
            .unwrap_or(0.0);
        let rmse = root_mean_square_error(&*self.predicted_disp_ring_mm, &*self.true_disp_ring_mm)
            .unwrap_or(0.0);
        let base_r_mm = state.primary_target.range_m() * 1.0e3;
        let err = spatial_accuracy_error(base_r_mm + kalman_pos_mm, base_r_mm + true_mm);
        self.eval_metrics = [r, rmse, if err.is_finite() { err } else { 0.0 }];
    }

    /// Builds a stack-resident [`GatingTelemetry`] snapshot from the current pipeline state.
    #[must_use]
    pub fn latest_telemetry(&self, step_event: bool) -> GatingTelemetry {
        GatingTelemetry {
            seq: 0,
            sim_time_s: self.sim_time_s,
            true_disp_mm: self.latest_disp_mm[0],
            measured_disp_mm: self.latest_disp_mm[1],
            predicted_disp_mm: self.latest_disp_mm[2],
            velocity_mm_s: self.fast_kalman.state[1] * 1.0e3,
            duty_cycle_pct: self.duty_cycle_percent(),
            beam_state: self.gating.state(),
            pearson_r: self.eval_metrics[0],
            rmse_mm: self.eval_metrics[1],
            spatial_err_pct: self.eval_metrics[2],
            step_event,
        }
    }

    /// Current cumulative LINAC beam-on duty cycle in `[0.0, 100.0]` percent.
    pub fn duty_cycle_percent(&self) -> f64 {
        ratio_pct(self.active_frames, self.total_frames)
    }

    /// Cumulative delayed physical LINAC beam-on duty cycle in `[0.0, 100.0]` percent.
    pub fn physical_duty_cycle_percent(&self) -> f64 {
        ratio_pct(self.physical_active_frames, self.total_frames)
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

fn ratio_pct(active: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * (active as f64) / (total as f64)
    }
}

fn push_ring<T: Copy>(buf: &mut [T], item: T) {
    let len = buf.len();
    if len > 1 {
        buf.copy_within(1..len, 0);
    }
    if len > 0 {
        buf[len - 1] = item;
    }
}

// [cite:mmwave_radiotherapy_setup]
