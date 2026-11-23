//! Serializable configuration state and `ToolStateBounds` invariants for `RadarGatingTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::ToolStateBounds;

/// Hard upper bound on ADC samples per FMCW chirp to guarantee 60 FPS and URL budget compliance.
pub const MAX_SAMPLES_PER_CHIRP: usize = 64;
/// Minimum ADC samples per chirp required for subspace estimation.
pub const MIN_SAMPLES_PER_CHIRP: usize = 16;
/// Hard upper bound on coherent MIMO snapshots accumulated per MUSIC frame.
pub const MAX_SNAPSHOTS: usize = 32;
/// Minimum coherent snapshots required for covariance conditioning.
pub const MIN_SNAPSHOTS: usize = 6;
/// Hard upper bound on spatial range sweep bins for CZT and MUSIC evaluation.
pub const MAX_SWEEP_BINS: usize = 512;
/// Minimum spatial range sweep bins.
pub const MIN_SWEEP_BINS: usize = 64;

/// A point scatterer inside the 2D clinical treatment room (`+y` is radar boresight depth).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScattererPoint {
    /// Lateral cross-range offset in meters (`[-1.5, 1.5]`).
    pub x_m: f64,
    /// Down-range depth from the 60 GHz radar aperture in meters (`[0.20, 2.50]`).
    pub y_m: f64,
    /// Relative radar cross-section amplitude (`[0.05, 5.0]`).
    pub rcs: f64,
}

impl ScattererPoint {
    /// Euclidean slant range $R = \sqrt{x^2 + y^2}$ in meters.
    pub fn range_m(&self) -> f64 {
        self.x_m.hypot(self.y_m)
    }

    /// Off-boresight azimuth $\theta = \operatorname{atan2}(x, y)$ in radians.
    pub fn angle_rad(&self) -> f64 {
        self.x_m.atan2(self.y_m)
    }

    /// Validates finite room coordinates and physical RCS bounds.
    pub fn validate(&self, label: &str) -> Result<(), String> {
        if !self.x_m.is_finite() || !(-1.5..=1.5).contains(&self.x_m) {
            return Err(format!("{label}.x_m out of [-1.5, 1.5]: {}", self.x_m));
        }
        if !self.y_m.is_finite() || !(0.20..=2.50).contains(&self.y_m) {
            return Err(format!("{label}.y_m out of [0.20, 2.50]: {}", self.y_m));
        }
        if !self.rcs.is_finite() || !(0.05..=5.0).contains(&self.rcs) {
            return Err(format!("{label}.rcs out of [0.05, 5.0]: {}", self.rcs));
        }
        Ok(())
    }
}

/// Complete serializable state for the 60 GHz MIMO radar super-resolution and LINAC gating twin.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RadarGatingState {
    /// Primary thoracic chest-wall scatterer tracked for respiratory gating.
    pub primary_target: ScattererPoint,
    /// Secondary scatterer (hospital gown, immobilization mask, or couch reflector).
    pub secondary_target: ScattererPoint,
    /// FMCW sweep bandwidth $B$ in GHz (`[0.25, 10.0]`).
    pub bandwidth_ghz: f64,
    /// ULA digital beamformer steering angle in degrees (`[-65.0, 65.0]`).
    pub steering_deg: f64,
    /// Number of virtual ULA receive elements (`[2, 16]`).
    pub num_antennas: usize,
    /// Fast-time ADC samples per chirp (`[16, 64]`).
    pub samples_per_chirp: usize,
    /// Slow-time snapshots used by `MusicEstimator` (`[6, 32]`).
    pub snapshots: usize,
    /// Spatial range bins evaluated in CZT and MUSIC sweeps (`[64, 512]`).
    pub sweep_bins: usize,
    /// Deterministic thermal IQ noise standard deviation (`[0.0, 1.0]`).
    pub noise_sigma: f64,
    /// Sliding window length for `EllipticalFilter` static clutter rejection (`[4, 64]`).
    pub clutter_window: usize,
    /// Stationary room/couch clutter phasor magnitude (`[0.0, 5.0]`).
    pub static_clutter_gain: f64,
    /// Peak-to-baseline respiratory excursion in millimeters (`[0.5, 25.0]`).
    pub breath_amplitude_mm: f64,
    /// Single respiratory breath cycle duration in seconds (`[1.0, 20.0]`).
    pub breath_period_s: f64,
    /// `ConstantVelocityModel` process noise variance $Q$ (`[1e-8, 1.0]`).
    pub kalman_process_var: f64,
    /// `ConstantVelocityModel` measurement noise variance $R$ (`[1e-9, 1.0]`).
    pub kalman_meas_var: f64,
    /// Schmitt-trigger end-exhale gating threshold in millimeters (`[-20.0, 20.0]`).
    pub gating_threshold_mm: f64,
    /// Schmitt-trigger half-band hysteresis in millimeters (`[0.05, 10.0]`).
    pub gating_hysteresis_mm: f64,
    /// LINAC beam-delivery lookahead compensation latency in milliseconds (`[0.0, 500.0]`).
    pub system_latency_ms: f64,
    /// Deterministic `OxidizeRng` seed for reproducible native and WASM traces.
    pub rng_seed: u64,
}

impl Default for RadarGatingState {
    fn default() -> Self {
        Self {
            primary_target: ScattererPoint {
                x_m: 0.02,
                y_m: 0.700,
                rcs: 1.0,
            },
            secondary_target: ScattererPoint {
                x_m: -0.02,
                y_m: 0.716,
                rcs: 0.90,
            },
            bandwidth_ghz: 4.0,
            steering_deg: 0.0,
            num_antennas: 8,
            samples_per_chirp: 40,
            snapshots: 20,
            sweep_bins: 200,
            noise_sigma: 0.015,
            clutter_window: 24,
            static_clutter_gain: 0.40,
            breath_amplitude_mm: 5.0,
            breath_period_s: 4.0,
            kalman_process_var: 2.0e-4,
            kalman_meas_var: 2.0e-5,
            gating_threshold_mm: -1.5,
            gating_hysteresis_mm: 0.5,
            system_latency_ms: 50.0,
            rng_seed: 42,
        }
    }
}

impl RadarGatingState {
    /// Euclidean slant-range separation $|R_2 - R_1|$ between the two scatterers in meters.
    pub fn target_separation_m(&self) -> f64 {
        (self.secondary_target.range_m() - self.primary_target.range_m()).abs()
    }

    fn validate_rf_bounds(&self) -> Result<(), String> {
        self.primary_target.validate("primary_target")?;
        self.secondary_target.validate("secondary_target")?;
        check_range("bandwidth_ghz", self.bandwidth_ghz, 0.25, 10.0)?;
        check_range("steering_deg", self.steering_deg, -65.0, 65.0)?;
        check_usize("num_antennas", self.num_antennas, 2, 16)?;
        check_usize(
            "samples_per_chirp",
            self.samples_per_chirp,
            MIN_SAMPLES_PER_CHIRP,
            MAX_SAMPLES_PER_CHIRP,
        )?;
        check_usize("snapshots", self.snapshots, MIN_SNAPSHOTS, MAX_SNAPSHOTS)?;
        check_usize(
            "sweep_bins",
            self.sweep_bins,
            MIN_SWEEP_BINS,
            MAX_SWEEP_BINS,
        )?;
        check_range("noise_sigma", self.noise_sigma, 0.0, 1.0)?;
        check_usize("clutter_window", self.clutter_window, 4, 64)?;
        check_range("static_clutter_gain", self.static_clutter_gain, 0.0, 5.0)
    }

    fn validate_gating_bounds(&self) -> Result<(), String> {
        check_range("breath_amplitude_mm", self.breath_amplitude_mm, 0.5, 25.0)?;
        check_range("breath_period_s", self.breath_period_s, 1.0, 20.0)?;
        check_range("kalman_process_var", self.kalman_process_var, 1.0e-8, 1.0)?;
        check_range("kalman_meas_var", self.kalman_meas_var, 1.0e-9, 1.0)?;
        check_range("gating_threshold_mm", self.gating_threshold_mm, -20.0, 20.0)?;
        check_range(
            "gating_hysteresis_mm",
            self.gating_hysteresis_mm,
            0.05,
            10.0,
        )?;
        check_range("system_latency_ms", self.system_latency_ms, 0.0, 500.0)
    }
}

fn check_range(name: &str, val: f64, min: f64, max: f64) -> Result<(), String> {
    if !val.is_finite() || !(min..=max).contains(&val) {
        return Err(format!("{name} out of [{min}, {max}]: {val}"));
    }
    Ok(())
}

fn check_usize(name: &str, val: usize, min: usize, max: usize) -> Result<(), String> {
    if !(min..=max).contains(&val) {
        return Err(format!("{name} out of [{min}, {max}]: {val}"));
    }
    Ok(())
}

impl RadarGatingState {
    /// Strict `ToolStateBounds` fitting comfortably within `MAX_DECODED_URL_STATE_BYTES` (`4 KiB`).
    pub const STATE_BOUNDS: ToolStateBounds = ToolStateBounds::new(4096, 8, 64);

    /// Validates all RF, spatial, Kalman, and gating bounds.
    pub fn validate_bounds(&self) -> Result<(), String> {
        self.validate_rf_bounds()?;
        self.validate_gating_bounds()
    }
}

// [cite:mmwave_radiotherapy_setup]
