#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::SimWorkerState;
use math_explorer::physics::optics::refraction::OpticsPreset;
use serde::{Deserialize, Serialize};

pub(crate) fn default_grid_size() -> usize {
    128
}

pub(crate) fn default_ray_count() -> usize {
    15
}

pub(crate) fn default_adc_bits() -> u32 {
    12
}

/// Validated serializable state schema for `OpticsWaveTool` presets and `#tab=optics_waves` URLs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpticsWaveToolState {
    /// Active optical element or aperture preset.
    pub preset: OpticsPreset,
    /// Square FDTD grid dimension (`64..=256`).
    #[serde(default = "default_grid_size")]
    pub grid_size: usize,
    /// Coherent source frequency $f$ (`4.0..=24.0`).
    pub frequency: f64,
    /// Double-slit center-to-center separation $d$ (`0.08..=0.40`).
    pub slit_separation: f64,
    /// Double-slit opening width $a$ (`0.015..=0.14`).
    pub aperture_width: f64,
    /// Characteristic lens radius $R$ (`0.10..=0.34`).
    pub lens_radius: f64,
    /// Virtual vertical detector x-coordinate (`0.45..=0.90`).
    pub detector_x: f64,
    /// Fraction of the 2D CFL stability step (`0.10..=1.00`).
    pub cfl_scale: f64,
    /// Whether geometric ray trajectories and caustics are overlaid.
    pub show_rays: bool,
    /// Whether time-smoothed irradiance $\langle u^2\rangle$ is rendered.
    pub show_irradiance: bool,
    /// Number of rays in the Hamiltonian bundle (`3..=31`).
    #[serde(default = "default_ray_count")]
    pub ray_count: usize,
    /// Quantizer bit depth (`4..=16`).
    #[serde(default = "default_adc_bits")]
    pub adc_bits: u32,
    /// Optional worker execution state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<SimWorkerState>,
    /// Legacy boolean pause flag accepted on deserialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Deferred start flag keeping hidden tabs idle until first rendered frame.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pending_initial_start: bool,
}

impl OpticsWaveToolState {
    /// Validates finiteness and domain bounds for all 2D wave and ray parameters.
    pub fn validate(&self) -> Result<(), String> {
        if !(64..=256).contains(&self.grid_size) {
            return Err(format!(
                "OpticsWaveTool grid_size {} must be within 64..=256",
                self.grid_size
            ));
        }
        if !(3..=31).contains(&self.ray_count) {
            return Err(format!(
                "OpticsWaveTool ray_count {} must be within 3..=31",
                self.ray_count
            ));
        }
        if !(4..=16).contains(&self.adc_bits) {
            return Err(format!(
                "OpticsWaveTool adc_bits {} must be within 4..=16",
                self.adc_bits
            ));
        }
        let checks = [
            ("frequency", self.frequency, 4.0, 24.0),
            ("slit_separation", self.slit_separation, 0.08, 0.40),
            ("aperture_width", self.aperture_width, 0.015, 0.14),
            ("lens_radius", self.lens_radius, 0.10, 0.34),
            ("detector_x", self.detector_x, 0.45, 0.90),
            ("cfl_scale", self.cfl_scale, 0.10, 1.00),
        ];
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "OpticsWaveTool parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn resolve_target_state(&self) -> SimWorkerState {
        self.state.unwrap_or_else(|| {
            if self.paused.unwrap_or(false) {
                SimWorkerState::Ready
            } else {
                SimWorkerState::Running
            }
        })
    }
}
