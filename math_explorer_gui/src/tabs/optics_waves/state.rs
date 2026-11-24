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

impl super::tool::OpticsWaveTool {
    pub(super) fn build_export_raster_image(&self) -> Option<eframe::egui::ColorImage> {
        use super::runner::{render_wavefield_image, OpticsWaveRunner, WaveRenderInput};
        use crate::async_sim::SimulationRunner;
        use eframe::egui::ColorImage;
        use std::sync::Arc;

        if let Some((w, h, ref px)) = self.stepped_pixels {
            if w > 0 && h > 0 && px.len() == w * h {
                return Some(ColorImage {
                    size: [w, h],
                    pixels: px.clone(),
                    ..Default::default()
                });
            }
        }
        if let Some(ref frame) = self.latest_frame {
            let (show_irradiance, high_contrast) = {
                let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                (p.show_irradiance, p.high_contrast)
            };
            if frame.field.len() == self.grid_size * self.grid_size {
                return Some(render_wavefield_image(WaveRenderInput {
                    field: &frame.field,
                    irradiance: &frame.irradiance,
                    refractive_index: &frame.refractive_index,
                    obstacle: &frame.obstacle,
                    width: self.grid_size,
                    height: self.grid_size,
                    show_irradiance,
                    high_contrast,
                }));
            }
        }
        if self.pixels.len() == self.grid_size * self.grid_size && !self.pixels.is_empty() {
            return Some(ColorImage {
                size: [self.grid_size, self.grid_size],
                pixels: self.pixels.clone(),
                ..Default::default()
            });
        }
        let mut preview = OpticsWaveRunner::new(Arc::clone(&self.params), self.grid_size);
        preview.step();
        let snap = preview.get_snapshot();
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        Some(ColorImage {
            size: [snap.width, snap.height],
            pixels: px.clone(),
            ..Default::default()
        })
    }

    pub(super) fn build_export_scalar_field(
        &self,
    ) -> Option<math_commons::math_kernel::marching_squares::ScalarField2D> {
        use super::runner::{OpticsFrameData, OpticsWaveRunner};
        use crate::async_sim::SimulationRunner;
        use math_commons::math_kernel::marching_squares::ScalarField2D;
        use std::sync::Arc;

        let show_irradiance = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .show_irradiance;
        if let Some((w, h, ref vals)) = self.stepped_scalar {
            if w > 0 && h > 0 && vals.len() == w * h {
                return Some(ScalarField2D {
                    width: w,
                    height: h,
                    values: vals.clone(),
                    periodic: false,
                    default_levels: 8,
                });
            }
        }
        let from_frame = |f: &OpticsFrameData| {
            if show_irradiance {
                f.irradiance.clone()
            } else {
                f.field.clone()
            }
        };
        if let Some(ref frame) = self.latest_frame {
            let values = from_frame(frame);
            if values.len() == self.grid_size * self.grid_size && self.grid_size > 0 {
                return Some(ScalarField2D {
                    width: self.grid_size,
                    height: self.grid_size,
                    values,
                    periodic: false,
                    default_levels: 8,
                });
            }
        }
        let mut preview = OpticsWaveRunner::new(Arc::clone(&self.params), self.grid_size);
        preview.step();
        let snap = preview.get_snapshot();
        let frame = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<OpticsFrameData>())?;
        Some(ScalarField2D {
            width: snap.width,
            height: snap.height,
            values: from_frame(frame),
            periodic: false,
            default_levels: 8,
        })
    }

    pub(super) fn advance_optics_step(&mut self) {
        use super::runner::{OpticsFrameData, OpticsWaveRunner};
        use crate::async_sim::SimulationRunner;
        use std::sync::Arc;

        let show_irradiance = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .show_irradiance;
        let runner = self
            .step_runner
            .get_or_insert_with(|| OpticsWaveRunner::new(Arc::clone(&self.params), self.grid_size));
        for _ in 0..4 {
            runner.step();
        }
        let snap = runner.get_snapshot();
        let px = snap.pixels.read().unwrap_or_else(|e| e.into_inner());
        self.stepped_pixels = Some((snap.width, snap.height, px.clone()));
        if let Some(frame) = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<OpticsFrameData>())
        {
            let vals = if show_irradiance {
                frame.irradiance.clone()
            } else {
                frame.field.clone()
            };
            self.stepped_scalar = Some((snap.width, snap.height, vals));
        }
    }
}
