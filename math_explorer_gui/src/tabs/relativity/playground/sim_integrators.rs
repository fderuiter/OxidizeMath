#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Relativistic Doppler three-velocity helper and polyphonic [`SonificationFrame`] builder
//! for [`GeodesicPlaygroundTool`].

use crate::audio::frame::{SonificationFrame, WaveformPreset};
use crate::audio::mappings::{geodesic_bundle_frame, GeodesicVoiceSample};

use super::sim::MAX_BUNDLE_RAYS;
use super::{GeodesicMode, GeodesicPlaygroundTool};

/// Converts coordinate radial and angular velocities into a bounded planar three-velocity $\boldsymbol{\beta} = \mathbf{v} / c$.
#[inline]
pub(super) fn planar_velocity_beta(r: f64, phi: f64, dr_dct: f64, dphi_dct: f64) -> [f64; 3] {
    let (sin_p, cos_p) = phi.sin_cos();
    let v_rad = dr_dct.clamp(-0.92, 0.92);
    let v_tan = (r * dphi_dct).clamp(-0.92, 0.92);
    let vx = v_rad * cos_p - v_tan * sin_p;
    let vy = v_rad * sin_p + v_tan * cos_p;
    let mag_sq = vx * vx + vy * vy;
    if mag_sq > 0.90 {
        let s = 0.948 / mag_sq.sqrt();
        [vx * s, vy * s, 0.0]
    } else {
        [vx, vy, 0.0]
    }
}

impl GeodesicPlaygroundTool {
    /// Builds a zero-allocation polyphonic [`SonificationFrame`] across active geodesic rays.
    pub fn build_audio_frame(&self) -> SonificationFrame {
        let mut voices = [GeodesicVoiceSample {
            clock_rate: 0.0,
            source_beta: [0.0; 3],
            line_of_sight: [1.0, 0.0, 0.0],
            magnetic_u_b: 0.0,
            pan: 0.0,
            halted: true,
        }; MAX_BUNDLE_RAYS];
        let count = self.active_ray_count.clamp(1, MAX_BUNDLE_RAYS);
        let rs = self.state.scale_radius.max(0.5);
        for (slot, ray) in self.bundle_rays.iter().take(count).enumerate() {
            if ray.samples.is_empty() {
                continue;
            }
            let idx = self.state.scrub_index.min(ray.samples.len() - 1);
            let sample = ray.samples[idx];
            let at_end_halted = ray.halted && idx + 1 >= ray.samples.len();
            let proj = self.state.camera.project(&sample.embed_xyz);
            let r_eff = sample.radial_coord.abs().max(0.5 * rs);
            let audio_clock = match self.state.mode {
                GeodesicMode::NullPhoton => sample.grav_dilation.clamp(0.05, 1.0),
                GeodesicMode::TimelikeParticle => sample.clock_rate,
            };
            voices[slot] = GeodesicVoiceSample {
                clock_rate: audio_clock,
                source_beta: sample.velocity_beta,
                line_of_sight: [0.85, 0.0, 0.526],
                magnetic_u_b: 4.0 * (rs / r_eff).powi(3),
                pan: (proj[0] / 12.0).clamp(-1.0, 1.0) as f32,
                halted: at_end_halted,
            };
        }
        let waveform = match self.state.mode {
            GeodesicMode::NullPhoton => WaveformPreset::QuantumModal,
            GeodesicMode::TimelikeParticle => WaveformPreset::FourierWarm,
        };
        geodesic_bundle_frame(&voices[..count], waveform)
    }

    pub(super) fn sample_geodesic_wav_frames(
        &mut self,
        num_frames: usize,
    ) -> Vec<SonificationFrame> {
        use crate::framework::InteractiveTool;
        if !self.has_finite_parameters() {
            return Vec::new();
        }
        if self.samples.is_empty() {
            self.rebuild_trajectory(560);
        }
        let count = num_frames.max(1);
        let len = self.samples.len();
        let prev_scrub = self.state.scrub_index;
        let denom = count.saturating_sub(1).max(1);
        let mut out = Vec::with_capacity(count);
        for idx in 0..count {
            self.state.scrub_index = if len <= 1 {
                0
            } else {
                (idx * (len - 1)) / denom
            };
            out.push(self.build_audio_frame().sanitize());
        }
        self.state.scrub_index = prev_scrub;
        out
    }
}
