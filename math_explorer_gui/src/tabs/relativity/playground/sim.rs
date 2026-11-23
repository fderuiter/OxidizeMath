#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation RK4 multi-ray bundle orchestration, exact Kerr potentials, and audio extraction.

use std::f64::consts::TAU;

use math_explorer::physics::high_energy::{
    effective_potential, GeodesicKind, KerrEquatorialMetric, MorrisThorneWormhole,
};

use crate::audio::frame::{SonificationFrame, WaveformPreset};
use crate::audio::mappings::{geodesic_bundle_frame, GeodesicVoiceSample};

use super::sim_integrators::{
    integrate_kerr_ray, integrate_schwarzschild_ray, integrate_wormhole_ray, RayLaunchSpec,
};
use super::{GeodesicInteractionMode, GeodesicMode, GeodesicPlaygroundTool, SpacetimePreset};

/// Fixed upper bound on stored trajectory samples per ray (pre-allocated once at startup).
pub const MAX_TRAJECTORY_SAMPLES: usize = 720;
/// Maximum simultaneous geodesics integrated in `PhotonFan` mode.
pub const MAX_BUNDLE_RAYS: usize = 16;

/// Pre-allocated trajectory storage for a single ray in a geodesic bundle.
#[derive(Debug, Clone)]
pub struct BundleRay {
    /// Sampled worldline points up to [`MAX_TRAJECTORY_SAMPLES`].
    pub samples: Vec<TrajectorySample>,
    /// Whether this ray halted at the horizon singularity guard.
    pub halted: bool,
    /// Impact parameter $b$ or angular momentum $\ell$ assigned to this ray.
    pub impact_b: f64,
}

impl BundleRay {
    /// Pre-allocates sample buffer for zero-allocation stepping.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            samples: Vec::with_capacity(cap),
            halted: false,
            impact_b: 0.0,
        }
    }
}

/// Single integrated worldline sample with 3D embedding coordinates, Kerr clocks, and Doppler velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectorySample {
    /// Coordinate time $ct$.
    pub coord_ct: f64,
    /// Observer proper parameter $\tau$ (or affine $\lambda$ for null rays).
    pub proper_tau: f64,
    /// Radial coordinate $r$ (or proper throat coordinate $l$ for wormholes).
    pub radial_coord: f64,
    /// Azimuthal angle $\varphi$.
    pub phi: f64,
    /// Radial derivative $dr/d\lambda$ or $dl/d\lambda$.
    pub radial_vel: f64,
    /// Coordinate angular velocity $\Omega = d\varphi / d(ct)$.
    pub dphi_dct: f64,
    /// Local spatial three-velocity $\boldsymbol{\beta} = \mathbf{v} / c$ for Doppler beaming.
    pub velocity_beta: [f64; 3],
    /// Embedded 3D point $[x, y, z]$ on Flamm's paraboloid or Morris-Thorne surface.
    pub embed_xyz: [f64; 3],
    /// Quadratic metric norm $g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu$.
    pub metric_norm: f64,
    /// Instantaneous proper-to-coordinate clock rate $d\tau / dt$ (or gravitational frequency factor).
    pub clock_rate: f64,
    /// Instantaneous static/ZAMO gravitational dilation factor.
    pub grav_dilation: f64,
    /// Instantaneous local Lorentz factor $\gamma(v)$.
    pub lorentz_gamma: f64,
}

impl GeodesicPlaygroundTool {
    pub(super) fn clamp_state_in_place(&mut self) {
        let s = &mut self.state;
        s.scale_radius = clamp_or(s.scale_radius, 0.5, 5.0, 2.0);
        s.spin_chi = clamp_or(s.spin_chi, -0.92, 0.92, 0.65);
        let min_r = match s.spacetime {
            SpacetimePreset::Schwarzschild => 1.15 * s.scale_radius,
            SpacetimePreset::KerrEquatorial => {
                let a = 0.5 * s.scale_radius * s.spin_chi;
                let r_plus = KerrEquatorialMetric::new(s.scale_radius, a)
                    .map(|k| k.outer_horizon_radius())
                    .unwrap_or(s.scale_radius);
                1.12 * r_plus
            }
            SpacetimePreset::MorrisThorneWormhole => -12.0,
        };
        s.initial_radius = clamp_or(s.initial_radius, min_r, 18.0, 10.0);
        s.impact_or_momentum = clamp_or(s.impact_or_momentum, -14.0, 14.0, 5.35);
        s.radial_bias = clamp_or(s.radial_bias, -0.98, 0.98, -0.82);
        s.step_size = clamp_or(s.step_size, 0.005, 0.12, 0.04);
        s.steps_per_frame = s.steps_per_frame.clamp(1, 32);
        s.fan_count = s.fan_count.clamp(2, MAX_BUNDLE_RAYS);
        s.fan_spread = clamp_or(s.fan_spread, 0.1, 8.0, 2.4);
        s.launch_azimuth = clamp_or(s.launch_azimuth, -TAU, TAU, 0.0);
        if !s.camera.is_valid() {
            s.camera = crate::camera::Camera3D::new(0.48, -0.62, 1.0);
        }
    }

    /// Recomputes 1 ray (`Orbit`/`Slingshot`) or up to 16 rays (`PhotonFan`) with zero heap allocations.
    pub fn rebuild_trajectory(&mut self, step_budget: usize) {
        let steps = step_budget.clamp(2, MAX_TRAJECTORY_SAMPLES - 1);
        let ray_count = if self.state.interaction_mode == GeodesicInteractionMode::PhotonFan {
            self.state.fan_count.clamp(2, MAX_BUNDLE_RAYS)
        } else {
            1
        };
        self.active_ray_count = ray_count;
        let center_idx = ray_count / 2;

        for idx in 0..ray_count {
            let offset = if ray_count > 1 {
                let frac = (idx as f64) / ((ray_count - 1) as f64) - 0.5;
                frac * self.state.fan_spread
            } else {
                0.0
            };
            let impact_b = (self.state.impact_or_momentum + offset).clamp(-16.0, 16.0);
            self.integrate_single_bundle_ray(idx, impact_b, steps);
        }

        self.samples.clear();
        if let Some(primary) = self.bundle_rays.get(center_idx) {
            self.samples.extend_from_slice(&primary.samples);
        }
        self.halted_near_singularity = self.bundle_rays[..ray_count].iter().all(|r| r.halted);
        if !self.samples.is_empty() {
            self.state.scrub_index = self.state.scrub_index.min(self.samples.len() - 1);
        } else {
            self.state.scrub_index = 0;
        }
    }

    /// Appends up to `extra_steps` RK4 steps onto all active rays during live playback.
    pub fn advance_steps(&mut self, extra_steps: usize) {
        if self.samples.is_empty() {
            self.rebuild_trajectory(260);
        }
        let max_len = self.bundle_rays[..self.active_ray_count.max(1)]
            .iter()
            .map(|r| r.samples.len())
            .max()
            .unwrap_or(0);
        if self.halted_near_singularity || max_len >= MAX_TRAJECTORY_SAMPLES - 1 {
            if self.state.is_playing {
                self.state.is_playing = false;
                crate::audio::notify_playback_paused();
            }
            return;
        }
        let target = (max_len + extra_steps).min(MAX_TRAJECTORY_SAMPLES - 1);
        self.rebuild_trajectory(target);
        self.state.scrub_index = self.samples.len().saturating_sub(1);
        self.step_lensing_backgrounds();
    }

    fn integrate_single_bundle_ray(&mut self, ray_idx: usize, impact_b: f64, steps: usize) {
        let launch = RayLaunchSpec {
            mode: self.state.mode,
            rs: self.state.scale_radius,
            spin_chi: self.state.spin_chi,
            r0_cfg: self.state.initial_radius,
            phi0: self.state.launch_azimuth,
            impact_b,
            radial_bias: self.state.radial_bias,
            dt: self.state.step_size,
            steps,
        };
        let spacetime = self.state.spacetime;
        let Some(ray) = self.bundle_rays.get_mut(ray_idx) else {
            return;
        };
        ray.samples.clear();
        ray.halted = false;
        ray.impact_b = impact_b;

        match spacetime {
            SpacetimePreset::Schwarzschild => integrate_schwarzschild_ray(ray, &launch),
            SpacetimePreset::KerrEquatorial => integrate_kerr_ray(ray, &launch),
            SpacetimePreset::MorrisThorneWormhole => integrate_wormhole_ray(ray, &launch),
        }
    }

    /// Evaluates exact Kerr or Schwarzschild or Morris-Thorne $V_{\text{eff}}(r)$ without fallbacks.
    pub(super) fn effective_potential_curve(&self, points: usize) -> Vec<[f64; 2]> {
        let n = points.max(16);
        let mut curve = Vec::with_capacity(n);
        let kind = match self.state.mode {
            GeodesicMode::NullPhoton => GeodesicKind::Null,
            GeodesicMode::TimelikeParticle => GeodesicKind::Timelike,
        };
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild => {
                let rs = self.state.scale_radius;
                for i in 0..n {
                    let r = 1.05 * rs + 16.0 * (i as f64) / ((n - 1) as f64);
                    if let Ok(v) = effective_potential(rs, r, self.state.impact_or_momentum, kind) {
                        curve.push([r, v]);
                    }
                }
            }
            SpacetimePreset::KerrEquatorial => {
                let rs = self.state.scale_radius;
                let a = 0.5 * rs * self.state.spin_chi;
                if let Ok(kerr) = KerrEquatorialMetric::new(rs, a) {
                    let r_min = 1.05 * kerr.outer_horizon_radius();
                    for i in 0..n {
                        let r = r_min + 16.0 * (i as f64) / ((n - 1) as f64);
                        if let Ok(v) =
                            kerr.effective_potential(r, 1.0, self.state.impact_or_momentum, kind)
                        {
                            curve.push([r, v]);
                        }
                    }
                }
            }
            SpacetimePreset::MorrisThorneWormhole => {
                if let Ok(wh) = MorrisThorneWormhole::new(self.state.scale_radius) {
                    for i in 0..n {
                        let l = -10.0 + 20.0 * (i as f64) / ((n - 1) as f64);
                        if let Ok(v) = wh.effective_potential(l, self.state.impact_or_momentum) {
                            curve.push([l, v]);
                        }
                    }
                }
            }
        }
        curve
    }

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
            voices[slot] = GeodesicVoiceSample {
                clock_rate: sample.clock_rate,
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

    /// Generates projected 2D curves for CSV/PNG/SVG export and `Plot3DViewport` rendering.
    pub(super) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::with_capacity(self.active_ray_count + 3);
        for (idx, ray) in self
            .bundle_rays
            .iter()
            .take(self.active_ray_count.max(1))
            .enumerate()
        {
            let limit = self
                .state
                .scrub_index
                .saturating_add(1)
                .min(ray.samples.len());
            if limit > 0 {
                let pts = ray.samples[..limit]
                    .iter()
                    .map(|s| self.state.camera.project(&s.embed_xyz))
                    .collect();
                datasets.push((format!("Geodesic Ray {idx} (b={:.2})", ray.impact_b), pts));
            }
        }
        let ring = self.reference_ring_projected(self.state.scale_radius, 0.0, 48);
        datasets.push(("Horizon or Throat Ring".to_string(), ring));
        datasets.push((
            "Effective Potential V_eff(r)".to_string(),
            self.effective_potential_curve(64),
        ));
        datasets
    }

    pub(super) fn reference_ring_projected(
        &self,
        radius: f64,
        z_height: f64,
        segments: usize,
    ) -> Vec<[f64; 2]> {
        let n = segments.max(12);
        let mut pts = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let angle = TAU * (i as f64) / (n as f64);
            let (s, c) = angle.sin_cos();
            pts.push(
                self.state
                    .camera
                    .project(&[radius * c, radius * s, z_height]),
            );
        }
        pts
    }
}

fn clamp_or(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        fallback
    }
}
