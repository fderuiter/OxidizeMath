#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Thin UI trajectory orchestration delegating 4D, wormhole, custom-lapse, and `PhotonFan`
//! RK4 integration to `math_explorer::physics::high_energy`.

use std::f64::consts::TAU;

use math_explorer::physics::high_energy::{
    integrate_custom_lapse_worldline_into, integrate_custom_wormhole_worldline_into,
    integrate_kerr_worldline_into, integrate_photon_fan_bundle_into,
    integrate_schwarzschild_worldline_into, integrate_wormhole_worldline_into, GeodesicKind,
    KerrEquatorialMetric, WorldlineLaunchSpec,
};
pub use math_explorer::physics::high_energy::{
    BundleRay, TrajectorySample, MAX_BUNDLE_RAYS, MAX_TRAJECTORY_SAMPLES,
};

use super::{GeodesicInteractionMode, GeodesicMode, GeodesicPlaygroundTool, SpacetimePreset};

#[inline]
pub(super) const fn to_geodesic_kind(mode: GeodesicMode) -> GeodesicKind {
    match mode {
        GeodesicMode::NullPhoton => GeodesicKind::Null,
        GeodesicMode::TimelikeParticle => GeodesicKind::Timelike,
    }
}

impl GeodesicPlaygroundTool {
    pub(super) fn launch_spec(&self) -> WorldlineLaunchSpec {
        WorldlineLaunchSpec {
            kind: to_geodesic_kind(self.state.mode),
            scale_radius: self.state.scale_radius,
            spin_chi: self.state.spin_chi,
            initial_radius: self.state.initial_radius,
            launch_azimuth: self.state.launch_azimuth,
            radial_bias: self.state.radial_bias,
            step_size: self.state.step_size,
        }
    }

    pub(super) fn min_launch_radius(&self) -> f64 {
        let s = &self.state;
        match s.spacetime {
            SpacetimePreset::Schwarzschild => 1.15 * s.scale_radius,
            SpacetimePreset::KerrEquatorial => {
                let a = 0.5 * s.scale_radius * s.spin_chi.clamp(-0.95, 0.95);
                let r_plus = KerrEquatorialMetric::new(s.scale_radius, a)
                    .map_or(s.scale_radius, |k| k.outer_horizon_radius());
                1.15 * r_plus
            }
            SpacetimePreset::CustomRadialLapse => {
                (1.15 * self.sandbox.estimate_lapse_horizon(s.scale_radius)).max(0.5)
            }
            SpacetimePreset::MorrisThorneWormhole | SpacetimePreset::CustomWormholeProfile => -12.0,
        }
    }

    pub(super) fn clamp_state_in_place(&mut self) {
        self.state.scale_radius = finite_clamp(self.state.scale_radius, 0.5, 5.0, 2.0);
        self.state.spin_chi = finite_clamp(self.state.spin_chi, -0.95, 0.95, 0.65);
        let min_r = self.min_launch_radius();
        let max_r = if self.state.spacetime.is_wormhole() {
            12.0
        } else {
            18.0
        };
        let s = &mut self.state;
        s.initial_radius =
            finite_clamp(s.initial_radius, min_r, max_r, 10.0_f64.clamp(min_r, max_r));
        s.impact_or_momentum = finite_clamp(s.impact_or_momentum, -14.0, 14.0, 5.35);
        s.twin_delta_b = finite_clamp(s.twin_delta_b, -2.0, 2.0, 0.0);
        s.radial_bias = finite_clamp(s.radial_bias, -0.98, 0.98, -0.82);
        s.step_size = finite_clamp(s.step_size, 0.005, 0.12, 0.04);
        s.steps_per_frame = s.steps_per_frame.clamp(1, 32);
        s.fan_count = s.fan_count.clamp(2, MAX_BUNDLE_RAYS);
        s.fan_spread = finite_clamp(s.fan_spread, 0.1, 8.0, 2.4);
        s.launch_azimuth = finite_clamp(s.launch_azimuth, -TAU, TAU, 0.0);
        if !s.camera.is_valid() {
            s.camera = crate::camera::Camera3D::new(0.48, -0.62, 1.0);
        }
    }

    /// Recomputes primary/twin or up to 16 `PhotonFan` geodesic buffers with zero heap allocations.
    pub fn rebuild_trajectory(&mut self, step_budget: usize) {
        let count = step_budget.clamp(2, MAX_TRAJECTORY_SAMPLES);
        if self.state.interaction_mode == GeodesicInteractionMode::PhotonFan {
            self.rebuild_photon_fan_bundle(count);
        } else {
            self.rebuild_primary_and_twin(count);
        }
        self.state.scrub_index = if self.samples.is_empty() {
            0
        } else {
            self.state.scrub_index.min(self.samples.len() - 1)
        };
    }

    fn rebuild_photon_fan_bundle(&mut self, count: usize) {
        self.twin_samples.clear();
        self.twin_halted_near_singularity = false;
        let spec = self.launch_spec();
        let spacetime = self.state.spacetime;
        let sandbox = &self.sandbox;
        let mut bundle = core::mem::take(&mut self.bundle_rays);
        let (ray_count, all_halted, center_energy) = integrate_photon_fan_bundle_into(
            self.state.fan_count,
            self.state.impact_or_momentum,
            self.state.fan_spread,
            count,
            &mut bundle,
            |steps, impact_b, out| {
                integrate_preset_worldline(spacetime, &spec, sandbox, steps, impact_b, out)
            },
        );
        self.active_ray_count = ray_count;
        self.launch_energy = center_energy;
        self.twin_launch_energy = center_energy;
        self.samples.clear();
        if ray_count > 0 {
            self.samples
                .extend_from_slice(&bundle[ray_count / 2].samples);
        }
        self.bundle_rays = bundle;
        self.halted_near_singularity = all_halted;
    }

    fn rebuild_primary_and_twin(&mut self, count: usize) {
        self.active_ray_count = 1;
        self.samples.clear();
        self.twin_samples.clear();
        self.halted_near_singularity = false;
        self.twin_halted_near_singularity = false;
        let b0 = self.state.impact_or_momentum;
        let spec = self.launch_spec();
        let mut primary = core::mem::take(&mut self.samples);
        let (halted, e_val) = integrate_preset_worldline(
            self.state.spacetime,
            &spec,
            &self.sandbox,
            count,
            b0,
            &mut primary,
        );
        self.bundle_rays[0].samples.clear();
        self.bundle_rays[0].samples.extend_from_slice(&primary);
        self.bundle_rays[0].halted = halted;
        self.bundle_rays[0].impact_b = b0;
        self.samples = primary;
        self.halted_near_singularity = halted;
        self.launch_energy = e_val;

        if self.state.twin_delta_b.abs() > 1e-9 {
            let b_twin = (b0 + self.state.twin_delta_b).clamp(-16.0, 16.0);
            let mut twin = core::mem::take(&mut self.twin_samples);
            let (t_halted, t_e) = integrate_preset_worldline(
                self.state.spacetime,
                &spec,
                &self.sandbox,
                count,
                b_twin,
                &mut twin,
            );
            self.twin_samples = twin;
            self.twin_halted_near_singularity = t_halted;
            self.twin_launch_energy = t_e;
        } else {
            self.twin_launch_energy = e_val;
        }
    }

    /// Appends up to `extra_steps` RK4 steps onto all active rays during live playback.
    pub fn advance_steps(&mut self, extra_steps: usize) {
        if self.samples.is_empty() {
            self.rebuild_trajectory(260);
        }
        let current_len = self.samples.len();
        if self.halted_near_singularity || current_len > MAX_TRAJECTORY_SAMPLES {
            if self.state.is_playing {
                self.state.is_playing = false;
                crate::audio::notify_playback_paused();
            }
            return;
        }
        self.rebuild_trajectory((current_len + extra_steps).min(MAX_TRAJECTORY_SAMPLES));
        self.state.scrub_index = self.samples.len().saturating_sub(1);
        self.step_lensing_backgrounds();
    }
}

fn integrate_preset_worldline(
    spacetime: SpacetimePreset,
    spec: &WorldlineLaunchSpec,
    sandbox: &super::MetricSandbox,
    steps: usize,
    impact_val: f64,
    out: &mut Vec<TrajectorySample>,
) -> (bool, f64) {
    match spacetime {
        SpacetimePreset::Schwarzschild => {
            integrate_schwarzschild_worldline_into(spec, steps, impact_val, out)
        }
        SpacetimePreset::KerrEquatorial => {
            integrate_kerr_worldline_into(spec, steps, impact_val, out)
        }
        SpacetimePreset::MorrisThorneWormhole => {
            integrate_wormhole_worldline_into(spec, steps, impact_val, out)
        }
        SpacetimePreset::CustomRadialLapse => match sandbox.lapse_bar.active_compiled() {
            Some(compiled) => integrate_custom_lapse_worldline_into(
                spec,
                compiled,
                sandbox.lapse_bar.param_slice(),
                steps,
                impact_val,
                out,
            ),
            None => {
                out.clear();
                (false, 1.0)
            }
        },
        SpacetimePreset::CustomWormholeProfile => match sandbox.wormhole_bar.active_compiled() {
            Some(compiled) => integrate_custom_wormhole_worldline_into(
                spec,
                compiled,
                sandbox.wormhole_bar.param_slice(),
                steps,
                impact_val,
                out,
            ),
            None => {
                out.clear();
                (false, 1.0)
            }
        },
    }
}

#[inline]
fn finite_clamp(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        fallback
    }
}
