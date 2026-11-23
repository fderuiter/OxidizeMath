#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation RK4 trajectory integration, Flamm/wormhole embeddings, and twin clock math.

use std::f64::consts::{FRAC_PI_2, TAU};

use math_explorer::nalgebra::SVector;
use math_explorer::physics::high_energy::{
    combined_clock_rate, effective_potential, flamm_paraboloid_z, GeodesicKind,
    KerrEquatorialMetric, MorrisThorneWormhole, SchwarzschildBlackHole, C,
};
use math_explorer::pure_math::analysis::ode::{ArrayState, RungeKutta4};
use math_explorer::pure_math::tensor::{GeodesicState, GeodesicStepStatus, GeodesicSystem};

use super::{GeodesicMode, GeodesicPlaygroundTool, SpacetimePreset};

/// Fixed upper bound on stored trajectory samples (pre-allocated once at tool startup).
pub const MAX_TRAJECTORY_SAMPLES: usize = 720;

/// Single integrated worldline sample with 3D embedding coordinates and clock readouts.
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
    /// Embedded 3D point $[x, y, z]$ on Flamm's paraboloid or Morris-Thorne surface.
    pub embed_xyz: [f64; 3],
    /// Quadratic metric norm $g_{\mu\nu}\dot{x}^\mu\dot{x}^\nu$.
    pub metric_norm: f64,
    /// Instantaneous proper-to-coordinate clock rate $d\tau / dt$.
    pub clock_rate: f64,
    /// Instantaneous gravitational dilation $\sqrt{1 - R_s/r}$ (1.0 for zero-tidal wormhole).
    pub grav_dilation: f64,
    /// Instantaneous local Lorentz factor $\gamma(v)$.
    pub lorentz_gamma: f64,
}

impl GeodesicPlaygroundTool {
    pub(super) fn clamp_state_in_place(&mut self) {
        let s = &mut self.state;
        s.scale_radius = if s.scale_radius.is_finite() {
            s.scale_radius.clamp(0.5, 5.0)
        } else {
            2.0
        };
        s.spin_chi = if s.spin_chi.is_finite() {
            s.spin_chi.clamp(-0.92, 0.92)
        } else {
            0.65
        };
        let min_r = match s.spacetime {
            SpacetimePreset::Schwarzschild | SpacetimePreset::KerrEquatorial => {
                1.15 * s.scale_radius
            }
            SpacetimePreset::MorrisThorneWormhole => -12.0,
        };
        s.initial_radius = if s.initial_radius.is_finite() {
            s.initial_radius.clamp(min_r, 18.0)
        } else {
            10.0
        };
        s.impact_or_momentum = if s.impact_or_momentum.is_finite() {
            s.impact_or_momentum.clamp(-14.0, 14.0)
        } else {
            5.35
        };
        s.radial_bias = if s.radial_bias.is_finite() {
            s.radial_bias.clamp(-0.98, 0.98)
        } else {
            -0.82
        };
        s.step_size = if s.step_size.is_finite() {
            s.step_size.clamp(0.005, 0.12)
        } else {
            0.04
        };
        s.steps_per_frame = s.steps_per_frame.clamp(1, 32);
        if !s.camera.is_valid() {
            s.camera = crate::camera::Camera3D::new(0.48, -0.62, 1.0);
        }
    }

    /// Recomputes the geodesic buffer up to `step_budget` RK4 steps without reallocating `samples`.
    pub fn rebuild_trajectory(&mut self, step_budget: usize) {
        self.samples.clear();
        self.halted_near_singularity = false;
        let count = step_budget.clamp(2, MAX_TRAJECTORY_SAMPLES - 1);
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild => self.integrate_schwarzschild(count),
            SpacetimePreset::KerrEquatorial => self.integrate_kerr(count),
            SpacetimePreset::MorrisThorneWormhole => self.integrate_wormhole(count),
        }
        if !self.samples.is_empty() {
            self.state.scrub_index = self.state.scrub_index.min(self.samples.len() - 1);
        } else {
            self.state.scrub_index = 0;
        }
    }

    /// Appends up to `extra_steps` RK4 steps onto the active trajectory during live playback.
    pub fn advance_steps(&mut self, extra_steps: usize) {
        if self.samples.is_empty() {
            self.rebuild_trajectory(260);
        }
        let current_len = self.samples.len();
        if self.halted_near_singularity || current_len >= MAX_TRAJECTORY_SAMPLES {
            self.state.is_playing = false;
            return;
        }
        let target = (current_len + extra_steps).min(MAX_TRAJECTORY_SAMPLES - 1);
        self.rebuild_trajectory(target);
        self.state.scrub_index = self.samples.len().saturating_sub(1);
    }

    fn integrate_schwarzschild(&mut self, steps: usize) {
        let Ok(bh) = SchwarzschildBlackHole::from_schwarzschild_radius(self.state.scale_radius)
        else {
            return;
        };
        let rs = self.state.scale_radius;
        let r0 = self.state.initial_radius.max(1.15 * rs);
        let (pos0, vel0) = self.initial_4d_state(rs, 0.0, r0);
        let sys = GeodesicSystem::new(bh);
        let mut state = ArrayState::<8>::from_pos_vel(&pos0, &vel0);
        let mut solver = RungeKutta4::new(&state);
        let dt = self.state.step_size;

        let norm0 = sys.metric_norm(&state).unwrap_or(0.0);
        self.push_4d_sample(&bh, rs, 0.0, &state, norm0);
        for step in 0..steps {
            let lambda = (step as f64) * dt;
            match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
                GeodesicStepStatus::Advanced { metric_norm } => {
                    if state.0[1] > 28.0 {
                        self.push_4d_sample(&bh, rs, lambda + dt, &state, metric_norm);
                        break;
                    }
                    self.push_4d_sample(&bh, rs, lambda + dt, &state, metric_norm);
                }
                GeodesicStepStatus::HaltedSingularity => {
                    self.halted_near_singularity = true;
                    break;
                }
            }
        }
    }

    fn integrate_kerr(&mut self, steps: usize) {
        let rs = self.state.scale_radius;
        let a = 0.5 * rs * self.state.spin_chi;
        let Ok(kerr) = KerrEquatorialMetric::new(rs, a) else {
            return;
        };
        let Ok(bh_ref) = SchwarzschildBlackHole::from_schwarzschild_radius(rs) else {
            return;
        };
        let r0 = self
            .state
            .initial_radius
            .max(1.15 * kerr.outer_horizon_radius());
        let (pos0, vel0) = self.initial_4d_state(rs, a, r0);
        let sys = GeodesicSystem::new(kerr);
        let mut state = ArrayState::<8>::from_pos_vel(&pos0, &vel0);
        let mut solver = RungeKutta4::new(&state);
        let dt = self.state.step_size;

        let norm0 = sys.metric_norm(&state).unwrap_or(0.0);
        self.push_4d_sample(&bh_ref, rs, 0.0, &state, norm0);
        for step in 0..steps {
            let lambda = (step as f64) * dt;
            match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
                GeodesicStepStatus::Advanced { metric_norm } => {
                    self.push_4d_sample(&bh_ref, rs, lambda + dt, &state, metric_norm);
                    if state.0[1] > 28.0 {
                        break;
                    }
                }
                GeodesicStepStatus::HaltedSingularity => {
                    self.halted_near_singularity = true;
                    break;
                }
            }
        }
    }

    fn integrate_wormhole(&mut self, steps: usize) {
        let b0 = self.state.scale_radius;
        let Ok(wh) = MorrisThorneWormhole::new(b0) else {
            return;
        };
        let l0 = self.state.initial_radius;
        let kappa = match self.state.mode {
            GeodesicMode::NullPhoton => 0.0,
            GeodesicMode::TimelikeParticle => 1.0,
        };
        let r2 = b0 * b0 + l0 * l0;
        let ell = self.state.impact_or_momentum;
        let uphi = ell / r2;
        let e_sq = kappa + (ell * ell) / r2 + self.state.radial_bias * self.state.radial_bias;
        let ut = e_sq.sqrt();
        let ul =
            self.state.radial_bias.signum() * (e_sq - kappa - (ell * ell) / r2).max(0.0).sqrt();

        let sys = GeodesicSystem::new(wh);
        let mut state = ArrayState::<6>::from_pos_vel(
            &SVector::<f64, 3>::new(0.0, l0, 0.0),
            &SVector::<f64, 3>::new(ut, ul, uphi),
        );
        let mut solver = RungeKutta4::new(&state);
        let dt = self.state.step_size;

        let norm0 = sys.metric_norm(&state).unwrap_or(-kappa);
        self.push_wormhole_sample(&wh, 0.0, &state, norm0);
        for step in 0..steps {
            let lambda = (step as f64) * dt;
            match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
                GeodesicStepStatus::Advanced { metric_norm } => {
                    self.push_wormhole_sample(&wh, lambda + dt, &state, metric_norm);
                    if state.0[1].abs() > 20.0 {
                        break;
                    }
                }
                GeodesicStepStatus::HaltedSingularity => {
                    self.halted_near_singularity = true;
                    break;
                }
            }
        }
    }

    fn initial_4d_state(
        &self,
        rs: f64,
        spin_a: f64,
        r0: f64,
    ) -> (SVector<f64, 4>, SVector<f64, 4>) {
        let kappa = match self.state.mode {
            GeodesicMode::NullPhoton => 0.0,
            GeodesicMode::TimelikeParticle => 1.0,
        };
        let delta = (r0 * r0 - rs * r0 + spin_a * spin_a).max(1e-6);
        let g00 = -(1.0 - rs / r0);
        let g03 = -(spin_a * rs) / r0;
        let g11 = (r0 * r0) / delta;
        let g33 = r0 * r0 + spin_a * spin_a + (spin_a * spin_a * rs) / r0;

        let uphi = self.state.impact_or_momentum / (r0 * r0);
        let ur = self.state.radial_bias * 0.45;
        // Solve g00 * ut^2 + 2 * g03 * ut * uphi + (g11 * ur^2 + g33 * uphi^2 + kappa) = 0
        let spatial = g11 * ur * ur + g33 * uphi * uphi + kappa;
        let ut = if g00 < -1e-6 {
            let b_quad = 2.0 * g03 * uphi;
            let disc = (b_quad * b_quad - 4.0 * g00 * spatial).max(0.0);
            (-b_quad - disc.sqrt()) / (2.0 * g00)
        } else {
            1.5
        };
        (
            SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, 0.0),
            SVector::<f64, 4>::new(ut.max(0.1), ur, 0.0, uphi),
        )
    }

    fn push_4d_sample(
        &mut self,
        bh: &SchwarzschildBlackHole,
        rs: f64,
        proper_tau: f64,
        state: &ArrayState<8>,
        metric_norm: f64,
    ) {
        if self.samples.len() >= MAX_TRAJECTORY_SAMPLES {
            return;
        }
        let r = state.0[1].max(rs);
        let phi = state.0[3];
        let z = flamm_paraboloid_z(rs, r).unwrap_or(0.0);
        let (sin_p, cos_p) = phi.sin_cos();
        let grav = bh
            .gravitational_time_dilation(r.max(1.001 * rs))
            .unwrap_or(0.0);
        let ut = state.0[4].max(1e-6);
        let dtau_dt = match self.state.mode {
            GeodesicMode::NullPhoton => 0.0,
            GeodesicMode::TimelikeParticle => (1.0 / ut).clamp(0.0, 1.0),
        };
        let beta_local = if grav > 1e-6 && dtau_dt > 0.0 {
            let ratio = (dtau_dt / grav).clamp(0.0, 1.0);
            (1.0 - ratio * ratio).max(0.0).sqrt().min(0.999)
        } else {
            0.0
        };
        let rate = combined_clock_rate(bh, r.max(1.001 * rs), beta_local * C).unwrap_or(dtau_dt);
        let gamma = if beta_local > 0.0 {
            1.0 / (1.0 - beta_local * beta_local).sqrt()
        } else {
            1.0
        };

        self.samples.push(TrajectorySample {
            coord_ct: state.0[0],
            proper_tau,
            radial_coord: r,
            phi,
            radial_vel: state.0[5],
            embed_xyz: [r * cos_p, r * sin_p, z],
            metric_norm,
            clock_rate: rate,
            grav_dilation: grav,
            lorentz_gamma: gamma,
        });
    }

    fn push_wormhole_sample(
        &mut self,
        wh: &MorrisThorneWormhole,
        proper_tau: f64,
        state: &ArrayState<6>,
        metric_norm: f64,
    ) {
        if self.samples.len() >= MAX_TRAJECTORY_SAMPLES {
            return;
        }
        let l = state.0[1];
        let phi = state.0[2];
        let xyz = wh.embedding_xyz(l, phi).unwrap_or([0.0, 0.0, 0.0]);
        let ut = state.0[3].max(1.0);
        let clock_rate = match self.state.mode {
            GeodesicMode::NullPhoton => 0.0,
            GeodesicMode::TimelikeParticle => (1.0 / ut).clamp(0.0, 1.0),
        };
        self.samples.push(TrajectorySample {
            coord_ct: state.0[0],
            proper_tau,
            radial_coord: l,
            phi,
            radial_vel: state.0[4],
            embed_xyz: xyz,
            metric_norm,
            clock_rate,
            grav_dilation: 1.0,
            lorentz_gamma: ut,
        });
    }

    /// Generates projected 2D curves for CSV/PNG/SVG export and `Plot3DViewport` rendering.
    pub(super) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        if self.samples.is_empty() {
            let mut seeded = Self {
                state: self.state.clone(),
                samples: Vec::with_capacity(MAX_TRAJECTORY_SAMPLES),
                halted_near_singularity: false,
                last_tick: self.last_tick,
                revision: self.revision,
            };
            seeded.rebuild_trajectory(260);
            return seeded.collect_export_datasets();
        }
        let mut datasets = Vec::with_capacity(3);
        let limit = self
            .state
            .scrub_index
            .saturating_add(1)
            .min(self.samples.len());
        let traj: Vec<[f64; 2]> = self.samples[..limit]
            .iter()
            .map(|s| self.state.camera.project(&s.embed_xyz))
            .collect();
        datasets.push(("Geodesic Trajectory (Projected)".to_string(), traj));

        let ring = self.reference_ring_projected(self.state.scale_radius, 0.0, 48);
        datasets.push(("Horizon or Throat Ring".to_string(), ring));

        let veff = self.effective_potential_curve(64);
        datasets.push(("Effective Potential V_eff(r)".to_string(), veff));
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

    pub(super) fn effective_potential_curve(&self, points: usize) -> Vec<[f64; 2]> {
        let n = points.max(16);
        let mut curve = Vec::with_capacity(n);
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild | SpacetimePreset::KerrEquatorial => {
                let rs = self.state.scale_radius;
                let kind = match self.state.mode {
                    GeodesicMode::NullPhoton => GeodesicKind::Null,
                    GeodesicMode::TimelikeParticle => GeodesicKind::Timelike,
                };
                for i in 0..n {
                    let r = 1.05 * rs + 16.0 * (i as f64) / ((n - 1) as f64);
                    if let Ok(v) = effective_potential(rs, r, self.state.impact_or_momentum, kind) {
                        curve.push([r, v]);
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
}
