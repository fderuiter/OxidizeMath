#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation RK4 trajectory integration, normalized photon launch ($e = 1$),
//! and twin worldline sampling for [`GeodesicPlaygroundTool`].

use std::f64::consts::FRAC_PI_2;

use math_explorer::nalgebra::SVector;
use math_explorer::physics::high_energy::{
    KerrEquatorialMetric, MorrisThorneWormhole, SchwarzschildBlackHole,
};
use math_explorer::pure_math::analysis::ode::{ArrayState, OdeSystem, RungeKutta4};
use math_explorer::pure_math::tensor::{
    GeodesicState, GeodesicStepStatus, GeodesicSystem, StaticMetric,
};

use super::{GeodesicMode, GeodesicPlaygroundTool, SpacetimePreset};

/// Fixed upper bound on RK4 steps per trajectory (`samples` reserves `MAX_TRAJECTORY_SAMPLES + 8`).
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
    /// Instantaneous gravitational/lapse dilation factor.
    pub grav_dilation: f64,
    /// Instantaneous local Lorentz factor $\gamma(v)$.
    pub lorentz_gamma: f64,
}

#[derive(Clone, Copy)]
pub(super) struct Sample4dMeta<'a> {
    pub(super) mode: GeodesicMode,
    pub(super) bh: Option<&'a SchwarzschildBlackHole>,
    pub(super) throat_r: f64,
    pub(super) spin_a: f64,
}

/// Refines a pre-horizon RK4 halt using stack-allocated micro-steps so capture lands near $1.02 r_+$.
pub(crate) fn refine_halt_4d<M>(
    sys: &GeodesicSystem<M, 4>,
    solver: &mut RungeKutta4<ArrayState<8>>,
    lambda: f64,
    state: &mut ArrayState<8>,
    dt: f64,
) -> (bool, f64)
where
    M: StaticMetric<4>,
    GeodesicSystem<M, 4>: OdeSystem<ArrayState<8>>,
{
    let sub_dt = dt * 0.1;
    let mut cur_lambda = lambda;
    let mut advanced_any = false;
    for _ in 0..10 {
        let prev = *state;
        if let GeodesicStepStatus::Advanced { .. } = sys.step_rk4(solver, cur_lambda, state, sub_dt)
        {
            let pos = SVector::<f64, 4>::new(state.0[0], state.0[1], FRAC_PI_2, state.0[3]);
            if sys.metric().christoffel_at(&pos).is_err() {
                *state = prev;
                break;
            }
            cur_lambda += sub_dt;
            advanced_any = true;
        } else {
            break;
        }
    }
    (advanced_any, cur_lambda)
}

impl GeodesicPlaygroundTool {
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
        if !s.camera.is_valid() {
            s.camera = crate::camera::Camera3D::new(0.48, -0.62, 1.0);
        }
    }

    /// Recomputes primary and twin geodesic buffers up to `step_budget` RK4 steps with zero heap allocations.
    pub fn rebuild_trajectory(&mut self, step_budget: usize) {
        self.samples.clear();
        self.twin_samples.clear();
        self.halted_near_singularity = false;
        self.twin_halted_near_singularity = false;
        let count = step_budget.clamp(2, MAX_TRAJECTORY_SAMPLES);
        let b0 = self.state.impact_or_momentum;
        let mut primary = core::mem::take(&mut self.samples);
        let (halted, e_val) = self.integrate_single_worldline(count, b0, &mut primary);
        self.samples = primary;
        self.halted_near_singularity = halted;
        self.launch_energy = e_val;

        if self.state.twin_delta_b.abs() > 1e-9 {
            let b_twin = (b0 + self.state.twin_delta_b).clamp(-16.0, 16.0);
            let mut twin = core::mem::take(&mut self.twin_samples);
            let (t_halted, t_e) = self.integrate_single_worldline(count, b_twin, &mut twin);
            self.twin_samples = twin;
            self.twin_halted_near_singularity = t_halted;
            self.twin_launch_energy = t_e;
        } else {
            self.twin_launch_energy = e_val;
        }

        self.state.scrub_index = if self.samples.is_empty() {
            0
        } else {
            self.state.scrub_index.min(self.samples.len() - 1)
        };
    }

    fn integrate_single_worldline(
        &self,
        steps: usize,
        impact_val: f64,
        out: &mut Vec<TrajectorySample>,
    ) -> (bool, f64) {
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild => {
                self.integrate_schwarzschild_into(steps, impact_val, out)
            }
            SpacetimePreset::KerrEquatorial => self.integrate_kerr_into(steps, impact_val, out),
            SpacetimePreset::MorrisThorneWormhole => {
                self.integrate_wormhole_into(steps, impact_val, out)
            }
            SpacetimePreset::CustomRadialLapse => {
                self.integrate_custom_lapse_into(steps, impact_val, out)
            }
            SpacetimePreset::CustomWormholeProfile => {
                self.integrate_custom_wormhole_into(steps, impact_val, out)
            }
        }
    }

    /// Appends up to `extra_steps` RK4 steps onto the active trajectory during live playback.
    pub fn advance_steps(&mut self, extra_steps: usize) {
        if self.samples.is_empty() {
            self.rebuild_trajectory(260);
        }
        let current_len = self.samples.len();
        if self.halted_near_singularity || current_len > MAX_TRAJECTORY_SAMPLES {
            self.state.is_playing = false;
            return;
        }
        self.rebuild_trajectory((current_len + extra_steps).min(MAX_TRAJECTORY_SAMPLES));
        self.state.scrub_index = self.samples.len().saturating_sub(1);
    }

    fn integrate_schwarzschild_into(
        &self,
        steps: usize,
        impact_val: f64,
        out: &mut Vec<TrajectorySample>,
    ) -> (bool, f64) {
        let rs = self.state.scale_radius;
        let Ok(bh) = SchwarzschildBlackHole::from_schwarzschild_radius(rs) else {
            return (false, 1.0);
        };
        let r0 = self.state.initial_radius.max(1.05 * rs);
        let (pos0, vel0, e_val) = self.initial_4d_state(rs, 0.0, r0, impact_val);
        let meta = Sample4dMeta {
            mode: self.state.mode,
            bh: Some(&bh),
            throat_r: rs,
            spin_a: 0.0,
        };
        let halted = Self::run_4d_rk4(bh, pos0, vel0, self.state.step_size, steps, meta, out);
        (halted, e_val)
    }

    fn integrate_kerr_into(
        &self,
        steps: usize,
        impact_val: f64,
        out: &mut Vec<TrajectorySample>,
    ) -> (bool, f64) {
        let rs = self.state.scale_radius;
        let a = 0.5 * rs * self.state.spin_chi.clamp(-0.95, 0.95);
        let Ok(kerr) = KerrEquatorialMetric::new(rs, a) else {
            return (false, 1.0);
        };
        let r_plus = kerr.outer_horizon_radius();
        let r0 = self.state.initial_radius.max(1.05 * r_plus);
        let (pos0, vel0, e_val) = self.initial_4d_state(rs, a, r0, impact_val);
        let meta = Sample4dMeta {
            mode: self.state.mode,
            bh: None,
            throat_r: r_plus,
            spin_a: a,
        };
        let halted = Self::run_4d_rk4(kerr, pos0, vel0, self.state.step_size, steps, meta, out);
        (halted, e_val)
    }

    fn run_4d_rk4<M>(
        metric: M,
        pos0: SVector<f64, 4>,
        vel0: SVector<f64, 4>,
        dt: f64,
        steps: usize,
        meta: Sample4dMeta<'_>,
        out: &mut Vec<TrajectorySample>,
    ) -> bool
    where
        M: StaticMetric<4>,
        GeodesicSystem<M, 4>: OdeSystem<ArrayState<8>>,
    {
        let sys = GeodesicSystem::new(metric);
        let mut state = ArrayState::<8>::from_pos_vel(&pos0, &vel0);
        let mut solver = RungeKutta4::new(&state);
        let norm0 = sys.metric_norm(&state).unwrap_or(0.0);
        Self::push_4d_sample(out, meta, 0.0, &state, norm0);

        for step in 0..steps {
            let lambda = (step as f64) * dt;
            match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
                GeodesicStepStatus::Advanced { metric_norm } => {
                    Self::push_4d_sample(out, meta, lambda + dt, &state, metric_norm);
                    if state.0[1] > 28.0 {
                        return false;
                    }
                }
                GeodesicStepStatus::HaltedSingularity => {
                    let (did_adv, sub_lam) =
                        refine_halt_4d(&sys, &mut solver, lambda, &mut state, dt);
                    if did_adv {
                        if let Ok(n_end) = sys.metric_norm(&state) {
                            Self::push_4d_sample(out, meta, sub_lam, &state, n_end);
                        }
                    }
                    return true;
                }
            }
        }
        false
    }

    pub(super) fn wormhole_initial_vel(
        mode: GeodesicMode,
        r2: f64,
        impact_val: f64,
        radial_bias: f64,
    ) -> (f64, f64, f64, f64) {
        let safe_r2 = r2.max(0.01);
        let sign = if radial_bias > 0.0 { 1.0 } else { -1.0 };
        match mode {
            GeodesicMode::NullPhoton => {
                let max_b = safe_r2.sqrt();
                let ell = impact_val.clamp(-max_b, max_b);
                let uphi = ell / safe_r2;
                let ul = sign * (1.0 - (ell * ell) / safe_r2).max(0.0).sqrt();
                (1.0, ul, uphi, 1.0)
            }
            GeodesicMode::TimelikeParticle => {
                let uphi = impact_val / safe_r2;
                let e =
                    (1.0 + (impact_val * impact_val) / safe_r2 + radial_bias * radial_bias).sqrt();
                (e, sign * radial_bias.abs(), uphi, e)
            }
        }
    }

    fn integrate_wormhole_into(
        &self,
        steps: usize,
        impact_val: f64,
        out: &mut Vec<TrajectorySample>,
    ) -> (bool, f64) {
        let b0 = self.state.scale_radius;
        let Ok(wh) = MorrisThorneWormhole::new(b0) else {
            return (false, 1.0);
        };
        let l0 = self.state.initial_radius;
        let (ut, ul, uphi, e_val) = Self::wormhole_initial_vel(
            self.state.mode,
            b0 * b0 + l0 * l0,
            impact_val,
            self.state.radial_bias,
        );
        let sys = GeodesicSystem::new(wh);
        let mut state = ArrayState::<6>::from_pos_vel(
            &SVector::<f64, 3>::new(0.0, l0, 0.0),
            &SVector::<f64, 3>::new(ut, ul, uphi),
        );
        let mut solver = RungeKutta4::new(&state);
        let dt = self.state.step_size;
        let norm0 = sys.metric_norm(&state).unwrap_or(0.0);
        Self::push_wormhole_sample(out, self.state.mode, &wh, 0.0, &state, norm0);

        for step in 0..steps {
            let lambda = (step as f64) * dt;
            match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
                GeodesicStepStatus::Advanced { metric_norm } => {
                    Self::push_wormhole_sample(
                        out,
                        self.state.mode,
                        &wh,
                        lambda + dt,
                        &state,
                        metric_norm,
                    );
                    if state.0[1].abs() > 20.0 {
                        return (false, e_val);
                    }
                }
                GeodesicStepStatus::HaltedSingularity => return (true, e_val),
            }
        }
        (false, e_val)
    }

    fn initial_4d_state(
        &self,
        rs: f64,
        spin_a: f64,
        r0: f64,
        impact_val: f64,
    ) -> (SVector<f64, 4>, SVector<f64, 4>, f64) {
        let delta = (r0 * r0 - rs * r0 + spin_a * spin_a).max(1e-6);
        let sqrt_delta = delta.sqrt();
        let g03 = -(spin_a * rs) / r0;
        let g11 = (r0 * r0) / delta;
        let g33 = r0 * r0 + spin_a * spin_a + (spin_a * spin_a * rs) / r0;

        match self.state.mode {
            GeodesicMode::NullPhoton => {
                let mut ell = impact_val;
                if sqrt_delta - g03 > 1e-9 {
                    ell = ell.min(g33 / (sqrt_delta - g03));
                }
                if sqrt_delta + g03 > 1e-9 {
                    ell = ell.max(-g33 / (sqrt_delta + g03));
                }
                let ut = (g33 - (spin_a * rs / r0) * ell) / delta;
                let uphi = ((spin_a * rs / r0) + (1.0 - rs / r0) * ell) / delta;
                let ur_sq = ((delta / (r0 * r0)) * (ut - ell * uphi)).max(0.0);
                let sign = if self.state.radial_bias > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                (
                    SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, 0.0),
                    SVector::<f64, 4>::new(ut.max(1e-4), sign * ur_sq.sqrt(), 0.0, uphi),
                    1.0,
                )
            }
            GeodesicMode::TimelikeParticle => {
                let ell = impact_val;
                let ur = self.state.radial_bias * 0.45;
                let omega0 = (spin_a * rs) / (r0 * g33);
                let ut = ((g33 / delta) * (1.0 + (ell * ell) / g33 + g11 * ur * ur))
                    .max(1e-6)
                    .sqrt();
                let e = omega0 * ell + (delta / g33) * ut;
                let uphi = ((spin_a * rs / r0) * e + (1.0 - rs / r0) * ell) / delta;
                (
                    SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, 0.0),
                    SVector::<f64, 4>::new(ut, ur, 0.0, uphi),
                    e,
                )
            }
        }
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
