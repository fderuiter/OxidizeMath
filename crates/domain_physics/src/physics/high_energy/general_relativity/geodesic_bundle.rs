//! Pre-allocated zero-allocation Schwarzschild geodesic ray/orbit bundle and Newtonian comparison.

use std::f64::consts::{FRAC_PI_2, PI};

use nalgebra::Vector4;
use pure_math::pure_math::tensor::GeodesicStepStatus;

use crate::error::HighEnergyError;

use super::schwarzschild::{GeodesicKind, SchwarzschildBlackHole};

/// Maximum number of simultaneous trajectories in [`SchwarzschildGeodesicBundle`].
pub const MAX_GEODESIC_TRAJECTORIES: usize = 64;

/// Single trajectory state inside [`SchwarzschildGeodesicBundle`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicTrajectoryState {
    /// Four-position $x^\mu = (ct, r, \theta, \varphi)$ in units where $R_s$ sets length scale.
    pub pos: Vector4<f64>,
    /// Four-velocity $u^\mu = dx^\mu / d\lambda$.
    pub vel: Vector4<f64>,
    /// Causal classification (`Null` photon ray or `Timelike` test mass).
    pub kind: GeodesicKind,
    /// Impact parameter $b$ (null rays) or specific angular momentum $\ell$ (timelike).
    pub conserved_l_or_b: f64,
    /// Initial azimuth $\varphi_0$ at launch.
    pub initial_phi: f64,
    /// Initial Killing energy $e_0 = (1 - R_s / r_0) u^0$.
    pub initial_energy: f64,
    /// Initial Killing angular momentum $\ell_0 = r_0^2 \sin^2\theta_0 u^\varphi$.
    pub initial_angular_momentum: f64,
    /// Accumulated affine parameter $\lambda$ ($c\tau$ for timelike plunges).
    pub affine_elapsed: f64,
    /// Whether this trajectory halted at the pre-horizon singularity guard ($r \le 1.02 R_s$).
    pub halted_horizon: bool,
}

impl Default for GeodesicTrajectoryState {
    fn default() -> Self {
        Self {
            pos: Vector4::new(0.0, 10.0, FRAC_PI_2, 0.0),
            vel: Vector4::new(1.0, 0.0, 0.0, 0.0),
            kind: GeodesicKind::Null,
            conserved_l_or_b: 4.0,
            initial_phi: 0.0,
            initial_energy: 1.0,
            initial_angular_momentum: 0.0,
            affine_elapsed: 0.0,
            halted_horizon: false,
        }
    }
}

/// Planar Newtonian point-mass comparison state $(x, y, v_x, v_y)$ under $-GM / r^2 = -0.5 R_s / r^2$.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NewtonianOrbitState {
    /// Cartesian $x$ and $y$ coordinates and velocities $(v_x, v_y)$, plus minimum radius $r_{\min}$.
    pub x: f64,
    /// Cartesian $y$ coordinate.
    pub y: f64,
    /// Velocity $dx/dt$ (in units of $c$).
    pub vx: f64,
    /// Velocity $dy/dt$ (in units of $c$).
    pub vy: f64,
    /// Minimum radius $r_{\min}$ visited along the Newtonian orbit.
    pub min_radius: f64,
}

/// Pre-allocated geodesic and Newtonian comparison bundle (`<= 64` rays) for zero-allocation RK4 stepping.
#[derive(Debug, Clone)]
pub struct SchwarzschildGeodesicBundle {
    /// Underlying Schwarzschild black hole spacetime and cached horizon radius $R_s > 0$.
    pub black_hole: SchwarzschildBlackHole,
    /// Cached Schwarzschild radius $R_s > 0$.
    pub r_s: f64,
    active_count: usize,
    trail_capacity: usize,
    trail_head: usize,
    trail_len: usize,
    trajectories: Vec<GeodesicTrajectoryState>,
    trail_ring: Vec<[f64; 2]>,
    newtonian: NewtonianOrbitState,
    newtonian_trail: Vec<[f64; 2]>,
}

#[verified_engine::verified]
fn newtonian_accel(x: f64, y: f64, gm: f64, min_r: f64) -> [f64; 2] {
    let r2 = (x * x + y * y).max(min_r * min_r);
    let inv_r3 = gm / (r2 * r2.sqrt());
    [-inv_r3 * x, -inv_r3 * y]
}

#[verified_engine::verified]
fn step_newtonian_rk4(state: &mut NewtonianOrbitState, r_s: f64, dt: f64) {
    let gm = 0.5 * r_s;
    let min_r = 0.25 * r_s;
    let a1 = newtonian_accel(state.x, state.y, gm, min_r);

    let x2 = state.x + 0.5 * dt * state.vx;
    let y2 = state.y + 0.5 * dt * state.vy;
    let vx2 = state.vx + 0.5 * dt * a1[0];
    let vy2 = state.vy + 0.5 * dt * a1[1];
    let a2 = newtonian_accel(x2, y2, gm, min_r);

    let x3 = state.x + 0.5 * dt * vx2;
    let y3 = state.y + 0.5 * dt * vy2;
    let vx3 = state.vx + 0.5 * dt * a2[0];
    let vy3 = state.vy + 0.5 * dt * a2[1];
    let a3 = newtonian_accel(x3, y3, gm, min_r);

    let x4 = state.x + dt * vx3;
    let y4 = state.y + dt * vy3;
    let vx4 = state.vx + dt * a3[0];
    let vy4 = state.vy + dt * a3[1];
    let a4 = newtonian_accel(x4, y4, gm, min_r);

    state.x += (dt / 6.0) * (state.vx + 2.0 * vx2 + 2.0 * vx3 + vx4);
    state.y += (dt / 6.0) * (state.vy + 2.0 * vy2 + 2.0 * vy3 + vy4);
    state.vx += (dt / 6.0) * (a1[0] + 2.0 * a2[0] + 2.0 * a3[0] + a4[0]);
    state.vy += (dt / 6.0) * (a1[1] + 2.0 * a2[1] + 2.0 * a3[1] + a4[1]);
    let r_now = state.x.hypot(state.y);
    if r_now.is_finite() && r_now < state.min_radius {
        state.min_radius = r_now;
    }
}

impl SchwarzschildGeodesicBundle {
    /// Allocates fixed-capacity trajectory (`<= 64`) and trail buffers (`# Errors`: `r_s <= 0`).
    pub fn new(r_s: f64, max_rays: usize, trail_capacity: usize) -> Result<Self, HighEnergyError> {
        let black_hole = SchwarzschildBlackHole::from_schwarzschild_radius(r_s)?;
        let cap_rays = max_rays.clamp(1, MAX_GEODESIC_TRAJECTORIES);
        let cap_trail = trail_capacity.max(4);
        Ok(Self {
            black_hole,
            r_s,
            active_count: 1,
            trail_capacity: cap_trail,
            trail_head: 0,
            trail_len: 0,
            trajectories: vec![GeodesicTrajectoryState::default(); cap_rays],
            trail_ring: vec![[0.0, 0.0]; cap_rays.saturating_mul(cap_trail)],
            newtonian: NewtonianOrbitState::default(),
            newtonian_trail: vec![[0.0, 0.0]; cap_trail],
        })
    }

    /// Read-only slice of active trajectories.
    #[must_use]
    #[verified_engine::verified]
    pub fn trajectories(&self) -> &[GeodesicTrajectoryState] {
        &self.trajectories[..self.active_count]
    }
    /// Current planar Newtonian comparison orbit state.
    #[must_use]
    #[verified_engine::verified]
    pub fn newtonian_state(&self) -> &NewtonianOrbitState {
        &self.newtonian
    }

    /// Reseeds a parallel photon-beam lensing fan of `ray_count` null rays centered around
    /// `probe_b_over_rs` (index `0` is always the exact probe ray at `probe_b_over_rs * R_s`).
    #[verified_engine::verified]
    pub fn seed_photon_lensing_fan(
        &mut self,
        ray_count: usize,
        probe_b_over_rs: f64,
        launch_x_over_rs: f64,
        fan_half_span_over_rs: f64,
    ) {
        let n = ray_count.clamp(1, self.trajectories.len());
        self.active_count = n;
        let x0 = -(launch_x_over_rs.abs().clamp(4.0, 60.0)) * self.r_s;
        let b_probe = probe_b_over_rs.clamp(0.2, 15.0) * self.r_s;
        let span = fan_half_span_over_rs.clamp(0.0, 8.0) * self.r_s;

        let mut i = 0;
        while i < n {
            let b = if i == 0 || n == 1 {
                b_probe
            } else {
                let frac = ((i - 1) as f64) / ((n - 2).max(1) as f64) - 0.5;
                (b_probe + 2.0 * span * frac).max(0.15 * self.r_s)
            };
            let r0 = x0.hypot(b).max(1.5 * self.r_s);
            let phi0 = b.atan2(x0);
            let f0 = 1.0 - self.r_s / r0;
            let ut0 = 1.0 / f0;
            let uphi0 = -b / (r0 * r0);
            let ur0 = -(1.0 - f0 * (b * b) / (r0 * r0)).max(0.0).sqrt();
            let pos = Vector4::new(0.0, r0, FRAC_PI_2, phi0);
            let vel = Vector4::new(ut0, ur0, 0.0, uphi0);
            let (e0, l0) = self.black_hole.killing_invariants(&pos, &vel);
            self.trajectories[i] = GeodesicTrajectoryState {
                pos,
                vel,
                kind: GeodesicKind::Null,
                conserved_l_or_b: b,
                initial_phi: phi0,
                initial_energy: e0,
                initial_angular_momentum: l0,
                affine_elapsed: 0.0,
                halted_horizon: false,
            };
            i += 1;
        }
        self.newtonian = NewtonianOrbitState {
            x: x0,
            y: b_probe,
            vx: 1.0,
            vy: 0.0,
            min_radius: x0.hypot(b_probe),
        };
        self.reset_trail_ring();
    }

    /// Reseeds a timelike orbit/plunge at initial radius `r0_over_rs * R_s` with angular momentum
    /// `l_over_l_isco * L_ISCO` alongside the matching Newtonian point-mass orbit.
    #[verified_engine::verified]
    pub fn seed_timelike_isco_comparison(
        &mut self,
        r0_over_rs: f64,
        l_over_l_isco: f64,
        inward_radial_nudge: f64,
    ) {
        self.active_count = 1;
        let r0 = r0_over_rs.clamp(1.6, 25.0) * self.r_s;
        let l_ratio = l_over_l_isco.clamp(0.1, 2.5);
        let l_isco = 3.0_f64.sqrt() * self.r_s;
        let l = l_ratio * l_isco;
        let ur0 = -inward_radial_nudge.abs().clamp(0.0, 0.5);
        let f0 = 1.0 - self.r_s / r0;
        let uphi0 = l / (r0 * r0);
        let ut0 = ((1.0 + (ur0 * ur0) / f0 + r0 * r0 * uphi0 * uphi0) / f0).sqrt();
        let pos = Vector4::new(0.0, r0, FRAC_PI_2, 0.0);
        let vel = Vector4::new(ut0, ur0, 0.0, uphi0);
        let (e0, l0) = self.black_hole.killing_invariants(&pos, &vel);
        self.trajectories[0] = GeodesicTrajectoryState {
            pos,
            vel,
            kind: GeodesicKind::Timelike,
            conserved_l_or_b: l,
            initial_phi: 0.0,
            initial_energy: e0,
            initial_angular_momentum: l0,
            affine_elapsed: 0.0,
            halted_horizon: false,
        };

        let v_circ_newt = (0.5 * self.r_s / r0).sqrt();
        self.newtonian = NewtonianOrbitState {
            x: r0,
            y: 0.0,
            vx: ur0,
            vy: l_ratio * v_circ_newt,
            min_radius: r0,
        };
        self.reset_trail_ring();
    }

    /// Seeds a mixed ensemble of `total_count` (`<= 64`) null and timelike geodesics for
    /// multi-trajectory stress and Killing-invariant verification.
    #[verified_engine::verified]
    pub fn seed_mixed_ensemble(&mut self, total_count: usize) {
        let n = total_count.clamp(2, self.trajectories.len());
        self.active_count = n;
        let half = n / 2;
        let mut i = 0;
        while i < n {
            let (pos, vel, kind, conserved, phi0) = if i < half {
                let b = (2.75 + 0.12 * (i as f64)) * self.r_s;
                let x0 = -16.0 * self.r_s;
                let r0 = x0.hypot(b);
                let p0 = b.atan2(x0);
                let f0 = 1.0 - self.r_s / r0;
                let v = Vector4::new(
                    1.0 / f0,
                    -(1.0 - f0 * b * b / (r0 * r0)).sqrt(),
                    0.0,
                    -b / (r0 * r0),
                );
                (
                    Vector4::new(0.0, r0, FRAC_PI_2, p0),
                    v,
                    GeodesicKind::Null,
                    b,
                    p0,
                )
            } else {
                let r0 = (3.2 + 0.15 * ((i - half) as f64)) * self.r_s;
                let l = (0.5 * self.r_s * r0 * r0 / (r0 - 1.5 * self.r_s)).sqrt();
                let f0 = 1.0 - self.r_s / r0;
                let uphi0 = l / (r0 * r0);
                let ut0 = ((1.0 + r0 * r0 * uphi0 * uphi0) / f0).sqrt();
                (
                    Vector4::new(0.0, r0, FRAC_PI_2, 0.0),
                    Vector4::new(ut0, 0.0, 0.0, uphi0),
                    GeodesicKind::Timelike,
                    l,
                    0.0,
                )
            };
            let (e0, l0) = self.black_hole.killing_invariants(&pos, &vel);
            self.trajectories[i] = GeodesicTrajectoryState {
                pos,
                vel,
                kind,
                conserved_l_or_b: conserved,
                initial_phi: phi0,
                initial_energy: e0,
                initial_angular_momentum: l0,
                affine_elapsed: 0.0,
                halted_horizon: false,
            };
            i += 1;
        }
        self.newtonian = NewtonianOrbitState {
            x: 3.0 * self.r_s,
            y: 0.0,
            vx: 0.0,
            vy: (1.0 / 6.0_f64).sqrt(),
            min_radius: 3.0 * self.r_s,
        };
        self.reset_trail_ring();
    }

    #[verified_engine::verified]
    fn reset_trail_ring(&mut self) {
        self.trail_head = 0;
        self.trail_len = 0;
        self.record_trail_snapshot();
    }

    #[verified_engine::verified]
    fn record_trail_snapshot(&mut self) {
        let cap = self.trail_capacity;
        if cap == 0 || self.trail_ring.is_empty() {
            return;
        }
        let slot = self.trail_head;
        let mut i = 0;
        while i < self.active_count {
            let r = self.trajectories[i].pos[1];
            let (sin_p, cos_p) = self.trajectories[i].pos[3].sin_cos();
            let idx = math_commons::math_kernel::types::flatten_2d_index(slot, i, cap);
            if idx < self.trail_ring.len() {
                self.trail_ring[idx] = [r * cos_p, r * sin_p];
            }
            i += 1;
        }
        if slot < self.newtonian_trail.len() {
            self.newtonian_trail[slot] = [self.newtonian.x, self.newtonian.y];
        }
        self.trail_head = (slot + 1) % cap;
        if self.trail_len < cap {
            self.trail_len += 1;
        }
    }

    /// Advances all active trajectories and the Newtonian reference orbit by one RK4 step `d_lambda`
    /// with zero heap allocations.
    #[verified_engine::verified]
    pub fn step_once(&mut self, d_lambda: f64) {
        if !d_lambda.is_finite() || d_lambda <= 0.0 {
            return;
        }
        let dt = d_lambda.min(1.0);
        let mut i = 0;
        while i < self.active_count {
            if !self.trajectories[i].halted_horizon {
                let mut p = self.trajectories[i].pos;
                let mut v = self.trajectories[i].vel;
                match self.black_hole.step_geodesic_rk4(&mut p, &mut v, dt) {
                    GeodesicStepStatus::Advanced { .. } => {
                        self.trajectories[i].pos = p;
                        self.trajectories[i].vel = v;
                        self.trajectories[i].affine_elapsed += dt;
                    }
                    GeodesicStepStatus::HaltedSingularity => {
                        self.trajectories[i].halted_horizon = true;
                    }
                }
            }
            i += 1;
        }
        step_newtonian_rk4(&mut self.newtonian, self.r_s, dt);
        self.record_trail_snapshot();
    }

    /// Advances all active trajectories by `substeps` RK4 steps of `d_lambda` each.
    #[verified_engine::verified]
    pub fn step_n(&mut self, d_lambda: f64, substeps: usize) {
        let bounded = substeps.min(4_096);
        let mut s = 0;
        while s < bounded {
            self.step_once(d_lambda);
            s += 1;
        }
    }

    /// Executes [`Self::step_n`] inside [`verified_engine::engine::VerifiedEngine::run_verified`],
    /// proving zero heap allocations and returning `(function_calls, arithmetic_ops)`.
    pub fn step_n_verified(&mut self, d_lambda: f64, substeps: usize) -> (u64, u64) {
        let ((), metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            self.step_n(d_lambda, substeps);
        });
        (metrics.function_calls, metrics.arithmetic_ops)
    }

    /// Returns the worst-case relative Killing energy or angular-momentum drift across all active
    /// non-halted trajectories.
    #[must_use]
    #[verified_engine::verified]
    pub fn max_killing_drift(&self) -> f64 {
        let mut max_drift = 0.0_f64;
        let mut i = 0;
        while i < self.active_count {
            let tr = &self.trajectories[i];
            let (e_now, l_now) = self.black_hole.killing_invariants(&tr.pos, &tr.vel);
            let de = (e_now - tr.initial_energy).abs() / tr.initial_energy.abs().max(1e-12);
            let dl = (l_now - tr.initial_angular_momentum).abs()
                / tr.initial_angular_momentum.abs().max(1.0);
            max_drift = max_drift.max(de).max(dl);
            i += 1;
        }
        max_drift
    }

    /// Copies the chronological 2D Cartesian trail $(x, y)$ of trajectory `index` into `out`.
    #[verified_engine::verified]
    pub fn copy_trajectory_trail(&self, index: usize, out: &mut [[f64; 2]]) -> usize {
        if index >= self.active_count || self.trail_len == 0 || out.is_empty() {
            return 0;
        }
        let cap = self.trail_capacity;
        let count = self.trail_len.min(out.len());
        let start = if self.trail_len == cap {
            self.trail_head
        } else {
            0
        };
        let mut k = 0;
        while k < count {
            let flat =
                math_commons::math_kernel::types::flatten_2d_index((start + k) % cap, index, cap);
            out[k] = self.trail_ring[flat];
            k += 1;
        }
        count
    }

    /// Copies the chronological 2D Cartesian trail $(x, y)$ of the Newtonian comparison orbit into `out`.
    #[verified_engine::verified]
    pub fn copy_newtonian_trail(&self, out: &mut [[f64; 2]]) -> usize {
        if self.trail_len == 0 || out.is_empty() {
            return 0;
        }
        let cap = self.trail_capacity;
        let count = self.trail_len.min(out.len());
        let start = if self.trail_len == cap {
            self.trail_head
        } else {
            0
        };
        let mut k = 0;
        while k < count {
            out[k] = self.newtonian_trail[(start + k) % cap];
            k += 1;
        }
        count
    }

    /// Computes total swept deflection angle $|\Delta\varphi| - \pi$ (radians) of the primary
    /// probe ray (`index = 0`), returning `None` if captured by the event horizon.
    #[must_use]
    #[verified_engine::verified]
    pub fn probe_photon_deflection_rad(&self) -> Option<f64> {
        let tr = self.trajectories.first()?;
        if tr.halted_horizon || tr.kind != GeodesicKind::Null {
            return None;
        }
        let swept = (tr.initial_phi - tr.pos[3]).abs();
        Some((swept - PI).max(0.0))
    }
}

/// Analytical Newtonian periapsis radius $r_{\min} = \frac{r_0 \eta^2}{2 - \eta^2}$ ($\eta = L / L_{\text{circ}}$).
#[must_use]
#[verified_engine::verified]
pub fn newtonian_periapsis_radius(r0: f64, l_over_l_circ: f64) -> f64 {
    let eta = l_over_l_circ.clamp(0.0, 1.0);
    let eta2 = eta * eta;
    (r0 * eta2) / (2.0 - eta2)
}

// [cite:general_relativity_geodesics]
