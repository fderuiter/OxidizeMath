#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Stack-allocated RK4 ray integrators and sample extractors for Schwarzschild, Kerr, and Morris-Thorne metrics.

use std::f64::consts::FRAC_PI_2;

use math_explorer::nalgebra::SVector;
use math_explorer::physics::high_energy::{
    combined_clock_rate, flamm_paraboloid_z, KerrEquatorialMetric, MorrisThorneWormhole,
    SchwarzschildBlackHole, C,
};
use math_explorer::pure_math::analysis::ode::{ArrayState, RungeKutta4};
use math_explorer::pure_math::tensor::{GeodesicState, GeodesicStepStatus, GeodesicSystem};

use super::sim::{BundleRay, TrajectorySample, MAX_TRAJECTORY_SAMPLES};
use super::GeodesicMode;

pub(super) struct RayLaunchSpec {
    pub(super) mode: GeodesicMode,
    pub(super) rs: f64,
    pub(super) spin_chi: f64,
    pub(super) r0_cfg: f64,
    pub(super) phi0: f64,
    pub(super) impact_b: f64,
    pub(super) radial_bias: f64,
    pub(super) dt: f64,
    pub(super) steps: usize,
}

pub(super) fn integrate_schwarzschild_ray(ray: &mut BundleRay, spec: &RayLaunchSpec) {
    let Ok(bh) = SchwarzschildBlackHole::from_schwarzschild_radius(spec.rs) else {
        return;
    };
    let r0 = spec.r0_cfg.max(1.15 * spec.rs);
    let (pos0, vel0) = initial_4d_state(spec, 0.0, r0);
    let sys = GeodesicSystem::new(bh);
    let mut state = ArrayState::<8>::from_pos_vel(&pos0, &vel0);
    let mut solver = RungeKutta4::new(&state);

    let norm0 = sys.metric_norm(&state).unwrap_or(0.0);
    push_schwarzschild_sample(ray, spec.mode, &bh, spec.rs, 0.0, &state, norm0);
    for step in 0..spec.steps {
        let lambda = (step as f64) * spec.dt;
        match sys.step_rk4(&mut solver, lambda, &mut state, spec.dt) {
            GeodesicStepStatus::Advanced { metric_norm } => {
                push_schwarzschild_sample(
                    ray,
                    spec.mode,
                    &bh,
                    spec.rs,
                    lambda + spec.dt,
                    &state,
                    metric_norm,
                );
                if state.0[1] > 28.0 {
                    break;
                }
            }
            GeodesicStepStatus::HaltedSingularity => {
                ray.halted = true;
                break;
            }
        }
    }
}

pub(super) fn integrate_kerr_ray(ray: &mut BundleRay, spec: &RayLaunchSpec) {
    let a = 0.5 * spec.rs * spec.spin_chi;
    let Ok(kerr) = KerrEquatorialMetric::new(spec.rs, a) else {
        return;
    };
    let r0 = spec.r0_cfg.max(1.12 * kerr.outer_horizon_radius());
    let (pos0, vel0) = initial_4d_state(spec, a, r0);
    let sys = GeodesicSystem::new(kerr);
    let mut state = ArrayState::<8>::from_pos_vel(&pos0, &vel0);
    let mut solver = RungeKutta4::new(&state);

    let norm0 = sys.metric_norm(&state).unwrap_or(0.0);
    push_kerr_sample(ray, spec.mode, &kerr, 0.0, &state, norm0);
    for step in 0..spec.steps {
        let lambda = (step as f64) * spec.dt;
        match sys.step_rk4(&mut solver, lambda, &mut state, spec.dt) {
            GeodesicStepStatus::Advanced { metric_norm } => {
                push_kerr_sample(ray, spec.mode, &kerr, lambda + spec.dt, &state, metric_norm);
                if state.0[1] > 28.0 {
                    break;
                }
            }
            GeodesicStepStatus::HaltedSingularity => {
                ray.halted = true;
                break;
            }
        }
    }
}

pub(super) fn integrate_wormhole_ray(ray: &mut BundleRay, spec: &RayLaunchSpec) {
    let b0 = spec.rs;
    let Ok(wh) = MorrisThorneWormhole::new(b0) else {
        return;
    };
    let kappa = match spec.mode {
        GeodesicMode::NullPhoton => 0.0,
        GeodesicMode::TimelikeParticle => 1.0,
    };
    let l0 = spec.r0_cfg;
    let r2 = b0 * b0 + l0 * l0;
    let ell = spec.impact_b;
    let uphi = ell / r2;
    let e_sq = kappa + (ell * ell) / r2 + spec.radial_bias * spec.radial_bias;
    let ut = e_sq.sqrt().max(0.1);
    let ul = spec.radial_bias.signum() * (e_sq - kappa - (ell * ell) / r2).max(0.0).sqrt();

    let sys = GeodesicSystem::new(wh);
    let mut state = ArrayState::<6>::from_pos_vel(
        &SVector::<f64, 3>::new(0.0, l0, spec.phi0),
        &SVector::<f64, 3>::new(ut, ul, uphi),
    );
    let mut solver = RungeKutta4::new(&state);

    let norm0 = sys.metric_norm(&state).unwrap_or(-kappa);
    push_wormhole_sample(ray, spec.mode, &wh, 0.0, &state, norm0);
    for step in 0..spec.steps {
        let lambda = (step as f64) * spec.dt;
        match sys.step_rk4(&mut solver, lambda, &mut state, spec.dt) {
            GeodesicStepStatus::Advanced { metric_norm } => {
                push_wormhole_sample(ray, spec.mode, &wh, lambda + spec.dt, &state, metric_norm);
                if state.0[1].abs() > 20.0 {
                    break;
                }
            }
            GeodesicStepStatus::HaltedSingularity => {
                ray.halted = true;
                break;
            }
        }
    }
}

fn initial_4d_state(
    spec: &RayLaunchSpec,
    spin_a: f64,
    r0: f64,
) -> (SVector<f64, 4>, SVector<f64, 4>) {
    let kappa = match spec.mode {
        GeodesicMode::NullPhoton => 0.0,
        GeodesicMode::TimelikeParticle => 1.0,
    };
    let rs = spec.rs;
    let delta = (r0 * r0 - rs * r0 + spin_a * spin_a).max(1e-6);
    let g00 = -(1.0 - rs / r0);
    let g03 = -(spin_a * rs) / r0;
    let g11 = (r0 * r0) / delta;
    let g33 = r0 * r0 + spin_a * spin_a + (spin_a * spin_a * rs) / r0;

    let uphi = spec.impact_b / (r0 * r0);
    let ur = spec.radial_bias * 0.45;
    let spatial = g11 * ur * ur + g33 * uphi * uphi + kappa;
    let ut = if g00 < -1e-6 {
        let b_quad = 2.0 * g03 * uphi;
        let disc = (b_quad * b_quad - 4.0 * g00 * spatial).max(0.0);
        (-b_quad - disc.sqrt()) / (2.0 * g00)
    } else {
        (spatial / 0.25).sqrt().max(1.2)
    };
    (
        SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, spec.phi0),
        SVector::<f64, 4>::new(ut.max(0.1), ur, 0.0, uphi),
    )
}

fn push_schwarzschild_sample(
    ray: &mut BundleRay,
    mode: GeodesicMode,
    bh: &SchwarzschildBlackHole,
    rs: f64,
    proper_tau: f64,
    state: &ArrayState<8>,
    metric_norm: f64,
) {
    if ray.samples.len() >= MAX_TRAJECTORY_SAMPLES {
        return;
    }
    let r = state.0[1].max(1.001 * rs);
    let phi = state.0[3];
    let z = flamm_paraboloid_z(rs, r).unwrap_or(0.0);
    let (sin_p, cos_p) = phi.sin_cos();
    let grav = bh.gravitational_time_dilation(r).unwrap_or(0.0);
    let ut = state.0[4].max(1e-6);
    let dphi_dct = state.0[7] / ut;
    let beta_vec = planar_velocity_beta(r, phi, state.0[5] / ut, dphi_dct);
    let beta_mag = (beta_vec[0] * beta_vec[0] + beta_vec[1] * beta_vec[1])
        .sqrt()
        .min(0.995);
    let clock_rate = match mode {
        GeodesicMode::NullPhoton => grav.max(0.05),
        GeodesicMode::TimelikeParticle => {
            combined_clock_rate(bh, r, beta_mag * C).unwrap_or((1.0 / ut).clamp(0.0, 1.0))
        }
    };
    let gamma = 1.0 / (1.0 - beta_mag * beta_mag).max(1e-4).sqrt();
    ray.samples.push(TrajectorySample {
        coord_ct: state.0[0],
        proper_tau,
        radial_coord: r,
        phi,
        radial_vel: state.0[5],
        dphi_dct,
        velocity_beta: beta_vec,
        embed_xyz: [r * cos_p, r * sin_p, z],
        metric_norm,
        clock_rate,
        grav_dilation: grav,
        lorentz_gamma: gamma,
    });
}

fn push_kerr_sample(
    ray: &mut BundleRay,
    mode: GeodesicMode,
    kerr: &KerrEquatorialMetric,
    proper_tau: f64,
    state: &ArrayState<8>,
    metric_norm: f64,
) {
    if ray.samples.len() >= MAX_TRAJECTORY_SAMPLES {
        return;
    }
    let r_plus = kerr.outer_horizon_radius();
    let r = state.0[1].max(1.005 * r_plus);
    let phi = state.0[3];
    let z = flamm_paraboloid_z(r_plus, r).unwrap_or(0.0);
    let (sin_p, cos_p) = phi.sin_cos();
    let ut = state.0[4].max(1e-6);
    let dphi_dct = state.0[7] / ut;
    let zamo_grav = kerr.zamo_clock_dilation(r).unwrap_or(0.0);
    let exact_kerr_clock = kerr
        .clock_dilation(r, dphi_dct)
        .unwrap_or_else(|_| (1.0 / ut).clamp(0.0, 1.0));
    let clock_rate = match mode {
        GeodesicMode::NullPhoton => zamo_grav.max(0.05),
        GeodesicMode::TimelikeParticle => exact_kerr_clock,
    };
    let beta_vec = planar_velocity_beta(r, phi, state.0[5] / ut, dphi_dct);
    let beta_sq = beta_vec[0] * beta_vec[0] + beta_vec[1] * beta_vec[1];
    let gamma = 1.0 / (1.0 - beta_sq.min(0.99)).sqrt();
    ray.samples.push(TrajectorySample {
        coord_ct: state.0[0],
        proper_tau,
        radial_coord: r,
        phi,
        radial_vel: state.0[5],
        dphi_dct,
        velocity_beta: beta_vec,
        embed_xyz: [r * cos_p, r * sin_p, z],
        metric_norm,
        clock_rate,
        grav_dilation: zamo_grav,
        lorentz_gamma: gamma,
    });
}

fn push_wormhole_sample(
    ray: &mut BundleRay,
    mode: GeodesicMode,
    wh: &MorrisThorneWormhole,
    proper_tau: f64,
    state: &ArrayState<6>,
    metric_norm: f64,
) {
    if ray.samples.len() >= MAX_TRAJECTORY_SAMPLES {
        return;
    }
    let l = state.0[1];
    let phi = state.0[2];
    let xyz = wh.embedding_xyz(l, phi).unwrap_or([0.0, 0.0, 0.0]);
    let ut = state.0[3].max(1.0);
    let dphi_dct = state.0[5] / ut;
    let r_areal = wh.areal_radius(l).unwrap_or(wh.throat_radius);
    let beta_vec = planar_velocity_beta(r_areal, phi, state.0[4] / ut, dphi_dct);
    let clock_rate = match mode {
        GeodesicMode::NullPhoton => 1.0,
        GeodesicMode::TimelikeParticle => (1.0 / ut).clamp(0.05, 1.0),
    };
    ray.samples.push(TrajectorySample {
        coord_ct: state.0[0],
        proper_tau,
        radial_coord: l,
        phi,
        radial_vel: state.0[4],
        dphi_dct,
        velocity_beta: beta_vec,
        embed_xyz: xyz,
        metric_norm,
        clock_rate,
        grav_dilation: 1.0,
        lorentz_gamma: ut,
    });
}

fn planar_velocity_beta(r: f64, phi: f64, dr_dct: f64, dphi_dct: f64) -> [f64; 3] {
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
