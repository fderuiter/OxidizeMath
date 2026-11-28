//! Zero-allocation RK4 worldline and photon-fan bundle integrators across Schwarzschild,
//! Kerr equatorial, Morris-Thorne wormhole, and custom `CompiledExpr` spacetimes.

use std::f64::consts::FRAC_PI_2;

use math_commons::ast::CompiledExpr;
use nalgebra::SVector;
use pure_math::pure_math::analysis::ode::{ArrayState, OdeSystem, RungeKutta4};
use pure_math::pure_math::tensor::{
    GeodesicState, GeodesicStepStatus, GeodesicSystem, StaticMetric,
};

use super::custom_metric::{
    MIN_REGULAR_LAPSE, MIN_REGULAR_THROAT, build_lapse_static_metric, build_wormhole_static_metric,
};
use super::kerr::KerrEquatorialMetric;
use super::schwarzschild::{GeodesicKind, SchwarzschildBlackHole};
use super::worldline::{
    BundleRay, CustomWorldlineCtx, MAX_BUNDLE_RAYS, MAX_TRAJECTORY_SAMPLES, Sample4dMeta,
    TrajectorySample, initial_diagonal_4d_state, initial_kerr_4d_state, push_4d_worldline_sample,
    push_custom_lapse_worldline_sample, push_custom_wormhole_worldline_sample,
    push_wormhole_worldline_sample, wormhole_initial_vel,
};
use super::wormhole::MorrisThorneWormhole;

/// Launch parameters shared across 4D black hole, custom-lapse, and 3D wormhole worldline integrators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldlineLaunchSpec {
    /// Null photon vs massive timelike particle.
    pub kind: GeodesicKind,
    /// Schwarzschild radius $R_s$ or wormhole throat scale $b_0$.
    pub scale_radius: f64,
    /// Dimensionless Kerr spin $\chi \in (-0.95, 0.95)$.
    pub spin_chi: f64,
    /// Initial radial coordinate $r_0$ or proper throat distance $l_0$.
    pub initial_radius: f64,
    /// Initial azimuthal angle $\varphi_0$ in radians.
    pub launch_azimuth: f64,
    /// Inward/outward radial velocity bias in $[-0.98, 0.98]$.
    pub radial_bias: f64,
    /// Affine/proper RK4 integration step $\Delta\lambda$.
    pub step_size: f64,
}

/// Refines a pre-horizon RK4 halt using stack-allocated micro-steps so capture lands near $1.02 r_+$.
#[verified_engine::verified]
pub fn refine_halt_4d<M>(
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
    debug_assert!(cur_lambda.is_finite());
    (advanced_any, cur_lambda)
}

/// Integrates a 4D Schwarzschild worldline into a pre-allocated `Vec<TrajectorySample>`.
///
/// # Examples
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::high_energy::{
///     integrate_schwarzschild_worldline_into, GeodesicKind, TrajectorySample,
///     WorldlineLaunchSpec, MAX_TRAJECTORY_SAMPLES,
/// };
///
/// let spec = WorldlineLaunchSpec {
///     kind: GeodesicKind::Null,
///     scale_radius: 2.0,
///     spin_chi: 0.0,
///     initial_radius: 10.0,
///     launch_azimuth: 0.0,
///     radial_bias: -0.82,
///     step_size: 0.04,
/// };
/// let mut out = Vec::with_capacity(MAX_TRAJECTORY_SAMPLES + 8);
/// let (halted, energy) = integrate_schwarzschild_worldline_into(&spec, 120, 6.0, &mut out);
/// assert!(!halted && (energy - 1.0).abs() < 1.0e-12 && out.len() > 2);
/// ```
#[verified_engine::verified]
pub fn integrate_schwarzschild_worldline_into(
    spec: &WorldlineLaunchSpec,
    steps: usize,
    impact_val: f64,
    out: &mut Vec<TrajectorySample>,
) -> (bool, f64) {
    out.clear();
    let rs = spec.scale_radius;
    let Ok(bh) = SchwarzschildBlackHole::from_schwarzschild_radius(rs) else {
        return (false, 1.0);
    };
    let r0 = spec.initial_radius.max(1.05 * rs);
    let (pos0, vel0, e_val) = initial_kerr_4d_state(
        spec.kind,
        rs,
        0.0,
        r0,
        spec.launch_azimuth,
        impact_val,
        spec.radial_bias,
    );
    let meta = Sample4dMeta {
        kind: spec.kind,
        bh: Some(&bh),
        throat_r: rs,
        spin_a: 0.0,
    };
    let halted = run_4d_worldline_rk4(bh, pos0, vel0, spec.step_size, steps, meta, out);
    (halted, e_val)
}

/// Integrates a 4D equatorial Kerr worldline into a pre-allocated `Vec<TrajectorySample>`.
#[verified_engine::verified]
pub fn integrate_kerr_worldline_into(
    spec: &WorldlineLaunchSpec,
    steps: usize,
    impact_val: f64,
    out: &mut Vec<TrajectorySample>,
) -> (bool, f64) {
    out.clear();
    let rs = spec.scale_radius;
    let a = 0.5 * rs * spec.spin_chi.clamp(-0.95, 0.95);
    let Ok(kerr) = KerrEquatorialMetric::new(rs, a) else {
        return (false, 1.0);
    };
    let r_plus = kerr.outer_horizon_radius();
    let r0 = spec.initial_radius.max(1.05 * r_plus);
    let (pos0, vel0, e_val) = initial_kerr_4d_state(
        spec.kind,
        rs,
        a,
        r0,
        spec.launch_azimuth,
        impact_val,
        spec.radial_bias,
    );
    let meta = Sample4dMeta {
        kind: spec.kind,
        bh: None,
        throat_r: r_plus,
        spin_a: a,
    };
    let halted = run_4d_worldline_rk4(kerr, pos0, vel0, spec.step_size, steps, meta, out);
    (halted, e_val)
}

#[verified_engine::verified]
fn run_4d_worldline_rk4<M>(
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
    push_4d_worldline_sample(out, meta, 0.0, &state, norm0);

    for step in 0..steps.min(MAX_TRAJECTORY_SAMPLES) {
        let lambda = (step as f64) * dt;
        match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
            GeodesicStepStatus::Advanced { metric_norm } => {
                push_4d_worldline_sample(out, meta, lambda + dt, &state, metric_norm);
                if state.0[1] > 28.0 {
                    return false;
                }
            }
            GeodesicStepStatus::HaltedSingularity => {
                let (did_adv, sub_lam) = refine_halt_4d(&sys, &mut solver, lambda, &mut state, dt);
                if did_adv && let Ok(n_end) = sys.metric_norm(&state) {
                    push_4d_worldline_sample(out, meta, sub_lam, &state, n_end);
                }
                return true;
            }
        }
    }
    false
}

/// Integrates a 3D Morris-Thorne traversable wormhole worldline into a pre-allocated `Vec<TrajectorySample>`.
#[verified_engine::verified]
pub fn integrate_wormhole_worldline_into(
    spec: &WorldlineLaunchSpec,
    steps: usize,
    impact_val: f64,
    out: &mut Vec<TrajectorySample>,
) -> (bool, f64) {
    out.clear();
    let b0 = spec.scale_radius;
    let Ok(wh) = MorrisThorneWormhole::new(b0) else {
        return (false, 1.0);
    };
    let l0 = spec.initial_radius;
    let (ut, ul, uphi, e_val) =
        wormhole_initial_vel(spec.kind, b0 * b0 + l0 * l0, impact_val, spec.radial_bias);
    let sys = GeodesicSystem::new(wh);
    let mut state = ArrayState::<6>::from_pos_vel(
        &SVector::<f64, 3>::new(0.0, l0, spec.launch_azimuth),
        &SVector::<f64, 3>::new(ut, ul, uphi),
    );
    let mut solver = RungeKutta4::new(&state);
    let dt = spec.step_size;
    let norm0 = sys.metric_norm(&state).unwrap_or(0.0);
    push_wormhole_worldline_sample(out, spec.kind, &wh, 0.0, &state, norm0);

    for step in 0..steps.min(MAX_TRAJECTORY_SAMPLES) {
        let lambda = (step as f64) * dt;
        match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
            GeodesicStepStatus::Advanced { metric_norm } => {
                push_wormhole_worldline_sample(
                    out,
                    spec.kind,
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

/// Integrates a 4D custom radial-lapse $f(r; R_s)$ worldline into a pre-allocated `Vec<TrajectorySample>`.
#[verified_engine::verified]
pub fn integrate_custom_lapse_worldline_into(
    spec: &WorldlineLaunchSpec,
    compiled: &CompiledExpr,
    params: &[f64],
    steps: usize,
    impact_val: f64,
    out: &mut Vec<TrajectorySample>,
) -> (bool, f64) {
    out.clear();
    let rs = spec.scale_radius;
    let r0 = spec.initial_radius.max(0.25);
    let f0 = compiled.eval_2d(r0, rs, params);
    if !f0.is_finite() || f0 <= MIN_REGULAR_LAPSE {
        return (true, 1.0);
    }
    let (pos0, vel0, e_val) = initial_diagonal_4d_state(
        spec.kind,
        r0,
        spec.launch_azimuth,
        f0,
        impact_val,
        spec.radial_bias,
    );
    let ctx = CustomWorldlineCtx {
        kind: spec.kind,
        compiled,
        scale: rs,
        params,
    };
    let halted = step_custom_lapse_rk4(ctx, pos0, vel0, spec.step_size, steps, out);
    (halted, e_val)
}

#[verified_engine::verified]
fn step_custom_lapse_rk4(
    ctx: CustomWorldlineCtx<'_>,
    pos0: SVector<f64, 4>,
    vel0: SVector<f64, 4>,
    dt: f64,
    steps: usize,
    out: &mut Vec<TrajectorySample>,
) -> bool {
    let sys = GeodesicSystem::new(build_lapse_static_metric(
        ctx.compiled,
        ctx.scale,
        ctx.params,
    ));
    let mut state = ArrayState::<8>::from_pos_vel(&pos0, &vel0);
    let mut solver = RungeKutta4::new(&state);
    push_custom_lapse_worldline_sample(
        out,
        ctx,
        0.0,
        &state,
        sys.metric_norm(&state).unwrap_or(0.0),
    );
    for step in 0..steps.min(MAX_TRAJECTORY_SAMPLES) {
        let lambda = (step as f64) * dt;
        let prev = state;
        match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
            GeodesicStepStatus::Advanced { metric_norm } => {
                let pos = SVector::<f64, 4>::new(state.0[0], state.0[1], FRAC_PI_2, state.0[3]);
                if sys.metric().christoffel_at(&pos).is_err() {
                    state = prev;
                    let (_, sub_lam) = refine_halt_4d(&sys, &mut solver, lambda, &mut state, dt);
                    if let Ok(n_end) = sys.metric_norm(&state) {
                        push_custom_lapse_worldline_sample(out, ctx, sub_lam, &state, n_end);
                    }
                    return true;
                }
                push_custom_lapse_worldline_sample(out, ctx, lambda + dt, &state, metric_norm);
                if state.0[1] > 28.0 {
                    return false;
                }
            }
            GeodesicStepStatus::HaltedSingularity => {
                let (did_adv, sub_lam) = refine_halt_4d(&sys, &mut solver, lambda, &mut state, dt);
                if did_adv && let Ok(n_end) = sys.metric_norm(&state) {
                    push_custom_lapse_worldline_sample(out, ctx, sub_lam, &state, n_end);
                }
                return true;
            }
        }
    }
    false
}

/// Integrates a 3D custom wormhole profile $b(l; b_0)$ worldline into a pre-allocated `Vec<TrajectorySample>`.
#[verified_engine::verified]
pub fn integrate_custom_wormhole_worldline_into(
    spec: &WorldlineLaunchSpec,
    compiled: &CompiledExpr,
    params: &[f64],
    steps: usize,
    impact_val: f64,
    out: &mut Vec<TrajectorySample>,
) -> (bool, f64) {
    out.clear();
    let b0 = spec.scale_radius;
    let l0 = spec.initial_radius;
    let b_init = compiled.eval_2d(l0, b0, params);
    if !b_init.is_finite() || b_init <= MIN_REGULAR_THROAT {
        return (true, 1.0);
    }
    let (ut, ul, uphi, e_val) =
        wormhole_initial_vel(spec.kind, b_init * b_init, impact_val, spec.radial_bias);
    let ctx = CustomWorldlineCtx {
        kind: spec.kind,
        compiled,
        scale: b0,
        params,
    };
    let sys = GeodesicSystem::new(build_wormhole_static_metric(compiled, b0, params));
    let mut state = ArrayState::<6>::from_pos_vel(
        &SVector::<f64, 3>::new(0.0, l0, spec.launch_azimuth),
        &SVector::<f64, 3>::new(ut, ul, uphi),
    );
    let mut solver = RungeKutta4::new(&state);
    let dt = spec.step_size;
    push_custom_wormhole_worldline_sample(
        out,
        ctx,
        0.0,
        &state,
        sys.metric_norm(&state).unwrap_or(0.0),
    );
    for step in 0..steps.min(MAX_TRAJECTORY_SAMPLES) {
        let lambda = (step as f64) * dt;
        let prev = state;
        match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
            GeodesicStepStatus::Advanced { metric_norm } => {
                let pos = SVector::<f64, 3>::new(state.0[0], state.0[1], state.0[2]);
                if sys.metric().christoffel_at(&pos).is_err() {
                    state = prev;
                    let _ = state;
                    return (true, e_val);
                }
                push_custom_wormhole_worldline_sample(out, ctx, lambda + dt, &state, metric_norm);
                if state.0[1].abs() > 20.0 {
                    return (false, e_val);
                }
            }
            GeodesicStepStatus::HaltedSingularity => return (true, e_val),
        }
    }
    (false, e_val)
}

/// Integrates a fan of `2..=MAX_BUNDLE_RAYS` rays centered at `center_b` with total spread `fan_spread`
/// into pre-allocated [`BundleRay`] buffers without heap allocations.
#[verified_engine::verified]
pub fn integrate_photon_fan_bundle_into<F>(
    fan_count: usize,
    center_b: f64,
    fan_spread: f64,
    steps: usize,
    bundle_rays: &mut [BundleRay],
    mut integrate_ray: F,
) -> (usize, bool, f64)
where
    F: FnMut(usize, f64, &mut Vec<TrajectorySample>) -> (bool, f64),
{
    let cap = bundle_rays.len().min(MAX_BUNDLE_RAYS);
    if cap < 2 {
        return (0, false, 1.0);
    }
    let ray_count = fan_count.clamp(2, cap);
    let center_idx = ray_count / 2;
    let mut center_energy = 1.0;

    for (idx, ray) in bundle_rays.iter_mut().take(ray_count).enumerate() {
        let frac = (idx as f64) / ((ray_count - 1) as f64) - 0.5;
        let impact_b = (center_b + frac * fan_spread).clamp(-16.0, 16.0);
        ray.samples.clear();
        let (halted, e_val) = integrate_ray(steps, impact_b, &mut ray.samples);
        ray.halted = halted;
        ray.impact_b = impact_b;
        if idx == center_idx {
            center_energy = e_val;
        }
    }
    let all_halted = bundle_rays[..ray_count].iter().all(|r| r.halted);
    debug_assert!(center_energy.is_finite());
    (ray_count, all_halted, center_energy)
}

// [cite:general_relativity_geodesics]

#[cfg(test)]
#[path = "worldline_integrators_tests.rs"]
mod tests;
