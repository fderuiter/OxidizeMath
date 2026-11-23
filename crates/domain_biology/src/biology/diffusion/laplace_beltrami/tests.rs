// [cite:graph_parameters_rust]
use super::*;
use crate::biology::morphogenesis::reaction::{GrayScottKinetics, SchnakenbergKinetics};
use crate::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
use math_commons::ast::{CompiledExpr, parse};
use math_commons::math_kernel::types::Dimension;
use pure_math::pure_math::differential_geometry::buckled_flow::BuckledSurfaceFlow;
use pure_math::pure_math::differential_geometry::surface::{
    CompiledHeightfieldSurface, KleinBottle, Sphere, Torus,
};
use std::f64::consts::{PI, TAU};
use std::sync::atomic::Ordering;
use verified_engine::allocator::MEMORY_LOCK;
use verified_engine::engine::VerifiedEngine;

#[test]
fn turing_and_buckled_flow_steps_allocate_zero_bytes_under_verified_engine() {
    let torus = Torus {
        major_radius: 2.2,
        minor_radius: 0.7,
    };
    let (nu, nv) = (18, 18);
    let diff = LaplaceBeltramiDiffusion2D::from_surface(
        &torus,
        SurfaceTopology::PeriodicTorus,
        (0.0, TAU),
        (0.0, TAU),
        nu,
        nv,
    );
    let schnak = SchnakenbergKinetics::new(0.1, 0.9);
    let len = nu * nv;
    let mut turing = TuringSystem::new_with_kinetics(
        Dimension(len),
        DiffusionCoeff(0.05),
        DiffusionCoeff(1.0),
        schnak,
        diff.clone(),
    );
    for k in 0..len {
        turing.u_mut()[k] = 1.0 + 0.05 * (k as f64 * 0.37).sin();
        turing.v_mut()[k] = 0.9;
    }
    let mut flow = BuckledSurfaceFlow::from_parametric(
        &torus,
        SurfaceTopology::PeriodicTorus,
        (0.0, TAU),
        (0.0, TAU),
        nu,
        nv,
    );

    let (ok, metrics) = VerifiedEngine::run_verified(|| {
        turing.step(0.005);
        flow.step_buckled(turing.u(), 1.0, 0.2, 0.05, 0.1, 0.005)
    });
    assert!(ok);
    assert!(metrics.function_calls >= 2);
    assert!(!MEMORY_LOCK.load(Ordering::SeqCst));

    let gs = GrayScottKinetics::new(0.037, 0.06);
    let mut gs_sys = TuringSystem::new_with_kinetics(
        Dimension(len),
        DiffusionCoeff(0.02),
        DiffusionCoeff(0.01),
        gs,
        diff,
    );
    gs_sys.u_mut().fill(1.0);
    gs_sys.v_mut().fill(0.25);
    let ((), gs_metrics) = VerifiedEngine::run_verified(|| {
        gs_sys.step(0.01);
    });
    assert!(gs_metrics.function_calls >= 1);
}

#[test]
fn torus_and_klein_bottle_propagate_seamlessly_across_wrapped_and_twisted_seams() {
    let torus = Torus {
        major_radius: 2.0,
        minor_radius: 0.6,
    };
    let (nu, nv) = (20, 20);
    let periodic = LaplaceBeltramiDiffusion2D::from_surface(
        &torus,
        SurfaceTopology::PeriodicTorus,
        (0.0, TAU),
        (0.0, TAU),
        nu,
        nv,
    );
    let clamped = LaplaceBeltramiDiffusion2D::from_surface(
        &torus,
        SurfaceTopology::OpenNeumann,
        (0.0, TAU),
        (0.0, TAU),
        nu,
        nv,
    );
    let mut state_periodic = vec![0.0_f64; nu * nv];
    state_periodic[flatten_2d_index(0, 0, nv)] = 1.0;
    let mut state_clamped = state_periodic.clone();
    let mut next = vec![0.0_f64; nu * nv];

    for _ in 0..6 {
        periodic.map_diffusion([&state_periodic], [0.2], |idx, curr, diffs| {
            next[idx] = curr[0] + 0.005 * diffs[0];
        });
        state_periodic.copy_from_slice(&next);

        clamped.map_diffusion([&state_clamped], [0.2], |idx, curr, diffs| {
            next[idx] = curr[0] + 0.005 * diffs[0];
        });
        state_clamped.copy_from_slice(&next);
    }

    assert!(state_periodic[flatten_2d_index(0, nu - 1, nv)] > 1e-3);
    assert!(state_periodic[flatten_2d_index(nv - 1, 0, nv)] > 1e-3);
    assert_eq!(state_clamped[flatten_2d_index(0, nu - 1, nv)], 0.0);

    // KleinBottle twisted seam maps (i = nu - 1, j = 3) across u = 2pi -> (i = 0, j = 13)
    let klein = KleinBottle { radius: 2.5 };
    let klein_op = LaplaceBeltramiDiffusion2D::from_surface(
        &klein,
        SurfaceTopology::TwistedKlein,
        (0.0, TAU),
        (0.0, TAU),
        nu,
        nv,
    );
    let mut k_state = vec![0.0_f64; nu * nv];
    k_state[flatten_2d_index(3, nu - 1, nv)] = 1.0;
    klein_op.map_diffusion([&k_state], [0.25], |idx, curr, diffs| {
        next[idx] = curr[0] + 0.01 * diffs[0];
    });
    let twisted_target = next[flatten_2d_index(13, 0, nv)];
    let untwisted_target = next[flatten_2d_index(3, 0, nv)];
    assert!(
        twisted_target > 1e-3 && twisted_target > 50.0 * untwisted_target.abs(),
        "Klein seam must route flux to twisted partner j=13 ({twisted_target}) rather than j=3 ({untwisted_target})"
    );
}

#[test]
fn sphere_polar_caps_diffuse_isotropically_without_singularity() {
    let sphere = Sphere { radius: 1.2 };
    let (nu, nv) = (16, 13);
    let sph_op = LaplaceBeltramiDiffusion2D::from_surface(
        &sphere,
        SurfaceTopology::PolarSphere,
        (0.0, TAU),
        (0.0, PI),
        nu,
        nv,
    );
    let mut u_state = vec![0.0_f64; nu * nv];
    for i in 0..nu {
        u_state[flatten_2d_index(0, i, nv)] = 1.0;
    }
    let mut next = vec![0.0_f64; nu * nv];
    for _ in 0..8 {
        let dt = sph_op.clamp_dt(0.005, 0.15);
        sph_op.map_diffusion([&u_state], [0.15], |idx, curr, diffs| {
            next[idx] = curr[0] + dt * diffs[0];
        });
        u_state.copy_from_slice(&next);
    }
    assert!(u_state.iter().all(|v| v.is_finite()));
    let ring1_first = u_state[flatten_2d_index(1, 0, nv)];
    assert!(ring1_first > 0.0);
    for i in 1..nu {
        assert!((u_state[flatten_2d_index(1, i, nv)] - ring1_first).abs() < 1e-10);
    }
}

#[test]
fn torus_curvature_modulates_schnakenberg_patterns() {
    let torus = Torus {
        major_radius: 2.0,
        minor_radius: 0.9,
    };
    let (nu, nv) = (24, 24);
    let torus_op = LaplaceBeltramiDiffusion2D::from_surface(
        &torus,
        SurfaceTopology::PeriodicTorus,
        (0.0, TAU),
        (0.0, TAU),
        nu,
        nv,
    );
    let outer_idx = flatten_2d_index(0, 0, nv);
    let inner_idx = flatten_2d_index(nv / 2, 0, nv);
    assert!(torus_op.gaussian_curvature()[outer_idx] > 0.0);
    assert!(torus_op.gaussian_curvature()[inner_idx] < 0.0);
    assert!(torus_op.metric_anisotropy()[outer_idx] > 1.5);

    let schnak = SchnakenbergKinetics::new(0.1, 0.9);
    let mut sys = TuringSystem::new_with_kinetics(
        Dimension(nu * nv),
        DiffusionCoeff(0.08),
        DiffusionCoeff(1.2),
        schnak,
        torus_op,
    );
    sys.u_mut().fill(1.0);
    sys.v_mut().fill(0.9);
    for i in 0..nu {
        let wave = 0.25 * (4.0 * TAU * i as f64 / nu as f64).cos();
        sys.u_mut()[flatten_2d_index(0, i, nv)] += wave;
        sys.u_mut()[flatten_2d_index(nv / 2, i, nv)] += wave;
    }
    for _ in 0..80 {
        sys.step(0.015);
    }
    let u_after = sys.u();
    let outer_amp = (0..nu)
        .map(|i| (u_after[flatten_2d_index(0, i, nv)] - 1.0).abs())
        .fold(0.0_f64, f64::max);
    let inner_amp = (0..nu)
        .map(|i| (u_after[flatten_2d_index(nv / 2, i, nv)] - 1.0).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        outer_amp > inner_amp * 1.15,
        "Outer positive-K equator ({outer_amp}) must retain higher azimuthal harmonic amplitude than compressed negative-K throat ({inner_amp})"
    );
}

#[test]
fn saddle_anisotropy_modulates_gray_scott_patterns() {
    let expr =
        CompiledExpr::compile_with_vars(&parse("0.7 * (x^2 - y^2)").unwrap(), &["x", "y"]).unwrap();
    let saddle = CompiledHeightfieldSurface::new(expr, &[]);
    let (su, sv) = (21, 21);
    let saddle_op = LaplaceBeltramiDiffusion2D::from_surface(
        &saddle,
        SurfaceTopology::OpenNeumann,
        (-1.5, 1.5),
        (-1.5, 1.5),
        su,
        sv,
    );
    let center = flatten_2d_index(sv / 2, su / 2, sv);
    assert!(saddle_op.gaussian_curvature()[center] < -0.5);
    let off_axis = flatten_2d_index(sv / 2, su - 3, sv);
    assert!(saddle_op.metric_anisotropy()[off_axis] > 1.4);

    let gs = GrayScottKinetics::new(0.037, 0.06);
    let mut gs_saddle = TuringSystem::new_with_kinetics(
        Dimension(su * sv),
        DiffusionCoeff(0.04),
        DiffusionCoeff(0.02),
        gs,
        saddle_op,
    );
    gs_saddle.u_mut().fill(1.0);
    gs_saddle.v_mut().fill(0.0);
    gs_saddle.u_mut()[off_axis] = 0.5;
    gs_saddle.v_mut()[off_axis] = 0.25;
    for _ in 0..15 {
        gs_saddle.step(0.005);
    }
    let gv_after = gs_saddle.v();
    let spread_x = gv_after[flatten_2d_index(sv / 2, su - 2, sv)];
    let spread_y = gv_after[flatten_2d_index(sv / 2 + 1, su - 3, sv)];
    assert!(
        spread_y > spread_x * 1.2,
        "Stretched x-slope (E > G) on saddle must slow x-diffusion ({spread_x}) relative to flat y-direction ({spread_y})"
    );
}
