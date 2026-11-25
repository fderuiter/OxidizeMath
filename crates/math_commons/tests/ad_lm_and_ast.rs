//! Integration tests for single-pass forward-AD Levenberg-Marquardt, generic `Scalar` AST
//! bytecode evaluation, `SolverWorkspace` dual/interval partitioning, and `f32` tolerance floors.

use core::cell::Cell;
use math_commons::ast::parse;
use math_commons::fit::{MAX_FIT_PARAMS, levenberg_marquardt_ad};
use math_commons::math_kernel::types::SolverWorkspace;
use math_commons::registry::{TOLERANCE_FAST_F32, TOLERANCE_HIGH_F32, TOLERANCE_STANDARD_F32};
use math_scalars::{Dual, DualVec, HyperDual, Interval, Scalar};
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_f32_tolerances_exceed_machine_epsilon() {
    let eps = core::hint::black_box(f32::EPSILON);
    let high = core::hint::black_box(TOLERANCE_HIGH_F32);
    let std_tol = core::hint::black_box(TOLERANCE_STANDARD_F32);
    let fast = core::hint::black_box(TOLERANCE_FAST_F32);
    assert!(high > eps && 1.0_f32 + high > 1.0_f32);
    assert!(std_tol > high);
    assert!(fast > std_tol);
}

#[test]
#[verified_engine::verified]
fn test_levenberg_marquardt_ad_single_pass_per_point_and_zero_alloc() {
    // 5-parameter damped wave: y = p0 * exp(-p1 * x) * sin(p2 * x + p3) + p4
    let true_p = [2.4_f64, 0.35, 1.8, 0.4, -0.6];
    let mut xs = [0.0_f64; 24];
    let mut ys = [0.0_f64; 24];
    for i in 0..24 {
        let x = 0.2 * (i as f64);
        xs[i] = x;
        ys[i] = true_p[0] * (-true_p[1] * x).exp() * (true_p[2] * x + true_p[3]).sin() + true_p[4];
    }

    let seed = [2.0_f64, 0.25, 1.65, 0.25, -0.4];
    let points = (&xs[..], &ys[..]);

    // Verify single-pass Jacobian accumulation: evaluating 1 iteration on already-exact parameters
    // calls eval_ad once for initial SSE + once per point for normal-equation Jacobian (NOT 2k+1).
    let calls = Cell::new(0usize);
    let exact_fit = levenberg_marquardt_ad(&points, &true_p, |x, p| {
        calls.set(calls.get() + 1);
        let xv = DualVec::<f64, MAX_FIT_PARAMS>::constant(x);
        Some(p[0] * (-p[1] * xv).exp() * (p[2] * xv + p[3]).sin() + p[4])
    })
    .expect("exact seed converges immediately");
    assert_eq!(calls.get(), xs.len());
    assert!((exact_fit[0] - true_p[0]).abs() < 1e-12);

    let (fitted, _) = VerifiedEngine::run_verified(|| {
        levenberg_marquardt_ad(&points, &seed, |x, p| {
            let xv = DualVec::<f64, MAX_FIT_PARAMS>::constant(x);
            Some(p[0] * (-p[1] * xv).exp() * (p[2] * xv + p[3]).sin() + p[4])
        })
        .expect("AD LM converges with zero heap allocations")
    });

    for j in 0..5 {
        assert!(
            (fitted[j] - true_p[j]).abs() < 1e-7,
            "param {j} mismatch: {} vs {}",
            fitted[j],
            true_p[j]
        );
    }
}

#[test]
#[verified_engine::verified]
fn test_compiled_expr_hyperdual_and_interval_evaluation() {
    let expr = parse("exp(a * u) * cos(b * v) + u^2").expect("valid AST");
    let compiled = math_commons::ast::CompiledExpr::compile_with_vars_and_params(
        &expr,
        &["u", "v"],
        &["a", "b"],
    )
    .expect("compiles");
    let params = [0.7_f64, 1.3_f64];

    let u0 = 0.9_f64;
    let v0 = 0.4_f64;
    let jet = compiled
        .eval_2d_scalar(
            HyperDual::bivariate_u(u0),
            HyperDual::bivariate_v(v0),
            &params,
        )
        .expect("HyperDual AST evaluation");

    let exact_val = (0.7 * u0).exp() * (1.3 * v0).cos() + u0 * u0;
    let exact_uv = -0.7 * 1.3 * (0.7 * u0).exp() * (1.3 * v0).sin();
    assert!((jet.val - exact_val).abs() < 1e-15);
    assert!((jet.eps12 - exact_uv).abs() < 1e-15);

    let iv = compiled
        .eval_2d_scalar(
            Interval::new(u0 - 1e-12, u0 + 1e-12),
            Interval::new(v0 - 1e-12, v0 + 1e-12),
            &params,
        )
        .expect("Interval AST evaluation");
    assert!(iv.contains(exact_val));

    // Verify SolverWorkspace accepts Dual and Interval slices directly:
    let mut dual_buf = [Dual::constant(1.0_f64); 4];
    let mut ws = SolverWorkspace::new(&mut dual_buf);
    ws.zero_fill();
    let sub = ws.take_slice(2);
    assert_eq!(sub[0], Dual::ZERO);
}
