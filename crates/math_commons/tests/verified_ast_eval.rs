//! Zero-allocation runtime verification tests for AST expression evaluators.

use math_commons::ast::{CompiledExpr, Expr, parse};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::{E, PI, TAU};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn check_compiled_hot_loops(
    c1d: &CompiledExpr,
    c2d: &CompiledExpr,
    c3d: &CompiledExpr,
    cz: &CompiledExpr,
) {
    let params_1d = [1.5, 0.8];
    let mut acc = 0.0;
    for i in -50..=50 {
        let x = f64::from(i) * 0.04;
        acc += c1d.eval_1d(x, &params_1d);
    }
    assert!(acc.is_finite());

    for ix in -8..=8 {
        for iy in -8..=8 {
            let z_surf = c2d.eval_2d(f64::from(ix) * 0.2, f64::from(iy) * 0.2, &[0.6]);
            assert!(z_surf.is_finite());
        }
    }

    let mse_term = c3d.eval_3d(0.5, 1.2, -0.4, &[0.25]);
    assert!(mse_term.is_finite());

    for step in 1..=100 {
        let re = f64::from(step) * 0.03;
        let w = cz.eval_complex_z_params(Complex64::new(re, 0.2), &[0.85]);
        assert!(w.re.is_finite() && w.im.is_finite());
    }
}

fn check_real_eval(expr: &Expr, vars: &HashMap<String, f64>) {
    let val = expr.eval(vars).expect("eval should succeed");
    let expected = 2.5 * 4.0 + PI + TAU - E;
    assert!((val - expected).abs() < 1e-12);
}

fn check_ode_rk4_eval_2d(expr: &Expr, unknown: &Expr, params: &HashMap<String, f64>) {
    let dt = 0.01;
    let mut t = 0.0;
    let mut y = 1.0;
    for _ in 0..200 {
        let k1 = expr.eval_2d(t, y, params);
        let k2 = expr.eval_2d(t + 0.5 * dt, y + 0.5 * dt * k1, params);
        let k3 = expr.eval_2d(t + 0.5 * dt, y + 0.5 * dt * k2, params);
        let k4 = expr.eval_2d(t + dt, y + dt * k3, params);
        y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += dt;
    }
    assert!(y.is_finite());
    assert!(unknown.eval_2d(t, y, params).is_nan());
}

fn check_complex_eval(expr: &Expr, vars: &HashMap<String, Complex64>) {
    let val = expr
        .eval_complex(vars)
        .expect("eval_complex should succeed");
    let expected =
        Complex64::new(2.0, 1.0) * Complex64::new(3.0, -0.5) + Complex64::new(PI + E, TAU);
    assert!((val - expected).norm() < 1e-12);
}

fn check_conformal_eval_complex_z(expr: &Expr, unknown: &Expr) {
    for step in 1..=250 {
        let re = f64::from(step) * 0.02;
        let z = Complex64::new(re, -0.5 * re);
        let w = expr.eval_complex_z(z);
        assert!(w.re.is_finite() && w.im.is_finite());
    }
    let bad = unknown.eval_complex_z(Complex64::new(1.0, 1.0));
    assert!(bad.re.is_nan() && bad.im.is_nan());
}

#[test]
fn test_ast_evaluators_zero_allocations_with_global_allocator() {
    // Initialize all AST nodes and variable maps prior to locking allocations.
    let real_expr = parse("k * alpha + PI + Tau - E").expect("parse real");
    let mut real_vars = HashMap::with_capacity(4);
    real_vars.insert("k".to_string(), 2.5);
    real_vars.insert("Alpha".to_string(), 4.0);

    let ode_expr = parse("-k * Y * sin(omega * T + PI / Tau) + 0 * E").expect("parse ode");
    let unknown_expr = parse("missing_identifier + 1").expect("parse unknown");
    let mut ode_params = HashMap::with_capacity(4);
    ode_params.insert("k".to_string(), 1.2);
    ode_params.insert("Omega".to_string(), 2.0);

    let complex_expr = parse("z0 * gain + PI + E + Tau * I").expect("parse complex");
    let mut complex_vars = HashMap::with_capacity(4);
    complex_vars.insert("z0".to_string(), Complex64::new(2.0, 1.0));
    complex_vars.insert("Gain".to_string(), Complex64::new(3.0, -0.5));

    let conformal_expr = parse("Z^3 - 1 / z + PI * I + Tau - E").expect("parse conformal");
    let compiled_1d = CompiledExpr::compile_with_vars(
        &parse(
            "clamp(a * sin(x), -b, b) + step(x) - rect(x) + min(x, 1) + max(x, -1) + atan2(x, 1)",
        )
        .unwrap(),
        &["x"],
    )
    .expect("compile 1d");
    let compiled_2d =
        CompiledExpr::compile_with_vars(&parse("a * (x^2 - y^2)").unwrap(), &["x", "y"])
            .expect("compile 2d");
    let compiled_3d =
        CompiledExpr::compile_with_vars(&parse("w * sin(b * x + phi)").unwrap(), &["x", "w", "b"])
            .expect("compile 3d");
    let compiled_cz = CompiledExpr::compile_complex(&parse("z + a^2 / z + i").unwrap(), &["z"])
        .expect("compile complex z");

    // Keep all VerifiedEngine::run_verified assertions in one test to prevent MEMORY_LOCK races.
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        check_real_eval(&real_expr, &real_vars);
        check_ode_rk4_eval_2d(&ode_expr, &unknown_expr, &ode_params);
        check_complex_eval(&complex_expr, &complex_vars);
        check_conformal_eval_complex_z(&conformal_expr, &unknown_expr);
        check_compiled_hot_loops(&compiled_1d, &compiled_2d, &compiled_3d, &compiled_cz);
    });

    assert_eq!(metrics.memory_loads, 0);
}
