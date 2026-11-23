// [cite:math_commons]

use super::*;
use std::collections::HashMap;
use std::f64::consts::{E, PI, TAU};

#[test]
fn test_eval_planar_coupled_variables_and_constants() {
    let mut params = HashMap::new();
    params.insert("k".to_string(), 1.5);
    params.insert("r".to_string(), 2.0);
    params.insert("x".to_string(), 99.0);
    params.insert("y".to_string(), -99.0);
    params.insert("t".to_string(), 42.0);

    let (_, dx_expr) = parse_equation("x' = r * x - k * x * y").expect("parse x'");
    let (_, dy_expr) = parse_equation("y' = k * X * Y - y + sin(T)").expect("parse y'");

    let dx = dx_expr.eval_planar(0.5 * PI, 2.0, 3.0, &params);
    let dy = dy_expr.eval_planar(0.5 * PI, 2.0, 3.0, &params);

    // Explicit (t, x, y) arguments override any stale entries in params
    assert!((dx - (2.0 * 2.0 - 1.5 * 2.0 * 3.0)).abs() < 1e-12);
    assert!((dy - (1.5 * 2.0 * 3.0 - 3.0 + 1.0)).abs() < 1e-12);

    let const_expr = parse("PI + Tau - E + x - y").expect("parse constants");
    let expected = PI + TAU - E + 4.0 - 1.0;
    assert!((const_expr.eval_planar(0.0, 4.0, 1.0, &params) - expected).abs() < 1e-12);
}

#[test]
fn test_eval_planar_error_and_nan_propagation() {
    let params = HashMap::new();
    let div_expr = parse("1 / (x - y)").expect("parse division");
    assert!(div_expr.eval_planar(0.0, 2.0, 2.0, &params).is_nan());
    assert!((div_expr.eval_planar(0.0, 5.0, 3.0, &params) - 0.5).abs() < 1e-12);

    let unknown_expr = parse("x + missing_param").expect("parse unknown");
    assert!(unknown_expr.eval_planar(1.0, 2.0, 3.0, &params).is_nan());
}
