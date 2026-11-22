#![allow(missing_docs)]

use math_commons::ast::{Complex64, EvalError, Func, parse};
use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOC: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_eval_3d_sdf_functions_and_case_insensitive_coords() {
    let expr = parse("max(sqrt(X^2 + y^2 + Z^2) - r, -min(abs(x), abs(Y)) + 0.4)")
        .expect("parse CSG SDF expression");
    let mut params = HashMap::new();
    params.insert("r".to_string(), 2.5);

    let inside = expr.eval_3d(1.0, 1.0, 0.5, &params);
    let sphere_term = (1.0_f64 + 1.0 + 0.25).sqrt() - 2.5;
    let slot_term = -1.0_f64 + 0.4;
    assert!((inside - sphere_term.max(slot_term)).abs() < 1e-12);

    let angle_expr = parse("atan2(y, x) + MIN(z, 2) - MAX(x, -1)").expect("parse atan2/min/max");
    let val = angle_expr.eval_3d(1.0, 1.0, 3.0, &params);
    assert!((val - (FRAC_PI_4 + 2.0 - 1.0)).abs() < 1e-12);

    let neg_axis = parse("atan2(1, 0)").expect("parse atan2 quadrant");
    assert!((neg_axis.eval_3d(0.0, 0.0, 0.0, &params) - FRAC_PI_2).abs() < 1e-12);
}

#[test]
fn test_binary_func_arity_latex_and_nan_propagation() {
    let vars = HashMap::from([("x".to_string(), 3.0), ("y".to_string(), -2.0)]);
    let expr = parse("max(min(x, 1.5), atan2(y, x))").expect("parse nested binary funcs");
    assert_eq!(
        expr.to_latex(),
        r"\max\left(\min\left(x, 1.5\right), \mathrm{atan2}\left(y, x\right)\right)"
    );
    let expected = 1.5_f64.max((-2.0_f64).atan2(3.0));
    assert!((expr.eval(&vars).unwrap() - expected).abs() < 1e-12);

    let wrong_min = parse("min(1)").expect("parse wrong arity");
    assert_eq!(
        wrong_min.eval(&vars),
        Err(EvalError::WrongArgumentCount {
            func: Func::Min,
            expected: 2,
            found: 1,
        })
    );
    assert!(wrong_min.eval_3d(1.0, 2.0, 3.0, &vars).is_nan());

    let nan_prop = parse("min(1 / 0, 5)").expect("parse nan propagation");
    assert!(nan_prop.eval(&vars).unwrap().is_nan());
    assert!(nan_prop.eval_3d(0.0, 0.0, 0.0, &vars).is_nan());

    let cvars = HashMap::from([("z".to_string(), Complex64::new(1.0, 2.0))]);
    let complex_min = parse("min(z, 1)").expect("parse complex min");
    assert!(complex_min.eval_complex(&cvars).unwrap().re.is_nan());
}

#[test]
fn test_eval_3d_zero_heap_allocations_across_64_cubed_voxels() {
    let expr = parse(
        "max(sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x) - iso, atan2(y, x) * 0.01 - 4.0)",
    )
    .expect("parse gyroid + atan2 expression");
    let mut params = HashMap::new();
    params.insert("iso".to_string(), 0.15);

    let dim = 64_usize;
    let total_voxels = dim * dim * dim;
    assert_eq!(total_voxels, 262_144);

    let (accum, _metrics) = VerifiedEngine::run_verified(|| {
        let mut checksum = 0.0_f64;
        let step = 2.0 * std::f64::consts::PI / dim as f64;
        for iz in 0..dim {
            let z = -std::f64::consts::PI + (iz as f64 + 0.5) * step;
            for iy in 0..dim {
                let y = -std::f64::consts::PI + (iy as f64 + 0.5) * step;
                for ix in 0..dim {
                    let x = -std::f64::consts::PI + (ix as f64 + 0.5) * step;
                    checksum += expr.eval_3d(x, y, z, &params);
                }
            }
        }
        checksum
    });

    assert!(accum.is_finite(), "64^3 checksum must be finite: {accum}");
}
