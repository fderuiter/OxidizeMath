use super::*;
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

#[test]
fn test_eval_3d_spatial_coordinates_and_constants() {
    let gyroid =
        parse("sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)").expect("parse gyroid");
    let val = gyroid.eval_3d(0.5, -1.0, 1.5);
    let expected = 0.5_f64.sin() * (-1.0_f64).cos()
        + (-1.0_f64).sin() * 1.5_f64.cos()
        + 1.5_f64.sin() * 0.5_f64.cos();
    assert!((val - expected).abs() < 1e-12);

    // Case-insensitive X, Y, Z and built-in constants
    let mixed = parse("X + 2 * Y - 3 * Z + Pi").expect("parse mixed case");
    assert!((mixed.eval_3d(1.0, 2.0, 3.0) - (1.0 + 4.0 - 9.0 + PI)).abs() < 1e-12);

    // Unknown identifiers in eval_3d return NaN without allocating
    let unknown = parse("x + y + z + omega").expect("parse unknown");
    assert!(unknown.eval_3d(1.0, 2.0, 3.0).is_nan());
}

#[test]
fn test_min_and_max_csg_operators_and_variadic_arity() {
    let csg = parse("max(abs(x) - 10, abs(y) - 10, abs(z) - 10)").expect("parse box csg");
    assert_eq!(csg.eval_3d(0.0, 0.0, 0.0), -10.0);
    assert_eq!(csg.eval_3d(12.0, 3.0, -4.0), 2.0);

    let union_csg = parse("min(x^2 + y^2 + z^2 - 4, (x - 3)^2 + y^2 + z^2 - 4)").expect("union");
    assert_eq!(union_csg.eval_3d(0.0, 0.0, 0.0), -4.0);
    assert_eq!(union_csg.eval_3d(3.0, 0.0, 0.0), -4.0);

    let vars = HashMap::from([
        ("a".to_string(), 3.0),
        ("b".to_string(), -2.0),
        ("c".to_string(), 7.5),
    ]);
    assert_eq!(parse("MIN(a, b, c)").unwrap().eval(&vars).unwrap(), -2.0);
    assert_eq!(parse("Max(a, b, c)").unwrap().eval(&vars).unwrap(), 7.5);

    // NaN propagation
    let nan_min = parse("min(1 / 0, 5)").unwrap();
    assert!(nan_min.eval(&vars).unwrap().is_nan());
    assert!(nan_min.eval_3d(0.0, 0.0, 0.0).is_nan());
    let nan_max = parse("max(5, 1 / 0)").unwrap();
    assert!(nan_max.eval(&vars).unwrap().is_nan());
    assert!(nan_max.eval_3d(0.0, 0.0, 0.0).is_nan());
}

#[test]
fn test_min_and_max_arity_errors_complex_domain_and_latex() {
    let rvars = HashMap::new();
    let cvars = HashMap::from([("z".to_string(), Complex64::new(1.0, 2.0))]);

    for (src, func, found) in [
        ("min()", Func::Min, 0),
        ("min(1)", Func::Min, 1),
        ("max()", Func::Max, 0),
        ("max(1)", Func::Max, 1),
    ] {
        let expr = parse(src).unwrap();
        assert_eq!(
            expr.eval(&rvars),
            Err(EvalError::WrongArgumentCount {
                func,
                expected: 2,
                found
            })
        );
        assert!(expr.eval_3d(1.0, 2.0, 3.0).is_nan());
    }

    let csg = parse("max(abs(x) - 10, abs(y) - 10, abs(z) - 10)").expect("parse box csg");
    let csg_compiled =
        CompiledExpr::compile_with_vars(&csg, &["x", "y", "z"]).expect("compile variadic csg");
    assert_eq!(csg_compiled.eval_3d(0.0, 0.0, 0.0, &[]), -10.0);
    assert_eq!(csg_compiled.eval_3d(12.0, 3.0, -4.0, &[]), 2.0);

    let c_min = parse("min(z, 0.5)").unwrap();
    let c_max = parse("max(z, 2.5)").unwrap();
    assert_eq!(
        c_min.eval_complex(&cvars).unwrap(),
        Complex64::new(0.5, 0.0)
    );
    assert_eq!(
        c_max.eval_complex(&cvars).unwrap(),
        Complex64::new(2.5, 0.0)
    );
    assert_eq!(
        c_min.eval_complex_z(Complex64::new(1.0, 2.0)),
        Complex64::new(0.5, 0.0)
    );
    assert!(
        c_min
            .eval_complex_z(Complex64::new(f64::NAN, 0.0))
            .re
            .is_nan()
    );

    let latex = parse("max(x, min(y, z))").unwrap().to_latex();
    assert_eq!(latex, r"\max\left(x, \min\left(y, z\right)\right)");
}
