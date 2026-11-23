use super::glsl_sim_test_eval::{eval_glsl_complex, eval_glsl_scalar};
use super::*;
use num_complex::Complex64;
use std::collections::HashMap;

#[test]
fn test_glsl_ident_validation_and_literals() {
    assert_eq!(format_glsl_float(2.0).unwrap(), "2.0");
    assert_eq!(format_glsl_float(-0.0).unwrap(), "0.0");
    assert!(matches!(
        format_glsl_float(f64::NAN),
        Err(GlslError::NonFiniteLiteral(_))
    ));
    assert!(matches!(
        format_glsl_float(f64::INFINITY),
        Err(GlslError::NonFiniteLiteral(_))
    ));
    assert!(matches!(
        format_glsl_float(1.0e300),
        Err(GlslError::NonFiniteLiteral(_))
    ));

    for bad in [
        "float",
        "vec2",
        "uniform",
        "main",
        "return",
        "gl_FragCoord",
        "a__b",
        "",
    ] {
        let err = Expr::Variable(bad.to_string()).to_glsl_scalar();
        assert!(
            matches!(err, Err(GlslError::InvalidIdentifier(_))),
            "Expected reject for `{bad}`"
        );
    }
    let expr = parse("Z^2 + 1").unwrap();
    assert_eq!(
        expr.to_glsl_complex_with_vars(&["z"]).unwrap(),
        "cpow(z, vec2(2.0, 0.0)) + vec2(1.0, 0.0)"
    );
    assert!(matches!(
        parse("z + w").unwrap().to_glsl_complex_with_vars(&["z"]),
        Err(GlslError::InvalidIdentifier(v)) if v == "w"
    ));
}

#[test]
fn test_glsl_precedence_associativity_and_no_decrement_token() {
    let sub_right = parse("a - (b - c)").unwrap().to_glsl_scalar().unwrap();
    let sub_left = parse("(a - b) - c").unwrap().to_glsl_scalar().unwrap();
    assert_eq!(sub_right, "a - (b - c)");
    assert_eq!(sub_left, "a - b - c");

    let neg_pow = parse("-3^2").unwrap().to_glsl_scalar().unwrap();
    let paren_pow = parse("(-3)^2").unwrap().to_glsl_scalar().unwrap();
    assert_eq!(eval_glsl_scalar(&neg_pow, &HashMap::new()), -9.0);
    assert_eq!(eval_glsl_scalar(&paren_pow, &HashMap::new()), 9.0);

    let double_neg = parse("-(-x)").unwrap().to_glsl_scalar().unwrap();
    let sub_neg = parse("a - (-b)").unwrap().to_glsl_scalar().unwrap();
    assert!(!double_neg.contains("--"), "Emitted `--` in `{double_neg}`");
    assert!(!sub_neg.contains("--"), "Emitted `--` in `{sub_neg}`");

    let bad_fn = Expr::Function {
        func: Func::Sin,
        args: vec![Expr::Number(1.0), Expr::Number(2.0)],
    };
    assert_eq!(
        bad_fn.to_glsl_scalar(),
        Err(GlslError::WrongArgumentCount {
            func: Func::Sin,
            expected: 1,
            found: 2,
        })
    );
    let bad_clamp = Expr::Function {
        func: Func::Clamp,
        args: vec![Expr::Number(1.0), Expr::Number(2.0)],
    };
    assert_eq!(
        bad_clamp.to_glsl_scalar(),
        Err(GlslError::WrongArgumentCount {
            func: Func::Clamp,
            expected: 3,
            found: 2,
        })
    );
}

#[test]
fn test_glsl_iterative_depth_64_and_100k_node_chain_without_stack_overflow() {
    let mut exact_64 = Expr::Variable("z".to_string());
    for _ in 0..MAX_EVAL_DEPTH {
        exact_64 = Expr::Unary {
            op: UnaryOp::Pos,
            expr: Box::new(exact_64),
        };
    }
    assert_eq!(exact_64.to_glsl_complex_with_vars(&["z"]).unwrap(), "z");

    let depth_65 = Expr::Unary {
        op: UnaryOp::Pos,
        expr: Box::new(exact_64),
    };
    assert_eq!(
        depth_65.to_glsl_complex_with_vars(&["z"]),
        Err(GlslError::DepthLimitExceeded(MAX_EVAL_DEPTH))
    );

    let mut deep = Expr::Number(1.0);
    for _ in 0..100_000 {
        deep = Expr::Unary {
            op: UnaryOp::Pos,
            expr: Box::new(deep),
        };
    }
    assert_eq!(
        deep.to_glsl_scalar(),
        Err(GlslError::DepthLimitExceeded(MAX_EVAL_DEPTH))
    );
    assert_eq!(
        deep.to_glsl_complex(),
        Err(GlslError::DepthLimitExceeded(MAX_EVAL_DEPTH))
    );
}

const SCALAR_PARITY_FORMULAS: &[&str] = &[
    "sin(x)",
    "cos(x)",
    "tan(x)",
    "exp(x)",
    "ln(x)",
    "log10(x)",
    "sqrt(x)",
    "abs(-x)",
    "sinh(x)",
    "cosh(x)",
    "tanh(x)",
    "asin(x)",
    "acos(x)",
    "atan(x)",
    "min(x, k)",
    "max(x, y)",
    "clamp(t, -0.5, 0.8)",
    "step(x) + step(y)",
    "rect(x) + rect(t)",
    "atan2(y, x)",
    "-k * y * sin(t) + pi / tau + e",
];

const SCALAR_GUARDED_FORMULAS: &[&str] = &[
    "1 / 0",
    "ln(-2)",
    "log10(0)",
    "sqrt(-4)",
    "asin(2)",
    "acos(-2)",
    "0^0",
    "0^3",
    "clamp(0.5, 2, -2)",
    "atan2(0, 0)",
    "atan2(0, -1)",
];

#[test]
fn test_scalar_numerical_parity_across_all_20_funcs_and_guards() {
    let mut cpu_vars = HashMap::new();
    cpu_vars.insert("x".to_string(), 0.42_f64);
    cpu_vars.insert("k".to_string(), 1.7_f64);
    cpu_vars.insert("y".to_string(), -0.85_f64);
    cpu_vars.insert("t".to_string(), 1.25_f64);
    let gpu_vars = HashMap::from([
        ("x", 0.42_f32),
        ("k", 1.7_f32),
        ("y", -0.85_f32),
        ("t", 1.25_f32),
    ]);

    for &src in SCALAR_PARITY_FORMULAS {
        let ast = parse(src).unwrap();
        let cpu = ast.eval(&cpu_vars).unwrap() as f32;
        let glsl = ast.to_glsl_scalar_with_vars(&["x", "k", "y", "t"]).unwrap();
        let gpu = eval_glsl_scalar(&glsl, &gpu_vars);
        assert!(
            (cpu - gpu).abs() < 1.0e-4,
            "Mismatch on `{src}`: cpu={cpu}, gpu={gpu}"
        );
    }

    for &guarded in SCALAR_GUARDED_FORMULAS {
        let glsl = parse(guarded).unwrap().to_glsl_scalar().unwrap();
        let gpu = eval_glsl_scalar(&glsl, &HashMap::new());
        assert!(
            gpu.is_finite(),
            "Non-finite guard result on `{guarded}`: {gpu}"
        );
    }
}

const COMPLEX_PARITY_FORMULAS: &[&str] = &[
    "sin(z)",
    "cos(z)",
    "tan(z)",
    "exp(z)",
    "ln(z)",
    "log10(z)",
    "sqrt(z)",
    "abs(z)",
    "sinh(z)",
    "cosh(z)",
    "tanh(z)",
    "asin(z)",
    "acos(z)",
    "atan(z)",
    "min(z, 0.5)",
    "max(z, -0.5)",
    "clamp(z, -1, 0.5)",
    "step(z)",
    "rect(z)",
    "atan2(z, -0.5)",
    "z^3 - 1/z",
    "z + 1/z",
    "sin(z) + z^2",
    "z^i + pi * e / tau",
];

#[test]
fn test_complex_numerical_parity_across_all_20_funcs_and_branch_cuts() {
    let samples = [
        Complex64::new(0.65, -0.35),
        Complex64::new(-1.15, 0.75),
        Complex64::new(-2.0, 0.0),
    ];

    for &src in COMPLEX_PARITY_FORMULAS {
        let ast = parse(src).unwrap();
        let glsl = ast.to_glsl_complex_with_vars(&["z"]).unwrap();
        for z in samples {
            let cpu = ast.eval_complex_z(z);
            let gpu = eval_glsl_complex(&glsl, [z.re as f32, z.im as f32]);
            assert!(
                (cpu.re as f32 - gpu[0]).abs() < 2.0e-4,
                "Re mismatch `{src}` at {z}: cpu={}, gpu={}",
                cpu.re,
                gpu[0]
            );
            assert!(
                (cpu.im as f32 - gpu[1]).abs() < 2.0e-4,
                "Im mismatch `{src}` at {z}: cpu={}, gpu={}",
                cpu.im,
                gpu[1]
            );
        }
    }

    let zero_div = parse("1 / z")
        .unwrap()
        .to_glsl_complex_with_vars(&["z"])
        .unwrap();
    assert_eq!(eval_glsl_complex(&zero_div, [0.0, 0.0]), [0.0, 0.0]);
    let zero_zero = parse("z^z")
        .unwrap()
        .to_glsl_complex_with_vars(&["z"])
        .unwrap();
    assert_eq!(eval_glsl_complex(&zero_zero, [0.0, 0.0]), [1.0, 0.0]);
}
