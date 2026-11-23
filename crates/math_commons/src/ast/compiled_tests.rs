// [cite:math_commons]
use super::*;
use num_complex::Complex64;
use std::collections::HashMap;

#[test]
fn test_multi_arity_and_piecewise_functions_real_and_complex() {
    let vars = HashMap::from([("x".to_string(), 0.25), ("y".to_string(), -1.5)]);

    let expr_min = parse("min(x, y)").expect("parse min");
    let expr_max = parse("max(x, y)").expect("parse max");
    let expr_clamp = parse("clamp(y, -1, 1)").expect("parse clamp");
    let expr_step_neg = parse("step(-0.1)").expect("parse step neg");
    let expr_step_zero = parse("step(0)").expect("parse step zero");
    let expr_rect_in = parse("rect(0.5)").expect("parse rect inside");
    let expr_rect_out = parse("rect(0.6)").expect("parse rect outside");
    let expr_atan2 = parse("atan2(1, -1)").expect("parse atan2");

    assert!((expr_min.eval(&vars).unwrap() - -1.5).abs() < 1e-12);
    assert!((expr_max.eval(&vars).unwrap() - 0.25).abs() < 1e-12);
    assert!((expr_clamp.eval(&vars).unwrap() - -1.0).abs() < 1e-12);
    assert_eq!(expr_step_neg.eval(&vars).unwrap(), 0.0);
    assert_eq!(expr_step_zero.eval(&vars).unwrap(), 1.0);
    assert_eq!(expr_rect_in.eval(&vars).unwrap(), 1.0);
    assert_eq!(expr_rect_out.eval(&vars).unwrap(), 0.0);
    assert!((expr_atan2.eval(&vars).unwrap() - (3.0 * std::f64::consts::FRAC_PI_4)).abs() < 1e-12);

    // Inverted clamp bounds must return NaN without panicking
    let inverted_clamp = parse("clamp(0.2, 3.0, -1.0)").expect("parse inverted clamp");
    assert!(inverted_clamp.eval(&vars).unwrap().is_nan());
    assert!(
        CompiledExpr::compile(&inverted_clamp)
            .unwrap()
            .eval_slice(&[])
            .is_nan()
    );

    // Complex mode projections
    let c_clamp = CompiledExpr::compile_complex(&parse("clamp(z, -1, 1) + 2i").unwrap(), &["z"])
        .expect("compile complex clamp");
    let w = c_clamp.eval_complex_z(Complex64::new(3.5, 0.0));
    assert!((w - Complex64::new(1.0, 2.0)).norm() < 1e-12);
}

#[test]
fn test_exact_arity_checking_in_eval_and_compile() {
    let vars = HashMap::new();
    let bad_min = parse("min(1)").expect("parse 1-arg min");
    assert_eq!(
        bad_min.eval(&vars),
        Err(EvalError::WrongArgumentCount {
            func: Func::Min,
            expected: 2,
            found: 1,
        })
    );
    assert_eq!(
        CompiledExpr::compile(&bad_min),
        Err(EvalError::WrongArgumentCount {
            func: Func::Min,
            expected: 2,
            found: 1,
        })
    );

    let bad_clamp = parse("clamp(1, 2)").expect("parse 2-arg clamp");
    assert_eq!(
        CompiledExpr::compile(&bad_clamp),
        Err(EvalError::WrongArgumentCount {
            func: Func::Clamp,
            expected: 3,
            found: 2,
        })
    );
}

#[test]
fn test_unknown_function_rejected_with_one_indexed_position() {
    let err = parse("foo(x)").expect_err("unknown function must fail");
    assert_eq!(
        err,
        ParseError::UnknownFunction {
            name: "foo".into(),
            span: Span { start: 0, end: 3 },
        }
    );
    assert_eq!(err.user_message(), "At character 1: unknown function 'foo'");

    let err_offset = parse("1 + bar_baz(x, 2)").expect_err("offset unknown function must fail");
    assert_eq!(
        err_offset.user_message(),
        "At character 5: unknown function 'bar_baz'"
    );

    // Multi-arity function without parentheses must also fail cleanly
    let err_no_parens = parse("min 1").expect_err("min without parens must fail");
    assert!(err_no_parens.user_message().contains("character 5"));
}

#[test]
fn test_free_variables_and_latex_for_new_functions() {
    let expr = parse("a * clamp(sin(k * x + PI), -b, b) + Tau - E + step(x) + rect(x)").unwrap();
    assert_eq!(expr.free_variables(), vec!["a", "k", "x", "b"]);

    let latex = parse("min(a, max(b, clamp(x, -1, 1))) + atan2(y, x) + step(x) + rect(x)")
        .unwrap()
        .to_latex();
    assert!(latex.contains(r"\min\left("));
    assert!(latex.contains(r"\max\left("));
    assert!(latex.contains(r"\mathrm{clamp}\left("));
    assert!(latex.contains(r"\mathrm{atan2}\left("));
    assert!(latex.contains(r"\mathrm{step}\left("));
    assert!(latex.contains(r"\mathrm{rect}\left("));
}

#[test]
fn test_compiled_expr_1d_2d_3d_and_complex_parity() {
    let expr_1d = parse("a * sin(k * x) + clamp(x, -b, b)").unwrap();
    let compiled_1d = CompiledExpr::compile_with_vars(&expr_1d, &["x"]).unwrap();
    assert_eq!(compiled_1d.param_names(), &["a", "k", "b"]);

    let params = [1.5, 2.5, 0.4];
    for step in -10..=10 {
        let x = f64::from(step) * 0.15;
        let vars = HashMap::from([
            ("x".into(), x),
            ("a".into(), params[0]),
            ("k".into(), params[1]),
            ("b".into(), params[2]),
        ]);
        let expected = expr_1d.eval(&vars).unwrap();
        let actual = compiled_1d.eval_1d(x, &params);
        assert!((actual - expected).abs() < 1e-12);
    }

    let expr_2d = parse("a * (x^2 - y^2) + step(x) * rect(y)").unwrap();
    let compiled_2d = CompiledExpr::compile_with_vars(&expr_2d, &["x", "y"]).unwrap();
    assert_eq!(compiled_2d.param_names(), &["a"]);
    assert!((compiled_2d.eval_2d(0.5, 0.25, &[2.0]) - (2.0 * (0.25 - 0.0625) + 1.0)).abs() < 1e-12);

    let expr_3d = parse("w * sin(b * x + phi)").unwrap();
    let compiled_3d = CompiledExpr::compile_with_vars(&expr_3d, &["x", "w", "b"]).unwrap();
    assert_eq!(compiled_3d.param_names(), &["phi"]);
    let expected_3d = 1.8 * (0.7 * 0.5 + 0.3_f64).sin();
    assert!((compiled_3d.eval_3d(0.5, 1.8, 0.7, &[0.3]) - expected_3d).abs() < 1e-12);

    let expr_joukowsky = parse("z + a^2 / z").unwrap();
    let compiled_c = CompiledExpr::compile_complex(&expr_joukowsky, &["z"]).unwrap();
    assert_eq!(compiled_c.param_names(), &["a"]);
    let z = Complex64::new(1.2, 0.8);
    let w = compiled_c.eval_complex_z_params(z, &[0.9]);
    let expected_w = z + Complex64::new(0.81, 0.0) / z;
    assert!((w - expected_w).norm() < 1e-12);
    let pole = compiled_c.eval_complex_z_params(Complex64::new(0.0, 0.0), &[0.9]);
    assert!(pole.re.is_nan() && pole.im.is_nan());
}

#[test]
fn test_compiled_expr_deep_ast_respects_max_eval_depth() {
    let mut deep = Expr::Number(1.0);
    for _ in 0..100 {
        deep = Expr::unary(UnaryOp::Neg, deep);
    }
    assert_eq!(
        CompiledExpr::compile(&deep),
        Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH))
    );
}

#[test]
fn test_pipe_absolute_value_bars_nested_adjacent_and_errors() {
    let compiled_1d = CompiledExpr::compile_with_vars(&parse("||x| - 1|").unwrap(), &["x"])
        .expect("compile nested abs");
    assert!((compiled_1d.eval_1d(-2.5, &[]) - 1.5).abs() < 1e-12);
    assert!((compiled_1d.eval_1d(0.25, &[]) - 0.75).abs() < 1e-12);
    assert_eq!(
        parse("||x| - 1|").unwrap().to_latex(),
        r"\left|\left|x\right| - 1\right|"
    );

    let adj = CompiledExpr::compile_with_vars(
        &parse("2|y| + |x||y| + |(2|y|) - 1|").unwrap(),
        &["x", "y"],
    )
    .expect("compile adjacent and parenthesized abs");
    // x = -3, y = -2 => 2*2 + 3*2 + |2*2 - 1| = 4 + 6 + 3 = 13
    assert!((adj.eval_2d(-3.0, -2.0, &[]) - 13.0).abs() < 1e-12);

    let c_abs = CompiledExpr::compile_complex(&parse("|z| + i").unwrap(), &["z"]).unwrap();
    let w = c_abs.eval_complex_z(Complex64::new(3.0, -4.0));
    assert!((w - Complex64::new(5.0, 1.0)).norm() < 1e-12);

    let unclosed = parse("|y + 1").expect_err("unclosed pipe must fail");
    assert!(matches!(unclosed, ParseError::UnexpectedToken { .. }));
    assert!(unclosed.user_message().contains("'|'"));

    let deep_pipes = format!("{}x{}", "|".repeat(70), "|".repeat(70));
    assert_eq!(
        parse(&deep_pipes),
        Err(ParseError::RecursionLimitExceeded(MAX_PARSE_DEPTH))
    );
}

#[test]
fn test_unary_function_exponent_desmos_notation_and_depth_bounds() {
    let pythag = CompiledExpr::compile_with_vars(
        &parse("sin^2(t) + cos^2(t) + 2sin^3(t) - sin^3 t").unwrap(),
        &["t"],
    )
    .expect("compile trig powers");
    for step in -8..=8 {
        let t = f64::from(step) * 0.35;
        let expected = 1.0 + t.sin().powi(3);
        assert!((pythag.eval_1d(t, &[]) - expected).abs() < 1e-12);
    }

    let param_pow =
        CompiledExpr::compile_with_vars(&parse("-k * |y| * sin^2(t)").unwrap(), &["t", "y"])
            .expect("compile ode abs + sin^2");
    assert_eq!(param_pow.param_names(), &["k"]);
    let actual = param_pow.eval_2d(0.7, -2.0, &[1.5]);
    assert!((actual - (-1.5 * 2.0 * 0.7_f64.sin().powi(2))).abs() < 1e-12);

    assert!(matches!(
        parse("sin^2"),
        Err(ParseError::UnexpectedToken { .. })
    ));
    assert!(matches!(
        parse("min^2(x, y)"),
        Err(ParseError::UnexpectedToken { .. })
    ));

    let chain_62 = format!("{}1", "1 + ".repeat(MAX_PARSE_DEPTH - 2));
    let chain_63 = format!("{}1", "1 + ".repeat(MAX_PARSE_DEPTH - 1));
    assert!(parse(&format!("sin^2({chain_62})")).is_ok());
    assert_eq!(
        parse(&format!("sin^2({chain_63})")),
        Err(ParseError::RecursionLimitExceeded(MAX_PARSE_DEPTH))
    );
}
