use super::*;
use num_complex::Complex64;

#[test]
fn test_real_wgsl_and_glsl_operator_and_constant_lowering() {
    let expr = parse("x^3 - 1 / (x + pi) + log10(e * tau)").expect("parse real formula");
    let wgsl = expr.to_wgsl().expect("to_wgsl");
    let glsl = expr.to_glsl().expect("to_glsl");

    assert!(wgsl.contains("om_safe_pow(x, 3.0)"), "wgsl={wgsl}");
    assert!(
        wgsl.contains("om_safe_div(1.0, (x + 3.1415927))"),
        "wgsl={wgsl}"
    );
    assert!(
        wgsl.contains("om_log10((2.7182817 * 6.2831855))"),
        "wgsl={wgsl}"
    );
    assert_eq!(wgsl, glsl);

    let neg_pow = parse("-x^2 + (+x)").expect("parse unary");
    assert_eq!(
        neg_pow.to_wgsl().expect("wgsl unary"),
        "((-om_safe_pow(x, 2.0)) + x)"
    );
}

#[test]
fn test_complex_wgsl_and_glsl_lowering_and_all_14_functions() {
    let expr = parse("w = sin(z^3) - 1/z + i * pi").expect("parse complex equation");
    let wgsl = expr.to_wgsl_complex().expect("to_wgsl_complex");
    let glsl = expr.to_glsl_complex().expect("to_glsl_complex");

    assert!(
        wgsl.contains("c_sin(c_pow(z, vec2<f32>(3.0, 0.0)))"),
        "wgsl={wgsl}"
    );
    assert!(
        wgsl.contains("c_div(vec2<f32>(1.0, 0.0), z)"),
        "wgsl={wgsl}"
    );
    assert!(
        wgsl.contains("c_mul(vec2<f32>(0.0, 1.0), vec2<f32>(3.1415927, 0.0))"),
        "wgsl={wgsl}"
    );
    assert!(!wgsl.contains("vec2("));

    assert!(
        glsl.contains("c_sin(c_pow(z, vec2(3.0, 0.0)))"),
        "glsl={glsl}"
    );
    assert!(glsl.contains("c_div(vec2(1.0, 0.0), z)"), "glsl={glsl}");
    assert!(
        glsl.contains("c_mul(vec2(0.0, 1.0), vec2(3.1415927, 0.0))"),
        "glsl={glsl}"
    );
    assert!(!glsl.contains("vec2<f32>"));

    for (fn_name, expected_complex_fn, expected_real_prefix) in [
        ("sin", "c_sin(z)", "sin(x)"),
        ("cos", "c_cos(z)", "cos(x)"),
        ("tan", "c_tan(z)", "tan(x)"),
        ("exp", "c_exp(z)", "exp(clamp(x, -80.0, 80.0))"),
        ("ln", "c_ln(z)", "om_ln(x)"),
        ("log10", "c_log10(z)", "om_log10(x)"),
        ("sqrt", "c_sqrt(z)", "sqrt(max(x, 0.0))"),
        ("abs", "c_abs(z)", "abs(x)"),
        ("sinh", "c_sinh(z)", "sinh(clamp(x, -80.0, 80.0))"),
        ("cosh", "c_cosh(z)", "cosh(clamp(x, -80.0, 80.0))"),
        ("tanh", "c_tanh(z)", "tanh(x)"),
        ("asin", "c_asin(z)", "asin(clamp(x, -1.0, 1.0))"),
        ("acos", "c_acos(z)", "acos(clamp(x, -1.0, 1.0))"),
        ("atan", "c_atan(z)", "atan(x)"),
    ] {
        let r_ast = parse(&format!("{fn_name}(x)")).expect("parse real func");
        assert_eq!(r_ast.to_wgsl().unwrap(), expected_real_prefix);
        assert_eq!(r_ast.to_glsl().unwrap(), expected_real_prefix);

        let c_ast = parse(&format!("{fn_name}(Z)")).expect("parse complex func");
        assert_eq!(c_ast.to_wgsl_complex().unwrap(), expected_complex_fn);
        assert_eq!(c_ast.to_glsl_complex().unwrap(), expected_complex_fn);
    }
}

#[test]
fn test_complex_preludes_cover_all_helpers_and_target_intrinsics() {
    let wgsl_pre = wgsl_complex_prelude();
    let glsl_pre = glsl_complex_prelude();
    for helper in [
        "c_mul", "c_div", "c_pow", "c_sin", "c_cos", "c_tan", "c_exp", "c_ln", "c_log10", "c_sqrt",
        "c_abs", "c_sinh", "c_cosh", "c_tanh", "c_asin", "c_acos", "c_atan",
    ] {
        assert!(
            wgsl_pre.contains(&format!("fn {helper}(")),
            "missing {helper} in WGSL"
        );
        assert!(
            glsl_pre.contains(&format!("vec2 {helper}(")),
            "missing {helper} in GLSL"
        );
    }
    assert!(wgsl_pre.contains("atan2(") && !wgsl_pre.contains("? "));
    assert!(glsl_pre.contains("atan(z.y, z.x)") && !glsl_pre.contains("atan2("));

    let wgsl_r = wgsl_real_prelude();
    let glsl_r = glsl_real_prelude();
    for helper in [
        "om_safe_div",
        "om_safe_pow",
        "om_ln",
        "om_log10",
        "om_step",
        "om_rect",
        "om_clamp",
        "om_atan_yx",
    ] {
        assert!(wgsl_r.contains(helper) && glsl_r.contains(helper));
    }
}

#[test]
fn test_multi_arg_and_piecewise_func_shader_lowering() {
    for (src, real_expected, complex_expected) in [
        ("step(x)", "om_step(x)", "c_step(x)"),
        ("rect(x)", "om_rect(x)", "c_rect(x)"),
        ("min(x, y)", "min(x, y)", "c_min(x, y)"),
        ("max(x, y)", "max(x, y)", "c_max(x, y)"),
        ("atan2(y, x)", "om_atan_yx(y, x)", "c_atan_yx(y, x)"),
        (
            "clamp(x, -1, 1)",
            "om_clamp(x, (-1.0), 1.0)",
            "c_clamp(x, (-vec2<f32>(1.0, 0.0)), vec2<f32>(1.0, 0.0))",
        ),
    ] {
        let ast = parse(src).expect("parse multi/piecewise func");
        assert_eq!(ast.to_wgsl().unwrap(), real_expected);
        assert_eq!(ast.to_glsl().unwrap(), real_expected);
        assert_eq!(ast.to_wgsl_complex().unwrap(), complex_expected);
    }

    let bad_clamp = parse("clamp(x, 0)").expect("parse 2-arg clamp");
    assert_eq!(
        bad_clamp.to_wgsl(),
        Err(ShaderEmitError::WrongArgumentCount {
            func: Func::Clamp,
            expected: 3,
            found: 2,
        })
    );
}

#[test]
fn test_bounded_iterative_worklist_rejects_deep_ast_without_stack_overflow() {
    let mut deep = Expr::Number(1.0);
    for idx in 0..100_000 {
        deep = if idx % 2 == 0 {
            Expr::unary(UnaryOp::Neg, deep)
        } else {
            Expr::function(Func::Sin, vec![deep])
        };
    }
    for res in [
        deep.to_wgsl(),
        deep.to_glsl(),
        deep.to_wgsl_complex(),
        deep.to_glsl_complex(),
    ] {
        assert_eq!(
            res,
            Err(ShaderEmitError::RecursionLimitExceeded(MAX_EVAL_DEPTH))
        );
    }

    let exact_64 = parse(&format!("{}1", "1 + ".repeat(MAX_EVAL_DEPTH))).expect("depth 64");
    assert!(exact_64.to_wgsl().is_ok());
    assert!(exact_64.to_glsl_complex().is_ok());
}

#[test]
fn test_shader_emitter_rejects_bad_arity_nonfinite_and_injected_identifiers() {
    let zero_args = parse("sin()").expect("parse zero args");
    assert_eq!(
        zero_args.to_wgsl(),
        Err(ShaderEmitError::WrongArgumentCount {
            func: Func::Sin,
            expected: 1,
            found: 0,
        })
    );
    assert_eq!(
        Expr::Number(f64::NAN).to_glsl(),
        Err(ShaderEmitError::NonFiniteLiteral)
    );
    assert_eq!(
        Expr::Number(f64::INFINITY).to_wgsl_complex(),
        Err(ShaderEmitError::NonFiniteLiteral)
    );
    for bad_ident in [
        "",
        "z); discard; //",
        "return",
        "c_mul",
        "om_safe_div",
        "1abc",
    ] {
        assert!(matches!(
            Expr::Variable(bad_ident.to_string()).to_wgsl(),
            Err(ShaderEmitError::InvalidIdentifier(_))
        ));
    }
    assert_eq!(
        Expr::Variable("y'".to_string()).to_wgsl().unwrap(),
        "y_prime"
    );
}

#[test]
fn test_eval_complex_zc_matches_mandelbrot_and_custom_multibrot() {
    let expr = parse("z^3 + c").expect("parse z^3 + c");
    let z = Complex64::new(0.5, -0.5);
    let c = Complex64::new(-0.4, 0.6);
    let actual = expr.eval_complex_zc(z, c);
    let expected = z.powf(3.0) + c;
    assert!((actual - expected).norm() < 1e-12);
}
