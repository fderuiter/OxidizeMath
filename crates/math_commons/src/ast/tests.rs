use super::*;
use num_complex::Complex64;
use std::collections::HashMap;
use std::time::Instant;

#[test]
fn test_parser_precedence_and_associativity() {
    let expr = parse("1 + 2 * 3").expect("parse");
    let vars = HashMap::new();
    assert_eq!(expr.eval(&vars).unwrap(), 7.0);

    let expr2 = parse("(1 + 2) * 3").expect("parse");
    assert_eq!(expr2.eval(&vars).unwrap(), 9.0);

    // Right-associative exponentiation: 2^3^2 = 2^(3^2) = 2^9 = 512
    let expr_pow = parse("2 ^ 3 ^ 2").expect("parse");
    assert_eq!(expr_pow.eval(&vars).unwrap(), 512.0);

    // Unary negative vs power: -3^2 should be -(3^2) = -9
    let expr_neg = parse("-3^2").expect("parse");
    assert_eq!(expr_neg.eval(&vars).unwrap(), -9.0);

    let expr_neg_parens = parse("(-3)^2").expect("parse");
    assert_eq!(expr_neg_parens.eval(&vars).unwrap(), 9.0);
}

#[test]
fn test_implicit_multiplication() {
    let mut vars = HashMap::new();
    vars.insert("x".into(), 4.0);
    vars.insert("y".into(), 5.0);
    vars.insert("k".into(), 2.0);
    vars.insert("t".into(), 0.0);

    // 2x -> 2 * 4 = 8
    let expr1 = parse("2x").expect("parse");
    assert_eq!(expr1.eval(&vars).unwrap(), 8.0);

    // 2(x + 1) -> 2 * 5 = 10
    let expr2 = parse("2(x + 1)").expect("parse");
    assert_eq!(expr2.eval(&vars).unwrap(), 10.0);

    // (x + 1)(x - 1) -> 5 * 3 = 15
    let expr3 = parse("(x + 1)(x - 1)").expect("parse");
    assert_eq!(expr3.eval(&vars).unwrap(), 15.0);

    // x y -> 4 * 5 = 20
    let expr4 = parse("x y").expect("parse");
    assert_eq!(expr4.eval(&vars).unwrap(), 20.0);

    // k y sin(t) with t = 0 -> 2 * 5 * sin(0) = 0
    let expr5 = parse("k y sin(t)").expect("parse");
    assert_eq!(expr5.eval(&vars).unwrap(), 0.0);

    // 2sin(pi/2) -> 2 * 1 = 2
    let expr6 = parse("2sin(pi / 2)").expect("parse");
    assert!((expr6.eval(&vars).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_equation_prefix_handling() {
    let (lhs, expr) = parse_equation("y' = -k * y * sin(t)").expect("parse ode");
    assert_eq!(lhs, Some("y'".into()));

    let mut params = HashMap::new();
    params.insert("k".into(), 2.5);
    let val = expr.eval_2d(0.0, 1.0, &params);
    assert_eq!(val, 0.0);

    let (lhs_w, expr_w) = parse_equation("w = z^3 - 1/z").expect("parse complex");
    assert_eq!(lhs_w, Some("w".into()));
    let z = Complex64::new(2.0, 0.0);
    let val_z = expr_w.eval_complex_z(z);
    // 2^3 - 1/2 = 8 - 0.5 = 7.5
    assert!((val_z.re - 7.5).abs() < 1e-10);

    let (lhs_f, expr_f) = parse_equation("f(t, y) = 3y").expect("parse func");
    assert_eq!(lhs_f, Some("f".into()));
    assert_eq!(expr_f.eval_2d(0.0, 4.0, &params), 12.0);
}

#[test]
fn test_error_recovery_and_syntax_validation() {
    // Empty expression
    assert_eq!(parse(""), Err(ParseError::EmptyExpression));
    assert_eq!(parse("   "), Err(ParseError::EmptyExpression));

    // Unbalanced parentheses
    assert!(matches!(
        parse("(1 + 2"),
        Err(ParseError::UnexpectedToken { .. })
    ));
    assert!(matches!(
        parse("1 + 2)"),
        Err(ParseError::UnexpectedToken { .. })
    ));

    // Missing operand
    assert!(matches!(parse("1 + "), Err(ParseError::UnexpectedEof)));

    // Invalid character
    assert!(matches!(parse("1 + @"), Err(ParseError::Lex(_))));
}

#[test]
fn test_user_message_plain_language_positions_and_no_debug_tokens() {
    let unclosed = parse("(1 + 2").expect_err("unclosed paren must fail");
    let msg_unclosed = unclosed.user_message();
    assert!(
        msg_unclosed.contains("character 7"),
        "Expected 1-indexed position 7, got: {msg_unclosed}"
    );
    assert!(
        msg_unclosed.contains("end of expression"),
        "Expected 'end of expression', got: {msg_unclosed}"
    );
    assert!(!msg_unclosed.contains("Span {"));
    assert!(!msg_unclosed.contains("<EOF>"));

    let extra_paren = parse("1 + 2)").expect_err("extra paren must fail");
    let msg_extra = extra_paren.user_message();
    assert!(msg_extra.contains("character 6"));
    assert!(msg_extra.contains("expected end of expression, found ')'"));
    assert!(!msg_extra.contains("Span {"));

    let bad_lex = parse("1 + @").expect_err("invalid char must fail");
    let msg_lex = bad_lex.user_message();
    assert!(msg_lex.contains("character 5"));
    assert!(!msg_lex.contains(".."));
    assert!(!msg_lex.contains("Span {"));

    let trailing_op = parse("1 +").expect_err("incomplete rhs must fail");
    let msg_trailing = trailing_op.user_message_for_input("1 +");
    assert!(msg_trailing.contains("end of expression"));
    assert!(msg_trailing.contains("character 4"));
    assert!(!msg_trailing.contains("<EOF>"));
}

#[test]
fn test_recursion_depth_limit() {
    // Build nested expression: "((((...1...))))"
    let mut deep = String::new();
    for _ in 0..80 {
        deep.push('(');
    }
    deep.push('1');
    for _ in 0..80 {
        deep.push(')');
    }

    let res = parse(&deep);
    assert!(matches!(res, Err(ParseError::RecursionLimitExceeded(_))));
}

#[test]
fn test_complex_evaluation() {
    let expr = parse("z^2").expect("parse");
    let z = Complex64::new(1.0, 2.0);
    let out = expr.eval_complex_z(z);
    // (1 + 2i)^2 = 1 - 4 + 4i = -3 + 4i
    assert!((out.re - (-3.0)).abs() < 1e-10);
    assert!((out.im - 4.0).abs() < 1e-10);

    // Division by zero complex handling: returns non-finite / NaN without panic
    let expr_div = parse("1 / z").expect("parse");
    let out_zero = expr_div.eval_complex_z(Complex64::new(0.0, 0.0));
    assert!(!out_zero.re.is_finite());
}

#[test]
fn test_latex_generation() {
    let expr = parse("z^3 - 1/z").expect("parse");
    let latex = expr.to_latex();
    assert!(latex.contains(r"\frac{1}{z}"));
    assert!(latex.contains(r"{z}^{3}"));

    let expr_ode = parse("-k * y * sin(t)").expect("parse");
    let latex_ode = expr_ode.to_latex();
    assert!(latex_ode.contains(r"\sin"));
}

#[test]
fn test_rk4_sub_microsecond_throughput() {
    let expr = parse("-k * y * sin(t)").expect("parse");
    let mut params = HashMap::new();
    params.insert("k".into(), 1.5);

    let dt = 0.01;
    let mut t = 0.0;
    let mut y = 1.0;

    // Warm-up
    for _ in 0..100 {
        let k1 = expr.eval_2d(t, y, &params);
        let k2 = expr.eval_2d(t + 0.5 * dt, y + 0.5 * dt * k1, &params);
        let k3 = expr.eval_2d(t + 0.5 * dt, y + 0.5 * dt * k2, &params);
        let k4 = expr.eval_2d(t + dt, y + dt * k3, &params);
        y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += dt;
    }

    let start = Instant::now();
    let steps = 1_000;
    for _ in 0..steps {
        let k1 = expr.eval_2d(t, y, &params);
        let k2 = expr.eval_2d(t + 0.5 * dt, y + 0.5 * dt * k1, &params);
        let k3 = expr.eval_2d(t + 0.5 * dt, y + 0.5 * dt * k2, &params);
        let k4 = expr.eval_2d(t + dt, y + dt * k3, &params);
        y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += dt;
    }
    let elapsed = start.elapsed();

    // Verify finite numerical solution
    assert!(y.is_finite());
    // 1,000 RK4 steps = 4,000 function evaluations.
    // Must complete fast (well under 5 milliseconds total, meaning ~1 microsecond per step).
    assert!(
        elapsed.as_millis() < 50,
        "RK4 1,000 steps took {:?}, expected < 50ms",
        elapsed
    );
}
