#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{Func, ParseError, Span, parse, parse_equation};
use std::collections::HashMap;

#[test]
fn test_parse_rejects_zero_and_multi_arg_calls_to_unary_functions() {
    let unary_funcs = [
        ("sin", Func::Sin),
        ("cos", Func::Cos),
        ("tan", Func::Tan),
        ("exp", Func::Exp),
        ("ln", Func::Ln),
        ("log10", Func::Log10),
        ("sqrt", Func::Sqrt),
        ("abs", Func::Abs),
        ("sinh", Func::Sinh),
        ("cosh", Func::Cosh),
        ("tanh", Func::Tanh),
        ("asin", Func::Asin),
        ("acos", Func::Acos),
        ("atan", Func::Atan),
        ("step", Func::Step),
        ("rect", Func::Rect),
    ];

    for (name, expected_func) in unary_funcs {
        let zero_call = format!("{name}()");
        let err_zero = parse(&zero_call).expect_err("0-arg unary call must fail in parse");
        assert_eq!(
            err_zero,
            ParseError::WrongArgumentCount {
                func: expected_func,
                expected: 1,
                found: 0,
                span: Span {
                    start: 0,
                    end: name.len()
                },
            }
        );
        let msg_zero = err_zero.user_message_for_input(&zero_call);
        assert!(msg_zero.contains("expected 1 argument, found 0"));
        assert!(!msg_zero.contains("Span") && !msg_zero.contains("<EOF>"));

        let multi_call = format!("{name}(y, t)");
        let err_multi = parse(&multi_call).expect_err("2-arg unary call must fail in parse");
        assert_eq!(
            err_multi,
            ParseError::WrongArgumentCount {
                func: expected_func,
                expected: 1,
                found: 2,
                span: Span {
                    start: 0,
                    end: name.len()
                },
            }
        );
        let msg_multi = err_multi.user_message_for_input(&multi_call);
        assert!(msg_multi.contains("expected 1 argument, found 2"));
    }
}

#[test]
fn test_parse_rejects_wrong_arity_for_multi_arg_and_power_functions() {
    assert_eq!(
        parse_equation("y' = atan(y, t)").expect_err("atan(y, t) must fail"),
        ParseError::WrongArgumentCount {
            func: Func::Atan,
            expected: 1,
            found: 2,
            span: Span { start: 5, end: 9 },
        }
    );

    assert_eq!(
        parse("min(1)").expect_err("1-arg min must fail"),
        ParseError::WrongArgumentCount {
            func: Func::Min,
            expected: 2,
            found: 1,
            span: Span { start: 0, end: 3 },
        }
    );

    assert_eq!(
        parse("atan2(y)").expect_err("1-arg atan2 must fail"),
        ParseError::WrongArgumentCount {
            func: Func::Atan2,
            expected: 2,
            found: 1,
            span: Span { start: 0, end: 5 },
        }
    );

    assert_eq!(
        parse("clamp(x, 0)").expect_err("2-arg clamp must fail"),
        ParseError::WrongArgumentCount {
            func: Func::Clamp,
            expected: 3,
            found: 2,
            span: Span { start: 0, end: 5 },
        }
    );

    assert_eq!(
        parse("sin^2()").expect_err("0-arg sin^2() must fail"),
        ParseError::WrongArgumentCount {
            func: Func::Sin,
            expected: 1,
            found: 0,
            span: Span { start: 0, end: 3 },
        }
    );

    let err_conj = parse("conj(z)").expect_err("unknown function conj(z) must fail");
    assert!(matches!(err_conj, ParseError::UnknownFunction { .. }));
}

#[test]
fn test_implicit_multiplication_and_valid_calls_preserved() {
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    vars.insert("t".to_string(), 0.5);

    let expr = parse("2x + 3sin(t) + atan2(x, 1) + clamp(2x, 0, 5)")
        .expect("implicit multiplication and multi-arg functions must parse");
    let val = expr.eval(&vars).expect("evaluate valid expression");
    let expected = 6.0 + 3.0 * 0.5_f64.sin() + 3.0_f64.atan2(1.0) + 5.0;
    assert!((val - expected).abs() < 1e-12);
}
