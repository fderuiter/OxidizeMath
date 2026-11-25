// [cite:math_commons]
use super::*;

fn assert_stack_error_traits<
    E: Copy + Clone + PartialEq + Eq + core::fmt::Display + core::error::Error + Diagnostic,
>() {
}

#[test]
fn math_error_is_copy_eq_and_diagnostic_without_alloc() {
    assert_stack_error_traits::<MathError>();
    assert_stack_error_traits::<QmcError>();
    assert_stack_error_traits::<RngError>();
    let ws = MathError::WorkspaceTooSmall {
        required: 16,
        available: 8,
    };
    let ws_copy = ws;
    assert_eq!(ws, ws_copy);
    assert_eq!(ws.code(), "MATH-005");
    assert!(core::error::Error::source(&ws).is_none());

    let qmc_umb = MathCommonsError::from(QmcError::InvalidBase { base: 1 });
    assert!(core::error::Error::source(&qmc_umb).is_some());
    assert_eq!(qmc_umb.severity(), Severity::Error);

    let rng_umb = MathCommonsError::from(RngError::ZeroUpperBound);
    assert!(core::error::Error::source(&rng_umb).is_some());
    assert_eq!(rng_umb.severity(), Severity::Error);

    let nan_bounds = MathError::OutOfBounds {
        value: f64::NAN,
        min: -0.0,
        max: 1.0,
    };
    let pos_zero_bounds = MathError::OutOfBounds {
        value: f64::NAN,
        min: 0.0,
        max: 1.0,
    };
    assert_eq!(nan_bounds, pos_zero_bounds);
    assert_eq!(
        MathError::from("overflow"),
        MathError::ConversionError { reason: "overflow" }
    );
}

#[test]
fn interval_enclosure_errors_are_copy_eq_in_bare_no_std() {
    use crate::primitives::Interval;

    assert_eq!(
        Interval::from_approx(f64::NAN),
        Err(MathError::NumericalError {
            reason: "non-finite scalar cannot form an interval enclosure",
        })
    );
    let one = Interval::point(1.0).expect("finite point interval");
    let straddles_zero = Interval::new(-1.0, 1.0).expect("valid interval");
    assert_eq!(
        one.checked_div(straddles_zero),
        Err(MathError::NumericalError {
            reason: "interval division by zero-containing enclosure",
        })
    );
}

#[cfg(feature = "alloc")]
mod nostd_alloc {
    use super::*;
    use alloc::string::ToString;

    struct FieldCounter(usize);

    impl DiagnosticVisitor for FieldCounter {
        fn record_str(&mut self, _key: &str, _value: &str) {
            self.0 += 1;
        }

        fn record_display(&mut self, _key: &str, _value: &dyn core::fmt::Display) {
            self.0 += 1;
        }
    }

    #[test]
    fn ast_parse_compile_eval_and_all_umbrella_variants_work_without_std() {
        let expr = crate::ast::parse("atan2(y, x) + 2 * a").expect("parse in no_std");
        let compiled =
            crate::ast::CompiledExpr::compile_with_vars(&expr, &["x", "y"]).expect("compile");
        let val = compiled.eval_2d(1.0, 1.0, &[1.5]);
        assert!((val - (core::f64::consts::FRAC_PI_4 + 3.0)).abs() < 1e-12);

        let variants = [
            MathCommonsError::from(MathError::DimensionMismatch {
                expected: Dimension(2),
                actual: Dimension(3),
            }),
            MathCommonsError::from(FitError::SingularMatrix),
            MathCommonsError::from(LinalgError::SingularMatrix),
            MathCommonsError::from(QmcError::InvalidDimension {
                dimension: 0,
                max_supported: 64,
            }),
            MathCommonsError::from(RngError::ZeroUpperBound),
            MathCommonsError::from(LexError {
                start: 0,
                end: 1,
                message: "bad".to_string(),
            }),
            MathCommonsError::from(ParseError::EmptyExpression),
            MathCommonsError::from(EvalError::UnknownVariable("x".to_string())),
            MathCommonsError::from(GlslError::DepthLimitExceeded(64)),
            MathCommonsError::from(ShaderEmitError::RecursionLimitExceeded(64)),
        ];

        for err in &variants {
            assert!(core::error::Error::source(err).is_some());
            assert_eq!(err.severity(), Severity::Error);
            let mut counter = FieldCounter(0);
            err.visit_fields(&mut counter);
            assert!(counter.0 >= 3);
        }
    }
}

#[cfg(feature = "std")]
mod std_tests {
    use super::*;
    use std::error::Error as StdError;
    use verified_engine::engine::VerifiedEngine;

    struct NullVisitor(usize);

    impl DiagnosticVisitor for NullVisitor {
        fn record_str(&mut self, _key: &str, _value: &str) {
            self.0 += 1;
        }

        fn record_display(&mut self, _key: &str, _value: &dyn core::fmt::Display) {
            self.0 += 1;
        }
    }

    #[test]
    fn zero_allocation_diagnostic_visitation_for_linalg_math_qmc_rng_and_fit() {
        let linalg_err = LinalgError::WorkspaceTooSmall {
            required: 32,
            available: 16,
        };
        let math_err = MathError::DimensionMismatch {
            expected: Dimension(3),
            actual: Dimension(2),
        };
        let qmc_err = QmcError::InvalidDimension {
            dimension: 0,
            max_supported: 64,
        };
        let rng_err = RngError::ZeroUpperBound;
        let fit_err = FitError::from(linalg_err);
        assert!(fit_err.source().is_some());

        let (count, _metrics) = VerifiedEngine::run_verified(|| {
            let mut v = NullVisitor(0);
            linalg_err.visit_fields(&mut v);
            math_err.visit_fields(&mut v);
            qmc_err.visit_fields(&mut v);
            rng_err.visit_fields(&mut v);
            fit_err.visit_fields(&mut v);
            v.0
        });
        assert!(count >= 20);
        assert_eq!(math_err.code(), "MATH-001");
        assert_eq!(
            qmc_err.metadata().get("code").map(String::as_str),
            Some("RNG-QMC-001")
        );
    }

    #[test]
    fn math_commons_umbrella_preserves_sources_and_diagnostics() {
        let lex = LexError {
            start: 0,
            end: 1,
            message: "bad char".to_string(),
        };
        let parse = ParseError::from(lex.clone());
        assert!(parse.source().is_some());

        let unknown_fn = ParseError::UnknownFunction {
            name: "foo".to_string(),
            span: crate::ast::Span { start: 3, end: 6 },
        };
        let unknown_meta = unknown_fn.metadata();
        assert_eq!(
            unknown_meta.get("function").map(String::as_str),
            Some("foo")
        );
        assert_eq!(unknown_meta.get("start").map(String::as_str), Some("3"));

        let wrong_arity = ParseError::WrongArgumentCount {
            func: crate::ast::Func::Atan2,
            expected: 2,
            found: 1,
            span: crate::ast::Span { start: 4, end: 12 },
        };
        let arity_meta = wrong_arity.metadata();
        assert_eq!(
            arity_meta.get("function").map(String::as_str),
            Some("atan2")
        );
        assert_eq!(arity_meta.get("expected").map(String::as_str), Some("2"));
        assert_eq!(arity_meta.get("found").map(String::as_str), Some("1"));
        assert_eq!(arity_meta.get("start").map(String::as_str), Some("4"));

        let variants = [
            MathCommonsError::from(MathError::DimensionMismatch {
                expected: Dimension(2),
                actual: Dimension(3),
            }),
            MathCommonsError::from(FitError::SingularMatrix),
            MathCommonsError::from(LinalgError::SingularMatrix),
            MathCommonsError::from(QmcError::InvalidBase { base: 1 }),
            MathCommonsError::from(RngError::ZeroUpperBound),
            MathCommonsError::from(lex),
            MathCommonsError::from(parse),
            MathCommonsError::from(unknown_fn),
            MathCommonsError::from(wrong_arity),
            MathCommonsError::from(EvalError::UnknownVariable("x".to_string())),
            MathCommonsError::from(GlslError::DepthLimitExceeded(64)),
            MathCommonsError::from(ShaderEmitError::RecursionLimitExceeded(64)),
        ];

        for err in &variants {
            assert!(err.source().is_some());
            assert_eq!(err.severity(), Severity::Error);
            let meta = err.metadata();
            assert_eq!(
                meta.get("umbrella_type").map(String::as_str),
                Some("MathCommonsError")
            );
            assert!(meta.contains_key("error_type"));
        }
    }
}
