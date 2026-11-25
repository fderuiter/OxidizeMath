// [cite:math_commons]
#[test]
fn math_commons_serde_json_dev_dep_linked() {
    let v: usize = serde_json::from_str("3").expect("json scalar");
    assert_eq!(v, 3);
}

#[test]
fn bounded_floats_implement_try_from_f64() {
    use crate::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};

    assert_eq!(UnitInterval::try_from(0.25).unwrap().value(), 0.25);
    assert!(UnitInterval::try_from(-0.1).is_err());
    assert!(UnitInterval::try_from(f64::NAN).is_err());

    assert_eq!(PositiveFloat::try_from(2.5).unwrap().value(), 2.5);
    assert!(PositiveFloat::try_from(0.0).is_err());
    assert!(PositiveFloat::try_from(f64::INFINITY).is_err());

    assert_eq!(NonNegativeFloat::try_from(0.0).unwrap().value(), 0.0);
    assert!(NonNegativeFloat::try_from(-1e-9).is_err());
    assert!(NonNegativeFloat::try_from(f64::NEG_INFINITY).is_err());
}

#[cfg(feature = "serde")]
mod active {
    use crate::error::{FitError, MathError};
    use crate::fit::{CurveFit, FitModel, WeightedOlsAccum, fit_polynomial_scaled};
    use crate::math_kernel::types::{Dimension, GridIndex, StepSize};
    use crate::poi::{LinePoiScan, LocalExtremum};
    use crate::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};
    use crate::stochastic::{OuCalibration, SampleSummary, SccSummary};
    use crate::wigner::ClebschGordan;
    use alloc::string::ToString;
    use alloc::vec;
    use serde::Deserialize;
    use serde::de::value::{Error as ValueDeError, F64Deserializer};

    fn roundtrip_json<T>(val: &T) -> T
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let encoded = serde_json::to_string(val).expect("serialize to JSON");
        serde_json::from_slice(encoded.as_bytes()).expect("deserialize from JSON bytes")
    }

    #[test]
    fn bounded_floats_roundtrip_and_reject_invalid_loads() {
        for raw in [0.0, 0.5, 1.0] {
            let u = UnitInterval::new(raw).unwrap();
            assert_eq!(roundtrip_json(&u), u);
        }
        assert_eq!(
            roundtrip_json(&PositiveFloat::new(1.75).unwrap()),
            PositiveFloat::new(1.75).unwrap()
        );
        for raw in [0.0, 42.25] {
            let nn = NonNegativeFloat::new(raw).unwrap();
            assert_eq!(roundtrip_json(&nn), nn);
        }

        for bad in ["-0.0001", "1.0001"] {
            assert!(serde_json::from_str::<UnitInterval>(bad).is_err());
        }
        for bad in ["-2.0", "0.0"] {
            assert!(serde_json::from_str::<PositiveFloat>(bad).is_err());
        }
        assert!(serde_json::from_str::<NonNegativeFloat>("-1e-12").is_err());

        for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err_u = UnitInterval::deserialize(F64Deserializer::<ValueDeError>::new(non_finite))
                .expect_err("UnitInterval must reject non-finite");
            assert!(err_u.to_string().contains("outside"));

            let err_p =
                PositiveFloat::deserialize(F64Deserializer::<ValueDeError>::new(non_finite))
                    .expect_err("PositiveFloat must reject non-finite");
            assert!(err_p.to_string().contains("outside"));

            let err_nn =
                NonNegativeFloat::deserialize(F64Deserializer::<ValueDeError>::new(non_finite))
                    .expect_err("NonNegativeFloat must reject non-finite");
            assert!(err_nn.to_string().contains("outside"));
        }
    }

    #[test]
    fn kernel_fit_poi_stochastic_and_wigner_roundtrip_no_std() {
        assert_eq!(roundtrip_json(&StepSize(0.0125)), StepSize(0.0125));
        assert_eq!(roundtrip_json(&GridIndex(17)), GridIndex(17));
        assert_eq!(roundtrip_json(&Dimension(256)), Dimension(256));

        for model in FitModel::MENU_PRESETS {
            assert_eq!(roundtrip_json(&model), model);
        }
        let pts = [[1000.0, 2.0], [1001.0, 5.0], [1002.0, 10.0], [1003.0, 17.0]];
        let curve = fit_polynomial_scaled(&pts, 2, false, false).expect("quadratic fit");
        let decoded_curve = roundtrip_json(&curve);
        assert_eq!(decoded_curve, curve);
        assert!((decoded_curve.eval(1001.5) - curve.eval(1001.5)).abs() < 1e-12);

        let mut bad_json = serde_json::to_string(&curve).unwrap();
        bad_json = bad_json.replace("\"coeff_len\":3", "\"coeff_len\":9");
        assert!(serde_json::from_str::<CurveFit>(&bad_json).is_err());

        let mut accum = WeightedOlsAccum::default();
        accum.push(1.0, 2.0, 0.5);
        accum.push(2.0, 4.0, 1.5);
        assert_eq!(roundtrip_json(&accum), accum);

        let ext = LocalExtremum {
            point: [1.25, -3.5],
            is_max: false,
        };
        assert_eq!(roundtrip_json(&ext), ext);
        let scan = LinePoiScan {
            roots: vec![-1.0, 1.0],
            y_intercepts: vec![-1.0],
            extrema: vec![ext],
        };
        assert_eq!(roundtrip_json(&scan), scan);

        let scc = SccSummary {
            num_components: 3,
            num_sink_components: 1,
            unique_sink_size: 4,
        };
        assert_eq!(roundtrip_json(&scc), scc);
        let ou = OuCalibration {
            mu: 0.5,
            theta: 1.2,
            sigma: 0.3,
        };
        assert_eq!(roundtrip_json(&ou), ou);
        let summary = SampleSummary {
            mean: 1.0,
            std_dev: 0.2,
            median: 0.98,
            percentile_05: 0.65,
            percentile_95: 1.35,
            prob_above_threshold: 0.45,
        };
        assert_eq!(roundtrip_json(&summary), summary);

        let cg = ClebschGordan {
            tj1: 3,
            tm1: -1,
            tj2: 2,
            tm2: 2,
            tj12: 5,
            tm12: 1,
        };
        assert_eq!(roundtrip_json(&cg), cg);
    }

    #[test]
    fn math_and_fit_errors_roundtrip_no_std() {
        let math_errors = [
            MathError::DimensionMismatch {
                expected: Dimension(3),
                actual: Dimension(4),
            },
            MathError::OutOfBounds {
                value: 1.5,
                min: 0.0,
                max: 1.0,
            },
            MathError::StrictlyOutOfBounds {
                value: 0.0,
                min: 0.0,
                max: f64::MAX,
            },
            MathError::InvalidPermutation {
                index: 1,
                value: 5,
                len: 3,
            },
            MathError::NumericalError {
                reason: "singular".to_string(),
            },
            MathError::ConversionError {
                reason: "overflow".to_string(),
            },
        ];
        for err in math_errors {
            assert_eq!(roundtrip_json(&err), err);
        }

        let fit_errors = [
            FitError::InsufficientPoints {
                required: 3,
                actual: 1,
            },
            FitError::UnsupportedDegree { degree: 7 },
            FitError::InvalidDomain {
                reason: "exponential fit requires at least 2 points with y > 0",
            },
            FitError::SingularMatrix,
            FitError::NonConvergence { iterations: 200 },
        ];
        for err in fit_errors {
            assert_eq!(roundtrip_json(&err), err);
        }

        let unknown_domain: FitError =
            serde_json::from_str(r#"{"InvalidDomain":{"reason":"unrecognized custom reason"}}"#)
                .expect("unknown domain reason falls back to static message");
        assert_eq!(
            unknown_domain,
            FitError::InvalidDomain {
                reason: "invalid curve-fitting domain"
            }
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn ast_and_umbrella_errors_roundtrip_std() {
        use crate::ast::{
            BinaryOp, CompiledExpr, EvalError, Expr, Func, GlslError, Instruction, LexError,
            ParseError, ShaderEmitError, ShaderTarget, Span, Token, TokenKind, UnaryOp, parse,
        };
        use crate::error::MathCommonsError;
        use crate::math_kernel::colormap::ShaderPalette;

        for palette in ShaderPalette::ALL {
            assert_eq!(roundtrip_json(&palette), palette);
        }

        let expr: Expr = parse("sin(2 * x) + a").expect("parse expr");
        assert_eq!(roundtrip_json(&expr), expr);

        let compiled = CompiledExpr::compile_with_vars(&expr, &["x"]).expect("compile expr");
        let decoded_compiled = roundtrip_json(&compiled);
        assert_eq!(decoded_compiled, compiled);
        assert_eq!(decoded_compiled.param_names(), &["a".to_string()]);
        assert!(
            (decoded_compiled.eval_1d(0.5, &[1.25]) - compiled.eval_1d(0.5, &[1.25])).abs() < 1e-12
        );

        let bad_axis = r#"{"instructions":[{"PushConst":1.0}],"var_names":[],"axis_count":2}"#;
        assert!(serde_json::from_str::<CompiledExpr>(bad_axis).is_err());

        let bad_slot = r#"{"instructions":[{"LoadVar":3}],"var_names":["x"],"axis_count":1}"#;
        assert!(serde_json::from_str::<CompiledExpr>(bad_slot).is_err());

        let bad_underflow = r#"{"instructions":[{"Binary":"Add"}],"var_names":[],"axis_count":0}"#;
        assert!(serde_json::from_str::<CompiledExpr>(bad_underflow).is_err());

        let bad_unbalanced = r#"{"instructions":[{"PushConst":1.0},{"PushConst":2.0}],"var_names":[],"axis_count":0}"#;
        assert!(serde_json::from_str::<CompiledExpr>(bad_unbalanced).is_err());

        assert_eq!(
            roundtrip_json(&Instruction::Binary(BinaryOp::Add)),
            Instruction::Binary(BinaryOp::Add)
        );
        assert_eq!(
            roundtrip_json(&Instruction::Unary(UnaryOp::Neg)),
            Instruction::Unary(UnaryOp::Neg)
        );
        assert_eq!(
            roundtrip_json(&Instruction::Call(Func::Sin)),
            Instruction::Call(Func::Sin)
        );
        assert_eq!(roundtrip_json(&ShaderTarget::Wgsl), ShaderTarget::Wgsl);

        let token = Token {
            kind: TokenKind::Ident("alpha".to_string()),
            span: Span::new(0, 5),
        };
        assert_eq!(roundtrip_json(&token), token);

        let umbrella_cases = [
            MathCommonsError::from(LexError {
                start: 0,
                end: 1,
                message: "unexpected".to_string(),
            }),
            MathCommonsError::from(ParseError::EmptyExpression),
            MathCommonsError::from(EvalError::UnknownVariable("z".to_string())),
            MathCommonsError::from(GlslError::DepthLimitExceeded(64)),
            MathCommonsError::from(ShaderEmitError::RecursionLimitExceeded(64)),
        ];
        for err in umbrella_cases {
            assert_eq!(roundtrip_json(&err), err);
        }
    }
}
