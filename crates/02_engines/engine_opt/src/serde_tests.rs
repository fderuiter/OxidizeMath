// [cite:engine_opt]
#[test]
fn engine_opt_serde_json_dev_dep_linked() {
    let v: usize = serde_json::from_str("3").expect("json scalar");
    assert_eq!(v, 3);
}

#[cfg(feature = "serde")]
mod active {
    use crate::error::{FitError, LinalgError};
    use crate::fit::{CurveFit, FitModel, WeightedOlsAccum, fit_polynomial_scaled};

    fn roundtrip_json<T>(val: &T) -> T
    where
        T: serde::Serialize + serde::de::DeserializeOwned,
    {
        let encoded = serde_json::to_string(val).expect("serialize to JSON");
        serde_json::from_slice(encoded.as_bytes()).expect("deserialize from JSON bytes")
    }

    #[test]
    fn fit_models_and_curves_roundtrip_no_std() {
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
    }

    #[test]
    fn fit_errors_roundtrip_no_std() {
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
            FitError::Linalg(LinalgError::WorkspaceTooSmall {
                required: 16,
                available: 8,
            }),
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
}
