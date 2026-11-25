// [cite:engine_linalg]
#[test]
fn linalg_serde_json_dev_dep_linked() {
    let parsed: u32 = serde_json::from_str("7").expect("json scalar");
    assert_eq!(parsed, 7);
}

#[cfg(feature = "serde")]
#[test]
fn linalg_value_and_error_types_roundtrip_no_std() {
    use crate::{DampingStyle, KrylovReport, LinalgError};

    let errors = [
        LinalgError::DimensionMismatch,
        LinalgError::WorkspaceTooSmall {
            required: 64,
            available: 16,
        },
        LinalgError::InvalidInput,
        LinalgError::SingularMatrix,
        LinalgError::NotPositiveDefinite,
        LinalgError::NonConvergence,
    ];
    for err in errors {
        let encoded = serde_json::to_string(&err).expect("serialize LinalgError");
        let decoded: LinalgError = serde_json::from_str(&encoded).expect("deserialize LinalgError");
        assert_eq!(decoded, err);
    }

    let report64 = KrylovReport::<f64> {
        iterations: 19,
        residual_norm: 1.25e-11,
    };
    let encoded_r64 = serde_json::to_string(&report64).expect("serialize KrylovReport<f64>");
    let decoded_r64: KrylovReport<f64> =
        serde_json::from_str(&encoded_r64).expect("deserialize KrylovReport<f64>");
    assert_eq!(decoded_r64, report64);

    let report32 = KrylovReport::<f32> {
        iterations: 7,
        residual_norm: 2.5e-5,
    };
    let encoded_r32 = serde_json::to_string(&report32).expect("serialize KrylovReport<f32>");
    let decoded_r32: KrylovReport<f32> =
        serde_json::from_str(&encoded_r32).expect("deserialize KrylovReport<f32>");
    assert_eq!(decoded_r32, report32);

    let dampings = [
        DampingStyle::Isotropic(1e-4_f64),
        DampingStyle::Marquardt {
            lambda: 0.25,
            diag_floor: 1e-8,
        },
    ];
    for damping in dampings {
        let encoded = serde_json::to_string(&damping).expect("serialize DampingStyle");
        let decoded: DampingStyle<f64> =
            serde_json::from_str(&encoded).expect("deserialize DampingStyle");
        assert_eq!(decoded, damping);
    }

    let mut qr = crate::StreamingQr::<f64, 3>::new(2).expect("init StreamingQr");
    qr.push_row(&[1.0, 0.0], 2.0).unwrap();
    qr.push_row(&[1.0, 1.0], 5.0).unwrap();
    qr.push_row(&[1.0, 2.0], 8.0).unwrap();
    let qr_json = serde_json::to_string(&qr).expect("serialize StreamingQr");
    let qr_decoded: crate::StreamingQr<f64, 3> =
        serde_json::from_str(&qr_json).expect("deserialize StreamingQr");
    assert_eq!(qr_decoded, qr);
    assert_eq!(qr_decoded.solve(1e-12), qr.solve(1e-12));

    let bad_ncols = qr_json.replace("\"ncols\":2", "\"ncols\":4");
    assert!(serde_json::from_str::<crate::StreamingQr<f64, 3>>(&bad_ncols).is_err());
    let zero_ncols = qr_json.replace("\"ncols\":2", "\"ncols\":0");
    assert!(serde_json::from_str::<crate::StreamingQr<f64, 3>>(&zero_ncols).is_err());
}
