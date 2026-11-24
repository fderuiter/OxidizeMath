use std::sync::Arc;

use crate::fit::{
    CustomFitEvaluator, FitResult, FitType, fit_custom_evaluator, fit_model_with_axes,
};

#[derive(Debug)]
struct AffineScaleEval {
    names: Vec<String>,
}

impl CustomFitEvaluator for AffineScaleEval {
    fn param_names(&self) -> &[String] {
        &self.names
    }

    fn eval(&self, x: f64, params: &[f64]) -> Option<f64> {
        Some(params[0] * x + params[1])
    }
}

#[test]
fn fit_result_preset_roundtrips_json_and_preserves_evaluation() {
    let pts = [[100.0, 3.0], [101.0, 6.0], [102.0, 11.0], [103.0, 18.0]];
    let fit = fit_model_with_axes(&pts, FitType::Quadratic, "parabola", false, false, false)
        .expect("quadratic fit");

    let encoded = serde_json::to_string(&fit).expect("serialize FitResult");
    let decoded: FitResult = serde_json::from_str(&encoded).expect("deserialize FitResult");

    assert_eq!(decoded, fit);
    assert!((decoded.eval(101.5) - fit.eval(101.5)).abs() < 1e-12);
}

#[test]
fn fit_result_custom_skips_runtime_callback_handle() {
    let pts = [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]];
    let evaluator = Arc::new(AffineScaleEval {
        names: vec!["m".to_owned(), "b".to_owned()],
    });
    let fit = fit_custom_evaluator(&pts, evaluator, "m*x + b", &[1.0, 0.0], "affine")
        .expect("custom fit");
    assert!(fit.custom_evaluator.is_some());

    let encoded = serde_json::to_string(&fit).expect("serialize custom FitResult");
    assert!(!encoded.contains("custom_evaluator"));

    let decoded: FitResult = serde_json::from_str(&encoded).expect("deserialize custom FitResult");
    assert_eq!(decoded.fit_type, FitType::Custom);
    assert_eq!(decoded.coefficients, fit.coefficients);
    assert_eq!(decoded.dataset_name, "affine");
    assert!(decoded.custom_evaluator.is_none());
}
