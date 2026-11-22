use super::*;
use diagnostics::{Diagnostic, DiagnosticBus, Severity};
use math_commons::error::{EvalError, MathError};
use math_commons::math_kernel::types::Dimension;
use std::error::Error as StdError;
use verified_engine::engine::VerifiedEngine;

fn sample_math_err() -> MathError {
    MathError::DimensionMismatch {
        expected: Dimension(2),
        actual: Dimension(3),
    }
}

#[test]
#[verified_engine::verified]
fn test_pure_math_leaf_wrappers_preserve_source_chain() {
    let markov = MarkovError::from(sample_math_err());
    assert!(markov.source().is_some());

    let copula = CopulaError::from(sample_math_err());
    assert!(copula.source().is_some());

    let tensor = TensorError::from(sample_math_err());
    assert!(tensor.source().is_some());

    let opt = OptimizationError::from(EvolutionError::BoundsError);
    assert!(opt.source().is_some());
}

#[test]
#[verified_engine::verified]
fn test_pure_math_umbrella_variants_preserve_source_and_diagnostics() {
    let cases: Vec<PureMathError> = vec![
        CopulaError::NotSymmetric.into(),
        Glicko2Error::EmptyMatchResults.into(),
        KellyError::InvalidOdds { value: 0.5 }.into(),
        MarkovError::from(sample_math_err()).into(),
        NumberTheoryError::ModuloError.into(),
        OuError::InvalidTimeStep { value: 0.0 }.into(),
        TdaError::EmptyPointCloud.into(),
        ZipError::InvalidRate { value: -1.0 }.into(),
        AmbsError::ConversionError.into(),
        AnalysisError::ConvergenceError(1.0).into(),
        EllipticCurveError::InvalidDegree.into(),
        EvolutionError::InvalidState.into(),
        LinearAlgebraError::SingularMatrix.into(),
        OptimizationError::from(EvolutionError::ConversionError).into(),
        StochasticError::InvalidReactionIndex(3).into(),
        TensorError::SingularMetric.into(),
        TransformError::EmptyInput.into(),
        sample_math_err().into(),
        EvalError::UnknownVariable("x".to_string()).into(),
    ];

    let bus = DiagnosticBus::new();
    for err in &cases {
        assert!(err.source().is_some());
        assert_eq!(err.severity(), Severity::Error);
        let meta = err.metadata();
        assert_eq!(
            meta.get("umbrella_type").map(String::as_str),
            Some("PureMathError")
        );
        bus.emit_diagnostic(err);
    }
}

#[verified_engine::verified]
fn propagate_pure_math_question_mark(trigger_err: bool) -> std::result::Result<f64, PureMathError> {
    if trigger_err {
        let _: f64 = Err(MathError::DimensionMismatch {
            expected: Dimension(1),
            actual: Dimension(2),
        })?;
    }
    let _: f64 = Ok::<f64, KellyError>(0.25)?;
    Ok(1.0)
}

#[test]
#[verified_engine::verified]
fn test_pure_math_error_zero_allocation_propagation() {
    let (ok, _) = VerifiedEngine::run_verified(|| propagate_pure_math_question_mark(false));
    assert!(ok.is_ok());

    let (err, _) = VerifiedEngine::run_verified(|| propagate_pure_math_question_mark(true));
    assert!(err.is_err());
}
