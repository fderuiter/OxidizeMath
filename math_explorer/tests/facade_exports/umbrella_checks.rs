use math_explorer::diagnostics::{Diagnostic, DiagnosticBus, Severity};
use math_explorer::error::{self, Error, Result};
use math_explorer::math_kernel::types::Dimension;
use std::error::Error as StdError;
use std::sync::{Arc, Mutex};
use verified_engine::engine::VerifiedEngine;

fn sample_math() -> error::MathError {
    error::MathError::DimensionMismatch {
        expected: Dimension(2),
        actual: Dimension(3),
    }
}

fn check_leaf<E>(err: E, bus: &DiagnosticBus) -> Error
where
    E: Diagnostic + StdError,
    Error: From<E>,
{
    assert_eq!(err.severity(), Severity::Error);
    assert!(!err.metadata().is_empty());
    bus.emit_diagnostic(&err);
    let propagated: Result<()> = (|| -> Result<()> { Err(err)? })();
    let facade_err = propagated.expect_err("must convert into facade Error");
    assert!(facade_err.source().is_some());
    assert_eq!(facade_err.severity(), Severity::Error);
    assert_eq!(
        facade_err
            .metadata()
            .get("umbrella_type")
            .map(String::as_str),
        Some("math_explorer::error::Error")
    );
    facade_err
}

pub fn verify_core_and_pure_math_leaves(bus: &DiagnosticBus) -> Vec<Error> {
    vec![
        check_leaf(sample_math(), bus),
        check_leaf(error::FitError::SingularMatrix, bus),
        check_leaf(error::EvalError::UnknownVariable("x".to_string()), bus),
        check_leaf(error::GlslError::DepthLimitExceeded(64), bus),
        check_leaf(
            error::LexError {
                start: 0,
                end: 1,
                message: "bad".to_string(),
            },
            bus,
        ),
        check_leaf(error::ParseError::UnexpectedEof, bus),
        check_leaf(error::AmbsError::ConversionError, bus),
        check_leaf(error::AnalysisError::ConvergenceError(1.0), bus),
        check_leaf(error::CopulaError::NotSymmetric, bus),
        check_leaf(error::EllipticCurveError::InvalidDegree, bus),
        check_leaf(error::EvolutionError::BoundsError, bus),
        check_leaf(error::FractranError::InvalidFractionSyntax, bus),
        check_leaf(error::Glicko2Error::EmptyMatchResults, bus),
        check_leaf(error::KellyError::InvalidBankroll { value: -1.0 }, bus),
        check_leaf(error::LinearAlgebraError::SingularMatrix, bus),
        check_leaf(
            error::MarketPricingError::NoArbitrage {
                reciprocal_sum: 1.05,
            },
            bus,
        ),
        check_leaf(
            error::MarkovError::InvalidState {
                reason: "s".to_string(),
            },
            bus,
        ),
        check_leaf(error::NumberTheoryError::ConversionError, bus),
        check_leaf(error::OptimizationError::ConversionError, bus),
        check_leaf(error::OuError::InvalidVolatility { value: -0.1 }, bus),
        check_leaf(error::StochasticError::InvalidReactionIndex(2), bus),
        check_leaf(error::TdaError::EmptyPointCloud, bus),
        check_leaf(error::TensorError::SingularMetric, bus),
        check_leaf(error::TransformError::EmptyInput, bus),
        check_leaf(error::ZipError::InvalidRate { value: 0.0 }, bus),
    ]
}

pub fn verify_domain_leaves(bus: &DiagnosticBus) -> Vec<Error> {
    vec![
        check_leaf(error::AIError::OptimizationDivergence, bus),
        check_leaf(error::BatteryError::NegativeCycles(-5.0), bus),
        check_leaf(error::ClinicalTrialError::ZeroTotalCount, bus),
        check_leaf(
            error::EngineeringError::InvalidParameter {
                name: "n".to_string(),
                value: 0.0,
            },
            bus,
        ),
        check_leaf(
            error::GameTheoryError::NonSquarePayoffMatrix { rows: 1, cols: 2 },
            bus,
        ),
        check_leaf(
            error::IsosurfaceError::InvalidGrid("small".to_string()),
            bus,
        ),
        check_leaf(error::KalmanError::MatrixInversionError, bus),
        check_leaf(error::LoraError::KeyMismatch, bus),
        check_leaf(
            error::PharmacokineticsError::InvalidParameter("k".to_string()),
            bus,
        ),
        check_leaf(error::VoxelGridError::InvalidDimensions, bus),
        check_leaf(error::HodgkinHuxleyError::InvalidConductance(-1.0), bus),
        check_leaf(error::KineticsError::InvalidParameters, bus),
        check_leaf(error::ReactionDiffusionError::ZeroSpecies, bus),
        check_leaf(error::EpidemiologyError::SingularTransitionMatrix, bus),
        check_leaf(error::NetworkModelError::InvalidNodeCount, bus),
        check_leaf(
            error::ChaosError::InvalidParameter("sigma".to_string()),
            bus,
        ),
        check_leaf(error::DoseFluenceError::Singularity, bus),
        check_leaf(error::FluidError::InvalidViscosity { value: -0.1 }, bus),
        check_leaf(error::GatingError::InterlockLatched, bus),
        check_leaf(error::HighEnergyError::InvalidMomentum, bus),
        check_leaf(error::NuclearError::InvalidHalfLife, bus),
        check_leaf(
            error::RadarError::InvalidConfiguration("cfg".to_string()),
            bus,
        ),
        check_leaf(
            error::SolidStateError::ComputationError("err".to_string()),
            bus,
        ),
        check_leaf(error::StandardModelError::InvalidFlavors { nf: -1.0 }, bus),
        check_leaf(error::StatMechError::InvalidTemperature(-5.0), bus),
    ]
}

pub fn verify_six_umbrellas_and_sixteen_wrappers(bus: &DiagnosticBus) {
    let umbrellas: [Error; 6] = [
        error::MathCommonsError::from(sample_math()).into(),
        error::PureMathError::from(error::KellyError::InvalidOdds { value: 0.5 }).into(),
        error::AppliedError::from(error::LoraError::EmptyEnsemble).into(),
        error::BiologyError::from(error::KineticsError::InvalidParameters).into(),
        error::EpidemiologyDomainError::from(error::NetworkModelError::InvalidNodeCount).into(),
        error::PhysicsError::from(error::NuclearError::VolumeZero).into(),
    ];
    for u in &umbrellas {
        assert!(u.source().is_some());
        bus.emit_diagnostic(u);
    }

    let wrapper_sources_present = [
        error::ParseError::from(error::LexError {
            start: 0,
            end: 1,
            message: "bad".to_string(),
        })
        .source()
        .is_some(),
        error::MarkovError::from(sample_math()).source().is_some(),
        error::CopulaError::from(sample_math()).source().is_some(),
        error::TensorError::from(sample_math()).source().is_some(),
        error::OptimizationError::from(error::EvolutionError::BoundsError)
            .source()
            .is_some(),
        error::MarketPricingError::from(error::AnalysisError::ConvergenceError(0.0))
            .source()
            .is_some(),
        error::AIError::from(sample_math()).source().is_some(),
        error::PharmacokineticsError::from(error::AnalysisError::ConvergenceError(0.0))
            .source()
            .is_some(),
        error::IsosurfaceError::from(sample_math())
            .source()
            .is_some(),
        error::KalmanError::from(sample_math()).source().is_some(),
        error::ReactionDiffusionError::from(sample_math())
            .source()
            .is_some(),
        error::EpidemiologyError::from(sample_math())
            .source()
            .is_some(),
        error::ChaosError::from(sample_math()).source().is_some(),
        error::RadarError::from(sample_math()).source().is_some(),
        error::HighEnergyError::from(sample_math())
            .source()
            .is_some(),
        error::StatMechError::from(sample_math()).source().is_some(),
    ];
    assert_eq!(wrapper_sources_present.len(), 16);
    assert!(wrapper_sources_present.into_iter().all(|present| present));
}

pub fn verify_bus_capture_and_zero_alloc_facade_result() {
    let bus = DiagnosticBus::new();
    let events = Arc::new(Mutex::new(0_usize));
    let counter = Arc::clone(&events);
    bus.register_listener(move |_| {
        *counter.lock().unwrap() += 1;
    });

    let mut converted = verify_core_and_pure_math_leaves(&bus);
    converted.extend(verify_domain_leaves(&bus));
    assert_eq!(converted.len(), 50);
    assert_eq!(*events.lock().unwrap(), 50);

    verify_six_umbrellas_and_sixteen_wrappers(&bus);
    assert_eq!(*events.lock().unwrap(), 56);

    let (ok_run, _) = VerifiedEngine::run_verified(|| -> Result<f64> {
        let x: f64 = Ok::<f64, error::KellyError>(0.5)?;
        let y: f64 = Ok::<f64, error::BatteryError>(0.5)?;
        Ok(x + y)
    });
    assert_eq!(ok_run.unwrap(), 1.0);

    let (err_run, _) = VerifiedEngine::run_verified(|| -> Result<f64> {
        let _: f64 = Err(error::NuclearError::VolumeZero)?;
        Ok(0.0)
    });
    assert!(err_run.is_err());
}

fn verify_prelude_domain_leaves() {
    use math_explorer::prelude as p;

    let _applied_and_phys = core::mem::size_of::<(
        p::BatteryError,
        p::ClinicalTrialError,
        p::EngineeringError,
        p::GameTheoryError,
        p::IsosurfaceError,
        p::KalmanError,
        p::LoraError,
        p::PharmacokineticsError,
        p::VoxelGridError,
        p::ChaosError,
        p::DoseFluenceError,
        p::FluidError,
        p::GatingError,
        p::HighEnergyError,
        p::NuclearError,
        p::RadarError,
        p::SolidStateError,
        p::StandardModelError,
        p::StatMechError,
    )>();
}

pub fn verify_prelude_error_reexports() {
    use math_explorer::prelude as p;

    let res: p::Result<(
        p::MathCommonsError,
        p::PureMathError,
        p::AppliedError,
        p::BiologyError,
        p::EpidemiologyDomainError,
        p::PhysicsError,
    )> = Err(p::Error::from(p::MathError::ConversionError {
        reason: "bad".to_string(),
    }));
    assert!(res.is_err());
    let _core = core::mem::size_of::<(
        p::EvalError,
        p::FitError,
        p::GlslError,
        p::LexError,
        p::MathError,
        p::ParseError,
        p::AIError,
        p::HodgkinHuxleyError,
        p::KineticsError,
        p::ReactionDiffusionError,
        p::EpidemiologyError,
        p::NetworkModelError,
    )>();
    let _pure = core::mem::size_of::<(
        p::AmbsError,
        p::AnalysisError,
        p::CopulaError,
        p::EllipticCurveError,
        p::EvolutionError,
        p::FractranError,
        p::Glicko2Error,
        p::KellyError,
        p::LinearAlgebraError,
        p::MarketPricingError,
        p::MarkovError,
        p::NumberTheoryError,
        p::OptimizationError,
        p::OuError,
        p::StochasticError,
        p::TdaError,
        p::TensorError,
        p::TransformError,
        p::ZipError,
    )>();
    verify_prelude_domain_leaves();
}
