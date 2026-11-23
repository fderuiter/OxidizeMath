#![allow(missing_docs)]

#[cfg(all(
    feature = "pure_math",
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics"
))]
#[path = "facade_exports/umbrella_checks.rs"]
mod umbrella_checks;

fn assert_core_error_and_prelude_exports() {
    use math_explorer::diagnostics::{Diagnostic, DiagnosticBus, Severity};
    use math_explorer::error::{
        Error, EvalError, GlslError, LexError, MathCommonsError, MathError, MathResult, ParseError,
        Result,
    };
    use math_explorer::math_kernel::types::Dimension;
    use math_explorer::prelude as p;

    let err = MathError::DimensionMismatch {
        expected: Dimension(3),
        actual: Dimension(2),
    };
    let res: MathResult<f64> = Err(err.clone());
    assert!(res.is_err());
    assert_eq!(Diagnostic::severity(&err), Severity::Error);
    assert_eq!(p::Diagnostic::severity(&err), p::Severity::Error);

    let bus = DiagnosticBus::new();
    let facade_res: Result<()> = Err(EvalError::UnknownVariable("x".to_string()).into());
    let lex = LexError {
        start: 0,
        end: 1,
        message: "bad".to_string(),
    };
    let prelude_res: p::Result<()> = Err(p::Error::from(lex.clone()));
    assert!(facade_res.is_err() && prelude_res.is_err());
    bus.emit_diagnostic(&p::MathCommonsError::from(p::ParseError::UnexpectedEof));
    let (_e, _g, _l): (p::EvalError, p::GlslError, p::LexError) = (
        EvalError::UnknownVariable("x".into()),
        GlslError::DepthLimitExceeded(64),
        lex,
    );
    let (_u, _m): (MathCommonsError, p::MathError) = (p::MathCommonsError::from(err.clone()), err);
    let _err_ty: Error = ParseError::UnexpectedEof.into();
}

#[test]
fn core_only_facade_exports_work_without_domain_features() {
    use math_explorer::ast::{Complex64, parse};
    use math_explorer::constants::C;
    use math_explorer::prelude::{NonNegativeFloat, PositiveFloat, UnitInterval};
    use math_explorer::primitives::PositiveFloat as DirectPositiveFloat;
    use math_explorer::registry::TOLERANCE_STANDARD;
    use math_explorer::wigner::ClebschGordan;

    let pos = PositiveFloat::new(1.5).expect("valid positive float");
    let direct_pos = DirectPositiveFloat::new(1.5).expect("valid direct positive float");
    assert_eq!(pos, direct_pos);
    assert_eq!(NonNegativeFloat::new(0.0).unwrap().value(), 0.0);
    assert_eq!(UnitInterval::new(0.25).unwrap().value(), 0.25);

    let expr = parse("x + 1").expect("valid AST expression");
    assert!(!format!("{expr:?}").is_empty());
    assert_eq!(Complex64::new(1.0, -1.0).re, 1.0);
    assert_eq!(C, 299_792_458.0);
    assert_eq!(TOLERANCE_STANDARD, 1e-9);
    assert_core_error_and_prelude_exports();

    let cg = ClebschGordan {
        tj1: 3,
        tm1: -1,
        tj2: 2,
        tm2: 2,
        tj12: 5,
        tm12: 1,
    };
    assert!((cg.value() - (0.3_f64).sqrt()).abs() < 1e-6);
}

#[cfg(feature = "pure_math")]
fn assert_pure_math_eighteen_errors() {
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error as e;
    use math_explorer::prelude::PureMathError as PreludePureMathError;
    use math_explorer::pure_math::error as inner_error;

    let umbrella: PreludePureMathError = e::AmbsError::ModuloError.into();
    let _facade: e::Error = e::PureMathError::from(e::LinearAlgebraError::DimensionMismatch).into();
    let severities = [
        e::NumberTheoryError::ModuloError.severity(),
        e::KellyError::InvalidOdds { value: 0.5 }.severity(),
        e::MarkovError::InvalidProbability { value: -0.1 }.severity(),
        e::OuError::InvalidTimeStep { value: 0.0 }.severity(),
        e::ZipError::InvalidCount { value: -1.0 }.severity(),
        e::TdaError::EmptyPointCloud.severity(),
        e::CopulaError::NotSymmetric.severity(),
        e::Glicko2Error::EmptyMatchResults.severity(),
        e::AnalysisError::ConvergenceError(0.0).severity(),
        e::EllipticCurveError::InvalidDegree.severity(),
        e::EvolutionError::InvalidState.severity(),
        e::FractranError::InvalidFractionSyntax.severity(),
        e::OptimizationError::ConversionError.severity(),
        e::StochasticError::InvalidReactionIndex(1).severity(),
        e::TensorError::IndexOutOfBounds.severity(),
        e::TransformError::InvalidDimension.severity(),
        inner_error::NumberTheoryError::ModuloError.severity(),
        umbrella.severity(),
    ];
    assert!(severities.iter().all(|s| *s == Severity::Error));
}

#[cfg(feature = "pure_math")]
#[test]
fn pure_math_facade_exports_containers_third_party_types_and_all_eighteen_errors() {
    use math_explorer::math_types::{Float, IBig, Integer, Rational};
    use math_explorer::mesh::{Mesh, Point3D};
    use math_explorer::nalgebra::{DMatrix, DVector};
    use math_explorer::num_complex::Complex;
    use math_explorer::petgraph::Graph;
    use math_explorer::prelude::{
        DMatrix as PreludeMatrix, Integer as PreludeInteger, Mesh as PreludeMesh, OxidizeRng,
    };
    use math_explorer::pure_math::statistics::markov::ToPrimitive;
    use math_explorer::rand::RngCore;

    assert_eq!(Integer::from(42), PreludeInteger::from(IBig::from(42)));
    assert_eq!(42_i64.to_usize(), Some(42));
    assert_eq!(Float::with_val(53, 2.5).round(), 3.0);
    assert_eq!(
        Rational::from((3, 4)).numer() / Rational::from((3, 4)).denom(),
        0.75
    );

    let mut mesh = Mesh::default();
    mesh.vertices.push(Point3D::new(0.0, 1.0, 2.0));
    let prelude_mesh: PreludeMesh = mesh;
    assert_eq!(prelude_mesh.vertices.len(), 1);
    let _sample = RngCore::next_u32(&mut OxidizeRng::new(7));

    let mat = DMatrix::<f64>::identity(2, 2);
    let prod: DVector<f64> = &mat * DVector::<f64>::from_element(2, 3.0);
    assert_eq!(prod[0], 3.0);
    let _pmat: PreludeMatrix<f64> = mat;
    assert_eq!(Complex::new(2.0_f64, 3.0_f64).im, 3.0);

    let mut g: Graph<&str, f64> = Graph::new();
    let (a, b) = (g.add_node("a"), g.add_node("b"));
    g.add_edge(a, b, 1.0);
    assert_eq!(g.node_count(), 2);
    assert_pure_math_eighteen_errors();
}

#[cfg(feature = "ai")]
#[test]
fn ai_facade_exports_adam_and_ai_error_without_internal_crate_imports() {
    use math_explorer::ai::error::AIError as InnerAIError;
    use math_explorer::ai::optimization::{
        Adam, GenericAdam, GenericOptimizer, ParamType, mean_squared_error,
    };
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::prelude::{AIError, DMatrix, DVector, Error, PositiveFloat};

    let lr = PositiveFloat::new(0.01).expect("valid learning rate");
    let mut opt: Adam<f64> = Adam::new(lr.value()).expect("valid Adam");
    let mut weights = DMatrix::from_element(2, 2, 1.0_f64);
    let grads = DMatrix::from_element(2, 2, 0.1_f64);
    opt.update_matrix((0, ParamType::Weight), &mut weights, &grads)
        .expect("valid Adam update");
    assert!(weights[(0, 0)] < 1.0);

    let _generic: GenericAdam<f64, usize> =
        GenericAdam::new(lr.value()).expect("valid GenericAdam");
    let y_pred = DVector::from_vec(vec![1.0_f64, 2.0]);
    let y_true = DVector::from_vec(vec![1.0_f64, 1.0]);
    let mse = mean_squared_error(&y_pred, &y_true).expect("valid MSE");
    assert!((mse - 0.5).abs() < 1e-12);

    let e1 = AIError::OptimizationDivergence;
    let e2 = InnerAIError::UninitializedState {
        name: "m".to_string(),
    };
    assert_eq!(e1.severity(), Severity::Error);
    assert_eq!(e2.severity(), Severity::Error);
    let _facade: Error = e1.into();
}

#[cfg(feature = "applied")]
#[test]
fn applied_facade_exports_all_nine_errors_and_umbrella() {
    use math_explorer::applied::battery_degradation::{
        BatteryError as SubBatteryError, Capacity, Cycles, DepthOfDischarge, PowerLawModel,
    };
    use math_explorer::applied::error as inner_error;
    use math_explorer::applied::game_theory::mechanism_design::{ContinuousCDF, Distribution};
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error::{
        AppliedError, BatteryError, ClinicalTrialError, EngineeringError, Error, GameTheoryError,
        IsosurfaceError, KalmanError, LoraError, PharmacokineticsError, VoxelGridError,
    };

    let model = PowerLawModel::standard();
    let cap = model.capacity(
        Cycles::new_clamped(100.0),
        DepthOfDischarge::new_clamped(0.8),
    );
    assert!(cap / Capacity::new_clamped(1.0) > 0.0);
    let bad_err: SubBatteryError = Cycles::new(-1.0).expect_err("negative cycles must fail");
    let root_err: BatteryError = bad_err;
    assert_eq!(root_err.severity(), Severity::Error);

    let dist =
        math_explorer::applied::game_theory::mechanism_design::distribution::Uniform::new(0.0, 1.0)
            .expect("valid uniform");
    assert!((dist.cdf(0.5) - 0.5).abs() < 1e-12);
    assert!((dist.mean().unwrap() - 0.5).abs() < 1e-12);

    let umbrella = AppliedError::from(VoxelGridError::InvalidDimensions);
    let _facade: Error = umbrella.into();
    let severities = [
        BatteryError::InvalidCapacity(1.5).severity(),
        ClinicalTrialError::ZeroTotalCount.severity(),
        EngineeringError::InvalidParameter {
            name: "bits".to_string(),
            value: 0.0,
        }
        .severity(),
        GameTheoryError::NonSquarePayoffMatrix { rows: 2, cols: 3 }.severity(),
        IsosurfaceError::InvalidGrid("too small".to_string()).severity(),
        KalmanError::MatrixInversionError.severity(),
        LoraError::EmptyEnsemble.severity(),
        PharmacokineticsError::InvalidParameter("v".to_string()).severity(),
        VoxelGridError::DataSizeMismatch.severity(),
        inner_error::LoraError::EmptyWeights.severity(),
    ];
    assert!(severities.iter().all(|s| *s == Severity::Error));
}

#[cfg(feature = "biology")]
#[test]
fn biology_facade_exports_all_three_errors_and_umbrella() {
    use math_explorer::biology::error::HodgkinHuxleyError as InnerHhError;
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error::{
        BiologyError, Error, HodgkinHuxleyError, KineticsError, ReactionDiffusionError,
    };

    let e1 = HodgkinHuxleyError::InvalidGatingVariable(1.5);
    let e2 = InnerHhError::InvalidConductance(-0.2);
    let e3 = KineticsError::NegativeSubstrateConcentration;
    let e4 = ReactionDiffusionError::ZeroGridSize;
    assert_eq!(e1.severity(), Severity::Error);
    assert_eq!(e2.severity(), Severity::Error);
    assert_eq!(e3.severity(), Severity::Error);
    assert_eq!(e4.severity(), Severity::Error);
    let _facade: Error = BiologyError::from(e3).into();
}

#[cfg(feature = "climate")]
#[test]
fn climate_facade_exports_error_module_and_cera_types() {
    use math_explorer::climate::cera::CeraConfig;
    use math_explorer::climate::error as climate_error;

    let cfg = CeraConfig {
        learning_rate: 1e-3,
        lambda_pred: 1.0,
        lambda_emd: 0.1,
        epochs: 2,
        batch_size: 4,
        in_channels: 3,
        latent_channels: 4,
        aligned_channels: 2,
        num_levels: 5,
        output_size: 2,
    };
    assert_eq!(cfg.latent_channels, 4);
    #[allow(unused_imports)]
    use climate_error::*;
}

#[cfg(feature = "epidemiology")]
#[test]
fn epidemiology_facade_exports_both_errors_and_umbrella() {
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::epidemiology::error::EpidemiologyError as InnerEpiError;
    use math_explorer::error::{
        EpidemiologyDomainError, EpidemiologyError, Error, NetworkModelError,
    };

    let e1 = EpidemiologyError::SingularTransitionMatrix;
    let e2 = InnerEpiError::MissingParameter {
        name: "beta".to_string(),
    };
    let e3 = NetworkModelError::InvalidParameters;
    assert_eq!(e1.severity(), Severity::Error);
    assert_eq!(e2.severity(), Severity::Error);
    assert_eq!(e3.severity(), Severity::Error);
    let _facade: Error = EpidemiologyDomainError::from(e3).into();
}

#[cfg(feature = "physics")]
#[test]
fn physics_facade_exports_all_ten_physics_errors_and_umbrella() {
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error::{
        ChaosError, DoseFluenceError, Error, FluidError, GatingError, HighEnergyError,
        NuclearError, PhysicsError, RadarError, SolidStateError, StandardModelError, StatMechError,
    };
    use math_explorer::physics::error as inner_error;

    let _facade: Error = PhysicsError::from(NuclearError::VolumeZero).into();
    let severities = [
        ChaosError::InvalidParameter("dt".to_string()).severity(),
        DoseFluenceError::Singularity.severity(),
        FluidError::InvalidDensity { value: 0.0 }.severity(),
        GatingError::NotInterlocked.severity(),
        HighEnergyError::InvalidMomentum.severity(),
        NuclearError::InvalidVelocity.severity(),
        RadarError::InvalidConfiguration("antennas".to_string()).severity(),
        SolidStateError::InvalidParameter("gap".to_string()).severity(),
        StandardModelError::LandauPole { scale: 1e12 }.severity(),
        StatMechError::InvalidTemperature(-1.0).severity(),
        inner_error::DoseFluenceError::NegativeRadius.severity(),
    ];
    assert!(severities.iter().all(|s| *s == Severity::Error));
}

#[cfg(all(
    feature = "pure_math",
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics"
))]
#[test]
fn all_forty_eight_leaf_errors_and_umbrellas_convert_and_emit_diagnostics() {
    umbrella_checks::verify_prelude_error_reexports();
    umbrella_checks::verify_bus_capture_and_zero_alloc_facade_result();
}
