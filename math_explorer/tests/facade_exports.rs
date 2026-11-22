#![allow(missing_docs)]

#[test]
fn core_only_facade_exports_work_without_domain_features() {
    use math_explorer::ast::{Complex64, parse};
    use math_explorer::constants::C;
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error::{MathError, MathResult};
    use math_explorer::math_kernel::types::Dimension;
    use math_explorer::prelude::{
        Diagnostic as PreludeDiagnostic, MathError as PreludeMathError, NonNegativeFloat,
        PositiveFloat, Severity as PreludeSeverity, UnitInterval,
    };
    use math_explorer::primitives::PositiveFloat as DirectPositiveFloat;
    use math_explorer::registry::TOLERANCE_STANDARD;
    use math_explorer::wigner::ClebschGordan;

    let pos = PositiveFloat::new(1.5).expect("valid positive float");
    let direct_pos = DirectPositiveFloat::new(1.5).expect("valid direct positive float");
    assert_eq!(pos, direct_pos);

    let non_neg = NonNegativeFloat::new(0.0).expect("valid non-negative float");
    assert_eq!(non_neg.value(), 0.0);

    let unit = UnitInterval::new(0.25).expect("valid unit interval");
    assert_eq!(unit.value(), 0.25);

    let expr = parse("x + 1").expect("valid AST expression");
    assert!(!format!("{expr:?}").is_empty());
    let z = Complex64::new(1.0, -1.0);
    assert_eq!(z.re, 1.0);
    assert_eq!(C, 299_792_458.0);
    assert_eq!(TOLERANCE_STANDARD, 1e-9);

    let err = MathError::DimensionMismatch {
        expected: Dimension(3),
        actual: Dimension(2),
    };
    let res: MathResult<f64> = Err(err.clone());
    assert!(res.is_err());
    assert_eq!(Diagnostic::severity(&err), Severity::Error);
    assert_eq!(PreludeDiagnostic::severity(&err), PreludeSeverity::Error);
    let _same_err: PreludeMathError = err;

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
#[test]
fn pure_math_facade_exports_containers_third_party_types_and_eight_errors() {
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error::{
        CopulaError, Glicko2Error, KellyError, MarkovError, NumberTheoryError, OuError, TdaError,
        ZipError,
    };
    use math_explorer::math_types::{Float, IBig, Integer, Rational};
    use math_explorer::mesh::{Mesh, Point3D};
    use math_explorer::nalgebra::{DMatrix, DVector};
    use math_explorer::num_complex::Complex;
    use math_explorer::petgraph::Graph;
    use math_explorer::prelude::{
        DMatrix as PreludeMatrix, Integer as PreludeInteger, Mesh as PreludeMesh, OxidizeRng,
    };
    use math_explorer::pure_math::error as inner_error;
    use math_explorer::pure_math::statistics::markov::ToPrimitive;
    use math_explorer::rand::RngCore;

    let n = Integer::from(42);
    let n2 = PreludeInteger::from(IBig::from(42));
    assert_eq!(n, n2);
    assert_eq!(42_i64.to_usize(), Some(42));
    let f = Float::with_val(53, 2.5);
    let r = Rational::from((3, 4));
    assert_eq!(f.round(), 3.0);
    assert_eq!(r.numer() / r.denom(), 0.75);

    let mut mesh = Mesh::default();
    mesh.vertices.push(Point3D::new(0.0, 1.0, 2.0));
    let prelude_mesh: PreludeMesh = mesh;
    assert_eq!(prelude_mesh.vertices.len(), 1);

    let mut rng = OxidizeRng::new(7);
    let _sample = RngCore::next_u32(&mut rng);

    let mat = DMatrix::<f64>::identity(2, 2);
    let vec = DVector::<f64>::from_element(2, 3.0);
    let prod: DVector<f64> = &mat * vec;
    assert_eq!(prod[0], 3.0);
    let _pmat: PreludeMatrix<f64> = mat;

    let c = Complex::new(2.0_f64, 3.0_f64);
    assert_eq!(c.im, 3.0);
    let mut g: Graph<&str, f64> = Graph::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_edge(a, b, 1.0);
    assert_eq!(g.node_count(), 2);

    let severities = [
        NumberTheoryError::ModuloError.severity(),
        KellyError::InvalidOdds { value: 0.5 }.severity(),
        MarkovError::InvalidProbability { value: -0.1 }.severity(),
        OuError::InvalidTimeStep { value: 0.0 }.severity(),
        ZipError::InvalidCount { value: -1.0 }.severity(),
        TdaError::EmptyPointCloud.severity(),
        CopulaError::NotSymmetric.severity(),
        Glicko2Error::EmptyMatchResults.severity(),
        inner_error::TdaError::EmptyPointCloud.severity(),
    ];
    assert!(severities.iter().all(|s| *s == Severity::Error));
}

#[cfg(feature = "ai")]
#[test]
fn ai_facade_exports_adam_and_ai_error_without_internal_crate_imports() {
    use math_explorer::ai::error::AIError as InnerAIError;
    use math_explorer::ai::optimization::{
        Adam, GenericAdam, GenericOptimizer, ParamType, mean_squared_error,
    };
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::prelude::{AIError, DMatrix, DVector, PositiveFloat};

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
}

#[cfg(feature = "applied")]
#[test]
fn applied_facade_exports_six_errors_and_submodule_reexports() {
    use math_explorer::applied::battery_degradation::{
        BatteryError as SubBatteryError, Capacity, Cycles, DepthOfDischarge, PowerLawModel,
    };
    use math_explorer::applied::error as inner_error;
    use math_explorer::applied::game_theory::mechanism_design::{ContinuousCDF, Distribution};
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error::{
        BatteryError, EngineeringError, GameTheoryError, IsosurfaceError, LoraError,
        PharmacokineticsError,
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

    let severities = [
        BatteryError::InvalidCapacity(1.5).severity(),
        EngineeringError::InvalidParameter {
            name: "bits".to_string(),
            value: 0.0,
        }
        .severity(),
        GameTheoryError::NonSquarePayoffMatrix { rows: 2, cols: 3 }.severity(),
        IsosurfaceError::InvalidGrid("too small".to_string()).severity(),
        LoraError::EmptyEnsemble.severity(),
        PharmacokineticsError::InvalidParameter("v".to_string()).severity(),
        inner_error::LoraError::EmptyWeights.severity(),
    ];
    assert!(severities.iter().all(|s| *s == Severity::Error));
}

#[cfg(feature = "biology")]
#[test]
fn biology_facade_exports_hodgkin_huxley_error() {
    use math_explorer::biology::error::HodgkinHuxleyError as InnerHhError;
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error::HodgkinHuxleyError;

    let e1 = HodgkinHuxleyError::InvalidGatingVariable(1.5);
    let e2 = InnerHhError::InvalidConductance(-0.2);
    assert_eq!(e1.severity(), Severity::Error);
    assert_eq!(e2.severity(), Severity::Error);
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
fn epidemiology_facade_exports_epidemiology_error() {
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::epidemiology::error::EpidemiologyError as InnerEpiError;
    use math_explorer::error::EpidemiologyError;

    let e1 = EpidemiologyError::SingularTransitionMatrix;
    let e2 = InnerEpiError::MissingParameter {
        name: "beta".to_string(),
    };
    assert_eq!(e1.severity(), Severity::Error);
    assert_eq!(e2.severity(), Severity::Error);
}

#[cfg(feature = "physics")]
#[test]
fn physics_facade_exports_all_eight_physics_errors() {
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error::{
        ChaosError, DoseFluenceError, FluidError, HighEnergyError, RadarError, SolidStateError,
        StandardModelError, StatMechError,
    };
    use math_explorer::physics::chaos::{
        DoublePendulumBuilder, DoublePendulumState, RosslerBuilder, RosslerState,
        double_pendulum::DoublePendulum, rossler::RosslerSystem,
    };
    use math_explorer::physics::error as inner_error;

    let mut rossler: RosslerSystem = RosslerBuilder::new().build(RosslerState::new(0.1, 0.0, 0.0));
    rossler.step(0.01);
    assert!(rossler.state.vec.x.is_finite());

    let mut pendulum: DoublePendulum =
        DoublePendulumBuilder::new().build(DoublePendulumState::new(2.2, 2.2, 0.0, 0.0));
    pendulum.step(0.005);
    assert!(pendulum.total_energy().is_finite());

    let severities = [
        ChaosError::InvalidParameter("dt".to_string()).severity(),
        DoseFluenceError::Singularity.severity(),
        FluidError::InvalidDensity { value: 0.0 }.severity(),
        HighEnergyError::InvalidMomentum.severity(),
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
fn all_twenty_six_crate_error_enums_constructible_from_math_explorer_error() {
    use math_explorer::diagnostics::{Diagnostic, Severity};
    use math_explorer::error::{
        AIError, BatteryError, ChaosError, CopulaError, DoseFluenceError, EngineeringError,
        EpidemiologyError, FluidError, GameTheoryError, Glicko2Error, HighEnergyError,
        HodgkinHuxleyError, IsosurfaceError, KellyError, LoraError, MarkovError, MathError,
        NumberTheoryError, OuError, PharmacokineticsError, RadarError, SolidStateError,
        StandardModelError, StatMechError, TdaError, ZipError,
    };

    let all_26_severities = [
        MathError::ConversionError {
            reason: "bad".to_string(),
        }
        .severity(),
        NumberTheoryError::ConversionError.severity(),
        KellyError::InvalidBankroll { value: -1.0 }.severity(),
        MarkovError::InvalidState {
            reason: "s".to_string(),
        }
        .severity(),
        OuError::InvalidVolatility { value: -0.1 }.severity(),
        ZipError::InvalidRate { value: 0.0 }.severity(),
        TdaError::EmptyPointCloud.severity(),
        CopulaError::NotPositiveDefinite.severity(),
        Glicko2Error::EmptyMatchResults.severity(),
        AIError::OptimizationDivergence.severity(),
        BatteryError::NegativeCycles(-5.0).severity(),
        EngineeringError::InvalidParameter {
            name: "n".to_string(),
            value: 0.0,
        }
        .severity(),
        GameTheoryError::NonSquarePayoffMatrix { rows: 1, cols: 2 }.severity(),
        IsosurfaceError::InvalidGrid("small".to_string()).severity(),
        LoraError::KeyMismatch.severity(),
        PharmacokineticsError::InvalidParameter("k".to_string()).severity(),
        HodgkinHuxleyError::InvalidConductance(-1.0).severity(),
        EpidemiologyError::SingularTransitionMatrix.severity(),
        ChaosError::InvalidParameter("sigma".to_string()).severity(),
        DoseFluenceError::Singularity.severity(),
        FluidError::InvalidViscosity { value: -0.1 }.severity(),
        HighEnergyError::InvalidMomentum.severity(),
        RadarError::InvalidConfiguration("cfg".to_string()).severity(),
        SolidStateError::ComputationError("err".to_string()).severity(),
        StandardModelError::InvalidFlavors { nf: -1.0 }.severity(),
        StatMechError::InvalidTemperature(-5.0).severity(),
    ];
    assert_eq!(all_26_severities.len(), 26);
    assert!(all_26_severities.iter().all(|s| *s == Severity::Error));
}
