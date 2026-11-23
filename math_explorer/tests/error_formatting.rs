//! Verifies that all 27 public error enums re-exported by `math_explorer::error`
//! emit human-readable `Display` sentences and `Diagnostic::metadata` without `{:?}` Debug formatting.

#![cfg(all(
    feature = "pure_math",
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics"
))]

use math_explorer::diagnostics::Diagnostic;
use math_explorer::error::{
    AIError, BatteryError, ChaosError, ClimateError, CopulaError, DoseFluenceError,
    EngineeringError, EpidemiologyError, FluidError, GameTheoryError, Glicko2Error,
    HighEnergyError, HodgkinHuxleyError, IsosurfaceError, KellyError, LoraError, MarkovError,
    MathError, NumberTheoryError, OuError, PharmacokineticsError, RadarError, SolidStateError,
    StandardModelError, StatMechError, TdaError, ZipError,
};
use math_explorer::math_kernel::types::Dimension;

fn assert_readable_diagnostic<E: Diagnostic>(name: &str, err: &E) {
    let display = err.to_string();
    assert!(
        display.contains(' '),
        "{name} Display must be a human-readable sentence, got `{display}`"
    );
    assert!(
        !display.contains('{') && !display.contains('}') && !display.contains("Dimension("),
        "{name} Display must not contain Debug struct formatting: `{display}`"
    );
    let meta = err.metadata();
    let desc = meta
        .get("description")
        .unwrap_or_else(|| panic!("{name} metadata missing description"));
    assert!(
        !desc.contains('{') && !desc.contains('}') && !desc.contains("Dimension("),
        "{name} Diagnostic description must not contain Debug struct formatting: `{desc}`"
    );
}

#[test]
fn pure_math_and_core_errors_emit_readable_display_and_metadata() {
    assert_readable_diagnostic(
        "MathError",
        &MathError::DimensionMismatch {
            expected: Dimension(3),
            actual: Dimension(2),
        },
    );
    assert_readable_diagnostic(
        "MathError::OutOfBounds",
        &MathError::OutOfBounds {
            value: 1.5,
            min: 0.0,
            max: 1.0,
        },
    );
    assert_readable_diagnostic(
        "NumberTheoryError",
        &NumberTheoryError::DivisionByZeroQSeries,
    );
    assert_readable_diagnostic("KellyError", &KellyError::InvalidBankroll { value: -1.0 });
    assert_readable_diagnostic(
        "MarkovError",
        &MarkovError::InvalidProbability { value: 1.5 },
    );
    assert_readable_diagnostic("OuError", &OuError::InvalidVolatility { value: -0.1 });
    assert_readable_diagnostic("ZipError", &ZipError::InvalidRate { value: -0.5 });
    assert_readable_diagnostic("TdaError", &TdaError::EmptyPointCloud);
    assert_readable_diagnostic("CopulaError", &CopulaError::NotPositiveDefinite);
    assert_readable_diagnostic(
        "Glicko2Error",
        &Glicko2Error::InvalidSystemConstant { value: 0.0 },
    );
}

#[test]
fn applied_ai_bio_climate_epi_errors_emit_readable_display_and_metadata() {
    assert_readable_diagnostic(
        "AIError",
        &AIError::InvalidHeadConfiguration {
            d_model: 65,
            heads: 8,
        },
    );
    assert_readable_diagnostic("BatteryError", &BatteryError::NegativeCycles(-5.0));
    assert_readable_diagnostic(
        "ClimateError",
        &ClimateError::ShapeMismatch {
            context: "MSE",
            expected: (2, 3),
            actual: (3, 2),
        },
    );
    assert_readable_diagnostic("EngineeringError", &EngineeringError::SingularDesignMatrix);
    assert_readable_diagnostic(
        "GameTheoryError",
        &GameTheoryError::NonSquarePayoffMatrix { rows: 2, cols: 3 },
    );
    assert_readable_diagnostic(
        "IsosurfaceError",
        &IsosurfaceError::InvalidGrid("1x1x1".to_string()),
    );
    assert_readable_diagnostic("LoraError", &LoraError::KeyMismatch);
    assert_readable_diagnostic(
        "PharmacokineticsError",
        &PharmacokineticsError::InvalidParameter("volume".to_string()),
    );
    assert_readable_diagnostic(
        "HodgkinHuxleyError",
        &HodgkinHuxleyError::InvalidConductance(-1.0),
    );
    assert_readable_diagnostic(
        "EpidemiologyError",
        &EpidemiologyError::InvalidReproductionNumber { r0: 0.0 },
    );
}

#[test]
fn physics_and_wrapped_math_errors_emit_readable_display_and_metadata() {
    assert_readable_diagnostic(
        "ChaosError",
        &ChaosError::InvalidParameter("dt".to_string()),
    );
    assert_readable_diagnostic(
        "DoseFluenceError",
        &DoseFluenceError::InsufficientCalibrationPoints {
            minimum: 2,
            actual: 1,
        },
    );
    assert_readable_diagnostic(
        "FluidError",
        &FluidError::GridDimensionsOverflow {
            width: 10,
            height: 20,
        },
    );
    assert_readable_diagnostic(
        "HighEnergyError",
        &HighEnergyError::InvalidLorentzFactor { gamma: 0.5 },
    );
    assert_readable_diagnostic(
        "RadarError",
        &RadarError::AntennaWeightCountMismatch {
            expected: 4,
            actual: 2,
        },
    );
    assert_readable_diagnostic(
        "SolidStateError",
        &SolidStateError::PauliExclusionViolation { requested: 2 },
    );
    assert_readable_diagnostic(
        "StandardModelError",
        &StandardModelError::UnbrokenSymmetry { mu2: 1.0 },
    );
    assert_readable_diagnostic("StatMechError", &StatMechError::InvalidTemperature(-5.0));

    let math_err = MathError::StrictlyOutOfBounds {
        value: 0.0,
        min: 0.0,
        max: f64::INFINITY,
    };
    assert_readable_diagnostic("ClimateError::Math", &ClimateError::from(math_err.clone()));
    assert_readable_diagnostic(
        "EngineeringError::Math",
        &EngineeringError::from(math_err.clone()),
    );
    assert_readable_diagnostic("DoseFluenceError::Math", &DoseFluenceError::from(math_err));
}
