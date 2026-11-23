#![doc = include_str!("README.md")]

/// Stack-allocated Levenberg-Marquardt calibration for [`ElectrochemicalAgingModel`].
pub mod calibration;
/// Coupled multi-segment duty-cycle state-of-health evaluation and warranty solvers.
pub mod duty;
pub mod model;
/// Validated domain types (`DepthOfDischarge`, `Capacity`, `Cycles`) and `BatteryError`.
pub mod types;

pub use calibration::{CalibratedAgingFit, calibrate_electrochemical_model};
pub use duty::{
    DAYS_PER_YEAR, DutyLifetimeSummary, DutyPointBreakdown, DutySegment, MAX_DUTY_SEGMENTS,
    estimate_duty_lifetime, evaluate_duty_point,
};
pub use model::{
    ChemistryPreset, DegradationModel, ElectrochemicalAgingModel, MAX_C_RATE, MAX_EXPONENT_Z,
    MAX_TEMPERATURE_C, MIN_C_RATE, MIN_EXPONENT_Z, MIN_TEMPERATURE_C, PowerLawModel,
};
pub use types::{BatteryError, Capacity, Cycles, DepthOfDischarge};

// [cite:battery_degradation]

use pure_math::theory_verification;

theory_verification!(
    module = "battery_degradation",
    constants = {},
    test = {
        use approx::assert_relative_eq;
        let model = model::PowerLawModel::standard();
        assert_relative_eq!(
            model.capacity(
                types::Cycles::new_clamped(0.0),
                types::DepthOfDischarge::new_clamped(1.0)
            ) / types::Capacity::new_clamped(1.0),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
