#![doc = include_str!("README.md")]

pub mod model;
#[allow(missing_docs)]
pub mod types;

pub use model::PowerLawModel;
pub use types::{Capacity, Cycles, DepthOfDischarge};

// [cite:algorithmic_information_rust]

use pure_math::theory_verification;

theory_verification!(
    module = "battery_degradation",
    epsilon = math_commons::registry::TOLERANCE_FAST,
    constants = {
        TARGET_CAP = 0.8;
    },
    test = {
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
