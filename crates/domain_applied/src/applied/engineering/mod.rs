//! # Engineering Utilities
//!
//! Practical formulas and calculations for hardware engineering and reliability analysis.
//!
//! This module collects utility functions that don't fit into the larger simulation frameworks
//! but are essential for "back-of-the-napkin" estimation in embedded systems and manufacturing.

#[allow(missing_docs)]
pub mod cnc;
#[allow(missing_docs)]
pub mod uart;

pub use cnc::*;
pub use uart::*;

// [cite:graph_parameters_rust]

use pure_math::theory_verification;

theory_verification!(
    module = "engineering",
    constants = {
        TOLERANCE = 0.001;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(
            TOLERANCE,
            0.001,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
