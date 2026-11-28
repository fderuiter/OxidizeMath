//! # Engineering Utilities
//!
//! Practical formulas and calculations for hardware engineering and reliability analysis.
//!
//! This module collects utility functions that don't fit into the larger simulation frameworks
//! but are essential for "back-of-the-napkin" estimation in embedded systems and manufacturing.

/// Spindle speed, feed-rate, and material-removal-rate formulas for CNC machining.
pub mod cnc;
/// Baud-rate clock tolerance and frame-drift error budgets for asynchronous UART links.
pub mod uart;
/// Viswanath UPO reference trajectory and OGY Rayleigh-parameter EKF control primitives.
pub mod upo_control;

pub use cnc::*;
pub use uart::*;
pub use upo_control::*;

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
