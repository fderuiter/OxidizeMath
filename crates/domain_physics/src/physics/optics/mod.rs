pub mod sensing;

// [cite:graph_parameters_rust]

use pure_math::theory_verification;

theory_verification!(
    module = "optics",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
