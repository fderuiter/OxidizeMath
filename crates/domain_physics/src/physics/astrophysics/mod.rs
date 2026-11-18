/// The astrophysics module contains implementations of astrophysical formulas.
pub mod galaxies;

// [cite:graph_parameters_rust]

use pure_math::theory_verification;

theory_verification!(
    module = "astrophysics",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
