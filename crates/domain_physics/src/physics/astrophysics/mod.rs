/// The astrophysics module contains empirical galaxy models and gravitational N-body choreographies.
pub mod galaxies;
pub mod nbody;

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
