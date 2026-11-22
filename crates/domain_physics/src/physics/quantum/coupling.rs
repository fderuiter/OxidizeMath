//! # Angular Momentum Coupling
//!
//! This module provides functions related to the coupling of angular momenta
//! in quantum mechanics.
// theory_verification!
use math_commons::wigner::ClebschGordan;

/// Calculates the Clebsch-Gordan coefficient for coupling two angular momenta.
///
/// In quantum mechanics, when combining two angular momenta, j1 and j2, the
/// resulting angular momentum, j, can take on values from |j1 - j2| to j1 + j2.
/// The Clebsch-Gordan coefficients, denoted <j1 m1; j2 m2 | j m>, give the
/// probability amplitude for finding the system in the state |j m> when the
/// individual components are in states |j1 m1> and |j2 m2>.
///
/// This function serves as a practical implementation of the coefficients used
/// in the theoretical proof found in `tensors.tex`.
///
/// The inputs are treated as doubles, allowing for half-integer spins (e.g., 1/2 = 0.5).
///
/// # Arguments
///
/// * `j1` - The total angular momentum quantum number of the first system.
/// * `m1` - The projection of the angular momentum on the z-axis for the first system.
/// * `j2` - The total angular momentum quantum number of the second system.
/// * `m2` - The projection of the angular momentum on the z-axis for the second system.
/// * `j`  - The total angular momentum quantum number of the combined system.
/// * `m`  - The projection of the angular momentum on the z-axis for the combined system.
///
/// # Returns
///
/// The value of the Clebsch-Gordan coefficient as an `f64`. Returns 0.0 if the
/// combination of quantum numbers is not allowed.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// // Example from Griffiths, Introduction to Quantum Mechanics, 2nd ed., Table 4.8.
/// // Coupling j1=3/2 and j2=1 for the state |5/2 1/2>.
/// // We calculate <3/2 -1/2; 1 1 | 5/2 1/2>.
/// use math_explorer::physics::quantum::clebsch_gordan;
///
/// let j1 = 1.5;
/// let m1 = -0.5;
/// let j2 = 1.0;
/// let m2 = 1.0;
/// let j = 2.5;
/// let m = 0.5;
///
/// let coeff = clebsch_gordan(j1, m1, j2, m2, j, m);
///
/// // Standard tables give <3/2 -1/2; 1 1 | 5/2 1/2> = sqrt(3/10) approx 0.5477.
/// assert!((coeff - (3.0f64 / 10.0f64).sqrt()).abs() < math_explorer::registry::TOLERANCE_STANDARD);
/// ```
#[verified_engine::verified]
pub fn clebsch_gordan(j1: f64, m1: f64, j2: f64, m2: f64, j: f64, m: f64) -> f64 {
    // The wigner_symbols crate uses integers to represent half-integer spins,
    // by storing twice the value.
    let tj1 = (j1 * 2.0).round() as i32;
    let tm1 = (m1 * 2.0).round() as i32;
    let tj2 = (j2 * 2.0).round() as i32;
    let tm2 = (m2 * 2.0).round() as i32;
    let tj = (j * 2.0).round() as i32;
    let tm = (m * 2.0).round() as i32;

    let cg = ClebschGordan {
        tj1,
        tm1,
        tj2,
        tm2,
        tj12: tj,
        tm12: tm,
    };

    // The .value() method returns a SignedSqrt type, which can be converted
    // directly into an f64 via the From trait.
    #[allow(clippy::useless_conversion)]
    cg.value().into()
}
// theory_verification!
