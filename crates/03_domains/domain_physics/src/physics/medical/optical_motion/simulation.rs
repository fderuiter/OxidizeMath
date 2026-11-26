//! Simulation Models for Optical Motion.
//!
//! Provides mathematical models to simulate respiratory motion and mechanical translation
//! for validation and testing purposes.

/// Generates a respiratory signal using the Cosine Respiratory Curve Model.
///
/// $$ Z(t) = -b \cdot \cos(6\pi t / \tau + \pi / 2) $$
///
/// # Arguments
///
/// * `t` - Time in seconds.
/// * `amplitude` ($b$) - Amplitude of the breath.
/// * `period` ($\tau$) - Period of the respiratory cycle.
///
/// # Returns
///
/// * `f64` - The simulated height $Z(t)$.
#[verified_engine::verified]
pub fn cosine_respiratory_curve(t: f64, amplitude: f64, period: f64) -> f64 {
    // Note: The formula uses 6*pi, which implies 3 cycles per 'tau' if tau is the standard period?
    // Usually standard is 2*pi*t/T.
    // If tau is "period of respiratory curve", then usually 2*pi.
    // The prompt specifies: 6*pi*t / tau.
    // We strictly follow the prompt.
    let phase = (6.0 * std::f64::consts::PI * t) / period + std::f64::consts::FRAC_PI_2;
    -amplitude * phase.cos()
}

/// Models the motion of a translation stage with constant acceleration.
///
/// Used for determining beam-on/off time delays.
///
/// $$ y_1 = a \cdot (t - T)^2 + b $$
///
/// # Arguments
///
/// * `t` - Time.
/// * `half_period` ($T$) - Half the movement period.
/// * `coefficient_a` ($a$) - Acceleration coefficient.
/// * `coefficient_b` ($b$) - Offset.
///
/// # Returns
///
/// * `f64` - Position $y_1$.
#[verified_engine::verified]
pub fn translation_stage_motion(
    t: f64,
    half_period: f64,
    coefficient_a: f64,
    coefficient_b: f64,
) -> f64 {
    coefficient_a * (t - half_period).powi(2) + coefficient_b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_cosine_model() {
        // t=0, phase = pi/2. cos(pi/2) = 0. Z = 0.
        let z0 = cosine_respiratory_curve(0.0, 1.0, 5.0);
        assert!(z0.abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_translation_motion() {
        // t=T -> y = b
        let y = translation_stage_motion(5.0, 5.0, 2.0, 10.0);
        assert_eq!(y, 10.0);
    }
}
