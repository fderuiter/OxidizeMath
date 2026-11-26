//! # Body Mass Index (BMI)
//!
//! A function to calculate the Body Mass Index (BMI).

/// Calculates the Body Mass Index (BMI).
///
/// BMI is a measure of body fat based on height and weight that applies to adult men and women.
///
/// ## Formula
///
/// The formula for BMI is:
///
/// BMI = weight (kg) / height (m)^2
///
/// ## Parameters
///
/// * `weight_kg`: Weight in kilograms (kg).
/// * `height_m`: Height in meters (m).
///
/// ## Returns
///
/// The calculated BMI as a `f64`.
///
/// ## Example
///
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// let weight = 70.0; // kg
/// let height = 1.75; // m
/// use math_explorer::applied::win_ratio::bmi::calculate_bmi;
/// let bmi = calculate_bmi(weight, height);
/// assert!((bmi - 22.857).abs() < 1e-3);
/// ```
#[verified_engine::verified]
pub fn calculate_bmi(weight_kg: f64, height_m: f64) -> f64 {
    if height_m <= 0.0 {
        return 0.0;
    }
    weight_kg / (height_m * height_m)
}
