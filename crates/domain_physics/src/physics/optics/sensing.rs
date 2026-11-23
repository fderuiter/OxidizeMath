//! Optical Sensing Models.
//!
//! Models for light-based sensors and propagation.

// [cite:optics]

/// Calculates intensity using the Lambertian Intensity Profile.
///
/// $$ I_{tx}(r_i, \phi_i) = \frac{(n + 1) P_{tx}}{2\pi r_i^2} \cos^n(\phi_i) $$
///
/// # Arguments
///
/// * `power_tx` ($P_{tx}$) - Transmitted optical power.
/// * `distance` ($r_i$) - Distance from source to point.
/// * `angle` ($\phi_i$) - Off-normal angle in radians.
/// * `mode_n` ($n$) - Radiation pattern mode number (related to half-power angle).
///
/// # Returns
///
/// * `f64` - Intensity at the point.
#[verified_engine::verified]
pub fn lambertian_intensity(power_tx: f64, distance: f64, angle: f64, mode_n: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0; // Singularity
    }
    let numerator = (mode_n + 1.0) * power_tx * angle.cos().powf(mode_n);
    let denominator = 2.0 * std::f64::consts::PI * distance.powi(2);

    numerator / denominator
}

/// Calculates LWS Range Resolution (ADC-Limited).
///
/// $$ \Delta r_{min} = r_0 - [(r_0 - \Delta r_{max})^{-4} 2^{B_{min}} + r_0^{-4}]^{-1/4} $$
///
/// Note: This is a complex formula involving inverse fourth-power path loss.
/// The prompt text: `r_0 - [(r_0 - Delta r_max)^-4 * 2^B_min + r_0^-4]^-1/4`
/// Wait, the prompt says:
/// `Delta r_min = r_0 - [(r_0 - Delta r_max)^-4 * 2^B_min + r_0^-4]^-1/4`
/// Actually, looking at the prompt: `r_0 - [(r_0 - Delta r_max)^-4 2^{B_{min}} + r_0^{-4}]^{-1/4}`.
///
/// Let's implement exactly as written.
///
/// # Arguments
///
/// * `r0` ($r_0$) - Nominal range.
/// * `delta_r_max` ($\Delta r_{max}$) - Maximum displacement range (full scale of sensor?).
/// * `bits` ($B_{min}$) - Number of ADC bits.
///
/// # Returns
///
/// * `f64` - Minimum resolvable displacement ($\Delta r_{min}$).
#[verified_engine::verified]
pub fn lws_range_resolution(r0: f64, delta_r_max: f64, bits: u32) -> f64 {
    let term1 = (r0 - delta_r_max).powi(-4);
    let term2 = 2.0_f64.powi(bits as i32);
    let term3 = r0.powi(-4);

    let inner = term1 * term2 + term3;
    let bracket = inner.powf(-0.25); // ^-1/4

    r0 - bracket
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_lambertian() {
        // n=1 (Lambertian), phi=0 (cos=1), P=2pi, r=1
        // I = (2 * 2pi) / (2pi * 1) = 2
        let i = lambertian_intensity(2.0 * std::f64::consts::PI, 1.0, 0.0, 1.0);
        assert!((i - 2.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }
}
