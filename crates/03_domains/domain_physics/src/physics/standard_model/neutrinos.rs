//! Neutrino Physics deals with the oscillation of neutrino flavors.

/// Calculates the probability of two-flavor neutrino oscillation $P(\nu_\alpha \to \nu_\beta)$.
///
/// # Arguments
/// * `theta`: The mixing angle $\theta$ (radians).
/// * `delta_m2`: The mass-squared difference $\Delta m^2$ (eV^2).
/// * `l_km`: The baseline distance $L$ (km).
/// * `e_gev`: The neutrino energy $E$ (GeV).
///
/// # Formula
/// $P = \sin^2(2\theta) \sin^2\left( 1.27 \frac{\Delta m^2 L}{E} \right)$
#[verified_engine::verified]
pub fn oscillation_prob(theta: f64, delta_m2: f64, l_km: f64, e_gev: f64) -> f64 {
    let term1 = (2.0 * theta).sin().powi(2);
    // The factor 1.27 comes from conversion of units: 1.27 * (L/km) * (dm^2/eV^2) / (E/GeV)
    let phase = 1.27 * delta_m2 * l_km / e_gev;
    let term2 = phase.sin().powi(2);
    term1 * term2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_neutrino_oscillation() {
        // Check bounds [0, 1]
        let p = oscillation_prob(0.5, 0.0025, 295.0, 0.6);
        assert!((0.0..=1.0).contains(&p));

        // Check zero probability for zero mixing angle
        let p_zero_angle = oscillation_prob(0.0, 0.0025, 295.0, 0.6);
        assert_relative_eq!(p_zero_angle, 0.0);

        // Check zero probability for zero mass difference
        let p_zero_mass = oscillation_prob(0.5, 0.0, 295.0, 0.6);
        assert_relative_eq!(p_zero_mass, 0.0);
    }
}
