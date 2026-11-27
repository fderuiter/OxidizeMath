use super::types::*;

/// Calculates the Q-value of a reaction.
///
/// Formula: Q = (sum(m_in) - sum(m_out)) * c^2
///
/// # Arguments
/// * `input_masses` - Slice of input masses in MeV/c^2.
/// * `output_masses` - Slice of output masses in MeV/c^2.
#[verified_engine::verified]
pub fn q_value(input_masses: &[f64], output_masses: &[f64]) -> f64 {
    let sum_in: f64 = input_masses.iter().sum();
    let sum_out: f64 = output_masses.iter().sum();
    sum_in - sum_out
}

/// Calculates the Breit-Wigner cross-section shape.
///
/// Formula: sigma(E) ~ Gamma^2 / ((E - E_res)^2 + Gamma^2/4)
///
/// # Arguments
/// * `energy` - Energy E in MeV.
/// * `resonance_energy` - Resonance energy E_res in MeV.
/// * `gamma_width` - Decay width Gamma in MeV.
#[verified_engine::verified]
pub fn breit_wigner(
    energy: f64,
    resonance_energy: f64,
    gamma_width: f64,
) -> Result<f64, NuclearError> {
    if gamma_width <= 0.0 {
        return Err(NuclearError::InvalidGammaWidth);
    }
    let numerator = gamma_width.powi(2);
    let denominator = (energy - resonance_energy).powi(2) + (gamma_width.powi(2) / 4.0);
    Ok(numerator / denominator)
}
