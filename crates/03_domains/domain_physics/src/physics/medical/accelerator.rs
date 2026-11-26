/// Calculates the average energy for the Beam Loading Line.
///
/// $$ E = 5.925 - I_b \times 0.00808 $$
///
/// # Arguments
///
/// * `beam_current` ($I_b$) - Beam current in mA.
///
/// # Returns
///
/// * `f64` - Average energy in MeV.
#[verified_engine::verified]
pub fn beam_loading_energy(beam_current: f64) -> f64 {
    5.925 - beam_current * 0.00808
}
