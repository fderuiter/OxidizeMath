//! Thermal Physics for Medical Applications.
//!
//! Models heat transfer in biological tissues and dielectric properties of skin.

/// Calculates the temperature change rate using the 1-D Bio-Heat Transfer Equation.
///
/// $$ \rho c \frac{\partial T}{\partial t} = K \frac{\partial^2 T}{\partial z^2} + Q $$
///
/// This function calculates the time derivative of temperature ($\frac{\partial T}{\partial t}$)
/// given the spatial gradients and material properties.
///
/// # Arguments
///
/// * `density` ($\rho$) - Tissue density (kg/m^3).
/// * `specific_heat` ($c$) - Specific heat capacity (J/(kg·K)).
/// * `thermal_conductivity` ($K$) - Thermal conductivity (W/(m·K)).
/// * `second_derivative_T` ($\frac{\partial^2 T}{\partial z^2}$) - Curvature of temperature profile (K/m^2).
/// * `heat_deposition` ($Q$) - Volumetric heat generation rate (W/m^3).
///
/// # Returns
///
/// * `f64` - The rate of temperature change (K/s).
#[verified_engine::verified]
pub fn bio_heat_transfer_rate(
    density: f64,
    specific_heat: f64,
    thermal_conductivity: f64,
    second_derivative_t: f64,
    heat_deposition: f64,
) -> f64 {
    let term1 = thermal_conductivity * second_derivative_t;
    let total_heat_flux = term1 + heat_deposition;
    let heat_capacity = density * specific_heat;

    if heat_capacity <= 0.0 {
        return 0.0; // Avoid division by zero or invalid physics
    }

    total_heat_flux / heat_capacity
}

/// Calculates complex permittivity using the Debye Equation.
///
/// Models the dielectric relaxation of tissues (e.g., skin) at mmWave frequencies.
///
/// $$ \epsilon^* = \epsilon_{\infty} + \frac{\epsilon_s - \epsilon_{\infty}}{1 + j\omega\tau} + \frac{\sigma_i}{j\omega\epsilon_0} $$
///
/// # Arguments
///
/// * `epsilon_inf` ($\epsilon_{\infty}$) - Permittivity at high frequency.
/// * `epsilon_s` ($\epsilon_s$) - Static permittivity.
/// * `relaxation_time` ($\tau$) - Relaxation time constant (s).
/// * `sigma_i` ($\sigma_i$) - Ionic conductivity (S/m).
/// * `frequency` ($f$) - Frequency in Hz ($\omega = 2\pi f$).
///
/// # Returns
///
/// * `(f64, f64)` - The complex permittivity (Real part $\epsilon'$, Imaginary part $\epsilon''$).
///   Note: The term $\frac{1}{j}$ is $-j$, so the conductivity term contributes to the imaginary part.
#[verified_engine::verified]
pub fn debye_permittivity(
    epsilon_inf: f64,
    epsilon_s: f64,
    relaxation_time: f64,
    sigma_i: f64,
    frequency: f64,
) -> (f64, f64) {
    let omega = 2.0 * std::f64::consts::PI * frequency;
    let epsilon_0 = 8.854_187_817e-12; // Vacuum permittivity

    // Debye term: (es - einf) / (1 + j*w*t)
    // Multiply by conjugate (1 - j*w*t) / (1 + (w*t)^2)
    let wt = omega * relaxation_time;
    let denom = 1.0 + wt.powi(2);
    let delta_eps = epsilon_s - epsilon_inf;

    let debye_real = delta_eps / denom;
    let debye_imag = -(delta_eps * wt) / denom; // Negative because 1/(1+j) has -j component

    // Conductivity term: sigma / (j * w * eps0) = -j * sigma / (w * eps0)
    let conductivity_imag = -sigma_i / (omega * epsilon_0);

    let real_part = epsilon_inf + debye_real;
    let imag_part = debye_imag + conductivity_imag;

    (real_part, imag_part)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_bio_heat() {
        let rho = 1000.0;
        let c = 4000.0;
        let k = 0.5;
        let d2t = 10.0;
        let q = 100.0;

        // dT/dt = (0.5*10 + 100) / (1000*4000) = 105 / 4e6
        let rate = bio_heat_transfer_rate(rho, c, k, d2t, q);
        assert!((rate - 2.625e-5).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }
}
