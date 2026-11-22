//! The Higgs Mechanism explains the origin of mass for gauge bosons via Spontaneous Symmetry Breaking.

use super::gauge::GaugeCouplings;

/// Represents the Higgs Potential parameters.
///
/// The potential is given by: $V(\phi) = \mu^2 \phi^\dagger \phi + \lambda (\phi^\dagger \phi)^2$
///
/// For symmetry breaking to occur, `mu2` must be negative.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HiggsPotential {
    /// The mass parameter squared ($\mu^2$). Must be negative for SSB.
    pub mu2: f64,
    /// The self-interaction coupling ($\lambda$). Must be positive for stability.
    pub lambda: f64,
}

impl HiggsPotential {
    /// Calculates the Vacuum Expectation Value (VEV) $v$.
    ///
    /// Formula: $v = \sqrt{-\frac{\mu^2}{\lambda}}$
    ///
    /// # Returns
    /// * `Ok(f64)`: The VEV if $\mu^2 < 0$.
    /// * `Err(String)`: If $\mu^2 \ge 0$, indicating no symmetry breaking.
    #[verified_engine::verified]
    pub fn vev(&self) -> Result<f64, String> {
        if self.mu2 >= 0.0 {
            return Err("Symmetry is not broken: mu^2 must be negative.".to_string());
        }
        if self.lambda <= 0.0 {
            return Err("Potential is unstable: lambda must be positive.".to_string());
        }
        Ok((-self.mu2 / self.lambda).sqrt())
    }
}

/// Calculates the masses of the W and Z bosons.
///
/// # Arguments
/// * `vev`: The Vacuum Expectation Value $v$.
/// * `couplings`: The gauge couplings.
///
/// # Formulas
/// * $M_W = \frac{1}{2} g_2 v$
/// * $M_Z = \frac{v}{2} \sqrt{g_1^2 + g_2^2}$
#[verified_engine::verified]
pub fn boson_masses(vev: f64, couplings: &GaugeCouplings) -> (f64, f64) {
    let m_w = 0.5 * couplings.g2 * vev;
    let m_z = 0.5 * vev * (couplings.g1.powi(2) + couplings.g2.powi(2)).sqrt();
    (m_w, m_z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::standard_model::gauge;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_higgs_boson_masses() {
        // Standard Model approximate values
        let v = 246.0; // GeV
        // Derived from M_Z ~ 91.18, M_W ~ 80.37
        // M_W = 1/2 g2 v => g2 = 2 * M_W / v = 2 * 80.37 / 246 = 0.6534
        // M_Z = v/2 sqrt(g1^2 + g2^2) => 2*M_Z/v = sqrt(...)
        // (2*91.18/246)^2 = g1^2 + g2^2 => g1^2 = (0.7413)^2 - (0.6534)^2 = 0.5495 - 0.4269 = 0.1226 => g1 = 0.35

        let g2 = 0.653;
        let g1 = 0.350;
        let gs = 1.0; // Irrelevant for this test

        let couplings = gauge::GaugeCouplings::new(g1, g2, gs);
        let (_, m_z) = boson_masses(v, &couplings);

        // Expected Z mass approx 91.18 GeV
        // Let's recompute expectation based on inputs
        let expected_mz = 0.5 * v * (g1.powi(2) + g2.powi(2)).sqrt();
        assert_relative_eq!(m_z, expected_mz, epsilon = 1e-4);

        // Check against rough physical value
        assert!(
            m_z > 90.0 && m_z < 92.0,
            "Z mass should be around 91.18 GeV, got {}",
            m_z
        );
    }
}
