use crate::error::HighEnergyError;
use math_commons::constants::{C, EPSILON};

/// A struct representing a Four-Vector (Time + 3-Space).
/// The metric signature is -+++.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FourVector {
    /// Time component (e.g., t or ct, depending on context).
    /// For `invariant_interval`, this is interpreted as time `t` (seconds).
    /// For `is_valid_momentum`, this is interpreted as energy component `E/c`.
    pub t: f64,
    /// Spatial X component.
    pub x: f64,
    /// Spatial Y component.
    pub y: f64,
    /// Spatial Z component.
    pub z: f64,
}

impl FourVector {
    /// Creates a new FourVector.
    #[verified_engine::verified]
    pub fn new(t: f64, x: f64, y: f64, z: f64) -> Self {
        Self { t, x, y, z }
    }

    /// Calculates the spacetime interval s^2.
    /// Formula: s^2 = -(ct)^2 + x^2 + y^2 + z^2.
    /// This assumes `self.t` is time in seconds.
    #[verified_engine::verified]
    pub fn invariant_interval(&self) -> Result<f64, HighEnergyError> {
        let ct = C * self.t;
        Ok(-(ct * ct) + self.x * self.x + self.y * self.y + self.z * self.z)
    }

    /// Validates 4-Momentum Invariance.
    /// Checks if p^mu p_mu = -(mc)^2.
    ///
    /// Assumes the vector represents 4-momentum p^mu = (E/c, px, py, pz).
    /// Note: The input `mass` is the rest mass `m`.
    ///
    /// # Errors
    /// * `HighEnergyError::InvalidMass` if `mass < 0`.
    #[verified_engine::verified]
    pub fn is_valid_momentum(&self, mass: f64) -> Result<bool, HighEnergyError> {
        if mass < 0.0 {
            return Err(HighEnergyError::InvalidMass { mass });
        }
        // Contraction p^mu p_mu = - (p^0)^2 + (p^1)^2 + ...
        // Here self.t is p^0 (E/c).
        let p_sq = -(self.t * self.t) + self.x * self.x + self.y * self.y + self.z * self.z;
        let target = -(mass * C).powi(2);

        // Using relative error for robustness with large numbers, or absolute for small.
        let diff = (p_sq - target).abs();
        let tolerance = 1e-5; // Tolerance for floating point comparison

        if target.abs() > EPSILON {
            Ok(diff / target.abs() < tolerance)
        } else {
            // Mass is effectively 0 (photon). p_sq should be 0.
            Ok(diff < tolerance)
        }
    }
}

/// Calculates the Lorentz Factor gamma.
/// Formula: gamma = 1 / sqrt(1 - beta^2) where beta = v/c.
///
/// # Errors
/// * `HighEnergyError::InvalidVelocity` if `|v| >= c`.
#[verified_engine::verified]
pub fn calculate_lorentz_factor(v: f64) -> Result<f64, HighEnergyError> {
    let beta = v / C;
    if beta.abs() >= 1.0 {
        return Err(HighEnergyError::InvalidVelocity { v });
    }
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();
    Ok(gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_special_relativity() {
        // Lorentz Factor: v = 0.6c => gamma = 1.25
        let v = 0.6 * C;
        let gamma = calculate_lorentz_factor(v).expect("Failed to calc Lorentz factor");
        assert_relative_eq!(
            gamma,
            1.25,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        // Invariant Interval: ct=3, x=4, y=0, z=0 => s^2 = -9 + 16 = 7
        // t = 3/c.
        let fv = FourVector::new(3.0 / C, 4.0, 0.0, 0.0);
        let s2 = fv.invariant_interval().expect("Failed to calc interval");
        assert_relative_eq!(s2, 7.0, epsilon = math_commons::registry::TOLERANCE_FAST);

        // Momentum Invariance
        // m=1. p^mu = (gamma mc, gamma mv, 0, 0) / c?
        // p^0 = E/c = gamma m c.
        // p^1 = gamma m v.
        // p.p = -(gamma m c)^2 + (gamma m v)^2 = gamma^2 m^2 (v^2 - c^2) = gamma^2 m^2 (-c^2/gamma^2) = -m^2 c^2.
        // Let's use m=1, v=0.6c, gamma=1.25.
        // p^0 = 1.25 * 1 * C.
        // p^1 = 1.25 * 1 * 0.6 * C = 0.75 * C.
        let mass = 1.0;
        // Wait, FourVector.t is just the value.
        // If p^mu = (E/c, px, py, pz).
        // E/c = 1.25 * C.
        // px = 0.75 * C.
        let p_vec = FourVector::new(1.25 * C, 0.75 * C, 0.0, 0.0);
        assert!(
            p_vec
                .is_valid_momentum(mass)
                .expect("Failed to check momentum")
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_errors() {
        assert!(calculate_lorentz_factor(C).is_err());
        assert!(calculate_lorentz_factor(1.1 * C).is_err());
        let fv = FourVector::new(0.0, 0.0, 0.0, 0.0);
        assert!(fv.is_valid_momentum(-1.0).is_err());
    }
}
