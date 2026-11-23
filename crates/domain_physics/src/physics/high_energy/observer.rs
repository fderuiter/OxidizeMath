use crate::error::HighEnergyError;
use math_commons::constants::{C, EPSILON};

/// A struct representing a Four-Vector (Time + 3-Space).
/// The metric signature is -+++.
#[derive(Debug, Clone, Copy, PartialEq)]
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

    /// Applies a 3D Lorentz boost with dimensionless velocity $\boldsymbol{\beta} = \mathbf{v}/c$
    /// (`|beta| < 1`) to homogeneous Minkowski components $(x^0, x^1, x^2, x^3)$.
    ///
    /// Preserves the quadratic Minkowski norm $-(x^0)^2 + x^2 + y^2 + z^2$.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidVelocity`] if any component is non-finite or $|\boldsymbol{\beta}| \ge 1$.
    #[verified_engine::verified]
    pub fn lorentz_boost(&self, beta_vec: [f64; 3]) -> Result<Self, HighEnergyError> {
        let [bx, by, bz] = beta_vec;
        if !self.t.is_finite()
            || !self.x.is_finite()
            || !self.y.is_finite()
            || !self.z.is_finite()
            || !bx.is_finite()
            || !by.is_finite()
            || !bz.is_finite()
        {
            return Err(HighEnergyError::InvalidVelocity { v: f64::NAN });
        }
        let beta_sq = bx * bx + by * by + bz * bz;
        if beta_sq >= 1.0 {
            return Err(HighEnergyError::InvalidVelocity {
                v: beta_sq.sqrt() * C,
            });
        }
        if beta_sq <= 1e-30 {
            return Ok(*self);
        }
        let gamma = 1.0 / (1.0 - beta_sq).sqrt();
        let b_dot_x = bx * self.x + by * self.y + bz * self.z;
        let parallel_coeff = ((gamma * gamma) / (gamma + 1.0)) * b_dot_x - gamma * self.t;
        Ok(Self {
            t: gamma * (self.t - b_dot_x),
            x: self.x + parallel_coeff * bx,
            y: self.y + parallel_coeff * by,
            z: self.z + parallel_coeff * bz,
        })
    }

    /// Computes the relativistic Doppler frequency ratio $\delta = \nu_{\text{obs}} / \nu_{\text{emit}}$
    /// for radiation propagating along spatial direction `(self.x, self.y, self.z)` from a source
    /// moving with dimensionless three-velocity `source_beta` ($\mathbf{v}_{\text{src}} / c$).
    ///
    /// Evaluates $\delta = \left[\gamma\left(1 - \boldsymbol{\beta}_{\text{src}}\cdot\hat{\mathbf{n}}\right)\right]^{-1}$
    /// by Lorentz-boosting the observer-frame null direction $(1, \hat{\mathbf{n}})$ into the source rest frame.
    ///
    /// # Errors
    /// Returns [`HighEnergyError::InvalidVelocity`] if inputs are non-finite, the spatial propagation
    /// vector has zero length, or $|\boldsymbol{\beta}_{\text{src}}| \ge 1$.
    #[verified_engine::verified]
    pub fn relativistic_doppler_factor(
        &self,
        source_beta: [f64; 3],
    ) -> Result<f64, HighEnergyError> {
        if !self.t.is_finite() || !self.x.is_finite() || !self.y.is_finite() || !self.z.is_finite()
        {
            return Err(HighEnergyError::InvalidVelocity { v: f64::NAN });
        }
        let norm_sq = self.x * self.x + self.y * self.y + self.z * self.z;
        if norm_sq <= EPSILON * EPSILON {
            return Err(HighEnergyError::InvalidVelocity { v: 0.0 });
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        let null_ray = Self::new(1.0, self.x * inv_norm, self.y * inv_norm, self.z * inv_norm);
        let boosted = null_ray.lorentz_boost(source_beta)?;
        if boosted.t <= EPSILON {
            return Err(HighEnergyError::InvalidVelocity { v: C });
        }
        Ok(1.0 / boosted.t)
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

    #[test]
    #[verified_engine::verified]
    fn test_lorentz_boost_preserves_minkowski_norm() {
        let p = FourVector::new(5.0, 3.0, 1.5, -2.0);
        let norm_orig = -p.t * p.t + p.x * p.x + p.y * p.y + p.z * p.z;
        let boosted = p
            .lorentz_boost([0.48, -0.36, 0.25])
            .expect("subluminal boost should succeed");
        let norm_boosted = -boosted.t * boosted.t
            + boosted.x * boosted.x
            + boosted.y * boosted.y
            + boosted.z * boosted.z;
        assert_relative_eq!(
            norm_orig,
            norm_boosted,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        let unboosted = boosted
            .lorentz_boost([-0.48, 0.36, -0.25])
            .expect("inverse boost should succeed");
        assert_relative_eq!(unboosted.t, p.t, epsilon = 1e-10);
        assert_relative_eq!(unboosted.x, p.x, epsilon = 1e-10);
        assert_relative_eq!(unboosted.y, p.y, epsilon = 1e-10);
        assert_relative_eq!(unboosted.z, p.z, epsilon = 1e-10);
    }

    #[test]
    #[verified_engine::verified]
    fn test_relativistic_doppler_longitudinal_and_transverse() {
        let los = FourVector::new(1.0, 1.0, 0.0, 0.0);
        let beta = 0.6;
        let blue = los
            .relativistic_doppler_factor([beta, 0.0, 0.0])
            .expect("approaching Doppler factor");
        let expected_blue = ((1.0 + beta) / (1.0 - beta)).sqrt();
        assert_relative_eq!(blue, expected_blue, epsilon = 1e-10);

        let red = los
            .relativistic_doppler_factor([-beta, 0.0, 0.0])
            .expect("receding Doppler factor");
        let expected_red = ((1.0 - beta) / (1.0 + beta)).sqrt();
        assert_relative_eq!(red, expected_red, epsilon = 1e-10);

        let transverse = los
            .relativistic_doppler_factor([0.0, beta, 0.0])
            .expect("transverse Doppler factor");
        assert_relative_eq!(transverse, 0.8, epsilon = 1e-10);

        assert!(los.lorentz_boost([0.8, 0.8, 0.0]).is_err());
        assert!(
            FourVector::new(1.0, 0.0, 0.0, 0.0)
                .relativistic_doppler_factor([0.2, 0.0, 0.0])
                .is_err()
        );
    }
}
