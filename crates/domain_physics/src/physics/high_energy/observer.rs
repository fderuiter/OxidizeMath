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

    /// Applies a Lorentz boost along the $+z$ axis with dimensionless velocity $\beta = v / c$.
    ///
    /// Transforms contravariant components $(x^0, x, y, z) \mapsto (\gamma(x^0 - \beta z), x, y, \gamma(z - \beta x^0))$
    /// where `self.t` represents $x^0$ (e.g., $E/c$ or $ct$).
    ///
    /// # Errors
    /// * [`HighEnergyError::InvalidVelocity`] if `|beta| >= 1.0` or inputs are non-finite.
    #[verified_engine::verified]
    pub fn lorentz_boost_z(&self, beta: f64) -> Result<Self, HighEnergyError> {
        let gamma = calculate_lorentz_factor_from_beta(beta)?;
        if ![self.t, self.x, self.y, self.z]
            .into_iter()
            .all(f64::is_finite)
        {
            return Err(HighEnergyError::InvalidMomentum);
        }
        Ok(Self {
            t: gamma * (self.t - beta * self.z),
            x: self.x,
            y: self.y,
            z: gamma * (self.z - beta * self.t),
        })
    }

    /// Applies a general 3D Lorentz boost with dimensionless velocity vector $\vec{\beta} = \vec{v} / c$.
    ///
    /// Assumes `self.t` stores $x^0$ in the same units as spatial components $(x, y, z)$ (such as
    /// $(E/c, \vec{p})$ or $(ct, \vec{r})$).
    ///
    /// # Errors
    /// * [`HighEnergyError::InvalidVelocity`] if $|\vec{\beta}| \ge 1.0$ or inputs are non-finite.
    #[verified_engine::verified]
    pub fn lorentz_boost(&self, beta_vec: [f64; 3]) -> Result<Self, HighEnergyError> {
        if !beta_vec.into_iter().all(f64::is_finite)
            || ![self.t, self.x, self.y, self.z]
                .into_iter()
                .all(f64::is_finite)
        {
            return Err(HighEnergyError::InvalidVelocity { v: f64::NAN });
        }
        let beta_sq =
            beta_vec[0] * beta_vec[0] + beta_vec[1] * beta_vec[1] + beta_vec[2] * beta_vec[2];
        if beta_sq >= 1.0 {
            return Err(HighEnergyError::InvalidVelocity {
                v: beta_sq.sqrt() * C,
            });
        }
        if beta_sq <= EPSILON * EPSILON {
            return Ok(*self);
        }
        let gamma = 1.0 / (1.0 - beta_sq).sqrt();
        let b_dot_r = beta_vec[0] * self.x + beta_vec[1] * self.y + beta_vec[2] * self.z;
        let parallel_scale = ((gamma - 1.0) / beta_sq) * b_dot_r - gamma * self.t;
        Ok(Self {
            t: gamma * (self.t - b_dot_r),
            x: self.x + parallel_scale * beta_vec[0],
            y: self.y + parallel_scale * beta_vec[1],
            z: self.z + parallel_scale * beta_vec[2],
        })
    }
}

/// Calculates the Lorentz factor $\gamma = (1 - \beta^2)^{-1/2}$ from dimensionless speed $\beta = v/c$.
///
/// # Errors
/// * [`HighEnergyError::InvalidVelocity`] if `|beta| >= 1.0` or `beta` is non-finite.
#[verified_engine::verified]
pub fn calculate_lorentz_factor_from_beta(beta: f64) -> Result<f64, HighEnergyError> {
    if !beta.is_finite() || beta.abs() >= 1.0 {
        return Err(HighEnergyError::InvalidVelocity { v: beta * C });
    }
    Ok(1.0 / (1.0 - beta * beta).sqrt())
}

/// Calculates the Lorentz Factor gamma.
/// Formula: gamma = 1 / sqrt(1 - beta^2) where beta = v/c.
///
/// # Errors
/// * `HighEnergyError::InvalidVelocity` if `|v| >= c`.
#[verified_engine::verified]
pub fn calculate_lorentz_factor(v: f64) -> Result<f64, HighEnergyError> {
    calculate_lorentz_factor_from_beta(v / C)
}

/// Calculates the kinematic relativistic Doppler factor
/// $$D(\beta, \theta) = \frac{1}{\gamma (1 - \beta \cos\theta)}$$
/// for an emitter moving at dimensionless bulk speed $\beta = v / c$ at angle `viewing_angle_rad`
/// relative to the observer's line of sight ($\theta = 0$ approaching, $\theta = \pi$ receding).
///
/// # Errors
/// * [`HighEnergyError::InvalidVelocity`] if `|beta| >= 1.0` or inputs are non-finite.
#[verified_engine::verified]
pub fn relativistic_doppler_factor(
    beta: f64,
    viewing_angle_rad: f64,
) -> Result<f64, HighEnergyError> {
    if !viewing_angle_rad.is_finite() {
        return Err(HighEnergyError::InvalidVelocity { v: f64::NAN });
    }
    let gamma = calculate_lorentz_factor_from_beta(beta)?;
    let denom = gamma * (1.0 - beta * viewing_angle_rad.cos());
    if denom <= 0.0 || !denom.is_finite() {
        return Err(HighEnergyError::InvalidVelocity { v: beta * C });
    }
    Ok(1.0 / denom)
}

/// Calculates apparent transverse speed $\beta_{\text{app}} = \frac{\beta \sin\theta}{1 - \beta \cos\theta}$
/// in units of $c$ (exhibiting apparent superluminal motion $\beta_{\text{app}} > 1$ when $\beta > 1/\sqrt{2}$).
///
/// # Errors
/// * [`HighEnergyError::InvalidVelocity`] if `|beta| >= 1.0` or inputs are non-finite.
#[verified_engine::verified]
pub fn apparent_superluminal_beta(
    beta: f64,
    viewing_angle_rad: f64,
) -> Result<f64, HighEnergyError> {
    if !viewing_angle_rad.is_finite() {
        return Err(HighEnergyError::InvalidVelocity { v: f64::NAN });
    }
    let _ = calculate_lorentz_factor_from_beta(beta)?;
    let (sin_t, cos_t) = viewing_angle_rad.sin_cos();
    let denom = 1.0 - beta * cos_t;
    if denom <= 0.0 || !denom.is_finite() {
        return Err(HighEnergyError::InvalidVelocity { v: beta * C });
    }
    Ok((beta * sin_t).abs() / denom)
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
    fn test_lorentz_boosts_preserve_interval_and_doppler_factor() {
        let rest_p = FourVector::new(2.0 * C, 0.3 * C, -0.4 * C, 0.5 * C);
        let p_sq_before = -(rest_p.t * rest_p.t)
            + rest_p.x * rest_p.x
            + rest_p.y * rest_p.y
            + rest_p.z * rest_p.z;

        let boosted_z = rest_p.lorentz_boost_z(-0.8).expect("z boost");
        let p_sq_z = -(boosted_z.t * boosted_z.t)
            + boosted_z.x * boosted_z.x
            + boosted_z.y * boosted_z.y
            + boosted_z.z * boosted_z.z;
        assert_relative_eq!(p_sq_z, p_sq_before, max_relative = 1e-12);

        let boosted_3d = rest_p.lorentz_boost([0.3, -0.4, 0.5]).expect("3d boost");
        let p_sq_3d = -(boosted_3d.t * boosted_3d.t)
            + boosted_3d.x * boosted_3d.x
            + boosted_3d.y * boosted_3d.y
            + boosted_3d.z * boosted_3d.z;
        assert_relative_eq!(p_sq_3d, p_sq_before, max_relative = 1e-12);

        let theta = 15.0_f64.to_radians();
        let d_app = relativistic_doppler_factor(0.9, theta).expect("approaching Doppler");
        let d_rec = relativistic_doppler_factor(0.9, std::f64::consts::PI - theta)
            .expect("receding Doppler");
        assert_relative_eq!(d_app / d_rec, 14.306_111_944_788_727, epsilon = 1e-10);
        let b_app = apparent_superluminal_beta(0.9, theta).expect("superluminal beta");
        assert!(
            b_app > 1.7,
            "expected superluminal beta_app > 1.7, got {b_app}"
        );
    }
}
