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
    /// Computes the Minkowski norm squared $\eta_{\mu\nu} V^\mu V^\nu = -(V^0)^2 + V_x^2 + V_y^2 + V_z^2$
    /// in signature `(-, +, +, +)` where `self.t` is $V^0$ (e.g., $E/c$ or $ct$).
    #[verified_engine::verified]
    pub fn minkowski_norm_squared(&self) -> f64 {
        -(self.t * self.t) + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Boosts a homogeneous four-vector $V^\mu = (V^0, \mathbf{V})$ (such as 4-momentum $(E/c, \mathbf{p})$)
    /// into an observer frame moving at dimensionless velocity $\vec{\beta} = (\beta_x, \beta_y, \beta_z)$
    /// with $|\vec{\beta}| < 1$.
    ///
    /// Uses $\frac{\gamma - 1}{\beta^2} = \frac{\gamma^2}{\gamma + 1}$ to avoid $0/0$ cancellation at $\vec{\beta} = \mathbf{0}$.
    #[verified_engine::verified]
    pub fn lorentz_boost(
        &self,
        beta_x: f64,
        beta_y: f64,
        beta_z: f64,
    ) -> Result<Self, HighEnergyError> {
        let all_finite = self.t.is_finite()
            && self.x.is_finite()
            && self.y.is_finite()
            && self.z.is_finite()
            && beta_x.is_finite()
            && beta_y.is_finite()
            && beta_z.is_finite();
        if !all_finite {
            return Err(HighEnergyError::InvalidVelocity { v: f64::NAN });
        }
        let beta_sq = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z;
        if beta_sq >= 1.0 {
            return Err(HighEnergyError::InvalidVelocity {
                v: beta_sq.sqrt() * C,
            });
        }
        let gamma = 1.0 / (1.0 - beta_sq).sqrt();
        let beta_dot_r = beta_x * self.x + beta_y * self.y + beta_z * self.z;
        let transverse_scale = (gamma * gamma / (gamma + 1.0)) * beta_dot_r - gamma * self.t;
        Ok(Self {
            t: gamma * (self.t - beta_dot_r),
            x: self.x + transverse_scale * beta_x,
            y: self.y + transverse_scale * beta_y,
            z: self.z + transverse_scale * beta_z,
        })
    }

    /// Boosts a spacetime event $(t\ [\text{s}], x, y, z\ [\text{m}])$ by physical 3-velocity $(v_x, v_y, v_z)$ ($\text{m/s}$),
    /// preserving [`FourVector::invariant_interval`].
    #[verified_engine::verified]
    pub fn lorentz_boost_spacetime(
        &self,
        vx: f64,
        vy: f64,
        vz: f64,
    ) -> Result<Self, HighEnergyError> {
        let boosted =
            Self::new(C * self.t, self.x, self.y, self.z).lorentz_boost(vx / C, vy / C, vz / C)?;
        Ok(Self::new(boosted.t / C, boosted.x, boosted.y, boosted.z))
    }
}

/// Calculates the Lorentz Factor gamma.
/// Formula: gamma = 1 / sqrt(1 - beta^2) where beta = v/c.
///
/// # Errors
/// * `HighEnergyError::InvalidVelocity` if `|v| >= c`.
#[verified_engine::verified]
pub fn calculate_lorentz_factor(v: f64) -> Result<f64, HighEnergyError> {
    if !v.is_finite() {
        return Err(HighEnergyError::InvalidVelocity { v });
    }
    let beta = v / C;
    if beta.abs() >= 1.0 {
        return Err(HighEnergyError::InvalidVelocity { v });
    }
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();
    Ok(gamma)
}

/// Computes the relativistic Doppler factor $\delta = \frac{1}{\gamma (1 - \beta \cos\theta)}$
/// for dimensionless emitter speed $\beta = v/c \in (-1, 1)$ and line-of-sight angle $\theta$ (radians,
/// where $\theta = 0$ is head-on approach).
#[verified_engine::verified]
pub fn relativistic_doppler_factor(
    beta: f64,
    viewing_angle_rad: f64,
) -> Result<f64, HighEnergyError> {
    if !beta.is_finite() || beta.abs() >= 1.0 {
        return Err(HighEnergyError::InvalidVelocity { v: beta * C });
    }
    if !viewing_angle_rad.is_finite() {
        return Err(HighEnergyError::InvalidViewingAngle {
            angle_rad: viewing_angle_rad,
        });
    }
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();
    Ok(1.0 / (gamma * (1.0 - beta * viewing_angle_rad.cos())))
}

/// Computes the apparent transverse speed $\beta_{\text{app}} = \frac{\beta \sin\theta}{1 - \beta \cos\theta}$
/// (in units of $c$) for a relativistic jet component, exhibiting apparent superluminal motion when $\beta > 1/\sqrt{2}$.
#[verified_engine::verified]
pub fn apparent_transverse_beta(beta: f64, viewing_angle_rad: f64) -> Result<f64, HighEnergyError> {
    if !beta.is_finite() || beta.abs() >= 1.0 {
        return Err(HighEnergyError::InvalidVelocity { v: beta * C });
    }
    if !viewing_angle_rad.is_finite() {
        return Err(HighEnergyError::InvalidViewingAngle {
            angle_rad: viewing_angle_rad,
        });
    }
    Ok((beta * viewing_angle_rad.sin()) / (1.0 - beta * viewing_angle_rad.cos()))
}

/// Computes the relativistic Doppler flux-beaming amplification $\delta^{k + \alpha}$
/// where $k = 2$ for a steady continuous jet (`continuous_jet = true`) and $k = 3$ for a discrete
/// moving plasmoid (`continuous_jet = false`), for power-law spectrum $S_\nu \propto \nu^{-\alpha}$.
#[verified_engine::verified]
pub fn doppler_beaming_factor(
    doppler_factor: f64,
    spectral_index: f64,
    continuous_jet: bool,
) -> Result<f64, HighEnergyError> {
    if !doppler_factor.is_finite() || doppler_factor <= 0.0 {
        return Err(HighEnergyError::InvalidBeamingFactor {
            factor: doppler_factor,
        });
    }
    if !spectral_index.is_finite() {
        return Err(HighEnergyError::InvalidPowerLawIndex { p: spectral_index });
    }
    let k = if continuous_jet { 2.0 } else { 3.0 };
    Ok(doppler_factor.powf(k + spectral_index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    #[verified_engine::verified]
    fn test_special_relativity() {
        let v = 0.6 * C;
        let gamma = calculate_lorentz_factor(v).expect("Failed to calc Lorentz factor");
        assert_relative_eq!(
            gamma,
            1.25,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        let fv = FourVector::new(3.0 / C, 4.0, 0.0, 0.0);
        let s2 = fv.invariant_interval().expect("Failed to calc interval");
        assert_relative_eq!(s2, 7.0, epsilon = math_commons::registry::TOLERANCE_FAST);

        let mass = 1.0;
        let p_vec = FourVector::new(1.25 * C, 0.75 * C, 0.0, 0.0);
        assert!(
            p_vec
                .is_valid_momentum(mass)
                .expect("Failed to check momentum")
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_lorentz_boost_and_doppler_factors() {
        let mass = 2.5;
        let rest_p = FourVector::new(mass * C, 0.0, 0.0, 0.0);
        let boosted = rest_p.lorentz_boost(-0.6, 0.0, 0.36).unwrap();
        assert!(boosted.is_valid_momentum(mass).unwrap());
        assert_relative_eq!(
            boosted.minkowski_norm_squared(),
            rest_p.minkowski_norm_squared(),
            max_relative = 1e-12
        );

        // Inverse boost restores original 4-vector; zero boost is exact identity.
        let restored = boosted.lorentz_boost(0.6, 0.0, -0.36).unwrap();
        assert_relative_eq!(restored.t, rest_p.t, epsilon = 1e-6);
        assert_relative_eq!(restored.x, rest_p.x, epsilon = 1e-6);
        assert_eq!(rest_p.lorentz_boost(0.0, 0.0, 0.0).unwrap(), rest_p);

        // Spacetime interval invariance under general 3D boost.
        let event = FourVector::new(5.0 / C, 2.0, -1.5, 3.0);
        let event_boosted = event
            .lorentz_boost_spacetime(0.4 * C, -0.3 * C, 0.2 * C)
            .unwrap();
        assert_relative_eq!(
            event.invariant_interval().unwrap(),
            event_boosted.invariant_interval().unwrap(),
            epsilon = 1e-10
        );

        // Relativistic Doppler: beta = 0.6 => head-on 2.0, receding 0.5, transverse 0.8.
        assert_relative_eq!(relativistic_doppler_factor(0.6, 0.0).unwrap(), 2.0);
        assert_relative_eq!(relativistic_doppler_factor(0.6, PI).unwrap(), 0.5);
        assert_relative_eq!(relativistic_doppler_factor(0.6, 0.5 * PI).unwrap(), 0.8);

        // Superluminal apparent motion for beta = 0.98 at optimal viewing angle acos(beta): beta_app = gamma * beta > 4.9.
        let theta_opt = 0.98_f64.acos();
        let beta_app = apparent_transverse_beta(0.98, theta_opt).unwrap();
        assert!(beta_app > 4.9);
        assert_relative_eq!(
            doppler_beaming_factor(2.0, 0.5, true).unwrap(),
            2.0_f64.powf(2.5)
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_errors() {
        assert!(calculate_lorentz_factor(C).is_err());
        assert!(calculate_lorentz_factor(1.1 * C).is_err());
        let fv = FourVector::new(0.0, 0.0, 0.0, 0.0);
        assert!(fv.is_valid_momentum(-1.0).is_err());
        assert!(fv.lorentz_boost(0.8, 0.8, 0.0).is_err());
        assert!(relativistic_doppler_factor(1.0, 0.0).is_err());
    }
}
