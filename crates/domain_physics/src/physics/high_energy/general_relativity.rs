use crate::error::HighEnergyError;
use math_commons::constants::{C, G};

/// Struct representing a Schwarzschild Black Hole.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SchwarzschildBlackHole {
    #[allow(missing_docs)]
    pub mass: f64,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SchwarzschildBlackHole {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        struct Wire {
            mass: f64,
        }
        let wire = Wire::deserialize(deserializer)?;
        Self::new(wire.mass).map_err(serde::de::Error::custom)
    }
}

impl SchwarzschildBlackHole {
    /// Creates a new SchwarzschildBlackHole.
    ///
    /// # Errors
    /// Returns `HighEnergyError::InvalidMass` if `mass <= 0`.
    #[verified_engine::verified]
    pub fn new(mass: f64) -> Result<Self, HighEnergyError> {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(HighEnergyError::InvalidMass { mass });
        }
        Ok(Self { mass })
    }

    /// Calculates the Schwarzschild Radius.
    /// Formula: R_s = 2GM / c^2
    ///
    /// # Errors
    /// Returns error if calculation fails (though unlikely with valid mass).
    #[verified_engine::verified]
    pub fn schwarzschild_radius(&self) -> Result<f64, HighEnergyError> {
        // Technically this can't fail if mass is valid, but keeping Result for consistency/API stability if needed
        Ok((2.0 * G * self.mass) / C.powi(2))
    }

    /// Calculates the gravitational time dilation factor dtau/dt at radius r.
    /// Formula: dtau/dt = sqrt(1 - Rs/r).
    ///
    /// # Errors
    /// Returns `HighEnergyError::InvalidRadius` if `r <= Rs`.
    #[verified_engine::verified]
    pub fn gravitational_time_dilation(&self, r: f64) -> Result<f64, HighEnergyError> {
        let rs = self.schwarzschild_radius()?;
        if r <= rs {
            return Err(HighEnergyError::InvalidRadius {
                radius: r,
                limit: rs,
            });
        }
        Ok((1.0 - rs / r).sqrt())
    }

    /// Calculates the Innermost Stable Circular Orbit (ISCO).
    /// Formula: R_ISCO = 6GM / c^2
    #[verified_engine::verified]
    pub fn isco(&self) -> Result<f64, HighEnergyError> {
        Ok((6.0 * G * self.mass) / C.powi(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use math_commons::constants::SOLAR_MASS;

    #[test]
    #[verified_engine::verified]
    fn test_schwarzschild() {
        let mass = SOLAR_MASS;
        let bh = SchwarzschildBlackHole::new(mass).expect("Failed to create BH");

        let rs = bh.schwarzschild_radius().expect("Failed to calc Rs");
        let expected_rs = 2.0 * G * mass / (C * C);
        assert_relative_eq!(rs, expected_rs);

        let isco = bh.isco().expect("Failed to calc ISCO");
        assert_relative_eq!(isco, 3.0 * rs);

        // Time dilation at r = 2 Rs.
        // factor = sqrt(1 - 1/2) = sqrt(0.5) approx 0.707
        let factor = bh
            .gravitational_time_dilation(2.0 * rs)
            .expect("Failed to calc time dilation");
        assert_relative_eq!(factor, 0.5f64.sqrt());
    }

    #[test]
    #[verified_engine::verified]
    fn test_errors() {
        assert!(SchwarzschildBlackHole::new(-1.0).is_err());

        let bh = SchwarzschildBlackHole::new(SOLAR_MASS).unwrap();
        let rs = bh.schwarzschild_radius().unwrap();
        assert!(bh.gravitational_time_dilation(0.5 * rs).is_err());
    }
}
