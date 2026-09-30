//! Biochemical Kinetics (Enzymes)
//!
//! This module implements enzyme kinetics using the Michaelis-Menten framework and other models.
//!
//! # Kinetic Models
//!
//! * **Michaelis-Menten**: $v = \frac{V_{max}\[S\]}{K_m + \[S\]}$
//! * **Hill Kinetics**: $v = \frac{V_{max}\[S\]^n}{K_m^n + \[S\]^n}$
//!
//! ##  Quick Start
//!
//! Calculate the reaction rate of an enzyme at different substrate concentrations.
//!
//! ```rust
//! use domain_biology::biology::kinetics::{MichaelisMenten, KineticsModel};
//!
//! // 1. Define Enzyme Properties
//! // Vmax = 100.0 (Max rate), Km = 50.0 (Substrate conc at half max rate)
//! let enzyme = MichaelisMenten::new(100.0, 50.0).expect("Invalid parameters");
//!
//! // 2. Calculate Velocity at [S] = 50.0 (should be Vmax/2)
//! let velocity = enzyme.reaction_velocity(50.0).expect("Substrate must be non-negative");
//! println!("Reaction Velocity at Km: {:.2}", velocity);
//! assert!((velocity - 50.0).abs() < math_commons::registry::TOLERANCE_FAST);
//!
//! // 3. Calculate Velocity at saturation ([S] = 1000.0)
//! let v_sat = enzyme.reaction_velocity(1000.0).expect("Substrate must be non-negative");
//! println!("Reaction Velocity at Saturation: {:.2}", v_sat);
//! assert!(v_sat > 90.0);
//! ```

use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
#[allow(missing_docs)]
pub enum KineticsError {
    #[error("Substrate concentration cannot be negative")]
    #[allow(missing_docs)]
    NegativeSubstrateConcentration,
    #[error("Parameters must be non-negative")]
    #[allow(missing_docs)]
    InvalidParameters,
}

/// Defines the interface for kinetic models.
pub trait KineticsModel {
    /// Calculates the reaction velocity for a given substrate concentration $\[S\]$.
    #[verified_engine::verified]
    fn reaction_velocity(&self, substrate_conc: f64) -> Result<f64, KineticsError>;
}

/// Represents an enzymatic reaction following Michaelis-Menten kinetics.
///
/// $$ v = \frac{V_{max}\[S\]}{K_m + \[S\]} $$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MichaelisMenten {
    /// Maximum reaction rate ($V_{max}$).
    pub v_max: f64,
    /// Michaelis constant ($K_m$).
    pub k_m: f64,
}

impl MichaelisMenten {
    /// Creates a new MichaelisMenten model with given parameters.
    #[verified_engine::verified]
    pub fn new(v_max: f64, k_m: f64) -> Result<Self, KineticsError> {
        if v_max < 0.0 || k_m < 0.0 {
            return Err(KineticsError::InvalidParameters);
        }
        Ok(Self { v_max, k_m })
    }

    /// Calculates the reaction velocity (Legacy wrapper).
    ///
    /// Delegates to `KineticsModel` implementation.
    #[verified_engine::verified]
    pub fn reaction_velocity(&self, substrate_conc: f64) -> Result<f64, KineticsError> {
        <Self as KineticsModel>::reaction_velocity(self, substrate_conc)
    }
}

impl KineticsModel for MichaelisMenten {
    #[verified_engine::verified]
    fn reaction_velocity(&self, substrate_conc: f64) -> Result<f64, KineticsError> {
        if substrate_conc < 0.0 {
            return Err(KineticsError::NegativeSubstrateConcentration);
        }
        let denominator = self.k_m + substrate_conc;
        if denominator == 0.0 {
            return Ok(0.0);
        }
        Ok(self.v_max * substrate_conc / denominator)
    }
}

/// Represents an enzymatic reaction following Hill kinetics (cooperative binding).
///
/// $$ v = \frac{V_{max}\[S\]^n}{K_m^n + \[S\]^n} $$
///
/// # Examples
///
/// ```rust
/// use domain_biology::biology::kinetics::{HillKinetics, KineticsModel};
///
/// // Create a cooperative enzyme with Hill coefficient n=2.0
/// let cooperative_enzyme = HillKinetics::new(100.0, 50.0, 2.0).expect("Invalid parameters");
///
/// // Calculate velocity at substrate concentration 100.0
/// let velocity = cooperative_enzyme.reaction_velocity(100.0).expect("Substrate must be non-negative");
///
/// // With positive cooperativity, the velocity is higher than Michaelis-Menten
/// // v = 100 * 100^2 / (50^2 + 100^2) = 100 * 10000 / 12500 = 80.0
/// assert!((velocity - 80.0).abs() < math_commons::registry::TOLERANCE_FAST);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HillKinetics {
    /// Maximum reaction rate ($V_{max}$).
    pub v_max: f64,
    /// Michaelis constant ($K_m$).
    pub k_m: f64,
    /// Hill coefficient ($n$). $n>1$ indicates positive cooperativity.
    pub n: f64,
}

impl HillKinetics {
    /// Creates a new HillKinetics model.
    #[verified_engine::verified]
    pub fn new(v_max: f64, k_m: f64, n: f64) -> Result<Self, KineticsError> {
        if v_max < 0.0 || k_m < 0.0 || n < 0.0 {
            return Err(KineticsError::InvalidParameters);
        }
        Ok(Self { v_max, k_m, n })
    }
}

impl KineticsModel for HillKinetics {
    #[verified_engine::verified]
    fn reaction_velocity(&self, substrate_conc: f64) -> Result<f64, KineticsError> {
        if substrate_conc < 0.0 {
            return Err(KineticsError::NegativeSubstrateConcentration);
        }
        let s_n = substrate_conc.powf(self.n);
        let k_n = self.k_m.powf(self.n);
        let denominator = k_n + s_n;

        if denominator == 0.0 {
            return Ok(0.0);
        }

        Ok(self.v_max * s_n / denominator)
    }
}

// Backward compatibility

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_original_behavior() {
        let reaction = MichaelisMenten::new(100.0, 50.0).unwrap();
        // At [S] = Km, v = Vmax / 2
        let v = reaction.reaction_velocity(50.0).unwrap();
        assert!((v - 50.0).abs() < math_commons::registry::TOLERANCE_FAST);

        // At [S] = 0, v = 0
        let v_0 = reaction.reaction_velocity(0.0).unwrap();
        assert!((v_0 - 0.0).abs() < math_commons::registry::TOLERANCE_FAST);

        // At very high [S], v -> Vmax
        let v_high = reaction.reaction_velocity(10000.0).unwrap();
        assert!((v_high - 100.0).abs() < 1.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_hill_kinetics() {
        // n=1 should behave like Michaelis-Menten
        let hill_mm = HillKinetics::new(100.0, 50.0, 1.0).unwrap();
        let v_mm = hill_mm.reaction_velocity(50.0).unwrap();
        assert!((v_mm - 50.0).abs() < math_commons::registry::TOLERANCE_FAST);

        // n=2 (Positive Cooperativity)
        // v = Vmax * S^2 / (Km^2 + S^2)
        // If S = Km, v = Vmax / 2 (still true for Hill)
        let hill_coop = HillKinetics::new(100.0, 50.0, 2.0).unwrap();
        let v_coop_half = hill_coop.reaction_velocity(50.0).unwrap();
        assert!((v_coop_half - 50.0).abs() < math_commons::registry::TOLERANCE_FAST);

        // If S = 2*Km = 100
        // MM: v = 100 * 100 / (50 + 100) = 10000 / 150 = 66.67
        // Hill(n=2): v = 100 * 100^2 / (50^2 + 100^2) = 100 * 10000 / (2500 + 10000) = 1000000 / 12500 = 80.0
        // Higher velocity due to cooperativity
        let mm = MichaelisMenten::new(100.0, 50.0).unwrap();
        let v_mm_high = mm.reaction_velocity(100.0).unwrap();
        let v_hill_high = hill_coop.reaction_velocity(100.0).unwrap();

        assert!(v_hill_high > v_mm_high);
        assert!((v_hill_high - 80.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_error_handling() {
        assert_eq!(
            MichaelisMenten::new(-1.0, 50.0).unwrap_err(),
            KineticsError::InvalidParameters
        );
        assert_eq!(
            HillKinetics::new(100.0, -50.0, 1.0).unwrap_err(),
            KineticsError::InvalidParameters
        );

        let mm = MichaelisMenten::new(100.0, 50.0).unwrap();
        assert_eq!(
            mm.reaction_velocity(-10.0).unwrap_err(),
            KineticsError::NegativeSubstrateConcentration
        );
    }
}
