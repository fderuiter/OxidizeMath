//! Input Data Calibration: CT to Physical Density.

use crate::error::DoseFluenceError;
use math_commons::primitives::NonNegativeFloat;

/// Converts Hounsfield Units (HU) to physical density (g/cm³).
///
/// The conversion uses a bi-linear model typical for scanner calibration curves:
/// - **Air to Water range (HU < 0)**: Linearly interpolates between Air (-1000 HU, ~0.0 g/cm³) and Water (0 HU, 1.0 g/cm³).
/// - **Water to Bone range (HU >= 0)**: Linearly interpolates between Water (0 HU, 1.0 g/cm³) and dense bone.
///
/// # Arguments
///
/// * `hu` - The CT number in Hounsfield Units.
///
/// # Returns
///
/// * `Result<NonNegativeFloat, DoseFluenceError>` - Physical density in g/cm³.
///
/// # Formula
///
/// - If $HU < 0$: $\rho = 1.0 + (HU / 1000.0)$
/// - If $HU \geq 0$: $\rho = 1.0 + (HU / 500.0)$
#[verified_engine::verified]
pub fn hu_to_density(hu: f64) -> Result<NonNegativeFloat, DoseFluenceError> {
    let density = if hu < 0.0 {
        1.0 + (hu / 1000.0)
    } else {
        1.0 + (hu / 500.0)
    };

    Ok(NonNegativeFloat::new(density)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_hu_to_density() {
        // Water
        assert!(
            (hu_to_density(0.0).unwrap().value() - 1.0).abs()
                < math_commons::registry::TOLERANCE_FAST
        );
        // Air
        assert!(
            (hu_to_density(-1000.0).unwrap().value() - 0.0).abs()
                < math_commons::registry::TOLERANCE_FAST
        );
        // Bone
        // HU = 500 -> rho = 1 + 500/500 = 2.0
        assert!(
            (hu_to_density(500.0).unwrap().value() - 2.0).abs()
                < math_commons::registry::TOLERANCE_FAST
        );

        // Deep vacuum / noise (should error instead of clipping to 0)
        assert!(hu_to_density(-1500.0).is_err());
    }
}
