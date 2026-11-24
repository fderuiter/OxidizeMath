use crate::error::DoseFluenceError;

/// Defines a Point Spread Function (Kernel) for dose calculation.
pub trait DoseKernel {
    /// Calculates the kernel value at a given radial distance.
    #[verified_engine::verified]
    fn value_at(&self, radius: f64) -> Result<f64, DoseFluenceError>;
}

/// A simplified analytical Point Spread Function (Exponential Kernel).
///
/// Formula: $K(r) = \frac{A}{r^2} e^{-\beta r}$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExponentialKernel {
    #[allow(missing_docs)]
    pub amplitude: f64,
    #[allow(missing_docs)]
    pub beta: f64,
}

impl ExponentialKernel {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(amplitude: f64, beta: f64) -> Self {
        Self { amplitude, beta }
    }
}

impl DoseKernel for ExponentialKernel {
    #[verified_engine::verified]
    fn value_at(&self, radius: f64) -> Result<f64, DoseFluenceError> {
        if radius.abs() < math_commons::registry::TOLERANCE_FAST {
            return Err(DoseFluenceError::Singularity);
        }
        if radius < 0.0 {
            return Err(DoseFluenceError::NegativeRadius);
        }

        let val = (self.amplitude / (radius * radius)) * (-self.beta * radius).exp();
        Ok(val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_exponential_kernel() {
        let kernel = ExponentialKernel::new(4.0, 0.5);
        assert!(kernel.value_at(0.0).is_err());
        assert!(kernel.value_at(-1.0).is_err());

        let val = kernel.value_at(2.0).unwrap();
        // K = (4/4) * exp(-1) = 0.367879...
        assert!((val - (-1.0_f64).exp()).abs() < 1e-5);
    }
}
