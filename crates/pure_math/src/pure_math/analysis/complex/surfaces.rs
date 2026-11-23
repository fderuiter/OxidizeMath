use math_commons::error::MathError;
use num_complex::Complex64;
use std::f64::consts::PI;

/// A trait representing a Riemann Surface for a multivalued function.
///
/// Allows evaluation on different sheets and navigation between sheets via winding.
pub trait RiemannSurface {
    /// Returns the total number of sheets, or `None` if infinite.
    #[verified_engine::verified]
    fn num_sheets(&self) -> Option<usize>;

    /// Evaluates the function at `z` on the specified `sheet`.
    #[verified_engine::verified]
    fn evaluate(&self, z: Complex64, sheet: isize) -> Complex64;

    /// Calculates the target sheet index after winding around the branch point.
    /// `winding_number` represents the number of counter-clockwise turns.
    #[verified_engine::verified]
    fn next_sheet(&self, current_sheet: isize, winding_number: isize) -> isize;
}

/// Riemann surface for the n-th root function $f(z) = z^{1/n}$.
/// Has $n$ sheets.
pub struct NthRootSurface {
    #[allow(missing_docs)]
    pub n: usize,
}

impl NthRootSurface {
    /// Creates an $n$-sheeted Riemann surface for $z^{1/n}$ after verifying `n > 0`.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::StrictlyOutOfBounds`] when `n == 0`.
    #[verified_engine::verified]
    pub fn try_new(n: usize) -> Result<Self, MathError> {
        if n == 0 {
            return Err(MathError::StrictlyOutOfBounds {
                value: 0.0,
                min: 0.0,
                max: f64::INFINITY,
            });
        }
        Ok(Self { n })
    }

    /// Creates an $n$-sheeted Riemann surface for $z^{1/n}$.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`. Prefer [`Self::try_new`] for unvalidated runtime inputs.
    #[verified_engine::verified]
    pub fn new(n: usize) -> Self {
        Self::try_new(n).expect("NthRootSurface sheet count n must be positive")
    }
}

impl RiemannSurface for NthRootSurface {
    #[verified_engine::verified]
    fn num_sheets(&self) -> Option<usize> {
        Some(self.n)
    }

    #[verified_engine::verified]
    fn evaluate(&self, z: Complex64, sheet: isize) -> Complex64 {
        // Normalize sheet to 0..n-1
        let k = sheet.rem_euclid(self.n as isize);
        let (r, theta) = z.to_polar();
        // The argument on sheet k is theta + 2*pi*k.
        // Result is r^(1/n) * exp(i * (theta + 2*pi*k)/n).
        let new_theta = (theta + 2.0 * PI * (k as f64)) / (self.n as f64);
        Complex64::from_polar(r.powf(1.0 / self.n as f64), new_theta)
    }

    #[verified_engine::verified]
    fn next_sheet(&self, current_sheet: isize, winding_number: isize) -> isize {
        (current_sheet + winding_number).rem_euclid(self.n as isize)
    }
}

/// Riemann surface for the complex logarithm $f(z) = \log(z)$.
/// Has infinitely many sheets.
pub struct LogSurface;

impl RiemannSurface for LogSurface {
    #[verified_engine::verified]
    fn num_sheets(&self) -> Option<usize> {
        None
    }

    #[verified_engine::verified]
    fn evaluate(&self, z: Complex64, sheet: isize) -> Complex64 {
        // Principal log is ln(r) + i*theta, where theta in (-pi, pi].
        // Sheet k adds 2*pi*k to the imaginary part.
        z.ln() + Complex64::new(0.0, 2.0 * PI * (sheet as f64))
    }

    #[verified_engine::verified]
    fn next_sheet(&self, current_sheet: isize, winding_number: isize) -> isize {
        current_sheet + winding_number
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_nth_root_surface() {
        let surface = NthRootSurface::new(2); // Sqrt
        let z = Complex64::new(1.0, 0.0);

        // Sheet 0: sqrt(1) = 1
        let val0 = surface.evaluate(z, 0);
        assert_relative_eq!(val0.re, 1.0);
        assert_relative_eq!(val0.im, 0.0);

        // Sheet 1: sqrt(1) * e^(i*pi) = -1
        let val1 = surface.evaluate(z, 1);
        assert_relative_eq!(val1.re, -1.0);
        assert_relative_eq!(
            val1.im,
            0.0,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );

        // Winding
        assert_eq!(surface.next_sheet(0, 1), 1);
        assert_eq!(surface.next_sheet(1, 1), 0);
        assert_eq!(surface.next_sheet(0, 3), 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_log_surface() {
        let surface = LogSurface;
        let z = Complex64::new(1.0, 0.0);

        // Sheet 0: ln(1) = 0
        let val0 = surface.evaluate(z, 0);
        assert_relative_eq!(val0.norm(), 0.0);

        // Sheet 1: ln(1) + 2pi*i = 2pi*i
        let val1 = surface.evaluate(z, 1);
        assert_relative_eq!(val1.re, 0.0);
        assert_relative_eq!(val1.im, 2.0 * PI);

        // Winding
        assert_eq!(surface.next_sheet(0, 1), 1);
        assert_eq!(surface.next_sheet(5, -2), 3);
    }
}
