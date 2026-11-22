use super::integration::path_integrate;
use num_complex::Complex64;

/// Calculates the linear magnification factor of a conformal map w = f(z) at z0.
///
/// Returns $|f'(z_0)|$.
#[verified_engine::verified]
pub fn conformal_scale_factor<F>(derivative_f: F, z0: Complex64) -> f64
where
    F: Fn(Complex64) -> Complex64,
{
    derivative_f(z0).norm()
}

/// Calculates the area magnification factor of a conformal map w = f(z) at z0.
///
/// Returns $|f'(z_0)|^2$.
#[verified_engine::verified]
pub fn area_magnification<F>(derivative_f: F, z0: Complex64) -> f64
where
    F: Fn(Complex64) -> Complex64,
{
    derivative_f(z0).norm_sqr()
}

/// Represents a Schwarz-Christoffel transformation.
/// Maps the upper half plane to a polygon.
///
/// $$ \frac{dw}{dz} = A \prod (z - x_j)^{-k_j} $$
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SchwarzChristoffel {
    /// Points on the real axis corresponding to polygon vertices.
    pub prevertices: Vec<f64>,
    /// Exponents k_j.
    pub exponents: Vec<f64>,
    /// Scaling and rotation constant.
    pub a: Complex64,
    /// Translation constant.
    pub b: Complex64,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for SchwarzChristoffel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct SchwarzChristoffelRaw {
            prevertices: Vec<f64>,
            exponents: Vec<f64>,
            a: Complex64,
            b: Complex64,
        }
        let raw = SchwarzChristoffelRaw::deserialize(deserializer)?;
        if raw.prevertices.len() != raw.exponents.len() {
            return Err(serde::de::Error::custom(
                "SchwarzChristoffel requires equal numbers of prevertices and exponents",
            ));
        }
        Ok(Self {
            prevertices: raw.prevertices,
            exponents: raw.exponents,
            a: raw.a,
            b: raw.b,
        })
    }
}

impl SchwarzChristoffel {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(prevertices: Vec<f64>, exponents: Vec<f64>, a: Complex64, b: Complex64) -> Self {
        assert_eq!(
            prevertices.len(),
            exponents.len(),
            "Must have same number of prevertices and exponents"
        );
        Self {
            prevertices,
            exponents,
            a,
            b,
        }
    }

    /// Evaluates the derivative dw/dz at z.
    #[verified_engine::verified]
    pub fn derivative(&self, z: Complex64) -> Complex64 {
        let mut prod = Complex64::new(1.0, 0.0);
        for (x, k) in self.prevertices.iter().zip(self.exponents.iter()) {
            let term = z - Complex64::new(*x, 0.0);
            prod *= term.powf(-k);
        }
        self.a * prod
    }

    /// Transforms a point z in the upper half plane to the w plane.
    /// Integrates from a reference point `integration_start` to `z`.
    /// `w(z) = b + integral_{z_ref}^z f'(zeta) dzeta`.
    /// `b` is interpreted as `w(integration_start)`.
    #[verified_engine::verified]
    pub fn transform(&self, z: Complex64, integration_start: Complex64, steps: usize) -> Complex64 {
        let gamma = |t: f64| integration_start + (z - integration_start) * Complex64::new(t, 0.0);
        let dgamma = |_t: f64| z - integration_start;

        let f = |p: Complex64| self.derivative(p);

        let integral = path_integrate(f, gamma, dgamma, 0.0, 1.0, steps);
        self.b + integral
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_magnification() {
        // w = z^2 => w' = 2z
        let derivative = |z: Complex64| 2.0 * z;
        let z0 = Complex64::new(1.0, 1.0); // 1+i
        // |2(1+i)| = |2+2i| = sqrt(4+4) = sqrt(8) approx 2.828
        let scale = conformal_scale_factor(derivative, z0);
        assert_relative_eq!(scale, 8.0f64.sqrt(), epsilon = 1e-4);

        // Area mag = 8
        let area = area_magnification(derivative, z0);
        assert_relative_eq!(area, 8.0, epsilon = 1e-4);
    }

    #[test]
    #[verified_engine::verified]
    fn test_schwarz_christoffel_sqrt() {
        // Map z to z^0.5 (sqrt).
        // w' = 0.5 z^-0.5.
        // prevertex at 0, k=0.5. A=0.5. B=0 (w(0)=0).
        // Since 0 is singular for derivative (blows up), we integrate from 1 to z, and b = w(1) = 1.

        let sc = SchwarzChristoffel::new(
            vec![0.0],
            vec![0.5],
            Complex64::new(0.5, 0.0), // A
            Complex64::new(1.0, 0.0), // B = w(1)
        );

        let z_target = Complex64::new(4.0, 0.0);
        // w(4) should be 2.0.
        // We integrate from 1 to 4.
        let result = sc.transform(z_target, Complex64::new(1.0, 0.0), 1000);

        assert_relative_eq!(result.re, 2.0, epsilon = 1e-3);
        assert_relative_eq!(result.im, 0.0, epsilon = 1e-3);
    }
}
