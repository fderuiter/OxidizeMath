//! Laplace Transforms.
//!
//! Transforms a function $f(t)$ (for $t \ge 0$) into a function $F(s)$.
//!
//! # Definition
//! $$\mathcal{L}\{f(t)\} = F(s) = \int_{0}^{\infty} e^{-st} f(t) \, dt$$
//!
//! # Transforms of Derivatives
//! *   $\mathcal{L}\{f'(t)\} = s F(s) - f(0)$
//! *   $\mathcal{L}\{f''(t)\} = s^2 F(s) - s f(0) - f'(0)$

use crate::pure_math::analysis::integration::Integrator;
use num_complex::Complex64;

/// Computes the Laplace Transform $F(s)$ of a real-valued function $f(t)$.
///
/// $$F(s) = \int_{0}^{\infty} f(t) e^{-st} \, dt$$
///
/// Note: This function performs numerical integration over a finite range `[0, max_t]`.
///
/// # Arguments
/// * `f` - The function to transform.
/// * `s` - The complex frequency parameter $s = \sigma + i\omega$.
/// * `max_t` - The upper limit for integration (approximating infinity). Choose a value where $e^{-st}f(t)$ is negligible.
/// * `integrator` - The integration strategy.
#[verified_engine::verified]
pub fn laplace_transform<F, I>(f: F, s: Complex64, max_t: f64, integrator: &I) -> Complex64
where
    F: Fn(f64) -> f64 + Copy,
    I: Integrator,
{
    let min = 0.0;
    let max = max_t;
    let eps = math_commons::registry::TOLERANCE_FAST;

    // integrand = f(t) * e^{-s t}
    // e^{-s t} = e^{-(sigma + i omega) t} = e^{-sigma t} * (cos(omega t) - i sin(omega t))
    // real part = f(t) * e^{-sigma t} * cos(omega t)
    // imag part = f(t) * e^{-sigma t} * -sin(omega t)

    let func_re = |t: f64| {
        let damping = (-s.re * t).exp();
        f(t) * damping * (s.im * t).cos()
    };

    let func_im = |t: f64| {
        let damping = (-s.re * t).exp();
        f(t) * damping * -(s.im * t).sin()
    };

    let res_re = integrator.integrate(func_re, min, max, eps);
    let res_im = integrator.integrate(func_im, min, max, eps);

    Complex64::new(res_re.value, res_im.value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::analysis::integration::ClenshawCurtis;

    #[test]
    #[verified_engine::verified]
    fn test_laplace_exponential() {
        // f(t) = e^{at}.
        // F(s) = 1 / (s - a).
        // Let a = -1. f(t) = e^{-t}.
        // Let s = 1 + 0i. F(1) = 1 / (1 - (-1)) = 1/2 = 0.5.

        let a = -1.0;
        let func = |t: f64| (a * t).exp();
        let s = Complex64::new(1.0, 0.0);
        let max_t = 20.0; // e^{-20} is very small
        let integrator = ClenshawCurtis;

        let result = laplace_transform(func, s, max_t, &integrator);

        // Expected: 1 / (s - a) = 1 / (1 - (-1)) = 0.5
        assert!((result.re - 0.5).abs() < 1e-5);
        assert!(result.im.abs() < 1e-5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_laplace_sine() {
        // f(t) = sin(kt).
        // F(s) = k / (s^2 + k^2).
        // Let k = 1. f(t) = sin(t).
        // Let s = 1 + 0i. F(1) = 1 / (1 + 1) = 0.5.

        let k = 1.0;
        let func = |t: f64| (k * t).sin();
        let s = Complex64::new(1.0, 0.0);
        let max_t = 20.0; // Decay is determined by e^{-st} part since sin(t) is bounded. e^{-t} at 20 is small.
        let integrator = ClenshawCurtis;

        let result = laplace_transform(func, s, max_t, &integrator);

        assert!((result.re - 0.5).abs() < 1e-5);
        assert!(result.im.abs() < 1e-5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_laplace_complex_s() {
        // f(t) = 1 (unit step).
        // F(s) = 1/s.
        // Let s = 1 + i. F(s) = 1 / (1+i) = (1-i)/2 = 0.5 - 0.5i.

        let func = |_t: f64| 1.0;
        let s = Complex64::new(1.0, 1.0);
        let max_t = 20.0;
        let integrator = ClenshawCurtis;

        let result = laplace_transform(func, s, max_t, &integrator);

        assert!((result.re - 0.5).abs() < 1e-5);
        assert!((result.im - (-0.5)).abs() < 1e-5);
    }
}
