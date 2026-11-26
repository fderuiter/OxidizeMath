//! Fourier Analysis.
//!
//! Represents functions as superpositions of continuous waves.
//!
//! # Fourier Series
//! A periodic function $f(x)$ with period $2L$ can be represented as:
//! $$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\frac{n\pi x}{L}\right) + b_n \sin\left(\frac{n\pi x}{L}\right) \right]$$
//!
//! # Fourier Transform
//! For non-periodic functions:
//! $$\tilde{f}(\omega) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \, dt$$

use crate::pure_math::analysis::integration::Integrator;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Calculates real Fourier coefficients for a periodic function $f(x)$ with period $2L$.
///
/// Returns $(a_0, a_n, b_n)$.
///
/// # Arguments
/// * `f` - The function to analyze.
/// * `l` - Half-period $L$. The function is assumed periodic over $[-L, L]$.
/// * `n_terms` - Number of terms $N$ to calculate (for $n=1$ to $N$).
/// * `integrator` - The integration strategy.
#[verified_engine::verified]
pub fn calculate_fourier_coefficients_real<F, I>(
    f: F,
    l: f64,
    n_terms: usize,
    integrator: &I,
) -> (f64, Vec<f64>, Vec<f64>)
where
    F: Fn(f64) -> f64 + Copy,
    I: Integrator,
{
    let min = -l;
    let max = l;
    // Target error for integration. Using a small value.
    let eps = math_commons::registry::TOLERANCE_FAST;

    // a0 = (1/L) * int(f(x))
    let a0_res = integrator.integrate(f, min, max, eps);
    let a0 = (1.0 / l) * a0_res.value;

    let mut an = Vec::with_capacity(n_terms);
    let mut bn = Vec::with_capacity(n_terms);

    for n in 1..=n_terms {
        let n_f = n as f64;
        let factor = n_f * PI / l;

        let func_cos = |x: f64| f(x) * (factor * x).cos();
        let func_sin = |x: f64| f(x) * (factor * x).sin();

        let res_cos = integrator.integrate(func_cos, min, max, eps);
        let res_sin = integrator.integrate(func_sin, min, max, eps);

        an.push((1.0 / l) * res_cos.value);
        bn.push((1.0 / l) * res_sin.value);
    }

    (a0, an, bn)
}

/// Evaluates the real Fourier series at a given point $x$.
#[verified_engine::verified]
pub fn evaluate_fourier_series_real(x: f64, l: f64, a0: f64, an: &[f64], bn: &[f64]) -> f64 {
    let mut sum = a0 / 2.0;
    for (i, (a, b)) in an.iter().zip(bn.iter()).enumerate() {
        let n = (i + 1) as f64;
        let term_arg = n * PI * x / l;
        sum += a * term_arg.cos() + b * term_arg.sin();
    }
    sum
}

/// Calculates complex Fourier coefficients $c_n$ for a function $f(x)$ with period $2L$.
///
/// $$c_n = \frac{1}{2L} \int_{-L}^{L} f(x) e^{-in\pi x/L} \, dx$$
///
/// # Arguments
/// * `f` - The function to analyze (assumed real-valued for simplicity here, but result is complex).
/// * `l` - Half-period $L$.
/// * `n_terms` - Number of terms to calculate (from $-N$ to $N$).
/// * `integrator` - The integration strategy.
#[verified_engine::verified]
pub fn calculate_fourier_coefficients_complex<F, I>(
    f: F,
    l: f64,
    n_terms: usize,
    integrator: &I,
) -> Vec<Complex64>
where
    F: Fn(f64) -> f64 + Copy,
    I: Integrator,
{
    let min = -l;
    let max = l;
    let eps = math_commons::registry::TOLERANCE_FAST;
    let range = n_terms as isize;
    let mut coeffs = Vec::with_capacity(2 * n_terms + 1);

    for n in -range..=range {
        let n_f = n as f64;
        let factor = n_f * PI / l;

        // e^{-i theta} = cos(theta) - i sin(theta)
        // integrand = f(x) * (cos(n pi x / L) - i sin(n pi x / L))
        // real part = f(x) cos(...)
        // imag part = -f(x) sin(...)

        // Note: argument is n * pi * x / L.
        // We use -n in the exponent formula, but here n runs from -N to N so we just use n directly
        // in the exponent e^{-i * n ...}?
        // Formula: c_n = ... e^{-i n pi x / L}
        // So argument to trig functions is (n pi x / L), and we take cos - i sin.

        let func_re = |x: f64| f(x) * (factor * x).cos();
        let func_im = |x: f64| f(x) * -(factor * x).sin();

        let res_re = integrator.integrate(func_re, min, max, eps);
        let res_im = integrator.integrate(func_im, min, max, eps);

        let c_n = Complex64::new(
            (1.0 / (2.0 * l)) * res_re.value,
            (1.0 / (2.0 * l)) * res_im.value,
        );
        coeffs.push(c_n);
    }
    coeffs
}

/// Computes the Fourier Transform $\tilde{f}(\omega)$ of a real-valued function $f(t)$.
///
/// $$\tilde{f}(\omega) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \, dt$$
///
/// Note: This function performs numerical integration over a finite range `bounds`.
///
/// # Arguments
/// * `f` - The function to transform.
/// * `omega` - The angular frequency $\omega$.
/// * `bounds` - The integration bounds $(t_{min}, t_{max})$. Since numerical integration requires finite bounds, approximate infinity with a sufficiently large interval where $f(t) \to 0$.
/// * `integrator` - The integration strategy.
#[verified_engine::verified]
pub fn fourier_transform<F, I>(f: F, omega: f64, bounds: (f64, f64), integrator: &I) -> Complex64
where
    F: Fn(f64) -> f64 + Copy,
    I: Integrator,
{
    let (min, max) = bounds;
    let eps = math_commons::registry::TOLERANCE_FAST;

    // e^{-i omega t} = cos(omega t) - i sin(omega t)
    let func_re = |t: f64| f(t) * (omega * t).cos();
    let func_im = |t: f64| f(t) * -(omega * t).sin();

    let res_re = integrator.integrate(func_re, min, max, eps);
    let res_im = integrator.integrate(func_im, min, max, eps);

    let factor = 1.0 / (2.0 * PI).sqrt();
    Complex64::new(factor * res_re.value, factor * res_im.value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::analysis::integration::ClenshawCurtis;

    #[test]
    #[verified_engine::verified]
    fn test_fourier_series_smooth() {
        // Smooth function: f(x) = cos(pi x / L) + 0.5 * sin(2 pi x / L)
        // L = 1. f(x) = cos(pi x) + 0.5 * sin(2 pi x).
        // Coefficients:
        // a1 = 1.0, b2 = 0.5. All others 0.

        let l = 1.0;
        let func = |x: f64| (PI * x).cos() + 0.5 * (2.0 * PI * x).sin();
        let integrator = ClenshawCurtis;

        let (a0, an, bn) = calculate_fourier_coefficients_real(func, l, 3, &integrator);

        assert!(a0.abs() < math_commons::registry::TOLERANCE_STANDARD);

        // a1 should be 1.0
        assert!((an[0] - 1.0).abs() < 1e-4);
        // a2, a3 should be 0.0
        assert!(an[1].abs() < 1e-4);
        assert!(an[2].abs() < 1e-4);

        // b1 should be 0.0
        assert!(bn[0].abs() < 1e-4);
        // b2 should be 0.5
        assert!((bn[1] - 0.5).abs() < 1e-4);
        // b3 should be 0.0
        assert!(bn[2].abs() < 1e-4);
    }

    #[test]
    #[verified_engine::verified]
    fn test_fourier_transform_gaussian() {
        // Gaussian: f(t) = e^{-t^2}
        // Transform: F(w) = 1/sqrt(2) * e^{-w^2/4}
        // (Using definition with 1/sqrt(2pi))

        let gaussian = |t: f64| (-t * t).exp();
        let integrator = ClenshawCurtis;
        let omega = 1.0;

        // Use bounds large enough to capture the gaussian
        let bounds = (-10.0, 10.0);

        let result = fourier_transform(gaussian, omega, bounds, &integrator);

        let expected = 1.0 / 2.0_f64.sqrt() * (-omega * omega / 4.0).exp();

        assert!((result.re - expected).abs() < 1e-5);
        assert!(result.im.abs() < 1e-5); // Should be real
    }
}
