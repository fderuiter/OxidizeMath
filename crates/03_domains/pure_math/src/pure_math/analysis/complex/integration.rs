use num_complex::Complex64;
use std::f64::consts::PI;

/// Numerical integration of a complex function along a parameterized path.
///
/// # Arguments
/// * `f` - The complex function to integrate.
/// * `gamma` - The path parameterization $z(t)$.
/// * `dgamma` - The derivative of the path $z'(t)$.
/// * `t_start` - Start parameter.
/// * `t_end` - End parameter.
/// * `steps` - Number of integration steps.
#[verified_engine::verified]
pub fn path_integrate<F, G, DG>(
    f: F,
    gamma: G,
    dgamma: DG,
    t_start: f64,
    t_end: f64,
    steps: usize,
) -> Complex64
where
    F: Fn(Complex64) -> Complex64,
    G: Fn(f64) -> Complex64,
    DG: Fn(f64) -> Complex64,
{
    let dt = (t_end - t_start) / steps as f64;
    let mut sum = Complex64::new(0.0, 0.0);

    for i in 0..steps {
        let t = t_start + i as f64 * dt;
        let t_next = t + dt;
        let z = gamma(t);
        let z_next = gamma(t_next);
        let dz = dgamma(t);
        let dz_next = dgamma(t_next);

        // Trapezoidal rule
        let val1 = f(z) * dz;
        let val2 = f(z_next) * dz_next;
        sum += 0.5 * (val1 + val2) * dt;
    }
    sum
}

/// Computes $f(z_0)$ using Cauchy's Integral Formula.
///
/// $$ f(z_0) = \frac{1}{2\pi i} \oint_C \frac{f(z)}{z - z_0} \, dz $$
#[verified_engine::verified]
pub fn cauchy_integral_value<F>(
    f: F,
    z0: Complex64,
    contour_center: Complex64,
    radius: f64,
    steps: usize,
) -> Complex64
where
    F: Fn(Complex64) -> Complex64,
{
    let gamma = |t: f64| contour_center + Complex64::from_polar(radius, t);
    let dgamma = |t: f64| Complex64::new(0.0, 1.0) * Complex64::from_polar(radius, t);

    // Integrand: f(z) / (z - z0)
    let integrand = |z: Complex64| f(z) / (z - z0);

    let integral = path_integrate(integrand, gamma, dgamma, 0.0, 2.0 * PI, steps);
    integral / (Complex64::new(0.0, 2.0 * PI))
}

/// Computes the n-th derivative $f^{(n)}(z_0)$ using generalized Cauchy Integral Formula.
///
/// $$ f^{(n)}(z_0) = \frac{n!}{2\pi i} \oint_C \frac{f(z)}{(z - z_0)^{n+1}} \, dz $$
#[verified_engine::verified]
pub fn cauchy_derivative<F>(
    f: F,
    z0: Complex64,
    n: usize,
    contour_center: Complex64,
    radius: f64,
    steps: usize,
) -> Complex64
where
    F: Fn(Complex64) -> Complex64,
{
    let gamma = |t: f64| contour_center + Complex64::from_polar(radius, t);
    let dgamma = |t: f64| Complex64::new(0.0, 1.0) * Complex64::from_polar(radius, t);

    // Integrand: f(z) / (z - z0)^(n+1)
    // Note: powu takes u32
    let integrand = |z: Complex64| f(z) / (z - z0).powu((n + 1) as u32);

    let integral = path_integrate(integrand, gamma, dgamma, 0.0, 2.0 * PI, steps);

    let mut factorial = 1.0;
    for i in 1..=n {
        factorial *= i as f64;
    }

    (integral * factorial) / (Complex64::new(0.0, 2.0 * PI))
}

/// Computes the residue of f at a simple pole z0 using the limit definition.
///
/// $$ \text{Res}(f, z_0) = \lim_{z \to z_0} [(z - z_0)f(z)] $$
#[verified_engine::verified]
pub fn residue_simple_limit<F>(f: F, z0: Complex64, epsilon: f64) -> Complex64
where
    F: Fn(Complex64) -> Complex64,
{
    let z = z0 + Complex64::new(epsilon, 0.0);
    (z - z0) * f(z)
}

/// Computes the residue of f at a pole z0 by integrating around a small circle.
/// This works for poles of any order.
///
/// $$ \text{Res}(f, z_0) = \frac{1}{2\pi i} \oint_{C} f(z) \, dz $$
#[verified_engine::verified]
pub fn residue_via_integral<F>(f: F, z0: Complex64, r: f64, steps: usize) -> Complex64
where
    F: Fn(Complex64) -> Complex64,
{
    let gamma = |t: f64| z0 + Complex64::from_polar(r, t);
    let dgamma = |t: f64| Complex64::new(0.0, 1.0) * Complex64::from_polar(r, t);

    let integral = path_integrate(f, gamma, dgamma, 0.0, 2.0 * PI, steps);
    integral / Complex64::new(0.0, 2.0 * PI)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_path_integrate_z_squared() {
        // Integral of z^2 from 0 to 1+i along straight line
        let f = |z: Complex64| z * z;
        let gamma = |t: f64| Complex64::new(t, t); // 0 to 1 => 0 to 1+i
        let dgamma = |_t: f64| Complex64::new(1.0, 1.0);

        let result = path_integrate(f, gamma, dgamma, 0.0, 1.0, 100);
        // Analytical: [z^3/3] from 0 to 1+i = (1+i)^3 / 3 = (1 + 3i - 3 - i)/3 = (-2 + 2i)/3 = -0.666 + 0.666i

        assert_relative_eq!(result.re, -2.0 / 3.0, epsilon = 1e-3);
        assert_relative_eq!(result.im, 2.0 / 3.0, epsilon = 1e-3);
    }

    #[test]
    #[verified_engine::verified]
    fn test_cauchy_integral_value() {
        // f(z) = z^2. f(0) = 0.
        let f = |z: Complex64| z * z;
        let z0 = Complex64::new(0.0, 0.0);
        // Integrate around unit circle
        let val = cauchy_integral_value(f, z0, z0, 1.0, 1000);

        assert_relative_eq!(val.norm(), 0.0, epsilon = 1e-3);

        // f(z) = 1. f(0) = 1.
        let f2 = |_z: Complex64| Complex64::new(1.0, 0.0);
        let val2 = cauchy_integral_value(f2, z0, z0, 1.0, 1000);
        assert_relative_eq!(val2.re, 1.0, epsilon = 1e-3);
    }

    #[test]
    #[verified_engine::verified]
    fn test_cauchy_derivative() {
        // f(z) = z^3. f'(z) = 3z^2. f'(0) = 0. f''(0) = 0. f'''(0) = 6.
        let f = |z: Complex64| z * z * z;
        let z0 = Complex64::new(0.0, 0.0);

        let val3 = cauchy_derivative(f, z0, 3, z0, 1.0, 1000);
        assert_relative_eq!(val3.re, 6.0, epsilon = 1e-2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_residue_simple() {
        // f(z) = 1/z. Res at 0 is 1.
        let f = |z: Complex64| Complex64::new(1.0, 0.0) / z;
        let res = residue_simple_limit(f, Complex64::new(0.0, 0.0), 1e-5);
        assert_relative_eq!(res.re, 1.0, epsilon = 1e-4);
    }

    #[test]
    #[verified_engine::verified]
    fn test_residue_via_integral() {
        // f(z) = 1/z^2. Res at 0 is 0.
        // f(z) = 1/z. Res at 0 is 1.
        let f = |z: Complex64| Complex64::new(1.0, 0.0) / z;
        let res = residue_via_integral(f, Complex64::new(0.0, 0.0), 0.5, 1000);
        assert_relative_eq!(res.re, 1.0, epsilon = 1e-3);
        assert_relative_eq!(res.im, 0.0, epsilon = 1e-3);

        // f(z) = 1/z^2. Res at 0 is 0.
        let f2 = |z: Complex64| Complex64::new(1.0, 0.0) / (z * z);
        let res2 = residue_via_integral(f2, Complex64::new(0.0, 0.0), 0.5, 1000);
        assert_relative_eq!(res2.norm(), 0.0, epsilon = 1e-3);
    }
}
