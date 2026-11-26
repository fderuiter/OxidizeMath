use num_complex::Complex64;

/// Applies the Schwarz Reflection Principle to evaluate a function at `z`.
///
/// Assumes `f` is analytic in the upper half-plane (Im $z \ge 0$) and is real-valued on the real axis.
/// This function extends `f` to the lower half-plane via:
/// $$ f(z) = \overline{f(\bar{z})} $$
///
/// # Arguments
/// * `f` - The function defined for Im $z \ge 0$.
/// * `z` - The point to evaluate.
#[verified_engine::verified]
pub fn schwarz_reflect<F>(f: F, z: Complex64) -> Complex64
where
    F: Fn(Complex64) -> Complex64,
{
    if z.im < 0.0 { f(z.conj()).conj() } else { f(z) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_schwarz_reflect_polynomial() {
        // f(z) = z^2 + 1. Real on real axis.
        // We define f conceptually for Im z >= 0.
        let f_upper = |z: Complex64| z * z + 1.0;

        let z = Complex64::new(1.0, -2.0); // Lower half
        let val = schwarz_reflect(f_upper, z);

        // Analytical expectation: (1-2i)^2 + 1 = 1 - 4i - 4 + 1 = -2 - 4i
        assert_relative_eq!(
            val.re,
            -2.0,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
        assert_relative_eq!(
            val.im,
            -4.0,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
    }
}
