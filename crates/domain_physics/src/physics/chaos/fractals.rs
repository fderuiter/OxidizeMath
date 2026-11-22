//! Fractal Dimension (Correlation Dimension) and Generation

use nalgebra::Vector3;
use num_complex::Complex;

/// Calculates the number of iterations before a point escapes the Mandelbrot set.
///
/// The Mandelbrot set is defined as the set of complex numbers $c$ for which the function
/// $f_c(z) = z^2 + c$ does not diverge when iterated from $z=0$.
///
/// Returns the number of iterations until $|z| > 2$ (escape radius), or `max_iter` if it stays bounded.
#[verified_engine::verified]
pub fn escape_time_mandelbrot(c: Complex<f64>, max_iter: u32) -> u32 {
    let mut z = Complex::new(0.0, 0.0);
    for i in 0..max_iter {
        if z.norm_sqr() > 4.0 {
            return i;
        }
        z = z * z + c;
    }
    max_iter
}

/// Calculates the number of iterations before a point escapes the Julia set for a given parameter $c$.
///
/// The Julia set is defined for a fixed complex parameter $c$ by iterating
/// $f_c(z) = z^2 + c$ starting from a point $z$ in the complex plane.
///
/// Returns the number of iterations until $|z| > 2$, or `max_iter` if it stays bounded.
#[verified_engine::verified]
pub fn escape_time_julia(z: Complex<f64>, c: Complex<f64>, max_iter: u32) -> u32 {
    let mut z = z;
    for i in 0..max_iter {
        if z.norm_sqr() > 4.0 {
            return i;
        }
        z = z * z + c;
    }
    max_iter
}

/// Calculates the Grassberger-Procaccia Correlation Sum $C(\epsilon)$ in place without heap allocation.
///
/// Sorts `trajectory` in place by $x$-coordinate so pair searches terminate as soon as
/// $\Delta x > \epsilon$, making it safe to execute inside [`verified_engine::engine::VerifiedEngine`]
/// zero-allocation regions.
///
/// # Example
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::Vector3;
/// use math_explorer::physics::chaos::fractals::correlation_dimension_in_place;
///
/// let mut scratch = [
///     Vector3::new(2.0, 0.0, 0.0),
///     Vector3::new(0.0, 0.0, 0.0),
///     Vector3::new(1.0, 0.0, 0.0),
/// ];
/// let c = correlation_dimension_in_place(&mut scratch, 1.1);
/// assert!((c - 2.0 / 3.0).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn correlation_dimension_in_place(trajectory: &mut [Vector3<f64>], epsilon: f64) -> f64 {
    let n = trajectory.len();
    if n < 2 || !epsilon.is_finite() || epsilon <= 0.0 {
        return 0.0;
    }

    let mut count = 0_usize;
    let epsilon_sq = epsilon * epsilon;

    // Sort in place by X-coordinate using stack-only unstable sort to prune search space.
    trajectory.sort_unstable_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..n {
        let p1 = trajectory[i];
        if !p1.x.is_finite() || !p1.y.is_finite() || !p1.z.is_finite() {
            continue;
        }

        // Because data is sorted by X, stop as soon as the X-distance exceeds epsilon.
        for p2 in trajectory.iter().skip(i + 1) {
            if !p2.x.is_finite() || !p2.y.is_finite() || !p2.z.is_finite() {
                continue;
            }
            let dx = p2.x - p1.x;
            if dx > epsilon {
                break;
            }

            let dy = p1.y - p2.y;
            let dz = p1.z - p2.z;
            if dx * dx + dy * dy + dz * dz < epsilon_sq {
                count += 1;
            }
        }
    }

    (2.0 * count as f64) / ((n * (n - 1)) as f64)
}

/// Calculates the Correlation Sum $C(\epsilon)$ for the Grassberger-Procaccia algorithm.
///
/// The Correlation Dimension is a measure of the dimensionality of the space occupied by a set of random points,
/// often referred to as a type of fractal dimension.
///
/// $C(\epsilon) = \frac{1}{N^2} \sum_{i, j} \Theta(\epsilon - |x_i - x_j|)$
/// where $\Theta$ is the Heaviside step function.
///
/// Note: This implementation returns the normalized count (proportion of pairs closer than $\epsilon$).
/// The actual dimension would be the slope of $\ln(C(\epsilon))$ vs $\ln(\epsilon)$.
#[verified_engine::verified]
pub fn correlation_dimension(trajectory: &[Vector3<f64>], epsilon: f64) -> f64 {
    if trajectory.len() < 2 || !epsilon.is_finite() || epsilon <= 0.0 {
        return 0.0;
    }
    let mut sorted_traj = trajectory.to_vec();
    correlation_dimension_in_place(&mut sorted_traj, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_mandelbrot_bounded() {
        // Point inside Mandelbrot set (0,0) should not escape
        let c = Complex::new(0.0, 0.0);
        let iter = escape_time_mandelbrot(c, 100);
        assert_eq!(iter, 100);
    }

    #[test]
    #[verified_engine::verified]
    fn test_mandelbrot_escape() {
        // Point outside Mandelbrot set (2,2) should escape quickly
        let c = Complex::new(2.0, 2.0);
        let iter = escape_time_mandelbrot(c, 100);
        assert!(iter < 100);
    }

    #[test]
    #[verified_engine::verified]
    fn test_julia_bounded() {
        // Julia set for c = 0 is the unit circle.
        // z=0 should stay at 0.
        let c = Complex::new(0.0, 0.0);
        let z = Complex::new(0.0, 0.0);
        let iter = escape_time_julia(z, c, 100);
        assert_eq!(iter, 100);
    }
}
