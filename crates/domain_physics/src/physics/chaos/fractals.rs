//! Fractal Dimension (Correlation Dimension) and Generation

use nalgebra::Vector3;
use num_complex::Complex;

/// Squared escape radius ($R^2 = 16^2 = 256$) used for smooth Douady-Hubbard escape renormalization.
pub const SMOOTH_ESCAPE_RADIUS_SQR: f64 = 256.0;

/// Computes a continuous, band-free fractional iteration count at escape using log-log interpolation.
///
/// Adapts automatically to any superlinear polynomial degree $d \ge 2$ by interpolating
/// $\ln\ln|z|$ across the escape horizon $R = 16$, falling back to quadratic ($d = 2$)
/// Douady-Hubbard renormalization when escaping on the initial step.
#[verified_engine::verified]
#[must_use]
pub fn renormalize_escape_step(iter: u32, prev_norm_sqr: f64, curr_norm_sqr: f64) -> f64 {
    if !curr_norm_sqr.is_finite() || curr_norm_sqr <= SMOOTH_ESCAPE_RADIUS_SQR {
        return f64::from(iter);
    }
    let log_esc = (0.5 * SMOOTH_ESCAPE_RADIUS_SQR.ln()).ln();
    let log_curr = (0.5 * curr_norm_sqr.ln()).max(1.0e-12).ln();
    if iter > 0 && prev_norm_sqr.is_finite() && prev_norm_sqr > 1.05 {
        let log_prev = (0.5 * prev_norm_sqr.ln()).max(1.0e-12).ln();
        let denom = log_curr - log_prev;
        if denom > 1.0e-6 {
            let frac = ((log_esc - log_prev) / denom).clamp(0.0, 1.0);
            return (f64::from(iter) - 1.0 + frac).max(0.0);
        }
    }
    let nu = 1.0 - ((log_curr - log_esc) / std::f64::consts::LN_2);
    (f64::from(iter) + nu.clamp(0.0, 1.0)).max(0.0)
}

/// Calculates the continuous Douady-Hubbard renormalized escape time for the Mandelbrot set.
///
/// Returns `f64::from(max_iter)` when the orbit stays bounded within `SMOOTH_ESCAPE_RADIUS_SQR`.
#[verified_engine::verified]
#[must_use]
pub fn smooth_escape_mandelbrot(c: Complex<f64>, max_iter: u32) -> f64 {
    let (cx, cy) = (c.re, c.im);
    let cy2 = cy * cy;
    if (cx + 1.0) * (cx + 1.0) + cy2 <= 0.0625 {
        return f64::from(max_iter);
    }
    let q = (cx - 0.25) * (cx - 0.25) + cy2;
    if q * (q + (cx - 0.25)) <= 0.25 * cy2 {
        return f64::from(max_iter);
    }
    let (mut zr, mut zi) = (0.0, 0.0);
    let mut prev_norm_sqr = 0.0;
    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let norm_sqr = zr2 + zi2;
        if norm_sqr > SMOOTH_ESCAPE_RADIUS_SQR {
            return renormalize_escape_step(i, prev_norm_sqr, norm_sqr).min(f64::from(max_iter));
        }
        prev_norm_sqr = norm_sqr;
        zi = 2.0 * zr * zi + cy;
        zr = zr2 - zi2 + cx;
    }
    f64::from(max_iter)
}

/// Calculates the continuous Douady-Hubbard renormalized escape time for a Julia set with parameter $c$.
///
/// Returns `f64::from(max_iter)` when the orbit stays bounded within `SMOOTH_ESCAPE_RADIUS_SQR`.
#[verified_engine::verified]
#[must_use]
pub fn smooth_escape_julia(z: Complex<f64>, c: Complex<f64>, max_iter: u32) -> f64 {
    let (mut zr, mut zi) = (z.re, z.im);
    let (cx, cy) = (c.re, c.im);
    let mut prev_norm_sqr = zr * zr + zi * zi;
    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let norm_sqr = zr2 + zi2;
        if norm_sqr > SMOOTH_ESCAPE_RADIUS_SQR {
            return renormalize_escape_step(i, prev_norm_sqr, norm_sqr).min(f64::from(max_iter));
        }
        prev_norm_sqr = norm_sqr;
        zi = 2.0 * zr * zi + cy;
        zr = zr2 - zi2 + cx;
    }
    f64::from(max_iter)
}

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
    let n = trajectory.len();
    if n < 2 {
        return 0.0;
    }

    let mut count = 0;
    let epsilon_sq = epsilon * epsilon;

    // Profiler Optimization: Sort by X-coordinate to prune search space.
    // This reduces complexity from O(N^2) to O(N * k) where k is the neighborhood size.
    // For small epsilon (the relevant case for fractal dimension), this yields massive speedups.

    let mut sorted_traj = trajectory.to_vec();
    // Use unstable_sort_by for speed; floating point sort handles NaN by treating as equal (safe assumption here).
    sorted_traj.sort_unstable_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

    for i in 0..n {
        let p1 = sorted_traj[i];

        // Because data is sorted by X, we only look ahead.
        // We can stop as soon as the X-distance exceeds epsilon.
        for p2 in sorted_traj.iter().skip(i + 1) {
            let dx = p2.x - p1.x; // p2.x >= p1.x due to sort

            if dx > epsilon {
                break;
            }

            let dy = p1.y - p2.y;
            let dz = p1.z - p2.z;

            // Manual squared distance check to avoid Vector3 overhead in the hot loop
            if dx * dx + dy * dy + dz * dz < epsilon_sq {
                count += 1;
            }
        }
    }

    // Multiply by 2 because of symmetry (pair (i,j) and (j,i)), and divide by total possible pairs N*(N-1)
    // C(eps) = (2 * count) / (N * (N - 1))

    (2.0 * count as f64) / ((n * (n - 1)) as f64)
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

    #[test]
    #[verified_engine::verified]
    fn test_smooth_escape_mandelbrot_and_julia_continuity() {
        let interior = smooth_escape_mandelbrot(Complex::new(0.0, 0.0), 80);
        assert_eq!(interior, 80.0);

        let far = smooth_escape_mandelbrot(Complex::new(2.1, 0.0), 80);
        let closer = smooth_escape_mandelbrot(Complex::new(1.2, 0.0), 80);
        assert!(far.is_finite() && closer.is_finite());
        assert!(far < closer && closer < 80.0);

        let julia_val = smooth_escape_julia(Complex::new(1.5, 0.25), Complex::new(-0.4, 0.6), 80);
        assert!(julia_val.is_finite() && julia_val < 80.0);
    }
}
