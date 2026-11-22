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

#[inline(always)]
#[verified_engine::verified]
fn renormalize_escape_potential(iter: u32, norm_sq: f64, max_iter: u32) -> f64 {
    let log_zn = 0.5 * norm_sq.ln();
    let nu = log_zn.ln() / std::f64::consts::LN_2;
    ((iter as f64) + 1.0 - nu).clamp(0.0, max_iter as f64)
}

/// Computes the continuous renormalized escape-time potential for the Mandelbrot set.
///
/// Uses an enlarged escape radius ($|z|^2 > 256$) with Douady-Hubbard logarithmic potential
/// renormalization $\nu = n + 1 - \log_2(\ln |z_n|)$ to eliminate integer terracing.
/// Bounded interior points return `max_iter as f64`.
#[verified_engine::verified]
pub fn smooth_escape_mandelbrot(c: Complex<f64>, max_iter: u32) -> f64 {
    if max_iter == 0 {
        return 0.0;
    }
    let mut z = Complex::new(0.0, 0.0);
    for i in 0..max_iter {
        let norm_sq = z.norm_sqr();
        if norm_sq > 256.0 {
            return renormalize_escape_potential(i, norm_sq, max_iter);
        }
        z = z * z + c;
    }
    let final_norm_sq = z.norm_sqr();
    if final_norm_sq > 4.0 {
        renormalize_escape_potential(max_iter, final_norm_sq, max_iter)
    } else {
        max_iter as f64
    }
}

/// Computes the continuous renormalized escape-time potential for the Julia set of parameter $c$.
///
/// Uses an enlarged escape radius ($|z|^2 > 256$) with logarithmic potential renormalization
/// $\nu = n + 1 - \log_2(\ln |z_n|)$ to produce smooth, band-free scalar fields.
/// Bounded interior points return `max_iter as f64`.
#[verified_engine::verified]
pub fn smooth_escape_julia(z: Complex<f64>, c: Complex<f64>, max_iter: u32) -> f64 {
    if max_iter == 0 {
        return 0.0;
    }
    let mut z_curr = z;
    for i in 0..max_iter {
        let norm_sq = z_curr.norm_sqr();
        if norm_sq > 256.0 {
            return renormalize_escape_potential(i, norm_sq, max_iter);
        }
        z_curr = z_curr * z_curr + c;
    }
    let final_norm_sq = z_curr.norm_sqr();
    if final_norm_sq > 4.0 {
        renormalize_escape_potential(max_iter, final_norm_sq, max_iter)
    } else {
        max_iter as f64
    }
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
}
