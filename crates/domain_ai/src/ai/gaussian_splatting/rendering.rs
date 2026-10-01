use super::structs::Gaussian2D;
use nalgebra::Vector3;

/// Computes the accumulated color for a pixel using alpha blending.
///
/// C = sum(c_i * alpha_i * T_i)
/// where T_i = prod(1 - alpha_j) for j < i
#[verified_engine::verified]
pub fn blend_gaussians(
    sorted_gaussians: &[Gaussian2D],
    pixel_coord: &nalgebra::Point2<f64>,
) -> Vector3<f64> {
    let mut c = Vector3::zeros();
    let mut transmittance = 1.0;

    for g in sorted_gaussians {
        let alpha = evaluate_gaussian_opacity(g, pixel_coord);
        // Alpha blending contribution
        // contribution = alpha * transmittance
        // color += c_i * contribution
        // transmittance *= (1 - alpha)

        let contribution = alpha * transmittance;
        c += g.color * contribution;
        transmittance *= 1.0 - alpha;

        // Optimization: Stop if transmittance is negligible
        if transmittance < 0.0001 {
            break;
        }
    }

    c
}

/// Evaluates the opacity of a 2D Gaussian at a specific point.
///
/// alpha = alpha_raw * exp(-0.5 * (x - mu)^T * Sigma^-1 * (x - mu))
#[inline]
#[verified_engine::verified]
pub fn evaluate_gaussian_opacity(gaussian: &Gaussian2D, point: &nalgebra::Point2<f64>) -> f64 {
    let dx = point.x - gaussian.mean.x;
    let dy = point.y - gaussian.mean.y;

    // Power = d^T * Conic * d
    // Conic is symmetric: [[a, b], [b, c]] (where conic = -0.5 * Sigma^-1)
    // Result = a*x^2 + 2*b*x*y + c*y^2
    let a = gaussian.conic[(0, 0)];
    let b = gaussian.conic[(0, 1)];
    let c = gaussian.conic[(1, 1)];

    let power = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy;

    if !(-16.0..=0.0).contains(&power) {
        return 0.0;
    }

    gaussian.opacity * power.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Point2};

    #[test]
    #[verified_engine::verified]
    fn test_evaluate_gaussian() {
        let g = Gaussian2D {
            mean: Point2::new(0.0, 0.0),
            conic: Matrix2::from_diagonal_element(-0.5),
            opacity: 1.0,
            color: Vector3::new(1.0, 0.0, 0.0),
            depth: 1.0,
        };

        // At mean, exp(0) = 1, so returns opacity * 1 = 1.0
        let opacity = evaluate_gaussian_opacity(&g, &Point2::new(0.0, 0.0));
        assert!((opacity - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);

        // At 1 sigma away (x=1), exp(-0.5 * 1 * 1 * 1) = exp(-0.5) = 0.6065
        let opacity_sigma = evaluate_gaussian_opacity(&g, &Point2::new(1.0, 0.0));
        assert!((opacity_sigma - (-0.5f64).exp()).abs() < math_commons::registry::TOLERANCE_FAST);

        // Far away point (power < -16.0), early exit returns 0.0
        let opacity_far = evaluate_gaussian_opacity(&g, &Point2::new(10.0, 0.0));
        assert_eq!(opacity_far, 0.0);

        // Positive power (invalid conic), early exit returns 0.0
        let g_pos = Gaussian2D {
            conic: Matrix2::from_diagonal_element(0.5),
            ..g
        };
        let opacity_pos = evaluate_gaussian_opacity(&g_pos, &Point2::new(1.0, 0.0));
        assert_eq!(opacity_pos, 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_blend_gaussians() {
        let g1 = Gaussian2D {
            mean: Point2::new(0.0, 0.0),
            conic: Matrix2::from_diagonal_element(-0.5),
            opacity: 0.5,
            color: Vector3::new(1.0, 0.0, 0.0), // Red
            depth: 1.0,
        };
        let g2 = Gaussian2D {
            mean: Point2::new(0.0, 0.0),
            conic: Matrix2::from_diagonal_element(-0.5),
            opacity: 0.5,
            color: Vector3::new(0.0, 1.0, 0.0), // Green
            depth: 2.0,
        };

        // At mean (0,0), alpha = opacity * 1 = 0.5 for both.
        // C = g1.c * 0.5 * 1.0 + g2.c * 0.5 * (1 - 0.5)
        //   = (1,0,0) * 0.5 + (0,1,0) * 0.25
        //   = (0.5, 0.25, 0.0)

        let gaussians = vec![g1, g2];
        let color = blend_gaussians(&gaussians, &Point2::new(0.0, 0.0));

        assert!((color.x - 0.5).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((color.y - 0.25).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((color.z - 0.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }
}
