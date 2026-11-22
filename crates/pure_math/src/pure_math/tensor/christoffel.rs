use super::metric::Metric;
use super::types::TensorError;
use nalgebra::{DMatrix, DVector};

/// Computes the Christoffel symbols of the second kind $\Gamma^k_{ij}$ at a given point.
///
/// The Christoffel symbols are derived from the metric tensor and are essential for defining
/// the affine connection and computing covariant derivatives on a Riemannian manifold.
/// This implementation uses a central difference numerical approximation for the metric derivatives.
///
/// # Arguments
///
/// * `metric` - An implementation of the [`Metric`] trait representing the Riemannian metric $g_{ij}$.
/// * `point` - The coordinate vector $x$ where the Christoffel symbols should be evaluated.
///
/// # Returns
///
/// Returns a `Result` containing a `Vec` of `DMatrix<f64>` on success.
/// The outer vector represents the upper index $k$, and the inner matrix represents the lower indices $i$ and $j$.
/// Therefore, `result[k][(i, j)]` corresponds to $\Gamma^k_{ij}$.
///
/// # Errors
///
/// Returns [`TensorError::SingularMetric`] if the metric tensor is singular (non-invertible) at the given point,
/// or another [`TensorError`] variant if operations fail (e.g., dimension mismatch).
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::tensor::christoffel::christoffel_symbols;
/// use math_explorer::pure_math::tensor::metric::RiemannianMetric;
/// use math_explorer::nalgebra::{DMatrix, DVector};
///
/// // Define a 2D polar metric: g_{rr} = 1, g_{\theta\theta} = r^2, off-diagonals = 0
/// let polar_metric = RiemannianMetric::new(|p: &DVector<f64>| {
///     let r = p[0];
///     DMatrix::from_vec(2, 2, vec![
///         1.0, 0.0,
///         0.0, r * r,
///     ])
/// });
///
/// // Evaluate at r = 2.0, \theta = \pi/4
/// let point = DVector::from_vec(vec![2.0, std::f64::consts::PI / 4.0]);
/// let gammas = christoffel_symbols(&polar_metric, &point).unwrap();
///
/// // In polar coordinates:
/// // \Gamma^r_{\theta\theta} = -r
/// assert!((gammas[0][(1, 1)] - (-2.0)).abs() < 1e-4);
/// // \Gamma^\theta_{r\theta} = \Gamma^\theta_{\theta r} = 1/r
/// assert!((gammas[1][(0, 1)] - 0.5).abs() < 1e-4);
/// assert!((gammas[1][(1, 0)] - 0.5).abs() < 1e-4);
/// ```
#[verified_engine::verified]
pub fn christoffel_symbols(
    metric: &impl Metric,
    point: &DVector<f64>,
) -> Result<Vec<DMatrix<f64>>, TensorError> {
    let dim = point.len();
    let h = 1e-5;

    let g_inv = metric.inverse_metric_at(point)?;

    // Precompute metric derivatives: partial_g[k][i, j] = d(g_ij)/dx^k
    let mut partial_g = vec![DMatrix::zeros(dim, dim); dim];

    // Bolt Optimization: Allocate point buffer once to prevent 2*dim heap allocations
    let mut p_eval = point.clone();

    for k in 0..dim {
        // Forward shift
        p_eval[k] += h;
        let g_plus = metric.metric_at(&p_eval)?;

        // Backward shift
        p_eval[k] -= 2.0 * h;
        let g_minus = metric.metric_at(&p_eval)?;

        // Restore point buffer
        p_eval[k] += h;

        partial_g[k] = (g_plus - g_minus) / (2.0 * h);
    }

    let mut gammas = vec![DMatrix::zeros(dim, dim); dim];

    for m in 0..dim {
        // Upper index
        for i in 0..dim {
            // Lower index 1
            for j in 0..dim {
                // Lower index 2
                let mut sum = 0.0;
                for k in 0..dim {
                    // Summation index
                    // \Gamma^m_{ij} = 0.5 * g^{mk} * (dg_jk/du^i + dg_ki/du^j - dg_ij/du^k)
                    let term = partial_g[i][(j, k)] + partial_g[j][(k, i)] - partial_g[k][(i, j)];
                    sum += g_inv[(m, k)] * term;
                }
                gammas[m][(i, j)] = 0.5 * sum;
            }
        }
    }

    Ok(gammas)
}
