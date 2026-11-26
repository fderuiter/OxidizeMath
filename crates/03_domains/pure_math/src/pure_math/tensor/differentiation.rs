use super::christoffel::christoffel_symbols;
use super::metric::Metric;
use super::types::{ContravariantVector, CovariantVector, Rank2Tensor, TensorError};
use nalgebra::{DMatrix, DVector};

/// Computes the covariant derivative of a contravariant vector field $A^i_{;j}$.
///
/// In differential geometry, the covariant derivative extends the concept of a directional
/// derivative to manifolds. For a contravariant vector field (often just called a vector field),
/// it measures how the field changes as it is transported along the manifold, accounting for
/// the manifold's intrinsic curvature via the Levi-Civita connection (Christoffel symbols).
///
/// Mathematically, it is defined as:
/// $A^i_{;j} = \partial_j A^i + \Gamma^i_{kj} A^k$
///
/// # Arguments
///
/// * `field` - A closure representing the contravariant vector field to differentiate.
///   It maps a point on the manifold (as a `DVector<f64>`) to a [`ContravariantVector`].
/// * `metric` - The metric tensor of the manifold, implementing the [`Metric`] trait.
///   This is required to compute the Christoffel symbols.
/// * `point` - The specific coordinate point at which to evaluate the derivative.
///
/// # Errors
///
/// Returns a [`TensorError`] if the metric is singular or degenerate at the given point,
/// which prevents the computation of the inverse metric and subsequently the Christoffel symbols.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::tensor::differentiation::covariant_derivative_contravariant;
/// use math_explorer::pure_math::tensor::{RiemannianMetric, ContravariantVector};
/// use math_explorer::nalgebra::{DMatrix, DVector};
///
/// // Define a flat Euclidean metric space
/// let metric = RiemannianMetric::new(|_| DMatrix::identity(2, 2));
///
/// // Define a vector field: A(x, y) = (x^2, xy)
/// let field = |p: &DVector<f64>| ContravariantVector::new(DVector::from_vec(vec![p[0] * p[0], p[0] * p[1]]));
///
/// // Evaluate at point (2.0, 3.0)
/// let point = DVector::from_vec(vec![2.0, 3.0]);
///
/// let deriv = covariant_derivative_contravariant(field, &metric, &point).unwrap();
///
/// // In a flat space, Christoffel symbols are zero, so the covariant derivative
/// // simplifies to the standard Jacobian matrix (partial derivatives):
/// // dA^x/dx = 2x = 4.0
/// // dA^x/dy = 0.0
/// // dA^y/dx = y  = 3.0
/// // dA^y/dy = x  = 2.0
/// assert!((deriv.0[(0, 0)] - 4.0).abs() < 1e-4);
/// assert!((deriv.0[(0, 1)] - 0.0).abs() < 1e-4);
/// assert!((deriv.0[(1, 0)] - 3.0).abs() < 1e-4);
/// assert!((deriv.0[(1, 1)] - 2.0).abs() < 1e-4);
/// ```
#[verified_engine::verified]
pub fn covariant_derivative_contravariant<F>(
    field: F,
    metric: &impl Metric,
    point: &DVector<f64>,
) -> Result<Rank2Tensor, TensorError>
where
    F: Fn(&DVector<f64>) -> ContravariantVector,
{
    let dim = point.len();
    let h = 1e-5;
    let gammas = christoffel_symbols(metric, point)?;
    let vec_at_point = field(point);

    // Compute partial derivatives \partial A^i / \partial u^j
    let mut partial_derivatives = DMatrix::zeros(dim, dim);

    // Bolt Optimization: Allocate point buffer once to prevent 2*dim heap allocations
    let mut p_eval = point.clone();

    for j in 0..dim {
        // Forward shift
        p_eval[j] += h;
        let vec_plus = field(&p_eval);

        // Backward shift
        p_eval[j] -= 2.0 * h;
        let vec_minus = field(&p_eval);

        // Restore point buffer
        p_eval[j] += h;

        let derivative_vec = (vec_plus.0 - vec_minus.0) / (2.0 * h);
        for i in 0..dim {
            partial_derivatives[(i, j)] = derivative_vec[i];
        }
    }

    // Add Christoffel term: + \Gamma^i_{kj} A^k
    let mut result = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut gamma_term = 0.0;
            for k in 0..dim {
                // Formula: \Gamma^i_{kj}. m=i, lower1=k, lower2=j.
                // So access gammas[i][(k, j)].
                let gamma_val = gammas[i][(k, j)];
                gamma_term += gamma_val * vec_at_point.0[k];
            }

            result[(i, j)] = partial_derivatives[(i, j)] + gamma_term;
        }
    }

    Ok(Rank2Tensor(result))
}

/// Computes the covariant derivative of a covariant vector field $A_{i;j}$.
///
/// For a covariant vector field (also known as a 1-form or covector field), the covariant
/// derivative accounts for the manifold's curvature by subtracting a correction term
/// based on the Christoffel symbols. This ensures the resulting derivative transforms
/// properly as a rank-2 tensor.
///
/// Mathematically, it is defined as:
/// $A_{i;j} = \partial_j A_i - \Gamma^k_{ij} A_k$
///
/// # Arguments
///
/// * `field` - A closure representing the covariant vector field to differentiate.
///   It maps a point on the manifold (as a `DVector<f64>`) to a [`CovariantVector`].
/// * `metric` - The metric tensor of the manifold, implementing the [`Metric`] trait.
/// * `point` - The specific coordinate point at which to evaluate the derivative.
///
/// # Errors
///
/// Returns a [`TensorError`] if the metric is singular or degenerate at the given point,
/// making it impossible to compute the required Christoffel symbols.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::tensor::differentiation::covariant_derivative_covariant;
/// use math_explorer::pure_math::tensor::{RiemannianMetric, CovariantVector};
/// use math_explorer::nalgebra::{DMatrix, DVector};
///
/// // Define a flat Euclidean metric space
/// let metric = RiemannianMetric::new(|_| DMatrix::identity(2, 2));
///
/// // Define a covector field: A_i(x, y) = (x^2, xy)
/// let field = |p: &DVector<f64>| CovariantVector::new(DVector::from_vec(vec![p[0] * p[0], p[0] * p[1]]));
///
/// // Evaluate at point (2.0, 3.0)
/// let point = DVector::from_vec(vec![2.0, 3.0]);
///
/// let deriv = covariant_derivative_covariant(field, &metric, &point).unwrap();
///
/// // In a flat space, the covariant derivative equals the partial derivatives
/// assert!((deriv.0[(0, 0)] - 4.0).abs() < 1e-4);
/// assert!((deriv.0[(0, 1)] - 0.0).abs() < 1e-4);
/// assert!((deriv.0[(1, 0)] - 3.0).abs() < 1e-4);
/// assert!((deriv.0[(1, 1)] - 2.0).abs() < 1e-4);
/// ```
#[verified_engine::verified]
pub fn covariant_derivative_covariant<F>(
    field: F,
    metric: &impl Metric,
    point: &DVector<f64>,
) -> Result<Rank2Tensor, TensorError>
where
    F: Fn(&DVector<f64>) -> CovariantVector,
{
    let dim = point.len();
    let h = 1e-5;
    let gammas = christoffel_symbols(metric, point)?;
    let vec_at_point = field(point);

    // Compute partial derivatives \partial A_i / \partial u^j
    let mut partial_derivatives = DMatrix::zeros(dim, dim);

    // Bolt Optimization: Allocate point buffer once to prevent 2*dim heap allocations
    let mut p_eval = point.clone();

    for j in 0..dim {
        // Forward shift
        p_eval[j] += h;
        let vec_plus = field(&p_eval);

        // Backward shift
        p_eval[j] -= 2.0 * h;
        let vec_minus = field(&p_eval);

        // Restore point buffer
        p_eval[j] += h;

        let derivative_vec = (vec_plus.0 - vec_minus.0) / (2.0 * h);
        for i in 0..dim {
            partial_derivatives[(i, j)] = derivative_vec[i];
        }
    }

    // Subtract Christoffel term: - \Gamma^k_{ij} A_k
    let mut result = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut gamma_term = 0.0;
            for (k, gamma_k) in gammas.iter().enumerate().take(dim) {
                // Formula: \Gamma^k_{ij}. Upper=k, Lower1=i, Lower2=j.
                // Access gammas[k][(i, j)].
                let gamma_val = gamma_k[(i, j)];
                gamma_term += gamma_val * vec_at_point.0[k];
            }

            result[(i, j)] = partial_derivatives[(i, j)] - gamma_term;
        }
    }

    Ok(Rank2Tensor(result))
}
