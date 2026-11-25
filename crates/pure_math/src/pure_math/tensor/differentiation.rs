use super::christoffel::christoffel_symbols;
use super::metric::Metric;
use super::types::{ContravariantVector, CovariantVector, Rank2Tensor, TensorError};
use math_scalars::Dual;
use nalgebra::{DMatrix, DVector};

/// Computes the exact covariant derivative $A^i_{;j} = \partial_j A^i + \Gamma^i_{kj} A^k$ of a
/// contravariant vector field via forward-mode [`Dual<f64>`] automatic differentiation.
///
/// Reference: Carroll (2004), *Spacetime and Geometry*, Addison-Wesley;
/// Griewank & Walther (2008), *Evaluating Derivatives*, SIAM, DOI: 10.1137/1.9780898717761.
///
/// # Stability
/// Evaluates $\partial_j A^i$ exactly via `dim` tangent sweeps over [`Dual<f64>`] and combines with
/// [`christoffel_symbols`], achieving machine-precision accuracy (`< 1e-13`) when paired with
/// [`super::metric::AdRiemannianMetric`].
///
/// # Domain of Validity
/// Requires `point` to lie in the regular domain of `metric` where $g_{ij}(x)$ is invertible and
/// `field` returns `point.len()` finite components.
#[verified_engine::verified]
pub fn covariant_derivative_contravariant_ad<F>(
    field: F,
    metric: &impl Metric,
    point: &DVector<f64>,
) -> Result<Rank2Tensor, TensorError>
where
    F: Fn(&[Dual<f64>]) -> Vec<Dual<f64>>,
{
    let dim = point.len();
    let gammas = christoffel_symbols(metric, point)?;
    let mut coords: Vec<Dual<f64>> = point.iter().map(|&v| Dual::constant(v)).collect();
    let base = field(&coords);
    if base.len() != dim || !base.iter().all(|d| d.val.is_finite()) {
        return Err(TensorError::SingularMetric);
    }

    let mut partial_derivatives = DMatrix::zeros(dim, dim);
    for j in 0..dim {
        coords[j] = Dual::variable(point[j]);
        let eval = field(&coords);
        coords[j] = Dual::constant(point[j]);
        if eval.len() != dim || !eval.iter().all(|d| d.eps.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        for i in 0..dim {
            partial_derivatives[(i, j)] = eval[i].eps;
        }
    }

    let mut result = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut gamma_term = 0.0;
            for k in 0..dim {
                gamma_term += gammas[i][(k, j)] * base[k].val;
            }
            result[(i, j)] = partial_derivatives[(i, j)] + gamma_term;
        }
    }
    Ok(Rank2Tensor(result))
}

/// Computes the exact covariant derivative $A_{i;j} = \partial_j A_i - \Gamma^k_{ij} A_k$ of a
/// covariant 1-form field via forward-mode [`Dual<f64>`] automatic differentiation.
///
/// Reference: Carroll (2004), *Spacetime and Geometry*, Addison-Wesley;
/// Griewank & Walther (2008), *Evaluating Derivatives*, SIAM, DOI: 10.1137/1.9780898717761.
///
/// # Stability
/// Eliminates finite-difference step-size cancellation in $\partial_j A_i$; matches analytical
/// `f64` covariant derivatives within `< 1e-13` when `metric` supplies exact partials.
///
/// # Domain of Validity
/// Requires `metric` to be invertible at `point` and `field` to return `point.len()` finite components.
#[verified_engine::verified]
pub fn covariant_derivative_covariant_ad<F>(
    field: F,
    metric: &impl Metric,
    point: &DVector<f64>,
) -> Result<Rank2Tensor, TensorError>
where
    F: Fn(&[Dual<f64>]) -> Vec<Dual<f64>>,
{
    let dim = point.len();
    let gammas = christoffel_symbols(metric, point)?;
    let mut coords: Vec<Dual<f64>> = point.iter().map(|&v| Dual::constant(v)).collect();
    let base = field(&coords);
    if base.len() != dim || !base.iter().all(|d| d.val.is_finite()) {
        return Err(TensorError::SingularMetric);
    }

    let mut partial_derivatives = DMatrix::zeros(dim, dim);
    for j in 0..dim {
        coords[j] = Dual::variable(point[j]);
        let eval = field(&coords);
        coords[j] = Dual::constant(point[j]);
        if eval.len() != dim || !eval.iter().all(|d| d.eps.is_finite()) {
            return Err(TensorError::SingularMetric);
        }
        for i in 0..dim {
            partial_derivatives[(i, j)] = eval[i].eps;
        }
    }

    let mut result = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut gamma_term = 0.0;
            for (k, gamma_k) in gammas.iter().enumerate().take(dim) {
                gamma_term += gamma_k[(i, j)] * base[k].val;
            }
            result[(i, j)] = partial_derivatives[(i, j)] - gamma_term;
        }
    }
    Ok(Rank2Tensor(result))
}

/// Computes the covariant derivative of a contravariant vector field $A^i_{;j}$.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::tensor::differentiation::covariant_derivative_contravariant;
/// use math_explorer::pure_math::tensor::{RiemannianMetric, ContravariantVector};
/// use math_explorer::nalgebra::{DMatrix, DVector};
///
/// let metric = RiemannianMetric::new(|_| DMatrix::identity(2, 2));
/// let field = |p: &DVector<f64>| ContravariantVector::new(DVector::from_vec(vec![p[0] * p[0], p[0] * p[1]]));
/// let point = DVector::from_vec(vec![2.0, 3.0]);
/// let deriv = covariant_derivative_contravariant(field, &metric, &point).unwrap();
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
    let gammas = christoffel_symbols(metric, point)?;
    let vec_at_point = field(point);
    let mut partial_derivatives = DMatrix::zeros(dim, dim);
    let mut p_eval = point.clone();

    for j in 0..dim {
        let h = f64::EPSILON.sqrt() * point[j].abs().max(1.0);
        p_eval[j] = point[j] + h;
        let vec_plus = field(&p_eval);
        p_eval[j] = point[j] - h;
        let vec_minus = field(&p_eval);
        p_eval[j] = point[j];

        let derivative_vec = (vec_plus.0 - vec_minus.0) / (2.0 * h);
        for i in 0..dim {
            partial_derivatives[(i, j)] = derivative_vec[i];
        }
    }

    let mut result = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut gamma_term = 0.0;
            for k in 0..dim {
                gamma_term += gammas[i][(k, j)] * vec_at_point.0[k];
            }
            result[(i, j)] = partial_derivatives[(i, j)] + gamma_term;
        }
    }
    Ok(Rank2Tensor(result))
}

/// Computes the covariant derivative of a covariant vector field $A_{i;j}$.
///
/// # Examples
///
/// ```rust
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::tensor::differentiation::covariant_derivative_covariant;
/// use math_explorer::pure_math::tensor::{RiemannianMetric, CovariantVector};
/// use math_explorer::nalgebra::{DMatrix, DVector};
///
/// let metric = RiemannianMetric::new(|_| DMatrix::identity(2, 2));
/// let field = |p: &DVector<f64>| CovariantVector::new(DVector::from_vec(vec![p[0] * p[0], p[0] * p[1]]));
/// let point = DVector::from_vec(vec![2.0, 3.0]);
/// let deriv = covariant_derivative_covariant(field, &metric, &point).unwrap();
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
    let gammas = christoffel_symbols(metric, point)?;
    let vec_at_point = field(point);
    let mut partial_derivatives = DMatrix::zeros(dim, dim);
    let mut p_eval = point.clone();

    for j in 0..dim {
        let h = f64::EPSILON.sqrt() * point[j].abs().max(1.0);
        p_eval[j] = point[j] + h;
        let vec_plus = field(&p_eval);
        p_eval[j] = point[j] - h;
        let vec_minus = field(&p_eval);
        p_eval[j] = point[j];

        let derivative_vec = (vec_plus.0 - vec_minus.0) / (2.0 * h);
        for i in 0..dim {
            partial_derivatives[(i, j)] = derivative_vec[i];
        }
    }

    let mut result = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            let mut gamma_term = 0.0;
            for (k, gamma_k) in gammas.iter().enumerate().take(dim) {
                gamma_term += gamma_k[(i, j)] * vec_at_point.0[k];
            }
            result[(i, j)] = partial_derivatives[(i, j)] - gamma_term;
        }
    }
    Ok(Rank2Tensor(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::tensor::metric::AdRiemannianMetric;
    use math_scalars::Scalar;

    #[test]
    fn test_ad_metric_christoffel_and_covariant_derivatives_machine_precision() {
        // 2D polar metric g = diag(1, r^2)
        let polar = AdRiemannianMetric::new(|p: &[Dual<f64>]| {
            let r = p[0];
            vec![Dual::ONE, Dual::ZERO, Dual::ZERO, r * r]
        });
        let r = 2.75_f64;
        let theta = 0.92_f64;
        let pt = DVector::from_vec(vec![r, theta]);
        let gammas = christoffel_symbols(&polar, &pt).expect("valid polar metric");

        // Exact: \Gamma^r_{\theta\theta} = -r, \Gamma^\theta_{r\theta} = \Gamma^\theta_{\theta r} = 1/r, others 0.
        assert!((gammas[0][(1, 1)] - (-r)).abs() < 1e-15);
        assert!((gammas[1][(0, 1)] - (1.0 / r)).abs() < 1e-15);
        assert!((gammas[1][(1, 0)] - (1.0 / r)).abs() < 1e-15);
        assert!(gammas[0][(0, 0)].abs() < 1e-15);

        // Contravariant field A^r = r^2 cos(\theta), A^\theta = sin(\theta) / r
        let contra = covariant_derivative_contravariant_ad(
            |p| vec![p[0] * p[0] * p[1].cos(), p[1].sin() / p[0]],
            &polar,
            &pt,
        )
        .expect("contravariant AD");
        // Exact A^r_{;r} = 2 r cos(\theta)
        // Exact A^r_{;\theta} = -r^2 sin(\theta) + \Gamma^r_{\theta\theta} A^\theta = -r^2 sin(\theta) - sin(\theta)
        // Exact A^\theta_{;r} = -sin(\theta)/r^2 + \Gamma^\theta_{\theta r} A^\theta = 0!
        // Exact A^\theta_{;\theta} = cos(\theta)/r + \Gamma^\theta_{r\theta} A^r = cos(\theta)/r + r cos(\theta)
        assert!((contra.0[(0, 0)] - 2.0 * r * theta.cos()).abs() < 1e-14);
        assert!((contra.0[(0, 1)] - (-(r * r + 1.0) * theta.sin())).abs() < 1e-14);
        assert!(contra.0[(1, 0)].abs() < 1e-14);
        assert!((contra.0[(1, 1)] - (theta.cos() / r + r * theta.cos())).abs() < 1e-14);

        // Covariant field B_r = r sin(\theta), B_\theta = r^2 cos(\theta)
        let cov = covariant_derivative_covariant_ad(
            |p| vec![p[0] * p[1].sin(), p[0] * p[0] * p[1].cos()],
            &polar,
            &pt,
        )
        .expect("covariant AD");
        // Exact B_{r;r} = sin(\theta)
        // Exact B_{r;\theta} = r cos(\theta) - \Gamma^\theta_{r\theta} B_\theta = r cos(\theta) - r cos(\theta) = 0!
        // Exact B_{\theta;r} = 2 r cos(\theta) - \Gamma^\theta_{\theta r} B_\theta = r cos(\theta)
        // Exact B_{\theta;\theta} = -r^2 sin(\theta) - \Gamma^r_{\theta\theta} B_r = 0!
        assert!((cov.0[(0, 0)] - theta.sin()).abs() < 1e-14);
        assert!(cov.0[(0, 1)].abs() < 1e-14);
        assert!((cov.0[(1, 0)] - r * theta.cos()).abs() < 1e-14);
        assert!(cov.0[(1, 1)].abs() < 1e-14);
    }
}
