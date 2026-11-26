use super::coordinates::OrthogonalCoordinateSystem;
use nalgebra::Vector3;

const H: f64 = 1e-5;

/// Computes the partial derivative of a scalar function `f` with respect to the
/// `i`-th coordinate using a centered finite difference method.
#[verified_engine::verified]
fn partial_derivative<F>(i: usize, point: &Vector3<f64>, f: F) -> f64
where
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(i < 3);
    let mut p_plus = *point;
    p_plus[i] += H;
    let mut p_minus = *point;
    p_minus[i] -= H;
    (f(&p_plus) - f(&p_minus)) / (2.0 * H)
}

/// Computes the gradient of a scalar field $\nabla \Phi$.
/// Returns the components of the gradient in the local basis vectors of the coordinate system.
#[verified_engine::verified]
pub fn gradient<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let factors = coords.scale_factors(point);
    let mut grad = Vector3::zeros();

    for i in 0..3 {
        let df = partial_derivative(i, point, &field);
        grad[i] = df / factors[i];
    }

    grad
}

/// Computes the divergence of a vector field $\nabla \cdot \mathbf{A}$.
/// The field `A` should return components in the curvilinear basis.
#[verified_engine::verified]
pub fn divergence<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let factors = coords.scale_factors(point);
    let h1h2h3 = factors[0] * factors[1] * factors[2];

    let mut sum = 0.0;

    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;

        let term_func = |p: &Vector3<f64>| {
            let h = coords.scale_factors(p);
            let a = field(p);
            h[j] * h[k] * a[i]
        };

        sum += partial_derivative(i, point, term_func);
    }

    if h1h2h3.abs() < 1e-12 {
        0.0 // Avoid division by zero at singularities
    } else {
        sum / h1h2h3
    }
}

/// Computes the curl of a vector field $\nabla \times \mathbf{A}$.
/// Returns the components in the curvilinear basis.
#[verified_engine::verified]
pub fn curl<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let h = coords.scale_factors(point);
    let mut result = Vector3::zeros();

    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;

        let term_1 = {
            let func = |p: &Vector3<f64>| coords.scale_factors(p)[k] * field(p)[k];
            partial_derivative(j, point, func)
        };

        let term_2 = {
            let func = |p: &Vector3<f64>| coords.scale_factors(p)[j] * field(p)[j];
            partial_derivative(k, point, func)
        };

        result[i] = (term_1 - term_2) / (h[j] * h[k]);
    }

    result
}

/// Computes the Laplacian of a scalar field $\nabla^2 \Phi$.
#[verified_engine::verified]
pub fn laplacian<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let h_factors = coords.scale_factors(point);
    let h1h2h3 = h_factors[0] * h_factors[1] * h_factors[2];

    let mut sum = 0.0;

    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;

        let term_func = |p: &Vector3<f64>| {
            let h = coords.scale_factors(p);
            let dphi = partial_derivative(i, p, &field);
            (h[j] * h[k] / h[i]) * dphi
        };

        sum += partial_derivative(i, point, term_func);
    }

    if h1h2h3.abs() < 1e-12 {
        0.0
    } else {
        sum / h1h2h3
    }
}
