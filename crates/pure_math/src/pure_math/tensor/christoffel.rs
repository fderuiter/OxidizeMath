use super::metric::Metric;
use super::static_metric::{StaticMetric, static_christoffel_symbols};
use super::types::TensorError;
use engine_symbolic::Dual;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

#[verified_engine::verified]
fn compute_dynamic_metric_partials(
    metric: &impl Metric,
    point: &DVector<f64>,
) -> Result<Vec<DMatrix<f64>>, TensorError> {
    let dim = point.len();
    let mut partial_g = vec![DMatrix::zeros(dim, dim); dim];
    let mut dual_pt: Vec<Dual<f64>> = point.iter().copied().map(Dual::constant).collect();
    let mut used_dual = true;

    for (k, slot) in partial_g.iter_mut().enumerate() {
        dual_pt[k] = Dual::variable(point[k]);
        let maybe_dual = metric.metric_dual_at(&dual_pt);
        dual_pt[k] = Dual::constant(point[k]);
        if let Some(res) = maybe_dual {
            let g_dual = res?;
            let diff = DMatrix::from_fn(dim, dim, |r, c| g_dual[(r, c)].eps);
            if !diff.iter().all(|v| v.is_finite()) {
                return Err(TensorError::SingularMetric);
            }
            *slot = diff;
        } else {
            used_dual = false;
            break;
        }
    }
    if used_dual {
        return Ok(partial_g);
    }

    let h = 1e-5;
    let mut p_eval = point.clone();
    for (k, slot) in partial_g.iter_mut().enumerate() {
        p_eval[k] += h;
        let g_plus = metric.metric_at(&p_eval)?;
        p_eval[k] -= 2.0 * h;
        let g_minus = metric.metric_at(&p_eval)?;
        p_eval[k] += h;
        *slot = (g_plus - g_minus) / (2.0 * h);
    }
    Ok(partial_g)
}

/// Computes the Christoffel symbols of the second kind $\Gamma^k_{ij}$ at a given point.
///
/// Uses forward-mode [`Dual<f64>`] automatic differentiation whenever [`Metric::metric_dual_at`]
/// is implemented (achieving machine precision right up to event-horizon singularities without
/// stencil step-size tuning), falling back to central differences for legacy `f64`-only closures.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::engine_symbolic::Dual;
/// use math_explorer::pure_math::tensor::christoffel::christoffel_symbols;
/// use math_explorer::pure_math::tensor::metric::DualRiemannianMetric;
/// use math_explorer::nalgebra::{DMatrix, DVector};
///
/// // Define a 2D polar metric: g_{rr} = 1, g_{\theta\theta} = r^2, off-diagonals = 0
/// let polar_metric = DualRiemannianMetric::new(|p: &[Dual<f64>]| {
///     let r = p[0];
///     DMatrix::from_vec(2, 2, vec![
///         Dual::ONE, Dual::ZERO,
///         Dual::ZERO, r * r,
///     ])
/// });
///
/// // Evaluate at r = 2.0, \theta = \pi/4
/// let point = DVector::from_vec(vec![2.0, std::f64::consts::PI / 4.0]);
/// let gammas = christoffel_symbols(&polar_metric, &point).unwrap();
///
/// // Exact forward-mode Dual derivatives match analytical polar connection to machine precision:
/// assert!((gammas[0][(1, 1)] - (-2.0)).abs() < 1e-14);
/// assert!((gammas[1][(0, 1)] - 0.5).abs() < 1e-14);
/// assert!((gammas[1][(1, 0)] - 0.5).abs() < 1e-14);
/// ```
#[verified_engine::verified]
pub fn christoffel_symbols(
    metric: &impl Metric,
    point: &DVector<f64>,
) -> Result<Vec<DMatrix<f64>>, TensorError> {
    if !point.iter().all(|v| v.is_finite()) {
        return Err(TensorError::SingularMetric);
    }
    let dim = point.len();
    let g_inv = metric.inverse_metric_at(point)?;
    let partial_g = compute_dynamic_metric_partials(metric, point)?;
    let mut gammas = vec![DMatrix::zeros(dim, dim); dim];

    for m in 0..dim {
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    let term = partial_g[i][(j, k)] + partial_g[j][(k, i)] - partial_g[k][(i, j)];
                    sum += g_inv[(m, k)] * term;
                }
                if !sum.is_finite() {
                    return Err(TensorError::SingularMetric);
                }
                gammas[m][(i, j)] = 0.5 * sum;
            }
        }
    }

    Ok(gammas)
}

/// Computes the Christoffel symbols of the second kind $\Gamma^k_{ij}$ on stack-allocated
/// [`SMatrix`] buffers without `DMatrix` heap allocations.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::engine_symbolic::Dual;
/// use math_explorer::nalgebra::{SMatrix, SVector};
/// use math_explorer::pure_math::tensor::DualStaticRiemannianMetric;
/// use math_explorer::pure_math::tensor::christoffel::christoffel_symbols_fixed;
///
/// let polar = DualStaticRiemannianMetric::<_, 2>::new(|p: &SVector<Dual<f64>, 2>| {
///     let r = p[0];
///     Ok(SMatrix::<Dual<f64>, 2, 2>::new(Dual::ONE, Dual::ZERO, Dual::ZERO, r * r))
/// });
/// let pt = SVector::<f64, 2>::new(2.0, std::f64::consts::FRAC_PI_4);
/// let gammas = christoffel_symbols_fixed(&polar, &pt).unwrap();
/// assert!((gammas[0][(1, 1)] - (-2.0)).abs() < 1e-14);
/// assert!((gammas[1][(0, 1)] - 0.5).abs() < 1e-14);
/// ```
#[verified_engine::verified]
pub fn christoffel_symbols_fixed<M: StaticMetric<N> + ?Sized, const N: usize>(
    metric: &M,
    point: &SVector<f64, N>,
) -> Result<[SMatrix<f64, N, N>; N], TensorError> {
    static_christoffel_symbols(metric, point)
}

#[cfg(test)]
mod fixed_tests {
    use super::*;
    use crate::pure_math::tensor::{
        DualRiemannianMetric, DualStaticRiemannianMetric, StaticMetric, TwoSphereMetric,
    };

    #[test]
    #[verified_engine::verified]
    fn test_christoffel_symbols_fixed_zero_alloc_and_schwarzschild_horizon_precision() {
        // Schwarzschild R_s = 1.0 evaluated at r = 1.00002 (2e-5 from horizon singularity!)
        let rs = 1.0_f64;
        let dyn_schwarzschild = DualRiemannianMetric::new(move |p: &[Dual<f64>]| {
            let r = p[1];
            let theta = p[2];
            let f = Dual::ONE - Dual::constant(rs) / r;
            let sin_t = theta.sin();
            let z = Dual::ZERO;
            DMatrix::from_row_slice(
                4,
                4,
                &[
                    -f,
                    z,
                    z,
                    z,
                    z,
                    f.recip(),
                    z,
                    z,
                    z,
                    z,
                    r * r,
                    z,
                    z,
                    z,
                    z,
                    r * r * sin_t * sin_t,
                ],
            )
        });
        let fixed_schwarzschild =
            DualStaticRiemannianMetric::<_, 4>::new(move |p: &SVector<Dual<f64>, 4>| {
                let r = p[1];
                let theta = p[2];
                if !r.re.is_finite() || r.re <= rs {
                    return Err(TensorError::SingularMetric);
                }
                let f = Dual::ONE - Dual::constant(rs) / r;
                let sin_t = theta.sin();
                let mut g = SMatrix::<Dual<f64>, 4, 4>::from_element(Dual::ZERO);
                g[(0, 0)] = -f;
                g[(1, 1)] = f.recip();
                g[(2, 2)] = r * r;
                g[(3, 3)] = r * r * sin_t * sin_t;
                Ok(g)
            });

        let r = 1.00002_f64;
        let theta = 1.1_f64;
        let dyn_pt = DVector::from_vec(vec![0.0, r, theta, 0.3]);
        let fixed_pt = SVector::<f64, 4>::new(0.0, r, theta, 0.3);

        let dyn_gamma =
            christoffel_symbols(&dyn_schwarzschild, &dyn_pt).expect("dyn schwarzschild");
        let sphere = TwoSphereMetric::new(2.5).expect("valid radius");
        let sphere_pt = SVector::<f64, 2>::new(3.0, 0.8);

        let ((fixed_gamma, sphere_gamma), metrics) =
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                let g1 = christoffel_symbols_fixed(&fixed_schwarzschild, &fixed_pt)
                    .expect("fixed schwarzschild");
                let g2 = christoffel_symbols_fixed(&sphere, &sphere_pt).expect("fixed sphere");
                (g1, g2)
            });
        assert!(metrics.function_calls > 0);

        let (sin_t, cos_t) = theta.sin_cos();
        let exact_g0_01 = rs / (2.0 * r * (r - rs));
        let exact_g1_00 = rs * (r - rs) / (2.0 * r * r * r);
        let exact_g1_11 = -exact_g0_01;
        let exact_g1_22 = -(r - rs);
        let exact_g1_33 = -(r - rs) * sin_t * sin_t;
        let exact_g2_12 = 1.0 / r;
        let exact_g2_33 = -sin_t * cos_t;
        let exact_g3_23 = cos_t / sin_t;

        let checks = [
            (0, 0, 1, exact_g0_01),
            (1, 0, 0, exact_g1_00),
            (1, 1, 1, exact_g1_11),
            (1, 2, 2, exact_g1_22),
            (1, 3, 3, exact_g1_33),
            (2, 1, 2, exact_g2_12),
            (2, 3, 3, exact_g2_33),
            (3, 2, 3, exact_g3_23),
        ];
        for (m, i, j, exact) in checks {
            let rel_dyn = (dyn_gamma[m][(i, j)] - exact).abs() / exact.abs();
            let rel_fix = (fixed_gamma[m][(i, j)] - exact).abs() / exact.abs();
            assert!(
                rel_dyn < 1e-11,
                "dyn Gamma^{m}_{{{i}{j}}} rel err {rel_dyn:e} (got {}, exact {exact})",
                dyn_gamma[m][(i, j)]
            );
            assert!(
                rel_fix < 1e-11,
                "fixed Gamma^{m}_{{{i}{j}}} rel err {rel_fix:e} (got {}, exact {exact})",
                fixed_gamma[m][(i, j)]
            );
        }

        let exact_sphere = sphere.christoffel_at(&sphere_pt).expect("exact sphere");
        assert!((sphere_gamma[0][(1, 1)] - exact_sphere[0][(1, 1)]).abs() < 1e-14);
        assert!((sphere_gamma[1][(0, 1)] - exact_sphere[1][(0, 1)]).abs() < 1e-14);
    }
}
