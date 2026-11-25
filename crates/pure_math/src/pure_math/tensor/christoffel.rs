use super::metric::Metric;
use super::static_metric::{StaticMetric, static_christoffel_symbols};
use super::types::TensorError;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

/// Computes the Christoffel symbols of the second kind $\Gamma^k_{ij}$ at a given point.
///
/// The Christoffel symbols are derived from the metric tensor and are essential for defining
/// the affine connection and computing covariant derivatives on a Riemannian manifold.
/// Evaluates exact forward-mode [`math_scalars::Dual<f64>`] metric derivatives when `metric` is an
/// [`super::metric::AdRiemannianMetric`] (falling back to scale-aware $\varepsilon^{1/3}$ central
/// differences for legacy closure metrics).
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
    let g_inv = metric.inverse_metric_at(point)?;
    let partial_g = metric.metric_partials_at(point)?;

    let mut gammas = vec![DMatrix::zeros(dim, dim); dim];

    for m in 0..dim {
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for k in 0..dim {
                    let term = partial_g[i][(j, k)] + partial_g[j][(k, i)] - partial_g[k][(i, j)];
                    sum += g_inv[(m, k)] * term;
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
/// Uses fixed-dimension [`StaticMetric`] central-difference evaluation so interactive geodesic,
/// Paczyński-Wiita connection validation, and WebAssembly loops run with zero heap allocations.
///
/// # Errors
///
/// Returns [`TensorError::SingularMetric`] if `point` contains non-finite coordinates or if the
/// metric tensor is singular at `point` or any central-difference stencil evaluation.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::{SMatrix, SVector};
/// use math_explorer::pure_math::tensor::StaticRiemannianMetric;
/// use math_explorer::pure_math::tensor::christoffel::christoffel_symbols_fixed;
///
/// let polar = StaticRiemannianMetric::<_, 2>::new(|p: &SVector<f64, 2>| {
///     let r = p[0];
///     Ok(SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, r * r))
/// });
/// let pt = SVector::<f64, 2>::new(2.0, std::f64::consts::FRAC_PI_4);
/// let gammas = christoffel_symbols_fixed(&polar, &pt).unwrap();
/// assert!((gammas[0][(1, 1)] - (-2.0)).abs() < 1e-6);
/// assert!((gammas[1][(0, 1)] - 0.5).abs() < 1e-6);
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
        AdRiemannianMetric, RiemannianMetric, StaticAdMetric, StaticRiemannianMetric,
        TwoSphereMetric,
    };
    use math_scalars::{Dual, Scalar};

    #[test]
    #[verified_engine::verified]
    fn test_christoffel_symbols_fixed_zero_alloc_and_matches_dynamic() {
        let dyn_polar = RiemannianMetric::new(|p: &DVector<f64>| {
            let r = p[0];
            DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, r * r])
        });
        let fixed_polar = StaticRiemannianMetric::<_, 2>::new(|p: &SVector<f64, 2>| {
            let r = p[0];
            Ok(SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, r * r))
        });
        let sphere = TwoSphereMetric::new(2.5).expect("valid radius");

        let dyn_pt = DVector::from_vec(vec![3.0, 0.8]);
        let fixed_pt = SVector::<f64, 2>::new(3.0, 0.8);
        let dyn_gamma = christoffel_symbols(&dyn_polar, &dyn_pt).expect("dynamic christoffel");

        let ((fixed_gamma, sphere_gamma), metrics) =
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                let g1 = christoffel_symbols_fixed(&fixed_polar, &fixed_pt).expect("fixed polar");
                let g2 = christoffel_symbols_fixed(&sphere, &fixed_pt).expect("fixed sphere");
                (g1, g2)
            });
        assert!(metrics.function_calls > 0);

        for k in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert!((fixed_gamma[k][(i, j)] - dyn_gamma[k][(i, j)]).abs() < 1e-6);
                }
            }
        }
        let exact_sphere = sphere.christoffel_at(&fixed_pt).expect("exact sphere");
        assert!((sphere_gamma[0][(1, 1)] - exact_sphere[0][(1, 1)]).abs() < 1e-14);
        assert!((sphere_gamma[1][(0, 1)] - exact_sphere[1][(0, 1)]).abs() < 1e-14);
    }

    #[test]
    #[verified_engine::verified]
    fn test_christoffel_ad_spherical_and_schwarzschild_within_1e_13() {
        let sph_ad = AdRiemannianMetric::new(|p: &[Dual<f64>]| {
            let (r, th) = (p[0], p[1]);
            let r2 = r * r;
            let st = th.sin();
            vec![
                Dual::ONE,
                Dual::ZERO,
                Dual::ZERO,
                Dual::ZERO,
                r2,
                Dual::ZERO,
                Dual::ZERO,
                Dual::ZERO,
                r2 * st * st,
            ]
        });
        let (r, th, ph) = (3.4_f64, 0.85_f64, 1.2_f64);
        let pt = DVector::from_vec(vec![r, th, ph]);
        let g = christoffel_symbols(&sph_ad, &pt).expect("3D spherical AD christoffel");
        assert!((g[0][(1, 1)] - (-r)).abs() < 1e-14);
        assert!((g[0][(2, 2)] - (-r * th.sin().powi(2))).abs() < 1e-14);
        assert!((g[1][(0, 1)] - (1.0 / r)).abs() < 1e-14);
        assert!((g[1][(2, 2)] - (-th.sin() * th.cos())).abs() < 1e-14);
        assert!((g[2][(0, 2)] - (1.0 / r)).abs() < 1e-14);
        assert!((g[2][(1, 2)] - (th.cos() / th.sin())).abs() < 1e-14);

        let rs = 1.0_f64;
        let schw = StaticAdMetric::<_, 4>::new(move |x: &[Dual<f64>; 4]| {
            let r_d = x[1];
            let f: Dual<f64> = Dual::<f64>::ONE - Dual::<f64>::from_f64(rs) / r_d;
            let st = x[2].sin();
            let z: Dual<f64> = Dual::ZERO;
            Ok([
                [-f, z, z, z],
                [z, Dual::<f64>::ONE / f, z, z],
                [z, z, r_d * r_d, z],
                [z, z, z, r_d * r_d * st * st],
            ])
        });
        let spt = SVector::<f64, 4>::new(0.0, 5.0, 1.1, 0.4);
        let sg = christoffel_symbols_fixed(&schw, &spt).expect("4D Schwarzschild AD christoffel");
        let f_exact = 1.0 - rs / 5.0;
        let gamma_t_tr = rs / (2.0 * 5.0 * 5.0 * f_exact);
        let gamma_r_tt = 0.5 * f_exact * rs / (5.0 * 5.0);
        assert!((sg[0][(0, 1)] - gamma_t_tr).abs() < 1e-14);
        assert!((sg[1][(0, 0)] - gamma_r_tt).abs() < 1e-14);
    }
}
