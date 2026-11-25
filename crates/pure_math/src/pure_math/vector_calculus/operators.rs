// [cite:quasiperfect_alcf]
//! Curvilinear `gradient`, `divergence`, `curl`, and `laplacian` operators with exact
//! `DualVec<f64, 3>` / `HyperDual<f64>` forward-AD variants and cancellation-free metric expansion.

use math_scalars::{DualVec, HyperDual};
use nalgebra::Vector3;

use super::coordinates::{DifferentiableCoordSystem, OrthogonalCoordinateSystem};

/// Computes the exact curvilinear gradient $\nabla \Phi$ in a single `DualVec<f64, 3>` pass.
#[verified_engine::verified]
pub fn gradient_ad<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: DifferentiableCoordSystem,
    F: Fn(&[DualVec<f64, 3>; 3]) -> DualVec<f64, 3>,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let h = coords.scale_factors(point);
    let seeded = [
        DualVec::variable(point[0], 0),
        DualVec::variable(point[1], 1),
        DualVec::variable(point[2], 2),
    ];
    let phi = field(&seeded);
    Vector3::new(phi.eps[0] / h[0], phi.eps[1] / h[1], phi.eps[2] / h[2])
}

/// Computes the exact curvilinear divergence $\nabla \cdot \mathbf{A}$ in a single `DualVec<f64, 3>` pass.
#[verified_engine::verified]
pub fn divergence_ad<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: DifferentiableCoordSystem,
    F: Fn(&[DualVec<f64, 3>; 3]) -> [DualVec<f64, 3>; 3],
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let seeded = [
        DualVec::variable(point[0], 0),
        DualVec::variable(point[1], 1),
        DualVec::variable(point[2], 2),
    ];
    let h = coords.scale_factors_scalar(&seeded);
    let vol = h[0].val * h[1].val * h[2].val;
    if vol.abs() < 1e-12 {
        return 0.0;
    }
    let a = field(&seeded);
    let t0 = h[1] * h[2] * a[0];
    let t1 = h[2] * h[0] * a[1];
    let t2 = h[0] * h[1] * a[2];
    (t0.eps[0] + t1.eps[1] + t2.eps[2]) / vol
}

/// Computes the exact curvilinear curl $\nabla \times \mathbf{A}$ in a single `DualVec<f64, 3>` pass.
#[verified_engine::verified]
pub fn curl_ad<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: DifferentiableCoordSystem,
    F: Fn(&[DualVec<f64, 3>; 3]) -> [DualVec<f64, 3>; 3],
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let seeded = [
        DualVec::variable(point[0], 0),
        DualVec::variable(point[1], 1),
        DualVec::variable(point[2], 2),
    ];
    let h = coords.scale_factors_scalar(&seeded);
    let a = field(&seeded);
    let ha = [h[0] * a[0], h[1] * a[1], h[2] * a[2]];
    let mut out = Vector3::zeros();
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let denom = h[j].val * h[k].val;
        out[i] = if denom.abs() < 1e-12 {
            0.0
        } else {
            (ha[k].eps[j] - ha[j].eps[k]) / denom
        };
    }
    out
}

/// Computes the exact curvilinear Laplacian $\nabla^2 \Phi$ using [`HyperDual`] second derivatives.
#[verified_engine::verified]
pub fn laplacian_ad<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: DifferentiableCoordSystem,
    F: Fn(&[HyperDual<f64>; 3]) -> HyperDual<f64>,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let h0 = coords.scale_factors(point);
    let vol = h0[0] * h0[1] * h0[2];
    if vol.abs() < 1e-12 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let mut p_hd = [
            HyperDual::constant(point[0]),
            HyperDual::constant(point[1]),
            HyperDual::constant(point[2]),
        ];
        p_hd[i] = HyperDual::second_order(point[i]);
        let h = coords.scale_factors_scalar(&p_hd);
        let phi = field(&p_hd);
        let w = (h[j] * h[k]) / h[i];
        sum += w.eps1 * phi.eps1 + w.val * phi.eps12;
    }
    sum / vol
}

#[verified_engine::verified]
fn partial_4th<F>(i: usize, point: &Vector3<f64>, h: f64, f: F) -> f64
where
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(i < 3);
    let mut p_p2 = *point;
    let mut p_p1 = *point;
    let mut p_m1 = *point;
    let mut p_m2 = *point;
    p_p2[i] += 2.0 * h;
    p_p1[i] += h;
    p_m1[i] -= h;
    p_m2[i] -= 2.0 * h;
    (-f(&p_p2) + 8.0 * f(&p_p1) - 8.0 * f(&p_m1) + f(&p_m2)) / (12.0 * h)
}

#[verified_engine::verified]
fn second_partial_4th<F>(i: usize, point: &Vector3<f64>, h: f64, f0: f64, f: F) -> f64
where
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(i < 3);
    let mut p_p2 = *point;
    let mut p_p1 = *point;
    let mut p_m1 = *point;
    let mut p_m2 = *point;
    p_p2[i] += 2.0 * h;
    p_p1[i] += h;
    p_m1[i] -= h;
    p_m2[i] -= 2.0 * h;
    (-f(&p_p2) + 16.0 * f(&p_p1) - 30.0 * f0 + 16.0 * f(&p_m1) - f(&p_m2)) / (12.0 * h * h)
}

/// Computes the gradient of a scalar field $\nabla \Phi$ in local curvilinear basis components.
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
        let step = 1e-4 * point[i].abs().max(1.0);
        grad[i] = partial_4th(i, point, step, &field) / factors[i];
    }
    grad
}

/// Computes the divergence of a vector field $\nabla \cdot \mathbf{A}$ in curvilinear coordinates.
#[verified_engine::verified]
pub fn divergence<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let factors = coords.scale_factors(point);
    let h1h2h3 = factors[0] * factors[1] * factors[2];
    if h1h2h3.abs() < 1e-12 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let step = 1e-4 * point[i].abs().max(1.0);
        sum += partial_4th(i, point, step, |p| {
            let h = coords.scale_factors(p);
            h[j] * h[k] * field(p)[i]
        });
    }
    sum / h1h2h3
}

/// Computes the curl of a vector field $\nabla \times \mathbf{A}$ in local curvilinear components.
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
        let sj = 1e-4 * point[j].abs().max(1.0);
        let sk = 1e-4 * point[k].abs().max(1.0);
        let t1 = partial_4th(j, point, sj, |p| coords.scale_factors(p)[k] * field(p)[k]);
        let t2 = partial_4th(k, point, sk, |p| coords.scale_factors(p)[j] * field(p)[j]);
        result[i] = (t1 - t2) / (h[j] * h[k]);
    }
    result
}

/// Computes the Laplacian of a scalar field $\nabla^2 \Phi$ using exact AD scale-factor derivatives
/// when available and single-level 4th-order stencils (avoiding nested finite-difference cancellation).
#[verified_engine::verified]
pub fn laplacian<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    if let Some((h, dh)) = coords.scale_factors_jet1(point) {
        let vol = h[0] * h[1] * h[2];
        if vol.abs() < 1e-12 {
            return 0.0;
        }
        let f0 = field(point);
        let mut sum = 0.0;
        for i in 0..3 {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;
            let step = 1e-3 * point[i].abs().max(1.0);
            let dphi = partial_4th(i, point, step, &field);
            let d2phi = second_partial_4th(i, point, step, f0, &field);
            let w0 = (h[j] * h[k]) / h[i];
            let dw = ((dh[j][i] * h[k] + h[j] * dh[k][i]) * h[i] - h[j] * h[k] * dh[i][i])
                / (h[i] * h[i]);
            sum += dw * dphi + w0 * d2phi;
        }
        return sum / vol;
    }

    let h_factors = coords.scale_factors(point);
    let h1h2h3 = h_factors[0] * h_factors[1] * h_factors[2];
    if h1h2h3.abs() < 1e-12 {
        return 0.0;
    }
    let step = 1e-4;
    let mut sum = 0.0;
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        sum += partial_4th(i, point, step, |p| {
            let h = coords.scale_factors(p);
            (h[j] * h[k] / h[i]) * partial_4th(i, p, step, &field)
        });
    }
    sum / h1h2h3
}

#[cfg(test)]
mod ad_tests {
    use super::*;
    use crate::pure_math::vector_calculus::coordinates::Spherical;
    use math_scalars::Scalar;

    #[test]
    fn test_spherical_ad_operators_machine_precision() {
        let sph = Spherical;
        let pt = Vector3::new(2.2, 0.85, 1.3);
        let lap = laplacian_ad(&sph, |p| p[0].sqr() * p[1].cos(), &pt);
        let exact_lap = 4.0 * pt[1].cos();
        assert!((lap - exact_lap).abs() < 1e-13);

        let grad = gradient_ad(&sph, |p| p[0].sqr() * p[1].cos(), &pt);
        assert!((grad[0] - 2.0 * pt[0] * pt[1].cos()).abs() < 1e-14);
        assert!((grad[1] + pt[0] * pt[1].sin()).abs() < 1e-14);
        assert!(grad[2].abs() < 1e-15);

        let div = divergence_ad(&sph, |p| [p[0].sqr(), Scalar::ZERO, Scalar::ZERO], &pt);
        assert!((div - 4.0 * pt[0]).abs() < 1e-14);

        let rot = curl_ad(
            &sph,
            |p| [Scalar::ZERO, Scalar::ZERO, p[0] * p[1].sin()],
            &pt,
        );
        assert!((rot[0] - 2.0 * pt[1].cos()).abs() < 1e-14);
        assert!((rot[1] + 2.0 * pt[1].sin()).abs() < 1e-14);
        assert!(rot[2].abs() < 1e-15);
    }
}
