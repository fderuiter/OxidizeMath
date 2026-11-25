// [cite:quasiperfect_alcf]
use super::coordinates::{DifferentiableCoordinates, OrthogonalCoordinateSystem};
use math_scalars::{Dual, DualVec, HyperDual};
use nalgebra::Vector3;

#[inline]
#[verified_engine::verified]
fn partial_derivative_4th<F>(i: usize, point: &Vector3<f64>, f: &F) -> f64
where
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(i < 3);
    let step = 1e-4 * point[i].abs().max(1.0);
    let mut p_p1 = *point;
    let mut p_m1 = *point;
    let mut p_p2 = *point;
    let mut p_m2 = *point;
    p_p1[i] += step;
    p_m1[i] -= step;
    p_p2[i] += 2.0 * step;
    p_m2[i] -= 2.0 * step;
    (-f(&p_p2) + 8.0 * f(&p_p1) - 8.0 * f(&p_m1) + f(&p_m2)) / (12.0 * step)
}

#[inline]
#[verified_engine::verified]
fn second_partial_derivative_4th<F>(i: usize, point: &Vector3<f64>, f0: f64, f: &F) -> f64
where
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(i < 3);
    let step = 1e-4 * point[i].abs().max(1.0);
    let mut p_p1 = *point;
    let mut p_m1 = *point;
    let mut p_p2 = *point;
    let mut p_m2 = *point;
    p_p1[i] += step;
    p_m1[i] -= step;
    p_p2[i] += 2.0 * step;
    p_m2[i] -= 2.0 * step;
    (-f(&p_p2) + 16.0 * f(&p_p1) - 30.0 * f0 + 16.0 * f(&p_m1) - f(&p_m2)) / (12.0 * step * step)
}

/// Computes the exact curvilinear gradient $\nabla \Phi$ in a single [`DualVec<f64, 3>`] forward pass.
#[verified_engine::verified]
pub fn gradient_ad<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: DifferentiableCoordinates + ?Sized,
    F: Fn(&[DualVec<f64, 3>; 3]) -> DualVec<f64, 3>,
{
    let h = coords.scale_factors(point);
    let seed = DualVec::<f64, 3>::jacobian_seed([point[0], point[1], point[2]]);
    let dphi = field(&seed).eps;
    let mut out = Vector3::zeros();
    for i in 0..3 {
        out[i] = if h[i].abs() < 1e-12 {
            0.0
        } else {
            dphi[i] / h[i]
        };
    }
    out
}

/// Computes the exact curvilinear divergence $\nabla \cdot \mathbf{A}$ in a single [`DualVec<f64, 3>`] pass.
#[verified_engine::verified]
pub fn divergence_ad<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: DifferentiableCoordinates + ?Sized,
    F: Fn(&[DualVec<f64, 3>; 3]) -> [DualVec<f64, 3>; 3],
{
    let seed = DualVec::<f64, 3>::jacobian_seed([point[0], point[1], point[2]]);
    let h_d = coords.scale_factors_scalar(&seed);
    let h1h2h3 = h_d[0].val * h_d[1].val * h_d[2].val;
    if h1h2h3.abs() < 1e-12 {
        return 0.0;
    }
    let a_d = field(&seed);
    let mut sum = 0.0;
    for (i, &a_i) in a_d.iter().enumerate() {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        sum += (h_d[j] * h_d[k] * a_i).eps[i];
    }
    sum / h1h2h3
}

/// Computes the exact curvilinear curl $\nabla \times \mathbf{A}$ in a single [`DualVec<f64, 3>`] pass.
#[verified_engine::verified]
pub fn curl_ad<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: DifferentiableCoordinates + ?Sized,
    F: Fn(&[DualVec<f64, 3>; 3]) -> [DualVec<f64, 3>; 3],
{
    let seed = DualVec::<f64, 3>::jacobian_seed([point[0], point[1], point[2]]);
    let h_d = coords.scale_factors_scalar(&seed);
    let a_d = field(&seed);
    let ha = [h_d[0] * a_d[0], h_d[1] * a_d[1], h_d[2] * a_d[2]];
    let mut out = Vector3::zeros();
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let denom = h_d[j].val * h_d[k].val;
        out[i] = if denom.abs() < 1e-12 {
            0.0
        } else {
            (ha[k].eps[j] - ha[j].eps[k]) / denom
        };
    }
    out
}

/// Computes the exact curvilinear Laplacian $\nabla^2 \Phi$ via [`HyperDual<f64>`] and [`Dual<f64>`]
/// without nested finite-difference cancellation.
#[verified_engine::verified]
pub fn laplacian_ad<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: DifferentiableCoordinates + ?Sized,
    F: Fn(&[HyperDual<f64>; 3]) -> HyperDual<f64>,
{
    let h0 = coords.scale_factors(point);
    let h1h2h3 = h0[0] * h0[1] * h0[2];
    if h1h2h3.abs() < 1e-12 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..3 {
        if h0[i].abs() < 1e-12 {
            continue;
        }
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let mut p_dual = [
            Dual::constant(point[0]),
            Dual::constant(point[1]),
            Dual::constant(point[2]),
        ];
        p_dual[i] = Dual::variable(point[i]);
        let hd_scale = coords.scale_factors_scalar(&p_dual);
        let weight = (hd_scale[j] * hd_scale[k]) / hd_scale[i];

        let mut p_hd = [
            HyperDual::constant(point[0]),
            HyperDual::constant(point[1]),
            HyperDual::constant(point[2]),
        ];
        p_hd[i] = HyperDual::new(point[i], 1.0, 1.0, 0.0);
        let phi_jet = field(&p_hd);
        sum += weight.eps * phi_jet.eps1 + weight.val * phi_jet.eps12;
    }
    sum / h1h2h3
}

/// Computes the gradient of a scalar field $\nabla \Phi$ in the local curvilinear basis.
#[verified_engine::verified]
pub fn gradient<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: OrthogonalCoordinateSystem + ?Sized,
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let factors = coords.scale_factors(point);
    let mut grad = Vector3::zeros();
    for i in 0..3 {
        grad[i] = if factors[i].abs() < 1e-12 {
            0.0
        } else {
            partial_derivative_4th(i, point, &field) / factors[i]
        };
    }
    grad
}

/// Computes the divergence of a vector field $\nabla \cdot \mathbf{A}$ using exact scale-factor partials.
#[verified_engine::verified]
pub fn divergence<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem + ?Sized,
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let h = coords.scale_factors(point);
    let h1h2h3 = h[0] * h[1] * h[2];
    if h1h2h3.abs() < 1e-12 {
        return 0.0;
    }
    let dh = coords.scale_factor_partials(point);
    let a0 = field(point);
    let mut sum = 0.0;
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let da_di = partial_derivative_4th(i, point, &|p: &Vector3<f64>| field(p)[i]);
        let dh_jk_di = dh[j][i] * h[k] + h[j] * dh[k][i];
        sum += dh_jk_di * a0[i] + h[j] * h[k] * da_di;
    }
    sum / h1h2h3
}

/// Computes the curl of a vector field $\nabla \times \mathbf{A}$ using exact scale-factor partials.
#[verified_engine::verified]
pub fn curl<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> Vector3<f64>
where
    S: OrthogonalCoordinateSystem + ?Sized,
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let h = coords.scale_factors(point);
    let dh = coords.scale_factor_partials(point);
    let a0 = field(point);
    let mut result = Vector3::zeros();

    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let denom = h[j] * h[k];
        if denom.abs() < 1e-12 {
            result[i] = 0.0;
            continue;
        }
        let dak_dj = partial_derivative_4th(j, point, &|p: &Vector3<f64>| field(p)[k]);
        let daj_dk = partial_derivative_4th(k, point, &|p: &Vector3<f64>| field(p)[j]);
        let term_1 = dh[k][j] * a0[k] + h[k] * dak_dj;
        let term_2 = dh[j][k] * a0[j] + h[j] * daj_dk;
        result[i] = (term_1 - term_2) / denom;
    }
    result
}

/// Computes the Laplacian of a scalar field $\nabla^2 \Phi$ using exact scale-factor partials
/// and un-nested fourth-order second-derivative stencils.
#[verified_engine::verified]
pub fn laplacian<S, F>(coords: &S, field: F, point: &Vector3<f64>) -> f64
where
    S: OrthogonalCoordinateSystem + ?Sized,
    F: Fn(&Vector3<f64>) -> f64,
{
    debug_assert!(point.iter().all(|x| x.is_finite()));
    let h = coords.scale_factors(point);
    let h1h2h3 = h[0] * h[1] * h[2];
    if h1h2h3.abs() < 1e-12 {
        return 0.0;
    }
    let dh = coords.scale_factor_partials(point);
    let f0 = field(point);
    let mut sum = 0.0;

    for i in 0..3 {
        if h[i].abs() < 1e-12 {
            continue;
        }
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        let w = (h[j] * h[k]) / h[i];
        let num_prime = (dh[j][i] * h[k] + h[j] * dh[k][i]) * h[i] - h[j] * h[k] * dh[i][i];
        let dw_di = num_prime / (h[i] * h[i]);
        let dphi = partial_derivative_4th(i, point, &field);
        let d2phi = second_partial_derivative_4th(i, point, f0, &field);
        sum += dw_di * dphi + w * d2phi;
    }
    sum / h1h2h3
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::vector_calculus::coordinates::{Cylindrical, Spherical};
    use math_scalars::Scalar;

    #[test]
    fn test_spherical_and_cylindrical_ad_operators_machine_precision() {
        let sph = Spherical;
        let pt = Vector3::new(2.3, 0.85, 1.15);
        let (r, theta, phi) = (pt[0], pt[1], pt[2]);

        // Scalar field \Phi(r, \theta, \phi) = r^2 \cos(\theta) \sin(\phi) = z * y / sin(\theta)... wait:
        // In Cartesian: r^2 sin(\theta) cos(\theta) sin(\phi) = y z / r -> let's test harmonic \Phi = r^2 cos(\theta) * ...
        // Better: exact harmonic Y_1^0 * r = r cos(\theta) = z (Laplacian == 0!) and r^3 cos(\theta):
        // \nabla^2 (r^3 \cos\theta) = (1/r^2)\partial_r(r^2 * 3r^2 \cos\theta) - 2 r \cos\theta = (12 - 2) r \cos\theta = 10 r \cos\theta!
        let lap = laplacian_ad(&sph, |q| q[0] * q[0] * q[0] * q[1].cos(), &pt);
        let exact_lap = 10.0 * r * theta.cos();
        assert!(
            ((lap - exact_lap) / exact_lap.abs()).abs() < 1e-14,
            "lap={lap}, exact={exact_lap}"
        );

        let grad = gradient_ad(&sph, |q| q[0] * q[0] * q[1].sin() * q[2].cos(), &pt);
        let exact_grad = Vector3::new(
            2.0 * r * theta.sin() * phi.cos(),
            r * theta.cos() * phi.cos(),
            -r * phi.sin(),
        );
        assert!((grad - exact_grad).norm() / exact_grad.norm() < 1e-14);

        // Spherical divergence & curl of A = (r^2, r sin\theta, r cos\theta sin\phi)
        let div = divergence_ad(
            &sph,
            |q| {
                [
                    q[0] * q[0],
                    q[0] * q[1].sin(),
                    q[0] * q[1].cos() * q[2].sin(),
                ]
            },
            &pt,
        );
        // div = (1 / (r^2 sin\theta)) [ \partial_r(r^2 sin\theta * r^2) + \partial_\theta(r sin\theta * r sin\theta) + \partial_\phi(r * r cos\theta sin\phi) ]
        //     = 4 r + 2 cos\theta + cot\theta cos\phi
        let exact_div = 4.0 * r + 2.0 * theta.cos() + (theta.cos() / theta.sin()) * phi.cos();
        assert!(((div - exact_div) / exact_div.abs()).abs() < 1e-14);

        let cyl = Cylindrical;
        let cpt = Vector3::new(1.8, 0.6, -0.7);
        let (rho, cphi, z) = (cpt[0], cpt[1], cpt[2]);
        // A = (rho z, rho^2, sin(phi) * z^2)
        // curl_rho = (1/rho) \partial_\phi A_z - \partial_z A_\phi = (z^2 cos\phi)/rho - 0
        // curl_phi = \partial_z A_\rho - \partial_\rho A_z = rho
        // curl_z = (1/rho) [ \partial_\rho(rho * rho^2) - \partial_\phi(rho z) ] = 3 rho
        let c_ad = curl_ad(
            &cyl,
            |q| [q[0] * q[2], q[0] * q[0], q[1].sin() * q[2] * q[2]],
            &cpt,
        );
        let exact_curl = Vector3::new((z * z * cphi.cos()) / rho, rho, 3.0 * rho);
        assert!((c_ad - exact_curl).norm() / exact_curl.norm() < 1e-14);
    }
}
