//! Curvilinear integral theorems (Gauss divergence, Stokes circulation, and Helmholtz-Hodge split).

// [cite:vector_calculus]

use super::coordinates::OrthogonalCoordinateSystem;
use super::operators::{curl, divergence};
use nalgebra::Vector3;

pub use super::helmholtz::{HelmholtzSummary, HelmholtzWorkspace, helmholtz_decompose};

/// A rectangular box domain in curvilinear coordinate space defined by `min` and `max` bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Domain {
    /// Lower coordinate bounds `(u1_min, u2_min, u3_min)`.
    pub min: Vector3<f64>,
    /// Upper coordinate bounds `(u1_max, u2_max, u3_max)`.
    pub max: Vector3<f64>,
}

/// An oriented curvilinear coordinate surface patch $\Sigma$ for Stokes' theorem verification.
///
/// Holds coordinate `normal_axis` (`0`, `1`, or `2`) constant at `fixed_value`, while the cyclic
/// tangent coordinates `(axis_u, axis_v) = ((normal_axis + 1) % 3, (normal_axis + 2) % 3)` sweep
/// `[u_range.0, u_range.1] x [v_range.0, v_range.1]` with right-handed orientation $+\hat{e}_n$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StokesPatch {
    /// Coordinate index (`0`, `1`, or `2`) normal to the patch.
    pub normal_axis: usize,
    /// Constant coordinate value along `normal_axis`.
    pub fixed_value: f64,
    /// Parameter interval `(u_min, u_max)` along `(normal_axis + 1) % 3`.
    pub u_range: (f64, f64),
    /// Parameter interval `(v_min, v_max)` along `(normal_axis + 2) % 3`.
    pub v_range: (f64, f64),
}

impl StokesPatch {
    /// Constructs a right-handed coordinate patch normal to `normal_axis` (`0..3`).
    #[must_use]
    #[verified_engine::verified]
    pub fn new(
        normal_axis: usize,
        fixed_value: f64,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) -> Self {
        assert!(normal_axis < 3, "normal_axis must be 0, 1, or 2");
        assert!(u_range.1 > u_range.0 && v_range.1 > v_range.0);
        Self {
            normal_axis,
            fixed_value,
            u_range,
            v_range,
        }
    }

    /// Evaluates a 3D curvilinear coordinate point `(u, v)` on this patch.
    #[must_use]
    #[inline]
    pub fn point_at(&self, u: f64, v: f64) -> Vector3<f64> {
        let mut p = Vector3::zeros();
        let axis_u = (self.normal_axis + 1) % 3;
        let axis_v = (self.normal_axis + 2) % 3;
        p[self.normal_axis] = self.fixed_value;
        p[axis_u] = u;
        p[axis_v] = v;
        p
    }
}

/// Verifies Gauss's Divergence Theorem:
/// $\iiint_V (\nabla \cdot \mathbf{A})\,dV = \oiint_{\partial V} (\mathbf{A} \cdot \mathbf{n})\,dS$.
///
/// Returns `(volume_integral, boundary_flux, |volume_integral - boundary_flux|)`.
#[verified_engine::verified]
pub fn verify_divergence_theorem<S, F>(
    coords: &S,
    domain: &Domain,
    field: F,
    steps: usize,
) -> (f64, f64, f64)
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64> + Copy,
{
    assert!(steps > 0, "Integration steps must be positive");
    let lhs = integrate_volume(coords, domain, steps, |p| divergence(coords, field, p));
    let rhs = integrate_surface_flux(coords, domain, steps, field);
    let diff = (lhs - rhs).abs();
    assert!(diff.is_finite());
    (lhs, rhs, diff)
}

/// Verifies Kelvin-Stokes' Circulation Theorem on an oriented curvilinear coordinate patch:
/// $\iint_\Sigma (\nabla \times \mathbf{A}) \cdot d\mathbf{S} = \oint_{\partial \Sigma} \mathbf{A} \cdot d\mathbf{l}$.
///
/// Returns `(surface_curl_flux, boundary_circulation, |surface_curl_flux - boundary_circulation|)`.
#[verified_engine::verified]
pub fn verify_stokes_theorem<S, F>(
    coords: &S,
    patch: &StokesPatch,
    field: F,
    steps: usize,
) -> (f64, f64, f64)
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64> + Copy,
{
    assert!(steps > 0, "Integration steps must be positive");
    assert!(patch.normal_axis < 3, "normal_axis must be in 0..3");
    let lhs = integrate_patch_curl(coords, patch, steps, field);
    let rhs = integrate_patch_boundary(coords, patch, steps, field);
    let diff = (lhs - rhs).abs();
    assert!(diff.is_finite());
    (lhs, rhs, diff)
}

#[verified_engine::verified]
fn integrate_patch_curl<S, F>(coords: &S, patch: &StokesPatch, steps: usize, field: F) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64> + Copy,
{
    assert!(steps > 0);
    let axis_n = patch.normal_axis;
    let axis_u = (axis_n + 1) % 3;
    let axis_v = (axis_n + 2) % 3;
    let du = (patch.u_range.1 - patch.u_range.0) / steps as f64;
    let dv = (patch.v_range.1 - patch.v_range.0) / steps as f64;
    let mut sum = 0.0;

    for i in 0..steps {
        let u = patch.u_range.0 + (i as f64 + 0.5) * du;
        for j in 0..steps {
            let v = patch.v_range.0 + (j as f64 + 0.5) * dv;
            let p = patch.point_at(u, v);
            let h = coords.scale_factors(&p);
            let d_area = h[axis_u] * h[axis_v] * du * dv;
            let curl_n = curl(coords, field, &p)[axis_n];
            sum += curl_n * d_area;
        }
    }
    sum
}

#[verified_engine::verified]
fn integrate_patch_boundary<S, F>(coords: &S, patch: &StokesPatch, steps: usize, field: F) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    assert!(steps > 0);
    let axis_u = (patch.normal_axis + 1) % 3;
    let axis_v = (patch.normal_axis + 2) % 3;
    let du = (patch.u_range.1 - patch.u_range.0) / steps as f64;
    let dv = (patch.v_range.1 - patch.v_range.0) / steps as f64;
    let mut circulation = 0.0;

    for i in 0..steps {
        let u = patch.u_range.0 + (i as f64 + 0.5) * du;
        let p_bottom = patch.point_at(u, patch.v_range.0);
        let p_top = patch.point_at(u, patch.v_range.1);
        circulation += field(&p_bottom)[axis_u] * coords.scale_factors(&p_bottom)[axis_u] * du;
        circulation -= field(&p_top)[axis_u] * coords.scale_factors(&p_top)[axis_u] * du;
    }

    for j in 0..steps {
        let v = patch.v_range.0 + (j as f64 + 0.5) * dv;
        let p_right = patch.point_at(patch.u_range.1, v);
        let p_left = patch.point_at(patch.u_range.0, v);
        circulation += field(&p_right)[axis_v] * coords.scale_factors(&p_right)[axis_v] * dv;
        circulation -= field(&p_left)[axis_v] * coords.scale_factors(&p_left)[axis_v] * dv;
    }

    circulation
}

#[verified_engine::verified]
fn integrate_volume<S, F>(coords: &S, domain: &Domain, steps: usize, func: F) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> f64,
{
    assert!(steps > 0);
    let d = (domain.max - domain.min) / (steps as f64);
    let mut sum = 0.0;

    for i in 0..steps {
        for j in 0..steps {
            for k in 0..steps {
                let u1 = domain.min[0] + (i as f64 + 0.5) * d[0];
                let u2 = domain.min[1] + (j as f64 + 0.5) * d[1];
                let u3 = domain.min[2] + (k as f64 + 0.5) * d[2];
                let p = Vector3::new(u1, u2, u3);

                let h = coords.scale_factors(&p);
                let volume_element = h[0] * h[1] * h[2] * d[0] * d[1] * d[2];
                sum += func(&p) * volume_element;
            }
        }
    }
    sum
}

#[verified_engine::verified]
fn integrate_axis_face_pair<S, F>(
    coords: &S,
    domain: &Domain,
    axis: usize,
    steps: usize,
    field: &F,
) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    assert!(axis < 3 && steps > 0);
    let a = (axis + 1) % 3;
    let b = (axis + 2) % 3;
    let d = (domain.max - domain.min) / (steps as f64);
    let mut flux = 0.0;

    for ia in 0..steps {
        let ua = domain.min[a] + (ia as f64 + 0.5) * d[a];
        for ib in 0..steps {
            let ub = domain.min[b] + (ib as f64 + 0.5) * d[b];

            let mut p_min = Vector3::zeros();
            p_min[axis] = domain.min[axis];
            p_min[a] = ua;
            p_min[b] = ub;
            let h_min = coords.scale_factors(&p_min);
            flux -= field(&p_min)[axis] * h_min[a] * h_min[b] * d[a] * d[b];

            let mut p_max = p_min;
            p_max[axis] = domain.max[axis];
            let h_max = coords.scale_factors(&p_max);
            flux += field(&p_max)[axis] * h_max[a] * h_max[b] * d[a] * d[b];
        }
    }
    flux
}

#[verified_engine::verified]
fn integrate_surface_flux<S, F>(coords: &S, domain: &Domain, steps: usize, field: F) -> f64
where
    S: OrthogonalCoordinateSystem,
    F: Fn(&Vector3<f64>) -> Vector3<f64>,
{
    assert!(steps > 0);
    integrate_axis_face_pair(coords, domain, 0, steps, &field)
        + integrate_axis_face_pair(coords, domain, 1, steps, &field)
        + integrate_axis_face_pair(coords, domain, 2, steps, &field)
}
