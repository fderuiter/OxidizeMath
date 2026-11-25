// [cite:pure_math]
use super::super::Psu11;
use super::{DiskPoint, HyperbolicError};
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

/// Computes the signed Euclidean shoelace area of `vertices` to detect CCW (`> 0`) vs CW (`< 0`) winding.
#[verified_engine::verified]
fn signed_euclidean_area(vertices: &[Complex64]) -> f64 {
    assert!(vertices.len() >= 3);
    let n = vertices.len();
    let mut sum = 0.0;
    for i in 0..n {
        let a = vertices[i];
        let b = vertices[(i + 1) % n];
        sum += a.re * b.im - b.re * a.im;
    }
    debug_assert!(sum.is_finite());
    0.5 * sum
}

/// Computes the interior hyperbolic angle $\alpha_i \in [0, 2\pi)$ at vertex `curr`.
#[verified_engine::verified]
fn single_vertex_interior_angle(
    prev: Complex64,
    curr: Complex64,
    next: Complex64,
    is_ccw: bool,
) -> Result<f64, HyperbolicError> {
    if !curr.re.is_finite() || !curr.im.is_finite() || curr.norm_sqr() > 1.0 + 1.0e-9 {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "polygon vertex lies outside the closed unit disk",
        });
    }
    if (curr - prev).norm_sqr() <= 1.0e-22 || (next - curr).norm_sqr() <= 1.0e-22 {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "polygon has coincident adjacent vertices",
        });
    }
    if curr.norm_sqr() >= 1.0 - 1.0e-11 {
        return Ok(0.0);
    }
    let pt = DiskPoint::from_complex(curr)?;
    let to_origin = Psu11::translation_to_origin(pt);
    let u_prev = to_origin.apply_complex(prev);
    let u_next = to_origin.apply_complex(next);
    let ang_prev = u_prev.im.atan2(u_prev.re);
    let ang_next = u_next.im.atan2(u_next.re);
    let raw = if is_ccw {
        (ang_prev - ang_next).rem_euclid(TAU)
    } else {
        (ang_next - ang_prev).rem_euclid(TAU)
    };
    assert!(raw.is_finite() && (0.0..=TAU).contains(&raw));
    Ok(raw)
}

/// Computes the exact interior hyperbolic angles $\alpha_0, \dots, \alpha_{n-1}$ of a geodesic $n$-gon in $\overline{\mathbb{D}}$.
#[verified_engine::verified]
pub fn hyperbolic_polygon_angles(
    vertices: &[Complex64],
    out_angles: &mut [f64],
) -> Result<usize, HyperbolicError> {
    let n = vertices.len();
    if n < 3 {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "polygon requires at least 3 vertices",
        });
    }
    if out_angles.len() < n {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "output angle slice is smaller than vertex count",
        });
    }
    let area2 = signed_euclidean_area(vertices);
    if area2.abs() <= 1.0e-15 {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "polygon vertices are collinear or degenerate",
        });
    }
    let is_ccw = area2 > 0.0;
    for i in 0..n {
        let prev = vertices[(i + n - 1) % n];
        let curr = vertices[i];
        let next = vertices[(i + 1) % n];
        out_angles[i] = single_vertex_interior_angle(prev, curr, next, is_ccw)?;
    }
    assert!(n >= 3);
    Ok(n)
}

/// Computes the exact Gauss-Bonnet hyperbolic area $(n - 2)\pi - \sum_{i=0}^{n-1} \alpha_i$ ($K = -1$) of a simple geodesic $n$-gon.
#[verified_engine::verified]
pub fn gauss_bonnet_polygon_area(vertices: &[Complex64]) -> Result<f64, HyperbolicError> {
    let n = vertices.len();
    if n < 3 {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "polygon requires at least 3 vertices",
        });
    }
    let area2 = signed_euclidean_area(vertices);
    if area2.abs() <= 1.0e-15 {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "polygon vertices are collinear or degenerate",
        });
    }
    let is_ccw = area2 > 0.0;
    let mut angle_sum = 0.0;
    for i in 0..n {
        let prev = vertices[(i + n - 1) % n];
        let curr = vertices[i];
        let next = vertices[(i + 1) % n];
        angle_sum += single_vertex_interior_angle(prev, curr, next, is_ccw)?;
    }
    let gb_area = ((n - 2) as f64) * PI - angle_sum;
    if !gb_area.is_finite() || gb_area <= 0.0 {
        return Err(HyperbolicError::InvalidPolygon {
            reason: "polygon does not enclose positive hyperbolic area",
        });
    }
    assert!(gb_area > 0.0);
    Ok(gb_area)
}
