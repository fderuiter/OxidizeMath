//! Zero-allocation Gauss linking integral and spatial Gauss writhe quadrature.

// [cite:knot_braid_topology]

use super::KnotError;
use super::curves::ClosedLink3D;
use math_commons::math_kernel::types::flatten_2d_index;
use nalgebra::Point3;
use std::f64::consts::PI;

/// 2-point Gauss-Legendre nodes on $[0, 1]$: $\frac{1}{2} \mp \frac{1}{2\sqrt{3}}$.
const GL2_NODES: [f64; 2] = [0.211_324_865_405_187_13, 0.788_675_134_594_812_9];

/// Evaluates the 2-point Gauss-Legendre double integral over straight segments
/// $A_0 \to A_1$ and $B_0 \to B_1$:
/// $$\iint_{[0,1]^2} \frac{(r_1(s) - r_2(t)) \cdot (r_1'(s) \times r_2'(t))}{\|r_1(s) - r_2(t)\|^3}\,ds\,dt.$$
#[inline]
#[verified_engine::verified]
fn segment_pair_gauss_contribution(
    a0: &Point3<f64>,
    a1: &Point3<f64>,
    b0: &Point3<f64>,
    b1: &Point3<f64>,
) -> f64 {
    let da = a1 - a0;
    let db = b1 - b0;
    let cross = da.cross(&db);
    if cross.norm_squared() <= 1e-30 {
        return 0.0;
    }
    let mut accum = 0.0;
    for &s in &GL2_NODES {
        let pa = a0 + da * s;
        for &t in &GL2_NODES {
            let pb = b0 + db * t;
            let diff = pa - pb;
            let r2 = diff.norm_squared();
            if r2 > 1e-20 {
                let inv_r3 = 1.0 / (r2 * r2.sqrt());
                accum += diff.dot(&cross) * inv_r3;
            }
        }
    }
    0.25 * accum
}

/// Computes the continuous 3D Gauss writhe
/// $$W_r(C) = \frac{1}{4\pi} \oint_C \oint_C \frac{(r(s) - r(t)) \cdot (dr(s) \times dr(t))}{\|r(s) - r(t)\|^3}$$
/// of a closed polyline with **zero heap allocations**.
///
/// Identical and immediately adjacent segments ($i = j$ or $i \equiv j \pm 1 \pmod N$) are
/// coplanar/tangent-regular and contribute zero to the principal-value self-integral.
///
/// # Errors
/// Returns [`KnotError::InsufficientSamples`] if `curve.len() < 3`.
#[verified_engine::verified]
pub fn gauss_writhe_quadrature(curve: &[Point3<f64>]) -> Result<f64, KnotError> {
    let n = curve.len();
    if n < 3 {
        return Err(KnotError::InsufficientSamples {
            required: 3,
            actual: n,
        });
    }
    let mut sum = 0.0;
    for i in 0..n {
        let a0 = &curve[i];
        let a1 = &curve[(i + 1) % n];
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            let b0 = &curve[j];
            let b1 = &curve[(j + 1) % n];
            sum += 2.0 * segment_pair_gauss_contribution(a0, a1, b0, b1);
        }
    }
    Ok(sum / (4.0 * PI))
}

/// Computes the Gauss linking integral
/// $$\operatorname{Lk}(C_1, C_2) = \frac{1}{4\pi} \oint_{C_1} \oint_{C_2} \frac{(r_1 - r_2) \cdot (dr_1 \times dr_2)}{\|r_1 - r_2\|^3}$$
/// between two disjoint closed polylines with **zero heap allocations**.
///
/// # Errors
/// Returns [`KnotError::InsufficientSamples`] if either curve has fewer than 3 sample points.
#[verified_engine::verified]
pub fn gauss_linking_quadrature(
    curve_a: &[Point3<f64>],
    curve_b: &[Point3<f64>],
) -> Result<f64, KnotError> {
    let na = curve_a.len();
    let nb = curve_b.len();
    if na < 3 || nb < 3 {
        return Err(KnotError::InsufficientSamples {
            required: 3,
            actual: na.min(nb),
        });
    }
    let mut sum = 0.0;
    for i in 0..na {
        let a0 = &curve_a[i];
        let a1 = &curve_a[(i + 1) % na];
        for j in 0..nb {
            let b0 = &curve_b[j];
            let b1 = &curve_b[(j + 1) % nb];
            sum += segment_pair_gauss_contribution(a0, a1, b0, b1);
        }
    }
    Ok(sum / (4.0 * PI))
}

/// Rounds [`gauss_linking_quadrature`] to the exact integer topological linking number $\in \mathbb{Z}$.
///
/// # Errors
/// Propagates any [`KnotError`] from [`gauss_linking_quadrature`].
#[verified_engine::verified]
pub fn gauss_linking_number(
    curve_a: &[Point3<f64>],
    curve_b: &[Point3<f64>],
) -> Result<i64, KnotError> {
    let raw = gauss_linking_quadrature(curve_a, curve_b)?;
    Ok(raw.round() as i64)
}

/// Fills an $m \times m$ row-major matrix `out_matrix` (`m == link.component_count()`) with
/// exact integer pairwise Gauss linking numbers (`0` on the diagonal) with **zero heap allocations**.
///
/// # Errors
/// Returns [`KnotError`] if `out_matrix.len() != m * m` or any component has fewer than 3 samples.
#[verified_engine::verified]
pub fn pairwise_linking_numbers_in_place(
    link: &ClosedLink3D,
    out_matrix: &mut [i64],
) -> Result<(), KnotError> {
    let m = link.component_count();
    let expected = m * m;
    if out_matrix.len() != expected {
        return Err(KnotError::BufferLengthMismatch {
            expected,
            actual: out_matrix.len(),
        });
    }
    for i in 0..m {
        out_matrix[flatten_2d_index(i, i, m)] = 0;
        for j in (i + 1)..m {
            let lk = gauss_linking_number(&link.components[i], &link.components[j])?;
            out_matrix[flatten_2d_index(i, j, m)] = lk;
            out_matrix[flatten_2d_index(j, i, m)] = lk;
        }
    }
    Ok(())
}

/// Returns a symmetric $m \times m$ grid (`Vec<Vec<i64>>`) of pairwise Gauss linking numbers.
///
/// # Errors
/// Propagates any [`KnotError`] from [`gauss_linking_number`].
#[verified_engine::verified]
pub fn pairwise_linking_matrix(link: &ClosedLink3D) -> Result<Vec<Vec<i64>>, KnotError> {
    let m = link.component_count();
    let mut flat = vec![0_i64; m * m];
    pairwise_linking_numbers_in_place(link, &mut flat)?;
    let mut rows = Vec::with_capacity(m);
    for i in 0..m {
        let mut row = Vec::with_capacity(m);
        for j in 0..m {
            row.push(flat[flatten_2d_index(i, j, m)]);
        }
        rows.push(row);
    }
    Ok(rows)
}
