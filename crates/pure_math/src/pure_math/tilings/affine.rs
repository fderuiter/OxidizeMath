//! 2D Euclidean similarity transforms and `Tile(a, b)` polykite base polygons.

use math_commons::contours::Point2D;
use std::f64::consts::PI;

/// 2D affine matrix `[a, b, c, d, tx, ty]` mapping `(x, y)` to `(a*x + b*y + tx, c*x + d*y + ty)`.
pub(crate) type Affine2D = [f64; 6];

/// Identity 2D affine transform.
#[verified_engine::verified]
pub(crate) fn identity() -> Affine2D {
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}

/// Pure 2D translation by `(tx, ty)`.
#[verified_engine::verified]
pub(crate) fn translate(tx: f64, ty: f64) -> Affine2D {
    [1.0, 0.0, 0.0, 1.0, tx, ty]
}

/// Horizontal reflection `(x, y) -> (-x, y)`.
#[verified_engine::verified]
pub(crate) fn reflect_x() -> Affine2D {
    [-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
}

/// Counter-clockwise rotation by `deg` degrees about the origin.
#[verified_engine::verified]
pub(crate) fn rotate_deg(deg: f64) -> Affine2D {
    let rad = deg * (PI / 180.0);
    let (s, c) = rad.sin_cos();
    [c, -s, s, c, 0.0, 0.0]
}

/// Composes `lhs * rhs` (applies `rhs` first, then `lhs`).
#[verified_engine::verified]
pub(crate) fn mul_affine(lhs: Affine2D, rhs: Affine2D) -> Affine2D {
    let [a1, b1, c1, d1, tx1, ty1] = lhs;
    let [a2, b2, c2, d2, tx2, ty2] = rhs;
    [
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        a1 * tx2 + b1 * ty2 + tx1,
        c1 * tx2 + d1 * ty2 + ty1,
    ]
}

/// Applies `mat` to point `pt`.
#[verified_engine::verified]
pub(crate) fn apply_point(mat: Affine2D, pt: Point2D) -> Point2D {
    [
        mat[0] * pt[0] + mat[1] * pt[1] + mat[4],
        mat[2] * pt[0] + mat[3] * pt[1] + mat[5],
    ]
}

/// Applies `mat` to a slice of points.
#[verified_engine::verified]
pub(crate) fn transform_polygon(mat: Affine2D, poly: &[Point2D]) -> Vec<Point2D> {
    poly.iter().map(|&pt| apply_point(mat, pt)).collect()
}

/// Determinant `a*d - b*c` of the linear part (`< 0` indicates orientation reflection).
#[verified_engine::verified]
pub(crate) fn det_affine(mat: Affine2D) -> f64 {
    mat[0] * mat[3] - mat[1] * mat[2]
}

/// Inverts a non-singular similarity transform.
#[verified_engine::verified]
pub(crate) fn invert_affine(mat: Affine2D) -> Affine2D {
    let [a, b, c, d, tx, ty] = mat;
    let inv_det = 1.0 / (a * d - b * c);
    let ia = d * inv_det;
    let ib = -b * inv_det;
    let ic = -c * inv_det;
    let id = a * inv_det;
    [ia, ib, ic, id, -(ia * tx + ib * ty), -(ic * tx + id * ty)]
}

/// Direct similarity transform mapping `p` to `(0, 0)` and `q` onto `p->q` frame.
#[verified_engine::verified]
fn match_segment(p: Point2D, q: Point2D) -> Affine2D {
    [
        q[0] - p[0],
        p[1] - q[1],
        q[1] - p[1],
        q[0] - p[0],
        p[0],
        p[1],
    ]
}

/// Direct similarity transform mapping segment `p1 -> q1` onto `p2 -> q2`.
#[verified_engine::verified]
pub(crate) fn match_two_segments(p1: Point2D, q1: Point2D, p2: Point2D, q2: Point2D) -> Affine2D {
    mul_affine(match_segment(p2, q2), invert_affine(match_segment(p1, q1)))
}

/// Arithmetic mean of vertices in `poly`.
#[verified_engine::verified]
pub(crate) fn polygon_centroid(poly: &[Point2D]) -> Point2D {
    if poly.is_empty() {
        return [0.0, 0.0];
    }
    let inv_n = 1.0 / (poly.len() as f64);
    let mut sx = 0.0;
    let mut sy = 0.0;
    for &p in poly {
        sx += p[0];
        sy += p[1];
    }
    [sx * inv_n, sy * inv_n]
}

/// Hexagonal lattice point `a * (1, 0) + b * (1/2, √3/2)`.
#[verified_engine::verified]
pub(crate) fn hex_point(a: f64, b: f64) -> Point2D {
    let hr3 = 3.0_f64.sqrt() * 0.5;
    [a + 0.5 * b, hr3 * b]
}

/// 13-vertex polykite Hat outline used for metatile alignment, plus 14-vertex split outline
/// where the length-2 edge is subdivided at `(1, -2)` so all unit kite edges match adjacent tiles.
#[verified_engine::verified]
pub(crate) fn hat_base_outlines() -> ([Point2D; 13], [Point2D; 14]) {
    let hat13 = [
        hex_point(0.0, 0.0),
        hex_point(-1.0, -1.0),
        hex_point(0.0, -2.0),
        hex_point(2.0, -2.0),
        hex_point(2.0, -1.0),
        hex_point(4.0, -2.0),
        hex_point(5.0, -1.0),
        hex_point(4.0, 0.0),
        hex_point(3.0, 0.0),
        hex_point(2.0, 2.0),
        hex_point(0.0, 3.0),
        hex_point(0.0, 2.0),
        hex_point(-1.0, 2.0),
    ];
    let hat14 = [
        hex_point(0.0, 0.0),
        hex_point(-1.0, -1.0),
        hex_point(0.0, -2.0),
        hex_point(1.0, -2.0),
        hex_point(2.0, -2.0),
        hex_point(2.0, -1.0),
        hex_point(4.0, -2.0),
        hex_point(5.0, -1.0),
        hex_point(4.0, 0.0),
        hex_point(3.0, 0.0),
        hex_point(2.0, 2.0),
        hex_point(0.0, 3.0),
        hex_point(0.0, 2.0),
        hex_point(-1.0, 2.0),
    ];
    (hat13, hat14)
}

/// 14-vertex chiral `Spectre` (`Tile(1, 1)`) monotile outline.
#[verified_engine::verified]
pub(crate) fn spectre_base_outline() -> [Point2D; 14] {
    const DIRS: [usize; 13] = [0, 10, 1, 3, 0, 2, 5, 7, 4, 6, 6, 8, 11];
    let mut pts = [[0.0, 0.0]; 14];
    for (idx, &dir) in DIRS.iter().enumerate() {
        let ang = (dir as f64) * (PI / 6.0);
        pts[idx + 1] = [pts[idx][0] + ang.cos(), pts[idx][1] + ang.sin()];
    }
    pts
}

// [cite:graph_parameters_rust]
