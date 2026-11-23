//! Penrose `P2` (Kite & Dart) and `P3` (Thick & Thin Rhomb) aperiodic tilings via
//! Robinson golden-triangle deflation and symmetry-diagonal pairing.

use super::{TilePolygon, TileRole, quantize_vertex};
use math_commons::contours::Point2D;
use std::collections::HashMap;
use std::f64::consts::PI;

#[derive(Clone, Copy)]
struct RobinsonTri {
    kind: u8,
    a: Point2D,
    b: Point2D,
    c: Point2D,
}

#[verified_engine::verified]
fn lerp_pt(p: Point2D, q: Point2D, t: f64) -> Point2D {
    [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]
}

/// Generates a Penrose `P2` (Kite & Dart) patch from a 5-fold Kite Sun at `depth` (`0..=6`).
#[verified_engine::verified]
pub(crate) fn generate_penrose_p2(depth: usize) -> Vec<TilePolygon> {
    let psi = (5.0_f64.sqrt() - 1.0) * 0.5;
    let psi2 = 1.0 - psi;
    let radius = 10.0;
    let mut tris = Vec::with_capacity(10);

    for k in 0..10 {
        let a1 = (k as f64) * (PI / 5.0);
        let a2 = ((k + 1) as f64) * (PI / 5.0);
        let p1 = [radius * a1.cos(), radius * a1.sin()];
        let p2 = [radius * a2.cos(), radius * a2.sin()];
        let (b, c) = if k % 2 == 0 { (p1, p2) } else { (p2, p1) };
        tris.push(RobinsonTri {
            kind: 0,
            a: [0.0, 0.0],
            b,
            c,
        });
    }

    for _ in 0..depth {
        let mut next = Vec::with_capacity(tris.len() * 3);
        for tri in tris {
            if tri.kind == 0 {
                let q = lerp_pt(tri.a, tri.b, psi);
                let p = lerp_pt(tri.a, tri.c, psi2);
                next.push(RobinsonTri {
                    kind: 0,
                    a: tri.c,
                    b: tri.b,
                    c: q,
                });
                next.push(RobinsonTri {
                    kind: 0,
                    a: tri.c,
                    b: p,
                    c: q,
                });
                next.push(RobinsonTri {
                    kind: 1,
                    a: tri.a,
                    b: q,
                    c: p,
                });
            } else {
                let r = lerp_pt(tri.a, tri.b, psi);
                next.push(RobinsonTri {
                    kind: 0,
                    a: tri.a,
                    b: r,
                    c: tri.c,
                });
                next.push(RobinsonTri {
                    kind: 1,
                    a: tri.b,
                    b: tri.c,
                    c: r,
                });
            }
        }
        tris = next;
    }

    pair_symmetry_diagonals(&tris, true)
}

/// Generates a Penrose `P3` (Thick & Thin Rhomb) patch from a 5-fold Thick-Rhomb Star at `depth`.
#[verified_engine::verified]
pub(crate) fn generate_penrose_p3(depth: usize) -> Vec<TilePolygon> {
    let phi = (1.0 + 5.0_f64.sqrt()) * 0.5;
    let psi = phi - 1.0;
    let radius = 10.0;
    let mut tris = Vec::with_capacity(10);

    for m in 0..5 {
        let ang_c = ((2 * m + 1) as f64) * (PI / 5.0);
        let ang_b1 = ((2 * m) as f64) * (PI / 5.0);
        let ang_b2 = ((2 * m + 2) as f64) * (PI / 5.0);
        let c = [phi * radius * ang_c.cos(), phi * radius * ang_c.sin()];
        let b1 = [radius * ang_b1.cos(), radius * ang_b1.sin()];
        let b2 = [radius * ang_b2.cos(), radius * ang_b2.sin()];
        tris.push(RobinsonTri {
            kind: 1,
            a: b1,
            b: [0.0, 0.0],
            c,
        });
        tris.push(RobinsonTri {
            kind: 1,
            a: b2,
            b: [0.0, 0.0],
            c,
        });
    }

    for _ in 0..depth {
        let mut next = Vec::with_capacity(tris.len() * 3);
        for tri in tris {
            if tri.kind == 0 {
                let p = lerp_pt(tri.a, tri.b, psi);
                next.push(RobinsonTri {
                    kind: 0,
                    a: tri.c,
                    b: p,
                    c: tri.b,
                });
                next.push(RobinsonTri {
                    kind: 1,
                    a: p,
                    b: tri.c,
                    c: tri.a,
                });
            } else {
                let r = lerp_pt(tri.b, tri.c, psi);
                let q = lerp_pt(tri.b, tri.a, psi);
                next.push(RobinsonTri {
                    kind: 1,
                    a: q,
                    b: r,
                    c: tri.b,
                });
                next.push(RobinsonTri {
                    kind: 0,
                    a: r,
                    b: q,
                    c: tri.a,
                });
                next.push(RobinsonTri {
                    kind: 1,
                    a: r,
                    b: tri.c,
                    c: tri.a,
                });
            }
        }
        tris = next;
    }

    pair_symmetry_diagonals(&tris, false)
}

type SymmetryDiagKey = (u8, super::QuantizedVertex, super::QuantizedVertex);

#[verified_engine::verified]
fn pair_symmetry_diagonals(tris: &[RobinsonTri], is_p2: bool) -> Vec<TilePolygon> {
    let inv_tol = 1e5;
    let mut groups: HashMap<SymmetryDiagKey, Vec<RobinsonTri>> = HashMap::with_capacity(tris.len());
    let mut order_keys = Vec::new();

    for &tri in tris {
        let (d0, d1) = if is_p2 {
            (tri.a, tri.c)
        } else {
            (tri.b, tri.c)
        };
        let k0 = quantize_vertex(d0, inv_tol);
        let k1 = quantize_vertex(d1, inv_tol);
        let diag_key = if k0 <= k1 {
            (tri.kind, k0, k1)
        } else {
            (tri.kind, k1, k0)
        };
        let entry = groups.entry(diag_key).or_insert_with(|| {
            order_keys.push(diag_key);
            Vec::with_capacity(2)
        });
        entry.push(tri);
    }

    let mut tiles = Vec::with_capacity(order_keys.len());
    for key in order_keys {
        let Some(pair) = groups.get(&key) else {
            continue;
        };
        if pair.len() != 2 {
            continue;
        }
        let (t0, t1) = (pair[0], pair[1]);
        let (vertices, role) = if is_p2 {
            let role = if t0.kind == 0 {
                TileRole::Kite
            } else {
                TileRole::Dart
            };
            (vec![t0.a, t0.b, t0.c, t1.b], role)
        } else {
            let role = if t0.kind == 1 {
                TileRole::ThickRhomb
            } else {
                TileRole::ThinRhomb
            };
            (vec![t0.b, t0.a, t0.c, t1.a], role)
        };
        tiles.push(TilePolygon { vertices, role });
    }
    tiles
}

// [cite:graph_parameters_rust]
