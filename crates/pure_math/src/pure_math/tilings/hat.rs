//! Smith–Myers–Kaplan–Goodman-Strauss Einstein `Hat` (`Tile(1, √3)`) polykite monotile
//! patches via `H–T–P–F` metatile assembly with simply-connected shared-edge verification.

use super::affine::{
    Affine2D, det_affine, hat_base_outlines, identity, match_two_segments, mul_affine,
    polygon_centroid, transform_polygon, translate,
};
use super::{TilePolygon, TileRole, quantize_vertex};
use math_commons::contours::Point2D;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq)]
enum MetaKind {
    H,
    T,
    P,
    F,
}

struct PlacedMetatile {
    kind: MetaKind,
    transform: Affine2D,
    outline: Vec<Point2D>,
}

type EdgeKey = ((i64, i64), (i64, i64));

const TRANSITIONS: &[(MetaKind, usize, MetaKind, usize)] = &[
    (MetaKind::H, 0, MetaKind::P, 2),
    (MetaKind::H, 2, MetaKind::P, 0),
    (MetaKind::H, 4, MetaKind::P, 2),
    (MetaKind::H, 4, MetaKind::F, 3),
    (MetaKind::H, 4, MetaKind::P, 0),
    (MetaKind::H, 0, MetaKind::P, 0),
    (MetaKind::H, 2, MetaKind::P, 2),
    (MetaKind::P, 0, MetaKind::H, 2),
    (MetaKind::P, 2, MetaKind::H, 0),
    (MetaKind::P, 0, MetaKind::H, 4),
    (MetaKind::P, 2, MetaKind::H, 4),
    (MetaKind::P, 0, MetaKind::H, 0),
    (MetaKind::P, 2, MetaKind::H, 2),
    (MetaKind::P, 0, MetaKind::F, 3),
    (MetaKind::P, 2, MetaKind::F, 3),
    (MetaKind::F, 2, MetaKind::F, 1),
    (MetaKind::F, 1, MetaKind::F, 2),
    (MetaKind::F, 0, MetaKind::T, 0),
    (MetaKind::F, 0, MetaKind::T, 2),
    (MetaKind::T, 0, MetaKind::F, 0),
    (MetaKind::T, 1, MetaKind::F, 0),
    (MetaKind::T, 2, MetaKind::F, 0),
    (MetaKind::F, 3, MetaKind::H, 4),
    (MetaKind::F, 3, MetaKind::H, 0),
    (MetaKind::F, 3, MetaKind::H, 2),
    (MetaKind::F, 3, MetaKind::P, 0),
    (MetaKind::F, 3, MetaKind::P, 2),
];

struct LocalMetatileHats {
    h: [Affine2D; 4],
    t: [Affine2D; 1],
    p: [Affine2D; 2],
    f: [Affine2D; 2],
}

impl LocalMetatileHats {
    #[verified_engine::verified]
    fn slice(&self, kind: MetaKind) -> &[Affine2D] {
        match kind {
            MetaKind::H => &self.h,
            MetaKind::T => &self.t,
            MetaKind::P => &self.p,
            MetaKind::F => &self.f,
        }
    }
}

/// Generates a simply-connected Einstein `Hat` (`Tile(1, √3)`) patch at `depth` (`0..=6`),
/// containing both unreflected `Hat` and reflected `AntiHat` polykites with zero interior gaps.
#[verified_engine::verified]
pub(crate) fn generate_einstein_hat(depth: usize) -> Vec<TilePolygon> {
    let (hat13, hat14) = hat_base_outlines();
    let h_out = metatile_outline(MetaKind::H);
    let local_hats = LocalMetatileHats {
        h: h_hat_transforms(&hat13, &h_out),
        t: t_hat_transforms(),
        p: p_hat_transforms(),
        f: f_hat_transforms(),
    };

    let mut placed = vec![PlacedMetatile {
        kind: MetaKind::H,
        transform: identity(),
        outline: h_out,
    }];
    let mut edge_counts: HashMap<EdgeKey, usize> = HashMap::new();
    for &th in &local_hats.h {
        insert_hat_edges(th, &hat14, &mut edge_counts);
    }

    for _ in 0..depth {
        let cur_len = placed.len();
        for idx in 0..cur_len {
            let parent_kind = placed[idx].kind;
            let parent_poly = placed[idx].outline.clone();
            for &(pk, pe, ck, ce) in TRANSITIONS {
                if pk == parent_kind {
                    try_attach_metatile(
                        ck,
                        ce,
                        &parent_poly,
                        pe,
                        &hat14,
                        &local_hats,
                        &mut placed,
                        &mut edge_counts,
                    );
                }
            }
        }
    }

    emit_hat_polygons(&placed, &local_hats, &hat14)
}

#[verified_engine::verified]
fn emit_hat_polygons(
    placed: &[PlacedMetatile],
    local_hats: &LocalMetatileHats,
    hat14: &[Point2D; 14],
) -> Vec<TilePolygon> {
    let scale2: Affine2D = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let mut tiles = Vec::with_capacity(placed.len() * 3);
    for meta in placed {
        for &th in local_hats.slice(meta.kind) {
            let full_t = mul_affine(scale2, mul_affine(meta.transform, th));
            let role = if det_affine(full_t) < 0.0 {
                TileRole::AntiHat
            } else {
                TileRole::Hat
            };
            tiles.push(TilePolygon {
                vertices: transform_polygon(full_t, hat14),
                role,
            });
        }
    }
    tiles
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn try_attach_metatile(
    ck: MetaKind,
    ce: usize,
    parent_poly: &[Point2D],
    pe: usize,
    hat14: &[Point2D; 14],
    local_hats: &LocalMetatileHats,
    placed: &mut Vec<PlacedMetatile>,
    edge_counts: &mut HashMap<EdgeKey, usize>,
) {
    let child_base = metatile_outline(ck);
    let tc = match_two_segments(
        child_base[(ce + 1) % child_base.len()],
        child_base[ce],
        parent_poly[pe],
        parent_poly[(pe + 1) % parent_poly.len()],
    );
    let c_poly = transform_polygon(tc, &child_base);
    let c_center = quantize_vertex(polygon_centroid(&c_poly), 100.0);
    if placed
        .iter()
        .any(|ex| quantize_vertex(polygon_centroid(&ex.outline), 100.0) == c_center)
    {
        return;
    }

    let mut trial = edge_counts.clone();
    let mut shared = 0usize;
    for &th in local_hats.slice(ck) {
        for ek in hat_quantized_edges(mul_affine(tc, th), hat14) {
            if edge_counts.contains_key(&ek) {
                shared += 1;
            }
            let cnt = trial.entry(ek).or_insert(0);
            *cnt += 1;
            if *cnt > 2 {
                return;
            }
        }
    }

    if shared >= 3 && has_single_simple_boundary(&trial) {
        placed.push(PlacedMetatile {
            kind: ck,
            transform: tc,
            outline: c_poly,
        });
        *edge_counts = trial;
    }
}

#[verified_engine::verified]
fn metatile_outline(kind: MetaKind) -> Vec<Point2D> {
    let hr3 = 3.0_f64.sqrt() * 0.5;
    match kind {
        MetaKind::H => vec![
            [0.0, 0.0],
            [4.0, 0.0],
            [4.5, hr3],
            [2.5, 5.0 * hr3],
            [1.5, 5.0 * hr3],
            [-0.5, hr3],
        ],
        MetaKind::T => vec![[0.0, 0.0], [3.0, 0.0], [1.5, 3.0 * hr3]],
        MetaKind::P => vec![[0.0, 0.0], [4.0, 0.0], [3.0, 2.0 * hr3], [-1.0, 2.0 * hr3]],
        MetaKind::F => vec![
            [0.0, 0.0],
            [3.0, 0.0],
            [3.5, hr3],
            [3.0, 2.0 * hr3],
            [-1.0, 2.0 * hr3],
        ],
    }
}

#[verified_engine::verified]
fn h_hat_transforms(hat13: &[Point2D; 13], h_out: &[Point2D]) -> [Affine2D; 4] {
    let hr3 = 3.0_f64.sqrt() * 0.5;
    [
        match_two_segments(hat13[5], hat13[7], h_out[5], h_out[0]),
        match_two_segments(hat13[9], hat13[11], h_out[1], h_out[2]),
        match_two_segments(hat13[5], hat13[7], h_out[3], h_out[4]),
        mul_affine(
            translate(2.5, hr3),
            [-0.25, 0.5 * hr3, 0.5 * hr3, 0.25, 0.0, 0.0],
        ),
    ]
}

#[verified_engine::verified]
fn t_hat_transforms() -> [Affine2D; 1] {
    let hr3 = 3.0_f64.sqrt() * 0.5;
    [[0.5, 0.0, 0.0, 0.5, 0.5, hr3]]
}

#[verified_engine::verified]
fn p_hat_transforms() -> [Affine2D; 2] {
    let hr3 = 3.0_f64.sqrt() * 0.5;
    [
        [0.5, 0.0, 0.0, 0.5, 1.5, hr3],
        [0.25, 0.5 * hr3, -0.5 * hr3, 0.25, 0.0, 2.0 * hr3],
    ]
}

#[verified_engine::verified]
fn f_hat_transforms() -> [Affine2D; 2] {
    let hr3 = 3.0_f64.sqrt() * 0.5;
    [
        [0.25, 0.5 * hr3, -0.5 * hr3, 0.25, 0.0, 2.0 * hr3],
        [0.25, -0.5 * hr3, 0.5 * hr3, 0.25, 1.5, -hr3],
    ]
}

#[verified_engine::verified]
fn hat_quantized_edges(mat: Affine2D, hat14: &[Point2D; 14]) -> [EdgeKey; 14] {
    let scale2: Affine2D = [2.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let poly = transform_polygon(mul_affine(scale2, mat), hat14);
    let mut keys = [((0, 0), (0, 0)); 14];
    for idx in 0..14 {
        let k0 = quantize_vertex(poly[idx], 1e4);
        let k1 = quantize_vertex(poly[(idx + 1) % 14], 1e4);
        keys[idx] = if k0 <= k1 { (k0, k1) } else { (k1, k0) };
    }
    keys
}

#[verified_engine::verified]
fn insert_hat_edges(mat: Affine2D, hat14: &[Point2D; 14], counts: &mut HashMap<EdgeKey, usize>) {
    for ek in hat_quantized_edges(mat, hat14) {
        *counts.entry(ek).or_insert(0) += 1;
    }
}

#[verified_engine::verified]
fn has_single_simple_boundary(edge_counts: &HashMap<EdgeKey, usize>) -> bool {
    let mut adj: HashMap<(i64, i64), Vec<(i64, i64)>> = HashMap::new();
    for (&(a, b), &cnt) in edge_counts {
        if cnt == 1 {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
    }
    let Some(&start) = adj.keys().next() else {
        return false;
    };
    if adj.values().any(|nbrs| nbrs.len() != 2) {
        return false;
    }

    let max_verts = adj.len();
    let mut visited: HashSet<(i64, i64)> = HashSet::with_capacity(max_verts);
    let mut stack = vec![start];
    for _ in 0..=max_verts {
        let Some(cur) = stack.pop() else {
            break;
        };
        if !visited.insert(cur) {
            continue;
        }
        if let Some(nbrs) = adj.get(&cur) {
            for &nxt in nbrs {
                if !visited.contains(&nxt) {
                    stack.push(nxt);
                }
            }
        }
    }
    visited.len() == max_verts
}

// [cite:graph_parameters_rust]
