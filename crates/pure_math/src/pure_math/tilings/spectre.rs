//! Chiral `Spectre` (`Tile(1, 1)`) aperiodic monotile substitution system
//! (`Gamma, Delta, Theta, Lambda, Xi, Pi, Sigma, Phi, Psi`).

use super::affine::{
    Affine2D, apply_point, identity, mul_affine, polygon_centroid, reflect_x, rotate_deg,
    spectre_base_outline, transform_polygon, translate,
};
use super::{TilePolygon, TileRole, quantize_vertex};
use math_commons::contours::Point2D;
use std::collections::{HashMap, HashSet};

const NUM_LABELS: usize = 9;
const GAMMA: usize = 0;
const DELTA: usize = 1;
const THETA: usize = 2;
const LAMBDA: usize = 3;
const XI: usize = 4;
const PI_L: usize = 5;
const SIGMA: usize = 6;
const PHI: usize = 7;
const PSI: usize = 8;

const T_RULES: [(f64, usize, usize); 7] = [
    (60.0, 3, 1),
    (0.0, 2, 0),
    (60.0, 3, 1),
    (60.0, 3, 1),
    (0.0, 2, 0),
    (60.0, 3, 1),
    (-120.0, 3, 3),
];

const SUBST: [[Option<usize>; 8]; NUM_LABELS] = [
    [
        Some(PI_L),
        Some(DELTA),
        None,
        Some(THETA),
        Some(SIGMA),
        Some(XI),
        Some(PHI),
        Some(GAMMA),
    ],
    [
        Some(XI),
        Some(DELTA),
        Some(XI),
        Some(PHI),
        Some(SIGMA),
        Some(PI_L),
        Some(PHI),
        Some(GAMMA),
    ],
    [
        Some(PSI),
        Some(DELTA),
        Some(PI_L),
        Some(PHI),
        Some(SIGMA),
        Some(PI_L),
        Some(PHI),
        Some(GAMMA),
    ],
    [
        Some(PSI),
        Some(DELTA),
        Some(XI),
        Some(PHI),
        Some(SIGMA),
        Some(PI_L),
        Some(PHI),
        Some(GAMMA),
    ],
    [
        Some(PSI),
        Some(DELTA),
        Some(PI_L),
        Some(PHI),
        Some(SIGMA),
        Some(PSI),
        Some(PHI),
        Some(GAMMA),
    ],
    [
        Some(PSI),
        Some(DELTA),
        Some(XI),
        Some(PHI),
        Some(SIGMA),
        Some(PSI),
        Some(PHI),
        Some(GAMMA),
    ],
    [
        Some(XI),
        Some(DELTA),
        Some(XI),
        Some(PHI),
        Some(SIGMA),
        Some(PI_L),
        Some(LAMBDA),
        Some(GAMMA),
    ],
    [
        Some(PSI),
        Some(DELTA),
        Some(PSI),
        Some(PHI),
        Some(SIGMA),
        Some(PI_L),
        Some(PHI),
        Some(GAMMA),
    ],
    [
        Some(PSI),
        Some(DELTA),
        Some(PSI),
        Some(PHI),
        Some(SIGMA),
        Some(PSI),
        Some(PHI),
        Some(GAMMA),
    ],
];

struct SupertileSystem {
    quad: [Point2D; 4],
    tiles: [Vec<Affine2D>; NUM_LABELS],
}

/// Generates a chiral `Spectre` (`Tile(1, 1)`) monotile patch at `depth` (`0..=6`).
///
/// For `depth >= 4`, supertile recursion is capped at level 3 (`559` tiles) or level 4 (`4,471`
/// tiles) with radial connected-core windowing so every depth `0..=6` remains fast, strictly
/// monotonic in patch size, crack-free, and memory-bounded on both desktop and `wasm32`.
#[verified_engine::verified]
pub(crate) fn generate_einstein_spectre(depth: usize) -> Vec<TilePolygon> {
    let base = spectre_base_outline();
    let mystic = mul_affine(translate(base[8][0], base[8][1]), rotate_deg(30.0));
    let mut sys = SupertileSystem {
        quad: [base[3], base[5], base[7], base[11]],
        tiles: [
            vec![identity(), mystic],
            vec![identity()],
            vec![identity()],
            vec![identity()],
            vec![identity()],
            vec![identity()],
            vec![identity()],
            vec![identity()],
            vec![identity()],
        ],
    };

    let subst_steps = depth.min(4);
    for _ in 0..subst_steps {
        sys = advance_supertile(&sys);
    }

    let raw_transforms = if depth == 0 {
        &sys.tiles[GAMMA]
    } else {
        &sys.tiles[DELTA]
    };
    let unreflect = subst_steps % 2 == 1;
    let refl = reflect_x();

    let all_polys: Vec<Vec<Point2D>> = raw_transforms
        .iter()
        .map(|&mat| {
            let eff = if unreflect {
                mul_affine(refl, mat)
            } else {
                mat
            };
            transform_polygon(eff, &base)
        })
        .collect();

    let max_tiles = match depth {
        0 => 2,
        1 => 9,
        2 => 71,
        3 => 180,
        4 => 320,
        5 => 500,
        _ => 720,
    };

    let selected = if all_polys.len() <= max_tiles {
        all_polys
    } else {
        window_connected_core(&all_polys, max_tiles)
    };

    selected
        .into_iter()
        .map(|vertices| TilePolygon {
            vertices,
            role: TileRole::Spectre,
        })
        .collect()
}

#[verified_engine::verified]
fn advance_supertile(sys: &SupertileSystem) -> SupertileSystem {
    let quad = sys.quad;
    let mut trans = [identity(); 8];
    let mut total_ang = 0.0;
    let mut cur_quad = quad;

    for (idx, &(ang, from_i, to_i)) in T_RULES.iter().enumerate() {
        total_ang += ang;
        let rot = rotate_deg(total_ang);
        let rot_to = apply_point(rot, quad[to_i]);
        let target = cur_quad[from_i];
        let next_t = mul_affine(translate(target[0] - rot_to[0], target[1] - rot_to[1]), rot);
        trans[idx + 1] = next_t;
        for slot in 0..4 {
            cur_quad[slot] = apply_point(next_t, quad[slot]);
        }
    }

    let refl = reflect_x();
    for slot in &mut trans {
        *slot = mul_affine(refl, *slot);
    }

    let super_quad = [
        apply_point(trans[6], quad[2]),
        apply_point(trans[5], quad[1]),
        apply_point(trans[3], quad[2]),
        apply_point(trans[0], quad[1]),
    ];

    let mut new_tiles: [Vec<Affine2D>; NUM_LABELS] = Default::default();
    for label in 0..NUM_LABELS {
        let mut list = Vec::new();
        for (slot_i, maybe_child) in SUBST[label].iter().enumerate() {
            if let Some(ch_label) = *maybe_child {
                for &child_t in &sys.tiles[ch_label] {
                    list.push(mul_affine(trans[slot_i], child_t));
                }
            }
        }
        new_tiles[label] = list;
    }

    SupertileSystem {
        quad: super_quad,
        tiles: new_tiles,
    }
}

#[verified_engine::verified]
fn window_connected_core(polys: &[Vec<Point2D>], target_count: usize) -> Vec<Vec<Point2D>> {
    let centroids: Vec<Point2D> = polys.iter().map(|p| polygon_centroid(p)).collect();
    let patch_center = polygon_centroid(&centroids);

    let mut seed_idx = 0usize;
    let mut best_d2 = f64::INFINITY;
    for (idx, &c) in centroids.iter().enumerate() {
        let d2 = (c[0] - patch_center[0]).powi(2) + (c[1] - patch_center[1]).powi(2);
        if d2 < best_d2 {
            best_d2 = d2;
            seed_idx = idx;
        }
    }

    let adj = build_tile_adjacency(polys);
    let mut visited: HashSet<usize> = HashSet::with_capacity(target_count);
    let mut selected_indices = Vec::with_capacity(target_count);
    let mut frontier = vec![seed_idx];
    visited.insert(seed_idx);

    for _ in 0..target_count {
        if frontier.is_empty() {
            break;
        }
        let mut best_pos = 0usize;
        let mut min_dist = f64::INFINITY;
        for (f_pos, &cand_i) in frontier.iter().enumerate() {
            let c = centroids[cand_i];
            let d2 = (c[0] - patch_center[0]).powi(2) + (c[1] - patch_center[1]).powi(2);
            if d2 < min_dist {
                min_dist = d2;
                best_pos = f_pos;
            }
        }
        let chosen = frontier.swap_remove(best_pos);
        selected_indices.push(chosen);
        for &nbr in &adj[chosen] {
            if visited.insert(nbr) {
                frontier.push(nbr);
            }
        }
    }

    selected_indices.sort_unstable();
    selected_indices
        .into_iter()
        .map(|i| polys[i].clone())
        .collect()
}

#[verified_engine::verified]
fn build_tile_adjacency(polys: &[Vec<Point2D>]) -> Vec<Vec<usize>> {
    let mut edge_owner: HashMap<super::QuantizedEdgeKey, usize> = HashMap::new();
    let mut adj = vec![Vec::new(); polys.len()];
    for (idx, poly) in polys.iter().enumerate() {
        let n = poly.len();
        for k in 0..n {
            let a = quantize_vertex(poly[k], 1e4);
            let b = quantize_vertex(poly[(k + 1) % n], 1e4);
            let ek = if a <= b { (a, b) } else { (b, a) };
            if let Some(&other) = edge_owner.get(&ek) {
                if other != idx {
                    adj[idx].push(other);
                    adj[other].push(idx);
                }
            } else {
                edge_owner.insert(ek, idx);
            }
        }
    }
    adj
}

// [cite:graph_parameters_rust]
