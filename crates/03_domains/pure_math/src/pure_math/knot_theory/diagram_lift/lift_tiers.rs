//! Non-crossing 2D polyline overlap detection and disjoint $z$-tier assignment for 3D lifts.

// [cite:knot_braid_topology]

use super::lift_spline::CrossingFrame;
use crate::pure_math::knot_theory::diagram::PlanarDiagram;

#[verified_engine::verified]
pub(super) fn compute_segment_z_tiers(
    diagram: &PlanarDiagram,
    polylines: &[Vec<[f64; 2]>],
    frames: &[CrossingFrame],
) -> Vec<usize> {
    let m = polylines.len();
    debug_assert_eq!(m, diagram.segments.len());
    debug_assert_eq!(frames.len(), diagram.crossings.len());
    let mut adj = vec![Vec::new(); m];
    for i in 0..m {
        for j in (i + 1)..m {
            if polylines_intersect_2d(&polylines[i], &polylines[j]) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    let mut needs_lift = vec![false; m];
    for (s, poly) in polylines.iter().enumerate() {
        let seg = &diagram.segments[s];
        for (ci, f) in frames.iter().enumerate() {
            if ci != seg.tail.crossing
                && ci != seg.head.crossing
                && polyline_near_point(poly, f.pos, 1.15 * f.collar)
            {
                needs_lift[s] = true;
            }
        }
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by_key(|&s| {
        let seg = &diagram.segments[s];
        let max_c = seg.tail.crossing.max(seg.head.crossing);
        let over_cnt = usize::from(seg.tail.slot != 2) + usize::from(seg.head.slot != 0);
        (max_c, over_cnt, s)
    });
    let tiers = assign_greedy_tiers(diagram, &order, &adj, &needs_lift);
    debug_assert_eq!(tiers.len(), m);
    tiers
}

#[verified_engine::verified]
fn assign_greedy_tiers(
    diagram: &PlanarDiagram,
    order: &[usize],
    adj: &[Vec<usize>],
    needs_lift: &[bool],
) -> Vec<usize> {
    let m = order.len();
    debug_assert_eq!(adj.len(), m);
    debug_assert_eq!(needs_lift.len(), m);
    let mut tier = vec![usize::MAX; m];
    for &s in order {
        if adj[s].is_empty() && !needs_lift[s] {
            tier[s] = 0;
            continue;
        }
        let mut used = Vec::new();
        for &nbr in &adj[s] {
            if tier[nbr] != usize::MAX {
                used.push(tier[nbr]);
            }
        }
        if !needs_lift[s] && !used.contains(&0) {
            tier[s] = 0;
            continue;
        }
        let pref = preferred_strand_tier(diagram, s, &tier);
        if let Some(p) = pref
            && p >= 1
            && !used.contains(&p)
        {
            tier[s] = p;
            continue;
        }
        let mut cand = 1_usize;
        while used.contains(&cand) {
            cand += 1;
        }
        tier[s] = cand;
    }
    debug_assert!(!tier.contains(&usize::MAX));
    tier
}

#[verified_engine::verified]
fn preferred_strand_tier(diagram: &PlanarDiagram, s: usize, tier: &[usize]) -> Option<usize> {
    debug_assert!(s < diagram.segments.len());
    let seg = &diagram.segments[s];
    let tc = &diagram.crossings[seg.tail.crossing];
    let prev = if seg.tail.slot == 2 {
        tc.under_in()
    } else {
        tc.over_in()
    };
    if tier[prev] != usize::MAX && tier[prev] >= 1 {
        return Some(tier[prev]);
    }
    let hc = &diagram.crossings[seg.head.crossing];
    let next = if seg.head.slot == 0 {
        hc.under_out()
    } else {
        hc.over_out()
    };
    (tier[next] != usize::MAX && tier[next] >= 1).then_some(tier[next])
}

#[verified_engine::verified]
fn polylines_intersect_2d(a: &[[f64; 2]], b: &[[f64; 2]]) -> bool {
    debug_assert!(a.len() >= 2 && b.len() >= 2);
    for w1 in a.windows(2) {
        for w2 in b.windows(2) {
            if segments_cross_2d(w1[0], w1[1], w2[0], w2[1]) {
                return true;
            }
        }
    }
    false
}

#[verified_engine::verified]
fn segments_cross_2d(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> bool {
    debug_assert!(a0[0].is_finite() && b0[0].is_finite());
    let (dax, day) = (a1[0] - a0[0], a1[1] - a0[1]);
    let (dbx, dby) = (b1[0] - b0[0], b1[1] - b0[1]);
    let denom = dax * dby - day * dbx;
    if denom.abs() <= 1e-12 {
        return false;
    }
    let (rx, ry) = (b0[0] - a0[0], b0[1] - a0[1]);
    let s = (rx * dby - ry * dbx) / denom;
    let t = (rx * day - ry * dax) / denom;
    (-1e-4..=1.0 + 1e-4).contains(&s) && (-1e-4..=1.0 + 1e-4).contains(&t)
}

#[verified_engine::verified]
fn polyline_near_point(poly: &[[f64; 2]], pt: [f64; 2], tol: f64) -> bool {
    debug_assert!(tol >= 0.0);
    let tol2 = tol * tol;
    for w in poly.windows(2) {
        let (dx, dy) = (w[1][0] - w[0][0], w[1][1] - w[0][1]);
        let l2 = dx * dx + dy * dy;
        let u = if l2 <= 1e-24 {
            0.0
        } else {
            (((pt[0] - w[0][0]) * dx + (pt[1] - w[0][1]) * dy) / l2).clamp(0.0, 1.0)
        };
        let (qx, qy) = (w[0][0] + u * dx - pt[0], w[0][1] + u * dy - pt[1]);
        if qx * qx + qy * qy <= tol2 {
            return true;
        }
    }
    false
}
