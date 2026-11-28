//! General-position 3D-to-2D link projection with deterministic $SO(3)$ micro-tilts.

// [cite:knot_braid_topology]

use crate::pure_math::knot_theory::KnotError;
use crate::pure_math::knot_theory::curves::ClosedLink3D;
use crate::pure_math::knot_theory::diagram::{
    Crossing, CrossingSign, DiagramSegment, Endpoint, MAX_DIAGRAM_CROSSINGS, PlanarDiagram,
};
use nalgebra::{Matrix3, Point3};

const VERTEX_EPS: f64 = 1e-7;
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
const MAX_TILT_ATTEMPTS: usize = 24;

#[derive(Clone, Copy)]
struct HitRecord {
    pos_xy: [f64; 2],
    sign: CrossingSign,
    over_comp: usize,
    over_seg: usize,
    over_t: f64,
    under_comp: usize,
    under_seg: usize,
    under_t: f64,
}

#[derive(Clone, Copy)]
struct SegmentRef<'a> {
    p0: &'a Point3<f64>,
    p1: &'a Point3<f64>,
    comp: usize,
    seg: usize,
}

#[derive(Clone, Copy)]
enum SegmentTestOutcome {
    Disjoint,
    Degenerate,
    Hit(HitRecord),
}

#[verified_engine::verified]
pub(super) fn project_3d_link_to_diagram(link: &ClosedLink3D) -> Result<PlanarDiagram, KnotError> {
    let bound = link.bounding_radius().max(1e-6);
    for attempt in 0..MAX_TILT_ATTEMPTS {
        let rotated = rotated_components(&link.components, attempt);
        if let Some(hits) = try_collect_general_position_hits(&rotated, bound, true)
            && let Ok(diagram) = assemble_projected_diagram(&rotated, &hits, bound)
        {
            return Ok(diagram);
        }
    }
    let rotated = rotated_components(&link.components, MAX_TILT_ATTEMPTS);
    let Some(mut hits) = try_collect_general_position_hits(&rotated, bound, false) else {
        return Err(KnotError::InvalidDiagram {
            reason: "3D link contains degenerate self-intersecting or overlapping segments",
        });
    };
    deduplicate_endpoint_hits(&mut hits, &rotated);
    assemble_projected_diagram(&rotated, &hits, bound)
}

#[verified_engine::verified]
fn rotated_components(components: &[Vec<Point3<f64>>], attempt: usize) -> Vec<Vec<Point3<f64>>> {
    if attempt == 0 {
        return components.to_vec();
    }
    let tilt = 3.5e-4 * (attempt as f64);
    let phi = (attempt as f64) * GOLDEN_ANGLE;
    let (ax, ay) = (tilt * phi.cos(), tilt * phi.sin());
    let (sx, cx) = ax.sin_cos();
    let (sy, cy) = ay.sin_cos();
    let rot_x = Matrix3::new(1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx);
    let rot_y = Matrix3::new(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy);
    let r = rot_x * rot_y;
    components
        .iter()
        .map(|comp| comp.iter().map(|p| r * p).collect())
        .collect()
}

#[verified_engine::verified]
fn try_collect_general_position_hits(
    components: &[Vec<Point3<f64>>],
    bound: f64,
    strict_vertices: bool,
) -> Option<Vec<HitRecord>> {
    let mut hits = Vec::new();
    for (c1, pts1) in components.iter().enumerate() {
        let m1 = pts1.len();
        for i1 in 0..m1 {
            let s1 = SegmentRef {
                p0: &pts1[i1],
                p1: &pts1[(i1 + 1) % m1],
                comp: c1,
                seg: i1,
            };
            for (c2, pts2) in components.iter().enumerate().skip(c1) {
                let m2 = pts2.len();
                let j_start = if c1 == c2 { i1 + 2 } else { 0 };
                for i2 in j_start..m2 {
                    if c1 == c2 && i1 == 0 && i2 + 1 == m2 {
                        continue;
                    }
                    let s2 = SegmentRef {
                        p0: &pts2[i2],
                        p1: &pts2[(i2 + 1) % m2],
                        comp: c2,
                        seg: i2,
                    };
                    match intersect_segments_xy(s1, s2, bound, strict_vertices) {
                        SegmentTestOutcome::Disjoint => {}
                        SegmentTestOutcome::Degenerate => return None,
                        SegmentTestOutcome::Hit(h) => hits.push(h),
                    }
                }
            }
        }
    }
    if strict_vertices && has_coincident_hits(&hits) {
        return None;
    }
    Some(hits)
}

#[verified_engine::verified]
fn intersect_segments_xy(
    a: SegmentRef<'_>,
    b: SegmentRef<'_>,
    bound: f64,
    strict_vertices: bool,
) -> SegmentTestOutcome {
    let margin = 1e-7 * bound;
    if a.p0.x.max(a.p1.x) + margin < b.p0.x.min(b.p1.x)
        || b.p0.x.max(b.p1.x) + margin < a.p0.x.min(a.p1.x)
        || a.p0.y.max(a.p1.y) + margin < b.p0.y.min(b.p1.y)
        || b.p0.y.max(b.p1.y) + margin < a.p0.y.min(a.p1.y)
    {
        return SegmentTestOutcome::Disjoint;
    }
    let (dax, day) = (a.p1.x - a.p0.x, a.p1.y - a.p0.y);
    let (dbx, dby) = (b.p1.x - b.p0.x, b.p1.y - b.p0.y);
    let len_scale = dax.hypot(day) * dbx.hypot(dby);
    if len_scale <= 1e-24 * bound * bound {
        return SegmentTestOutcome::Disjoint;
    }
    let denom = dax * dby - day * dbx;
    if denom.abs() <= 1e-10 * len_scale {
        return if segments_close_xy(a.p0, a.p1, b.p0, b.p1, margin) {
            SegmentTestOutcome::Degenerate
        } else {
            SegmentTestOutcome::Disjoint
        };
    }
    let (rx, ry) = (b.p0.x - a.p0.x, b.p0.y - a.p0.y);
    let s = (rx * dby - ry * dbx) / denom;
    let t = (rx * day - ry * dax) / denom;
    if s < -VERTEX_EPS || s > 1.0 + VERTEX_EPS || t < -VERTEX_EPS || t > 1.0 + VERTEX_EPS {
        return SegmentTestOutcome::Disjoint;
    }
    if strict_vertices
        && (s <= VERTEX_EPS || 1.0 - s <= VERTEX_EPS || t <= VERTEX_EPS || 1.0 - t <= VERTEX_EPS)
    {
        return SegmentTestOutcome::Degenerate;
    }
    if !strict_vertices && (!(0.0..=1.0).contains(&s) || !(0.0..=1.0).contains(&t)) {
        return SegmentTestOutcome::Disjoint;
    }
    build_hit_record(a, b, (s.clamp(0.0, 1.0), t.clamp(0.0, 1.0)), denom, bound)
}

#[verified_engine::verified]
fn build_hit_record(
    a: SegmentRef<'_>,
    b: SegmentRef<'_>,
    st: (f64, f64),
    denom: f64,
    bound: f64,
) -> SegmentTestOutcome {
    let (s, t) = st;
    let za = a.p0.z + s * (a.p1.z - a.p0.z);
    let zb = b.p0.z + t * (b.p1.z - b.p0.z);
    if (za - zb).abs() <= 1e-9 * bound {
        return SegmentTestOutcome::Degenerate;
    }
    let px = (a.p0.x + s * (a.p1.x - a.p0.x)) / bound * 0.85;
    let py = (a.p0.y + s * (a.p1.y - a.p0.y)) / bound * 0.85;
    if za > zb {
        SegmentTestOutcome::Hit(HitRecord {
            pos_xy: [px, py],
            sign: if denom > 0.0 {
                CrossingSign::Positive
            } else {
                CrossingSign::Negative
            },
            over_comp: a.comp,
            over_seg: a.seg,
            over_t: s,
            under_comp: b.comp,
            under_seg: b.seg,
            under_t: t,
        })
    } else {
        SegmentTestOutcome::Hit(HitRecord {
            pos_xy: [px, py],
            sign: if denom < 0.0 {
                CrossingSign::Positive
            } else {
                CrossingSign::Negative
            },
            over_comp: b.comp,
            over_seg: b.seg,
            over_t: t,
            under_comp: a.comp,
            under_seg: a.seg,
            under_t: s,
        })
    }
}

#[verified_engine::verified]
fn segments_close_xy(
    a0: &Point3<f64>,
    a1: &Point3<f64>,
    b0: &Point3<f64>,
    b1: &Point3<f64>,
    tol: f64,
) -> bool {
    let tol2 = tol * tol;
    pt_seg_dist2_xy(a0, b0, b1) <= tol2
        || pt_seg_dist2_xy(a1, b0, b1) <= tol2
        || pt_seg_dist2_xy(b0, a0, a1) <= tol2
        || pt_seg_dist2_xy(b1, a0, a1) <= tol2
}

#[verified_engine::verified]
fn pt_seg_dist2_xy(p: &Point3<f64>, s0: &Point3<f64>, s1: &Point3<f64>) -> f64 {
    let (dx, dy) = (s1.x - s0.x, s1.y - s0.y);
    let l2 = dx * dx + dy * dy;
    let u = if l2 <= 1e-30 {
        0.0
    } else {
        (((p.x - s0.x) * dx + (p.y - s0.y) * dy) / l2).clamp(0.0, 1.0)
    };
    let (qx, qy) = (s0.x + u * dx - p.x, s0.y + u * dy - p.y);
    qx * qx + qy * qy
}

#[verified_engine::verified]
fn has_coincident_hits(hits: &[HitRecord]) -> bool {
    for i in 0..hits.len() {
        for j in (i + 1)..hits.len() {
            let dx = hits[i].pos_xy[0] - hits[j].pos_xy[0];
            let dy = hits[i].pos_xy[1] - hits[j].pos_xy[1];
            if dx * dx + dy * dy <= 1e-12 {
                return true;
            }
        }
    }
    false
}

#[verified_engine::verified]
fn deduplicate_endpoint_hits(hits: &mut Vec<HitRecord>, components: &[Vec<Point3<f64>>]) {
    let mut keep = vec![true; hits.len()];
    for i in 0..hits.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..hits.len() {
            if !keep[j] {
                continue;
            }
            let same_pair = hits[i].over_comp == hits[j].over_comp
                && hits[i].under_comp == hits[j].under_comp
                && cyclic_seg_adj(
                    hits[i].over_seg,
                    hits[j].over_seg,
                    components[hits[i].over_comp].len(),
                )
                && cyclic_seg_adj(
                    hits[i].under_seg,
                    hits[j].under_seg,
                    components[hits[i].under_comp].len(),
                );
            let dx = hits[i].pos_xy[0] - hits[j].pos_xy[0];
            let dy = hits[i].pos_xy[1] - hits[j].pos_xy[1];
            if same_pair && dx * dx + dy * dy <= 1e-9 {
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    hits.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

#[verified_engine::verified]
fn cyclic_seg_adj(a: usize, b: usize, m: usize) -> bool {
    a == b || (a + 1) % m == b || (b + 1) % m == a
}

#[verified_engine::verified]
fn assemble_projected_diagram(
    components: &[Vec<Point3<f64>>],
    hits: &[HitRecord],
    bound: f64,
) -> Result<PlanarDiagram, KnotError> {
    let c_count = hits.len();
    if c_count > MAX_DIAGRAM_CROSSINGS {
        return Err(KnotError::InvalidDiagram {
            reason: "projected 3D link exceeds maximum crossing bound",
        });
    }
    let mut crossings: Vec<Crossing> = hits
        .iter()
        .map(|h| Crossing {
            sign: h.sign,
            slots: [0; 4],
            position: h.pos_xy,
        })
        .collect();
    let mut per_comp: Vec<Vec<(usize, f64, usize, bool)>> = vec![Vec::new(); components.len()];
    for (cid, h) in hits.iter().enumerate() {
        per_comp[h.over_comp].push((h.over_seg, h.over_t, cid, true));
        per_comp[h.under_comp].push((h.under_seg, h.under_t, cid, false));
    }

    let mut segments = Vec::with_capacity(2 * c_count);
    let mut free_loops = 0;
    for (comp_idx, events) in per_comp.iter_mut().enumerate() {
        if events.is_empty() {
            free_loops += 1;
            continue;
        }
        events.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)));
        append_component_segments(
            &components[comp_idx],
            events,
            bound,
            &mut crossings,
            &mut segments,
        );
    }
    let diagram = PlanarDiagram {
        crossings,
        segments,
        free_loops,
    };
    diagram.validate()?;
    Ok(diagram)
}

#[verified_engine::verified]
fn append_component_segments(
    pts: &[Point3<f64>],
    events: &[(usize, f64, usize, bool)],
    bound: f64,
    crossings: &mut [Crossing],
    segments: &mut Vec<DiagramSegment>,
) {
    let m_ev = events.len();
    let n_pts = pts.len();
    for idx in 0..m_ev {
        let (seg_start, _, c_tail, is_over_tail) = events[idx];
        let (seg_end, _, c_head, is_over_head) = events[(idx + 1) % m_ev];
        let tail_slot = if is_over_tail {
            crossings[c_tail].sign.over_out_slot()
        } else {
            2
        };
        let head_slot = if is_over_head {
            crossings[c_head].sign.over_in_slot()
        } else {
            0
        };
        let waypoints = extract_subsampled_waypoints(pts, n_pts, seg_start, seg_end, bound);
        let seg_id = segments.len();
        segments.push(DiagramSegment {
            tail: Endpoint {
                crossing: c_tail,
                slot: tail_slot,
            },
            head: Endpoint {
                crossing: c_head,
                slot: head_slot,
            },
            waypoints,
        });
        crossings[c_tail].slots[tail_slot as usize] = seg_id;
        crossings[c_head].slots[head_slot as usize] = seg_id;
    }
}

#[verified_engine::verified]
fn extract_subsampled_waypoints(
    pts: &[Point3<f64>],
    n_pts: usize,
    seg_start: usize,
    seg_end: usize,
    bound: f64,
) -> Vec<[f64; 2]> {
    let steps = if seg_end >= seg_start {
        seg_end - seg_start
    } else {
        n_pts - seg_start + seg_end
    };
    if steps <= 1 {
        return Vec::new();
    }
    let count = (steps - 1).min(8);
    let mut out = Vec::with_capacity(count);
    for k in 1..=count {
        let offset = (k * steps) / (count + 1);
        let p = pts[(seg_start + offset) % n_pts];
        out.push([p.x / bound * 0.85, p.y / bound * 0.85]);
    }
    out
}
