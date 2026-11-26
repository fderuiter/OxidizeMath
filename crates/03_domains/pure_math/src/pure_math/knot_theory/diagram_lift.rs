//! Planar projection from [`ClosedLink3D`], 3D diagram lifting, and canonical presets.

// [cite:knot_braid_topology]

use super::KnotError;
use super::curves::ClosedLink3D;
use super::diagram::{Crossing, CrossingSign, DiagramSegment, Endpoint, PlanarDiagram};
use crate::pure_math::algebra::BraidWord;
use nalgebra::Point3;
use std::f64::consts::TAU;

impl PlanarDiagram {
    /// Canonical 3-crossing right-handed (`positive == true`) or left-handed trefoil diagram.
    #[must_use]
    #[verified_engine::verified]
    pub fn trefoil(positive: bool) -> Self {
        let g = if positive { 1 } else { -1 };
        let braid = BraidWord::new(2, vec![g, g, g]).expect("valid 2-strand trefoil braid");
        Self::from_braid(&braid)
    }

    /// Canonical 4-crossing alternating figure-eight ($4_1$) diagram.
    #[must_use]
    #[verified_engine::verified]
    pub fn figure_eight() -> Self {
        let braid = BraidWord::new(3, vec![1, -2, 1, -2]).expect("valid 3-strand figure-8 braid");
        Self::from_braid(&braid)
    }

    /// Canonical 2-crossing Hopf link diagram (`positive` controls linking number $\pm 1$).
    #[must_use]
    #[verified_engine::verified]
    pub fn hopf_link(positive: bool) -> Self {
        let g = if positive { 1 } else { -1 };
        let braid = BraidWord::new(2, vec![g, g]).expect("valid 2-strand Hopf braid");
        Self::from_braid(&braid)
    }

    /// Canonical 6-crossing alternating Borromean rings ($6^3_2$) diagram.
    #[must_use]
    #[verified_engine::verified]
    pub fn borromean_rings() -> Self {
        let braid =
            BraidWord::new(3, vec![1, -2, 1, -2, 1, -2]).expect("valid 3-strand Borromean braid");
        Self::from_braid(&braid)
    }

    /// Projects a generic 3D [`ClosedLink3D`] onto the $(x, y)$ plane to extract an oriented [`PlanarDiagram`].
    ///
    /// # Errors
    /// Returns [`KnotError`] if the projection exceeds [`super::diagram::MAX_DIAGRAM_CROSSINGS`]
    /// or exhibits degenerate coincident crossings.
    #[verified_engine::verified]
    pub fn from_3d_link(link: &ClosedLink3D) -> Result<Self, KnotError> {
        let bound = link.bounding_radius().max(1e-6);
        let mut raw_hits: Vec<HitRecord> = Vec::new();
        let n_comp = link.components.len();

        for c1 in 0..n_comp {
            let pts1 = &link.components[c1];
            let m1 = pts1.len();
            for i1 in 0..m1 {
                let a0 = pts1[i1];
                let a1 = pts1[(i1 + 1) % m1];
                for c2 in c1..n_comp {
                    let pts2 = &link.components[c2];
                    let m2 = pts2.len();
                    let j_start = if c1 == c2 { i1 + 2 } else { 0 };
                    for i2 in j_start..m2 {
                        if c1 == c2 && i1 == 0 && i2 == m2 - 1 {
                            continue;
                        }
                        let b0 = pts2[i2];
                        let b1 = pts2[(i2 + 1) % m2];
                        if let Some(hit) =
                            intersect_segments_xy(&a0, &a1, &b0, &b1, c1, i1, c2, i2, bound)
                        {
                            raw_hits.push(hit);
                        }
                    }
                }
            }
        }
        assemble_projected_diagram(link, &raw_hits, bound)
    }

    /// Lifts `self` into a smooth 3D [`ClosedLink3D`] with over-strands arched to $+z$ and
    /// under-strands dipped to $-z$, enabling live 3D writhe and mesh export after Reidemeister moves.
    ///
    /// # Errors
    /// Returns [`KnotError`] if `samples_per_segment < 4` or `self.validate()` fails.
    #[verified_engine::verified]
    pub fn lift_to_3d_link(&self, samples_per_segment: usize) -> Result<ClosedLink3D, KnotError> {
        self.validate()?;
        if samples_per_segment < 4 {
            return Err(KnotError::InsufficientSamples {
                required: 4,
                actual: samples_per_segment,
            });
        }
        let m = self.segments.len();
        let mut visited = vec![false; m];
        let mut components: Vec<Vec<Point3<f64>>> = Vec::new();

        for start in 0..m {
            if visited[start] {
                continue;
            }
            let mut cycle = Vec::new();
            let mut cur = start;
            let mut guard = 0;
            while !visited[cur] && guard <= m {
                visited[cur] = true;
                cycle.push(cur);
                let head = self.segments[cur].head;
                let c = &self.crossings[head.crossing];
                cur = if head.slot == 0 {
                    c.under_out()
                } else {
                    c.over_out()
                };
                guard += 1;
            }
            components.push(sample_lifted_cycle(self, &cycle, samples_per_segment));
        }

        for k in 0..self.free_loops {
            let n_pts = samples_per_segment.max(32);
            let r = 1.25 + 0.35 * (k as f64);
            let mut loop_pts = Vec::with_capacity(n_pts);
            for i in 0..n_pts {
                let th = TAU * (i as f64) / (n_pts as f64);
                loop_pts.push(Point3::new(r * th.cos(), r * th.sin(), 0.0));
            }
            components.push(loop_pts);
        }
        ClosedLink3D::new(components)
    }
}

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

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn intersect_segments_xy(
    a0: &Point3<f64>,
    a1: &Point3<f64>,
    b0: &Point3<f64>,
    b1: &Point3<f64>,
    c1: usize,
    i1: usize,
    c2: usize,
    i2: usize,
    bound: f64,
) -> Option<HitRecord> {
    let dax = a1.x - a0.x;
    let day = a1.y - a0.y;
    let dbx = b1.x - b0.x;
    let dby = b1.y - b0.y;
    let denom = dax * dby - day * dbx;
    if denom.abs() <= 1e-12 {
        return None;
    }
    let rx = b0.x - a0.x;
    let ry = b0.y - a0.y;
    let s = (rx * dby - ry * dbx) / denom;
    let t = (rx * day - ry * dax) / denom;
    if !(0.0..1.0).contains(&s) || !(0.0..1.0).contains(&t) {
        return None;
    }
    let za = a0.z + s * (a1.z - a0.z);
    let zb = b0.z + t * (b1.z - b0.z);
    if (za - zb).abs() <= 1e-9 {
        return None;
    }
    let px = (a0.x + s * dax) / bound;
    let py = (a0.y + s * day) / bound;
    if za > zb {
        let sign = if denom > 0.0 {
            CrossingSign::Positive
        } else {
            CrossingSign::Negative
        };
        Some(HitRecord {
            pos_xy: [px * 0.85, py * 0.85],
            sign,
            over_comp: c1,
            over_seg: i1,
            over_t: s,
            under_comp: c2,
            under_seg: i2,
            under_t: t,
        })
    } else {
        let det_ba = -denom;
        let sign = if det_ba > 0.0 {
            CrossingSign::Positive
        } else {
            CrossingSign::Negative
        };
        Some(HitRecord {
            pos_xy: [px * 0.85, py * 0.85],
            sign,
            over_comp: c2,
            over_seg: i2,
            over_t: t,
            under_comp: c1,
            under_seg: i1,
            under_t: s,
        })
    }
}

#[verified_engine::verified]
fn assemble_projected_diagram(
    link: &ClosedLink3D,
    hits: &[HitRecord],
    bound: f64,
) -> Result<PlanarDiagram, KnotError> {
    let c_count = hits.len();
    if c_count > super::diagram::MAX_DIAGRAM_CROSSINGS {
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
    let mut per_comp: Vec<Vec<(usize, f64, usize, bool)>> = vec![Vec::new(); link.components.len()];
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
        let m_ev = events.len();
        let pts = &link.components[comp_idx];
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
    let diagram = PlanarDiagram {
        crossings,
        segments,
        free_loops,
    };
    diagram.validate()?;
    Ok(diagram)
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
    let count = (steps - 1).min(4);
    let mut out = Vec::with_capacity(count);
    for k in 1..=count {
        let offset = (k * steps) / (count + 1);
        let p = pts[(seg_start + offset) % n_pts];
        out.push([p.x / bound * 0.85, p.y / bound * 0.85]);
    }
    out
}

#[verified_engine::verified]
fn sample_lifted_cycle(
    diagram: &PlanarDiagram,
    cycle: &[usize],
    samples_per_segment: usize,
) -> Vec<Point3<f64>> {
    let scale_xy = 1.85;
    let z_bridge = 0.28;
    let mut control_pts: Vec<Point3<f64>> = Vec::with_capacity(cycle.len() * 3);
    let m_total = diagram.segments.len().max(1) as f64;

    for &seg_id in cycle {
        let seg = &diagram.segments[seg_id];
        let p_tail = diagram.crossings[seg.tail.crossing].position;
        let p_head = diagram.crossings[seg.head.crossing].position;
        let z_tail = if seg.tail.slot == 2 {
            -z_bridge
        } else {
            z_bridge
        };
        let z_head = if seg.head.slot == 0 {
            -z_bridge
        } else {
            z_bridge
        };
        control_pts.push(Point3::new(
            p_tail[0] * scale_xy,
            p_tail[1] * scale_xy,
            z_tail,
        ));
        let z_bias = 0.035 * (((seg_id as f64) / m_total) - 0.5);
        if seg.waypoints.is_empty() {
            control_pts.push(Point3::new(
                0.5 * (p_tail[0] + p_head[0]) * scale_xy,
                0.5 * (p_tail[1] + p_head[1]) * scale_xy,
                0.5 * (z_tail + z_head) + z_bias,
            ));
        } else {
            let nw = (seg.waypoints.len() + 1) as f64;
            for (wi, wp) in seg.waypoints.iter().enumerate() {
                let alpha = ((wi + 1) as f64) / nw;
                let z_w = (1.0 - alpha) * z_tail + alpha * z_head + z_bias;
                control_pts.push(Point3::new(wp[0] * scale_xy, wp[1] * scale_xy, z_w));
            }
        }
    }

    // Evaluate periodic uniform cubic B-spline through control_pts for C^2 smoothness
    let k = control_pts.len();
    let sub = (samples_per_segment / 2).max(4);
    let mut out = Vec::with_capacity(k * sub);
    for i in 0..k {
        let p0 = control_pts[(i + k - 1) % k].coords;
        let p1 = control_pts[i].coords;
        let p2 = control_pts[(i + 1) % k].coords;
        let p3 = control_pts[(i + 2) % k].coords;
        for s in 0..sub {
            let t = (s as f64) / (sub as f64);
            let t2 = t * t;
            let t3 = t2 * t;
            let b0 = (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0;
            let b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
            let b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
            let b3 = t3 / 6.0;
            out.push(Point3::from(p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3));
        }
    }
    out
}
