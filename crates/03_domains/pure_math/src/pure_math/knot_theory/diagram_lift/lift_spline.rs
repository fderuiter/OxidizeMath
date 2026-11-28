//! Interpolating centripetal Catmull-Rom 3D lift with crossing bridge collars.

// [cite:knot_braid_topology]

use super::lift_tiers::compute_segment_z_tiers;
use crate::pure_math::knot_theory::KnotError;
use crate::pure_math::knot_theory::curves::ClosedLink3D;
use crate::pure_math::knot_theory::diagram::{Endpoint, PlanarDiagram};
use nalgebra::{Point3, Vector3};
use std::f64::consts::TAU;

const SCALE_XY: f64 = 1.85;
const Z_BRIDGE: f64 = 0.28;
const TIER_STEP_Z: f64 = 0.30;

#[derive(Clone, Copy)]
pub(super) struct CrossingFrame {
    pub(super) pos: [f64; 2],
    pub(super) t_over: [f64; 2],
    pub(super) t_under: [f64; 2],
    pub(super) collar: f64,
}

#[verified_engine::verified]
pub(super) fn lift_diagram_to_3d(
    diagram: &PlanarDiagram,
    samples_per_segment: usize,
) -> Result<ClosedLink3D, KnotError> {
    debug_assert!(samples_per_segment >= 4);
    let m = diagram.segments.len();
    let frames = build_crossing_frames(diagram);
    let polylines_2d = build_segment_polylines_2d(diagram, &frames);
    let z_tiers = compute_segment_z_tiers(diagram, &polylines_2d, &frames);

    let mut visited = vec![false; m];
    let mut components: Vec<Vec<Point3<f64>>> = Vec::new();
    for start in 0..m {
        if visited[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut cur = start;
        while !visited[cur] {
            visited[cur] = true;
            cycle.push(cur);
            let head = diagram.segments[cur].head;
            let c = &diagram.crossings[head.crossing];
            cur = if head.slot == 0 {
                c.under_out()
            } else {
                c.over_out()
            };
        }
        let knots = build_cycle_knots(diagram, &cycle, &frames, &polylines_2d, &z_tiers);
        components.push(eval_periodic_centripetal_catmull_rom(
            &knots,
            samples_per_segment,
        ));
    }

    append_detached_free_loops(&mut components, diagram.free_loops, samples_per_segment);
    debug_assert_eq!(components.len(), diagram.component_count());
    ClosedLink3D::new(components)
}

#[verified_engine::verified]
fn append_detached_free_loops(
    components: &mut Vec<Vec<Point3<f64>>>,
    free_loops: usize,
    samples_per_segment: usize,
) {
    let prev_len = components.len();
    let max_xy_r = components
        .iter()
        .flat_map(|c| c.iter())
        .map(|p| p.x.hypot(p.y))
        .fold(0.8_f64, f64::max);
    let base_r = if components.is_empty() {
        1.25
    } else {
        max_xy_r + 0.45
    };
    for k in 0..free_loops {
        let n_pts = samples_per_segment.max(32);
        let r = base_r + 0.35 * (k as f64);
        let loop_pts = (0..n_pts)
            .map(|i| {
                let th = TAU * (i as f64) / (n_pts as f64);
                Point3::new(r * th.cos(), r * th.sin(), 0.0)
            })
            .collect();
        components.push(loop_pts);
    }
    debug_assert_eq!(components.len(), prev_len + free_loops);
}

#[verified_engine::verified]
fn build_crossing_frames(diagram: &PlanarDiagram) -> Vec<CrossingFrame> {
    let c_len = diagram.crossings.len();
    let mut frames = Vec::with_capacity(c_len);
    for (ci, cr) in diagram.crossings.iter().enumerate() {
        let pos = [cr.position[0] * SCALE_XY, cr.position[1] * SCALE_XY];
        let (oi, oo) = (
            guide_before_head(diagram, cr.over_in()),
            guide_after_tail(diagram, cr.over_out()),
        );
        let (ui, uo) = (
            guide_before_head(diagram, cr.under_in()),
            guide_after_tail(diagram, cr.under_out()),
        );
        let t_over = bisector_dir(oi, pos, oo, [1.0, 0.0]);
        let mut t_under = bisector_dir(ui, pos, uo, [0.0, 1.0]);
        let det = t_over[0] * t_under[1] - t_over[1] * t_under[0];
        let sgn = cr.sign.value() as f64;
        if sgn * det < 0.25 {
            let ang = sgn * 1.15;
            let (sa, ca) = ang.sin_cos();
            t_under = [
                t_over[0] * ca - t_over[1] * sa,
                t_over[0] * sa + t_over[1] * ca,
            ];
        }
        let mut min_d = [oi, oo, ui, uo]
            .into_iter()
            .map(|q| (q[0] - pos[0]).hypot(q[1] - pos[1]))
            .filter(|&d| d > 1e-6)
            .fold(0.8_f64, f64::min);
        for (cj, other) in diagram.crossings.iter().enumerate() {
            if ci != cj {
                let d = ((other.position[0] - cr.position[0])
                    .hypot(other.position[1] - cr.position[1]))
                    * SCALE_XY;
                if d > 1e-6 {
                    min_d = min_d.min(0.75 * d);
                }
            }
        }
        let collar = (0.18 * min_d).clamp(1e-4, 0.12);
        frames.push(CrossingFrame {
            pos,
            t_over,
            t_under,
            collar,
        });
    }
    debug_assert_eq!(frames.len(), c_len);
    frames
}

#[verified_engine::verified]
fn guide_after_tail(diagram: &PlanarDiagram, seg_id: usize) -> [f64; 2] {
    debug_assert!(seg_id < diagram.segments.len());
    let seg = &diagram.segments[seg_id];
    let raw = seg
        .waypoints
        .first()
        .copied()
        .unwrap_or(diagram.crossings[seg.head.crossing].position);
    [raw[0] * SCALE_XY, raw[1] * SCALE_XY]
}

#[verified_engine::verified]
fn guide_before_head(diagram: &PlanarDiagram, seg_id: usize) -> [f64; 2] {
    debug_assert!(seg_id < diagram.segments.len());
    let seg = &diagram.segments[seg_id];
    let raw = seg
        .waypoints
        .last()
        .copied()
        .unwrap_or(diagram.crossings[seg.tail.crossing].position);
    [raw[0] * SCALE_XY, raw[1] * SCALE_XY]
}

#[verified_engine::verified]
fn bisector_dir(p_in: [f64; 2], center: [f64; 2], p_out: [f64; 2], fallback: [f64; 2]) -> [f64; 2] {
    let v1 = unit_vec2([center[0] - p_in[0], center[1] - p_in[1]], fallback);
    let v2 = unit_vec2([p_out[0] - center[0], p_out[1] - center[1]], v1);
    let out = unit_vec2([v1[0] + v2[0], v1[1] + v2[1]], v1);
    debug_assert!(out[0].is_finite() && out[1].is_finite());
    out
}

#[verified_engine::verified]
fn unit_vec2(v: [f64; 2], fallback: [f64; 2]) -> [f64; 2] {
    let n = v[0].hypot(v[1]);
    let out = if n > 1e-9 {
        [v[0] / n, v[1] / n]
    } else {
        fallback
    };
    debug_assert!(out[0].is_finite() && out[1].is_finite());
    out
}

#[verified_engine::verified]
fn endpoint_collar_xy(frames: &[CrossingFrame], ep: Endpoint, is_tail: bool) -> [f64; 2] {
    debug_assert!(ep.crossing < frames.len());
    let f = &frames[ep.crossing];
    let dir = if ep.slot == 0 || ep.slot == 2 {
        f.t_under
    } else {
        f.t_over
    };
    let sign = if is_tail { 1.0 } else { -1.0 };
    [
        f.pos[0] + sign * f.collar * dir[0],
        f.pos[1] + sign * f.collar * dir[1],
    ]
}

#[verified_engine::verified]
fn build_segment_polylines_2d(
    diagram: &PlanarDiagram,
    frames: &[CrossingFrame],
) -> Vec<Vec<[f64; 2]>> {
    let out: Vec<Vec<[f64; 2]>> = diagram
        .segments
        .iter()
        .map(|seg| {
            let start = endpoint_collar_xy(frames, seg.tail, true);
            let end = endpoint_collar_xy(frames, seg.head, false);
            let ft = &frames[seg.tail.crossing];
            let fh = &frames[seg.head.crossing];
            let mut pts = Vec::with_capacity(seg.waypoints.len() + 3);
            pts.push(start);
            for wp in &seg.waypoints {
                let p = [wp[0] * SCALE_XY, wp[1] * SCALE_XY];
                let dt = (p[0] - ft.pos[0]).hypot(p[1] - ft.pos[1]);
                let dh = (p[0] - fh.pos[0]).hypot(p[1] - fh.pos[1]);
                if dt > 1.05 * ft.collar && dh > 1.05 * fh.collar {
                    pts.push(p);
                }
            }
            if pts.len() == 1 {
                pts.push([0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])]);
            }
            pts.push(end);
            pts
        })
        .collect();
    debug_assert_eq!(out.len(), diagram.segments.len());
    out
}

#[verified_engine::verified]
fn build_cycle_knots(
    diagram: &PlanarDiagram,
    cycle: &[usize],
    frames: &[CrossingFrame],
    polylines: &[Vec<[f64; 2]>],
    z_tiers: &[usize],
) -> Vec<Point3<f64>> {
    debug_assert!(!cycle.is_empty());
    let m_total = diagram.segments.len().max(1) as f64;
    let mut knots = Vec::with_capacity(cycle.len() * 6);
    for &seg_id in cycle {
        let seg = &diagram.segments[seg_id];
        let z_tail = if seg.tail.slot == 2 {
            -Z_BRIDGE
        } else {
            Z_BRIDGE
        };
        let z_head = if seg.head.slot == 0 {
            -Z_BRIDGE
        } else {
            Z_BRIDGE
        };
        let ft = &frames[seg.tail.crossing];
        let poly = &polylines[seg_id];
        push_unique_knot(&mut knots, Point3::new(ft.pos[0], ft.pos[1], z_tail));
        push_unique_knot(&mut knots, Point3::new(poly[0][0], poly[0][1], z_tail));
        if z_tiers[seg_id] >= 1 {
            let z_flat = Z_BRIDGE + TIER_STEP_Z * (z_tiers[seg_id] as f64);
            for frac in [0.08, 0.28, 0.50, 0.72, 0.92] {
                let xy = sample_polyline_at(poly, frac);
                push_unique_knot(&mut knots, Point3::new(xy[0], xy[1], z_flat));
            }
        } else {
            let z_bias = 0.02 * (((seg_id as f64) / m_total) - 0.5);
            let inner = &poly[1..poly.len() - 1];
            let denom = (inner.len() + 1) as f64;
            for (idx, wp) in inner.iter().enumerate() {
                let a = ((idx + 1) as f64) / denom;
                let z = (1.0 - a) * z_tail + a * z_head + z_bias;
                push_unique_knot(&mut knots, Point3::new(wp[0], wp[1], z));
            }
        }
        let end = poly[poly.len() - 1];
        push_unique_knot(&mut knots, Point3::new(end[0], end[1], z_head));
    }
    debug_assert!(knots.len() >= 3);
    knots
}

#[verified_engine::verified]
fn push_unique_knot(knots: &mut Vec<Point3<f64>>, p: Point3<f64>) {
    debug_assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
    if knots
        .last()
        .is_none_or(|prev| (prev - p).norm_squared() > 1e-12)
    {
        knots.push(p);
    }
}

#[verified_engine::verified]
fn sample_polyline_at(poly: &[[f64; 2]], frac: f64) -> [f64; 2] {
    debug_assert!(!poly.is_empty());
    let mut total = 0.0;
    for w in poly.windows(2) {
        total += (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]);
    }
    if total <= 1e-12 {
        return poly[0];
    }
    let target = frac.clamp(0.0, 1.0) * total;
    let mut acc = 0.0;
    for w in poly.windows(2) {
        let len = (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]);
        if acc + len >= target && len > 1e-12 {
            let u = (target - acc) / len;
            return [
                w[0][0] + u * (w[1][0] - w[0][0]),
                w[0][1] + u * (w[1][1] - w[0][1]),
            ];
        }
        acc += len;
    }
    poly[poly.len() - 1]
}

#[verified_engine::verified]
fn eval_periodic_centripetal_catmull_rom(
    knots: &[Point3<f64>],
    samples_per_segment: usize,
) -> Vec<Point3<f64>> {
    let k = knots.len();
    if k < 3 {
        return knots.to_vec();
    }
    let sub = (samples_per_segment / 3).max(3);
    let mut out = Vec::with_capacity(k * sub);
    for i in 0..k {
        let p0 = knots[(i + k - 1) % k].coords;
        let p1 = knots[i].coords;
        let p2 = knots[(i + 1) % k].coords;
        let p3 = knots[(i + 2) % k].coords;
        out.push(Point3::from(p1));
        for s in 1..sub {
            let u = (s as f64) / (sub as f64);
            out.push(Point3::from(catmull_rom_centripetal_point(
                p0, p1, p2, p3, u,
            )));
        }
    }
    debug_assert_eq!(out.len(), k * sub);
    out
}

#[verified_engine::verified]
fn catmull_rom_centripetal_point(
    p0: Vector3<f64>,
    p1: Vector3<f64>,
    p2: Vector3<f64>,
    p3: Vector3<f64>,
    u: f64,
) -> Vector3<f64> {
    debug_assert!((0.0..=1.0).contains(&u));
    let dt01 = (p1 - p0).norm().sqrt().max(1e-5);
    let dt12 = (p2 - p1).norm().sqrt().max(1e-5);
    let dt23 = (p3 - p2).norm().sqrt().max(1e-5);
    let t1 = dt01;
    let t2 = t1 + dt12;
    let t3 = t2 + dt23;
    let t = t1 + u * dt12;

    let a1 = p0 * ((t1 - t) / dt01) + p1 * (t / dt01);
    let a2 = p1 * ((t2 - t) / dt12) + p2 * ((t - t1) / dt12);
    let a3 = p2 * ((t3 - t) / dt23) + p3 * ((t - t2) / dt23);

    let b1 = a1 * ((t2 - t) / t2) + a2 * (t / t2);
    let b2 = a2 * ((t3 - t) / (t3 - t1)) + a3 * ((t - t1) / (t3 - t1));

    b1 * ((t2 - t) / dt12) + b2 * ((t - t1) / dt12)
}
