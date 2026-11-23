//! Runge-Kutta streamline tracing and $\psi$-level seed selection for [`PotentialFlowField`].

use super::potential_flow::{FlowElement, PotentialFlowField};
use math_commons::contours::{
    ContourPath, GridBounds2D, Point2D, StreamlineConfig, trace_streamlines_rk4,
};

impl PotentialFlowField {
    /// Generates seed coordinates across `[-view_range, view_range]^2` ordered and filtered by
    /// [`FlowElement::stream_function`] $\psi(x, y)$ so streamlines sample distinct flow tubes.
    #[verified_engine::verified]
    pub fn default_streamline_seeds(&self, view_range: f64, count: usize) -> Vec<Point2D> {
        if !view_range.is_finite() || view_range <= 0.0 || count == 0 || self.elements.is_empty() {
            return Vec::new();
        }
        let target = count.clamp(2, 64);
        let candidates = collect_seed_candidates(self, view_range, (target * 3).max(16));
        if candidates.is_empty() {
            return Vec::new();
        }
        select_psi_spaced_seeds(&candidates, view_range, target)
    }

    /// Traces bidirectional RK4 streamlines across `[-view_range, view_range]^2`, extracts any
    /// closed stagnation dividing-streamline body contours (such as a cylinder or Rankine oval),
    /// and filters out duplicate or interior-singularity trajectories.
    #[verified_engine::verified]
    pub fn trace_streamlines(&self, view_range: f64, count: usize) -> Vec<ContourPath> {
        if !view_range.is_finite() || view_range <= 0.0 || count == 0 || self.elements.is_empty() {
            return Vec::new();
        }
        let Ok(bounds) = GridBounds2D::new(-view_range, view_range, -view_range, view_range) else {
            return Vec::new();
        };
        let Ok(cfg) = StreamlineConfig::new(bounds, (view_range * 0.035).max(1e-3), 450, 1e-4)
        else {
            return Vec::new();
        };
        let body_loops = extract_dividing_body_loops(self, &bounds);
        let seeds = self.default_streamline_seeds(view_range, count);
        let raw = trace_streamlines_rk4(&seeds, &cfg, |x, y| {
            let v = self.velocity(x, y);
            [v.x, v.y]
        })
        .unwrap_or_default();

        let exterior: Vec<ContourPath> = raw
            .into_iter()
            .filter(|p| {
                p.points.len() >= 3
                    && !body_loops
                        .iter()
                        .any(|body| point_in_polygon(p.points[p.points.len() / 2], &body.points))
            })
            .collect();
        let mut kept = deduplicate_streamlines_by_psi(self, exterior, view_range);
        kept.extend(body_loops);
        kept
    }
}

#[verified_engine::verified]
fn collect_seed_candidates(
    field: &PotentialFlowField,
    view_range: f64,
    samples: usize,
) -> Vec<(Point2D, f64)> {
    let mut candidates = Vec::with_capacity(samples * 3);
    let span = 2.0 * view_range;
    let inset = view_range * 0.96;
    for idx in 0..samples {
        let frac = ((idx as f64) + 0.5) / (samples as f64);
        let coord = -view_range + frac * span;
        for pt in [[-inset, coord], [coord, -inset], [0.0, coord]] {
            let v = field.velocity(pt[0], pt[1]);
            let psi = field.stream_function(pt[0], pt[1]);
            if v.x.is_finite() && v.y.is_finite() && v.norm_squared() > 1e-8 && psi.is_finite() {
                candidates.push((pt, psi));
            }
        }
    }
    candidates
}

#[verified_engine::verified]
fn select_psi_spaced_seeds(
    candidates: &[(Point2D, f64)],
    view_range: f64,
    target: usize,
) -> Vec<Point2D> {
    let (psi_min, psi_max) = candidates
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(_, p)| {
            (lo.min(p), hi.max(p))
        });
    if (psi_max - psi_min).abs() <= 1e-9 {
        let stride = (candidates.len() / target).max(1);
        return candidates
            .iter()
            .step_by(stride)
            .take(target)
            .map(|(pt, _)| *pt)
            .collect();
    }
    let min_dist_sq = (0.08 * view_range).powi(2);
    let mut selected: Vec<Point2D> = Vec::with_capacity(target);
    for level_idx in 0..target {
        let target_psi =
            psi_min + (((level_idx as f64) + 0.5) / (target as f64)) * (psi_max - psi_min);
        if let Some((best_pt, _)) = candidates
            .iter()
            .filter(|(cand, _)| {
                selected.iter().all(|s| {
                    let dx = cand[0] - s[0];
                    let dy = cand[1] - s[1];
                    dx * dx + dy * dy > min_dist_sq
                })
            })
            .min_by(|a, b| {
                (a.1 - target_psi)
                    .abs()
                    .total_cmp(&(b.1 - target_psi).abs())
            })
        {
            selected.push(*best_pt);
        }
    }
    selected
}

#[verified_engine::verified]
fn deduplicate_streamlines_by_psi(
    field: &PotentialFlowField,
    paths: Vec<ContourPath>,
    view_range: f64,
) -> Vec<ContourPath> {
    let min_sep_sq = (view_range * 0.045).powi(2);
    let mut accepted: Vec<ContourPath> = Vec::with_capacity(paths.len());
    let mut accepted_samples: Vec<(Point2D, f64)> = Vec::new();

    for path in paths {
        let mid = path.points[path.points.len() / 2];
        let psi_mid = field.stream_function(mid[0], mid[1]);
        let overlaps = accepted_samples.iter().any(|&(prev_pt, prev_psi)| {
            let dx = mid[0] - prev_pt[0];
            let dy = mid[1] - prev_pt[1];
            dx * dx + dy * dy < min_sep_sq
                && (psi_mid - prev_psi).abs() < 1e-2 * view_range.max(1.0)
        });
        if !overlaps {
            let stride = (path.points.len() / 6).max(1);
            for &pt in path.points.iter().step_by(stride) {
                accepted_samples.push((pt, field.stream_function(pt[0], pt[1])));
            }
            accepted.push(path);
        }
    }
    accepted
}

#[verified_engine::verified]
fn extract_dividing_body_loops(
    field: &PotentialFlowField,
    bounds: &GridBounds2D,
) -> Vec<ContourPath> {
    let stag_pts = find_stagnation_points(field, bounds);
    if stag_pts.is_empty() {
        return Vec::new();
    }
    let span = bounds.span_x().min(bounds.span_y());
    let eps = 0.008 * span;
    let ds = 0.01 * span;
    let mut loops = Vec::new();

    for (start_idx, &stag) in stag_pts.iter().enumerate() {
        let Some(dir) = unstable_eigenvector(field, stag, 1e-5 * span) else {
            continue;
        };
        let seed_pos = [stag[0] + eps * dir[0], stag[1] + eps * dir[1]];
        let seed_neg = [stag[0] - eps * dir[0], stag[1] - eps * dir[1]];
        let pos_arc = trace_forward_separatrix(field, bounds, stag, seed_pos, &stag_pts, ds, eps);
        let neg_arc = trace_forward_separatrix(field, bounds, stag, seed_neg, &stag_pts, ds, eps);
        if let (Some((end_a, mut pts_a)), Some((end_b, pts_b))) = (pos_arc, neg_arc)
            && end_a == end_b
            && (end_a != start_idx || pts_a.len() > 12)
        {
            for &pt in pts_b[1..pts_b.len().saturating_sub(1)].iter().rev() {
                pts_a.push(pt);
            }
            if pts_a.len() >= 8 {
                loops.push(ContourPath::new(pts_a, true));
            }
        }
    }
    loops
}

#[verified_engine::verified]
fn find_stagnation_points(field: &PotentialFlowField, bounds: &GridBounds2D) -> Vec<Point2D> {
    let span = bounds.span_x().min(bounds.span_y());
    let v_ref = estimate_boundary_speed(field, bounds);
    if v_ref <= 1e-9 {
        return Vec::new();
    }
    let grid_n = 28usize;
    let mut found: Vec<Point2D> = Vec::new();
    for r in 1..grid_n {
        let y = bounds.min_y + ((r as f64) / (grid_n as f64)) * bounds.span_y();
        for c in 1..grid_n {
            let x = bounds.min_x + ((c as f64) / (grid_n as f64)) * bounds.span_x();
            if field.velocity(x, y).norm() > 0.75 * v_ref {
                continue;
            }
            if let Some(stag) = refine_stagnation_newton(field, bounds, [x, y], span, v_ref) {
                let dup = found
                    .iter()
                    .any(|p| (p[0] - stag[0]).hypot(p[1] - stag[1]) < 1e-2 * span);
                if !dup {
                    found.push(stag);
                }
            }
        }
    }
    found
}

#[verified_engine::verified]
fn estimate_boundary_speed(field: &PotentialFlowField, bounds: &GridBounds2D) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..8 {
        let t = ((i as f64) + 0.5) / 8.0;
        let x = bounds.min_x + t * bounds.span_x();
        let y = bounds.min_y + t * bounds.span_y();
        for [px, py] in [
            [bounds.min_x, y],
            [bounds.max_x, y],
            [x, bounds.min_y],
            [x, bounds.max_y],
        ] {
            let spd = field.velocity(px, py).norm();
            if spd.is_finite() {
                sum += spd;
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / (count as f64)
    }
}

#[verified_engine::verified]
fn refine_stagnation_newton(
    field: &PotentialFlowField,
    bounds: &GridBounds2D,
    start: Point2D,
    span: f64,
    v_ref: f64,
) -> Option<Point2D> {
    let h = (1e-6 * span).max(1e-8);
    let mut pt = start;
    for _ in 0..12 {
        let f = field.velocity(pt[0], pt[1]);
        if !f.x.is_finite() || !f.y.is_finite() {
            return None;
        }
        if f.norm() < 1e-8 * v_ref.max(1.0) {
            return Some(pt);
        }
        let fx = (field.velocity(pt[0] + h, pt[1]) - field.velocity(pt[0] - h, pt[1])) / (2.0 * h);
        let fy = (field.velocity(pt[0], pt[1] + h) - field.velocity(pt[0], pt[1] - h)) / (2.0 * h);
        let det = fx.x * fy.y - fy.x * fx.y;
        if !det.is_finite() || det.abs() < 1e-14 {
            return None;
        }
        let dx = (fy.y * f.x - fy.x * f.y) / det;
        let dy = (-fx.y * f.x + fx.x * f.y) / det;
        if dx.hypot(dy) > 0.25 * span {
            return None;
        }
        pt[0] -= dx;
        pt[1] -= dy;
        if !bounds.contains(pt) {
            return None;
        }
    }
    (field.velocity(pt[0], pt[1]).norm() < 1e-6 * v_ref.max(1.0)).then_some(pt)
}

#[verified_engine::verified]
fn unstable_eigenvector(field: &PotentialFlowField, stag: Point2D, h: f64) -> Option<Point2D> {
    let dfdx =
        (field.velocity(stag[0] + h, stag[1]) - field.velocity(stag[0] - h, stag[1])) / (2.0 * h);
    let dfdy =
        (field.velocity(stag[0], stag[1] + h) - field.velocity(stag[0], stag[1] - h)) / (2.0 * h);
    let a = 0.5 * (dfdx.x - dfdy.y);
    let b = 0.5 * (dfdy.x + dfdx.y);
    let lambda = a.hypot(b);
    if !lambda.is_finite() || lambda <= 1e-10 {
        return None;
    }
    let cand = if (lambda - a) >= (lambda + a) {
        [b, lambda - a]
    } else {
        [lambda + a, b]
    };
    let n = cand[0].hypot(cand[1]);
    (n > 1e-12).then_some([cand[0] / n, cand[1] / n])
}

#[verified_engine::verified]
fn trace_forward_separatrix(
    field: &PotentialFlowField,
    bounds: &GridBounds2D,
    start_stag: Point2D,
    seed: Point2D,
    stag_pts: &[Point2D],
    ds: f64,
    capture_radius: f64,
) -> Option<(usize, Vec<Point2D>)> {
    let mut pts = Vec::with_capacity(256);
    pts.push(start_stag);
    pts.push(seed);
    let mut cur = seed;
    for step in 0..700 {
        let v1 = unit_vel(field, cur)?;
        let p2 = [cur[0] + 0.5 * ds * v1[0], cur[1] + 0.5 * ds * v1[1]];
        let v2 = unit_vel(field, p2)?;
        let p3 = [cur[0] + 0.5 * ds * v2[0], cur[1] + 0.5 * ds * v2[1]];
        let v3 = unit_vel(field, p3)?;
        let p4 = [cur[0] + ds * v3[0], cur[1] + ds * v3[1]];
        let v4 = unit_vel(field, p4)?;
        let next = [
            cur[0] + (ds / 6.0) * (v1[0] + 2.0 * v2[0] + 2.0 * v3[0] + v4[0]),
            cur[1] + (ds / 6.0) * (v1[1] + 2.0 * v2[1] + 2.0 * v3[1] + v4[1]),
        ];
        if !bounds.contains(next) {
            return None;
        }
        pts.push(next);
        cur = next;
        if step >= 4 {
            for (idx, &sp) in stag_pts.iter().enumerate() {
                if (cur[0] - sp[0]).hypot(cur[1] - sp[1]) <= 1.8 * capture_radius {
                    pts.push(sp);
                    return Some((idx, pts));
                }
            }
        }
    }
    None
}

#[verified_engine::verified]
fn unit_vel(field: &PotentialFlowField, pt: Point2D) -> Option<Point2D> {
    let v = field.velocity(pt[0], pt[1]);
    let n = v.norm();
    (v.x.is_finite() && v.y.is_finite() && n > 1e-10 && n < 1e6).then_some([v.x / n, v.y / n])
}

#[verified_engine::verified]
fn point_in_polygon(pt: Point2D, poly: &[Point2D]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let pi = poly[i];
        let pj = poly[j];
        let crosses = (pi[1] > pt[1]) != (pj[1] > pt[1]);
        if crosses {
            let x_cross = (pj[0] - pi[0]) * (pt[1] - pi[1]) / (pj[1] - pi[1]) + pi[0];
            if pt[0] < x_cross {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use crate::physics::fluid_dynamics::potential_flow::{
        Doublet, FlowElement, PotentialFlowField, UniformFlow,
    };
    use std::f64::consts::PI;

    #[test]
    fn test_cylinder_potential_flow_traces_rk4_streamlines_preserving_psi() {
        let mut field = PotentialFlowField::new();
        let u_inf = 10.0;
        let radius = 2.0;
        let kappa = 2.0 * PI * u_inf * radius * radius;
        field.add(Box::new(UniformFlow::new(u_inf, 0.0)));
        field.add(Box::new(Doublet::new(kappa, 0.0, 0.0)));

        let lines = field.trace_streamlines(10.0, 12);
        assert!(
            lines.len() >= 6,
            "expected multiple streamlines, got {}",
            lines.len()
        );

        let open_line = lines
            .iter()
            .find(|p| !p.closed && p.points.iter().all(|pt| pt[0].hypot(pt[1]) > radius * 1.05))
            .expect("expected exterior open RK4 streamline");
        let psi0 = field.stream_function(open_line.points[0][0], open_line.points[0][1]);
        for pt in &open_line.points {
            let psi = field.stream_function(pt[0], pt[1]);
            assert!(
                (psi - psi0).abs() < 0.25,
                "RK4 streamline drifted off stream_function level: {psi} vs {psi0}"
            );
        }

        let dividing_loop = lines
            .iter()
            .find(|p| p.closed)
            .expect("expected closed psi=0 stagnation dividing streamline around cylinder");
        for pt in &dividing_loop.points {
            let r = pt[0].hypot(pt[1]);
            assert!(
                (r - radius).abs() < 0.05,
                "stagnation dividing loop point {pt:?} deviated from cylinder radius {radius} (r={r})"
            );
        }
    }
}
