//! Pairwise curve-curve and curve-reference-line intersection solvers.

use crate::{
    items::PlotPoint,
    poi::{
        CriticalPoint, CriticalPointKind, LineSeriesRef, MAX_CRITICAL_POINTS, RefLineEntry, interp,
        push_if_visible,
    },
    transform::{PlotBounds, PlotTransform},
};

pub(crate) fn collect_reference_intersections(
    lines: &[LineSeriesRef<'_>],
    hlines: &[RefLineEntry],
    vlines: &[RefLineEntry],
    bounds: &PlotBounds,
    transform: &PlotTransform,
    out: &mut Vec<CriticalPoint>,
) {
    for line in lines {
        for hline in hlines {
            for x in interp::interpolate_threshold_crossings(line.points, hline.value) {
                push_if_visible(
                    out,
                    CriticalPoint {
                        kind: CriticalPointKind::Intersection,
                        point: [x, hline.value],
                        primary_series: line.name.clone(),
                        secondary_series: Some(hline.name.clone()),
                        item_id: line.id,
                    },
                    bounds,
                    transform,
                );
            }
        }
        for vline in vlines {
            for y in interp::interpolate_y_at_x(line.points, vline.value) {
                push_if_visible(
                    out,
                    CriticalPoint {
                        kind: CriticalPointKind::Intersection,
                        point: [vline.value, y],
                        primary_series: line.name.clone(),
                        secondary_series: Some(vline.name.clone()),
                        item_id: line.id,
                    },
                    bounds,
                    transform,
                );
            }
        }
    }
    for hline in hlines {
        for vline in vlines {
            push_if_visible(
                out,
                CriticalPoint {
                    kind: CriticalPointKind::Intersection,
                    point: [vline.value, hline.value],
                    primary_series: hline.name.clone(),
                    secondary_series: Some(vline.name.clone()),
                    item_id: hline.id,
                },
                bounds,
                transform,
            );
        }
    }
}

/// Maximum number of hoverable line series eligible for $O(K^2)$ pairwise intersection scans.
pub const MAX_PAIRWISE_INTERSECTION_SERIES: usize = 16;

#[derive(Clone, Copy)]
struct SeriesBoundsSummary {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    monotone: bool,
    visible: bool,
}

fn summarize_series_bounds(pts: &[PlotPoint], bounds: &PlotBounds) -> SeriesBoundsSummary {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut monotone = pts.len() >= 2;
    let mut finite_count = 0_usize;
    let mut prev_x = f64::NEG_INFINITY;
    for (idx, &pt) in pts.iter().enumerate() {
        if !interp::is_finite_pt(pt) {
            monotone = false;
            continue;
        }
        if idx > 0 && pt.x <= prev_x {
            monotone = false;
        }
        prev_x = pt.x;
        finite_count += 1;
        min_x = min_x.min(pt.x);
        max_x = max_x.max(pt.x);
        min_y = min_y.min(pt.y);
        max_y = max_y.max(pt.y);
    }
    let tol_x = (bounds.width().abs() * 1e-9).max(1e-12);
    let tol_y = (bounds.height().abs() * 1e-9).max(1e-12);
    let visible = finite_count >= 2
        && max_x >= bounds.min[0] - tol_x
        && min_x <= bounds.max[0] + tol_x
        && max_y >= bounds.min[1] - tol_y
        && min_y <= bounds.max[1] + tol_y;
    SeriesBoundsSummary {
        min_x,
        max_x,
        min_y,
        max_y,
        monotone,
        visible,
    }
}

pub(crate) fn collect_pairwise_line_intersections(
    lines: &[LineSeriesRef<'_>],
    bounds: &PlotBounds,
    transform: &PlotTransform,
    out: &mut Vec<CriticalPoint>,
) {
    if lines.len() < 2 || lines.len() > MAX_PAIRWISE_INTERSECTION_SERIES {
        return;
    }
    let summaries: Vec<SeriesBoundsSummary> = lines
        .iter()
        .map(|line| summarize_series_bounds(line.points, bounds))
        .collect();
    for i in 0..lines.len() {
        let sa = summaries[i];
        if !sa.visible {
            continue;
        }
        for j in (i + 1)..lines.len() {
            if out.len() >= MAX_CRITICAL_POINTS * 2 {
                return;
            }
            let sb = summaries[j];
            if !sb.visible
                || sa.max_y < sb.min_y
                || sb.max_y < sa.min_y
                || sa.max_x < sb.min_x
                || sb.max_x < sa.min_x
            {
                continue;
            }
            let a = &lines[i];
            let b = &lines[j];
            let crossings = if sa.monotone && sb.monotone {
                intersect_monotone_polylines(a.points, b.points, bounds)
            } else {
                intersect_general_polylines(a.points, b.points, bounds)
            };
            for pt in crossings {
                push_if_visible(
                    out,
                    CriticalPoint {
                        kind: CriticalPointKind::Intersection,
                        point: pt,
                        primary_series: a.name.clone(),
                        secondary_series: Some(b.name.clone()),
                        item_id: a.id,
                    },
                    bounds,
                    transform,
                );
            }
        }
    }
}

fn intersect_monotone_polylines(
    a: &[PlotPoint],
    b: &[PlotPoint],
    bounds: &PlotBounds,
) -> Vec<[f64; 2]> {
    let x_lo = a[0].x.max(b[0].x).max(bounds.min[0]);
    let x_hi = a[a.len() - 1].x.min(b[b.len() - 1].x).min(bounds.max[0]);
    if !x_lo.is_finite() || !x_hi.is_finite() || x_hi <= x_lo {
        return Vec::new();
    }
    let knots = merged_knots(a, b, x_lo, x_hi);
    let mut hits = Vec::new();
    let mut ia = 0_usize;
    let mut ib = 0_usize;
    for k in 0..knots.len().saturating_sub(1) {
        let (x0, x1) = (knots[k], knots[k + 1]);
        if x1 <= x0 {
            continue;
        }
        advance_cursor(a, x0, &mut ia);
        advance_cursor(b, x0, &mut ib);
        let (Some(ya0), Some(yb0)) = (
            interp::eval_segment_at_x(a, ia, x0),
            interp::eval_segment_at_x(b, ib, x0),
        ) else {
            continue;
        };
        advance_cursor(a, x1, &mut ia);
        advance_cursor(b, x1, &mut ib);
        let (Some(ya1), Some(yb1)) = (
            interp::eval_segment_at_x(a, ia, x1),
            interp::eval_segment_at_x(b, ib, x1),
        ) else {
            continue;
        };
        let d0 = ya0 - yb0;
        let d1 = ya1 - yb1;
        if d0 == 0.0 && d1 == 0.0 {
            continue;
        }
        if d0 == 0.0 {
            hits.push([x0, ya0]);
        } else if (d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0) {
            if let Some(pt) = refine_monotone_crossing(a, b, x0, x1, d0, d1) {
                hits.push(pt);
            }
        } else if k + 2 == knots.len() && d1 == 0.0 {
            hits.push([x1, ya1]);
        }
    }
    hits
}

fn merged_knots(a: &[PlotPoint], b: &[PlotPoint], x_lo: f64, x_hi: f64) -> Vec<f64> {
    let mut knots = Vec::with_capacity(a.len() + b.len() + 2);
    knots.push(x_lo);
    knots.extend(a.iter().map(|p| p.x).filter(|&x| x > x_lo && x < x_hi));
    knots.extend(b.iter().map(|p| p.x).filter(|&x| x > x_lo && x < x_hi));
    knots.push(x_hi);
    knots.sort_by(f64::total_cmp);
    knots.dedup_by(|u, v| (*u - *v).abs() <= 1e-12);
    knots
}

fn advance_cursor(pts: &[PlotPoint], x: f64, idx: &mut usize) {
    while *idx + 2 < pts.len() && pts[*idx + 1].x < x {
        *idx += 1;
    }
    while *idx > 0 && pts[*idx].x > x {
        *idx -= 1;
    }
}

fn refine_monotone_crossing(
    a: &[PlotPoint],
    b: &[PlotPoint],
    mut lo: f64,
    mut hi: f64,
    mut d_lo: f64,
    mut d_hi: f64,
) -> Option<[f64; 2]> {
    let mut ia = 0_usize;
    let mut ib = 0_usize;
    for _ in 0..14 {
        let denom = d_hi - d_lo;
        let sec = if denom.abs() > 1e-18 {
            lo - d_lo * (hi - lo) / denom
        } else {
            0.5 * (lo + hi)
        };
        let mid = if sec > lo && sec < hi {
            0.5 * (sec + 0.5 * (lo + hi))
        } else {
            0.5 * (lo + hi)
        };
        advance_cursor(a, mid, &mut ia);
        advance_cursor(b, mid, &mut ib);
        let ya = interp::eval_segment_at_x(a, ia, mid)?;
        let yb = interp::eval_segment_at_x(b, ib, mid)?;
        let dm = ya - yb;
        if dm.abs() <= 1e-12 || (hi - lo).abs() <= 1e-12 {
            return Some([mid, 0.5 * (ya + yb)]);
        }
        if (dm > 0.0) == (d_lo > 0.0) {
            lo = mid;
            d_lo = dm;
        } else {
            hi = mid;
            d_hi = dm;
        }
    }
    let xm = 0.5 * (lo + hi);
    advance_cursor(a, xm, &mut ia);
    let ym = interp::eval_segment_at_x(a, ia, xm)?;
    Some([xm, ym])
}

fn intersect_general_polylines(
    a: &[PlotPoint],
    b: &[PlotPoint],
    bounds: &PlotBounds,
) -> Vec<[f64; 2]> {
    let mut segs_b: Vec<(f64, f64, PlotPoint, PlotPoint)> = b
        .windows(2)
        .filter(|w| interp::is_finite_pt(w[0]) && interp::is_finite_pt(w[1]))
        .filter_map(|w| {
            let min_x = w[0].x.min(w[1].x);
            let max_x = w[0].x.max(w[1].x);
            (max_x >= bounds.min[0] && min_x <= bounds.max[0]).then_some((min_x, max_x, w[0], w[1]))
        })
        .collect();
    segs_b.sort_by(|u, v| u.0.total_cmp(&v.0));

    let mut hits = Vec::new();
    for wa in a.windows(2) {
        if !interp::is_finite_pt(wa[0]) || !interp::is_finite_pt(wa[1]) {
            continue;
        }
        let a_min_x = wa[0].x.min(wa[1].x);
        let a_max_x = wa[0].x.max(wa[1].x);
        if a_max_x < bounds.min[0] || a_min_x > bounds.max[0] {
            continue;
        }
        let a_min_y = wa[0].y.min(wa[1].y);
        let a_max_y = wa[0].y.max(wa[1].y);
        for &(b_min_x, b_max_x, b0, b1) in &segs_b {
            if b_min_x > a_max_x {
                break;
            }
            if b_max_x < a_min_x || b0.y.max(b1.y) < a_min_y || b0.y.min(b1.y) > a_max_y {
                continue;
            }
            if let Some(pt) = segment_intersection_2d(wa[0], wa[1], b0, b1) {
                hits.push(pt);
                if hits.len() >= MAX_CRITICAL_POINTS {
                    return hits;
                }
            }
        }
    }
    hits
}

fn segment_intersection_2d(
    a0: PlotPoint,
    a1: PlotPoint,
    b0: PlotPoint,
    b1: PlotPoint,
) -> Option<[f64; 2]> {
    let r_x = a1.x - a0.x;
    let r_y = a1.y - a0.y;
    let s_x = b1.x - b0.x;
    let s_y = b1.y - b0.y;
    let denom = r_x * s_y - r_y * s_x;
    if !denom.is_finite() || denom.abs() <= 1e-15 {
        return None;
    }
    let qp_x = b0.x - a0.x;
    let qp_y = b0.y - a0.y;
    let t = (qp_x * s_y - qp_y * s_x) / denom;
    let u = (qp_x * r_y - qp_y * r_x) / denom;
    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        let x = a0.x + t * r_x;
        let y = a0.y + t * r_y;
        (x.is_finite() && y.is_finite()).then_some([x, y])
    } else {
        None
    }
}
