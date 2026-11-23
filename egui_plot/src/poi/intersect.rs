//! Pairwise curve-curve and curve-reference-line intersection solvers.

use crate::{
    items::PlotPoint,
    poi::{
        CriticalPoint, CriticalPointKind, LineSeriesRef, MAX_CRITICAL_POINTS, RefLineEntry,
        interp::{self, CoordSlice},
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
            for x in interp::interpolate_threshold_crossings_slice(&line.points, hline.value) {
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
            for y in interp::interpolate_y_at_x_slice(&line.points, vline.value) {
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

fn summarize_series_bounds<C: CoordSlice + ?Sized>(
    pts: &C,
    bounds: &PlotBounds,
) -> SeriesBoundsSummary {
    let n = pts.coord_len();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut monotone = n >= 2;
    let mut finite_count = 0_usize;
    let mut prev_x = f64::NEG_INFINITY;
    for idx in 0..n {
        let pt = pts.coord_at(idx);
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
        .map(|line| summarize_series_bounds(&line.points, bounds))
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
                intersect_monotone_polylines(&a.points, &b.points, bounds)
            } else {
                intersect_general_polylines(&a.points, &b.points, bounds)
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

/// Streamingly intersect two strictly $x$-increasing polylines over `bounds`
/// using a two-pointer knot cursor without allocating a merged knot vector.
pub(crate) fn intersect_monotone_polylines<A: CoordSlice + ?Sized, B: CoordSlice + ?Sized>(
    a: &A,
    b: &B,
    bounds: &PlotBounds,
) -> Vec<[f64; 2]> {
    let na = a.coord_len();
    let nb = b.coord_len();
    if na < 2 || nb < 2 {
        return Vec::new();
    }
    let x_lo = a.coord_at(0).x.max(b.coord_at(0).x).max(bounds.min[0]);
    let x_hi = a
        .coord_at(na - 1)
        .x
        .min(b.coord_at(nb - 1).x)
        .min(bounds.max[0]);
    if !x_lo.is_finite() || !x_hi.is_finite() || x_hi <= x_lo {
        return Vec::new();
    }

    let mut stream = KnotStream::new(a, b, x_lo, x_hi);
    let mut x0 = x_lo;
    let Some(mut x1) = stream.next_distinct_knot(x0) else {
        return Vec::new();
    };
    let mut ia = stream.initial_segment_a();
    let mut ib = stream.initial_segment_b();
    let mut hits = Vec::new();

    loop {
        let next_after = stream.next_distinct_knot(x1);
        let is_last = next_after.is_none();
        if x1 > x0 {
            evaluate_knot_interval(a, b, x0, x1, is_last, &mut ia, &mut ib, &mut hits);
        }
        let Some(x2) = next_after else {
            break;
        };
        x0 = x1;
        x1 = x2;
    }
    hits
}

struct KnotStream<'a, A: CoordSlice + ?Sized, B: CoordSlice + ?Sized> {
    a: &'a A,
    b: &'a B,
    ka: usize,
    kb: usize,
    init_ia: usize,
    init_ib: usize,
    x_hi: f64,
    emitted_hi: bool,
}

impl<'a, A: CoordSlice + ?Sized, B: CoordSlice + ?Sized> KnotStream<'a, A, B> {
    fn new(a: &'a A, b: &'a B, x_lo: f64, x_hi: f64) -> Self {
        let ka = first_gt_index(a, x_lo);
        let kb = first_gt_index(b, x_lo);
        Self {
            a,
            b,
            ka,
            kb,
            init_ia: ka.saturating_sub(1).min(a.coord_len().saturating_sub(2)),
            init_ib: kb.saturating_sub(1).min(b.coord_len().saturating_sub(2)),
            x_hi,
            emitted_hi: false,
        }
    }

    #[inline]
    const fn initial_segment_a(&self) -> usize {
        self.init_ia
    }

    #[inline]
    const fn initial_segment_b(&self) -> usize {
        self.init_ib
    }

    fn next_raw(&mut self) -> Option<f64> {
        let xa = (self.ka < self.a.coord_len())
            .then(|| self.a.coord_at(self.ka).x)
            .filter(|&x| x < self.x_hi);
        let xb = (self.kb < self.b.coord_len())
            .then(|| self.b.coord_at(self.kb).x)
            .filter(|&x| x < self.x_hi);
        match (xa, xb) {
            (Some(va), Some(vb)) => {
                if va.total_cmp(&vb).is_le() {
                    self.ka += 1;
                    Some(va)
                } else {
                    self.kb += 1;
                    Some(vb)
                }
            }
            (Some(va), None) => {
                self.ka += 1;
                Some(va)
            }
            (None, Some(vb)) => {
                self.kb += 1;
                Some(vb)
            }
            (None, None) => {
                if self.emitted_hi {
                    None
                } else {
                    self.emitted_hi = true;
                    Some(self.x_hi)
                }
            }
        }
    }

    fn next_distinct_knot(&mut self, prev_kept: f64) -> Option<f64> {
        while let Some(cand) = self.next_raw() {
            if (cand - prev_kept).abs() > 1e-12 {
                return Some(cand);
            }
        }
        None
    }
}

fn first_gt_index<C: CoordSlice + ?Sized>(pts: &C, x_lo: f64) -> usize {
    let mut lo = 0_usize;
    let mut hi = pts.coord_len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pts.coord_at(mid).x <= x_lo {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_knot_interval<A: CoordSlice + ?Sized, B: CoordSlice + ?Sized>(
    a: &A,
    b: &B,
    x0: f64,
    x1: f64,
    is_last: bool,
    ia: &mut usize,
    ib: &mut usize,
    hits: &mut Vec<[f64; 2]>,
) {
    advance_cursor(a, x0, ia);
    advance_cursor(b, x0, ib);
    let ia0 = *ia;
    let ib0 = *ib;
    let (Some(ya0), Some(yb0)) = (
        interp::eval_segment_at_x(a, ia0, x0),
        interp::eval_segment_at_x(b, ib0, x0),
    ) else {
        return;
    };
    advance_cursor(a, x1, ia);
    advance_cursor(b, x1, ib);
    let (Some(ya1), Some(yb1)) = (
        interp::eval_segment_at_x(a, *ia, x1),
        interp::eval_segment_at_x(b, *ib, x1),
    ) else {
        return;
    };
    let d0 = ya0 - yb0;
    let d1 = ya1 - yb1;
    if d0 == 0.0 && d1 == 0.0 {
        return;
    }
    if d0 == 0.0 {
        hits.push([x0, ya0]);
    } else if (d0 > 0.0 && d1 < 0.0) || (d0 < 0.0 && d1 > 0.0) {
        if let Some(pt) = refine_monotone_crossing(a, b, x0, x1, d0, d1, ia0, ib0) {
            hits.push(pt);
        }
    } else if is_last && d1 == 0.0 {
        hits.push([x1, ya1]);
    }
}

fn advance_cursor<C: CoordSlice + ?Sized>(pts: &C, x: f64, idx: &mut usize) {
    let n = pts.coord_len();
    while *idx + 2 < n && pts.coord_at(*idx + 1).x < x {
        *idx += 1;
    }
    while *idx > 0 && pts.coord_at(*idx).x > x {
        *idx -= 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_monotone_crossing<A: CoordSlice + ?Sized, B: CoordSlice + ?Sized>(
    a: &A,
    b: &B,
    mut lo: f64,
    mut hi: f64,
    mut d_lo: f64,
    mut d_hi: f64,
    mut ia: usize,
    mut ib: usize,
) -> Option<[f64; 2]> {
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

fn intersect_general_polylines<A: CoordSlice + ?Sized, B: CoordSlice + ?Sized>(
    a: &A,
    b: &B,
    bounds: &PlotBounds,
) -> Vec<[f64; 2]> {
    let nb = b.coord_len();
    let mut segs_b: Vec<(f64, f64, PlotPoint, PlotPoint)> = (0..nb.saturating_sub(1))
        .map(|i| (b.coord_at(i), b.coord_at(i + 1)))
        .filter(|&(p0, p1)| interp::is_finite_pt(p0) && interp::is_finite_pt(p1))
        .filter_map(|(p0, p1)| {
            let min_x = p0.x.min(p1.x);
            let max_x = p0.x.max(p1.x);
            (max_x >= bounds.min[0] && min_x <= bounds.max[0]).then_some((min_x, max_x, p0, p1))
        })
        .collect();
    segs_b.sort_by(|u, v| u.0.total_cmp(&v.0));

    let na = a.coord_len();
    let mut hits = Vec::new();
    for i in 0..na.saturating_sub(1) {
        let (wa0, wa1) = (a.coord_at(i), a.coord_at(i + 1));
        if !interp::is_finite_pt(wa0) || !interp::is_finite_pt(wa1) {
            continue;
        }
        let a_min_x = wa0.x.min(wa1.x);
        let a_max_x = wa0.x.max(wa1.x);
        if a_max_x < bounds.min[0] || a_min_x > bounds.max[0] {
            continue;
        }
        let a_min_y = wa0.y.min(wa1.y);
        let a_max_y = wa0.y.max(wa1.y);
        for &(b_min_x, b_max_x, b0, b1) in &segs_b {
            if b_min_x > a_max_x {
                break;
            }
            if b_max_x < a_min_x || b0.y.max(b1.y) < a_min_y || b0.y.min(b1.y) > a_max_y {
                continue;
            }
            if let Some(pt) = segment_intersection_2d(wa0, wa1, b0, b1) {
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
