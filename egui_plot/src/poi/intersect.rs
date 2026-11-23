//! Pairwise curve-curve and curve-reference-line intersection UI adapters delegating to `math_commons::poi`.

use math_commons::series::PointSeries;

use crate::{
    poi::{
        CriticalPoint, CriticalPointKind, LineSeriesRef, MAX_CRITICAL_POINTS, RefLineEntry,
        interpolate_threshold_crossings, interpolate_y_at_x, intersect_polylines_into,
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
            for x in interpolate_threshold_crossings(&line.points, hline.value) {
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
            for y in interpolate_y_at_x(&line.points, vline.value) {
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
    visible: bool,
}

fn summarize_series_bounds<P: PointSeries + ?Sized>(
    pts: &P,
    bounds: &PlotBounds,
) -> SeriesBoundsSummary {
    let n = pts.len();
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut finite_count = 0_usize;
    for idx in 0..n {
        let [x, y] = pts.point_at(idx);
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        finite_count += 1;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
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
    let mut stack_hits = [[0.0_f64; 2]; MAX_CRITICAL_POINTS];
    let x_bounds = [bounds.min[0], bounds.max[0]];
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
            let count = intersect_polylines_into(&a.points, &b.points, x_bounds, &mut stack_hits);
            for &pt in &stack_hits[..count] {
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
