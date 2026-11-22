//! Pairwise curve-curve and curve-reference-line intersection UI adapters delegating to `math_commons::poi`.

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
            for x in interpolate_threshold_crossings(line.points, hline.value) {
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
            for y in interpolate_y_at_x(line.points, vline.value) {
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

pub(crate) fn collect_pairwise_line_intersections(
    lines: &[LineSeriesRef<'_>],
    bounds: &PlotBounds,
    transform: &PlotTransform,
    out: &mut Vec<CriticalPoint>,
) {
    let mut stack_hits = [[0.0_f64; 2]; MAX_CRITICAL_POINTS];
    let x_bounds = [bounds.min[0], bounds.max[0]];
    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            if out.len() >= MAX_CRITICAL_POINTS * 2 {
                return;
            }
            let a = &lines[i];
            let b = &lines[j];
            let count = intersect_polylines_into(a.points, b.points, x_bounds, &mut stack_hits);
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
