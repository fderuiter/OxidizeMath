//! Min-max envelope polyline decimation for interactive high-step 1D trajectory display.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::PlotPoint;
use std::sync::Arc;

/// Maximum number of vertices emitted per display polyline (`4,096`) when decimating dense trajectories.
pub const MAX_DISPLAY_POLYLINE_POINTS: usize = 4_096;

fn decimate_indexed<F>(len: usize, max_points: usize, point_at: F) -> Arc<Vec<PlotPoint>>
where
    F: Fn(usize) -> PlotPoint,
{
    let cap = max_points.max(4);
    if len <= cap {
        return Arc::new((0..len).map(point_at).collect());
    }

    let bucket_count = (cap - 2) / 2;
    let interior_len = len - 2;
    let mut out = Vec::with_capacity(cap);
    out.push(point_at(0));

    for b in 0..bucket_count {
        let start = 1 + (b * interior_len) / bucket_count;
        let end = 1 + ((b + 1) * interior_len) / bucket_count;
        if start >= end {
            continue;
        }
        append_bucket_extrema(&mut out, start, end, &point_at);
    }

    out.push(point_at(len - 1));
    Arc::new(out)
}

fn append_bucket_extrema<F>(out: &mut Vec<PlotPoint>, start: usize, end: usize, point_at: &F)
where
    F: Fn(usize) -> PlotPoint,
{
    let first = point_at(start);
    let mut min_idx = start;
    let mut max_idx = start;
    let mut min_pt = first;
    let mut max_pt = first;

    for idx in (start + 1)..end {
        let pt = point_at(idx);
        if pt.y < min_pt.y {
            min_idx = idx;
            min_pt = pt;
        }
        if pt.y > max_pt.y {
            max_idx = idx;
            max_pt = pt;
        }
    }

    if min_idx == max_idx {
        out.push(min_pt);
    } else if min_idx < max_idx {
        out.push(min_pt);
        out.push(max_pt);
    } else {
        out.push(max_pt);
        out.push(min_pt);
    }
}

/// Decimates `PlotPoint` samples to at most `max_points` vertices using a min-max bucket envelope.
#[must_use]
pub fn decimate_plot_points_minmax(points: &[PlotPoint], max_points: usize) -> Arc<Vec<PlotPoint>> {
    decimate_indexed(points.len(), max_points, |i| points[i])
}

/// Decimates `[x, y]` coordinate pairs to at most `max_points` vertices using a min-max bucket envelope.
#[must_use]
pub fn decimate_pairs_minmax(pairs: &[[f64; 2]], max_points: usize) -> Arc<Vec<PlotPoint>> {
    decimate_indexed(pairs.len(), max_points, |i| {
        let [x, y] = pairs[i];
        PlotPoint::new(x, y)
    })
}

/// Decimates parallel `(xs, ys)` slices to at most `max_points` vertices using a min-max bucket envelope.
#[must_use]
pub fn decimate_xy_series_minmax(xs: &[f64], ys: &[f64], max_points: usize) -> Arc<Vec<PlotPoint>> {
    let len = xs.len().min(ys.len());
    decimate_indexed(len, max_points, |i| PlotPoint::new(xs[i], ys[i]))
}
