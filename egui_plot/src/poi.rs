//! Critical points (Points of Interest) detection for 2D plot polylines and reference lines.

mod intersect;
#[cfg(test)]
mod tests;

use egui::Id;

use crate::{
    items::{PlotItem, PlotPoint},
    transform::{PlotBounds, PlotTransform},
};

pub use self::intersect::MAX_PAIRWISE_INTERSECTION_SERIES;
pub use math_commons::poi::{
    LinePoiScan, LocalExtremum, closest_threshold_crossing, detect_local_extrema,
    detect_local_extrema_into, eval_segment_at_x, interpolate_threshold_crossings,
    interpolate_y_at_x, intersect_polylines, intersect_polylines_into, parabolic_vertex,
    peak_extremum, scan_line_pois, segment_intersection_2d, threshold_crossings_into, y_at_x_into,
};

/// Maximum number of visible critical points returned per plot frame.
pub const MAX_CRITICAL_POINTS: usize = 256;

/// Classification of a detected 2D plot critical point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum CriticalPointKind {
    /// Zero crossing where $y = 0$.
    Root,
    /// Axis crossing where $x = 0$.
    YIntercept,
    /// Interior local maximum turning point.
    LocalMax,
    /// Interior local minimum turning point.
    LocalMin,
    /// Pairwise crossing between two series or a series and a reference line.
    Intersection,
}

impl CriticalPointKind {
    /// Human-readable category label for tooltips, tables, and screen readers.
    #[inline]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Root => "Root",
            Self::YIntercept => "Y-Intercept",
            Self::LocalMax => "Local Max",
            Self::LocalMin => "Local Min",
            Self::Intersection => "Intersection",
        }
    }
}

/// A detected sub-segment critical point in plot data coordinates.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct CriticalPoint {
    pub kind: CriticalPointKind,
    pub point: [f64; 2],
    pub primary_series: String,
    pub secondary_series: Option<String>,
    pub item_id: Id,
}

impl CriticalPoint {
    /// Concise series label (`"A"` or `"A ∩ B"` for intersections).
    pub fn series_summary(&self) -> String {
        let primary = if self.primary_series.is_empty() {
            "Series"
        } else {
            self.primary_series.as_str()
        };
        match self.secondary_series.as_deref() {
            Some(sec) if !sec.is_empty() => format!("{primary} ∩ {sec}"),
            Some(_) => format!("{primary} ∩ Reference"),
            None => primary.to_owned(),
        }
    }

    /// Title shown in canvas hover/focus tooltips.
    pub fn tooltip_title(&self) -> String {
        format!("{} — {}", self.kind.label(), self.series_summary())
    }

    /// Persistent callout badge label (`<= 128` chars) when pinned on canvas.
    pub fn callout_label(&self) -> String {
        let x_str = crate::format::format_readout_value(self.point[0], 4, 4);
        let y_str = crate::format::format_readout_value(self.point[1], 4, 4);
        let raw = format!("{}: ({x_str}, {y_str})", self.kind.label());
        raw.chars().take(128).collect()
    }
}

pub(crate) struct LineSeriesRef<'a> {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) points: &'a [PlotPoint],
}

pub(crate) struct RefLineEntry {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) value: f64,
}

thread_local! {
    static DETECT_CRITICAL_POINTS_CALL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[doc(hidden)]
pub fn detect_critical_points_call_count() -> usize {
    DETECT_CRITICAL_POINTS_CALL_COUNT.with(std::cell::Cell::get)
}

#[doc(hidden)]
pub fn reset_detect_critical_points_call_count() {
    DETECT_CRITICAL_POINTS_CALL_COUNT.with(|c| c.set(0));
}

/// Extract bounded roots, y-intercepts, local extrema, and pairwise intersections
/// across visible `Line`, `HLine`, and `VLine` items, capped at [`MAX_CRITICAL_POINTS`].
pub fn detect_critical_points(
    items: &[Box<dyn PlotItem + '_>],
    transform: &PlotTransform,
) -> Vec<CriticalPoint> {
    DETECT_CRITICAL_POINTS_CALL_COUNT.with(|c| c.set(c.get() + 1));
    let bounds = *transform.bounds();
    if !bounds.is_finite() {
        return Vec::new();
    }
    let (lines, hlines, vlines) = collect_inspectable_items(items);
    let mut raw = Vec::new();

    for line in &lines {
        collect_single_line_pois(line, &bounds, transform, &mut raw);
    }
    intersect::collect_reference_intersections(
        &lines, &hlines, &vlines, &bounds, transform, &mut raw,
    );
    intersect::collect_pairwise_line_intersections(&lines, &bounds, transform, &mut raw);

    finalize_critical_points(raw, &bounds)
}

fn collect_inspectable_items<'a>(
    items: &'a [Box<dyn PlotItem + 'a>],
) -> (Vec<LineSeriesRef<'a>>, Vec<RefLineEntry>, Vec<RefLineEntry>) {
    let mut lines = Vec::new();
    let mut hlines = Vec::new();
    let mut vlines = Vec::new();
    for item in items {
        if !item.allow_hover() {
            continue;
        }
        if let Some(pts) = item.line_points() {
            if pts.len() >= 2 {
                lines.push(LineSeriesRef {
                    id: item.id(),
                    name: item.name().to_owned(),
                    points: pts,
                });
            }
        } else if let Some(y) = item.hline_y() {
            if y.is_finite() {
                hlines.push(RefLineEntry {
                    id: item.id(),
                    name: item.name().to_owned(),
                    value: y,
                });
            }
        } else if let Some(x) = item.vline_x()
            && x.is_finite()
        {
            vlines.push(RefLineEntry {
                id: item.id(),
                name: item.name().to_owned(),
                value: x,
            });
        }
    }
    (lines, hlines, vlines)
}

fn collect_single_line_pois(
    line: &LineSeriesRef<'_>,
    bounds: &PlotBounds,
    transform: &PlotTransform,
    out: &mut Vec<CriticalPoint>,
) {
    let want_roots = point_y_in_bounds(0.0, bounds) && (!transform.log_axis().y);
    let want_y_ints = point_x_in_bounds(0.0, bounds) && (!transform.log_axis().x);
    let LinePoiScan {
        roots,
        y_intercepts,
        extrema,
    } = scan_line_pois(line.points, want_roots, want_y_ints);
    for x_root in roots {
        push_if_visible(
            out,
            CriticalPoint {
                kind: CriticalPointKind::Root,
                point: [x_root, 0.0],
                primary_series: line.name.clone(),
                secondary_series: None,
                item_id: line.id,
            },
            bounds,
            transform,
        );
    }
    for y_int in y_intercepts {
        push_if_visible(
            out,
            CriticalPoint {
                kind: CriticalPointKind::YIntercept,
                point: [0.0, y_int],
                primary_series: line.name.clone(),
                secondary_series: None,
                item_id: line.id,
            },
            bounds,
            transform,
        );
    }
    for ext in extrema {
        push_if_visible(
            out,
            CriticalPoint {
                kind: if ext.is_max {
                    CriticalPointKind::LocalMax
                } else {
                    CriticalPointKind::LocalMin
                },
                point: ext.point,
                primary_series: line.name.clone(),
                secondary_series: None,
                item_id: line.id,
            },
            bounds,
            transform,
        );
    }
}

pub(crate) fn push_if_visible(
    out: &mut Vec<CriticalPoint>,
    poi: CriticalPoint,
    bounds: &PlotBounds,
    transform: &PlotTransform,
) {
    let [x, y] = poi.point;
    if !x.is_finite() || !y.is_finite() {
        return;
    }
    let plot_pt = PlotPoint::new(x, y);
    if !transform.is_point_valid_for_scale(&plot_pt) {
        return;
    }
    if point_x_in_bounds(x, bounds) && point_y_in_bounds(y, bounds) {
        out.push(poi);
    }
}

fn point_x_in_bounds(x: f64, bounds: &PlotBounds) -> bool {
    let tol = (bounds.width().abs() * 1e-9).max(1e-12);
    x >= bounds.min[0] - tol && x <= bounds.max[0] + tol
}

fn point_y_in_bounds(y: f64, bounds: &PlotBounds) -> bool {
    let tol = (bounds.height().abs() * 1e-9).max(1e-12);
    y >= bounds.min[1] - tol && y <= bounds.max[1] + tol
}

fn finalize_critical_points(
    mut raw: Vec<CriticalPoint>,
    bounds: &PlotBounds,
) -> Vec<CriticalPoint> {
    raw.sort_by(|a, b| {
        a.point[0]
            .total_cmp(&b.point[0])
            .then_with(|| a.kind.cmp(&b.kind))
            .then_with(|| a.point[1].total_cmp(&b.point[1]))
    });
    let eps_x = (bounds.width().abs() * 1e-6).max(1e-9);
    let eps_y = (bounds.height().abs() * 1e-6).max(1e-9);
    let mut deduped: Vec<CriticalPoint> = Vec::with_capacity(raw.len().min(MAX_CRITICAL_POINTS));
    for candidate in raw {
        let is_dup = deduped.iter().rev().take(8).any(|prev| {
            prev.kind == candidate.kind
                && prev.primary_series == candidate.primary_series
                && prev.secondary_series == candidate.secondary_series
                && (prev.point[0] - candidate.point[0]).abs() <= eps_x
                && (prev.point[1] - candidate.point[1]).abs() <= eps_y
        });
        if !is_dup {
            deduped.push(candidate);
            if deduped.len() >= MAX_CRITICAL_POINTS {
                break;
            }
        }
    }
    deduped
}
