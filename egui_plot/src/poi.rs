//! Critical points (Points of Interest) detection for 2D plot polylines and reference lines.

pub(crate) mod interp;
pub(crate) mod intersect;
#[cfg(test)]
mod tests;

use egui::Id;

use crate::{
    SharedPoints,
    items::{PlotItem, PlotPoint},
    transform::{PlotBounds, PlotTransform},
};

pub use self::interp::{
    interpolate_threshold_crossings, interpolate_threshold_crossings_pairs, interpolate_y_at_x,
};

/// Maximum number of visible critical points returned per plot frame.
pub const MAX_CRITICAL_POINTS: usize = 256;

/// Classification of a detected 2D plot critical point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
        let raw = format!(
            "{}: ({:.4}, {:.4})",
            self.kind.label(),
            self.point[0],
            self.point[1]
        );
        raw.chars().take(128).collect()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum SeriesCoords<'a> {
    PlotPoints(&'a [PlotPoint]),
    Shared(&'a [[f64; 2]]),
}

impl interp::CoordSlice for SeriesCoords<'_> {
    #[inline]
    fn coord_len(&self) -> usize {
        match self {
            Self::PlotPoints(pts) => pts.len(),
            Self::Shared(pts) => pts.len(),
        }
    }

    #[inline]
    fn coord_at(&self, idx: usize) -> PlotPoint {
        match self {
            Self::PlotPoints(pts) => pts[idx],
            Self::Shared(pts) => PlotPoint::from(pts[idx]),
        }
    }
}

pub(crate) struct LineSeriesRef<'a> {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) points: SeriesCoords<'a>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RefLineEntry {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) value: f64,
}

impl RefLineEntry {
    #[inline]
    pub(crate) fn matches(&self, id: Id, name: &str, value: f64) -> bool {
        self.id == id && self.name == name && self.value.to_bits() == value.to_bits()
    }
}

/// Extract bounded roots, y-intercepts, local extrema, and pairwise intersections
/// across visible `Line`, `HLine`, and `VLine` items, capped at [`MAX_CRITICAL_POINTS`].
pub fn detect_critical_points(
    items: &[Box<dyn PlotItem + '_>],
    transform: &PlotTransform,
) -> Vec<CriticalPoint> {
    let bounds = *transform.bounds();
    if !bounds.is_finite() {
        return Vec::new();
    }
    let (lines, hlines, vlines) = collect_inspectable_items(items);
    detect_critical_points_from_refs(&lines, &hlines, &vlines, &bounds, transform)
}

pub(crate) fn detect_critical_points_from_shared(
    lines: &[(Id, String, SharedPoints)],
    hlines: &[RefLineEntry],
    vlines: &[RefLineEntry],
    transform: &PlotTransform,
) -> Vec<CriticalPoint> {
    let bounds = *transform.bounds();
    if !bounds.is_finite() {
        return Vec::new();
    }
    let refs: Vec<LineSeriesRef<'_>> = lines
        .iter()
        .map(|(id, name, pts)| LineSeriesRef {
            id: *id,
            name: name.clone(),
            points: SeriesCoords::Shared(pts.as_slice()),
        })
        .collect();
    detect_critical_points_from_refs(&refs, hlines, vlines, &bounds, transform)
}

fn detect_critical_points_from_refs(
    lines: &[LineSeriesRef<'_>],
    hlines: &[RefLineEntry],
    vlines: &[RefLineEntry],
    bounds: &PlotBounds,
    transform: &PlotTransform,
) -> Vec<CriticalPoint> {
    let mut raw = Vec::new();
    for line in lines {
        collect_single_line_pois(line, bounds, transform, &mut raw);
    }
    intersect::collect_reference_intersections(lines, hlines, vlines, bounds, transform, &mut raw);
    intersect::collect_pairwise_line_intersections(lines, bounds, transform, &mut raw);
    finalize_critical_points(raw, bounds)
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
        if let Some(pts) = item.line_shared_points() {
            if pts.len() >= 2 {
                lines.push(LineSeriesRef {
                    id: item.id(),
                    name: item.name().to_owned(),
                    points: SeriesCoords::Shared(pts.as_slice()),
                });
            }
        } else if let Some(pts) = item.line_points() {
            if pts.len() >= 2 {
                lines.push(LineSeriesRef {
                    id: item.id(),
                    name: item.name().to_owned(),
                    points: SeriesCoords::PlotPoints(pts),
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
    if point_y_in_bounds(0.0, bounds) && (!transform.log_axis().y) {
        for x_root in interp::interpolate_threshold_crossings_slice(&line.points, 0.0) {
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
    }
    if point_x_in_bounds(0.0, bounds) && (!transform.log_axis().x) {
        for y_int in interp::interpolate_y_at_x_slice(&line.points, 0.0) {
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
    }
    for (pt, is_max) in interp::detect_local_extrema(&line.points) {
        push_if_visible(
            out,
            CriticalPoint {
                kind: if is_max {
                    CriticalPointKind::LocalMax
                } else {
                    CriticalPointKind::LocalMin
                },
                point: pt,
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
