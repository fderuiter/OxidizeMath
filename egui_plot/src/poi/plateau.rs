//! Fast 8-wide plateau compaction for dense constant-value polylines.

use crate::items::PlotPoint;

use super::SeriesCoords;

trait PlateauSample: Copy {
    fn xy(self) -> [f64; 2];
    fn chunk8_matches(slice: &[Self], y_plat: f64, x_min: f64, x_max: f64) -> bool;
}

#[inline(always)]
fn sample_matches(x: f64, y: f64, y_plat: f64, x_min: f64, x_max: f64) -> bool {
    (y == y_plat) & (x > x_min) & (x < x_max)
}

impl PlateauSample for PlotPoint {
    #[inline(always)]
    fn xy(self) -> [f64; 2] {
        [self.x, self.y]
    }

    #[inline(always)]
    fn chunk8_matches(slice: &[Self], y_plat: f64, x_min: f64, x_max: f64) -> bool {
        let [p0, p1, p2, p3, p4, p5, p6, p7, ..] = slice else {
            return false;
        };
        sample_matches(p0.x, p0.y, y_plat, x_min, x_max)
            & sample_matches(p1.x, p1.y, y_plat, x_min, x_max)
            & sample_matches(p2.x, p2.y, y_plat, x_min, x_max)
            & sample_matches(p3.x, p3.y, y_plat, x_min, x_max)
            & sample_matches(p4.x, p4.y, y_plat, x_min, x_max)
            & sample_matches(p5.x, p5.y, y_plat, x_min, x_max)
            & sample_matches(p6.x, p6.y, y_plat, x_min, x_max)
            & sample_matches(p7.x, p7.y, y_plat, x_min, x_max)
    }
}

impl PlateauSample for [f64; 2] {
    #[inline(always)]
    fn xy(self) -> [f64; 2] {
        self
    }

    #[inline(always)]
    fn chunk8_matches(slice: &[Self], y_plat: f64, x_min: f64, x_max: f64) -> bool {
        let [p0, p1, p2, p3, p4, p5, p6, p7, ..] = slice else {
            return false;
        };
        sample_matches(p0[0], p0[1], y_plat, x_min, x_max)
            & sample_matches(p1[0], p1[1], y_plat, x_min, x_max)
            & sample_matches(p2[0], p2[1], y_plat, x_min, x_max)
            & sample_matches(p3[0], p3[1], y_plat, x_min, x_max)
            & sample_matches(p4[0], p4[1], y_plat, x_min, x_max)
            & sample_matches(p5[0], p5[1], y_plat, x_min, x_max)
            & sample_matches(p6[0], p6[1], y_plat, x_min, x_max)
            & sample_matches(p7[0], p7[1], y_plat, x_min, x_max)
    }
}

#[inline(always)]
fn resolve_x_bounds(x0: f64, x1: f64, want_y_ints: bool) -> Option<(f64, f64)> {
    if !want_y_ints {
        Some((f64::NEG_INFINITY, f64::INFINITY))
    } else if x0 > 1e-12 && x1 > 1e-12 {
        Some((1e-12, f64::INFINITY))
    } else if x0 < -1e-12 && x1 < -1e-12 {
        Some((f64::NEG_INFINITY, -1e-12))
    } else {
        None
    }
}

fn compact_slice<T: PlateauSample>(pts: &[T], want_y_ints: bool) -> Option<Vec<[f64; 2]>> {
    let len = pts.len();
    if len < 32 {
        return None;
    }
    let mut out: Option<Vec<[f64; 2]>> = None;
    let mut copied_up_to = 0usize;
    let mut i = 2usize;
    while i + 8 <= len {
        let [x0, y_plat] = pts[i - 2].xy();
        let [x1, y1] = pts[i - 1].xy();
        let pre_ok = y1 == y_plat
            && x0 > f64::NEG_INFINITY
            && x0 < f64::INFINITY
            && x1 > f64::NEG_INFINITY
            && x1 < f64::INFINITY
            && y_plat > f64::NEG_INFINITY
            && y_plat < f64::INFINITY;
        let Some((x_min, x_max)) = resolve_x_bounds(x0, x1, want_y_ints) else {
            i += 8;
            continue;
        };
        if pre_ok && T::chunk8_matches(&pts[i..], y_plat, x_min, x_max) {
            let mut end = i + 8;
            while end + 8 <= len && T::chunk8_matches(&pts[end..], y_plat, x_min, x_max) {
                end += 8;
            }
            let buf = out.get_or_insert_with(|| Vec::with_capacity(64));
            buf.extend(pts[copied_up_to..i].iter().copied().map(T::xy));
            buf.push(pts[end - 2].xy());
            buf.push(pts[end - 1].xy());
            copied_up_to = end;
            i = end + 2;
        } else {
            i += 8;
        }
    }
    if let Some(ref mut buf) = out {
        buf.extend(pts[copied_up_to..].iter().copied().map(T::xy));
    }
    out
}

pub(super) fn compact_long_plateaus(
    coords: SeriesCoords<'_>,
    want_y_ints: bool,
) -> Option<Vec<[f64; 2]>> {
    match coords {
        SeriesCoords::PlotPoints(pts) => compact_slice(pts, want_y_ints),
        SeriesCoords::Shared(pts) => compact_slice(pts, want_y_ints),
    }
}
