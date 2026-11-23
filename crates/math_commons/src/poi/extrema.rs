//! Sub-segment parabolic vertex, linear-time plateau local extrema, and fused line scanning.
// [cite:math_commons]

use super::LocalExtremum;
use super::interp::{is_finite_pair, push_unique_vec, segment_threshold_hit, segment_vertical_hit};
use crate::series::PointSeries;

/// Output of a fused single-pass critical-point scan over a single 2D series.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinePoiScan {
    /// Interpolated horizontal coordinates where $y = 0$.
    pub roots: Vec<f64>,
    /// Interpolated vertical coordinates where $x = 0$.
    pub y_intercepts: Vec<f64>,
    /// Interior local maxima and minima (including flat plateau midpoints).
    pub extrema: Vec<LocalExtremum>,
}

#[derive(Clone, Copy)]
struct PlateauTracker {
    mode: u8,
    prev: [f64; 2],
    start: [f64; 2],
    second: [f64; 2],
    last: [f64; 2],
}

impl PlateauTracker {
    #[inline(always)]
    const fn new() -> Self {
        Self {
            mode: 0,
            prev: [0.0; 2],
            start: [0.0; 2],
            second: [0.0; 2],
            last: [0.0; 2],
        }
    }

    #[inline(always)]
    fn step_vec(&mut self, p0: [f64; 2], p1: [f64; 2], out: &mut Vec<LocalExtremum>) {
        if self.mode != 0 && p1[1] == self.start[1] {
            if self.mode == 2 {
                self.last = p1;
            } else if (self.prev[0] < self.start[0] && self.start[0] < p1[0])
                || (self.prev[0] > self.start[0] && self.start[0] > p1[0])
            {
                self.second = p1;
                self.last = p1;
                self.mode = 2;
            } else {
                self.mode = 0;
            }
            return;
        }
        if self.mode != 0 {
            self.finish_into_vec(p1, out);
        }
        if p1[1] != p0[1] {
            self.mode = 1;
            self.prev = p0;
            self.start = p1;
            self.last = p1;
        } else {
            self.mode = 0;
        }
    }

    #[inline(always)]
    fn step_slice(&mut self, p0: [f64; 2], p1: [f64; 2], out: &mut [LocalExtremum], n: &mut usize) {
        if self.mode != 0 && p1[1] == self.start[1] {
            if self.mode == 2 {
                self.last = p1;
            } else if (self.prev[0] < self.start[0] && self.start[0] < p1[0])
                || (self.prev[0] > self.start[0] && self.start[0] > p1[0])
            {
                self.second = p1;
                self.last = p1;
                self.mode = 2;
            } else {
                self.mode = 0;
            }
            return;
        }
        if self.mode != 0
            && let Some(ext) = self.finish(p1)
        {
            out[*n] = ext;
            *n += 1;
        }
        if p1[1] != p0[1] {
            self.mode = 1;
            self.prev = p0;
            self.start = p1;
            self.last = p1;
        } else {
            self.mode = 0;
        }
    }

    #[cold]
    fn finish_into_vec(&self, after: [f64; 2], out: &mut Vec<LocalExtremum>) {
        if let Some(ext) = self.finish(after) {
            out.push(ext);
        }
    }

    #[cold]
    fn finish(&self, after: [f64; 2]) -> Option<LocalExtremum> {
        let p2 = if self.mode == 1 { after } else { self.second };
        let x_ok = (self.prev[0] < self.start[0] && self.start[0] < p2[0])
            || (self.prev[0] > self.start[0] && self.start[0] > p2[0]);
        if !x_ok {
            return None;
        }
        let is_max =
            self.start[1] > self.prev[1] && self.start[1] > after[1] && self.start[1] >= p2[1];
        let is_min =
            self.start[1] < self.prev[1] && self.start[1] < after[1] && self.start[1] <= p2[1];
        if !is_max && !is_min {
            return None;
        }
        let point = if self.mode == 1 {
            parabolic_vertex_raw(self.prev, self.start, after)
        } else {
            [0.5 * (self.start[0] + self.last[0]), self.start[1]]
        };
        Some(LocalExtremum { point, is_max })
    }
}

#[inline(always)]
fn fast_forward_plateau<P: PointSeries + ?Sized>(
    pts: &P,
    n: usize,
    mut i: usize,
    want_y_intercepts: bool,
    p0: &mut [f64; 2],
    tracker: &mut PlateauTracker,
) -> usize {
    let flat_y = p0[1];
    let (x_min, x_max) = if !want_y_intercepts {
        (f64::MIN, f64::MAX)
    } else if p0[0] > 1e-12 {
        ((1e-12_f64).next_up(), f64::MAX)
    } else if p0[0] < -1e-12 {
        (f64::MIN, (-1e-12_f64).next_down())
    } else {
        return i;
    };
    let start_i = i;
    while i < n {
        let p = pts.point_at(i);
        if p[1] != flat_y || p[0] < x_min || p[0] > x_max {
            break;
        }
        i += 1;
    }
    if i > start_i {
        let end_p = pts.point_at(i - 1);
        if tracker.mode == 2 {
            tracker.last = end_p;
        }
        *p0 = end_p;
    }
    i
}

/// Fused $O(N)$ single-pass scan extracting $y = 0$ roots, $x = 0$ intercepts, and local extrema.
pub fn scan_line_pois<P: PointSeries + ?Sized>(
    pts: &P,
    want_roots: bool,
    want_y_intercepts: bool,
) -> LinePoiScan {
    let mut out = LinePoiScan::default();
    let n = pts.len();
    if n < 2 {
        return out;
    }
    let mut p0 = pts.point_at(0);
    let mut p0_fin = (p0[0] * 0.0 + p0[1] * 0.0) == 0.0;
    let mut tracker = PlateauTracker::new();
    let mut i = 1usize;
    while i < n {
        if p0_fin && (tracker.mode == 0 || tracker.mode == 2) {
            i = fast_forward_plateau(pts, n, i, want_y_intercepts, &mut p0, &mut tracker);
            if i >= n {
                break;
            }
        }
        let idx = i - 1;
        let p1 = pts.point_at(i);
        i += 1;
        let p1_fin = (p1[0] * 0.0 + p1[1] * 0.0) == 0.0;
        if p0_fin && p1_fin {
            if want_roots
                && p0[1] != p1[1]
                && let Some(xr) = segment_threshold_hit(pts, idx, n, p0, p1, 0.0)
            {
                push_unique_vec(&mut out.roots, xr);
            }
            if want_y_intercepts
                && p0[0] != p1[0]
                && !(p0[0] > 1e-12 && p1[0] > 1e-12)
                && !(p0[0] < -1e-12 && p1[0] < -1e-12)
                && let Some(yi) = segment_vertical_hit(idx, n, p0, p1, 0.0)
            {
                push_unique_vec(&mut out.y_intercepts, yi);
            }
            tracker.step_vec(p0, p1, &mut out.extrema);
        } else {
            tracker.mode = 0;
        }
        p0 = p1;
        p0_fin = p1_fin;
    }
    out
}

/// Writes interior local extrema refined via parabolic/plateau interpolation into `out`.
#[verified_engine::verified]
pub fn detect_local_extrema_into<P: PointSeries + ?Sized>(
    pts: &P,
    out: &mut [LocalExtremum],
) -> usize {
    detect_local_extrema_into_raw(pts, out)
}

fn detect_local_extrema_into_raw<P: PointSeries + ?Sized>(
    pts: &P,
    out: &mut [LocalExtremum],
) -> usize {
    let n = pts.len();
    if n < 3 || out.is_empty() {
        return 0;
    }
    let mut written = 0;
    let mut p0 = pts.point_at(0);
    let mut p0_fin = is_finite_pair(p0);
    let mut tracker = PlateauTracker::new();
    let mut i = 1usize;
    while i < n {
        let p1 = pts.point_at(i);
        i += 1;
        let p1_fin = is_finite_pair(p1);
        if p0_fin && p1_fin {
            tracker.step_slice(p0, p1, out, &mut written);
            if written == out.len() {
                break;
            }
        } else {
            tracker.mode = 0;
        }
        p0 = p1;
        p0_fin = p1_fin;
    }
    written
}

/// Convenience wrapper collecting interior local extrema into a `Vec<LocalExtremum>`.
pub fn detect_local_extrema<P: PointSeries + ?Sized>(pts: &P) -> Vec<LocalExtremum> {
    scan_line_pois(pts, false, false).extrema
}

/// Finds the global maximum in `pts`, refining interior peaks via [`parabolic_vertex`].
///
/// Executes with zero heap allocations on any [`PointSeries`].
#[verified_engine::verified]
pub fn peak_extremum<P: PointSeries + ?Sized>(pts: &P) -> Option<[f64; 2]> {
    peak_extremum_raw(pts)
}

fn peak_extremum_raw<P: PointSeries + ?Sized>(pts: &P) -> Option<[f64; 2]> {
    let n = pts.len();
    if n == 0 {
        return None;
    }
    let mut best_idx: Option<usize> = None;
    let mut best_pt = [0.0_f64; 2];
    for i in 0..n {
        let p = pts.point_at(i);
        if is_finite_pair(p) && best_idx.is_none_or(|_| p[1] > best_pt[1]) {
            best_idx = Some(i);
            best_pt = p;
        }
    }
    let idx = best_idx?;
    if idx > 0 && idx + 1 < n {
        let p0 = pts.point_at(idx - 1);
        let p2 = pts.point_at(idx + 1);
        if is_finite_pair(p0)
            && is_finite_pair(p2)
            && p0[0] < best_pt[0]
            && best_pt[0] < p2[0]
            && (best_pt[1] > p0[1] || best_pt[1] > p2[1])
        {
            let [vx, vy] = parabolic_vertex_raw(p0, best_pt, p2);
            return Some([vx, vy.max(best_pt[1])]);
        }
    }
    Some(best_pt)
}

/// Fits a 3-point Lagrange parabola through `(p0, p1, p2)` and returns its clamped vertex `[x, y]`.
#[verified_engine::verified]
pub fn parabolic_vertex(p0: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> [f64; 2] {
    parabolic_vertex_raw(p0, p1, p2)
}

pub(super) fn parabolic_vertex_raw(p0: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> [f64; 2] {
    let (x0, y0) = (p0[0], p0[1]);
    let (x1, y1) = (p1[0], p1[1]);
    let (x2, y2) = (p2[0], p2[1]);
    let denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
    if denom.abs() <= 1e-15 || x2 <= x0 {
        return p1;
    }
    let a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
    let b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
    if a.abs() <= 1e-15 {
        return p1;
    }
    let vx = (-b / (2.0 * a)).clamp(x0, x2);
    let d01 = x0 - x1;
    let d02 = x0 - x2;
    let d12 = x1 - x2;
    let l0 = ((vx - x1) * (vx - x2)) / (d01 * d02);
    let l1 = ((vx - x0) * (vx - x2)) / (-d01 * d12);
    let l2 = ((vx - x0) * (vx - x1)) / (d02 * d12);
    let vy = y0 * l0 + y1 * l1 + y2 * l2;
    if vx.is_finite() && vy.is_finite() {
        [vx, vy]
    } else {
        p1
    }
}
