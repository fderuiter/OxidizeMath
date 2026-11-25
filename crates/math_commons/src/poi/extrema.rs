//! Sub-segment parabolic vertex, linear-time plateau local extrema, and fused line scanning.
// [cite:math_commons]

use super::LocalExtremum;
use super::interp::is_finite_pair;
#[cfg(feature = "alloc")]
use super::interp::{push_unique_vec, segment_threshold_hit, segment_vertical_hit};
use crate::series::PointSeries;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use math_scalars::RealField;

/// Output of a fused single-pass critical-point scan over a single 2D series.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinePoiScan<T = f64> {
    /// Interpolated horizontal coordinates where $y = 0$.
    pub roots: Vec<T>,
    /// Interpolated vertical coordinates where $x = 0$.
    pub y_intercepts: Vec<T>,
    /// Interior local maxima and minima (including flat plateau midpoints).
    pub extrema: Vec<LocalExtremum<T>>,
}

#[derive(Clone, Copy)]
struct PlateauTracker<T> {
    mode: u8,
    prev: [T; 2],
    start: [T; 2],
    second: [T; 2],
    last: [T; 2],
}

impl<T: RealField> PlateauTracker<T> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            mode: 0,
            prev: [T::ZERO; 2],
            start: [T::ZERO; 2],
            second: [T::ZERO; 2],
            last: [T::ZERO; 2],
        }
    }

    #[cfg(feature = "alloc")]
    #[inline(always)]
    fn step_vec(&mut self, p0: [T; 2], p1: [T; 2], out: &mut Vec<LocalExtremum<T>>) {
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
    fn step_slice(&mut self, p0: [T; 2], p1: [T; 2], out: &mut [LocalExtremum<T>], n: &mut usize) {
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

    #[cfg(feature = "alloc")]
    #[cold]
    fn finish_into_vec(&self, after: [T; 2], out: &mut Vec<LocalExtremum<T>>) {
        if let Some(ext) = self.finish(after) {
            out.push(ext);
        }
    }

    #[cold]
    fn finish(&self, after: [T; 2]) -> Option<LocalExtremum<T>> {
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
            [T::HALF * (self.start[0] + self.last[0]), self.start[1]]
        };
        Some(LocalExtremum { point, is_max })
    }
}

#[cfg(feature = "alloc")]
#[inline(always)]
fn fast_forward_plateau<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    n: usize,
    i: usize,
    want_y_intercepts: bool,
    p0: &mut [T; 2],
    tracker: &mut PlateauTracker<T>,
) -> usize {
    let flat_y = p0[1];
    let tol = T::from_f64(1e-12);
    let strict_tol = tol * (T::ONE + T::DEFAULT_TOL_SCALE * T::EPSILON);
    let (x_min, x_max) = if !want_y_intercepts {
        (T::MIN_VALUE, T::MAX_VALUE)
    } else if p0[0] > tol {
        (strict_tol, T::MAX_VALUE)
    } else if p0[0] < -tol {
        (T::MIN_VALUE, -strict_tol)
    } else {
        return i;
    };
    let next_i = pts.advance_plateau_run(i, flat_y, x_min, x_max).min(n);
    if next_i > i {
        let end_p = pts.point_at(next_i - 1);
        if tracker.mode == 2 {
            tracker.last = end_p;
        }
        *p0 = end_p;
    }
    next_i
}

/// Fused $O(N)$ single-pass scan extracting $y = 0$ roots, $x = 0$ intercepts, and local extrema.
#[cfg(feature = "alloc")]
pub fn scan_line_pois<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    want_roots: bool,
    want_y_intercepts: bool,
) -> LinePoiScan<T> {
    let mut out = LinePoiScan::default();
    let n = pts.len();
    if n < 2 {
        return out;
    }
    let tol = T::from_f64(1e-12);
    let mut p0 = pts.point_at(0);
    let mut p0_fin = is_finite_pair(p0);
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
        let p1_fin = is_finite_pair(p1);
        if p0_fin && p1_fin {
            if want_roots
                && p0[1] != p1[1]
                && let Some(xr) = segment_threshold_hit(pts, idx, n, p0, p1, T::ZERO)
            {
                push_unique_vec(&mut out.roots, xr);
            }
            if want_y_intercepts
                && p0[0] != p1[0]
                && !(p0[0] > tol && p1[0] > tol)
                && !(p0[0] < -tol && p1[0] < -tol)
                && let Some(yi) = segment_vertical_hit(idx, n, p0, p1, T::ZERO)
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
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn detect_local_extrema_into<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    out: &mut [LocalExtremum<T>],
) -> usize {
    detect_local_extrema_into_raw(pts, out)
}

fn detect_local_extrema_into_raw<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
    out: &mut [LocalExtremum<T>],
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

/// Convenience wrapper collecting interior local extrema into a `Vec<LocalExtremum<T>>`.
#[cfg(feature = "alloc")]
pub fn detect_local_extrema<T: RealField, P: PointSeries<T> + ?Sized>(
    pts: &P,
) -> Vec<LocalExtremum<T>> {
    scan_line_pois(pts, false, false).extrema
}

/// Finds the global maximum in `pts`, refining interior peaks via [`parabolic_vertex`].
///
/// Executes with zero heap allocations on any [`PointSeries`].
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn peak_extremum<T: RealField, P: PointSeries<T> + ?Sized>(pts: &P) -> Option<[T; 2]> {
    peak_extremum_raw(pts)
}

fn peak_extremum_raw<T: RealField, P: PointSeries<T> + ?Sized>(pts: &P) -> Option<[T; 2]> {
    let n = pts.len();
    if n == 0 {
        return None;
    }
    let mut best_idx: Option<usize> = None;
    let mut best_pt = [T::ZERO; 2];
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
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn parabolic_vertex<T: RealField>(p0: [T; 2], p1: [T; 2], p2: [T; 2]) -> [T; 2] {
    parabolic_vertex_raw(p0, p1, p2)
}

pub(super) fn parabolic_vertex_raw<T: RealField>(p0: [T; 2], p1: [T; 2], p2: [T; 2]) -> [T; 2] {
    let (x0, y0) = (p0[0], p0[1]);
    let (x1, y1) = (p1[0], p1[1]);
    let (x2, y2) = (p2[0], p2[1]);
    let denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
    let tiny = T::from_f64(1e-15).max(T::EPSILON * T::from_f64(4.0));
    if denom.abs() <= tiny || x2 <= x0 {
        return p1;
    }
    let a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
    let b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
    if a.abs() <= tiny {
        return p1;
    }
    let vx = (-b / (T::TWO * a)).clamp(x0, x2);
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
