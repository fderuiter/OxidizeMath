//! Zero-allocation-fast-path 8-point chunked compaction for flat `PlotPoint` and `[f64; 2]` plateaus.

use std::borrow::Cow;

use crate::items::PlotPoint;

#[inline(always)]
fn x_bounds_for_plateau(x0: f64, want_y_ints: bool) -> Option<(f64, f64)> {
    if !(f64::MIN..=f64::MAX).contains(&x0) {
        return None;
    }
    if !want_y_ints {
        Some((f64::MIN, f64::MAX))
    } else if x0 > 1e-12 {
        Some(((1e-12_f64).next_up(), f64::MAX))
    } else if x0 < -1e-12 {
        Some((f64::MIN, (-1e-12_f64).next_down()))
    } else {
        None
    }
}

macro_rules! pt_x {
    (PlotPoint, $p:expr) => {
        $p.x
    };
    (Arr2, $p:expr) => {
        $p[0]
    };
}

macro_rules! pt_y {
    (PlotPoint, $p:expr) => {
        $p.y
    };
    (Arr2, $p:expr) => {
        $p[1]
    };
}

macro_rules! pt_ok {
    ($kind:ident, $p:expr, $yv:expr, $xmin:expr, $xmax:expr) => {
        (pt_y!($kind, $p) == $yv) & (pt_x!($kind, $p) >= $xmin) & (pt_x!($kind, $p) <= $xmax)
    };
}

macro_rules! impl_plateau_compactor {
    ($kind:ident, $ty:ty, $check_fn:ident, $has_fn:ident, $measure_fn:ident, $compact_fn:ident) => {
        #[inline(always)]
        fn $check_fn(c: &[$ty; 8], y_val: f64, x_min: f64, x_max: f64) -> bool {
            let &[a0, a1, a2, a3, a4, a5, a6, a7] = c;
            pt_ok!($kind, a0, y_val, x_min, x_max)
                & pt_ok!($kind, a1, y_val, x_min, x_max)
                & pt_ok!($kind, a2, y_val, x_min, x_max)
                & pt_ok!($kind, a3, y_val, x_min, x_max)
                & pt_ok!($kind, a4, y_val, x_min, x_max)
                & pt_ok!($kind, a5, y_val, x_min, x_max)
                & pt_ok!($kind, a6, y_val, x_min, x_max)
                & pt_ok!($kind, a7, y_val, x_min, x_max)
        }

        fn $has_fn(pts: &[$ty], want_y_ints: bool) -> bool {
            if pts.len() < 16 {
                return false;
            }
            let (chunks, _) = pts.as_chunks::<8>();
            for chunk in chunks {
                let y0 = pt_y!($kind, chunk[0]);
                if y0 != pt_y!($kind, chunk[7]) || !y0.is_finite() {
                    continue;
                }
                if let Some((x_min, x_max)) =
                    x_bounds_for_plateau(pt_x!($kind, chunk[0]), want_y_ints)
                    && $check_fn(chunk, y0, x_min, x_max)
                {
                    return true;
                }
            }
            false
        }

        fn $measure_fn(pts: &[$ty], y_val: f64, x_min: f64, x_max: f64) -> usize {
            let (chunks, _) = pts.as_chunks::<8>();
            let mut run_len = 0usize;
            for chunk in chunks {
                if !$check_fn(chunk, y_val, x_min, x_max) {
                    break;
                }
                run_len += 8;
            }
            let tail_end = (run_len + 8).min(pts.len());
            for p in &pts[run_len..tail_end] {
                if !pt_ok!($kind, *p, y_val, x_min, x_max) {
                    break;
                }
                run_len += 1;
            }
            run_len
        }

        pub(super) fn $compact_fn(pts: &[$ty], want_y_ints: bool) -> Cow<'_, [$ty]> {
            if !$has_fn(pts, want_y_ints) {
                return Cow::Borrowed(pts);
            }
            let n = pts.len();
            let mut out = Vec::with_capacity(64);
            let mut i = 0usize;
            while i < n {
                let yi = pt_y!($kind, pts[i]);
                if n - i >= 8
                    && yi == pt_y!($kind, pts[i + 7])
                    && yi.is_finite()
                    && let Some((x_min, x_max)) =
                        x_bounds_for_plateau(pt_x!($kind, pts[i]), want_y_ints)
                {
                    let run_len = $measure_fn(&pts[i..], yi, x_min, x_max);
                    if run_len >= 8 {
                        out.push(pts[i]);
                        out.push(pts[i + 1]);
                        out.push(pts[i + run_len - 2]);
                        out.push(pts[i + run_len - 1]);
                        i += run_len;
                        continue;
                    }
                }
                out.push(pts[i]);
                i += 1;
            }
            Cow::Owned(out)
        }
    };
}

impl_plateau_compactor!(
    PlotPoint,
    PlotPoint,
    is_valid_plot_chunk,
    has_any_plot_chunk,
    measure_plot_run,
    compact_plot_points
);

impl_plateau_compactor!(
    Arr2,
    [f64; 2],
    is_valid_shared_chunk,
    has_any_shared_chunk,
    measure_shared_run,
    compact_shared_points
);
