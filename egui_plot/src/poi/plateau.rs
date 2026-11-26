//! Fast 32/8-wide zero-copy plateau compaction for dense constant-value polylines.

use crate::items::PlotPoint;

use super::SeriesCoords;

const X_GAP: f64 = 1e-12_f64.next_up();

macro_rules! ok_pt {
    ($c:expr, $k:literal, $y_plat:expr, $x_lo:expr, $x_hi:expr, |$p:ident| ($px:expr, $py:expr)) => {{
        let $p = &$c[$k];
        ($py == $y_plat) & ($px >= $x_lo) & ($px <= $x_hi)
    }};
}

macro_rules! same_plateau8 {
    ($c:expr, $y_plat:expr, $x_lo:expr, $x_hi:expr, |$p:ident| ($px:expr, $py:expr)) => {{
        let c8 = $c;
        ok_pt!(c8, 0, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c8, 1, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c8, 2, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c8, 3, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c8, 4, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c8, 5, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c8, 6, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c8, 7, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
    }};
}

macro_rules! same_plateau32 {
    ($c:expr, $y_plat:expr, $x_lo:expr, $x_hi:expr, |$p:ident| ($px:expr, $py:expr)) => {{
        let c32 = $c;
        ok_pt!(c32, 0, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 1, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 2, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 3, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 4, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 5, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 6, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 7, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 8, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 9, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 10, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 11, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 12, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 13, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 14, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 15, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 16, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 17, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 18, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 19, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 20, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 21, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 22, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 23, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 24, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 25, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 26, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 27, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 28, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 29, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 30, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
            & ok_pt!(c32, 31, $y_plat, $x_lo, $x_hi, |$p| ($px, $py))
    }};
}

macro_rules! define_compact_fn {
    ($fn_name:ident, $elem:ty, |$p:ident| ($px:expr, $py:expr)) => {
        fn $fn_name(pts: &[$elem], want_y_ints: bool) -> Option<Vec<[f64; 2]>> {
            let n = pts.len();
            if n < 32 {
                return None;
            }
            let mut out: Option<Vec<[f64; 2]>> = None;
            let mut copied_up_to = 0usize;
            let mut i = 2usize;
            while i + 8 <= n {
                let (x0, y_plat) = {
                    let $p = &pts[i - 2];
                    ($px, $py)
                };
                let (x1, y1) = {
                    let $p = &pts[i - 1];
                    ($px, $py)
                };
                let (x_lo, x_hi) = if !want_y_ints {
                    (-f64::MAX, f64::MAX)
                } else if x0 >= X_GAP && x1 >= X_GAP {
                    (X_GAP, f64::MAX)
                } else if x0 <= -X_GAP && x1 <= -X_GAP {
                    (-f64::MAX, -X_GAP)
                } else {
                    i += 8;
                    continue;
                };
                let pre_ok = (y1 == y_plat)
                    && y_plat.is_finite()
                    && (x0 >= x_lo)
                    && (x0 <= x_hi)
                    && (x1 >= x_lo)
                    && (x1 <= x_hi);
                if pre_ok {
                    let (first8, _) = pts[i..i + 8].as_chunks::<8>();
                    if same_plateau8!(&first8[0], y_plat, x_lo, x_hi, |$p| ($px, $py)) {
                        let mut end = i + 8;
                        let (chunks32, _) = pts[end..].as_chunks::<32>();
                        let mut c32_idx = 0usize;
                        while c32_idx < chunks32.len()
                            && same_plateau32!(&chunks32[c32_idx], y_plat, x_lo, x_hi, |$p| (
                                $px, $py
                            ))
                        {
                            c32_idx += 1;
                        }
                        end += c32_idx << 5;
                        let (chunks8, _) = pts[end..].as_chunks::<8>();
                        let mut c8_idx = 0usize;
                        while c8_idx < chunks8.len()
                            && same_plateau8!(&chunks8[c8_idx], y_plat, x_lo, x_hi, |$p| ($px, $py))
                        {
                            c8_idx += 1;
                        }
                        end += c8_idx << 3;
                        let buf = out.get_or_insert_with(|| Vec::with_capacity(64));
                        for $p in &pts[copied_up_to..i] {
                            buf.push([$px, $py]);
                        }
                        let $p = &pts[end - 2];
                        buf.push([$px, $py]);
                        let $p = &pts[end - 1];
                        buf.push([$px, $py]);
                        copied_up_to = end;
                        i = end + 2;
                        continue;
                    }
                }
                i += 8;
            }
            if let Some(ref mut buf) = out {
                for $p in &pts[copied_up_to..] {
                    buf.push([$px, $py]);
                }
            }
            out
        }
    };
}

define_compact_fn!(compact_plot_points, PlotPoint, |p| (p.x, p.y));
define_compact_fn!(compact_shared, [f64; 2], |p| (p[0], p[1]));

pub(super) fn compact_long_plateaus(
    coords: SeriesCoords<'_>,
    want_y_ints: bool,
) -> Option<Vec<[f64; 2]>> {
    match coords {
        SeriesCoords::PlotPoints(pts) => compact_plot_points(pts, want_y_ints),
        SeriesCoords::Shared(pts) => compact_shared(pts, want_y_ints),
    }
}
