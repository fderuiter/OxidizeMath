//! Fast 32/8-wide plateau compaction for dense constant-value polylines.

use crate::items::PlotPoint;

use super::SeriesCoords;

macro_rules! pt_ok {
    ($x:expr, $y:expr, $yp:expr, $xl:expr, $xh:expr) => {
        ($y == $yp) & ($x > $xl) & ($x < $xh)
    };
}

macro_rules! chunk8_pairs_ok {
    ($c:expr, $yp:ident, $xl:ident, $xh:ident) => {{
        let c = $c;
        pt_ok!(c[0][0], c[0][1], $yp, $xl, $xh)
            & pt_ok!(c[1][0], c[1][1], $yp, $xl, $xh)
            & pt_ok!(c[2][0], c[2][1], $yp, $xl, $xh)
            & pt_ok!(c[3][0], c[3][1], $yp, $xl, $xh)
            & pt_ok!(c[4][0], c[4][1], $yp, $xl, $xh)
            & pt_ok!(c[5][0], c[5][1], $yp, $xl, $xh)
            & pt_ok!(c[6][0], c[6][1], $yp, $xl, $xh)
            & pt_ok!(c[7][0], c[7][1], $yp, $xl, $xh)
    }};
}

macro_rules! chunk8_plot_ok {
    ($c:expr, $yp:ident, $xl:ident, $xh:ident) => {{
        let c = $c;
        pt_ok!(c[0].x, c[0].y, $yp, $xl, $xh)
            & pt_ok!(c[1].x, c[1].y, $yp, $xl, $xh)
            & pt_ok!(c[2].x, c[2].y, $yp, $xl, $xh)
            & pt_ok!(c[3].x, c[3].y, $yp, $xl, $xh)
            & pt_ok!(c[4].x, c[4].y, $yp, $xl, $xh)
            & pt_ok!(c[5].x, c[5].y, $yp, $xl, $xh)
            & pt_ok!(c[6].x, c[6].y, $yp, $xl, $xh)
            & pt_ok!(c[7].x, c[7].y, $yp, $xl, $xh)
    }};
}

#[inline(always)]
fn plateau_x_bounds(
    x0: f64,
    y_plat: f64,
    x1: f64,
    y1: f64,
    want_y_ints: bool,
) -> Option<(f64, f64)> {
    let y_ok = (y1 == y_plat) & (y_plat > f64::NEG_INFINITY) & (y_plat < f64::INFINITY);
    if !y_ok {
        return None;
    }
    if !want_y_ints {
        if (x0 > f64::NEG_INFINITY)
            & (x0 < f64::INFINITY)
            & (x1 > f64::NEG_INFINITY)
            & (x1 < f64::INFINITY)
        {
            Some((f64::NEG_INFINITY, f64::INFINITY))
        } else {
            None
        }
    } else if (x0 > 1e-12) & (x0 < f64::INFINITY) & (x1 > 1e-12) & (x1 < f64::INFINITY) {
        Some((1e-12, f64::INFINITY))
    } else if (x0 < -1e-12) & (x0 > f64::NEG_INFINITY) & (x1 < -1e-12) & (x1 > f64::NEG_INFINITY) {
        Some((f64::NEG_INFINITY, -1e-12))
    } else {
        None
    }
}

fn count_pairs_chunks(tail8: &[[[f64; 2]; 8]], yp: f64, xl: f64, xh: f64) -> usize {
    let (quads, _) = tail8.as_chunks::<4>();
    let mut q = 0usize;
    while q < quads.len() {
        let [c0, c1, c2, c3] = &quads[q];
        if !(chunk8_pairs_ok!(c0, yp, xl, xh)
            & chunk8_pairs_ok!(c1, yp, xl, xh)
            & chunk8_pairs_ok!(c2, yp, xl, xh)
            & chunk8_pairs_ok!(c3, yp, xl, xh))
        {
            break;
        }
        q += 1;
    }
    let mut n = q * 4;
    while n < tail8.len() && chunk8_pairs_ok!(&tail8[n], yp, xl, xh) {
        n += 1;
    }
    n
}

fn compact_pairs(pts: &[[f64; 2]], want_y_ints: bool) -> Option<Vec<[f64; 2]>> {
    if pts.len() < 32 {
        return None;
    }
    let mut out: Option<Vec<[f64; 2]>> = None;
    let mut copied_up_to = 0usize;
    let mut i = 2usize;
    while let [[x0, yp], [x1, y1], rest @ ..] = &pts[i - 2..] {
        let [first8, tail8 @ ..] = rest.as_chunks::<8>().0 else {
            break;
        };
        let yp = *yp;
        if let Some((xl, xh)) = plateau_x_bounds(*x0, yp, *x1, *y1, want_y_ints)
            && chunk8_pairs_ok!(first8, yp, xl, xh)
        {
            let end = i + 8 * (1 + count_pairs_chunks(tail8, yp, xl, xh));
            let buf = out.get_or_insert_with(|| Vec::with_capacity(64));
            buf.extend_from_slice(&pts[copied_up_to..i]);
            buf.push(pts[end - 2]);
            buf.push(pts[end - 1]);
            copied_up_to = end;
            i = end + 2;
        } else {
            i += 8;
        }
    }
    if let Some(ref mut buf) = out {
        buf.extend_from_slice(&pts[copied_up_to..]);
    }
    out
}

fn count_plot_chunks(tail8: &[[PlotPoint; 8]], yp: f64, xl: f64, xh: f64) -> usize {
    let (quads, _) = tail8.as_chunks::<4>();
    let mut q = 0usize;
    while q < quads.len() {
        let [c0, c1, c2, c3] = &quads[q];
        if !(chunk8_plot_ok!(c0, yp, xl, xh)
            & chunk8_plot_ok!(c1, yp, xl, xh)
            & chunk8_plot_ok!(c2, yp, xl, xh)
            & chunk8_plot_ok!(c3, yp, xl, xh))
        {
            break;
        }
        q += 1;
    }
    let mut n = q * 4;
    while n < tail8.len() && chunk8_plot_ok!(&tail8[n], yp, xl, xh) {
        n += 1;
    }
    n
}

fn compact_plot_points(pts: &[PlotPoint], want_y_ints: bool) -> Option<Vec<[f64; 2]>> {
    if pts.len() < 32 {
        return None;
    }
    let mut out: Option<Vec<[f64; 2]>> = None;
    let mut copied_up_to = 0usize;
    let mut i = 2usize;
    while let [s0, s1, rest @ ..] = &pts[i - 2..] {
        let [first8, tail8 @ ..] = rest.as_chunks::<8>().0 else {
            break;
        };
        let yp = s0.y;
        if let Some((xl, xh)) = plateau_x_bounds(s0.x, yp, s1.x, s1.y, want_y_ints)
            && chunk8_plot_ok!(first8, yp, xl, xh)
        {
            let end = i + 8 * (1 + count_plot_chunks(tail8, yp, xl, xh));
            let buf = out.get_or_insert_with(|| Vec::with_capacity(64));
            buf.extend(pts[copied_up_to..i].iter().map(|p| [p.x, p.y]));
            let p_prev = pts[end - 2];
            let p_last = pts[end - 1];
            buf.push([p_prev.x, p_prev.y]);
            buf.push([p_last.x, p_last.y]);
            copied_up_to = end;
            i = end + 2;
        } else {
            i += 8;
        }
    }
    if let Some(ref mut buf) = out {
        buf.extend(pts[copied_up_to..].iter().map(|p| [p.x, p.y]));
    }
    out
}

pub(super) fn compact_long_plateaus(
    coords: SeriesCoords<'_>,
    want_y_ints: bool,
) -> Option<Vec<[f64; 2]>> {
    match coords {
        SeriesCoords::PlotPoints(pts) => compact_plot_points(pts, want_y_ints),
        SeriesCoords::Shared(pts) => compact_pairs(pts, want_y_ints),
    }
}
