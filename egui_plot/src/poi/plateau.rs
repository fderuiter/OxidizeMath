//! Fast 8-wide plateau compaction for dense constant-value polylines.

use crate::items::PlotPoint;

use super::SeriesCoords;

trait PlateauSample: Copy {
    fn xy(self) -> [f64; 2];
    fn chunk8_same_plateau(c: &[Self; 8], y_plat: f64, x_dir: f64) -> bool;
}

impl PlateauSample for PlotPoint {
    #[inline(always)]
    fn xy(self) -> [f64; 2] {
        [self.x, self.y]
    }

    #[inline(always)]
    fn chunk8_same_plateau(c: &[Self; 8], y_plat: f64, x_dir: f64) -> bool {
        let same_y = (c[0].y == y_plat)
            & (c[1].y == y_plat)
            & (c[2].y == y_plat)
            & (c[3].y == y_plat)
            & (c[4].y == y_plat)
            & (c[5].y == y_plat)
            & (c[6].y == y_plat)
            & (c[7].y == y_plat);
        let fin_x = ((c[0].x * 0.0)
            + (c[1].x * 0.0)
            + (c[2].x * 0.0)
            + (c[3].x * 0.0)
            + (c[4].x * 0.0)
            + (c[5].x * 0.0)
            + (c[6].x * 0.0)
            + (c[7].x * 0.0))
            == 0.0;
        let safe_x = (x_dir == 0.0)
            | ((c[0].x * x_dir > 1e-12)
                & (c[1].x * x_dir > 1e-12)
                & (c[2].x * x_dir > 1e-12)
                & (c[3].x * x_dir > 1e-12)
                & (c[4].x * x_dir > 1e-12)
                & (c[5].x * x_dir > 1e-12)
                & (c[6].x * x_dir > 1e-12)
                & (c[7].x * x_dir > 1e-12));
        same_y & fin_x & safe_x
    }
}

impl PlateauSample for [f64; 2] {
    #[inline(always)]
    fn xy(self) -> [f64; 2] {
        self
    }

    #[inline(always)]
    fn chunk8_same_plateau(c: &[Self; 8], y_plat: f64, x_dir: f64) -> bool {
        let same_y = (c[0][1] == y_plat)
            & (c[1][1] == y_plat)
            & (c[2][1] == y_plat)
            & (c[3][1] == y_plat)
            & (c[4][1] == y_plat)
            & (c[5][1] == y_plat)
            & (c[6][1] == y_plat)
            & (c[7][1] == y_plat);
        let fin_x = ((c[0][0] * 0.0)
            + (c[1][0] * 0.0)
            + (c[2][0] * 0.0)
            + (c[3][0] * 0.0)
            + (c[4][0] * 0.0)
            + (c[5][0] * 0.0)
            + (c[6][0] * 0.0)
            + (c[7][0] * 0.0))
            == 0.0;
        let safe_x = (x_dir == 0.0)
            | ((c[0][0] * x_dir > 1e-12)
                & (c[1][0] * x_dir > 1e-12)
                & (c[2][0] * x_dir > 1e-12)
                & (c[3][0] * x_dir > 1e-12)
                & (c[4][0] * x_dir > 1e-12)
                & (c[5][0] * x_dir > 1e-12)
                & (c[6][0] * x_dir > 1e-12)
                & (c[7][0] * x_dir > 1e-12));
        same_y & fin_x & safe_x
    }
}

fn compact_slice<T: PlateauSample>(pts: &[T], want_y_ints: bool) -> Option<Vec<[f64; 2]>> {
    if pts.len() < 32 {
        return None;
    }
    let mut out: Option<Vec<[f64; 2]>> = None;
    let mut copied_up_to = 0usize;
    let mut i = 2usize;
    while let Some(&[s0, s1, ref tail @ ..]) = pts.get(i - 2..) {
        let (chunks, _) = tail.as_chunks::<8>();
        let Some((first_chunk, rest_chunks)) = chunks.split_first() else {
            break;
        };
        let [x0, y_plat] = s0.xy();
        let [x1, y1] = s1.xy();
        let pre_ok = (x0 * 0.0 + y_plat * 0.0 + x1 * 0.0) == 0.0 && y1 == y_plat;
        let x_dir = if !want_y_ints {
            0.0
        } else if x0 > 1e-12 && x1 > 1e-12 {
            1.0
        } else if x0 < -1e-12 && x1 < -1e-12 {
            -1.0
        } else {
            i += 8;
            continue;
        };
        if pre_ok && T::chunk8_same_plateau(first_chunk, y_plat, x_dir) {
            let mut end = i + 8;
            for next in rest_chunks {
                if !T::chunk8_same_plateau(next, y_plat, x_dir) {
                    break;
                }
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
