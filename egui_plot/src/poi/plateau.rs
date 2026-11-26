//! Fast 8-wide plateau compaction for dense constant-value polylines.

use crate::items::PlotPoint;

use super::SeriesCoords;

trait PlateauSample: Copy {
    fn xy(self) -> [f64; 2];
    fn advance_plateau(tail: &[Self], y_plat: f64, x_dir: f64) -> usize;
}

macro_rules! impl_plateau_sample {
    ($ty:ty, |$p:ident| ($px:expr, $py:expr)) => {
        impl PlateauSample for $ty {
            #[inline(always)]
            fn xy(self) -> [f64; 2] {
                let $p = self;
                [$px, $py]
            }

            fn advance_plateau(tail: &[Self], y_plat: f64, x_dir: f64) -> usize {
                macro_rules! ok {
                    ($q:ident, $x_ok:expr) => {{
                        let $p = $q;
                        ($py == y_plat) & (($px * 0.0) == 0.0) & $x_ok
                    }};
                }
                let mut n = 0usize;
                let (c16, _) = tail.as_chunks::<16>();
                if x_dir == 0.0 {
                    for [a, b, c, d, e, f, g, h, j, k, l, m, o, q, r, s] in c16 {
                        if !(ok!(a, true)
                            & ok!(b, true)
                            & ok!(c, true)
                            & ok!(d, true)
                            & ok!(e, true)
                            & ok!(f, true)
                            & ok!(g, true)
                            & ok!(h, true)
                            & ok!(j, true)
                            & ok!(k, true)
                            & ok!(l, true)
                            & ok!(m, true)
                            & ok!(o, true)
                            & ok!(q, true)
                            & ok!(r, true)
                            & ok!(s, true))
                        {
                            break;
                        }
                        n += 16;
                    }
                    if let Some([a, b, c, d, e, f, g, h]) = tail[n..].as_chunks::<8>().0.first()
                        && (ok!(a, true)
                            & ok!(b, true)
                            & ok!(c, true)
                            & ok!(d, true)
                            & ok!(e, true)
                            & ok!(f, true)
                            & ok!(g, true)
                            & ok!(h, true))
                    {
                        n += 8;
                    }
                } else if x_dir > 0.0 {
                    macro_rules! pos {
                        ($q:ident) => {{
                            let $p = $q;
                            ($py == y_plat) & (($px * 0.0) == 0.0) & ($px > 1e-12)
                        }};
                    }
                    for [a, b, c, d, e, f, g, h, j, k, l, m, o, q, r, s] in c16 {
                        if !(pos!(a)
                            & pos!(b)
                            & pos!(c)
                            & pos!(d)
                            & pos!(e)
                            & pos!(f)
                            & pos!(g)
                            & pos!(h)
                            & pos!(j)
                            & pos!(k)
                            & pos!(l)
                            & pos!(m)
                            & pos!(o)
                            & pos!(q)
                            & pos!(r)
                            & pos!(s))
                        {
                            break;
                        }
                        n += 16;
                    }
                    if let Some([a, b, c, d, e, f, g, h]) = tail[n..].as_chunks::<8>().0.first()
                        && (pos!(a)
                            & pos!(b)
                            & pos!(c)
                            & pos!(d)
                            & pos!(e)
                            & pos!(f)
                            & pos!(g)
                            & pos!(h))
                    {
                        n += 8;
                    }
                } else {
                    macro_rules! neg {
                        ($q:ident) => {{
                            let $p = $q;
                            ($py == y_plat) & (($px * 0.0) == 0.0) & ($px < -1e-12)
                        }};
                    }
                    for [a, b, c, d, e, f, g, h, j, k, l, m, o, q, r, s] in c16 {
                        if !(neg!(a)
                            & neg!(b)
                            & neg!(c)
                            & neg!(d)
                            & neg!(e)
                            & neg!(f)
                            & neg!(g)
                            & neg!(h)
                            & neg!(j)
                            & neg!(k)
                            & neg!(l)
                            & neg!(m)
                            & neg!(o)
                            & neg!(q)
                            & neg!(r)
                            & neg!(s))
                        {
                            break;
                        }
                        n += 16;
                    }
                    if let Some([a, b, c, d, e, f, g, h]) = tail[n..].as_chunks::<8>().0.first()
                        && (neg!(a)
                            & neg!(b)
                            & neg!(c)
                            & neg!(d)
                            & neg!(e)
                            & neg!(f)
                            & neg!(g)
                            & neg!(h))
                    {
                        n += 8;
                    }
                }
                n
            }
        }
    };
}

impl_plateau_sample!(PlotPoint, |p| (p.x, p.y));
impl_plateau_sample!([f64; 2], |p| (p[0], p[1]));

fn compact_slice<T: PlateauSample>(pts: &[T], want_y_ints: bool) -> Option<Vec<[f64; 2]>> {
    if pts.len() < 32 {
        return None;
    }
    let mut out: Option<Vec<[f64; 2]>> = None;
    let mut copied_up_to = 0usize;
    let mut i = 2usize;
    while i + 8 <= pts.len() {
        let [x0, y_plat] = pts[i - 2].xy();
        let [x1, y1] = pts[i - 1].xy();
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
        let extra = if pre_ok {
            T::advance_plateau(&pts[i..], y_plat, x_dir)
        } else {
            0
        };
        if extra >= 8 {
            let end = i + extra;
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
