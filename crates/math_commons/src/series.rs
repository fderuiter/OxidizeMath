//! Zero-copy 2D coordinate series views for curve fitting and critical-point solvers.
// [cite:math_commons]

/// Unrolled 8-wide branchless plateau scan over contiguous `(x, y)` sample slices.
#[doc(hidden)]
#[macro_export]
macro_rules! advance_sample2d_plateau {
    ($slice:expr, $start:expr, $end:expr, $flat_y:expr, $x_min:expr, $x_max:expr, |$p:ident| ($x:expr, $y:expr)) => {{
        let start = $start;
        let end = ($end).min($slice.len());
        if start >= end {
            start
        } else {
            let flat_y = $flat_y;
            let x_min = $x_min;
            let x_max = $x_max;
            let (chunks, rem) = $slice[start..end].as_chunks::<8>();
            let mut skipped = 0usize;
            for &[a0, a1, a2, a3, a4, a5, a6, a7] in chunks {
                let ok = {
                    let $p = a0;
                    ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                } & {
                    let $p = a1;
                    ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                } & {
                    let $p = a2;
                    ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                } & {
                    let $p = a3;
                    ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                } & {
                    let $p = a4;
                    ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                } & {
                    let $p = a5;
                    ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                } & {
                    let $p = a6;
                    ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                } & {
                    let $p = a7;
                    ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                };
                if !ok {
                    for item in [a0, a1, a2, a3, a4, a5, a6, a7] {
                        let $p = item;
                        if !(($y == flat_y) & ($x >= x_min) & ($x <= x_max)) {
                            return start + skipped;
                        }
                        skipped += 1;
                    }
                }
                skipped += 8;
            }
            for &item in rem {
                let $p = item;
                if !(($y == flat_y) & ($x >= x_min) & ($x <= x_max)) {
                    break;
                }
                skipped += 1;
            }
            start + skipped
        }
    }};
}

/// Single 2D `(x, y)` coordinate sample enabling blanket [`PointSeries`] slice implementations.
pub trait Sample2D {
    /// Returns the horizontal (`x`) coordinate of this sample.
    fn sample_x(&self) -> f64;

    /// Returns the vertical (`y`) coordinate of this sample.
    fn sample_y(&self) -> f64;

    /// Advances from `start` up to `end` across `slice` while `y == flat_y` and `x_min <= x <= x_max`.
    #[inline(always)]
    fn advance_slice_plateau(
        slice: &[Self],
        mut start: usize,
        end: usize,
        flat_y: f64,
        x_min: f64,
        x_max: f64,
    ) -> usize
    where
        Self: Sized,
    {
        let end = end.min(slice.len());
        while start < end {
            let s = &slice[start];
            let (x, y) = (s.sample_x(), s.sample_y());
            if !((y == flat_y) & (x >= x_min) & (x <= x_max)) {
                break;
            }
            start += 1;
        }
        start
    }
}

impl Sample2D for [f64; 2] {
    #[inline(always)]
    fn sample_x(&self) -> f64 {
        self[0]
    }

    #[inline(always)]
    fn sample_y(&self) -> f64 {
        self[1]
    }

    #[inline(always)]
    fn advance_slice_plateau(
        slice: &[Self],
        start: usize,
        end: usize,
        flat_y: f64,
        x_min: f64,
        x_max: f64,
    ) -> usize {
        advance_sample2d_plateau!(slice, start, end, flat_y, x_min, x_max, |p| (p[0], p[1]))
    }
}

impl Sample2D for (f64, f64) {
    #[inline(always)]
    fn sample_x(&self) -> f64 {
        self.0
    }

    #[inline(always)]
    fn sample_y(&self) -> f64 {
        self.1
    }

    #[inline(always)]
    fn advance_slice_plateau(
        slice: &[Self],
        start: usize,
        end: usize,
        flat_y: f64,
        x_min: f64,
        x_max: f64,
    ) -> usize {
        advance_sample2d_plateau!(slice, start, end, flat_y, x_min, x_max, |p| (p.0, p.1))
    }
}

/// Zero-copy indexed view over a sequence of 2D `(x, y)` observations.
///
/// Implemented for `[f64; 2]` slices/arrays/vectors, `(f64, f64)` slices/arrays/vectors,
/// paired coordinate slices `(&[f64], &[f64])`, and downstream UI point types implementing
/// [`Sample2D`].
pub trait PointSeries {
    /// Number of accessible `(x, y)` points in the series.
    fn len(&self) -> usize;

    /// Returns `true` when the series contains zero points.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Horizontal (`x`) coordinate at `index`.
    fn x_at(&self, index: usize) -> f64;

    /// Vertical (`y`) coordinate at `index`.
    fn y_at(&self, index: usize) -> f64;

    /// Returns `[x, y]` at `index`.
    #[inline(always)]
    fn point_at(&self, index: usize) -> [f64; 2] {
        [self.x_at(index), self.y_at(index)]
    }

    /// Advances from `start` up to `end` while `y == flat_y` and `x_min <= x <= x_max`.
    #[inline(always)]
    fn advance_plateau(
        &self,
        mut start: usize,
        end: usize,
        flat_y: f64,
        x_min: f64,
        x_max: f64,
    ) -> usize {
        let end = end.min(self.len());
        while start < end {
            let p = self.point_at(start);
            if !((p[1] == flat_y) & (p[0] >= x_min) & (p[0] <= x_max)) {
                break;
            }
            start += 1;
        }
        start
    }
}

impl<S: Sample2D> PointSeries for [S] {
    #[inline(always)]
    fn len(&self) -> usize {
        <[S]>::len(self)
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> f64 {
        self[index].sample_x()
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> f64 {
        self[index].sample_y()
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [f64; 2] {
        let s = &self[index];
        [s.sample_x(), s.sample_y()]
    }

    #[inline(always)]
    fn advance_plateau(
        &self,
        start: usize,
        end: usize,
        flat_y: f64,
        x_min: f64,
        x_max: f64,
    ) -> usize {
        S::advance_slice_plateau(self, start, end, flat_y, x_min, x_max)
    }
}

impl<S: Sample2D, const N: usize> PointSeries for [S; N] {
    #[inline(always)]
    fn len(&self) -> usize {
        N
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> f64 {
        self[index].sample_x()
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> f64 {
        self[index].sample_y()
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [f64; 2] {
        let s = &self[index];
        [s.sample_x(), s.sample_y()]
    }

    #[inline(always)]
    fn advance_plateau(
        &self,
        start: usize,
        end: usize,
        flat_y: f64,
        x_min: f64,
        x_max: f64,
    ) -> usize {
        S::advance_slice_plateau(self, start, end, flat_y, x_min, x_max)
    }
}

impl<S: Sample2D> PointSeries for Vec<S> {
    #[inline(always)]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> f64 {
        self[index].sample_x()
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> f64 {
        self[index].sample_y()
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [f64; 2] {
        let s = &self[index];
        [s.sample_x(), s.sample_y()]
    }

    #[inline(always)]
    fn advance_plateau(
        &self,
        start: usize,
        end: usize,
        flat_y: f64,
        x_min: f64,
        x_max: f64,
    ) -> usize {
        S::advance_slice_plateau(self, start, end, flat_y, x_min, x_max)
    }
}

impl<T: PointSeries + ?Sized> PointSeries for &T {
    #[inline(always)]
    fn len(&self) -> usize {
        (**self).len()
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> f64 {
        (**self).x_at(index)
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> f64 {
        (**self).y_at(index)
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [f64; 2] {
        (**self).point_at(index)
    }

    #[inline(always)]
    fn advance_plateau(
        &self,
        start: usize,
        end: usize,
        flat_y: f64,
        x_min: f64,
        x_max: f64,
    ) -> usize {
        (**self).advance_plateau(start, end, flat_y, x_min, x_max)
    }
}

impl PointSeries for (&[f64], &[f64]) {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len().min(self.1.len())
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> f64 {
        self.0[index]
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> f64 {
        self.1[index]
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [f64; 2] {
        [self.0[index], self.1[index]]
    }
}

impl PointSeries for (&Vec<f64>, &Vec<f64>) {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len().min(self.1.len())
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> f64 {
        self.0[index]
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> f64 {
        self.1[index]
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [f64; 2] {
        [self.0[index], self.1[index]]
    }
}
