//! Zero-copy 2D coordinate series views for curve fitting and critical-point solvers.
// [cite:math_commons]

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Unrolled 16-wide branchless plateau scan over contiguous `(x, y)` sample slices.
#[doc(hidden)]
#[macro_export]
macro_rules! advance_sample2d_plateau {
    ($slice:expr, $start:expr, $end:expr, $flat_y:expr, $x_min:expr, $x_max:expr, |$p:ident| ($x:expr, $y:expr)) => {{
        let slice_ref = $slice;
        let start = $start;
        let end = ($end).min(slice_ref.len());
        if start >= end {
            start
        } else {
            let flat_y = $flat_y;
            let x_min = $x_min;
            let x_max = $x_max;
            let (chunks, rem) = slice_ref[start..end].as_chunks::<16>();
            let n_chunks = chunks.len();
            let mut ci = 0usize;
            while ci < n_chunks {
                let c = &chunks[ci];
                macro_rules! ok_at {
                    ($idx:literal) => {{
                        let $p = &c[$idx];
                        ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                    }};
                }
                let ok = ok_at!(0)
                    & ok_at!(1)
                    & ok_at!(2)
                    & ok_at!(3)
                    & ok_at!(4)
                    & ok_at!(5)
                    & ok_at!(6)
                    & ok_at!(7)
                    & ok_at!(8)
                    & ok_at!(9)
                    & ok_at!(10)
                    & ok_at!(11)
                    & ok_at!(12)
                    & ok_at!(13)
                    & ok_at!(14)
                    & ok_at!(15);
                if !ok {
                    let base = start + (ci << 4);
                    let mut k = 0usize;
                    while k < 16 {
                        let $p = &c[k];
                        if !(($y == flat_y) & ($x >= x_min) & ($x <= x_max)) {
                            return base + k;
                        }
                        k += 1;
                    }
                }
                ci += 1;
            }
            let base = start + (n_chunks << 4);
            let mut k = 0usize;
            while k < rem.len() {
                let $p = &rem[k];
                if !(($y == flat_y) & ($x >= x_min) & ($x <= x_max)) {
                    break;
                }
                k += 1;
            }
            base + k
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

    /// Advances from `start` while each sample satisfies `(y == flat_y) & (x >= x_min) & (x <= x_max)`.
    #[inline(always)]
    fn advance_plateau_run(&self, mut start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let n = self.len();
        while start < n {
            let [x, y] = self.point_at(start);
            if !((y == flat_y) & (x >= x_min) & (x <= x_max)) {
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
    fn advance_plateau_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        S::advance_slice_plateau(self, start, <[S]>::len(self), flat_y, x_min, x_max)
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
    fn advance_plateau_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        S::advance_slice_plateau(self, start, N, flat_y, x_min, x_max)
    }
}

#[cfg(feature = "alloc")]
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
    fn advance_plateau_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        S::advance_slice_plateau(self, start, Vec::len(self), flat_y, x_min, x_max)
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
    fn advance_plateau_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        (**self).advance_plateau_run(start, flat_y, x_min, x_max)
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

#[cfg(feature = "alloc")]
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
