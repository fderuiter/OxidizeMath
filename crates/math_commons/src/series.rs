//! Zero-copy 2D coordinate series views for curve fitting and critical-point solvers.
// [cite:math_commons]

/// Single 2D `(x, y)` coordinate sample enabling blanket [`PointSeries`] slice implementations.
pub trait Sample2D {
    /// Returns the horizontal (`x`) coordinate of this sample.
    fn sample_x(&self) -> f64;

    /// Returns the vertical (`y`) coordinate of this sample.
    fn sample_y(&self) -> f64;

    /// Counts leading samples in `slice` with `y == flat_y` and `x` in `[x_min, x_max]`.
    #[inline]
    fn advance_flat_slice(slice: &[Self], flat_y: f64, x_min: f64, x_max: f64) -> usize
    where
        Self: Sized,
    {
        let mut count = 0usize;
        for s in slice {
            let x = s.sample_x();
            let y = s.sample_y();
            if (y == flat_y) & (x >= x_min) & (x <= x_max) {
                count += 1;
            } else {
                break;
            }
        }
        count
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

    #[inline]
    fn advance_flat_slice(slice: &[Self], flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let mut rem = slice;
        let mut count = 0usize;
        while let [p0, p1, p2, p3, p4, p5, p6, p7, rest @ ..] = rem {
            let ok = (p0[1] == flat_y)
                & (p0[0] >= x_min)
                & (p0[0] <= x_max)
                & (p1[1] == flat_y)
                & (p1[0] >= x_min)
                & (p1[0] <= x_max)
                & (p2[1] == flat_y)
                & (p2[0] >= x_min)
                & (p2[0] <= x_max)
                & (p3[1] == flat_y)
                & (p3[0] >= x_min)
                & (p3[0] <= x_max)
                & (p4[1] == flat_y)
                & (p4[0] >= x_min)
                & (p4[0] <= x_max)
                & (p5[1] == flat_y)
                & (p5[0] >= x_min)
                & (p5[0] <= x_max)
                & (p6[1] == flat_y)
                & (p6[0] >= x_min)
                & (p6[0] <= x_max)
                & (p7[1] == flat_y)
                & (p7[0] >= x_min)
                & (p7[0] <= x_max);
            if !ok {
                break;
            }
            rem = rest;
            count += 8;
        }
        for p in rem {
            if (p[1] == flat_y) & (p[0] >= x_min) & (p[0] <= x_max) {
                count += 1;
            } else {
                break;
            }
        }
        count
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

    #[inline]
    fn advance_flat_slice(slice: &[Self], flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let mut rem = slice;
        let mut count = 0usize;
        while let [p0, p1, p2, p3, p4, p5, p6, p7, rest @ ..] = rem {
            let ok = (p0.1 == flat_y)
                & (p0.0 >= x_min)
                & (p0.0 <= x_max)
                & (p1.1 == flat_y)
                & (p1.0 >= x_min)
                & (p1.0 <= x_max)
                & (p2.1 == flat_y)
                & (p2.0 >= x_min)
                & (p2.0 <= x_max)
                & (p3.1 == flat_y)
                & (p3.0 >= x_min)
                & (p3.0 <= x_max)
                & (p4.1 == flat_y)
                & (p4.0 >= x_min)
                & (p4.0 <= x_max)
                & (p5.1 == flat_y)
                & (p5.0 >= x_min)
                & (p5.0 <= x_max)
                & (p6.1 == flat_y)
                & (p6.0 >= x_min)
                & (p6.0 <= x_max)
                & (p7.1 == flat_y)
                & (p7.0 >= x_min)
                & (p7.0 <= x_max);
            if !ok {
                break;
            }
            rem = rest;
            count += 8;
        }
        for p in rem {
            if (p.1 == flat_y) & (p.0 >= x_min) & (p.0 <= x_max) {
                count += 1;
            } else {
                break;
            }
        }
        count
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

    /// Advances past a contiguous flat-`y` run starting at `start` with `x` in `[x_min, x_max]`.
    #[inline]
    fn advance_flat_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let n = self.len();
        let mut i = start;
        while i < n {
            let p = self.point_at(i);
            if (p[1] == flat_y) & (p[0] >= x_min) & (p[0] <= x_max) {
                i += 1;
            } else {
                break;
            }
        }
        i
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
    fn advance_flat_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        if start >= self.len() {
            return self.len();
        }
        start + S::advance_flat_slice(&self[start..], flat_y, x_min, x_max)
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
    fn advance_flat_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        <[S] as PointSeries>::advance_flat_run(self.as_slice(), start, flat_y, x_min, x_max)
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
    fn advance_flat_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        <[S] as PointSeries>::advance_flat_run(self.as_slice(), start, flat_y, x_min, x_max)
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
    fn advance_flat_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        (**self).advance_flat_run(start, flat_y, x_min, x_max)
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
