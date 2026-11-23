//! Zero-copy 2D coordinate series views for curve fitting and critical-point solvers.
// [cite:math_commons]

/// Single 2D `(x, y)` coordinate sample enabling blanket [`PointSeries`] slice implementations.
pub trait Sample2D {
    /// Returns the horizontal (`x`) coordinate of this sample.
    fn sample_x(&self) -> f64;

    /// Returns the vertical (`y`) coordinate of this sample.
    fn sample_y(&self) -> f64;

    /// Counts consecutive leading samples in `slice` whose vertical coordinate equals `flat_y`
    /// and whose finite horizontal coordinate lies within `x_min..=x_max`.
    #[inline]
    fn flat_run_prefix_len(slice: &[Self], flat_y: f64, x_min: f64, x_max: f64) -> usize
    where
        Self: Sized,
    {
        let mut k = 0;
        let len = slice.len();
        while k < len {
            let s = &slice[k];
            let x = s.sample_x();
            let y = s.sample_y();
            if y != flat_y || x < x_min || x > x_max || (x * 0.0) != 0.0 {
                break;
            }
            k += 1;
        }
        k
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
    fn flat_run_prefix_len(slice: &[Self], flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let mut k = 0;
        let len = slice.len();
        while k < len {
            let p = slice[k];
            if p[1] != flat_y || p[0] < x_min || p[0] > x_max || (p[0] * 0.0) != 0.0 {
                break;
            }
            k += 1;
        }
        k
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
    fn flat_run_prefix_len(slice: &[Self], flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let mut k = 0;
        let len = slice.len();
        while k < len {
            let p = slice[k];
            if p.1 != flat_y || p.0 < x_min || p.0 > x_max || (p.0 * 0.0) != 0.0 {
                break;
            }
            k += 1;
        }
        k
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

    /// Returns the first index `i >= start` (clamped to `self.len()`) where `y_at(i) != flat_y`
    /// or `x_at(i)` falls outside the finite interval `x_min..=x_max`.
    #[inline]
    fn advance_flat_y_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let n = self.len();
        let mut i = start.min(n);
        while i < n {
            let p = self.point_at(i);
            if p[1] != flat_y || p[0] < x_min || p[0] > x_max || (p[0] * 0.0) != 0.0 {
                break;
            }
            i += 1;
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

    #[inline]
    fn advance_flat_y_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let s = start.min(self.len());
        s + S::flat_run_prefix_len(&self[s..], flat_y, x_min, x_max)
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

    #[inline]
    fn advance_flat_y_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let s = start.min(N);
        s + S::flat_run_prefix_len(&self[s..], flat_y, x_min, x_max)
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

    #[inline]
    fn advance_flat_y_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let s = start.min(self.len());
        s + S::flat_run_prefix_len(&self[s..], flat_y, x_min, x_max)
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

    #[inline]
    fn advance_flat_y_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        (**self).advance_flat_y_run(start, flat_y, x_min, x_max)
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

    #[inline]
    fn advance_flat_y_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        let n = self.0.len().min(self.1.len());
        let xs = &self.0[..n];
        let ys = &self.1[..n];
        let mut i = start.min(n);
        while i < n {
            let x = xs[i];
            if ys[i] != flat_y || x < x_min || x > x_max || (x * 0.0) != 0.0 {
                break;
            }
            i += 1;
        }
        i
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

    #[inline]
    fn advance_flat_y_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        (self.0.as_slice(), self.1.as_slice()).advance_flat_y_run(start, flat_y, x_min, x_max)
    }
}
