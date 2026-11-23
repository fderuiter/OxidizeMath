//! Zero-copy 2D coordinate series views for curve fitting and critical-point solvers.
// [cite:math_commons]

/// Single 2D `(x, y)` coordinate sample enabling blanket [`PointSeries`] slice implementations.
pub trait Sample2D {
    /// Returns the horizontal (`x`) coordinate of this sample.
    fn sample_x(&self) -> f64;

    /// Returns the vertical (`y`) coordinate of this sample.
    fn sample_y(&self) -> f64;
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

#[inline(always)]
fn advance_sample_slice_plateau<S: Sample2D>(
    slice: &[S],
    mut i: usize,
    flat_y: f64,
    x_min: f64,
    x_max: f64,
) -> usize {
    let (chunks, _) = slice[i..].as_chunks::<4>();
    for [s0, s1, s2, s3] in chunks {
        let (x0, y0) = (s0.sample_x(), s0.sample_y());
        let (x1, y1) = (s1.sample_x(), s1.sample_y());
        let (x2, y2) = (s2.sample_x(), s2.sample_y());
        let (x3, y3) = (s3.sample_x(), s3.sample_y());
        let ok = (y0 == flat_y)
            & (y1 == flat_y)
            & (y2 == flat_y)
            & (y3 == flat_y)
            & (x0 >= x_min)
            & (x0 <= x_max)
            & (x1 >= x_min)
            & (x1 <= x_max)
            & (x2 >= x_min)
            & (x2 <= x_max)
            & (x3 >= x_min)
            & (x3 <= x_max);
        if !ok {
            break;
        }
        i += 4;
    }
    for s in &slice[i..] {
        let (x, y) = (s.sample_x(), s.sample_y());
        if !((y == flat_y) & (x >= x_min) & (x <= x_max)) {
            break;
        }
        i += 1;
    }
    i
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
        advance_sample_slice_plateau(self, start, flat_y, x_min, x_max)
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
        advance_sample_slice_plateau(self, start, flat_y, x_min, x_max)
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
    fn advance_plateau_run(&self, start: usize, flat_y: f64, x_min: f64, x_max: f64) -> usize {
        advance_sample_slice_plateau(self, start, flat_y, x_min, x_max)
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
