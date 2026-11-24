//! Zero-copy 2D coordinate series views for curve fitting and critical-point solvers.
// [cite:math_commons]

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

/// Single 2D Cartesian sample $(x, y) \in \mathbb{R}^2$ enabling blanket [`PointSeries`] views (`@goldberg1991every`, `spec:math_commons_v1`).
///
/// - **Numerical stability**: Exact IEEE-754 `f64` coordinate projection with zero rounding error.
/// - **Domain of validity**: Valid for all IEEE-754 `f64` pairs; downstream solvers require finite $(x, y)$ where noted.
pub trait Sample2D {
    /// Projects horizontal coordinate $\pi_x(x, y) = x$ (`@goldberg1991every`, `spec:math_commons_v1`). Numerical stability: exact bitwise `f64` copy. Domain of validity: all `f64` values.
    fn sample_x(&self) -> f64;

    /// Projects vertical coordinate $\pi_y(x, y) = y$ (`@goldberg1991every`, `spec:math_commons_v1`). Numerical stability: exact bitwise `f64` copy. Domain of validity: all `f64` values.
    fn sample_y(&self) -> f64;

    /// Returns maximal slice index $m \in [s, \min(e, n)]$ satisfying $\forall k \in [s, m): y_k = y_{\text{flat}} \land x_k \in [x_{\min}, x_{\max}]$ (`@goldberg1991every`, `spec:math_commons_v1`).
    /// Numerical stability: exact IEEE-754 comparison (`NaN` terminates scan immediately). Domain of validity: $0 \le \text{start} \le \text{end}$ and $x_{\min} \le x_{\max}$.
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

/// Zero-copy indexed view over a finite 2D sample sequence $\{(x_k, y_k)\}_{k=0}^{n-1}$ (`@goldberg1991every`, `spec:math_commons_v1`).
///
/// - **Numerical stability**: Preserves exact source IEEE-754 `f64` coordinates without intermediate conversions or heap allocations.
/// - **Domain of validity**: Valid for finite index ranges $0 \le k < n$; paired slice tuples truncate safely to $n = \min(|X|, |Y|)$.
pub trait PointSeries {
    /// Returns cardinality $n = |\{(x_k, y_k)\}|$ of the series (`@goldberg1991every`, `spec:math_commons_v1`). Numerical stability: exact `usize` count. Domain of validity: $0 \le n \le \text{usize::MAX}$.
    fn len(&self) -> usize;

    /// Evaluates emptiness predicate $n = 0$ (`@goldberg1991every`, `spec:math_commons_v1`). Numerical stability: exact integer comparison. Domain of validity: all `PointSeries` instances.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reads abscissa $x_k$ at sample `index` $k$ (`@goldberg1991every`, `spec:math_commons_v1`). Numerical stability: exact `f64` read without rounding. Domain of validity: requires $0 \le k < \text{self.len()}$.
    fn x_at(&self, index: usize) -> f64;

    /// Reads ordinate $y_k$ at sample `index` $k$ (`@goldberg1991every`, `spec:math_commons_v1`). Numerical stability: exact `f64` read without rounding. Domain of validity: requires $0 \le k < \text{self.len()}$.
    fn y_at(&self, index: usize) -> f64;

    /// Reads paired sample $\mathbf{p}_k = [x_k, y_k]$ at `index` $k$ (`@goldberg1991every`, `spec:math_commons_v1`). Numerical stability: exact two-element `f64` copy. Domain of validity: requires $0 \le k < \text{self.len()}$.
    #[inline(always)]
    fn point_at(&self, index: usize) -> [f64; 2] {
        [self.x_at(index), self.y_at(index)]
    }

    /// Returns maximal index $m \ge s$ such that $\forall k \in [s, m): y_k = y_{\text{flat}} \land x_k \in [x_{\min}, x_{\max}]$ (`@goldberg1991every`, `spec:math_commons_v1`).
    /// Numerical stability: exact IEEE-754 comparison (`NaN` terminates plateau scan immediately). Domain of validity: $0 \le \text{start} \le \text{self.len()}$ and $x_{\min} \le x_{\max}$.
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
