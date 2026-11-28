//! Zero-copy 2D coordinate series views for curve fitting and critical-point solvers.
// [cite:math_scalars]

use crate::RealField;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Unrolled 16-wide branchless plateau scan over contiguous `(x, y)` sample slices.
#[doc(hidden)]
#[macro_export]
macro_rules! advance_sample2d_plateau {
    ($slice:expr, $start:expr, $end:expr, $flat_y:expr, $x_min:expr, $x_max:expr, |$p:pat_param| ($x:expr, $y:expr)) => {{
        let slice_ref = $slice;
        let start = $start;
        let end = ($end).min(slice_ref.len());
        if start >= end {
            start
        } else {
            'plateau: {
                let flat_y = $flat_y;
                let x_min = $x_min;
                let x_max = $x_max;
                let (chunks, rem) = slice_ref[start..end].as_chunks::<16>();
                let mut base = start;
                let mut chunk_slice = chunks;
                while let [c, rest @ ..] = chunk_slice {
                    let [
                        s0,
                        s1,
                        s2,
                        s3,
                        s4,
                        s5,
                        s6,
                        s7,
                        s8,
                        s9,
                        s10,
                        s11,
                        s12,
                        s13,
                        s14,
                        s15,
                    ] = c;
                    macro_rules! ok_sample {
                        ($s:ident) => {{
                            let $p = $s;
                            ($y == flat_y) & ($x >= x_min) & ($x <= x_max)
                        }};
                    }
                    let ok = ok_sample!(s0)
                        & ok_sample!(s1)
                        & ok_sample!(s2)
                        & ok_sample!(s3)
                        & ok_sample!(s4)
                        & ok_sample!(s5)
                        & ok_sample!(s6)
                        & ok_sample!(s7)
                        & ok_sample!(s8)
                        & ok_sample!(s9)
                        & ok_sample!(s10)
                        & ok_sample!(s11)
                        & ok_sample!(s12)
                        & ok_sample!(s13)
                        & ok_sample!(s14)
                        & ok_sample!(s15);
                    if !ok {
                        let mut tail: &[_] = c;
                        while let [$p, tail_rest @ ..] = tail {
                            if !(($y == flat_y) & ($x >= x_min) & ($x <= x_max)) {
                                break 'plateau base;
                            }
                            base += 1;
                            tail = tail_rest;
                        }
                        break 'plateau base;
                    }
                    base += 16;
                    chunk_slice = rest;
                }
                let mut rem_slice = rem;
                while let [$p, rest @ ..] = rem_slice {
                    if !(($y == flat_y) & ($x >= x_min) & ($x <= x_max)) {
                        break;
                    }
                    base += 1;
                    rem_slice = rest;
                }
                base
            }
        }
    }};
}

/// Single 2D `(x, y)` coordinate sample enabling blanket [`PointSeries`] slice implementations.
pub trait Sample2D<T: RealField = f64> {
    /// Returns the horizontal (`x`) coordinate of this sample.
    fn sample_x(&self) -> T;

    /// Returns the vertical (`y`) coordinate of this sample.
    fn sample_y(&self) -> T;

    /// Advances from `start` up to `end` across `slice` while `y == flat_y` and `x_min <= x <= x_max`.
    #[inline(always)]
    fn advance_slice_plateau(
        slice: &[Self],
        mut start: usize,
        end: usize,
        flat_y: T,
        x_min: T,
        x_max: T,
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

#[inline(always)]
fn advance_f64_array_plateau(
    slice: &[[f64; 2]],
    start: usize,
    end: usize,
    flat_y: f64,
    x_min: f64,
    x_max: f64,
) -> usize {
    advance_sample2d_plateau!(slice, start, end, flat_y, x_min, x_max, |[px, py]| (
        *px, *py
    ))
}

#[inline(always)]
fn advance_f32_array_plateau(
    slice: &[[f32; 2]],
    start: usize,
    end: usize,
    flat_y: f32,
    x_min: f32,
    x_max: f32,
) -> usize {
    advance_sample2d_plateau!(slice, start, end, flat_y, x_min, x_max, |[px, py]| (
        *px, *py
    ))
}

#[inline(always)]
fn advance_f64_tuple_plateau(
    slice: &[(f64, f64)],
    start: usize,
    end: usize,
    flat_y: f64,
    x_min: f64,
    x_max: f64,
) -> usize {
    advance_sample2d_plateau!(slice, start, end, flat_y, x_min, x_max, |(px, py)| (
        *px, *py
    ))
}

#[inline(always)]
fn advance_f32_tuple_plateau(
    slice: &[(f32, f32)],
    start: usize,
    end: usize,
    flat_y: f32,
    x_min: f32,
    x_max: f32,
) -> usize {
    advance_sample2d_plateau!(slice, start, end, flat_y, x_min, x_max, |(px, py)| (
        *px, *py
    ))
}

impl<T: RealField> Sample2D<T> for [T; 2] {
    #[inline(always)]
    fn sample_x(&self) -> T {
        self[0]
    }

    #[inline(always)]
    fn sample_y(&self) -> T {
        self[1]
    }

    #[inline(always)]
    fn advance_slice_plateau(
        slice: &[Self],
        start: usize,
        end: usize,
        flat_y: T,
        x_min: T,
        x_max: T,
    ) -> usize {
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
            debug_assert_eq!(
                core::mem::size_of::<Self>(),
                core::mem::size_of::<[f64; 2]>()
            );
            // SAFETY: `TypeId` proves `T` is `f64`, so `[T; 2]` and `T` have identical layout to `[f64; 2]` and `f64`.
            let s = unsafe {
                core::slice::from_raw_parts(slice.as_ptr().cast::<[f64; 2]>(), slice.len())
            };
            let fy = unsafe { core::ptr::read((&raw const flat_y).cast::<f64>()) };
            let xmin = unsafe { core::ptr::read((&raw const x_min).cast::<f64>()) };
            let xmax = unsafe { core::ptr::read((&raw const x_max).cast::<f64>()) };
            return advance_f64_array_plateau(s, start, end, fy, xmin, xmax);
        }
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
            debug_assert_eq!(
                core::mem::size_of::<Self>(),
                core::mem::size_of::<[f32; 2]>()
            );
            // SAFETY: `TypeId` proves `T` is `f32`, so `[T; 2]` and `T` have identical layout to `[f32; 2]` and `f32`.
            let s = unsafe {
                core::slice::from_raw_parts(slice.as_ptr().cast::<[f32; 2]>(), slice.len())
            };
            let fy = unsafe { core::ptr::read((&raw const flat_y).cast::<f32>()) };
            let xmin = unsafe { core::ptr::read((&raw const x_min).cast::<f32>()) };
            let xmax = unsafe { core::ptr::read((&raw const x_max).cast::<f32>()) };
            return advance_f32_array_plateau(s, start, end, fy, xmin, xmax);
        }
        advance_sample2d_plateau!(slice, start, end, flat_y, x_min, x_max, |[px, py]| (
            *px, *py
        ))
    }
}

impl<T: RealField> Sample2D<T> for (T, T) {
    #[inline(always)]
    fn sample_x(&self) -> T {
        self.0
    }

    #[inline(always)]
    fn sample_y(&self) -> T {
        self.1
    }

    #[inline(always)]
    fn advance_slice_plateau(
        slice: &[Self],
        start: usize,
        end: usize,
        flat_y: T,
        x_min: T,
        x_max: T,
    ) -> usize {
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
            debug_assert_eq!(
                core::mem::size_of::<Self>(),
                core::mem::size_of::<(f64, f64)>()
            );
            // SAFETY: `TypeId` proves `T` is `f64`, so `(T, T)` is identical to `(f64, f64)`.
            let s = unsafe {
                core::slice::from_raw_parts(slice.as_ptr().cast::<(f64, f64)>(), slice.len())
            };
            let fy = unsafe { core::ptr::read((&raw const flat_y).cast::<f64>()) };
            let xmin = unsafe { core::ptr::read((&raw const x_min).cast::<f64>()) };
            let xmax = unsafe { core::ptr::read((&raw const x_max).cast::<f64>()) };
            return advance_f64_tuple_plateau(s, start, end, fy, xmin, xmax);
        }
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
            debug_assert_eq!(
                core::mem::size_of::<Self>(),
                core::mem::size_of::<(f32, f32)>()
            );
            // SAFETY: `TypeId` proves `T` is `f32`, so `(T, T)` is identical to `(f32, f32)`.
            let s = unsafe {
                core::slice::from_raw_parts(slice.as_ptr().cast::<(f32, f32)>(), slice.len())
            };
            let fy = unsafe { core::ptr::read((&raw const flat_y).cast::<f32>()) };
            let xmin = unsafe { core::ptr::read((&raw const x_min).cast::<f32>()) };
            let xmax = unsafe { core::ptr::read((&raw const x_max).cast::<f32>()) };
            return advance_f32_tuple_plateau(s, start, end, fy, xmin, xmax);
        }
        advance_sample2d_plateau!(slice, start, end, flat_y, x_min, x_max, |(px, py)| (
            *px, *py
        ))
    }
}

/// Zero-copy indexed view over a sequence of 2D `(x, y)` observations over `T: RealField`.
///
/// Implemented for `[T; 2]` slices/arrays/vectors, `(T, T)` slices/arrays/vectors,
/// paired coordinate slices `(&[T], &[T])`, and downstream UI point types implementing
/// [`Sample2D`].
pub trait PointSeries<T: RealField = f64> {
    /// Number of accessible `(x, y)` points in the series.
    fn len(&self) -> usize;

    /// Returns `true` when the series contains zero points.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Horizontal (`x`) coordinate at `index`.
    fn x_at(&self, index: usize) -> T;

    /// Vertical (`y`) coordinate at `index`.
    fn y_at(&self, index: usize) -> T;

    /// Returns `[x, y]` at `index`.
    #[inline(always)]
    fn point_at(&self, index: usize) -> [T; 2] {
        [self.x_at(index), self.y_at(index)]
    }

    /// Advances from `start` while each sample satisfies `(y == flat_y) & (x >= x_min) & (x <= x_max)`.
    #[inline(always)]
    fn advance_plateau_run(&self, mut start: usize, flat_y: T, x_min: T, x_max: T) -> usize {
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

impl<T: RealField, S: Sample2D<T>> PointSeries<T> for [S] {
    #[inline(always)]
    fn len(&self) -> usize {
        <[S]>::len(self)
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> T {
        self[index].sample_x()
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> T {
        self[index].sample_y()
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [T; 2] {
        let s = &self[index];
        [s.sample_x(), s.sample_y()]
    }

    #[inline(always)]
    fn advance_plateau_run(&self, start: usize, flat_y: T, x_min: T, x_max: T) -> usize {
        S::advance_slice_plateau(self, start, <[S]>::len(self), flat_y, x_min, x_max)
    }
}

impl<T: RealField, S: Sample2D<T>, const N: usize> PointSeries<T> for [S; N] {
    #[inline(always)]
    fn len(&self) -> usize {
        N
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> T {
        self[index].sample_x()
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> T {
        self[index].sample_y()
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [T; 2] {
        let s = &self[index];
        [s.sample_x(), s.sample_y()]
    }

    #[inline(always)]
    fn advance_plateau_run(&self, start: usize, flat_y: T, x_min: T, x_max: T) -> usize {
        S::advance_slice_plateau(self, start, N, flat_y, x_min, x_max)
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField, S: Sample2D<T>> PointSeries<T> for Vec<S> {
    #[inline(always)]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> T {
        self[index].sample_x()
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> T {
        self[index].sample_y()
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [T; 2] {
        let s = &self[index];
        [s.sample_x(), s.sample_y()]
    }

    #[inline(always)]
    fn advance_plateau_run(&self, start: usize, flat_y: T, x_min: T, x_max: T) -> usize {
        S::advance_slice_plateau(self, start, Vec::len(self), flat_y, x_min, x_max)
    }
}

impl<T: RealField, P: PointSeries<T> + ?Sized> PointSeries<T> for &P {
    #[inline(always)]
    fn len(&self) -> usize {
        (**self).len()
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> T {
        (**self).x_at(index)
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> T {
        (**self).y_at(index)
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [T; 2] {
        (**self).point_at(index)
    }

    #[inline(always)]
    fn advance_plateau_run(&self, start: usize, flat_y: T, x_min: T, x_max: T) -> usize {
        (**self).advance_plateau_run(start, flat_y, x_min, x_max)
    }
}

impl<T: RealField> PointSeries<T> for (&[T], &[T]) {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len().min(self.1.len())
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> T {
        self.0[index]
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> T {
        self.1[index]
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [T; 2] {
        [self.0[index], self.1[index]]
    }
}

#[cfg(feature = "alloc")]
impl<T: RealField> PointSeries<T> for (&Vec<T>, &Vec<T>) {
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len().min(self.1.len())
    }

    #[inline(always)]
    fn x_at(&self, index: usize) -> T {
        self.0[index]
    }

    #[inline(always)]
    fn y_at(&self, index: usize) -> T {
        self.1[index]
    }

    #[inline(always)]
    fn point_at(&self, index: usize) -> [T; 2] {
        [self.0[index], self.1[index]]
    }
}
