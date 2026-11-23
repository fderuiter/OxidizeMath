//! Stack `ArrayState<N>` and optional `alloc`/`nalgebra` state implementations of [`VectorOperations`] and [`OdeState`].
// [cite:engine_diffeq]

use super::traits::{OdeState, VectorOperations};
use core::ops::{Add, AddAssign, Mul, MulAssign};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "nalgebra")]
use nalgebra::{DVector, Vector2, Vector3};

/// Zero-overhead fixed-size stack array implementing [`VectorOperations`] and [`OdeState`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrayState<const N: usize>(pub [f64; N]);

impl<const N: usize> Default for ArrayState<N> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self([0.0; N])
    }
}

impl<const N: usize> ArrayState<N> {
    /// Constructs a zero-initialized stack state vector of dimension `N`.
    #[must_use]
    #[verified_engine::verified]
    pub fn zeros() -> Self {
        Self([0.0; N])
    }

    /// Computes the $L_\infty$ (maximum absolute component) difference between two states.
    #[must_use]
    #[verified_engine::verified]
    pub fn max_abs_diff(&self, other: &Self) -> f64 {
        let mut max_diff = 0.0_f64;
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            let diff = (*a - *b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        max_diff
    }

    /// Returns `true` if every component of the state vector is finite (`!NaN && !Infinite`).
    #[must_use]
    #[verified_engine::verified]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl<const N: usize> Add for ArrayState<N> {
    type Output = Self;
    #[verified_engine::verified]
    fn add(self, rhs: Self) -> Self {
        let mut arr = [0.0; N];
        for (i, val) in arr.iter_mut().enumerate() {
            *val = self.0[i] + rhs.0[i];
        }
        Self(arr)
    }
}

impl<const N: usize> Mul<f64> for ArrayState<N> {
    type Output = Self;
    #[verified_engine::verified]
    fn mul(self, scalar: f64) -> Self {
        let mut arr = [0.0; N];
        for (i, val) in arr.iter_mut().enumerate() {
            *val = self.0[i] * scalar;
        }
        Self(arr)
    }
}

impl<const N: usize> AddAssign for ArrayState<N> {
    #[verified_engine::verified]
    fn add_assign(&mut self, rhs: Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a += b;
        }
    }
}

impl<const N: usize> MulAssign<f64> for ArrayState<N> {
    #[verified_engine::verified]
    fn mul_assign(&mut self, scalar: f64) {
        for val in self.0.iter_mut() {
            *val *= scalar;
        }
    }
}

impl<const N: usize> VectorOperations for ArrayState<N> {
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a += b * scale;
        }
    }

    #[verified_engine::verified]
    fn copy_from(&mut self, other: &Self) {
        self.0 = other.0;
    }

    #[verified_engine::verified]
    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        for ((dst, src), oth) in self.0.iter_mut().zip(source.0.iter()).zip(other.0.iter()) {
            *dst = *src + *oth * scale;
        }
    }
}

impl<const N: usize> OdeState for ArrayState<N> {
    #[verified_engine::verified]
    fn state_dim(&self) -> usize {
        N
    }

    #[verified_engine::verified]
    fn component(&self, idx: usize) -> f64 {
        self.0[idx]
    }

    #[verified_engine::verified]
    fn set_component(&mut self, idx: usize, value: f64) {
        self.0[idx] = value;
    }
}

/// Heap-allocated `Vec<f64>` state wrapper available when the `alloc` feature is enabled.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct VecState(pub Vec<f64>);

#[cfg(feature = "alloc")]
impl Add for VecState {
    type Output = Self;
    #[verified_engine::verified]
    fn add(mut self, rhs: Self) -> Self {
        let len = core::cmp::min(self.0.len(), rhs.0.len());
        self.0.truncate(len);
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a += b;
        }
        self
    }
}

#[cfg(feature = "alloc")]
impl AddAssign for VecState {
    #[verified_engine::verified]
    fn add_assign(&mut self, rhs: Self) {
        let len = core::cmp::min(self.0.len(), rhs.0.len());
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()).take(len) {
            *a += b;
        }
    }
}

#[cfg(feature = "alloc")]
impl Mul<f64> for VecState {
    type Output = Self;
    #[verified_engine::verified]
    fn mul(mut self, scalar: f64) -> Self {
        for val in self.0.iter_mut() {
            *val *= scalar;
        }
        self
    }
}

#[cfg(feature = "alloc")]
impl MulAssign<f64> for VecState {
    #[verified_engine::verified]
    fn mul_assign(&mut self, scalar: f64) {
        for val in self.0.iter_mut() {
            *val *= scalar;
        }
    }
}

#[cfg(feature = "alloc")]
impl VectorOperations for VecState {
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64) {
        let len = core::cmp::min(self.0.len(), other.0.len());
        for (a, b) in self.0.iter_mut().zip(other.0.iter()).take(len) {
            *a += b * scale;
        }
    }

    #[verified_engine::verified]
    fn copy_from(&mut self, other: &Self) {
        if self.0.len() != other.0.len() {
            self.0.resize(other.0.len(), 0.0);
        }
        self.0.copy_from_slice(&other.0);
    }

    #[verified_engine::verified]
    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        if self.0.len() != source.0.len() {
            self.0.resize(source.0.len(), 0.0);
        }
        for ((dst, src), oth) in self.0.iter_mut().zip(source.0.iter()).zip(other.0.iter()) {
            *dst = *src + *oth * scale;
        }
    }
}

#[cfg(feature = "alloc")]
impl OdeState for VecState {
    #[verified_engine::verified]
    fn state_dim(&self) -> usize {
        self.0.len()
    }

    #[verified_engine::verified]
    fn component(&self, idx: usize) -> f64 {
        self.0[idx]
    }

    #[verified_engine::verified]
    fn set_component(&mut self, idx: usize, value: f64) {
        self.0[idx] = value;
    }
}

#[cfg(feature = "nalgebra")]
impl VectorOperations for Vector2<f64> {
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64) {
        *self += other * scale;
    }

    #[verified_engine::verified]
    fn copy_from(&mut self, other: &Self) {
        *self = *other;
    }

    #[verified_engine::verified]
    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        *self = *source + *other * scale;
    }
}

#[cfg(feature = "nalgebra")]
impl OdeState for Vector2<f64> {
    #[verified_engine::verified]
    fn state_dim(&self) -> usize {
        2
    }

    #[verified_engine::verified]
    fn component(&self, idx: usize) -> f64 {
        self[idx]
    }

    #[verified_engine::verified]
    fn set_component(&mut self, idx: usize, value: f64) {
        self[idx] = value;
    }
}

#[cfg(feature = "nalgebra")]
impl VectorOperations for Vector3<f64> {
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64) {
        *self += other * scale;
    }

    #[verified_engine::verified]
    fn copy_from(&mut self, other: &Self) {
        *self = *other;
    }

    #[verified_engine::verified]
    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        *self = *source + *other * scale;
    }
}

#[cfg(feature = "nalgebra")]
impl OdeState for Vector3<f64> {
    #[verified_engine::verified]
    fn state_dim(&self) -> usize {
        3
    }

    #[verified_engine::verified]
    fn component(&self, idx: usize) -> f64 {
        self[idx]
    }

    #[verified_engine::verified]
    fn set_component(&mut self, idx: usize, value: f64) {
        self[idx] = value;
    }
}

#[cfg(feature = "nalgebra")]
impl VectorOperations for DVector<f64> {
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64) {
        for (a, b) in self.as_mut_slice().iter_mut().zip(other.as_slice().iter()) {
            *a += b * scale;
        }
    }

    #[verified_engine::verified(
        allow_recursion = "delegates trait copy_from to inherent Matrix::copy_from"
    )]
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other);
    }

    #[verified_engine::verified]
    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        if self.len() != source.len() {
            self.copy_from(source);
            self.scale_add(other, scale);
            return;
        }
        for ((dst, src), oth) in self
            .as_mut_slice()
            .iter_mut()
            .zip(source.as_slice().iter())
            .zip(other.as_slice().iter())
        {
            *dst = *src + *oth * scale;
        }
    }
}

#[cfg(feature = "nalgebra")]
impl OdeState for DVector<f64> {
    #[verified_engine::verified]
    fn state_dim(&self) -> usize {
        self.len()
    }

    #[verified_engine::verified]
    fn component(&self, idx: usize) -> f64 {
        self[idx]
    }

    #[verified_engine::verified]
    fn set_component(&mut self, idx: usize, value: f64) {
        self[idx] = value;
    }
}
