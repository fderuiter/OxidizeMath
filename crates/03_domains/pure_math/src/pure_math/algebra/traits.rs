use std::fmt::Debug;
use std::ops::{Div, Rem};

pub use math_scalars::{Field, RefOps, Ring, Scalar};

/// A set with an associative binary operation.
pub trait Semigroup: Clone + PartialEq + Debug {
    /// The binary operation (e.g., addition or multiplication).
    #[verified_engine::verified]
    fn operate(a: &Self, b: &Self) -> Self;
}

/// A Semigroup with an identity element.
pub trait Monoid: Semigroup {
    /// The identity element.
    #[verified_engine::verified]
    fn identity() -> Self;
}

/// A Monoid where every element has an inverse.
pub trait Group: Monoid {
    /// The inverse of the element.
    #[verified_engine::verified]
    fn inverse(&self) -> Self;
}

/// A Group where the operation is commutative.
pub trait AbelianGroup: Group {}

/// A Ring that supports Euclidean division (division with remainder).
pub trait EuclideanDomain: Ring + Div<Output = Self> + Rem<Output = Self> {}

impl EuclideanDomain for i32 {}
impl EuclideanDomain for i64 {}
impl EuclideanDomain for i128 {}
impl EuclideanDomain for isize {}
