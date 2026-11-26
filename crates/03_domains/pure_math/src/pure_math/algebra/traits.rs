use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

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

/// A Ring is a set with two binary operations: addition (+) and multiplication (*).
/// 1. (R, +) is an abelian group.
/// 2. (R, *) is a monoid (associative, has identity 1).
/// 3. Multiplication distributes over addition.
pub trait Ring:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Neg<Output = Self>
    + Clone
    + PartialEq
    + Debug
    + Sized
{
    /// The additive identity (0).
    #[verified_engine::verified]
    fn zero() -> Self;

    /// The multiplicative identity (1).
    #[verified_engine::verified]
    fn one() -> Self;

    /// Returns `true` if `self` equals the additive identity [`Self::zero`].
    #[verified_engine::verified]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

/// A Ring that supports Euclidean division (division with remainder).
pub trait EuclideanDomain: Ring + Div<Output = Self> + Rem<Output = Self> {}

/// A Field is a Commutative Ring where every nonzero element has a multiplicative inverse.
/// This implies division is defined.
pub trait Field: Ring + Div<Output = Self> {
    /// Multiplicative inverse of the element.
    #[verified_engine::verified]
    fn multiplicative_inverse(&self) -> Self;
}

// Implementations for primitives

impl Ring for i64 {
    #[verified_engine::verified]
    fn zero() -> Self {
        0
    }
    #[verified_engine::verified]
    fn one() -> Self {
        1
    }
}

impl EuclideanDomain for i64 {}

impl Ring for i128 {
    #[verified_engine::verified]
    fn zero() -> Self {
        0
    }
    #[verified_engine::verified]
    fn one() -> Self {
        1
    }
}

impl EuclideanDomain for i128 {}

impl Ring for f64 {
    #[verified_engine::verified]
    fn zero() -> Self {
        0.0
    }
    #[verified_engine::verified]
    fn one() -> Self {
        1.0
    }
}

impl Field for f64 {
    #[verified_engine::verified]
    fn multiplicative_inverse(&self) -> Self {
        1.0 / self
    }
}
