//! # Polynomial Ring
//!
//! Implementation of polynomials with coefficients in a Ring.

use crate::pure_math::algebra::traits::Ring;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A polynomial with coefficients in a Ring T.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Polynomial<T: Ring> {
    #[allow(missing_docs)]
    pub coeffs: Vec<T>, // coeffs[i] is coefficient of x^i
}

impl<T: Ring> Default for Polynomial<T> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: Ring> Polynomial<T> {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(coeffs: Vec<T>) -> Self {
        let mut p = Polynomial { coeffs };
        p.trim();
        p
    }

    /// Removes trailing zeros.
    #[verified_engine::verified]
    fn trim(&mut self) {
        while self.coeffs.len() > 1 && self.coeffs.last().is_some_and(|c| c.is_zero()) {
            self.coeffs.pop();
        }
        // Ensure at least one element (0) exists if empty?
        // Or representation: empty vec = 0?
        // Let's say empty vec = 0.
        if self.coeffs.len() == 1 && self.coeffs[0].is_zero() {
            self.coeffs.pop();
        }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None // Degree of zero polynomial is -infinity (represented as None)
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Evaluation at x using Horner's method.
    #[verified_engine::verified]
    pub fn eval(&self, x: T) -> T {
        if self.coeffs.is_empty() {
            return T::zero();
        }
        let mut result = T::zero();
        for c in self.coeffs.iter().rev() {
            result = result * x.clone() + c.clone();
        }
        result
    }

    /// Scale polynomial by a scalar.
    #[verified_engine::verified]
    pub fn scale(&self, scalar: &T) -> Self {
        if self.coeffs.is_empty() || scalar.is_zero() {
            return Polynomial::new(vec![]);
        }
        let mut new_coeffs = Vec::with_capacity(self.coeffs.len());
        for c in &self.coeffs {
            new_coeffs.push(c.clone() * scalar.clone());
        }
        Polynomial::new(new_coeffs)
    }
}

impl<T: Ring> Add for Polynomial<T> {
    type Output = Self;
    #[verified_engine::verified]
    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl<'b, T: Ring> Add<&'b Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn add(self, other: &'b Polynomial<T>) -> Polynomial<T> {
        let mut res = self.clone();
        res += other;
        res
    }
}

impl<'b, T: Ring> Add<&'b Polynomial<T>> for Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn add(mut self, other: &'b Polynomial<T>) -> Polynomial<T> {
        self += other;
        self
    }
}

impl<T: Ring> Add<Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn add(self, mut other: Polynomial<T>) -> Polynomial<T> {
        other += self;
        other
    }
}

impl<T: Ring> Mul for Polynomial<T> {
    type Output = Self;
    #[verified_engine::verified]
    fn mul(self, other: Self) -> Self {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Polynomial::new(vec![]);
        }
        let deg1 = self.coeffs.len() - 1;
        let deg2 = other.coeffs.len() - 1;
        let mut new_coeffs = vec![T::zero(); deg1 + deg2 + 1];
        let other_len = other.coeffs.len();

        for (i, c1) in self.coeffs.into_iter().enumerate() {
            for (j, c2) in other.coeffs[..other_len - 1].iter().enumerate() {
                new_coeffs[i + j] += c1.clone() * c2.clone();
            }
            if let Some(c2) = other.coeffs.last() {
                new_coeffs[i + other_len - 1] += c1 * c2.clone();
            }
        }
        Polynomial::new(new_coeffs)
    }
}

impl<'b, T: Ring> Mul<&'b Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn mul(self, other: &'b Polynomial<T>) -> Polynomial<T> {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Polynomial::new(vec![]);
        }
        let deg1 = self.coeffs.len() - 1;
        let deg2 = other.coeffs.len() - 1;
        let mut new_coeffs = vec![T::zero(); deg1 + deg2 + 1];

        for (i, c1) in self.coeffs.iter().enumerate() {
            for (j, c2) in other.coeffs.iter().enumerate() {
                new_coeffs[i + j] += c1.clone() * c2.clone();
            }
        }
        Polynomial::new(new_coeffs)
    }
}

impl<'b, T: Ring> Mul<&'b Polynomial<T>> for Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn mul(self, other: &'b Polynomial<T>) -> Polynomial<T> {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Polynomial::new(vec![]);
        }
        let deg1 = self.coeffs.len() - 1;
        let deg2 = other.coeffs.len() - 1;
        let mut new_coeffs = vec![T::zero(); deg1 + deg2 + 1];
        let other_len = other.coeffs.len();

        for (i, c1) in self.coeffs.into_iter().enumerate() {
            for (j, c2) in other.coeffs[..other_len - 1].iter().enumerate() {
                new_coeffs[i + j] += c1.clone() * c2.clone();
            }
            if let Some(c2) = other.coeffs.last() {
                new_coeffs[i + other_len - 1] += c1 * c2.clone();
            }
        }
        Polynomial::new(new_coeffs)
    }
}

impl<T: Ring> Mul<Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn mul(self, other: Polynomial<T>) -> Polynomial<T> {
        self * &other
    }
}

impl<T: Ring> Mul<T> for Polynomial<T> {
    type Output = Self;
    #[verified_engine::verified]
    fn mul(mut self, scalar: T) -> Self {
        if self.coeffs.is_empty() || scalar.is_zero() {
            return Polynomial::new(vec![]);
        }
        let len = self.coeffs.len();
        let mut new_coeffs = Vec::with_capacity(len);
        let last = self.coeffs.pop();
        for c in self.coeffs {
            new_coeffs.push(c * scalar.clone());
        }
        if let Some(c) = last {
            new_coeffs.push(c * scalar);
        }
        Polynomial::new(new_coeffs)
    }
}

impl<'b, T: Ring> Mul<&'b T> for Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn mul(self, scalar: &'b T) -> Polynomial<T> {
        self * scalar.clone()
    }
}

impl<'b, T: Ring> Mul<&'b T> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn mul(self, scalar: &'b T) -> Polynomial<T> {
        self.scale(scalar)
    }
}

impl<T: Ring> Mul<T> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn mul(self, scalar: T) -> Polynomial<T> {
        self.scale(&scalar)
    }
}

// Implement other Ring traits for Polynomial (omitted for brevity unless strictly needed by generic Ring bounds)
// Since we want Polynomial<T> to be a Ring, we must implement AddAssign, Sub, SubAssign, MulAssign, Neg, One, Zero.
// To satisfy the generic constraints for testing, I'll implement a subset or all if possible.
// For now, let's stick to basic operations demonstrated above. The user can use basic operators.
// Note: To implement `Ring` trait for `Polynomial<T>`, we need all of them.

impl<T: Ring> Ring for Polynomial<T> {
    #[verified_engine::verified]
    fn zero() -> Self {
        Polynomial::new(vec![])
    }
    #[verified_engine::verified]
    fn one() -> Self {
        Polynomial::new(vec![T::one()])
    }
}

impl<T: Ring> AddAssign for Polynomial<T> {
    #[verified_engine::verified]
    fn add_assign(&mut self, other: Self) {
        if self.coeffs.len() < other.coeffs.len() {
            self.coeffs.resize(other.coeffs.len(), T::zero());
        }
        for (i, c) in other.coeffs.into_iter().enumerate() {
            self.coeffs[i] += c;
        }
        self.trim();
    }
}

impl<T: Ring> AddAssign<&Polynomial<T>> for Polynomial<T> {
    #[verified_engine::verified]
    fn add_assign(&mut self, other: &Self) {
        if self.coeffs.len() < other.coeffs.len() {
            self.coeffs.resize(other.coeffs.len(), T::zero());
        }
        for (i, c) in other.coeffs.iter().enumerate() {
            self.coeffs[i] += c.clone();
        }
        self.trim();
    }
}

impl<T: Ring> Sub for Polynomial<T> {
    type Output = Self;
    #[verified_engine::verified]
    fn sub(mut self, other: Self) -> Self {
        self -= other;
        self
    }
}

impl<'b, T: Ring> Sub<&'b Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn sub(self, other: &'b Polynomial<T>) -> Polynomial<T> {
        let mut res = self.clone();
        res -= other;
        res
    }
}

impl<'b, T: Ring> Sub<&'b Polynomial<T>> for Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn sub(mut self, other: &'b Polynomial<T>) -> Polynomial<T> {
        self -= other;
        self
    }
}

impl<T: Ring> Sub<Polynomial<T>> for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn sub(self, other: Polynomial<T>) -> Polynomial<T> {
        self - &other
    }
}

impl<T: Ring> SubAssign for Polynomial<T> {
    #[verified_engine::verified]
    fn sub_assign(&mut self, other: Self) {
        if self.coeffs.len() < other.coeffs.len() {
            self.coeffs.resize(other.coeffs.len(), T::zero());
        }
        for (i, c) in other.coeffs.into_iter().enumerate() {
            self.coeffs[i] -= c;
        }
        self.trim();
    }
}

impl<T: Ring> SubAssign<&Polynomial<T>> for Polynomial<T> {
    #[verified_engine::verified]
    fn sub_assign(&mut self, other: &Self) {
        if self.coeffs.len() < other.coeffs.len() {
            self.coeffs.resize(other.coeffs.len(), T::zero());
        }
        for (i, c) in other.coeffs.iter().enumerate() {
            self.coeffs[i] -= c.clone();
        }
        self.trim();
    }
}

impl<T: Ring> MulAssign for Polynomial<T> {
    #[verified_engine::verified]
    fn mul_assign(&mut self, other: Self) {
        *self = std::mem::take(self) * other;
    }
}

impl<T: Ring> MulAssign<&Polynomial<T>> for Polynomial<T> {
    #[verified_engine::verified]
    fn mul_assign(&mut self, other: &Self) {
        *self = std::mem::take(self) * other;
    }
}

impl<T: Ring> Neg for Polynomial<T> {
    type Output = Self;
    #[verified_engine::verified]
    fn neg(mut self) -> Self {
        for c in &mut self.coeffs {
            *c = -std::mem::replace(c, T::zero());
        }
        self
    }
}

impl<T: Ring> Neg for &Polynomial<T> {
    type Output = Polynomial<T>;
    #[verified_engine::verified]
    fn neg(self) -> Polynomial<T> {
        Polynomial::new(self.coeffs.iter().map(|c| -c.clone()).collect())
    }
}
