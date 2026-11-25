#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:modular_polynomials_review]
//! Compile-time prime-validated finite field $\mathbb{F}_P$ (`Fp<const P: u64>`) backed by
//! carry-safe 128-bit Montgomery multiplication for odd primes up to $2^{64} - 59$ and an
//! explicit mod-2 branch for $P = 2$.

use crate::error::ScalarError;
use crate::modular::{
    MontgomeryParams, is_prime_u64, mod_add_u64, mod_inv_u64, mod_neg_u64, mod_reduce_i64,
    mod_reduce_i128, mod_sub_u64, tonelli_shanks_u64,
};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Element of the prime Galois field $\mathbb{F}_P$ for compile-time validated 64-bit prime $P$.
///
/// Composite moduli (such as `0`, `1`, `4`, `9`, `15`) are rejected at compile time:
///
/// ```compile_fail
/// use math_scalars::Fp;
/// let _bad = Fp::<9>::new(2);
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Fp<const P: u64> {
    mont: u64,
}

impl<const P: u64> Default for Fp<P> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const P: u64> Fp<P> {
    /// Compile-time primality check evaluated on every `Fp<P>` construction.
    pub const VALIDATE_PRIME: () = assert!(is_prime_u64(P), "Fp<P> modulus P must be prime");

    const MONT: MontgomeryParams = {
        let () = Self::VALIDATE_PRIME;
        if P == 2 {
            MontgomeryParams {
                p: 2,
                n_prime: 0,
                r_mod_p: 1,
                r2_mod_p: 1,
            }
        } else {
            MontgomeryParams::new_odd(P)
        }
    };

    /// Prime characteristic $P$.
    pub const MODULUS: u64 = {
        let () = Self::VALIDATE_PRIME;
        P
    };

    /// Additive identity $0 \in \mathbb{F}_P$.
    pub const ZERO: Self = {
        let () = Self::VALIDATE_PRIME;
        Self { mont: 0 }
    };

    /// Multiplicative identity $1 \in \mathbb{F}_P$.
    pub const ONE: Self = {
        let () = Self::VALIDATE_PRIME;
        if P == 2 {
            Self { mont: 1 }
        } else {
            Self {
                mont: Self::MONT.r_mod_p,
            }
        }
    };

    /// Reduces signed `val` into $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn new(val: i64) -> Self {
        let () = Self::VALIDATE_PRIME;
        Self::from_canonical(mod_reduce_i64(val, P))
    }

    /// Reduces unsigned `val` into $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn from_u64(val: u64) -> Self {
        let () = Self::VALIDATE_PRIME;
        Self::from_canonical(val % P)
    }

    /// Reduces signed 128-bit `val` into $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn from_i128(val: i128) -> Self {
        let () = Self::VALIDATE_PRIME;
        Self::from_canonical(mod_reduce_i128(val, P))
    }

    #[inline]
    const fn from_canonical(c: u64) -> Self {
        if P == 2 {
            Self { mont: c & 1 }
        } else {
            Self {
                mont: Self::MONT.to_mont(c),
            }
        }
    }

    /// Decodes `self` into its canonical least non-negative residue in `[0, P - 1]`.
    #[inline]
    #[must_use]
    pub const fn value(self) -> u64 {
        let () = Self::VALIDATE_PRIME;
        if P == 2 {
            self.mont & 1
        } else {
            Self::MONT.decode_mont(self.mont)
        }
    }

    /// Returns `true` iff `self == 0`.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.mont == 0
    }

    /// Carry-safe addition in $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn add_const(self, rhs: Self) -> Self {
        if P == 2 {
            Self {
                mont: (self.mont ^ rhs.mont) & 1,
            }
        } else {
            Self {
                mont: mod_add_u64(self.mont, rhs.mont, P),
            }
        }
    }

    /// Carry-safe subtraction in $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn sub_const(self, rhs: Self) -> Self {
        if P == 2 {
            Self {
                mont: (self.mont ^ rhs.mont) & 1,
            }
        } else {
            Self {
                mont: mod_sub_u64(self.mont, rhs.mont, P),
            }
        }
    }

    /// Carry-safe negation in $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn neg_const(self) -> Self {
        if P == 2 {
            self
        } else {
            Self {
                mont: mod_neg_u64(self.mont, P),
            }
        }
    }

    /// Carry-safe Montgomery multiplication in $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn mul_const(self, rhs: Self) -> Self {
        if P == 2 {
            Self {
                mont: (self.mont & rhs.mont) & 1,
            }
        } else {
            Self {
                mont: Self::MONT.mul_mont(self.mont, rhs.mont),
            }
        }
    }

    /// Binary exponentiation $a^e \in \mathbb{F}_P$ in Montgomery form.
    #[must_use]
    pub const fn pow(self, mut exp: u64) -> Self {
        let mut acc = Self::ONE;
        let mut cur = self;
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc.mul_const(cur);
            }
            exp >>= 1;
            if exp > 0 {
                cur = cur.mul_const(cur);
            }
        }
        acc
    }

    /// Fallible multiplicative inverse returning `None` iff `self == 0`.
    #[must_use]
    pub const fn checked_inv(self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        if P == 2 {
            return Some(Self::ONE);
        }
        match mod_inv_u64(self.value(), P) {
            Some(inv) => Some(Self::from_canonical(inv)),
            None => None,
        }
    }

    /// Fallible multiplicative inverse returning [`ScalarError::DivisionByZero`] when `self == 0`.
    pub const fn try_inverse(self) -> Result<Self, ScalarError> {
        match self.checked_inv() {
            Some(inv) => Ok(inv),
            None => Err(ScalarError::DivisionByZero),
        }
    }

    /// Fallible field division `self * rhs^-1`.
    pub const fn checked_div(self, rhs: Self) -> Result<Self, ScalarError> {
        match rhs.try_inverse() {
            Ok(inv) => Ok(self.mul_const(inv)),
            Err(e) => Err(e),
        }
    }

    /// Legendre symbol $(a / P) \in \{-1, 0, 1\}$.
    #[must_use]
    pub const fn legendre_symbol(self) -> i8 {
        if self.is_zero() {
            return 0;
        }
        if P == 2 {
            return 1;
        }
        let e = self.pow((P - 1) / 2).value();
        if e == 1 { 1 } else { -1 }
    }

    /// Solves $y^2 = \text{self}$ in $\mathbb{F}_P$, returning sorted canonical roots `([r0, r1], count)`.
    #[must_use]
    pub const fn sqrt_roots(self) -> ([Self; 2], usize) {
        let (roots, count) = tonelli_shanks_u64(self.value(), P);
        (
            [
                Self::from_canonical(roots[0]),
                Self::from_canonical(roots[1]),
            ],
            count,
        )
    }
}

impl<const P: u64> fmt::Debug for Fp<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp::<{P}>({})", self.value())
    }
}

impl<const P: u64> fmt::Display for Fp<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value(), P)
    }
}

impl<const P: u64> Add for Fp<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.add_const(rhs)
    }
}

impl<const P: u64> AddAssign for Fp<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_const(rhs);
    }
}

impl<const P: u64> Sub for Fp<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub_const(rhs)
    }
}

impl<const P: u64> SubAssign for Fp<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub_const(rhs);
    }
}

impl<const P: u64> Mul for Fp<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_const(rhs)
    }
}

impl<const P: u64> MulAssign for Fp<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_const(rhs);
    }
}

impl<const P: u64> Neg for Fp<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        self.neg_const()
    }
}

impl<const P: u64> Div for Fp<P> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let inv = rhs
            .checked_inv()
            .expect("Fp division by zero; use checked_div or try_inverse for fallible inversion");
        self.mul_const(inv)
    }
}

impl<const P: u64> DivAssign for Fp<P> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
