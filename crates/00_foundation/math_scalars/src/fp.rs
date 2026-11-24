#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
pub use crate::modular::is_prime_u64;
use crate::modular::{add_mod_u64, inv_mod_u64, sub_mod_u64, tonelli_shanks_u64};
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Element of the prime finite field $\mathbb{F}_P = \mathbb{Z}/P\mathbb{Z}$ for a compile-time
/// validated 64-bit prime $P \le 18{,}446{,}744{,}073{,}709{,}551{,}557$.
///
/// - Composite moduli ($P < 2$ or composite $P$) are rejected **at compile time** via `const`
///   deterministic 64-bit Miller-Rabin evaluation.
/// - Odd primes $P \ge 3$ store Montgomery residues $\bar{a} = a \cdot 2^{64} \bmod P$ and
///   multiply via split-carry 128-bit Montgomery reduction (`REDC`) that never overflows even when
///   $P > 2^{63}$ under `overflow-checks = true`.
/// - Even prime $P = 2$ uses an explicit constant-folded $\mathbb{F}_2$ path (`^` and `&`).
///
/// ```compile_fail
/// use math_scalars::Fp;
/// // Composite moduli fail to compile:
/// let _ = Fp::<15>::new(1);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp<const P: u64> {
    mont: u64,
}

const fn compute_n_prime(p: u64) -> u64 {
    if (p & 1) == 0 {
        return 0;
    }
    let mut inv = 1u64;
    let mut i = 0;
    while i < 6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(p.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg()
}

const fn compute_r_mod_p(p: u64) -> u64 {
    if p <= 2 {
        return 0;
    }
    ((1u128 << 64) % (p as u128)) as u64
}

const fn compute_r2_mod_p(p: u64) -> u64 {
    if p <= 2 {
        return 0;
    }
    let r = compute_r_mod_p(p) as u128;
    ((r * r) % (p as u128)) as u64
}

#[inline]
const fn mont_reduce(t: u128, p: u64, n_prime: u64) -> u64 {
    assert!(p >= 3);
    let t_lo = t as u64;
    let m = t_lo.wrapping_mul(n_prime);
    let mp = (m as u128) * (p as u128);
    let carry_lo = ((t_lo as u128) + ((mp as u64) as u128)) >> 64;
    let u = (t >> 64) + (mp >> 64) + carry_lo;
    let out = if u >= (p as u128) {
        (u - (p as u128)) as u64
    } else {
        u as u64
    };
    assert!(out < p);
    out
}

impl<const P: u64> Fp<P> {
    /// Compile-time primality proof for `P`.
    pub const VALID_PRIME: () = assert!(is_prime_u64(P), "Fp<P> modulus P must be a prime integer");

    const N_PRIME: u64 = compute_n_prime(P);
    const R_MOD_P: u64 = compute_r_mod_p(P);
    const R2_MOD_P: u64 = compute_r2_mod_p(P);

    /// Additive identity $0 \in \mathbb{F}_P$.
    pub const ZERO: Self = {
        let () = Self::VALID_PRIME;
        Self { mont: 0 }
    };

    /// Multiplicative identity $1 \in \mathbb{F}_P$.
    pub const ONE: Self = {
        let () = Self::VALID_PRIME;
        if P == 2 {
            Self { mont: 1 }
        } else {
            Self {
                mont: Self::R_MOD_P,
            }
        }
    };

    /// Constructs the prime modulus $P$.
    /// `[cite:math_scalars]`
    #[inline]
    #[must_use]
    pub const fn modulus() -> u64 {
        let () = Self::VALID_PRIME;
        assert!(P >= 2);
        P
    }

    /// Reduces an unsigned integer `val` into $\mathbb{F}_P$.
    /// `[cite:math_scalars]`
    #[inline]
    #[must_use]
    pub const fn new(val: u64) -> Self {
        let () = Self::VALID_PRIME;
        let rem = val % P;
        let out = if P == 2 {
            Self { mont: rem }
        } else {
            Self {
                mont: mont_reduce((rem as u128) * (Self::R2_MOD_P as u128), P, Self::N_PRIME),
            }
        };
        assert!(out.mont < P);
        out
    }

    /// Reduces a signed `i64` integer into $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn from_i64(val: i64) -> Self {
        let () = Self::VALID_PRIME;
        if val >= 0 {
            Self::new(val as u64)
        } else {
            let mag = val.unsigned_abs() % P;
            if mag == 0 {
                Self::ZERO
            } else {
                Self::new(P - mag)
            }
        }
    }

    /// Reduces a signed `i128` integer into $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn from_i128(val: i128) -> Self {
        let () = Self::VALID_PRIME;
        let p128 = P as u128;
        if val >= 0 {
            Self::new(((val as u128) % p128) as u64)
        } else {
            let mag = (val.unsigned_abs() % p128) as u64;
            if mag == 0 {
                Self::ZERO
            } else {
                Self::new(P - mag)
            }
        }
    }

    /// Converts `self` out of Montgomery form into its canonical residue in `0..P`.
    #[inline]
    #[must_use]
    pub const fn value(self) -> u64 {
        let () = Self::VALID_PRIME;
        if P == 2 {
            self.mont & 1
        } else {
            mont_reduce(self.mont as u128, P, Self::N_PRIME)
        }
    }

    /// Returns `true` when `self == 0`.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.mont == 0
    }

    /// Constant-friendly addition in $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn const_add(self, rhs: Self) -> Self {
        let () = Self::VALID_PRIME;
        if P == 2 {
            Self {
                mont: (self.mont ^ rhs.mont) & 1,
            }
        } else {
            Self {
                mont: add_mod_u64(self.mont, rhs.mont, P),
            }
        }
    }

    /// Constant-friendly subtraction in $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn const_sub(self, rhs: Self) -> Self {
        let () = Self::VALID_PRIME;
        if P == 2 {
            Self {
                mont: (self.mont ^ rhs.mont) & 1,
            }
        } else {
            Self {
                mont: sub_mod_u64(self.mont, rhs.mont, P),
            }
        }
    }

    /// Constant-friendly negation in $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn const_neg(self) -> Self {
        let () = Self::VALID_PRIME;
        if self.mont == 0 || P == 2 {
            self
        } else {
            Self {
                mont: P - self.mont,
            }
        }
    }

    /// Constant-friendly Montgomery multiplication in $\mathbb{F}_P$.
    #[inline]
    #[must_use]
    pub const fn const_mul(self, rhs: Self) -> Self {
        let () = Self::VALID_PRIME;
        if P == 2 {
            Self {
                mont: (self.mont & rhs.mont) & 1,
            }
        } else {
            Self {
                mont: mont_reduce((self.mont as u128) * (rhs.mont as u128), P, Self::N_PRIME),
            }
        }
    }

    /// Binary exponentiation `self^exp` in $\mathbb{F}_P$.
    #[must_use]
    pub const fn pow(self, mut exp: u64) -> Self {
        let () = Self::VALID_PRIME;
        let mut acc = Self::ONE;
        let mut base = self;
        while exp > 0 {
            if (exp & 1) == 1 {
                acc = acc.const_mul(base);
            }
            base = base.const_mul(base);
            exp >>= 1;
        }
        acc
    }

    /// Fallible multiplicative inverse $a^{-1} \in \mathbb{F}_P^\times$.
    ///
    /// # Errors
    /// Returns [`ScalarError::NonInvertible`] when `self == 0`.
    pub const fn checked_inverse(self) -> Result<Self, ScalarError> {
        let () = Self::VALID_PRIME;
        if self.mont == 0 {
            return Err(ScalarError::NonInvertible);
        }
        if P == 2 {
            return Ok(Self { mont: 1 });
        }
        match inv_mod_u64(self.value(), P) {
            Some(inv) => Ok(Self::new(inv)),
            None => Err(ScalarError::NonInvertible),
        }
    }

    /// Fallible division `self * rhs.checked_inverse()?`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] when `rhs == 0`.
    pub const fn checked_div(self, rhs: Self) -> Result<Self, ScalarError> {
        match rhs.checked_inverse() {
            Ok(inv) => Ok(self.const_mul(inv)),
            Err(_) => Err(ScalarError::DivisionByZero),
        }
    }

    /// Computes both square roots `[r0, r1]` ($r_0.\text{value}() \le r_1.\text{value}()$) when `self`
    /// is a square in $\mathbb{F}_P$, or `None` if `self` is a quadratic non-residue.
    #[must_use]
    pub const fn sqrt(self) -> Option<[Self; 2]> {
        let () = Self::VALID_PRIME;
        match tonelli_shanks_u64(self.value(), P) {
            Some([r0, r1]) => Some([Self::new(r0), Self::new(r1)]),
            None => None,
        }
    }
}

impl<const P: u64> Default for Fp<P> {
    #[inline]
    fn default() -> Self {
        Self::ZERO
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

impl<const P: u64> From<u64> for Fp<P> {
    #[inline]
    fn from(val: u64) -> Self {
        Self::new(val)
    }
}

impl<const P: u64> Neg for Fp<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        self.const_neg()
    }
}

impl<const P: u64> Add for Fp<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.const_add(rhs)
    }
}

impl<const P: u64> AddAssign for Fp<P> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.const_add(rhs);
    }
}

impl<const P: u64> Sub for Fp<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.const_sub(rhs)
    }
}

impl<const P: u64> SubAssign for Fp<P> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.const_sub(rhs);
    }
}

impl<const P: u64> Mul for Fp<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.const_mul(rhs)
    }
}

impl<const P: u64> MulAssign for Fp<P> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.const_mul(rhs);
    }
}

impl<const P: u64> Div for Fp<P> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs).expect("Fp division by zero")
    }
}

impl<const P: u64> DivAssign for Fp<P> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
