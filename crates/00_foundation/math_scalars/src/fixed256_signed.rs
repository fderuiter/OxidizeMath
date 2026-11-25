#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
//! Stack-allocated two's-complement 256-bit signed integer [`I256`] with checked and
//! Euclidean arithmetic (`checked_div_euclid`, `checked_rem_euclid`) in `#![no_std]`.

use crate::error::ScalarError;
use crate::fixed256::U256;
use core::cmp::Ordering;
use core::fmt;

/// Stack-allocated two's-complement 256-bit signed integer ($[-2^{255}, 2^{255} - 1]$).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct I256(U256);

impl I256 {
    /// Additive identity $0$.
    pub const ZERO: Self = Self(U256::ZERO);
    /// Multiplicative identity $1$.
    pub const ONE: Self = Self(U256::ONE);
    /// Constant $-1$.
    pub const NEG_ONE: Self = Self(U256::MAX);
    /// Smallest representable signed 256-bit integer $-2^{255}$.
    pub const MIN: Self = Self(U256::from_limbs([0, 0, 0, 1_u64 << 63]));
    /// Largest representable signed 256-bit integer $2^{255} - 1$.
    pub const MAX: Self = Self(U256::from_limbs([
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX >> 1,
    ]));

    /// Constructs an `I256` from raw two's-complement little-endian limbs.
    #[inline]
    #[must_use]
    pub const fn from_raw_u256(bits: U256) -> Self {
        Self(bits)
    }

    /// Returns the raw two's-complement `U256` bit pattern.
    #[inline]
    #[must_use]
    pub const fn to_raw_u256(self) -> U256 {
        self.0
    }

    /// Sign-extends an `i64` into `I256`.
    #[inline]
    #[must_use]
    pub const fn from_i64(v: i64) -> Self {
        let sign = if v < 0 { u64::MAX } else { 0 };
        Self(U256::from_limbs([v as u64, sign, sign, sign]))
    }

    /// Sign-extends an `i128` into `I256`.
    #[inline]
    #[must_use]
    pub const fn from_i128(v: i128) -> Self {
        let sign = if v < 0 { u64::MAX } else { 0 };
        let u = v as u128;
        Self(U256::from_limbs([u as u64, (u >> 64) as u64, sign, sign]))
    }

    /// Returns `true` when `self == 0`.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0.is_zero()
    }

    /// Returns `true` when `self < 0`.
    #[inline]
    #[must_use]
    pub const fn is_negative(self) -> bool {
        (self.0.0[3] >> 63) == 1
    }

    /// Returns `true` when `self > 0`.
    #[inline]
    #[must_use]
    pub const fn is_positive(self) -> bool {
        !self.is_negative() && !self.is_zero()
    }

    /// Exact unsigned magnitude $|x| \in [0, 2^{255}]$ without overflow (including `Self::MIN`).
    #[inline]
    #[must_use]
    pub const fn unsigned_abs(self) -> U256 {
        if self.is_negative() {
            self.0.wrapping_neg()
        } else {
            self.0
        }
    }

    /// Checked negation returning `None` for [`Self::MIN`].
    #[inline]
    #[must_use]
    pub const fn checked_neg(self) -> Option<Self> {
        if self.eq_const(Self::MIN) {
            None
        } else {
            Some(Self(self.0.wrapping_neg()))
        }
    }

    /// Checked absolute value returning `None` for [`Self::MIN`].
    #[inline]
    #[must_use]
    pub const fn checked_abs(self) -> Option<Self> {
        if self.is_negative() {
            self.checked_neg()
        } else {
            Some(self)
        }
    }

    /// Checked signed addition returning `None` on signed overflow.
    #[must_use]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        let sum = Self(self.0.wrapping_add(rhs.0));
        let same_sign = self.is_negative() == rhs.is_negative();
        if same_sign && sum.is_negative() != self.is_negative() {
            None
        } else {
            Some(sum)
        }
    }

    /// Checked signed subtraction returning `None` on signed overflow.
    #[must_use]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        let diff = Self(self.0.wrapping_sub(rhs.0));
        let diff_sign = self.is_negative() != rhs.is_negative();
        if diff_sign && diff.is_negative() != self.is_negative() {
            None
        } else {
            Some(diff)
        }
    }

    /// Checked signed multiplication returning `None` on signed overflow.
    #[must_use]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        if self.is_zero() || rhs.is_zero() {
            return Some(Self::ZERO);
        }
        let neg = self.is_negative() != rhs.is_negative();
        let Some(mag) = self.unsigned_abs().checked_mul(rhs.unsigned_abs()) else {
            return None;
        };
        Self::from_sign_and_mag(neg, mag)
    }

    /// Checked truncating quotient `self / rhs` (`None` on zero divisor or `MIN / -1`).
    #[must_use]
    pub const fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() || (self.eq_const(Self::MIN) && rhs.eq_const(Self::NEG_ONE)) {
            return None;
        }
        let Some((q_mag, _)) = self.unsigned_abs().checked_div_rem(rhs.unsigned_abs()) else {
            return None;
        };
        Self::from_sign_and_mag(self.is_negative() != rhs.is_negative(), q_mag)
    }

    /// Checked truncating remainder `self % rhs` (`None` on zero divisor or `MIN % -1`).
    #[must_use]
    pub const fn checked_rem(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() || (self.eq_const(Self::MIN) && rhs.eq_const(Self::NEG_ONE)) {
            return None;
        }
        let Some((_, r_mag)) = self.unsigned_abs().checked_div_rem(rhs.unsigned_abs()) else {
            return None;
        };
        Self::from_sign_and_mag(self.is_negative(), r_mag)
    }

    /// Checked Euclidean quotient $q$ satisfying $a = q b + r$ with $0 \le r < |b|$.
    #[must_use]
    pub const fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
        let Some(q) = self.checked_div(rhs) else {
            return None;
        };
        let Some(r) = self.checked_rem(rhs) else {
            return None;
        };
        if r.is_negative() {
            if rhs.is_positive() {
                q.checked_sub(Self::ONE)
            } else {
                q.checked_add(Self::ONE)
            }
        } else {
            Some(q)
        }
    }

    /// Checked non-negative Euclidean remainder $r \in [0, |b|)$ (`None` only when `rhs == 0` or `MIN, -1`).
    #[must_use]
    pub const fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
        let Some(r) = self.checked_rem(rhs) else {
            return None;
        };
        if r.is_negative() {
            if rhs.is_positive() {
                r.checked_add(rhs)
            } else {
                r.checked_sub(rhs)
            }
        } else {
            Some(r)
        }
    }

    const fn from_sign_and_mag(neg: bool, mag: U256) -> Option<Self> {
        if mag.is_zero() {
            return Some(Self::ZERO);
        }
        if neg {
            if matches!(mag.cmp_const(Self::MIN.0), Ordering::Greater) {
                None
            } else {
                Some(Self(mag.wrapping_neg()))
            }
        } else if matches!(mag.cmp_const(Self::MAX.0), Ordering::Greater) {
            None
        } else {
            Some(Self(mag))
        }
    }

    const fn eq_const(self, rhs: Self) -> bool {
        self.0.cmp_const(rhs.0) as i8 == Ordering::Equal as i8
    }

    const fn cmp_const(self, rhs: Self) -> Ordering {
        let a = U256::from_limbs([
            self.0.0[0],
            self.0.0[1],
            self.0.0[2],
            self.0.0[3] ^ (1_u64 << 63),
        ]);
        let b = U256::from_limbs([
            rhs.0.0[0],
            rhs.0.0[1],
            rhs.0.0[2],
            rhs.0.0[3] ^ (1_u64 << 63),
        ]);
        a.cmp_const(b)
    }
}

impl Ord for I256 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_const(*other)
    }
}

impl PartialOrd for I256 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<i64> for I256 {
    #[inline]
    fn from(v: i64) -> Self {
        Self::from_i64(v)
    }
}

impl From<i128> for I256 {
    #[inline]
    fn from(v: i128) -> Self {
        Self::from_i128(v)
    }
}

impl TryFrom<U256> for I256 {
    type Error = ScalarError;
    fn try_from(v: U256) -> Result<Self, Self::Error> {
        if v <= Self::MAX.0 {
            Ok(Self(v))
        } else {
            Err(ScalarError::ArithmeticOverflow)
        }
    }
}

impl fmt::Display for I256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_negative() {
            f.write_str("-")?;
        }
        fmt::Display::fmt(&self.unsigned_abs(), f)
    }
}
