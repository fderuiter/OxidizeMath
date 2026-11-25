#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:advanced_linear_algebra]
//! Stack-allocated 256-bit unsigned integer [`U256`] with checked, wrapping, and
//! Euclidean arithmetic available in `#![no_std]` without `alloc`.

use crate::error::ScalarError;
use core::cmp::Ordering;
use core::fmt;

/// Stack-allocated little-endian 256-bit unsigned integer (`limbs\[0\]` is least significant).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct U256(pub(crate) [u64; 4]);

impl U256 {
    /// Additive identity $0$.
    pub const ZERO: Self = Self([0; 4]);
    /// Multiplicative identity $1$.
    pub const ONE: Self = Self([1, 0, 0, 0]);
    /// Largest representable 256-bit unsigned integer $2^{256} - 1$.
    pub const MAX: Self = Self([u64::MAX; 4]);

    /// Constructs a `U256` from little-endian 64-bit limbs `[lo, l1, l2, hi]`.
    #[inline]
    #[must_use]
    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self(limbs)
    }

    /// Returns the underlying little-endian 64-bit limbs `[lo, l1, l2, hi]`.
    #[inline]
    #[must_use]
    pub const fn limbs(self) -> [u64; 4] {
        self.0
    }

    /// Widens a `u64` into `U256`.
    #[inline]
    #[must_use]
    pub const fn from_u64(v: u64) -> Self {
        Self([v, 0, 0, 0])
    }

    /// Widens a `u128` into `U256`.
    #[inline]
    #[must_use]
    pub const fn from_u128(v: u128) -> Self {
        Self([v as u64, (v >> 64) as u64, 0, 0])
    }

    /// Returns `true` when `self == 0`.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    /// Returns bit `idx` (`0..256`), or `false` if `idx >= 256`.
    #[inline]
    #[must_use]
    pub const fn bit(self, idx: usize) -> bool {
        if idx >= 256 {
            return false;
        }
        ((self.0[idx >> 6] >> (idx & 63)) & 1) == 1
    }

    /// Number of leading zero bits (`0..=256`).
    #[must_use]
    pub const fn leading_zeros(self) -> u32 {
        let mut i = 3;
        loop {
            if self.0[i] != 0 {
                return ((3 - i) as u32) * 64 + self.0[i].leading_zeros();
            }
            if i == 0 {
                return 256;
            }
            i -= 1;
        }
    }

    /// Minimum bit width required to represent `self` (`0` for zero).
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u32 {
        256 - self.leading_zeros()
    }

    /// Wrapping 256-bit addition returning `(sum, overflowed)`.
    #[must_use]
    pub const fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let mut out = [0_u64; 4];
        let mut carry = 0_u128;
        let mut i = 0;
        while i < 4 {
            let sum = (self.0[i] as u128) + (rhs.0[i] as u128) + carry;
            out[i] = sum as u64;
            carry = sum >> 64;
            i += 1;
        }
        (Self(out), carry != 0)
    }

    /// Checked 256-bit addition returning `None` on overflow.
    #[inline]
    #[must_use]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        let (sum, over) = self.overflowing_add(rhs);
        if over { None } else { Some(sum) }
    }

    /// Wrapping 256-bit addition modulo $2^{256}$.
    #[inline]
    #[must_use]
    pub const fn wrapping_add(self, rhs: Self) -> Self {
        self.overflowing_add(rhs).0
    }

    /// Wrapping 256-bit subtraction returning `(diff, borrowed)`.
    #[must_use]
    pub const fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let mut out = [0_u64; 4];
        let mut borrow = 0_u64;
        let mut i = 0;
        while i < 4 {
            let (d1, b1) = self.0[i].overflowing_sub(rhs.0[i]);
            let (d2, b2) = d1.overflowing_sub(borrow);
            out[i] = d2;
            borrow = (b1 as u64) + (b2 as u64);
            i += 1;
        }
        (Self(out), borrow != 0)
    }

    /// Checked 256-bit subtraction returning `None` on underflow.
    #[inline]
    #[must_use]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (diff, under) = self.overflowing_sub(rhs);
        if under { None } else { Some(diff) }
    }

    /// Wrapping 256-bit subtraction modulo $2^{256}$.
    #[inline]
    #[must_use]
    pub const fn wrapping_sub(self, rhs: Self) -> Self {
        self.overflowing_sub(rhs).0
    }

    /// Two's-complement negation modulo $2^{256}$.
    #[inline]
    #[must_use]
    pub const fn wrapping_neg(self) -> Self {
        Self([!self.0[0], !self.0[1], !self.0[2], !self.0[3]]).wrapping_add(Self::ONE)
    }

    /// Checked 256-bit multiplication returning `None` on overflow above $2^{256} - 1$.
    #[must_use]
    pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
        let mut acc = [0_u64; 4];
        let mut i = 0;
        while i < 4 {
            if self.0[i] != 0 {
                let mut carry = 0_u128;
                let mut j = 0;
                while j < 4 {
                    if i + j < 4 {
                        let prod =
                            (self.0[i] as u128) * (rhs.0[j] as u128) + (acc[i + j] as u128) + carry;
                        acc[i + j] = prod as u64;
                        carry = prod >> 64;
                    } else if rhs.0[j] != 0 {
                        return None;
                    }
                    j += 1;
                }
                if carry != 0 {
                    return None;
                }
            }
            i += 1;
        }
        Some(Self(acc))
    }

    /// Checked left shift by `bits` (`None` if `bits >= 256`).
    #[must_use]
    pub const fn checked_shl(self, bits: u32) -> Option<Self> {
        if bits >= 256 {
            return None;
        }
        if bits == 0 {
            return Some(self);
        }
        let word_shift = (bits / 64) as usize;
        let bit_shift = bits % 64;
        let mut out = [0_u64; 4];
        let mut i = word_shift;
        while i < 4 {
            out[i] = self.0[i - word_shift] << bit_shift;
            if bit_shift > 0 && i > word_shift {
                out[i] |= self.0[i - word_shift - 1] >> (64 - bit_shift);
            }
            i += 1;
        }
        Some(Self(out))
    }

    /// Checked right shift by `bits` (`None` if `bits >= 256`).
    #[must_use]
    pub const fn checked_shr(self, bits: u32) -> Option<Self> {
        if bits >= 256 {
            return None;
        }
        if bits == 0 {
            return Some(self);
        }
        let word_shift = (bits / 64) as usize;
        let bit_shift = bits % 64;
        let mut out = [0_u64; 4];
        let mut i = 0;
        while i + word_shift < 4 {
            out[i] = self.0[i + word_shift] >> bit_shift;
            if bit_shift > 0 && i + word_shift + 1 < 4 {
                out[i] |= self.0[i + word_shift + 1] << (64 - bit_shift);
            }
            i += 1;
        }
        Some(Self(out))
    }

    /// Checked quotient and remainder `(q, r)` with zero heap allocations (`None` if `rhs == 0`).
    #[must_use]
    pub const fn checked_div_rem(self, rhs: Self) -> Option<(Self, Self)> {
        if rhs.is_zero() {
            return None;
        }
        if matches!(self.cmp_const(rhs), Ordering::Less) {
            return Some((Self::ZERO, self));
        }
        let mut quot = [0_u64; 4];
        let mut rem = Self::ZERO;
        let mut bit = self.bits();
        while bit > 0 {
            bit -= 1;
            let shifted = match rem.checked_shl(1) {
                Some(v) => v,
                None => Self::ZERO,
            };
            let b = self.bit(bit as usize) as u64;
            rem = Self([shifted.0[0] | b, shifted.0[1], shifted.0[2], shifted.0[3]]);
            if !matches!(rem.cmp_const(rhs), Ordering::Less) {
                rem = rem.wrapping_sub(rhs);
                quot[(bit as usize) >> 6] |= 1_u64 << ((bit as usize) & 63);
            }
        }
        Some((Self(quot), rem))
    }

    /// Checked quotient `self / rhs`.
    #[inline]
    #[must_use]
    pub const fn checked_div(self, rhs: Self) -> Option<Self> {
        match self.checked_div_rem(rhs) {
            Some((q, _)) => Some(q),
            None => None,
        }
    }

    /// Checked remainder `self % rhs`.
    #[inline]
    #[must_use]
    pub const fn checked_rem(self, rhs: Self) -> Option<Self> {
        match self.checked_div_rem(rhs) {
            Some((_, r)) => Some(r),
            None => None,
        }
    }

    /// Checked Euclidean quotient (identical to [`Self::checked_div`] for unsigned integers).
    #[inline]
    #[must_use]
    pub const fn checked_div_euclid(self, rhs: Self) -> Option<Self> {
        self.checked_div(rhs)
    }

    /// Checked Euclidean remainder (identical to [`Self::checked_rem`] for unsigned integers).
    #[inline]
    #[must_use]
    pub const fn checked_rem_euclid(self, rhs: Self) -> Option<Self> {
        self.checked_rem(rhs)
    }

    /// Approximates `self` as `f64`.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        let two64 = 18_446_744_073_709_551_616.0_f64;
        (((self.0[3] as f64) * two64 + (self.0[2] as f64)) * two64 + (self.0[1] as f64)) * two64
            + (self.0[0] as f64)
    }

    pub(crate) const fn cmp_const(self, rhs: Self) -> Ordering {
        let mut i = 3;
        loop {
            if self.0[i] < rhs.0[i] {
                return Ordering::Less;
            }
            if self.0[i] > rhs.0[i] {
                return Ordering::Greater;
            }
            if i == 0 {
                return Ordering::Equal;
            }
            i -= 1;
        }
    }
}

impl Ord for U256 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_const(*other)
    }
}

impl PartialOrd for U256 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<u64> for U256 {
    #[inline]
    fn from(v: u64) -> Self {
        Self::from_u64(v)
    }
}

impl From<u128> for U256 {
    #[inline]
    fn from(v: u128) -> Self {
        Self::from_u128(v)
    }
}

impl TryFrom<U256> for u64 {
    type Error = ScalarError;
    fn try_from(v: U256) -> Result<Self, Self::Error> {
        if (v.0[1] | v.0[2] | v.0[3]) == 0 {
            Ok(v.0[0])
        } else {
            Err(ScalarError::ArithmeticOverflow)
        }
    }
}

impl TryFrom<U256> for u128 {
    type Error = ScalarError;
    fn try_from(v: U256) -> Result<Self, Self::Error> {
        if (v.0[2] | v.0[3]) == 0 {
            Ok((v.0[0] as u128) | ((v.0[1] as u128) << 64))
        } else {
            Err(ScalarError::ArithmeticOverflow)
        }
    }
}

impl fmt::Display for U256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let mut buf = [0_u8; 78];
        let mut pos = buf.len();
        let mut cur = *self;
        let ten = Self::from_u64(10);
        while !cur.is_zero() {
            let Some((q, r)) = cur.checked_div_rem(ten) else {
                break;
            };
            pos -= 1;
            buf[pos] = b'0' + (r.0[0] as u8);
            cur = q;
        }
        let Ok(s) = core::str::from_utf8(&buf[pos..]) else {
            return Err(fmt::Error);
        };
        f.write_str(s)
    }
}
