#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:math_scalars]

use crate::error::ScalarError;
use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Rem, Shl, Shr, Sub, SubAssign};

/// Stack-allocated 256-bit unsigned integer stored as four little-endian `u64` limbs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct U256(pub [u64; 4]);

impl U256 {
    /// Additive identity $0$.
    pub const ZERO: Self = Self([0, 0, 0, 0]);
    /// Multiplicative identity $1$.
    pub const ONE: Self = Self([1, 0, 0, 0]);
    /// Maximum representable 256-bit unsigned integer $2^{256} - 1$.
    pub const MAX: Self = Self([u64::MAX; 4]);

    /// Constructs a `U256` from little-endian 64-bit limbs `[l0, l1, l2, l3]`.
    #[inline]
    #[must_use]
    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self(limbs)
    }

    /// Returns the underlying little-endian 64-bit limbs.
    #[inline]
    #[must_use]
    pub const fn limbs(self) -> [u64; 4] {
        self.0
    }

    /// Constructs a `U256` from a `u64`.
    #[inline]
    #[must_use]
    pub const fn from_u64(v: u64) -> Self {
        Self([v, 0, 0, 0])
    }

    /// Constructs a `U256` from a `u128`.
    #[inline]
    #[must_use]
    pub const fn from_u128(v: u128) -> Self {
        Self([v as u64, (v >> 64) as u64, 0, 0])
    }

    /// Converts to `u64` if the value fits in 64 bits.
    #[inline]
    #[must_use]
    pub const fn to_u64(self) -> Option<u64> {
        if self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0 {
            Some(self.0[0])
        } else {
            None
        }
    }

    /// Converts to `u128` if the value fits in 128 bits.
    #[inline]
    #[must_use]
    pub const fn to_u128(self) -> Option<u128> {
        if self.0[2] == 0 && self.0[3] == 0 {
            Some((self.0[0] as u128) | ((self.0[1] as u128) << 64))
        } else {
            None
        }
    }

    /// Returns `true` when all four limbs are zero.
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
    }

    /// Returns the bit at `index` (`0..256`), or `false` if `index >= 256`.
    #[inline]
    #[must_use]
    pub const fn bit(self, index: usize) -> bool {
        if index >= 256 {
            return false;
        }
        ((self.0[index / 64] >> (index % 64)) & 1) != 0
    }

    /// Sets bit `index` (`0..256`) to `1`.
    #[inline]
    pub const fn set_bit(&mut self, index: usize) {
        if index < 256 {
            self.0[index / 64] |= 1u64 << (index % 64);
        }
    }

    /// Number of leading zero bits in `[0, 256]`.
    #[must_use]
    pub const fn leading_zeros(self) -> u32 {
        if self.0[3] != 0 {
            self.0[3].leading_zeros()
        } else if self.0[2] != 0 {
            64 + self.0[2].leading_zeros()
        } else if self.0[1] != 0 {
            128 + self.0[1].leading_zeros()
        } else {
            192 + self.0[0].leading_zeros()
        }
    }

    /// Minimum number of bits required to represent `self` (`0` for `ZERO`).
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u32 {
        256 - self.leading_zeros()
    }

    /// Adds two `U256` integers with carry-out flag `(sum, overflowed)`.
    #[must_use]
    pub const fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let mut out = [0u64; 4];
        let mut carry = 0u128;
        let mut i = 0;
        while i < 4 {
            let sum = (self.0[i] as u128) + (rhs.0[i] as u128) + carry;
            out[i] = sum as u64;
            carry = sum >> 64;
            i += 1;
        }
        (Self(out), carry != 0)
    }

    /// Subtracts `rhs` from `self` with borrow flag `(diff, underflowed)`.
    #[must_use]
    pub const fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let mut out = [0u64; 4];
        let mut borrow = 0u64;
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

    /// Multiplies two `U256` values modulo $2^{256}$ alongside overflow detection.
    #[must_use]
    pub const fn overflowing_mul(self, rhs: Self) -> (Self, bool) {
        let mut prod = [0u64; 8];
        let mut i = 0;
        while i < 4 {
            let mut carry = 0u128;
            let mut j = 0;
            while j < 4 {
                let acc = (prod[i + j] as u128) + (self.0[i] as u128) * (rhs.0[j] as u128) + carry;
                prod[i + j] = acc as u64;
                carry = acc >> 64;
                j += 1;
            }
            prod[i + 4] = carry as u64;
            i += 1;
        }
        let overflow = (prod[4] | prod[5] | prod[6] | prod[7]) != 0;
        (Self([prod[0], prod[1], prod[2], prod[3]]), overflow)
    }

    /// Wrapping addition modulo $2^{256}$.
    #[inline]
    #[must_use]
    pub const fn wrapping_add(self, rhs: Self) -> Self {
        self.overflowing_add(rhs).0
    }

    /// Wrapping subtraction modulo $2^{256}$.
    #[inline]
    #[must_use]
    pub const fn wrapping_sub(self, rhs: Self) -> Self {
        self.overflowing_sub(rhs).0
    }

    /// Wrapping multiplication modulo $2^{256}$.
    #[inline]
    #[must_use]
    pub const fn wrapping_mul(self, rhs: Self) -> Self {
        self.overflowing_mul(rhs).0
    }

    /// Two's-complement negation modulo $2^{256}$.
    #[inline]
    #[must_use]
    pub const fn wrapping_neg(self) -> Self {
        Self([!self.0[0], !self.0[1], !self.0[2], !self.0[3]]).wrapping_add(Self::ONE)
    }

    /// Checked addition returning [`ScalarError::ArithmeticOverflow`] on carry past 256 bits.
    ///
    /// # Errors
    /// Returns [`ScalarError::ArithmeticOverflow`] if $a + b \ge 2^{256}$.
    pub const fn checked_add(self, rhs: Self) -> Result<Self, ScalarError> {
        let (res, over) = self.overflowing_add(rhs);
        if over {
            Err(ScalarError::ArithmeticOverflow)
        } else {
            Ok(res)
        }
    }

    /// Checked subtraction returning [`ScalarError::ArithmeticOverflow`] when `self < rhs`.
    ///
    /// # Errors
    /// Returns [`ScalarError::ArithmeticOverflow`] if `self < rhs`.
    pub const fn checked_sub(self, rhs: Self) -> Result<Self, ScalarError> {
        let (res, under) = self.overflowing_sub(rhs);
        if under {
            Err(ScalarError::ArithmeticOverflow)
        } else {
            Ok(res)
        }
    }

    /// Checked multiplication returning [`ScalarError::ArithmeticOverflow`] on overflow.
    ///
    /// # Errors
    /// Returns [`ScalarError::ArithmeticOverflow`] if $a \cdot b \ge 2^{256}$.
    pub const fn checked_mul(self, rhs: Self) -> Result<Self, ScalarError> {
        let (res, over) = self.overflowing_mul(rhs);
        if over {
            Err(ScalarError::ArithmeticOverflow)
        } else {
            Ok(res)
        }
    }

    /// Shifts left by `shift` bits (`0` if `shift >= 256`).
    #[must_use]
    pub const fn shl_bits(self, shift: u32) -> Self {
        if shift >= 256 {
            return Self::ZERO;
        }
        let word_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;
        let mut out = [0u64; 4];
        let mut i = word_shift;
        while i < 4 {
            out[i] = self.0[i - word_shift] << bit_shift;
            if bit_shift > 0 && i > word_shift {
                out[i] |= self.0[i - word_shift - 1] >> (64 - bit_shift);
            }
            i += 1;
        }
        Self(out)
    }

    /// Shifts right logically by `shift` bits (`0` if `shift >= 256`).
    #[must_use]
    pub const fn shr_bits(self, shift: u32) -> Self {
        if shift >= 256 {
            return Self::ZERO;
        }
        let word_shift = (shift / 64) as usize;
        let bit_shift = shift % 64;
        let mut out = [0u64; 4];
        let mut i = 0;
        while i + word_shift < 4 {
            out[i] = self.0[i + word_shift] >> bit_shift;
            if bit_shift > 0 && i + word_shift + 1 < 4 {
                out[i] |= self.0[i + word_shift + 1] << (64 - bit_shift);
            }
            i += 1;
        }
        Self(out)
    }

    /// Constant-bounded shift-subtract division returning `(quotient, remainder)`.
    ///
    /// # Errors
    /// Returns [`ScalarError::DivisionByZero`] when `rhs == ZERO`.
    pub const fn checked_div_rem(self, rhs: Self) -> Result<(Self, Self), ScalarError> {
        if rhs.is_zero() {
            return Err(ScalarError::DivisionByZero);
        }
        if self.cmp_const(rhs) < 0 {
            return Ok((Self::ZERO, self));
        }
        let mut quot = Self::ZERO;
        let mut rem = Self::ZERO;
        let mut bit_idx = self.bits();
        while bit_idx > 0 {
            bit_idx -= 1;
            rem = rem.shl_bits(1);
            if self.bit(bit_idx as usize) {
                rem.0[0] |= 1;
            }
            if rem.cmp_const(rhs) >= 0 {
                rem = rem.wrapping_sub(rhs);
                quot.set_bit(bit_idx as usize);
            }
        }
        Ok((quot, rem))
    }

    /// Checked unsigned integer quotient `self / rhs` (`Err(DivisionByZero)` if `rhs == ZERO`).
    #[inline]
    pub const fn checked_div(self, rhs: Self) -> Result<Self, ScalarError> {
        match self.checked_div_rem(rhs) {
            Ok((q, _)) => Ok(q),
            Err(e) => Err(e),
        }
    }

    /// Checked unsigned remainder `self % rhs` (`Err(DivisionByZero)` if `rhs == ZERO`).
    #[inline]
    pub const fn checked_rem(self, rhs: Self) -> Result<Self, ScalarError> {
        match self.checked_div_rem(rhs) {
            Ok((_, r)) => Ok(r),
            Err(e) => Err(e),
        }
    }

    /// Checked Euclidean quotient (`Err(DivisionByZero)` if `rhs == ZERO`).
    #[inline]
    pub const fn checked_div_euclid(self, rhs: Self) -> Result<Self, ScalarError> {
        self.checked_div(rhs)
    }

    /// Checked Euclidean remainder (`Err(DivisionByZero)` if `rhs == ZERO`).
    #[inline]
    pub const fn checked_rem_euclid(self, rhs: Self) -> Result<Self, ScalarError> {
        self.checked_rem(rhs)
    }

    /// Parses an unsigned base-10 ASCII string into `U256` without heap allocation.
    ///
    /// # Errors
    /// Returns [`ScalarError::InvalidDigit`] on empty/non-digit input or [`ScalarError::ArithmeticOverflow`] on overflow.
    pub fn from_dec_str(s: &str) -> Result<Self, ScalarError> {
        let bytes = s.as_bytes();
        if bytes.is_empty() {
            return Err(ScalarError::InvalidDigit);
        }
        let ten = Self::from_u64(10);
        let mut acc = Self::ZERO;
        for &b in bytes {
            if b == b'_' {
                continue;
            }
            if !b.is_ascii_digit() {
                return Err(ScalarError::InvalidDigit);
            }
            acc = acc.checked_mul(ten)?;
            acc = acc.checked_add(Self::from_u64(u64::from(b - b'0')))?;
        }
        Ok(acc)
    }

    pub(crate) const fn cmp_const(self, rhs: Self) -> i8 {
        let mut i = 4;
        while i > 0 {
            i -= 1;
            if self.0[i] > rhs.0[i] {
                return 1;
            }
            if self.0[i] < rhs.0[i] {
                return -1;
            }
        }
        0
    }
}

impl PartialOrd for U256 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for U256 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.cmp_const(*other) {
            -1 => Ordering::Less,
            0 => Ordering::Equal,
            _ => Ordering::Greater,
        }
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

impl Add for U256 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs).expect("U256 addition overflow")
    }
}

impl AddAssign for U256 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for U256 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(rhs).expect("U256 subtraction underflow")
    }
}

impl SubAssign for U256 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for U256 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(rhs).expect("U256 multiplication overflow")
    }
}

impl Div for U256 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.checked_div(rhs).expect("U256 division by zero")
    }
}

impl Rem for U256 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self.checked_rem(rhs).expect("U256 remainder by zero")
    }
}

impl Shl<u32> for U256 {
    type Output = Self;
    #[inline]
    fn shl(self, rhs: u32) -> Self::Output {
        self.shl_bits(rhs)
    }
}

impl Shr<u32> for U256 {
    type Output = Self;
    #[inline]
    fn shr(self, rhs: u32) -> Self::Output {
        self.shr_bits(rhs)
    }
}

impl fmt::Display for U256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let mut buf = [0u8; 80];
        let mut pos = buf.len();
        let ten = Self::from_u64(10);
        let mut cur = *self;
        while !cur.is_zero() {
            let (q, r) = cur.checked_div_rem(ten).unwrap_or((Self::ZERO, Self::ZERO));
            pos -= 1;
            buf[pos] = b'0' + (r.0[0] as u8);
            cur = q;
        }
        let s = core::str::from_utf8(&buf[pos..]).unwrap_or("0");
        f.write_str(s)
    }
}
