#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
//! Safe `#![no_std]` boolean views (`FlagSlice` and `BitSlice`) over byte and word arena slices.
//!
//! Rust forbids reinterpreting arbitrary bytes as `&mut [bool]` because any bit pattern outside
//! `0..=1` triggers immediate undefined behavior. [`FlagSlice`] stores `0_u8` / `1_u8` bytes and
//! decodes booleans by value, while [`BitSlice`] packs `64` boolean flags per `u64` word.

use crate::error::ArenaError;

/// Mutable byte-per-flag boolean view over a zeroed `&'a mut [u8]` arena slice.
#[derive(Debug, PartialEq, Eq)]
pub struct FlagSlice<'a> {
    bytes: &'a mut [u8],
}

impl<'a> FlagSlice<'a> {
    /// Wraps a mutable byte slice and clears all entries to `false` (`0_u8`).
    #[inline]
    pub fn new_zeroed(bytes: &'a mut [u8]) -> Self {
        bytes.fill(0);
        Self { bytes }
    }

    /// Wraps an existing byte slice without clearing entries, preserving previously written flags.
    #[inline]
    pub fn from_existing_bytes(bytes: &'a mut [u8]) -> Self {
        Self { bytes }
    }

    /// Shrinks this view to its first `len` flags (`Err` if `len > self.len()`).
    ///
    /// # Errors
    /// Returns [`ArenaError::IndexOutOfBounds`] if `len > self.len()`.
    #[inline]
    pub fn try_into_prefix(self, len: usize) -> Result<Self, ArenaError> {
        let cur_len = self.bytes.len();
        if len > cur_len {
            return Err(ArenaError::IndexOutOfBounds {
                index: len,
                len: cur_len,
            });
        }
        Ok(Self {
            bytes: &mut self.bytes[..len],
        })
    }

    /// Shrinks this view to its first `len` flags (`panics` if `len > self.len()`).
    #[inline]
    #[must_use]
    pub fn into_prefix(self, len: usize) -> Self {
        self.try_into_prefix(len)
            .expect("FlagSlice prefix length exceeds capacity")
    }

    /// Number of addressable boolean flags.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns `true` when this slice contains zero flags.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Reads flag `index` (`panics` if `index >= self.len()`).
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        self.bytes[index] != 0
    }

    /// Checked read returning [`ArenaError::IndexOutOfBounds`] when `index >= self.len()`.
    ///
    /// # Errors
    /// Returns [`ArenaError::IndexOutOfBounds`] if `index >= self.len()`.
    #[inline]
    pub fn try_get(&self, index: usize) -> Result<bool, ArenaError> {
        self.bytes
            .get(index)
            .map(|&b| b != 0)
            .ok_or(ArenaError::IndexOutOfBounds {
                index,
                len: self.bytes.len(),
            })
    }

    /// Checked write returning [`ArenaError::IndexOutOfBounds`] when `index >= self.len()`.
    ///
    /// # Errors
    /// Returns [`ArenaError::IndexOutOfBounds`] if `index >= self.len()`.
    #[inline]
    pub fn try_set(&mut self, index: usize, value: bool) -> Result<(), ArenaError> {
        let len = self.bytes.len();
        let slot = self
            .bytes
            .get_mut(index)
            .ok_or(ArenaError::IndexOutOfBounds { index, len })?;
        *slot = u8::from(value);
        Ok(())
    }

    /// Writes `value` (`0_u8` or `1_u8`) into flag `index` (`panics` if `index >= self.len()`).
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        self.bytes[index] = u8::from(value);
    }

    /// Fills every flag in the slice with `value`.
    #[inline]
    pub fn fill(&mut self, value: bool) {
        self.bytes.fill(u8::from(value));
    }

    /// Resets every flag in the slice to `false` (`0_u8`).
    #[inline]
    pub fn clear(&mut self) {
        self.bytes.fill(0);
    }

    /// Underlying raw byte representation (`0` for `false`, non-zero for `true`).
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8] {
        self.bytes
    }

    /// Iterates boolean values across all flags in index order.
    #[inline]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = bool> + '_ {
        self.bytes.iter().map(|&b| b != 0)
    }
}

/// Mutable packed bitset view over a `&'a mut [u64]` arena word slice.
#[derive(Debug, PartialEq, Eq)]
pub struct BitSlice<'a> {
    words: &'a mut [u64],
    bit_len: usize,
}

impl<'a> BitSlice<'a> {
    /// Returns the number of `u64` words required to store `bit_len` bits.
    #[inline]
    #[must_use]
    pub const fn required_words(bit_len: usize) -> usize {
        bit_len.div_ceil(64)
    }

    /// Wraps `words` (requiring `words.len() >= required_words(bit_len)`) and zeroes all words.
    #[inline]
    pub fn new_zeroed(words: &'a mut [u64], bit_len: usize) -> Self {
        let needed = Self::required_words(bit_len);
        let used = &mut words[..needed];
        used.fill(0);
        Self {
            words: used,
            bit_len,
        }
    }

    /// Wraps `words` without zeroing existing bit patterns.
    #[inline]
    pub fn from_existing_words(words: &'a mut [u64], bit_len: usize) -> Self {
        let needed = Self::required_words(bit_len);
        Self {
            words: &mut words[..needed],
            bit_len,
        }
    }

    /// Number of addressable bits in `0..self.len()`.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bit_len
    }

    /// Returns `true` when `self.len() == 0`.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bit_len == 0
    }

    /// Checked bit read returning [`ArenaError::IndexOutOfBounds`] when `index >= self.len()`.
    ///
    /// # Errors
    /// Returns [`ArenaError::IndexOutOfBounds`] if `index >= self.len()`.
    #[inline]
    pub fn try_get(&self, index: usize) -> Result<bool, ArenaError> {
        if index >= self.bit_len {
            return Err(ArenaError::IndexOutOfBounds {
                index,
                len: self.bit_len,
            });
        }
        let word = self.words[index >> 6];
        Ok(((word >> (index & 63)) & 1) != 0)
    }

    /// Reads bit `index` (`panics` if `index >= self.len()`).
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        self.try_get(index).expect("BitSlice index out of bounds")
    }

    /// Checked bit write returning [`ArenaError::IndexOutOfBounds`] when `index >= self.len()`.
    ///
    /// # Errors
    /// Returns [`ArenaError::IndexOutOfBounds`] if `index >= self.len()`.
    #[inline]
    pub fn try_set(&mut self, index: usize, value: bool) -> Result<(), ArenaError> {
        if index >= self.bit_len {
            return Err(ArenaError::IndexOutOfBounds {
                index,
                len: self.bit_len,
            });
        }
        let mask = 1_u64 << (index & 63);
        if value {
            self.words[index >> 6] |= mask;
        } else {
            self.words[index >> 6] &= !mask;
        }
        Ok(())
    }

    /// Writes `value` into bit `index` (`panics` if `index >= self.len()`).
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        self.try_set(index, value)
            .expect("BitSlice index out of bounds");
    }

    /// Checked bit toggle returning [`ArenaError::IndexOutOfBounds`] when `index >= self.len()`.
    ///
    /// # Errors
    /// Returns [`ArenaError::IndexOutOfBounds`] if `index >= self.len()`.
    #[inline]
    pub fn try_toggle(&mut self, index: usize) -> Result<(), ArenaError> {
        if index >= self.bit_len {
            return Err(ArenaError::IndexOutOfBounds {
                index,
                len: self.bit_len,
            });
        }
        self.words[index >> 6] ^= 1_u64 << (index & 63);
        Ok(())
    }

    /// Toggles bit `index` (`panics` if `index >= self.len()`).
    #[inline]
    pub fn toggle(&mut self, index: usize) {
        self.try_toggle(index)
            .expect("BitSlice index out of bounds");
    }

    /// Clears all bits to `false`.
    #[inline]
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Sets all `self.len()` bits to `value`, masking unused trailing bits in the final word to `0`.
    #[inline]
    pub fn fill(&mut self, value: bool) {
        if !value || self.bit_len == 0 {
            self.words.fill(0);
            return;
        }
        self.words.fill(u64::MAX);
        let rem = self.bit_len & 63;
        if rem != 0
            && let Some(last) = self.words.last_mut()
        {
            *last = (1_u64 << rem) - 1;
        }
    }

    /// Counts set (`true`) bits across `0..self.len()`.
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        if self.bit_len == 0 {
            return 0;
        }
        let rem = self.bit_len & 63;
        let full_words = if rem == 0 {
            self.words.len()
        } else {
            self.words.len().saturating_sub(1)
        };
        let mut total = 0_usize;
        for &w in &self.words[..full_words] {
            total += w.count_ones() as usize;
        }
        if rem != 0 {
            let mask = (1_u64 << rem) - 1;
            total += (self.words[full_words] & mask).count_ones() as usize;
        }
        total
    }

    /// Immutable view of the underlying `u64` words.
    #[inline]
    #[must_use]
    pub const fn words(&self) -> &[u64] {
        self.words
    }

    /// Mutable view of the underlying `u64` words for SIMD/word-level XOR reductions.
    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64] {
        self.words
    }
}
