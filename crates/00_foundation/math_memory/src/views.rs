// [cite:math_memory]
//! Safe byte-backed boolean flag slices and `u64`-backed packed bit slices.
//!
//! Rust's `bool` layout has a niche validity invariant (`0` or `1` only), so casting arbitrary
//! `&mut [u8]` buffers to `&mut [bool]` is undefined behavior. [`FlagSlice`] and [`BitSlice`]
//! provide sound, zero-unsafe boolean scratch views over [`crate::ByteArena`] buffers.

use crate::error::ArenaError;

/// Safe byte-per-flag boolean view backed by `&'a mut [u8]`.
///
/// Reads interpret any non-zero byte as `true`, and writes store `0u8` or `1u8`.
#[derive(Debug, PartialEq, Eq)]
pub struct FlagSlice<'a> {
    bytes: &'a mut [u8],
}

impl<'a> FlagSlice<'a> {
    /// Zero-initializes `bytes` and wraps it as an all-`false` [`FlagSlice`].
    #[inline]
    pub fn new_zeroed(bytes: &'a mut [u8]) -> Self {
        bytes.fill(0);
        Self { bytes }
    }

    /// Normalizes existing `bytes` in place (`0 -> 0`, non-zero `-> 1`) and wraps them safely.
    #[inline]
    pub fn from_bytes(bytes: &'a mut [u8]) -> Self {
        for b in bytes.iter_mut() {
            *b = u8::from(*b != 0);
        }
        Self { bytes }
    }

    /// Number of addressable boolean flags.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns `true` if the slice contains zero flags.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Reads the boolean flag at `index`.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        self.bytes[index] != 0
    }

    /// Checked read of the boolean flag at `index`, returning `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn try_get(&self, index: usize) -> Option<bool> {
        self.bytes.get(index).map(|&b| b != 0)
    }

    /// Writes `value` into the flag at `index` (`0` for `false`, `1` for `true`).
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        self.bytes[index] = u8::from(value);
    }

    /// Resets all flags to `false`.
    #[inline]
    pub fn clear_all(&mut self) {
        self.bytes.fill(0);
    }

    /// Sets every flag in the slice to `value`.
    #[inline]
    pub fn fill(&mut self, value: bool) {
        self.bytes.fill(u8::from(value));
    }

    /// Counts how many flags are currently set to `true`.
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.bytes.iter().filter(|&&b| b != 0).count()
    }

    /// Immutable view of the underlying `0`/`1` byte representation.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8] {
        self.bytes
    }

    /// Iterates over all boolean values in index order.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        self.bytes.iter().map(|&b| b != 0)
    }
}

/// Packed bitset view backed by `&'a mut [u64]` words.
#[derive(Debug, PartialEq, Eq)]
pub struct BitSlice<'a> {
    words: &'a mut [u64],
    bit_len: usize,
}

impl<'a> BitSlice<'a> {
    /// Computes the number of `u64` words required to store `bit_len` bits without overflow.
    #[inline]
    #[must_use]
    pub const fn required_words(bit_len: usize) -> usize {
        if bit_len == 0 {
            0
        } else {
            ((bit_len - 1) >> 6) + 1
        }
    }

    /// Zero-initializes `words` and wraps the first `required_words(bit_len)` words as a [`BitSlice`].
    pub fn new_zeroed(words: &'a mut [u64], bit_len: usize) -> Result<Self, ArenaError> {
        let needed = Self::required_words(bit_len);
        if words.len() < needed {
            let requested = needed.checked_mul(core::mem::size_of::<u64>()).ok_or(
                ArenaError::CapacityOverflow {
                    elem_size: core::mem::size_of::<u64>(),
                    count: needed,
                },
            )?;
            let available = words.len().saturating_mul(core::mem::size_of::<u64>());
            return Err(ArenaError::InsufficientCapacity {
                requested,
                available,
            });
        }
        let active = &mut words[..needed];
        active.fill(0);
        Ok(Self {
            words: active,
            bit_len,
        })
    }

    /// Wraps existing `words` without clearing them, masking any unused tail bits in the final word.
    pub fn from_words(words: &'a mut [u64], bit_len: usize) -> Result<Self, ArenaError> {
        let needed = Self::required_words(bit_len);
        if words.len() < needed {
            let requested = needed.checked_mul(core::mem::size_of::<u64>()).ok_or(
                ArenaError::CapacityOverflow {
                    elem_size: core::mem::size_of::<u64>(),
                    count: needed,
                },
            )?;
            let available = words.len().saturating_mul(core::mem::size_of::<u64>());
            return Err(ArenaError::InsufficientCapacity {
                requested,
                available,
            });
        }
        let mut out = Self {
            words: &mut words[..needed],
            bit_len,
        };
        out.mask_tail();
        Ok(out)
    }

    /// Logical number of addressable bits.
    #[inline]
    #[must_use]
    pub const fn bit_len(&self) -> usize {
        self.bit_len
    }

    /// Returns `true` if `bit_len == 0`.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bit_len == 0
    }

    /// Number of backing `u64` words.
    #[inline]
    #[must_use]
    pub const fn word_len(&self) -> usize {
        self.words.len()
    }

    /// Reads bit `bit_index`, returning [`ArenaError::InsufficientCapacity`] when out of bounds.
    #[inline]
    pub fn checked_get(&self, bit_index: usize) -> Result<bool, ArenaError> {
        if bit_index >= self.bit_len {
            return Err(ArenaError::InsufficientCapacity {
                requested: bit_index.saturating_add(1),
                available: self.bit_len,
            });
        }
        Ok(((self.words[bit_index >> 6] >> (bit_index & 63)) & 1) != 0)
    }

    /// Reads bit `bit_index`.
    ///
    /// # Panics
    /// Panics if `bit_index >= self.bit_len()`; prefer [`Self::checked_get`] or [`Self::try_get`].
    #[inline]
    #[must_use]
    pub fn get(&self, bit_index: usize) -> bool {
        assert!(
            bit_index < self.bit_len,
            "BitSlice index {bit_index} out of bounds for bit_len {}",
            self.bit_len
        );
        let word = self.words[bit_index >> 6];
        ((word >> (bit_index & 63)) & 1) != 0
    }

    /// Checked bit read returning `None` when `bit_index >= self.bit_len()`.
    #[inline]
    #[must_use]
    pub fn try_get(&self, bit_index: usize) -> Option<bool> {
        if bit_index >= self.bit_len {
            return None;
        }
        Some(((self.words[bit_index >> 6] >> (bit_index & 63)) & 1) != 0)
    }

    /// Sets or clears bit `bit_index`, returning [`ArenaError::InsufficientCapacity`] when out of bounds.
    #[inline]
    pub fn checked_set(&mut self, bit_index: usize, value: bool) -> Result<(), ArenaError> {
        if bit_index >= self.bit_len {
            return Err(ArenaError::InsufficientCapacity {
                requested: bit_index.saturating_add(1),
                available: self.bit_len,
            });
        }
        let mask = 1_u64 << (bit_index & 63);
        if value {
            self.words[bit_index >> 6] |= mask;
        } else {
            self.words[bit_index >> 6] &= !mask;
        }
        Ok(())
    }

    /// Sets or clears bit `bit_index`.
    ///
    /// # Panics
    /// Panics if `bit_index >= self.bit_len()`; prefer [`Self::checked_set`].
    #[inline]
    pub fn set(&mut self, bit_index: usize, value: bool) {
        assert!(
            bit_index < self.bit_len,
            "BitSlice index {bit_index} out of bounds for bit_len {}",
            self.bit_len
        );
        let mask = 1_u64 << (bit_index & 63);
        if value {
            self.words[bit_index >> 6] |= mask;
        } else {
            self.words[bit_index >> 6] &= !mask;
        }
    }

    /// Clears all bits to `false`.
    #[inline]
    pub fn clear_all(&mut self) {
        self.words.fill(0);
    }

    /// Fills all `bit_len` bits with `value`, keeping trailing unused bits zeroed.
    #[inline]
    pub fn fill(&mut self, value: bool) {
        if value {
            self.words.fill(u64::MAX);
            self.mask_tail();
        } else {
            self.words.fill(0);
        }
    }

    /// Counts the number of `1` bits across `0..bit_len`.
    #[inline]
    #[must_use]
    pub fn count_ones(&self) -> usize {
        let Some((&last, init)) = self.words.split_last() else {
            return 0;
        };
        let mut total: usize = init.iter().map(|w| w.count_ones() as usize).sum();
        let rem = self.bit_len & 63;
        let masked_last = if rem == 0 {
            last
        } else {
            last & ((1_u64 << rem) - 1)
        };
        total += masked_last.count_ones() as usize;
        total
    }

    /// Immutable access to the underlying `u64` words.
    #[inline]
    #[must_use]
    pub const fn words(&self) -> &[u64] {
        self.words
    }

    /// Mutable access to the underlying `u64` words for bulk bitwise kernels.
    #[inline]
    pub fn words_mut(&mut self) -> &mut [u64] {
        self.words
    }

    #[inline]
    fn mask_tail(&mut self) {
        let rem = self.bit_len & 63;
        if rem != 0
            && let Some(last) = self.words.last_mut()
        {
            *last &= (1_u64 << rem) - 1;
        }
    }
}
