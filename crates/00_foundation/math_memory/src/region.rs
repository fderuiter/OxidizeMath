#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
//! Deterministic single-buffer layout planner (`ArenaLayout`) and typed region handles (`ArenaRegion`).

use crate::arena::{ByteArena, align_padding};
use crate::error::ArenaError;
use crate::views::{BitSlice, FlagSlice};
use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;

/// Base alignment guaranteed by [`crate::OwnedByteArena`] (`>= 8` on `wasm32`, `16` on 64-bit targets).
pub const MAX_ARENA_ALIGN: usize = core::mem::align_of::<u128>();

/// Typed byte-offset handle for a persistent `[T]` slice inside an [`crate::OwnedByteArena`].
pub struct ArenaRegion<T> {
    pub(crate) byte_offset: usize,
    pub(crate) len: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Copy for ArenaRegion<T> {}

impl<T> Clone for ArenaRegion<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> core::fmt::Debug for ArenaRegion<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArenaRegion")
            .field("byte_offset", &self.byte_offset)
            .field("len", &self.len)
            .finish()
    }
}

impl<T> PartialEq for ArenaRegion<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.byte_offset == other.byte_offset && self.len == other.len
    }
}

impl<T> Eq for ArenaRegion<T> {}

impl<T> Default for ArenaRegion<T> {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> ArenaRegion<T> {
    /// Creates an empty zero-length region at byte offset `0`.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            byte_offset: 0,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Element count reserved for this region.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when `self.len() == 0`.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte length (`len * size_of::<T>()`) of this region.
    #[inline]
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.len * core::mem::size_of::<T>()
    }
}

/// Typed byte-offset handle for a persistent [`FlagSlice`] inside an [`crate::OwnedByteArena`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FlagRegion {
    pub(crate) inner: ArenaRegion<u8>,
}

impl FlagRegion {
    /// Creates an empty zero-length flag region.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            inner: ArenaRegion::empty(),
        }
    }

    /// Number of boolean flags reserved in this region.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` when `self.len() == 0`.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Typed byte-offset handle for a persistent [`BitSlice`] inside an [`crate::OwnedByteArena`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BitRegion {
    pub(crate) words: ArenaRegion<u64>,
    pub(crate) bit_len: usize,
}

impl BitRegion {
    /// Creates an empty zero-length bit region.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            words: ArenaRegion::empty(),
            bit_len: 0,
        }
    }

    /// Number of addressable bits reserved in this region.
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
}

/// Pure `#![no_std]` layout planner computing aligned, non-overlapping [`ArenaRegion`] offsets.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ArenaLayout {
    total_bytes: usize,
}

impl ArenaLayout {
    /// Creates an empty layout starting at byte offset `0`.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { total_bytes: 0 }
    }

    /// Total aligned byte capacity required to hold all reserved regions.
    #[inline]
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Reserves an aligned region of `len` elements of `T` (`align_of::<T>() <= MAX_ARENA_ALIGN`).
    ///
    /// # Errors
    /// Returns [`ArenaError`] if `align_of::<T>() > MAX_ARENA_ALIGN` or arithmetic overflows.
    pub fn reserve<T: Pod + Zeroable>(&mut self, len: usize) -> Result<ArenaRegion<T>, ArenaError> {
        let align = core::mem::align_of::<T>();
        if !align.is_power_of_two() || align > MAX_ARENA_ALIGN {
            return Err(ArenaError::InvalidAlignment { align });
        }
        let byte_len = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(ArenaError::ArithmeticOverflow)?;
        if byte_len == 0 {
            return Ok(ArenaRegion {
                byte_offset: self.total_bytes,
                len: 0,
                _marker: PhantomData,
            });
        }
        let pad = align_padding(self.total_bytes, align)?;
        let start = self
            .total_bytes
            .checked_add(pad)
            .ok_or(ArenaError::ArithmeticOverflow)?;
        self.total_bytes = start
            .checked_add(byte_len)
            .ok_or(ArenaError::ArithmeticOverflow)?;
        Ok(ArenaRegion {
            byte_offset: start,
            len,
            _marker: PhantomData,
        })
    }

    /// Reserves a [`FlagRegion`] of `len` boolean flags.
    ///
    /// # Errors
    /// Returns [`ArenaError::ArithmeticOverflow`] if byte size overflows `usize`.
    #[inline]
    pub fn reserve_flags(&mut self, len: usize) -> Result<FlagRegion, ArenaError> {
        Ok(FlagRegion {
            inner: self.reserve::<u8>(len)?,
        })
    }

    /// Reserves a [`BitRegion`] of `bit_len` packed bits.
    ///
    /// # Errors
    /// Returns [`ArenaError`] if word byte size overflows `usize`.
    #[inline]
    pub fn reserve_bits(&mut self, bit_len: usize) -> Result<BitRegion, ArenaError> {
        Ok(BitRegion {
            words: self.reserve::<u64>(BitSlice::required_words(bit_len))?,
            bit_len,
        })
    }
}

/// Non-zeroing linear cursor splitting multiple persistent [`ArenaRegion`] slices simultaneously.
#[derive(Debug)]
pub struct RawByteCursor<'a> {
    base_offset: usize,
    tail: &'a mut [u8],
}

impl<'a> RawByteCursor<'a> {
    /// Wraps a mutable byte buffer starting at byte offset `0` without zeroing existing contents.
    #[inline]
    #[must_use]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self {
            base_offset: 0,
            tail: buf,
        }
    }

    /// Fallibly splits off `&'a mut [T]` for `region` without zeroing existing values.
    ///
    /// # Errors
    /// Returns [`ArenaError`] if `region` is out of reservation order, exceeds buffer bounds, or is misaligned.
    pub fn try_region_mut<T: Pod>(
        &mut self,
        region: ArenaRegion<T>,
    ) -> Result<&'a mut [T], ArenaError> {
        let byte_len = region.byte_len();
        if byte_len == 0 {
            return Ok(&mut []);
        }
        if region.byte_offset < self.base_offset {
            return Err(ArenaError::RegionOutOfOrder {
                region_offset: region.byte_offset,
                cursor_offset: self.base_offset,
            });
        }
        let skip = region.byte_offset - self.base_offset;
        let required = skip
            .checked_add(byte_len)
            .ok_or(ArenaError::ArithmeticOverflow)?;
        if required > self.tail.len() {
            return Err(ArenaError::InsufficientBytes {
                required,
                available: self.tail.len(),
                align: core::mem::align_of::<T>(),
            });
        }
        let full = core::mem::take(&mut self.tail);
        let (_, after_skip) = full.split_at_mut(skip);
        let (carved, rest) = after_skip.split_at_mut(byte_len);
        self.tail = rest;
        self.base_offset = region.byte_offset + byte_len;
        bytemuck::try_cast_slice_mut(carved).map_err(|_| ArenaError::InvalidAlignment {
            align: core::mem::align_of::<T>(),
        })
    }

    /// Splits off `&'a mut [T]` for `region` without zeroing existing values.
    ///
    /// Regions must be requested in monotonically increasing reservation order.
    #[inline]
    pub fn region_mut<T: Pod>(&mut self, region: ArenaRegion<T>) -> &'a mut [T] {
        self.try_region_mut(region)
            .expect("RawByteCursor region out of order or bounds")
    }

    /// Splits off a [`FlagSlice<'a>`] for `region` without zeroing existing flag values.
    #[inline]
    pub fn flags_mut(&mut self, region: FlagRegion) -> FlagSlice<'a> {
        FlagSlice::from_existing_bytes(self.region_mut(region.inner))
    }

    /// Splits off a [`BitSlice<'a>`] for `region` without zeroing existing bit patterns.
    #[inline]
    pub fn bits_mut(&mut self, region: BitRegion) -> BitSlice<'a> {
        BitSlice::from_existing_words(self.region_mut(region.words), region.bit_len)
    }

    /// Returns the remaining trailing bytes as a scratch [`ByteArena<'a>`].
    #[inline]
    #[must_use]
    pub fn into_trailing_arena(self) -> ByteArena<'a> {
        ByteArena::new(self.tail)
    }
}
