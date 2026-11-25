// [cite:math_memory]
//! Deterministic byte-layout planner and `alloc`-gated [`OwnedByteArena`] backing store.

use core::marker::PhantomData;

use bytemuck::Pod;
#[cfg(feature = "alloc")]
use bytemuck::Zeroable;

#[cfg(feature = "alloc")]
use crate::arena::{ArenaScope, ByteArena};
use crate::arena::{align_padding, slice_byte_len};
use crate::error::ArenaError;

/// Typed byte-range descriptor inside an 8-byte-aligned [`OwnedByteArena`].
#[derive(Debug, PartialEq, Eq)]
pub struct ArenaRegion<T> {
    byte_offset: usize,
    len: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for ArenaRegion<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ArenaRegion<T> {}

impl<T> Default for ArenaRegion<T> {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> ArenaRegion<T> {
    /// Empty zero-length region at byte offset `0`.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            byte_offset: 0,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Starting byte offset in the arena.
    #[inline]
    #[must_use]
    pub const fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Logical number of `T` elements in this region.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if `len == 0`.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Ending byte offset (`byte_offset + len * size_of::<T>()`).
    #[inline]
    #[must_use]
    pub const fn byte_end(&self) -> usize {
        self.byte_offset
            .saturating_add(self.len.saturating_mul(core::mem::size_of::<T>()))
    }
}

/// Checked byte-offset layout builder matching [`ArenaScope`] alignment rules from base offset `0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArenaLayout {
    total_bytes: usize,
}

impl ArenaLayout {
    /// Creates an empty layout starting at byte `0`.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self { total_bytes: 0 }
    }

    /// Appends an aligned slice region of `count` elements of `T`, returning its [`ArenaRegion`].
    pub fn add_slice<T: Pod>(&mut self, count: usize) -> Result<ArenaRegion<T>, ArenaError> {
        let elem_size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let byte_len = slice_byte_len::<T>(count)?;
        if byte_len == 0 {
            if !align.is_power_of_two() {
                return Err(ArenaError::InvalidAlignment { align });
            }
            return Ok(ArenaRegion::empty());
        }
        let pad = align_padding(self.total_bytes, align)?;
        let start = self
            .total_bytes
            .checked_add(pad)
            .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
        let end = start
            .checked_add(byte_len)
            .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
        self.total_bytes = end;
        Ok(ArenaRegion {
            byte_offset: start,
            len: count,
            _marker: PhantomData,
        })
    }

    /// Total aligned byte capacity required for all added regions.
    #[inline]
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

/// Reusable 8-byte-aligned owned byte arena available when the `alloc` feature is enabled.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OwnedByteArena {
    words: alloc::vec::Vec<u64>,
    byte_len: usize,
}

#[cfg(feature = "alloc")]
impl OwnedByteArena {
    /// Creates an empty [`OwnedByteArena`] that does not allocate until grown.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            words: alloc::vec::Vec::new(),
            byte_len: 0,
        }
    }

    #[inline]
    const fn words_for_bytes(byte_cap: usize) -> usize {
        if byte_cap == 0 {
            0
        } else {
            ((byte_cap - 1) >> 3) + 1
        }
    }

    /// Preallocates at least `byte_cap` zero-initialized bytes with 8-byte base alignment.
    #[must_use]
    pub fn with_capacity(byte_cap: usize) -> Self {
        let word_count = Self::words_for_bytes(byte_cap);
        Self {
            words: alloc::vec![0_u64; word_count],
            byte_len: word_count.saturating_mul(core::mem::size_of::<u64>()),
        }
    }

    /// Preallocates enough aligned bytes for `count` elements of `T`.
    pub fn with_elements<T: Pod>(count: usize) -> Result<Self, ArenaError> {
        let mut layout = ArenaLayout::new();
        let _ = layout.add_slice::<T>(count)?;
        Ok(Self::with_capacity(layout.total_bytes()))
    }

    /// Active 8-byte-aligned byte capacity.
    #[inline]
    #[must_use]
    pub const fn byte_capacity(&self) -> usize {
        self.byte_len
    }

    /// Returns `true` if `byte_capacity() == 0`.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.byte_len == 0
    }

    /// Ensures at least `min_bytes` are allocated, growing the backing `u64` buffer if needed.
    pub fn ensure_capacity(&mut self, min_bytes: usize) {
        if min_bytes > self.byte_len {
            let word_count = Self::words_for_bytes(min_bytes);
            self.words.resize(word_count, 0);
            self.byte_len = word_count.saturating_mul(core::mem::size_of::<u64>());
        }
    }

    /// Fallible capacity growth helper for `count` elements of `T`.
    pub fn ensure_elements<T: Pod>(&mut self, count: usize) -> Result<(), ArenaError> {
        let mut layout = ArenaLayout::new();
        let _ = layout.add_slice::<T>(count)?;
        self.ensure_capacity(layout.total_bytes());
        Ok(())
    }

    /// Zeroes all bytes in place without releasing capacity.
    #[inline]
    pub fn clear_zero(&mut self) {
        self.words.fill(0);
    }

    /// Immutable view of the backing bytes.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &bytemuck::cast_slice::<u64, u8>(&self.words)[..self.byte_len]
    }

    /// Mutable view of the backing bytes.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let len = self.byte_len;
        &mut bytemuck::cast_slice_mut::<u64, u8>(&mut self.words)[..len]
    }

    /// Borrows this container as a fresh [`ByteArena`] starting at offset `0`.
    #[inline]
    pub fn as_arena(&mut self) -> ByteArena<'_> {
        ByteArena::new(self.as_bytes_mut())
    }

    /// Opens a slice-splitting [`ArenaScope`] over the entire buffer.
    #[inline]
    pub fn reborrow(&mut self) -> ArenaScope<'_> {
        ArenaScope::new(self.as_bytes_mut())
    }

    /// Borrows a persistent [`ArenaRegion`] immutably in $O(1)$ time.
    #[inline]
    #[must_use]
    pub fn region<T: Pod>(&self, reg: ArenaRegion<T>) -> &[T] {
        if reg.is_empty() {
            return &[];
        }
        bytemuck::cast_slice(&self.as_bytes()[reg.byte_offset()..reg.byte_end()])
    }

    /// Borrows a persistent [`ArenaRegion`] mutably in $O(1)$ time.
    #[inline]
    pub fn region_mut<T: Pod + Zeroable>(&mut self, reg: ArenaRegion<T>) -> &mut [T] {
        if reg.is_empty() {
            return &mut [];
        }
        let start = reg.byte_offset();
        let end = reg.byte_end();
        bytemuck::cast_slice_mut(&mut self.as_bytes_mut()[start..end])
    }
}
