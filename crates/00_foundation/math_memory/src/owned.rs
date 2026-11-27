#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
//! Single-buffer `alloc`-backed [`OwnedByteArena`] aligned to [`MAX_ARENA_ALIGN`].

use crate::arena::{ArenaCheckpoint, ByteArena, ByteCursor, plan_slice_bytes};
use crate::error::ArenaError;
use crate::region::{ArenaLayout, ArenaRegion, BitRegion, FlagRegion, RawByteCursor};
use crate::views::{BitSlice, FlagSlice};
use crate::workspace::SolverWorkspace;
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};

const WORD_BYTES: usize = core::mem::size_of::<u128>();

#[inline]
const fn words_for_bytes(bytes: usize) -> usize {
    bytes.div_ceil(WORD_BYTES)
}

/// Heap-allocated byte arena backed by one `u128`-aligned buffer (`alloc` feature).
///
/// Storing `u128` words guarantees at least 8-byte alignment on 32-bit `wasm32-unknown-unknown`
/// and 16-byte alignment on 64-bit targets without `unsafe` custom `GlobalAlloc` layouts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OwnedByteArena {
    words: Vec<u128>,
    byte_len: usize,
    offset: usize,
}

impl OwnedByteArena {
    /// Creates an empty zero-capacity [`OwnedByteArena`] without heap allocation.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            words: Vec::new(),
            byte_len: 0,
            offset: 0,
        }
    }

    /// Fallibly allocates a zero-initialized arena with at least `byte_capacity` bytes.
    ///
    /// # Errors
    /// Returns [`ArenaError::AllocationFailed`] if the allocator rejects the reservation.
    pub fn try_with_capacity(byte_capacity: usize) -> Result<Self, ArenaError> {
        let word_count = words_for_bytes(byte_capacity);
        let mut words = Vec::new();
        words
            .try_reserve_exact(word_count)
            .map_err(|_| ArenaError::AllocationFailed {
                requested_bytes: byte_capacity,
            })?;
        words.resize(word_count, 0_u128);
        Ok(Self {
            words,
            byte_len: byte_capacity,
            offset: 0,
        })
    }

    /// Allocates a zero-initialized arena with `byte_capacity` usable bytes.
    #[must_use]
    pub fn with_capacity(byte_capacity: usize) -> Self {
        Self::try_with_capacity(byte_capacity).expect("OwnedByteArena::with_capacity failed")
    }

    /// Allocates a zero-initialized arena sized for `layout`.
    ///
    /// # Errors
    /// Returns [`ArenaError::AllocationFailed`] if the allocator rejects `layout.total_bytes()`.
    #[inline]
    pub fn try_from_layout(layout: &ArenaLayout) -> Result<Self, ArenaError> {
        Self::try_with_capacity(layout.total_bytes())
    }

    /// Allocates a zero-initialized arena sized for `layout`.
    #[inline]
    #[must_use]
    pub fn from_layout(layout: &ArenaLayout) -> Self {
        Self::with_capacity(layout.total_bytes())
    }

    /// Fallibly grows this arena to at least `min_bytes` usable bytes without shrinking.
    ///
    /// # Errors
    /// Returns [`ArenaError::AllocationFailed`] if reallocation fails.
    pub fn try_ensure_capacity(&mut self, min_bytes: usize) -> Result<(), ArenaError> {
        if min_bytes <= self.byte_len {
            return Ok(());
        }
        let needed_words = words_for_bytes(min_bytes);
        if needed_words > self.words.len() {
            let extra = needed_words - self.words.len();
            self.words
                .try_reserve(extra)
                .map_err(|_| ArenaError::AllocationFailed {
                    requested_bytes: min_bytes,
                })?;
            self.words.resize(needed_words, 0_u128);
        }
        self.byte_len = min_bytes;
        Ok(())
    }

    /// Grows this arena to at least `min_bytes` usable bytes (no-op if `min_bytes <= byte_capacity()`).
    pub fn ensure_capacity(&mut self, min_bytes: usize) {
        self.try_ensure_capacity(min_bytes)
            .expect("OwnedByteArena::ensure_capacity failed");
    }

    /// Total usable byte capacity of this arena.
    #[inline]
    #[must_use]
    pub const fn byte_capacity(&self) -> usize {
        self.byte_len
    }

    /// Bytes currently consumed by direct bump allocations on this arena.
    #[inline]
    #[must_use]
    pub const fn used_bytes(&self) -> usize {
        self.offset
    }

    /// Unallocated bytes remaining from the current cursor to `self.byte_capacity()`.
    #[inline]
    #[must_use]
    pub const fn remaining_bytes(&self) -> usize {
        self.byte_len.saturating_sub(self.offset)
    }

    /// Immutable view of all `self.byte_capacity()` bytes.
    #[inline]
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &bytemuck::cast_slice::<u128, u8>(&self.words)[..self.byte_len]
    }

    /// Mutable view of all `self.byte_capacity()` bytes.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        let len = self.byte_len;
        &mut bytemuck::cast_slice_mut::<u128, u8>(&mut self.words)[..len]
    }

    /// Zeroes all bytes in the backing buffer and resets the cursor to `0`.
    #[inline]
    pub fn clear_zero(&mut self) {
        self.words.fill(0);
        self.offset = 0;
    }

    /// Captures the current bump cursor offset.
    #[inline]
    #[must_use]
    pub const fn checkpoint(&self) -> ArenaCheckpoint {
        ArenaCheckpoint {
            offset: self.offset,
        }
    }

    /// Rewinds the bump cursor to `checkpoint`.
    ///
    /// # Errors
    /// Returns [`ArenaError::InvalidCheckpoint`] if `checkpoint` exceeds `self.used_bytes()`.
    #[inline]
    pub fn rewind(&mut self, checkpoint: ArenaCheckpoint) -> Result<(), ArenaError> {
        if checkpoint.offset > self.offset {
            return Err(ArenaError::InvalidCheckpoint {
                checkpoint_offset: checkpoint.offset,
                current_offset: self.offset,
            });
        }
        self.offset = checkpoint.offset;
        Ok(())
    }

    /// Borrows the entire buffer from byte `0` as a fresh [`ByteArena`].
    #[inline]
    #[must_use]
    pub fn as_arena(&mut self) -> ByteArena<'_> {
        ByteArena::new(self.as_bytes_mut())
    }

    /// Creates a non-zeroing [`RawByteCursor`] for simultaneous mutable access to [`ArenaRegion`]s.
    #[inline]
    #[must_use]
    pub fn raw_cursor(&mut self) -> RawByteCursor<'_> {
        RawByteCursor::new(self.as_bytes_mut())
    }

    /// Immutable slice view of a reserved [`ArenaRegion`].
    #[inline]
    #[must_use]
    pub fn region_slice<T: Pod>(&self, region: ArenaRegion<T>) -> &[T] {
        let byte_len = region.byte_len();
        if byte_len == 0 {
            return &[];
        }
        let end = region.byte_offset + byte_len;
        bytemuck::cast_slice(&self.as_bytes()[region.byte_offset..end])
    }

    /// Mutable slice view of a reserved [`ArenaRegion`].
    #[inline]
    pub fn region_slice_mut<T: Pod>(&mut self, region: ArenaRegion<T>) -> &mut [T] {
        let byte_len = region.byte_len();
        if byte_len == 0 {
            return &mut [];
        }
        let end = region.byte_offset + byte_len;
        bytemuck::cast_slice_mut(&mut self.as_bytes_mut()[region.byte_offset..end])
    }

    /// Reads a single boolean flag from a reserved [`FlagRegion`].
    #[inline]
    #[must_use]
    pub fn flag_get(&self, region: FlagRegion, index: usize) -> bool {
        self.region_slice(region.inner)[index] != 0
    }

    /// Mutable [`FlagSlice`] view of a reserved [`FlagRegion`].
    #[inline]
    pub fn flags_mut(&mut self, region: FlagRegion) -> FlagSlice<'_> {
        FlagSlice::from_existing_bytes(self.region_slice_mut(region.inner))
    }

    /// Immutable word view of a reserved [`BitRegion`].
    #[inline]
    #[must_use]
    pub fn bit_words(&self, region: BitRegion) -> &[u64] {
        self.region_slice(region.words)
    }

    /// Mutable [`BitSlice`] view of a reserved [`BitRegion`].
    #[inline]
    pub fn bits_mut(&mut self, region: BitRegion) -> BitSlice<'_> {
        BitSlice::from_existing_words(self.region_slice_mut(region.words), region.bit_len)
    }
}

impl SolverWorkspace for OwnedByteArena {
    #[inline]
    fn byte_capacity(&self) -> usize {
        self.byte_len
    }

    #[inline]
    fn remaining_bytes(&self) -> usize {
        OwnedByteArena::remaining_bytes(self)
    }

    #[inline]
    fn reset(&mut self) {
        self.offset = 0;
    }

    #[inline]
    fn reborrow(&mut self) -> ByteArena<'_> {
        let start = self.offset;
        let end = self.byte_len;
        let raw = &mut bytemuck::cast_slice_mut::<u128, u8>(&mut self.words)[start..end];
        ByteArena::new(raw)
    }

    #[inline]
    fn cursor(&mut self) -> ByteCursor<'_> {
        let start = self.offset;
        let end = self.byte_len;
        let raw = &mut bytemuck::cast_slice_mut::<u128, u8>(&mut self.words)[start..end];
        ByteCursor::new(raw)
    }

    fn alloc_slice<T: Pod + Zeroable>(&mut self, len: usize) -> Result<&mut [T], ArenaError> {
        let start_off = self.offset;
        let end_cap = self.byte_len;
        let full = &mut bytemuck::cast_slice_mut::<u128, u8>(&mut self.words)[..end_cap];
        let tail = &full[start_off..];
        let addr = tail.as_ptr() as usize;
        let (pad, byte_len, total) = plan_slice_bytes::<T>(addr, tail.len(), len)?;
        if byte_len == 0 {
            return Ok(&mut []);
        }
        let slice_start = start_off + pad;
        let slice_end = slice_start + byte_len;
        self.offset = start_off + total;
        let raw = &mut full[slice_start..slice_end];
        raw.fill(0);
        bytemuck::try_cast_slice_mut(raw).map_err(|_| ArenaError::InvalidAlignment {
            align: core::mem::align_of::<T>(),
        })
    }
}
