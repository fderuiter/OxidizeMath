// [cite:math_memory]
//! Fallible [`ByteArena`] and scoped [`ArenaScope`] bump allocators over borrowed byte slices.

use bytemuck::{Pod, Zeroable};

use crate::error::ArenaError;
use crate::views::{BitSlice, FlagSlice};

/// Computes `count * size_of::<T>()` using checked multiplication (`[cite:math_memory]`).
#[inline]
pub fn slice_byte_len<T>(count: usize) -> Result<usize, ArenaError> {
    let elem_size = core::mem::size_of::<T>();
    let total = count
        .checked_mul(elem_size)
        .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
    debug_assert!(elem_size == 0 || total / elem_size == count);
    Ok(total)
}

/// Computes a conservative upper-bound byte budget for `count` elements of `T`,
/// including up to `align_of::<T>() - 1` bytes of alignment padding (`[cite:math_memory]`).
#[inline]
pub fn padded_slice_bytes<T>(count: usize) -> Result<usize, ArenaError> {
    let elem_size = core::mem::size_of::<T>();
    let raw = slice_byte_len::<T>(count)?;
    if raw == 0 {
        return Ok(0);
    }
    let max_pad = core::mem::align_of::<T>().saturating_sub(1);
    let padded = raw
        .checked_add(max_pad)
        .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
    debug_assert!(padded >= raw);
    Ok(padded)
}

#[inline]
pub(crate) fn align_padding(addr: usize, align: usize) -> Result<usize, ArenaError> {
    if !align.is_power_of_two() {
        return Err(ArenaError::InvalidAlignment { align });
    }
    let rem = addr & (align - 1);
    let pad = if rem == 0 { 0 } else { align - rem };
    debug_assert!(pad < align);
    Ok(pad)
}

fn carve_bytes(
    tail: &mut [u8],
    elem_size: usize,
    align: usize,
    count: usize,
    zero_init: bool,
) -> Result<(&mut [u8], &mut [u8], usize), ArenaError> {
    if !align.is_power_of_two() {
        return Err(ArenaError::InvalidAlignment { align });
    }
    let byte_len = count
        .checked_mul(elem_size)
        .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
    if byte_len == 0 {
        return Ok((&mut [], tail, 0));
    }
    let pad = align_padding(tail.as_ptr() as usize, align)?;
    let total = pad
        .checked_add(byte_len)
        .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
    if total > tail.len() {
        return Err(ArenaError::InsufficientCapacity {
            requested: total,
            available: tail.len(),
        });
    }
    let (_padding, after_pad) = tail.split_at_mut(pad);
    let (region, rest) = after_pad.split_at_mut(byte_len);
    debug_assert_eq!((region.as_ptr() as usize) % align, 0);
    debug_assert_eq!(region.len(), byte_len);
    if zero_init {
        region.fill(0);
    }
    Ok((region, rest, total))
}

/// Saved byte-offset cursor produced by [`ByteArena::checkpoint`] for [`ByteArena::rewind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaCheckpoint {
    offset: usize,
}

impl ArenaCheckpoint {
    /// Byte offset recorded when the checkpoint was taken.
    #[inline]
    #[must_use]
    pub const fn offset(self) -> usize {
        self.offset
    }
}

/// Fallible bump allocator over a caller-borrowed `&'a mut [u8]` scratch buffer.
///
/// Supports explicit [`ArenaCheckpoint`] rewind as well as scoped [`ByteArena::reborrow`] frames
/// ([`ArenaScope`]) that partition multiple simultaneous disjoint slices and restore the parent
/// cursor automatically when dropped.
#[derive(Debug)]
pub struct ByteArena<'a> {
    buf: &'a mut [u8],
    offset: usize,
}

impl<'a> ByteArena<'a> {
    /// Wraps a mutable byte slice as a fresh [`ByteArena`] with cursor at `0`.
    #[inline]
    #[must_use]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, offset: 0 }
    }

    /// Wraps an existing typed [`Pod`] slice (e.g. `&mut [f64]`) as a byte-backed [`ByteArena`].
    #[inline]
    #[must_use]
    pub fn from_slice<T: Pod>(slice: &'a mut [T]) -> Self {
        Self::new(bytemuck::cast_slice_mut(slice))
    }

    /// Total byte capacity of the underlying buffer.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Bytes currently consumed from the front of the arena.
    #[inline]
    #[must_use]
    pub const fn used(&self) -> usize {
        self.offset
    }

    /// Unallocated bytes remaining after the current cursor.
    #[inline]
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.offset)
    }

    /// Returns `true` if zero unallocated bytes remain.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// Captures the current byte cursor so subsequent allocations can be undone via [`Self::rewind`].
    #[inline]
    #[must_use]
    pub const fn checkpoint(&self) -> ArenaCheckpoint {
        ArenaCheckpoint {
            offset: self.offset,
        }
    }

    /// Rewinds the arena cursor back to `checkpoint`.
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

    /// Resets the cursor to `0`, reclaiming the full buffer capacity.
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Creates a scoped [`ArenaScope`] over the remaining unallocated tail.
    ///
    /// Multiple disjoint typed slices can be carved simultaneously from the returned scope. When
    /// the scope drops at the end of its lexical block, `self.used()` remains unchanged at its
    /// pre-reborrow cursor.
    #[inline]
    pub fn reborrow(&mut self) -> ArenaScope<'_> {
        ArenaScope::new(&mut self.buf[self.offset..])
    }

    /// Consumes this [`ByteArena`] view into an [`ArenaScope`] with the same `'a` lifetime.
    #[inline]
    #[must_use]
    pub fn into_scope(self) -> ArenaScope<'a> {
        ArenaScope::new(&mut self.buf[self.offset..])
    }

    /// Carves a zero-initialized aligned slice of `count` elements of `T`, advancing `self.used()`.
    pub fn alloc_slice<T: Pod + Zeroable>(&mut self, count: usize) -> Result<&mut [T], ArenaError> {
        let elem_size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let (region, _rest, consumed) =
            carve_bytes(&mut self.buf[self.offset..], elem_size, align, count, true)?;
        self.offset = self
            .offset
            .checked_add(consumed)
            .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
        if region.is_empty() {
            return Ok(&mut []);
        }
        bytemuck::try_cast_slice_mut(region).map_err(|_| ArenaError::InvalidAlignment { align })
    }

    /// Carves a zero-initialized [`FlagSlice`] of `len` boolean flags, advancing `self.used()`.
    pub fn alloc_flags(&mut self, len: usize) -> Result<FlagSlice<'_>, ArenaError> {
        let bytes = self.alloc_slice::<u8>(len)?;
        Ok(FlagSlice::new_zeroed(bytes))
    }

    /// Carves a zero-initialized [`BitSlice`] of `bit_len` packed bits, advancing `self.used()`.
    pub fn alloc_bits(&mut self, bit_len: usize) -> Result<BitSlice<'_>, ArenaError> {
        let words_needed = BitSlice::required_words(bit_len);
        let words = self.alloc_slice::<u64>(words_needed)?;
        BitSlice::new_zeroed(words, bit_len)
    }
}

/// Scoped slice-splitting frame created by [`ByteArena::reborrow`] or [`ArenaScope::reborrow`].
///
/// Each allocation splits off an aligned prefix from the remaining `&'a mut [u8]` slice, allowing
/// callers to hold multiple heterogeneous mutable slices simultaneously in safe Rust while
/// automatically restoring the parent cursor when dropped.
#[derive(Debug)]
pub struct ArenaScope<'a> {
    rem: &'a mut [u8],
    consumed: usize,
}

impl<'a> ArenaScope<'a> {
    /// Creates a new [`ArenaScope`] over `rem`.
    #[inline]
    #[must_use]
    pub fn new(rem: &'a mut [u8]) -> Self {
        Self { rem, consumed: 0 }
    }

    /// Wraps a typed [`Pod`] slice directly as an [`ArenaScope`].
    #[inline]
    #[must_use]
    pub fn from_slice<T: Pod>(slice: &'a mut [T]) -> Self {
        Self::new(bytemuck::cast_slice_mut(slice))
    }

    /// Total byte capacity originally available to this scope.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.consumed.saturating_add(self.rem.len())
    }

    /// Bytes carved from this scope so far.
    #[inline]
    #[must_use]
    pub const fn used(&self) -> usize {
        self.consumed
    }

    /// Unallocated bytes remaining in this scope.
    #[inline]
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.rem.len()
    }

    /// Returns `true` if zero bytes remain in this scope.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rem.is_empty()
    }

    /// Opens a nested child scope over the remaining bytes; dropping the child restores `self`.
    #[inline]
    pub fn reborrow(&mut self) -> ArenaScope<'_> {
        ArenaScope::new(self.rem)
    }

    /// Converts the remaining bytes of this scope into a fresh [`ByteArena`].
    #[inline]
    #[must_use]
    pub fn into_byte_arena(self) -> ByteArena<'a> {
        ByteArena::new(self.rem)
    }

    fn alloc_raw<T: Pod>(
        &mut self,
        count: usize,
        zero_init: bool,
    ) -> Result<&'a mut [T], ArenaError> {
        let elem_size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let full = core::mem::take(&mut self.rem);
        match carve_bytes(full, elem_size, align, count, zero_init) {
            Ok((region, rest, delta)) => {
                self.rem = rest;
                self.consumed = self.consumed.saturating_add(delta);
                if region.is_empty() {
                    return Ok(&mut []);
                }
                bytemuck::try_cast_slice_mut(region)
                    .map_err(|_| ArenaError::InvalidAlignment { align })
            }
            Err(err) => Err(err),
        }
    }

    /// Carves a zero-initialized aligned slice `&'a mut [T]` of length `count`.
    pub fn alloc_slice<T: Pod + Zeroable>(
        &mut self,
        count: usize,
    ) -> Result<&'a mut [T], ArenaError> {
        let elem_size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let byte_len = count
            .checked_mul(elem_size)
            .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
        if byte_len == 0 {
            if !align.is_power_of_two() {
                return Err(ArenaError::InvalidAlignment { align });
            }
            return Ok(&mut []);
        }
        let pad = align_padding(self.rem.as_ptr() as usize, align)?;
        let total = pad
            .checked_add(byte_len)
            .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
        if total > self.rem.len() {
            return Err(ArenaError::InsufficientCapacity {
                requested: total,
                available: self.rem.len(),
            });
        }
        self.alloc_raw(count, true)
    }

    /// Carves an aligned slice `&'a mut [T]` preserving existing bytes in the backing buffer.
    pub fn retain_slice<T: Pod + Zeroable>(
        &mut self,
        count: usize,
    ) -> Result<&'a mut [T], ArenaError> {
        let elem_size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let byte_len = count
            .checked_mul(elem_size)
            .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
        if byte_len == 0 {
            if !align.is_power_of_two() {
                return Err(ArenaError::InvalidAlignment { align });
            }
            return Ok(&mut []);
        }
        let pad = align_padding(self.rem.as_ptr() as usize, align)?;
        let total = pad
            .checked_add(byte_len)
            .ok_or(ArenaError::CapacityOverflow { elem_size, count })?;
        if total > self.rem.len() {
            return Err(ArenaError::InsufficientCapacity {
                requested: total,
                available: self.rem.len(),
            });
        }
        self.alloc_raw(count, false)
    }

    /// Carves an aligned slice of length `src.len()` and copies `src` into it.
    pub fn alloc_slice_copy<T: Pod + Zeroable>(
        &mut self,
        src: &[T],
    ) -> Result<&'a mut [T], ArenaError> {
        let out = self.retain_slice::<T>(src.len())?;
        out.copy_from_slice(src);
        Ok(out)
    }

    /// Carves an aligned slice of length `count` and fills every element with `value`.
    pub fn alloc_slice_fill<T: Pod + Zeroable>(
        &mut self,
        count: usize,
        value: T,
    ) -> Result<&'a mut [T], ArenaError> {
        let out = self.retain_slice::<T>(count)?;
        out.fill(value);
        Ok(out)
    }

    /// Carves a zero-initialized [`FlagSlice`] of `len` boolean flags.
    pub fn alloc_flags(&mut self, len: usize) -> Result<FlagSlice<'a>, ArenaError> {
        let bytes = self.alloc_slice::<u8>(len)?;
        Ok(FlagSlice::new_zeroed(bytes))
    }

    /// Carves a zero-initialized [`BitSlice`] of `bit_len` packed bits.
    pub fn alloc_bits(&mut self, bit_len: usize) -> Result<BitSlice<'a>, ArenaError> {
        let words_needed = BitSlice::required_words(bit_len);
        let words = self.alloc_slice::<u64>(words_needed)?;
        BitSlice::new_zeroed(words, bit_len)
    }

    /// Fallible companion to [`Self::take_slice`], equivalent to [`Self::alloc_slice`].
    #[inline]
    pub fn try_take_slice<T: Pod + Zeroable>(
        &mut self,
        count: usize,
    ) -> Result<&'a mut [T], ArenaError> {
        self.alloc_slice(count)
    }

    /// Convenience helper carving `count` zero-initialized `T` elements when capacity was pre-verified.
    ///
    /// # Panics
    /// Panics if the scope lacks sufficient aligned byte capacity; prefer [`Self::try_take_slice`].
    #[inline]
    pub fn take_slice<T: Pod + Zeroable>(&mut self, count: usize) -> &'a mut [T] {
        self.alloc_slice(count)
            .expect("ArenaScope::take_slice exceeded preallocated workspace capacity")
    }
}
