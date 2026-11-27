#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
//! Fallible `#![no_std]` bump arena (`ByteArena`) and multi-slice splitter (`ByteCursor`).

use crate::error::ArenaError;
use crate::views::{BitSlice, FlagSlice};
use bytemuck::{Pod, Zeroable};

/// Saved cursor position inside a [`ByteArena`] or `OwnedByteArena`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaCheckpoint {
    pub(crate) offset: usize,
}

impl ArenaCheckpoint {
    /// Byte offset recorded when this checkpoint was created.
    #[inline]
    #[must_use]
    pub const fn byte_offset(self) -> usize {
        self.offset
    }
}

/// Computes `(align - (addr % align)) % align` after validating `align.is_power_of_two()`.
#[inline]
pub(crate) const fn align_padding(addr: usize, align: usize) -> Result<usize, ArenaError> {
    if !align.is_power_of_two() {
        return Err(ArenaError::InvalidAlignment { align });
    }
    let rem = addr & (align - 1);
    if rem == 0 { Ok(0) } else { Ok(align - rem) }
}

/// Validates and computes `(padding, byte_len, total_bytes)` for `len` elements of `T` at `base_addr`.
#[inline]
pub(crate) fn plan_slice_bytes<T>(
    base_addr: usize,
    available: usize,
    len: usize,
) -> Result<(usize, usize, usize), ArenaError> {
    let align = core::mem::align_of::<T>();
    if !align.is_power_of_two() {
        return Err(ArenaError::InvalidAlignment { align });
    }
    let elem_size = core::mem::size_of::<T>();
    let byte_len = len
        .checked_mul(elem_size)
        .ok_or(ArenaError::ArithmeticOverflow)?;
    if byte_len == 0 {
        return Ok((0, 0, 0));
    }
    let pad = align_padding(base_addr, align)?;
    let total = pad
        .checked_add(byte_len)
        .ok_or(ArenaError::ArithmeticOverflow)?;
    if total > available {
        return Err(ArenaError::InsufficientBytes {
            required: total,
            available,
            align,
        });
    }
    Ok((pad, byte_len, total))
}

/// Returns the byte count needed for `len` elements of `T` assuming worst-case `align - 1` padding.
#[inline]
pub const fn padded_slice_bytes<T>(len: usize) -> Option<usize> {
    let elem_size = core::mem::size_of::<T>();
    let align = core::mem::align_of::<T>();
    let Some(raw) = len.checked_mul(elem_size) else {
        return None;
    };
    if raw == 0 {
        return Some(0);
    }
    raw.checked_add(align.saturating_sub(1))
}

/// Fallible `#![no_std]` bump allocator over a caller-borrowed `&'a mut [u8]` slice.
#[derive(Debug)]
pub struct ByteArena<'a> {
    buf: &'a mut [u8],
    offset: usize,
}

impl<'a> ByteArena<'a> {
    /// Wraps a mutable byte slice with its cursor positioned at byte `0`.
    #[inline]
    #[must_use]
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, offset: 0 }
    }

    /// Wraps a typed `Pod` slice (e.g. `&mut [f64]`) as a byte arena preserving alignment.
    #[inline]
    #[must_use]
    pub fn from_pod_slice<T: Pod>(slice: &'a mut [T]) -> Self {
        Self {
            buf: bytemuck::cast_slice_mut(slice),
            offset: 0,
        }
    }

    /// Creates a zero-capacity arena over `&mut []`.
    #[inline]
    #[must_use]
    pub fn empty() -> Self {
        Self {
            buf: &mut [],
            offset: 0,
        }
    }

    /// Total capacity of this arena frame in bytes.
    #[inline]
    #[must_use]
    pub const fn byte_capacity(&self) -> usize {
        self.buf.len()
    }

    /// Bytes currently consumed from this arena frame (including alignment padding).
    #[inline]
    #[must_use]
    pub const fn used_bytes(&self) -> usize {
        self.offset
    }

    /// Unallocated bytes remaining from the current cursor to the end of this arena frame.
    #[inline]
    #[must_use]
    pub const fn remaining_bytes(&self) -> usize {
        self.buf.len().saturating_sub(self.offset)
    }

    /// Captures the current cursor offset for subsequent [`Self::rewind`].
    #[inline]
    #[must_use]
    pub const fn checkpoint(&self) -> ArenaCheckpoint {
        ArenaCheckpoint {
            offset: self.offset,
        }
    }

    /// Rewinds the cursor back to `checkpoint`, reclaiming all bytes allocated after it.
    ///
    /// # Errors
    /// Returns [`ArenaError::InvalidCheckpoint`] if `checkpoint` lies ahead of `self.used_bytes()`.
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

    /// Resets the cursor back to byte `0`.
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Creates a scoped sub-arena over the remaining unallocated bytes.
    ///
    /// When the returned `ByteArena` drops at the end of its scope, `self.used_bytes()` remains
    /// unchanged at its pre-`reborrow` position (automatic cursor rewind on drop).
    #[inline]
    #[must_use]
    pub fn reborrow(&mut self) -> ByteArena<'_> {
        ByteArena {
            buf: &mut self.buf[self.offset..],
            offset: 0,
        }
    }

    /// Creates a [`ByteCursor`] over the unallocated tail for carving multiple simultaneous slices.
    #[inline]
    #[must_use]
    pub fn cursor(&mut self) -> ByteCursor<'_> {
        ByteCursor {
            tail: &mut self.buf[self.offset..],
        }
    }

    /// Consumes `self` into a [`ByteCursor`] with the full `'a` lifetime of the underlying buffer.
    #[inline]
    #[must_use]
    pub fn into_cursor(self) -> ByteCursor<'a> {
        ByteCursor {
            tail: &mut self.buf[self.offset..],
        }
    }

    /// Carves an aligned, zero-initialized `&mut [T]` of length `len`, advancing `self`'s cursor.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on arithmetic overflow, invalid alignment, or insufficient capacity.
    pub fn alloc_slice<T: Pod + Zeroable>(&mut self, len: usize) -> Result<&mut [T], ArenaError> {
        let tail = &mut self.buf[self.offset..];
        let addr = tail.as_ptr() as usize;
        let (pad, byte_len, total) = plan_slice_bytes::<T>(addr, tail.len(), len)?;
        if byte_len == 0 {
            return Ok(&mut []);
        }
        let start = self.offset + pad;
        let end = start + byte_len;
        self.offset += total;
        let raw = &mut self.buf[start..end];
        raw.fill(0);
        bytemuck::try_cast_slice_mut(raw).map_err(|_| ArenaError::InvalidAlignment {
            align: core::mem::align_of::<T>(),
        })
    }

    /// Carves a zero-initialized [`FlagSlice`] of `len` boolean flags, advancing `self`'s cursor.
    ///
    /// # Errors
    /// Returns [`ArenaError`] if fewer than `len` bytes remain.
    #[inline]
    pub fn alloc_flags(&mut self, len: usize) -> Result<FlagSlice<'_>, ArenaError> {
        let bytes = self.alloc_slice::<u8>(len)?;
        Ok(FlagSlice::from_existing_bytes(bytes))
    }

    /// Carves a zero-initialized [`BitSlice`] of `bit_len` packed bits, advancing `self`'s cursor.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on word overflow or insufficient aligned `u64` capacity.
    #[inline]
    pub fn alloc_bits(&mut self, bit_len: usize) -> Result<BitSlice<'_>, ArenaError> {
        let words_needed = BitSlice::required_words(bit_len);
        let words = self.alloc_slice::<u64>(words_needed)?;
        Ok(BitSlice::from_existing_words(words, bit_len))
    }

    /// Carves an aligned `&mut [T]` and returns a sub-arena over the remaining unallocated suffix.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on overflow or insufficient capacity without advancing `self`.
    pub fn split_slice<T: Pod + Zeroable>(
        &mut self,
        len: usize,
    ) -> Result<(&mut [T], ByteArena<'_>), ArenaError> {
        let tail = &mut self.buf[self.offset..];
        let addr = tail.as_ptr() as usize;
        let (pad, byte_len, _) = plan_slice_bytes::<T>(addr, tail.len(), len)?;
        if byte_len == 0 {
            return Ok((&mut [], ByteArena::new(tail)));
        }
        let (_, after_pad) = tail.split_at_mut(pad);
        let (carved, rest) = after_pad.split_at_mut(byte_len);
        carved.fill(0);
        let typed =
            bytemuck::try_cast_slice_mut(carved).map_err(|_| ArenaError::InvalidAlignment {
                align: core::mem::align_of::<T>(),
            })?;
        Ok((typed, ByteArena::new(rest)))
    }
}

/// Linear splitter carving multiple disjoint `'a`-lifetime heterogeneous slices in one scope.
#[derive(Debug)]
pub struct ByteCursor<'a> {
    tail: &'a mut [u8],
}

impl<'a> ByteCursor<'a> {
    /// Wraps a mutable byte slice as a linear multi-slice splitter.
    #[inline]
    #[must_use]
    pub fn new(tail: &'a mut [u8]) -> Self {
        Self { tail }
    }

    /// Unallocated bytes remaining in this cursor.
    #[inline]
    #[must_use]
    pub const fn remaining_bytes(&self) -> usize {
        self.tail.len()
    }

    /// Carves an aligned, zero-initialized `&'a mut [T]` of length `len`.
    ///
    /// On error, `self` retains all previously unallocated bytes unchanged.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on arithmetic overflow or insufficient capacity.
    pub fn alloc_slice<T: Pod + Zeroable>(
        &mut self,
        len: usize,
    ) -> Result<&'a mut [T], ArenaError> {
        let addr = self.tail.as_ptr() as usize;
        let (pad, byte_len, _) = plan_slice_bytes::<T>(addr, self.tail.len(), len)?;
        if byte_len == 0 {
            return Ok(&mut []);
        }
        let full = core::mem::take(&mut self.tail);
        let (_, after_pad) = full.split_at_mut(pad);
        let (carved, rest) = after_pad.split_at_mut(byte_len);
        self.tail = rest;
        carved.fill(0);
        bytemuck::try_cast_slice_mut(carved).map_err(|_| ArenaError::InvalidAlignment {
            align: core::mem::align_of::<T>(),
        })
    }

    /// Carves a zero-initialized [`FlagSlice<'a>`] of `len` boolean flags.
    ///
    /// # Errors
    /// Returns [`ArenaError`] if fewer than `len` bytes remain.
    #[inline]
    pub fn alloc_flags(&mut self, len: usize) -> Result<FlagSlice<'a>, ArenaError> {
        let bytes = self.alloc_slice::<u8>(len)?;
        Ok(FlagSlice::from_existing_bytes(bytes))
    }

    /// Carves a zero-initialized [`BitSlice<'a>`] of `bit_len` packed bits.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on overflow or insufficient aligned `u64` capacity.
    #[inline]
    pub fn alloc_bits(&mut self, bit_len: usize) -> Result<BitSlice<'a>, ArenaError> {
        let words = self.alloc_slice::<u64>(BitSlice::required_words(bit_len))?;
        Ok(BitSlice::from_existing_words(words, bit_len))
    }

    /// Borrows the remaining unallocated bytes as a scoped [`ByteArena`].
    #[inline]
    #[must_use]
    pub fn reborrow(&mut self) -> ByteArena<'_> {
        ByteArena::new(self.tail)
    }

    /// Converts the remaining unallocated tail into a [`ByteArena<'a>`].
    #[inline]
    #[must_use]
    pub fn into_arena(self) -> ByteArena<'a> {
        ByteArena::new(self.tail)
    }

    /// Consumes this cursor and returns the remaining unallocated raw byte slice.
    #[inline]
    #[must_use]
    pub fn into_bytes(self) -> &'a mut [u8] {
        self.tail
    }
}
