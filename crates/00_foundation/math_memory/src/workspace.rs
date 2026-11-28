#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
//! Unified `#![no_std]` [`SolverWorkspace`] contract shared across borrowed and owned byte arenas.

use crate::arena::{ByteArena, ByteCursor};
use crate::error::ArenaError;
use crate::views::{BitSlice, FlagSlice};
use bytemuck::{Pod, Zeroable};

/// Shared zero-allocation workspace contract for `#![no_std]` and heap-preallocated solvers.
pub trait SolverWorkspace {
    /// Total byte capacity of the active workspace frame.
    fn byte_capacity(&self) -> usize;

    /// Unallocated bytes remaining from the current cursor to the end of the active frame.
    fn remaining_bytes(&self) -> usize;

    /// Resets the workspace cursor back to byte `0`.
    fn reset(&mut self);

    /// Creates a scoped [`ByteArena`] over the remaining unallocated bytes that rewinds on drop.
    fn reborrow(&mut self) -> ByteArena<'_>;

    /// Creates a [`ByteCursor`] over the remaining unallocated bytes for multi-slice carving.
    #[inline]
    fn cursor(&mut self) -> ByteCursor<'_> {
        self.reborrow().into_cursor()
    }

    /// Allocates an aligned, zero-initialized `&mut [T]` slice advancing `self`'s cursor.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on arithmetic overflow, invalid alignment, or insufficient capacity.
    fn alloc_slice<T: Pod + Zeroable>(&mut self, len: usize) -> Result<&mut [T], ArenaError>;

    /// Allocates a zero-initialized [`FlagSlice`] advancing `self`'s cursor.
    ///
    /// # Errors
    /// Returns [`ArenaError`] if fewer than `len` bytes remain.
    #[inline]
    fn alloc_flags(&mut self, len: usize) -> Result<FlagSlice<'_>, ArenaError> {
        let bytes = self.alloc_slice::<u8>(len)?;
        Ok(FlagSlice::from_existing_bytes(bytes))
    }

    /// Allocates a zero-initialized [`BitSlice`] of `bit_len` bits advancing `self`'s cursor.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on overflow or insufficient aligned `u64` capacity.
    #[inline]
    fn alloc_bits(&mut self, bit_len: usize) -> Result<BitSlice<'_>, ArenaError> {
        let words = self.alloc_slice::<u64>(BitSlice::required_words(bit_len))?;
        Ok(BitSlice::from_existing_words(words, bit_len))
    }
}

impl SolverWorkspace for ByteArena<'_> {
    #[inline]
    fn byte_capacity(&self) -> usize {
        ByteArena::byte_capacity(self)
    }

    #[inline]
    fn remaining_bytes(&self) -> usize {
        ByteArena::remaining_bytes(self)
    }

    #[inline]
    fn reset(&mut self) {
        ByteArena::reset(self);
    }

    #[inline]
    fn reborrow(&mut self) -> ByteArena<'_> {
        ByteArena::reborrow(self)
    }

    #[inline]
    fn cursor(&mut self) -> ByteCursor<'_> {
        ByteArena::cursor(self)
    }

    #[inline]
    fn alloc_slice<T: Pod + Zeroable>(&mut self, len: usize) -> Result<&mut [T], ArenaError> {
        ByteArena::alloc_slice(self, len)
    }
}
