// [cite:math_memory]
//! Unified [`SolverWorkspace`] trait bridging borrowed byte slices, [`ByteArena`], [`ArenaScope`],
//! and `alloc`-gated [`crate::OwnedByteArena`].

use crate::arena::{ArenaScope, ByteArena};
#[cfg(feature = "alloc")]
use crate::owned::OwnedByteArena;

/// Shared zero-allocation workspace contract for byte-arena access, scoped reborrowing, capacity
/// queries, and cursor resets across Tier 0–Tier 3 solvers.
pub trait SolverWorkspace {
    /// Borrows the workspace's active backing memory as a [`ByteArena`].
    fn as_byte_arena(&mut self) -> ByteArena<'_>;

    /// Total byte capacity managed by this workspace.
    fn byte_capacity(&self) -> usize;

    /// Resets any active allocation cursor so full scratch capacity can be reused.
    fn reset(&mut self);

    /// Opens a scoped [`ArenaScope`] whose carved slices release automatically when dropped.
    #[inline]
    fn reborrow(&mut self) -> ArenaScope<'_> {
        self.as_byte_arena().into_scope()
    }
}

impl SolverWorkspace for ByteArena<'_> {
    #[inline]
    fn as_byte_arena(&mut self) -> ByteArena<'_> {
        let scope = self.reborrow();
        scope.into_byte_arena()
    }

    #[inline]
    fn byte_capacity(&self) -> usize {
        self.remaining()
    }

    #[inline]
    fn reset(&mut self) {
        Self::reset(self);
    }

    #[inline]
    fn reborrow(&mut self) -> ArenaScope<'_> {
        Self::reborrow(self)
    }
}

impl SolverWorkspace for ArenaScope<'_> {
    #[inline]
    fn as_byte_arena(&mut self) -> ByteArena<'_> {
        self.reborrow().into_byte_arena()
    }

    #[inline]
    fn byte_capacity(&self) -> usize {
        self.remaining()
    }

    #[inline]
    fn reset(&mut self) {}

    #[inline]
    fn reborrow(&mut self) -> ArenaScope<'_> {
        Self::reborrow(self)
    }
}

impl SolverWorkspace for [u8] {
    #[inline]
    fn as_byte_arena(&mut self) -> ByteArena<'_> {
        ByteArena::new(self)
    }

    #[inline]
    fn byte_capacity(&self) -> usize {
        self.len()
    }

    #[inline]
    fn reset(&mut self) {}
}

impl<const N: usize> SolverWorkspace for [u8; N] {
    #[inline]
    fn as_byte_arena(&mut self) -> ByteArena<'_> {
        ByteArena::new(self.as_mut_slice())
    }

    #[inline]
    fn byte_capacity(&self) -> usize {
        N
    }

    #[inline]
    fn reset(&mut self) {}
}

#[cfg(feature = "alloc")]
impl SolverWorkspace for OwnedByteArena {
    #[inline]
    fn as_byte_arena(&mut self) -> ByteArena<'_> {
        self.as_arena()
    }

    #[inline]
    fn byte_capacity(&self) -> usize {
        Self::byte_capacity(self)
    }

    #[inline]
    fn reset(&mut self) {}

    #[inline]
    fn reborrow(&mut self) -> ArenaScope<'_> {
        Self::reborrow(self)
    }
}

impl<W: SolverWorkspace> SolverWorkspace for &mut W {
    #[inline]
    fn as_byte_arena(&mut self) -> ByteArena<'_> {
        (**self).as_byte_arena()
    }

    #[inline]
    fn byte_capacity(&self) -> usize {
        (**self).byte_capacity()
    }

    #[inline]
    fn reset(&mut self) {
        (**self).reset();
    }

    #[inline]
    fn reborrow(&mut self) -> ArenaScope<'_> {
        (**self).reborrow()
    }
}
