// [cite:math_memory]
//! Error types returned by fallible [`crate::ByteArena`] and [`crate::ArenaScope`] operations.

use core::fmt;

/// Fallible error returned when carving or rewinding byte-arena scratch regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ArenaError {
    /// Requested element count multiplied by element size (or alignment padding) overflowed `usize`.
    CapacityOverflow {
        /// Size in bytes of a single element `size_of::<T>()`.
        elem_size: usize,
        /// Requested element count.
        count: usize,
    },
    /// Remaining arena byte capacity was smaller than the aligned byte requirement.
    InsufficientCapacity {
        /// Total bytes required including alignment padding.
        requested: usize,
        /// Unallocated bytes remaining in the arena.
        available: usize,
    },
    /// Requested alignment was zero or not a power of two.
    InvalidAlignment {
        /// Invalid byte alignment value.
        align: usize,
    },
    /// Attempted to rewind to a checkpoint beyond the arena's current cursor offset.
    InvalidCheckpoint {
        /// Byte offset stored in the supplied checkpoint.
        checkpoint_offset: usize,
        /// Current arena cursor offset in bytes.
        current_offset: usize,
    },
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::CapacityOverflow { elem_size, count } => write!(
                f,
                "byte-arena capacity overflow for {count} elements of {elem_size} bytes"
            ),
            Self::InsufficientCapacity {
                requested,
                available,
            } => write!(
                f,
                "insufficient byte-arena capacity: requested {requested} bytes, only {available} bytes available"
            ),
            Self::InvalidAlignment { align } => {
                write!(
                    f,
                    "invalid byte-arena alignment {align} (must be a power of two)"
                )
            }
            Self::InvalidCheckpoint {
                checkpoint_offset,
                current_offset,
            } => write!(
                f,
                "invalid arena checkpoint offset {checkpoint_offset} beyond current offset {current_offset}"
            ),
        }
    }
}

impl core::error::Error for ArenaError {}

impl math_diagnostics::Diagnostic for ArenaError {
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        visitor.record_str("error_type", "ArenaError");
        visitor.record_display("description", self);
        match *self {
            Self::CapacityOverflow { elem_size, count } => {
                visitor.record_str("code", "ARENA-001");
                visitor.record_usize("elem_size", elem_size);
                visitor.record_usize("count", count);
            }
            Self::InsufficientCapacity {
                requested,
                available,
            } => {
                visitor.record_str("code", "ARENA-002");
                visitor.record_usize("requested", requested);
                visitor.record_usize("available", available);
            }
            Self::InvalidAlignment { align } => {
                visitor.record_str("code", "ARENA-003");
                visitor.record_usize("align", align);
            }
            Self::InvalidCheckpoint {
                checkpoint_offset,
                current_offset,
            } => {
                visitor.record_str("code", "ARENA-004");
                visitor.record_usize("checkpoint_offset", checkpoint_offset);
                visitor.record_usize("current_offset", current_offset);
            }
        }
    }
}
