#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
//! Stack-allocated `#![no_std]` error diagnostics for [`crate::ByteArena`] and [`crate::OwnedByteArena`].

use core::fmt;
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

/// Typed `#![no_std]` error returned when arena alignment, bounds, or allocation checks fail.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ArenaError {
    /// Requested alignment is zero or not a power of two.
    InvalidAlignment {
        /// Invalid alignment value in bytes.
        align: usize,
    },
    /// Byte length or alignment padding overflowed `usize`.
    ArithmeticOverflow,
    /// Remaining unallocated bytes in the arena cannot satisfy the aligned request.
    InsufficientBytes {
        /// Total bytes required including alignment padding.
        required: usize,
        /// Unallocated bytes remaining in the arena.
        available: usize,
        /// Target element alignment in bytes.
        align: usize,
    },
    /// Attempted to rewind to a checkpoint beyond the arena's current cursor offset.
    InvalidCheckpoint {
        /// Offset recorded inside the supplied [`crate::ArenaCheckpoint`].
        checkpoint_offset: usize,
        /// Current byte offset of the arena cursor.
        current_offset: usize,
    },
    /// Fallible heap allocation (`alloc` feature) failed for the requested byte count.
    AllocationFailed {
        /// Number of bytes requested from the global allocator.
        requested_bytes: usize,
    },
    /// Flag or bit index exceeded the view's logical length.
    IndexOutOfBounds {
        /// Requested element or bit index.
        index: usize,
        /// Logical view length.
        len: usize,
    },
    /// Persistent [`crate::ArenaRegion`] was requested out of reservation order or past buffer end.
    RegionOutOfOrder {
        /// Byte offset of the requested region.
        region_offset: usize,
        /// Current cursor byte offset.
        cursor_offset: usize,
    },
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlignment { align } => {
                write!(f, "invalid arena alignment {align}: must be a power of two")
            }
            Self::ArithmeticOverflow => {
                f.write_str("arena byte size or alignment calculation overflowed usize")
            }
            Self::InsufficientBytes {
                required,
                available,
                align,
            } => write!(
                f,
                "insufficient arena capacity for {align}-byte aligned slice: required {required} bytes, available {available}"
            ),
            Self::InvalidCheckpoint {
                checkpoint_offset,
                current_offset,
            } => write!(
                f,
                "invalid arena checkpoint offset {checkpoint_offset} exceeds current cursor {current_offset}"
            ),
            Self::AllocationFailed { requested_bytes } => {
                write!(
                    f,
                    "owned byte arena allocation failed for {requested_bytes} bytes"
                )
            }
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "arena view index {index} out of bounds for length {len}")
            }
            Self::RegionOutOfOrder {
                region_offset,
                cursor_offset,
            } => write!(
                f,
                "arena region offset {region_offset} out of order at cursor {cursor_offset}"
            ),
        }
    }
}

impl core::error::Error for ArenaError {}

impl Diagnostic for ArenaError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "ArenaError");
        visitor.record_display("description", self);
        match self {
            Self::InvalidAlignment { align } => {
                visitor.record_str("code", "MEM-001");
                visitor.record_usize("align", *align);
            }
            Self::ArithmeticOverflow => {
                visitor.record_str("code", "MEM-002");
            }
            Self::InsufficientBytes {
                required,
                available,
                align,
            } => {
                visitor.record_str("code", "MEM-003");
                visitor.record_usize("required", *required);
                visitor.record_usize("available", *available);
                visitor.record_usize("align", *align);
            }
            Self::InvalidCheckpoint {
                checkpoint_offset,
                current_offset,
            } => {
                visitor.record_str("code", "MEM-004");
                visitor.record_usize("checkpoint_offset", *checkpoint_offset);
                visitor.record_usize("current_offset", *current_offset);
            }
            Self::AllocationFailed { requested_bytes } => {
                visitor.record_str("code", "MEM-005");
                visitor.record_usize("requested_bytes", *requested_bytes);
            }
            Self::IndexOutOfBounds { index, len } => {
                visitor.record_str("code", "MEM-006");
                visitor.record_usize("index", *index);
                visitor.record_usize("len", *len);
            }
            Self::RegionOutOfOrder {
                region_offset,
                cursor_offset,
            } => {
                visitor.record_str("code", "MEM-007");
                visitor.record_usize("region_offset", *region_offset);
                visitor.record_usize("cursor_offset", *cursor_offset);
            }
        }
    }
}
