// [cite:math_memory]
#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod arena;
pub mod error;
pub mod owned;
pub mod views;
pub mod workspace;

pub use arena::{ArenaCheckpoint, ArenaScope, ByteArena, padded_slice_bytes, slice_byte_len};
pub use bytemuck::{Pod, Zeroable};
pub use error::ArenaError;
#[cfg(feature = "alloc")]
pub use owned::OwnedByteArena;
pub use owned::{ArenaLayout, ArenaRegion};
pub use views::{BitSlice, FlagSlice};
pub use workspace::SolverWorkspace;

#[cfg(test)]
mod tests;
