#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
#![doc = include_str!("../README.md")]

#[cfg(feature = "alloc")]
extern crate alloc;

mod arena;
mod error;
#[cfg(feature = "alloc")]
mod owned;
mod region;
mod views;
mod workspace;

pub use arena::{ArenaCheckpoint, ByteArena, ByteCursor, padded_slice_bytes};
pub use bytemuck::{Pod, Zeroable};
pub use error::ArenaError;
#[cfg(feature = "alloc")]
pub use owned::OwnedByteArena;
pub use region::{ArenaLayout, ArenaRegion, BitRegion, FlagRegion, MAX_ARENA_ALIGN, RawByteCursor};
pub use views::{BitSlice, FlagSlice};
pub use workspace::SolverWorkspace;

// Compile-time verification that heterogeneous primitive and complex scalars satisfy `Pod + Zeroable`.
const _: () = {
    const fn assert_arena_pod<T: bytemuck::Pod + bytemuck::Zeroable>() {}
    assert_arena_pod::<usize>();
    assert_arena_pod::<u64>();
    assert_arena_pod::<f32>();
    assert_arena_pod::<f64>();
    assert_arena_pod::<num_complex::Complex<f32>>();
    assert_arena_pod::<num_complex::Complex<f64>>();
};
