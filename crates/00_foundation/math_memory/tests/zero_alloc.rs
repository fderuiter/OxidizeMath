// [cite:math_memory]
//! Integration test enforcing zero dynamic heap allocations inside `VerifiedEngine::run_verified`
//! with `VerifiedAllocator` installed as the process `#[global_allocator]`.

#![cfg(feature = "std")]

use math_memory::{ArenaLayout, ByteArena, OwnedByteArena};
use num_complex::Complex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOC: VerifiedAllocator = VerifiedAllocator;

#[test]
fn verified_allocator_confirms_zero_heap_allocations_in_byte_arena() {
    let mut layout = ArenaLayout::new();
    let _ = layout.add_slice::<usize>(8).unwrap();
    let _ = layout.add_slice::<f64>(8).unwrap();
    let _ = layout.add_slice::<Complex<f64>>(4).unwrap();
    let _ = layout.add_slice::<u64>(4).unwrap();
    let mut owned = OwnedByteArena::with_capacity(layout.total_bytes() + 64);

    let (sum, _metrics) = VerifiedEngine::run_verified(|| {
        let mut scope = owned.reborrow();
        let idx = scope.alloc_slice::<usize>(4).unwrap();
        let vals = scope.alloc_slice::<f64>(4).unwrap();
        let zs = scope.alloc_slice::<Complex<f64>>(2).unwrap();
        let mut flags = scope.alloc_flags(8).unwrap();
        let mut bits = scope.alloc_bits(64).unwrap();
        idx[0] = 3;
        vals[0] = 7.5;
        zs[0] = Complex::new(1.25, -0.5);
        flags.set(2, true);
        bits.set(63, true);

        let mut stack_buf = [0_u8; 64];
        let mut stack_arena = ByteArena::new(&mut stack_buf);
        let s = stack_arena.alloc_slice::<f32>(4).unwrap();
        s[0] = 2.0;

        (idx[0] as f64)
            + vals[0]
            + zs[0].re
            + f64::from(s[0])
            + (flags.count_ones() as f64)
            + (bits.count_ones() as f64)
    });

    assert!((sum - 15.75).abs() < 1e-12);
}
