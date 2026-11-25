# `math_memory`

## 1. Overview & Architecture

`math_memory` is a Tier 0 `#![no_std]` foundation crate (`crates/00_foundation/math_memory`) providing fallible, zero-allocation byte-arena scratch partitioning and unified workspace contracts across OxidizeMath.

Key guarantees:
- **Strict `#![no_std]` & `#![forbid(unsafe_code)]`**: Compiles with `default = []` (no `alloc` or `std` required) and delegates typed slice reinterpretation exclusively to safe `bytemuck::Pod` and `bytemuck::Zeroable` casts.
- **Checked Alignment & Overflow Safety**: Every allocation computes byte lengths and alignment padding with `usize::checked_mul` and `usize::checked_add`, returning `Err(ArenaError)` on arithmetic overflow or buffer exhaustion rather than panicking.
- **Safe Boolean & Bit Views**: Provides [`FlagSlice`] (over `&mut [u8]`) and [`BitSlice`] (over `&mut [u64]`) so boolean scratch state never reinterprets arbitrary byte patterns as `&mut [bool]`.
- **Scoped Reborrowing & Reusable Containers**: [`ByteArena`] supports [`ArenaCheckpoint`] rewind and [`ByteArena::reborrow`] frames ([`ArenaScope`]) whose cursor resets automatically on drop. When the optional `alloc` feature is enabled, [`OwnedByteArena`] and [`ArenaLayout`] back persistent reusable workspaces with a single 8-byte-aligned buffer.

## 2. Developer Instructions

Check bare `#![no_std]`, `alloc`, and full `std` configurations:

```bash
cargo check -p math_memory --no-default-features
cargo check -p math_memory --no-default-features --features alloc
cargo test -p math_memory --all-features
```

Example carving heterogeneous scratch slices from a stack buffer:

```rust
use math_memory::{ByteArena, SolverWorkspace};
use num_complex::Complex;

let mut raw = [0u8; 256];
let mut arena = ByteArena::new(&mut raw);
{
    let mut scope = arena.reborrow();
    let pivots = scope.alloc_slice::<usize>(4).unwrap();
    let coeffs = scope.alloc_slice::<f64>(4).unwrap();
    let spectrum = scope.alloc_slice::<Complex<f64>>(2).unwrap();
    let mut visited = scope.alloc_flags(16).unwrap();
    pivots[0] = 1;
    coeffs[0] = 2.5;
    spectrum[0] = Complex::new(1.0, -1.0);
    visited.set(3, true);
    assert!(visited.get(3));
}
assert_eq!(arena.used(), 0);
```
