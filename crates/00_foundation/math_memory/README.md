# `math_memory`

![Maturity: Validated](https://img.shields.io/badge/maturity-Validated-1f6feb.svg)

> **Maturity Tier:** `Validated` — Tier 0 `#![no_std]`, `#![forbid(unsafe_code)]` `bytemuck`-backed byte arena, boolean/bitset slice views, and unified `SolverWorkspace` substrate verified under `VerifiedAllocator`.

Zero-allocation Tier 0 `#![no_std]` foundation crate providing aligned bump allocation over borrowed byte slices, single-buffer multi-region layout planning, and unified `SolverWorkspace` abstractions.

## 1. Overview & Architecture

`math_memory` sits at Tier 0 (`crates/00_foundation/math_memory`) with `default = []`, `#![no_std]`, `#![forbid(unsafe_code)]`, and zero upward dependencies on Tier 1, Tier 2, or Tier 3 crates.

Key components:
- **`ByteArena<'a>` & `ByteCursor<'a>`**: Fallible bump allocation over caller-supplied `&'a mut [u8]` slices with runtime pointer-alignment rounding up to 16 bytes, zero-initialization, `ArenaCheckpoint` rewind, and scoped `reborrow` frames that restore the cursor automatically on drop.
- **Heterogeneous Safe Views**: Carves aligned `&mut [T]` slices for `usize`, `u64`, `f32`, `f64`, fixed arrays `[T; N]`, and `num_complex::Complex<T>` via `bytemuck::Pod + bytemuck::Zeroable`, alongside `FlagSlice<'a>` (`0`/`1` byte flags without `&mut [bool]` UB) and packed `BitSlice<'a>` (`u64` bitsets).
- **`SolverWorkspace` Trait & `OwnedByteArena`**: Shared workspace contract implemented by both stack/borrowed `ByteArena<'a>` and `alloc`-gated `u128`-aligned `OwnedByteArena`.
- **`ArenaLayout` & `ArenaRegion<T>`**: Deterministic single-buffer region planning allowing persistent multi-array workspaces to replace dozens of separate `Vec` fields with one preallocated buffer.

## 2. Developer Instructions

Verify strict `#![no_std]` compilation on host and WebAssembly alongside the full feature test suite:

```bash
cargo check -p math_memory --no-default-features
cargo check -p math_memory --target wasm32-unknown-unknown --no-default-features
cargo test -p math_memory --all-features
```

### Usage Example

```rust
use math_memory::{ByteArena, SolverWorkspace};

let mut raw = [0_u8; 256];
let mut arena = ByteArena::new(&mut raw);
{
    let mut frame = arena.reborrow();
    let mut cur = frame.cursor();
    let pivots = cur.alloc_slice::<usize>(4).unwrap();
    let values = cur.alloc_slice::<f64>(8).unwrap();
    let mut flags = cur.alloc_flags(4).unwrap();
    pivots[0] = 2;
    values[0] = 3.5;
    flags.set(1, true);
    assert!(flags.get(1));
}
assert_eq!(arena.used_bytes(), 0);
```
