# `math_rng`

Zero-allocation Tier 0 `#![no_std]` foundation crate providing deterministic ChaCha8 pseudo-random streams, generic `f32`/`f64` scalar samplers, and borrowed-slice 64-dimensional Quasi-Monte Carlo (QMC) views over compact static 53-bit tables.

## 1. Overview & Architecture

`math_rng` sits at Tier 0 (`crates/00_foundation/math_rng`) with zero internal OxidizeMath runtime dependencies and zero OS-entropy dependencies (`getrandom` is never linked). Every random stream and low-discrepancy point sequence is 100% deterministic across native (`x86_64`, `aarch64`) and WebAssembly (`wasm32-unknown-unknown`) targets.

Key components:
- **`OxidizeRng` (`rng`)**: Deterministic `ChaCha8Rng` wrapper implementing `RngCore`, `CryptoRng`, and `SeedableRng`, with orthogonal ChaCha8 stream splitting (`with_stream`, `set_stream`, `split_stream`) and 128-bit word-position seeking (`jump_words`, `set_word_pos`, `word_pos`).
- **Generic Scalar & Index Samplers (`scalar`)**: `RngScalar` trait for `f32` and `f64` supporting half-open `[0, 1)`, strictly open `(0, 1)`, symmetric `(-1, 1)`, Box-Muller normal draws, Cranley-Patterson shift generation, and architecture-invariant unbiased bounded index draws (`sample_bounded_u64`, `sample_bounded_usize`).
- **Static-Table 64D QMC Views (`qmc`)**:
  - `SobolView`: 64D 53-bit Joe-Kuo direction tables with $O(d)$ Gray-code XOR stepping, optional 53-bit nested hash Owen scrambling, and Cranley-Patterson torus shifts.
  - `HaltonView`: 64D prime-radix sampling combining static Faure permutations (`FIRST_64_PRIMES`, $p_1 = 2 \dots p_{64} = 311$) with stateless hash Owen digit scrambling to eliminate high-dimensional collinearity.
  - `NiederreiterView`: 64D 53-bit base-2 `(t, s)`-sequence backed by irreducible `GF(2)` generator polynomials, $O(d)$ Gray-code stepping, and 53-bit nested hash Owen scrambling.
  - Optional owned wrappers (`SobolSequence`, `HaltonSequence`, `NiederreiterSequence`) gated behind the `alloc` feature.

All static QMC read-only tables combined occupy less than `16 KiB` of `.rodata`, and all sampling and QMC stepping routines execute with zero heap allocations under `VerifiedAllocator`.

## 2. Developer Instructions

Verify strict `#![no_std]` compilation on host and WebAssembly alongside the full feature test suite:

```bash
cargo check -p math_rng --no-default-features
cargo check -p math_rng --target wasm32-unknown-unknown --no-default-features
cargo test -p math_rng --all-features
```

### Usage Example

```rust
use math_rng::{BASE_SEED, OxidizeRng, SobolView, sample_open};

// Derive orthogonal ChaCha8 streams without ad-hoc XOR seed masks
let base = OxidizeRng::new(BASE_SEED);
let mut stream_a = base.split_stream(1);
let mut stream_b = base.split_stream(2);
let u_a: f64 = sample_open(&mut stream_a);
let u_b: f64 = sample_open(&mut stream_b);
assert!(u_a > 0.0 && u_a < 1.0 && (u_a - u_b).abs() > 0.0);

// Zero-allocation 64D Sobol sampling over caller-provided stack slices
let mut state = [0_u64; 64];
let mut shifts = [0.0_f64; 64];
let mut point = [0.0_f64; 64];
let mut view = SobolView::new(64, &mut state, &mut shifts)
    .expect("valid 64D slices")
    .with_scramble_seed(Some(42));
view.next_point(&mut point).expect("fills 64D point");
assert!(point.iter().all(|&x| x > 0.0 && x < 1.0));
```
