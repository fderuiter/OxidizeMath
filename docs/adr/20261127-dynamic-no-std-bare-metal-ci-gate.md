# ADR 20261127: Dynamic `#![no_std]` Discovery, Bare-Metal `thumbv7em-none-eabihf` CI Gate, and Allocator-Free `OxidizeRng` Serde

## Status

Accepted

## Context

Workspace CI jobs in `.github/workflows/ci.yml` and `apps/xtask/src/gates.rs` previously invoked Cargo doc, doctest, Clippy, and test commands exclusively with `--workspace --all-features`, while `xtask test-features` swept feature flags only on `math_explorer` and `math_explorer_gui`. Because Cargo feature unification enabled `std` and `alloc` across every workspace member during `--all-features` runs, regressions in standalone `#![no_std]` compilation, strict rustdoc intra-doc links, doctests, and feature-graph isolation went undetected:

1. **Ungated `std` integration tests and `Theory` surface**: `engine_diffeq/tests/test_checked_diffeq.rs`, `engine_pde/tests/test_checked_pde.rs`, and `verified_engine/tests/{allocator_trap,macro_integration}.rs` imported `std`-only `verified_engine::{allocator, engine, scientific_metadata}` without `required-features = ["std"]`, and `verified_engine` re-exported `Theory` unconditionally while gating `scientific_metadata` on `std`.
2. **Unguarded `alloc` and `std` rustdoc links and doctests**: Intra-doc links to `alloc`-gated workspace types (`OwnedWorkspace`, `PdeOwnedWorkspace`, `SolverWorkspace`, `NiederreiterSequence`, `SobolSequence`, `intersect_polylines`) broke `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` when `alloc` was disabled, and doctests in `math_commons` and `verified_engine` assumed `std`.
3. **`std` and `alloc` leakage under `--features serde`**: `math_commons` and `engine_opt` listed `"alloc"` and `"serde/alloc"` under `serde`, while `math_rng` delegated `OxidizeRng` serialization to `rand_chacha/serde1`, which unconditionally activates `serde` default features (`serde/std` and `serde/alloc`) and breaks bare-metal targets lacking `std`.

## Decision

1. **Gate `std`-only integration tests and `Theory` macro re-export**:
   - Declare `required-features = ["std"]` on `test_checked_diffeq` (`engine_diffeq`), `test_checked_pde` (`engine_pde`), and `allocator_trap` / `macro_integration` (`verified_engine`).
   - Gate `pub use verified_engine_macros::Theory;` under `#[cfg(feature = "std")]` alongside `scientific_metadata`, and guard `std`/`alloc` intra-doc links and doctests across `verified_engine`, `math_rng`, `engine_linalg`, `math_commons`, `engine_diffeq`, and `engine_pde`.
2. **Decouple `serde` from `alloc` and `std` with wire-compatible `#![no_std]` `OxidizeRng` serialization**:
   - Remove `"alloc"` and `"serde/alloc"` from `serde` in `math_commons/Cargo.toml` and `engine_opt/Cargo.toml`, moving `"serde?/alloc"` under `alloc` so `Vec`-backed types serialize when both `alloc` and `serde` are enabled while `--no-default-features --features serde` remains strictly `core`-only.
   - Remove `"rand_chacha/serde1"` from `math_rng/Cargo.toml` and implement `serde::Serialize` and `serde::Deserialize` directly on `OxidizeRng` over a stack-only `ChaCha8RngWire { seed: [u8; 32], stream: u64, word_pos: u128 }` representation (`#[serde(rename = "ChaCha8Rng")]`), preserving exact JSON map and positional tuple wire compatibility with `rand_chacha`.
3. **Add zero-dependency dynamic `xtask check-no-std` and bare-metal `thumbv7em-none-eabihf` CI enforcement**:
   - Add `thumbv7em-none-eabihf` to `rust-toolchain.toml` and the `pre-flight` job in `.github/workflows/ci.yml`.
   - Implement `apps/xtask/src/no_std.rs` using pure `std` manifest and `src/lib.rs` scanning to automatically discover every `#![no_std]` or `#![cfg_attr(not(feature = "std"), no_std)]` library crate in `[workspace].members`.
   - For each discovered crate, `xtask check-no-std` (also wired into `xtask lint-all`) verifies:
     - Zero `std` or `alloc` activation in `cargo tree -e normal -p <pkg> --no-default-features --features serde`.
     - Bare-metal `cargo check --lib --target thumbv7em-none-eabihf` across `--no-default-features`, `alloc`, `serde`, and `alloc,serde`.
     - Strict `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`, `cargo test --lib --tests`, and `cargo test --doc` across standalone `default` and all non-`std` feature legs.

```bash
cargo run -p xtask -- check-no-std
```

## Consequences

- Contributors can run standalone `cargo test -p <crate>` across all Tier 0 and Tier 2 `#![no_std]` crates without feature-unification workarounds.
- Embedded and bare-metal callers can enable `--no-default-features --features serde` on `math_diagnostics`, `math_scalars`, `math_rng`, `engine_linalg`, `engine_opt`, `math_commons`, `engine_diffeq`, and `engine_pde` without pulling `std` or requiring a global allocator, while existing serialized `OxidizeRng` payloads roundtrip unchanged.
- Any new `#![no_std]` crate added to `[workspace].members` is automatically enrolled in host and `thumbv7em-none-eabihf` CI gates without manual list maintenance.
