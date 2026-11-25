# Math Scalars

![Maturity: Certified](https://img.shields.io/badge/maturity-Certified-2ea44f.svg)

Zero-cost `#![no_std]` type-level SI dimensional quantities, unit-system frames, affine/absolute temperatures, and deterministic floating-point [`Scalar`] abstractions.

> **Maturity Tier:** `Certified` — `#[repr(transparent)]` type-level SI quantities, non-vacuous `theory_verification!` contracts, `VerifiedAllocator` zero-heap-allocation proofs, and SemVer signature locks.

## 1. Overview & Architecture

`math_scalars` forms OxidizeMath's Tier 0 physical dimensional substrate under `crates/00_foundation/math_scalars`. Every [`Quantity<T, D, F>`] is `#[repr(transparent)]` over its underlying scalar `T` (`f64` or `f32`), encoding all seven SI base exponents $(L, M, T, I, \Theta, N, J)$ in [`Dim<L, M, T, I, Theta, N, J>`] and a unit-system frame tag `F` ([`SiFrame`], [`NuclearMeVFmFrame`], [`GalacticKpcFrame`], [`ViewportFrame`], [`MillimeterFrame`]) at compile time with zero heap allocations and zero runtime memory overhead (`8` bytes for `f64`, `4` bytes for `f32`).

### Module Structure

- **`scalar`**: Deterministic `#![no_std]` [`Scalar`] trait implemented for `f64` and `f32` via `libm`.
- **`int`**: Open canonical Peano-backed signed type-level integers ([`Z0`], [`PInt`], [`NInt`], aliases [`N8`]..=[`P8`]) implementing [`AddExp`], [`NegExp`], [`SubExp`], and even-only [`HalfExp`] on stable Rust without nightly `generic_const_exprs`.
- **`dim`**: Seven-exponent SI dimension specification [`Dim<L, M, T, I, Theta, N, J>`] with open [`AddDim`], [`SubDim`], [`NegDim`], and [`HalfDim`] traits plus standard physical dimension aliases ([`Dimensionless`], [`DimLength`], [`DimVelocity`], [`DimEnergy`], [`DimAction`], [`DimMagneticFluxDensity`], etc.).
- **`frame`**: Disjoint unit-system tags ([`SiFrame`], [`NuclearMeVFmFrame`], [`GalacticKpcFrame`], [`ViewportFrame`], [`MillimeterFrame`]) preventing cross-scale arithmetic at compile time.
- **`quantity`**: `#[repr(transparent)]` [`Quantity<T, D, F>`] with dimension-preserving linear operations, exponent-deriving `*`, `/`, `.recip()`, even-exponent `.sqrt()`, and [`Dimensionless`]-restricted `.ln()`, `.exp()`, `.sin()`, `.cos()`, and `.powf()`.
- **`temperature`**: Distinct affine [`Celsius`] and absolute SI [`Kelvin`] representations requiring explicit `+ 273.15` offset conversion via [`Celsius::to_kelvin`].
- **`aliases`**: Coherent SI, nuclear `MeV`-`fm`, kiloparsec, viewport, and millimeter type aliases and explicit scale converters.

---

## 2. Developer Instructions

### Automatic SI Exponent Derivation and Dimensionless Transcendentals

```rust
use math_scalars::{
    Celsius, JoulesPerKelvin, Kelvin, Meters, Seconds,
};

let distance = Meters::new(120.0);
let elapsed = Seconds::new(8.0);
let speed = distance / elapsed;
assert_eq!(speed.get(), 15.0);

let area = distance * distance;
let recovered_length: Meters = area.sqrt();
assert_eq!(recovered_length.get(), 120.0);

let room_c = Celsius::new(25.0);
let room_k: Kelvin = room_c.to_kelvin();
assert!((room_k.get() - 298.15).abs() < 1e-12);

let kb = JoulesPerKelvin::new(1.380_649e-23);
let thermal_energy = kb * room_k;
let ratio = thermal_energy / thermal_energy;
assert!((ratio.ln().get()).abs() < 1e-12);
```
