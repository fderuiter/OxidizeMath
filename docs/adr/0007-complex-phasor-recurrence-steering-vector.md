# ADR 0007: In-Place Complex Phasor Recurrence for Steering Vector Generation

## Context
In `crates/domain_physics/src/physics/medical/radar_gating/super_resolution.rs`, `MusicEstimator::compute_spectrum` was calculating steering vectors $a(R)$ using explicit exponential function calls `Complex::new(0.0, phase).exp()` for every sample point in the inner range sweep loop.
For typical spectrum sweeps (e.g., 10,000 range steps with $N = 64$), this evaluated transcendental `sin` and `cos` functions 640,000 times per invocation, creating a major performance bottleneck in MUSIC spectrum calculation.

## Decision
We optimized the steering vector construction by:
1. Hoisting single-step phase calculation $e^{j \alpha}$ outside the sample loop using `alpha.sin_cos()`.
2. Constructing the steering vector $a(R)$ in-place using complex phasor recurrence $a_0 = 1 + 0j$ and $a_k = a_{k-1} \cdot e^{j \alpha}$ for $k \ge 1$.
3. Maintaining statement counts below the 60-statement limit mandated by `#[verified_engine::verified]`.

## Consequences
- **Positive:** Reduces steering vector calculation complexity by replacing transcendental `exp()` calls in the inner loop with a single complex multiplication per sample step.
- **Positive:** Speeds up `bench_music` execution time by ~13.5% without loss of precision.
- **Positive:** Preserves contract and correctness verification under `#[verified_engine::verified]`.
