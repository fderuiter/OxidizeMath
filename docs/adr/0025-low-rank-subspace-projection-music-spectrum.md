# ADR 0025: Low-Rank Orthonormal Subspace Projection in MUSIC Spectrum

## Status
Accepted

## Context
In `crates/domain_physics/src/physics/medical/radar_gating/super_resolution.rs`, `MusicEstimator::compute_spectrum` previously materialized a dense $N \times N$ noise projection matrix $P_{\text{noise}} = E_n E_n^H$ from $N - P$ outer products (`col * col.adjoint()`) and evaluated the pseudospectrum denominator $d(R) = a(R)^H P_{\text{noise}} a(R)$ across $K = \text{steps} + 1$ range bins via `p_noise.mul_to(&a_vec, &mut tmp)`.

For $N$ samples per chirp, $P$ signal targets, and $K$ range bins:
1. Constructing $P_{\text{noise}}$ allocated $N - P$ temporary $N \times N$ matrices on the heap.
2. Sweeping $K$ range bins executed $O(K \cdot N^2)$ complex multiply-adds, dominating over 97% of `bench_music` execution time ($N = 64, P = 2, K = 10{,}001$).

Because the Hermitian covariance eigendecomposition yields an orthonormal basis $[E_s, E_n] \in \mathbb{C}^{N \times N}$ satisfying $E_s E_s^H + E_n E_n^H = I_N$, the quadratic form admits two algebraically equivalent low-rank evaluations:
- Direct noise projection: $a^H E_n E_n^H a = \|E_n^H a\|_2^2 = \sum_{j=1}^{N-P} |e_{n,j}^H a|^2$.
- Signal-complement projection: $a^H E_n E_n^H a = \|a\|_2^2 - \|E_s^H a\|_2^2 = \|a\|_2^2 - \sum_{j=1}^{P} |e_{s,j}^H a|^2$.

## Decision
1. **Compact Column-Major Orthonormal Basis**: Replaced dense $P_{\text{noise}}$ construction with a single column-major `DMatrix<Complex<f64>>` of shape $N \times \min(P, N - P)$, packing the $P$ signal eigenvectors when $P < N - P$ and the $N - P$ noise eigenvectors otherwise.
2. **Zero-Allocation Low-Rank Range Sweep**: Removed the temporary `tmp` vector and `p_noise.mul_to` matrix-vector multiply. Each range step accumulates `proj_norm_sq` via `col.dotc(&a_vec).norm_sqr()` across basis columns and evaluates `(a_vec.norm_squared() - proj_norm_sq).max(0.0)` when $P < N - P$ (accounting for $O(N^2 \varepsilon)$ multiplicative phasor recurrence norm drift from [ADR 0007](0007-complex-phasor-recurrence-steering-vector.md) and clamping rounding residuals at exact spectral peaks) or `proj_norm_sq.max(0.0)` otherwise before adding $10^{-12}$.
3. **Verified Modular Helpers**: Decomposed `MusicEstimator::compute_spectrum` into `#[verified_engine::verified]` private helpers (`estimate_covariance`, `extract_compact_subspace`, and `CompactSubspace::evaluate_range_sweep`) and removed `#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]`, keeping every function within the 60-line Clippy threshold and 60-statement `#[verified]` limit.

## Consequences
- **Performance**: Reduces range-sweep complexity from $O(K \cdot N^2)$ to $O(K \cdot N \cdot \min(P, N - P))$ and eliminates $N - P$ dense $N \times N$ outer-product allocations per spectrum call.
- **Numerical Fidelity**: Preserves peak locations, finite positive spectral powers near exact noiseless targets, and tight numerical parity with dense noise-projector evaluation across both $P < N - P$ and $P \ge N - P$ regimes.
- **API & Safety Compliance**: Maintains identical public signatures on `MusicEstimator`, preserves all `RadarError` guards and `NaN` checks, and satisfies NASA Power of 10 function-length and statement-count constraints without lint suppressions.
