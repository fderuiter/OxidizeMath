# ADR 0015: Bluestein Fast Convolution Algorithm for Chirp-Z Transform

## Status

Accepted

## Context
In `crates/domain_physics/src/physics/medical/radar_gating/czt.rs`, `chirp_z_transform` previously evaluated the zoom-FFT discrete spectrum using direct summation over time-domain samples and output frequency bins. For an input signal of length $N$ and output spectrum of $K$ bins, this required $O(N \cdot K)$ complex operations. When computing high-resolution zoom spectra for FMCW radar distance gating ($N \ge 128, K \ge 128$), the quadratic computational scaling created unnecessary latency.

## Decision
We modernized `chirp_z_transform` to implement Bluestein's Chirp-Z algorithm via fast FFT convolution:
1. **Chirp Modulation Sequences**: Formulated the pre-chirp sequence $y_n = x_n e^{-j (n \omega_0 + \frac{\Delta\omega}{2} n^2)}$, chirp filter kernel $h_m = e^{j \frac{\Delta\omega}{2} m^2}$ with circular convolution embedding $h_{-m} = h_m$ stored at $L - m$, and post-chirp modulation $e^{-j \frac{\Delta\omega}{2} k^2}$ with scale factor $1/L$.
2. **Fast Convolution via FFT**: Evaluated the linear convolution of lengths $N$ and $K$ using circular FFT convolution of length $L = (N + K - 1).\text{next\_power\_of\_two()}$ via `rustfft::FftPlanner`, reducing time complexity to $O((N + K) \log(N + K))$.
3. **Zero-Overhead Direct Recurrence Fallback**: Retained direct summation via `direct_chirp_z_transform` for tiny bin counts or tiny signals ($K \le 4$ or $N \le 4$) to avoid FFT planning and buffer allocation overhead.
4. **Contract and Precision Verification**: Preserved `#[verified_engine::verified]` and theory embeddings, guaranteeing numerical equivalence within floating-point tolerance against direct summation across diverse problem sizes and spectral configurations.

## Consequences
- **Performance**: Asymptotic complexity reduced from $O(N \cdot K)$ to $O((N + K) \log(N + K))$, accelerating large-scale radar gating and high-resolution zoom-FFT transforms.
- **Compatibility**: Preserved identical API signatures and contracts for `chirp_z_transform` and `SpatialCztConfig::process`.
- **Verification**: Verified numerical equivalence against direct recurrence for all tested signal lengths, validated peak distance detection in radar tests, and maintained all architectural integrity invariants.
