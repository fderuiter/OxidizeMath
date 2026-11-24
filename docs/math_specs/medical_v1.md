# Medical Physics, Dosimetry & Radar Gating Specification (`spec:medical_v1`)

- **Spec ID**: `spec:medical_v1`
- **Traceability Module Key**: `medical`

## Governing Equations

High-resolution respiratory motion spectral zoom over band $[f_0, f_0 + B]$ uses Bluestein's Chirp Z-Transform (CZT) on $N$ complex samples $x_n$:

$$
X_k = \sum_{n=0}^{N-1} x_n \, e^{-i 2\pi \bigl(f_0 + \frac{k}{K} B\bigr) n / f_s}, \qquad k = 0, \dots, K - 1
$$

Radiobiological cell survival under fractionated dose $D = n d$ follows the linear-quadratic law $S(D) = \exp(-\alpha D - \beta d D)$.

## Numerical Stability & Error Bounds

- Bluestein FFT convolution evaluates unit-modulus chirp factors $|W^{n^2/2}| = 1$ in `f64`, achieving $\mathcal{O}((N + K)\log(N + K))$ complexity and $\mathcal{O}(\varepsilon_{\mathrm{mach}} \log(N + K))$ relative spectral error without geometric spiral overflow.

## Domain of Validity

- Sampling rate $f_s > 0$, non-empty finite signal $N \ge 1$, zoom bin count $K \ge 1$, Nyquist-bounded band $0 \le f_0 \le f_0 + B \le f_s$, and non-negative radiation dose $D \ge 0$.

## References

- `@bluestein1970linear` — Bluestein (1970), *A linear filtering approach to the computation of discrete Fourier transform*, `DOI:10.1109/TAU.1970.1162132`.
