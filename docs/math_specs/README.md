# Canonical Mathematical Specifications

This directory provides canonical 4-part mathematical contracts (governing equations, peer-reviewed bibliography citations, numerical stability bounds, and domains of validity) for public mathematical models, steppers, and solvers across all 12 mathematical and solver crates in OxidizeMath, linked directly to [`papers/references.bib`](../../papers/references.bib) and the LaTeX manuscripts in `papers/`.

## Crate Specifications

- [`math_commons`](math_commons.md): Polynomial and damped Levenberg-Marquardt least-squares curve fitting, stack-compiled AST expressions, and Wigner $3j$ / Clebsch-Gordan coupling.
- [`engine_linalg`](engine_linalg.md): `#![no_std]` zero-allocation LU, Cholesky, pivoted Householder QR, Golub-Kahan SVD, Tikhonov regularization, and sparse CSR Conjugate Gradient / restarted GMRES solvers.
- [`engine_diffeq`](engine_diffeq.md): `#![no_std]` multivariate Itô/Stratonovich Euler-Maruyama, Milstein, strong-order-$1.5$ `SRIW1` Stochastic Runge-Kutta, and geometric `B-A-O-A-B` Langevin solvers.
- [`engine_pde`](engine_pde.md): `#![no_std]` 1D/2D/3D fused finite-difference stencils, explicit/Strang-split SPDE steppers, and radix-2 Orszag $2/3$-dealiased semi-implicit Fourier SPDE steppers.
- [`pure_math`](pure_math.md): Arbitrary-precision modular arithmetic, dense linear systems, ODE/PDE solvers, discrete Kalman state estimation, evolutionary replicator dynamics, triangulated cortical surface meshes, symplectic/Langevin momentum dynamics, stochastic/Markov dynamics, optimization (`SGD`, `Adam`, $L_1$ least squares), and partition/quasiperfect number theory.
- [`domain_ai`](domain_ai.md): Multi-layer perceptrons, Transformer multi-head attention, 3D Gaussian Splatting, Score Distillation Sampling, self-calibration, and tabular reinforcement learning.
- [`domain_applied`](domain_applied.md): Markowitz portfolio optimization, age-structured cannibalism PDEs, clinical trials and win-ratio U-statistics, spatial/mean-field games, FreeSurfer cortical MRI/GLM, LoRAHub, Marching Cubes isosurfaces, and pharmacokinetics.
- [`domain_biology`](domain_biology.md): Hodgkin-Huxley membrane excitability, Kuramoto phase synchronization, Turing reaction-diffusion morphogenesis, 2D finite-difference diffusion, FFT Lenia, and enzyme kinetics.
- [`domain_climate`](domain_climate.md): Climate-Invariant Encoding and Representation Alignment (`Cera`, `AutoencoderModel`, `PredictorModel`) with Earth Mover's Distance alignment.
- [`domain_engineering`](domain_engineering.md): Millimeter-wave FMCW radar signal processing, Bluestein Chirp Z-Transform, MIMO/MUSIC estimation, Kalman motion tracking, coupled electrochemical battery degradation, Taylor CNC tool life, and UART frame throughput.
- [`domain_epidemiology`](domain_epidemiology.md): Deterministic compartmental `SIRModel` / `SEIRModel` systems and heterogeneous `NetworkEpidemicModel` contagion dynamics.
- [`domain_physics`](domain_physics.md): 1D Crank-Nicolson quantum Cayley propagation, non-equilibrium continuum SPDEs (Allen-Cahn, Cahn-Hilliard, Edwards-Wilkinson, KPZ), BCS superconductivity, D2Q9 Lattice Boltzmann, galactic potentials, nuclear mass models, clinical radiotherapy beam gating, and general relativity geodesics.

## 4-Part Docstring Contract

Every public mathematical solver, stepper, or model across the 12 mathematical and solver crates satisfies four verifiable docstring sections (inline or via `#[doc = include_str!("...")]`):

1. **Governing Equation**: Display (`$$...$$`) or inline (`$...$`) LaTeX equation, or compile-time `#[embed_theory("papers/...", label = "...")]` extraction.
2. **Citation**: Peer-reviewed `DOI` / `arXiv` identifier or relative link to the canonical `docs/math_specs/<crate>.md` specification anchored in [`papers/references.bib`](../../papers/references.bib).
3. **`# Stability`**: Algorithmic conditioning, step-size CFL or Lipschitz bounds, damping safeguards, and floating-point precision guarantees.
4. **`# Domain of Validity`**: Admissible parameter ranges, dimensionality invariants, positivity/finiteness preconditions, and typed error boundaries.
