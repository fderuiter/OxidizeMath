# Canonical Mathematical Specifications

This directory provides canonical 4-part mathematical contracts (governing equations, peer-reviewed bibliography citations, numerical stability bounds, and domains of validity) for public mathematical models and solvers across the 8 mathematical crates in OxidizeMath, linked directly to [`papers/references.bib`](../../papers/references.bib) and the LaTeX manuscripts in `papers/`.

## Crate Specifications

- [`math_commons`](math_commons.md): Polynomial and damped Levenberg-Marquardt least-squares curve fitting, stack-compiled AST expressions, and Wigner $3j$ / Clebsch-Gordan coupling.
- [`pure_math`](pure_math.md): Arbitrary-precision modular arithmetic, dense linear systems, ODE/PDE solvers, stochastic/Markov dynamics, optimization (`SGD`, `Adam`, $L_1$ least squares), and partition/quasiperfect number theory.
- [`domain_ai`](domain_ai.md): Multi-layer perceptrons, Transformer multi-head attention, 3D Gaussian Splatting, Score Distillation Sampling, self-calibration, and tabular reinforcement learning.
- [`domain_applied`](domain_applied.md): Kalman state estimation, battery electrochemical aging, age-structured cannibalism PDEs, clinical trials and win-ratio U-statistics, mean-field games, LoRAHub, Marching Cubes isosurfaces, and pharmacokinetics.
- [`domain_biology`](domain_biology.md): Hodgkin-Huxley membrane excitability, Kuramoto phase synchronization, Turing reaction-diffusion morphogenesis, 2D finite-difference diffusion, FFT Lenia, and enzyme kinetics.
- [`domain_climate`](domain_climate.md): Climate-Invariant Encoding and Representation Alignment (`Cera`, `AutoencoderModel`, `PredictorModel`) with Earth Mover's Distance alignment.
- [`domain_epidemiology`](domain_epidemiology.md): Deterministic compartmental `SIRModel` / `SEIRModel` systems and heterogeneous `NetworkEpidemicModel` contagion dynamics.
- [`domain_physics`](domain_physics.md): BCS superconductivity gap iteration, D2Q9 Lattice Boltzmann collision kernels, galactic gravitational potentials, nuclear semi-empirical mass models, FMCW radar gating / Bluestein CZT, and general relativity geodesics.

## 4-Part Docstring Contract

Every public mathematical solver or model across the 8 mathematical crates satisfies four verifiable docstring sections (inline or via `#[doc = include_str!("...")]`):

1. **Governing Equation**: Display (`$$...$$`) or inline (`$...$`) LaTeX equation, or compile-time `#[embed_theory("papers/...", label = "...")]` extraction.
2. **Citation**: Peer-reviewed `DOI` / `arXiv` identifier or relative link to the canonical `docs/math_specs/<crate>.md` specification anchored in [`papers/references.bib`](../../papers/references.bib).
3. **`# Stability`**: Algorithmic conditioning, step-size CFL or Lipschitz bounds, damping safeguards, and floating-point precision guarantees.
4. **`# Domain of Validity`**: Admissible parameter ranges, dimensionality invariants, positivity/finiteness preconditions, and typed error boundaries.
