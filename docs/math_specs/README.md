# Mathematical Domain Specifications (`docs/math_specs/`)

This directory holds verified Markdown mathematical specifications bound via `spec:<id>` entries in [`traceability.toml`](../../traceability.toml) and audited by `TraceabilityEngine` and `unified_verification`.

Every specification file MUST define:

1. **Governing Equations**: Explicit LaTeX mathematical formulations (`$$ ... $$` or `$ ... $`).
2. **Numerical Stability & Error Bounds**: Discretization, conditioning, floating-point rounding, and step-size stability guarantees.
3. **Domain of Validity**: Physical regime limits, parameter bounds, and mathematical preconditions.
4. **References**: Verified BibTeX keys (`@<key>`) resolving to entries in [`papers/references.bib`](../../papers/references.bib) that carry a persistent `doi` or `eprint` identifier.

## Specification Index

- [`transformer_v1.md`](transformer_v1.md) — Transformer self-attention and two-layer universal approximation (`ai`)
- [`wave_optics_caustics_v1.md`](wave_optics_caustics_v1.md) — Scalar diffraction, eikonal ray tracing, and optical caustics (`optics`)
- [`standard_model_v1.md`](standard_model_v1.md) — Electroweak symmetry breaking, CKM mixing, and relativistic kinematics (`standard_model`)
- [`stat_mech_v1.md`](stat_mech_v1.md) — Canonical ensembles, Ising spin lattices, and partition functions (`stat_mech`)
- [`fluid_dynamics_v1.md`](fluid_dynamics_v1.md) — Incompressible Navier-Stokes and Lattice Boltzmann BGK hydrodynamics (`fluid_dynamics`)
- [`chaos_v1.md`](chaos_v1.md) — Strange attractors, Lyapunov exponents, and nonlinear dynamical maps (`chaos`)
- [`nuclear_v1.md`](nuclear_v1.md) — Bateman decay chains, semi-empirical mass formula, and cross sections (`nuclear`)
- [`medical_v1.md`](medical_v1.md) — Dosimetry, Chirp Z-Transform respiratory radar gating, and medical physics (`medical`)
- [`mri_v1.md`](mri_v1.md) — Bloch magnetization dynamics and $k$-space signal reconstruction (`mri`)
- [`high_energy_v1.md`](high_energy_v1.md) — Relativistic two-body scattering and Mandelstam invariants (`high_energy`)
- [`solid_state_v1.md`](solid_state_v1.md) — Tight-binding band structures, Bloch electrons, and phonon dispersion (`solid_state`)
- [`epidemiology_v1.md`](epidemiology_v1.md) — Compartmental SIR/SEIR epidemic transmission dynamics (`epidemiology`)
- [`engineering_v1.md`](engineering_v1.md) — Linear elasticity, structural mechanics, and control systems (`engineering`)
- [`game_theory_v1.md`](game_theory_v1.md) — Strategic normal-form equilibria and replicator dynamics (`game_theory`)
- [`paradox_lab_v1.md`](paradox_lab_v1.md) — Braess network routing, decision-theoretic, and probability paradoxes (`paradox_lab`)
- [`battery_degradation_v1.md`](battery_degradation_v1.md) — Arrhenius SEI growth and power-law Li-ion capacity fade (`battery_degradation`)
- [`pharmacokinetics_v1.md`](pharmacokinetics_v1.md) — Multi-compartment absorption, distribution, and elimination ODEs (`pharmacokinetics`)
- [`climate_v1.md`](climate_v1.md) — Radiative energy balance, carbon-cycle forcing, and CERA rescaling (`climate`)
- [`biology_v1.md`](biology_v1.md) — Hodgkin-Huxley neurodynamics, Turing morphogenesis, and Lenia (`biology`)
- [`domain_physics_v1.md`](domain_physics_v1.md) — Classical Hamiltonian/Lagrangian mechanics and physical field solvers (`domain_physics`)
- [`math_commons_v1.md`](math_commons_v1.md) — IEEE-754 numerical primitives, series views, and least-squares kernels (`math_commons`)
- [`pure_math_v1.md`](pure_math_v1.md) — Initial-value ODE integrators, discrete-time Hidden Markov Models, and algebra (`pure_math`)
