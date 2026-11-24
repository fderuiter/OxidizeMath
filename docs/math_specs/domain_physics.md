# Canonical Mathematical Specification: `domain_physics`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/domain_physics`](../../crates/domain_physics/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and workspace physics manuscripts.

## 1. BCS Superconductivity, Lattice Boltzmann & Galactic/Nuclear Models (`solid_state`, `fluid_dynamics`, `astrophysics`, `nuclear`)

### Governing Equations

Self-consistent zero-temperature s-wave BCS pairing (`BcsGapSolver`, `IsotropicBCSModel`) iterates the damped gap operator over single-particle states $|\xi_k| \le \hbar\omega_D$:

$$
\Delta_{n+1} = (1 - \alpha)\Delta_n + \alpha \sum_{|\xi_k| \le \hbar\omega_D} \frac{V\,\Delta_n}{2\sqrt{\xi_k^2 + \Delta_n^2}}
$$

Discrete-velocity BGK Lattice Boltzmann (`D2Q9Model`, `CollisionModel`) relaxes populations $f_i^{\text{out}} = f_i - \frac{1}{\tau}(f_i - f_i^{\text{eq}}(\rho, \mathbf{u}))$, `GalaxyModel` evaluates axisymmetric/spherical circular velocity profiles $v_c^2(R) = R\,\partial_R \Phi(R, 0)$, and Bethe-Weizsaecker `LiquidDropModel` (`BindingEnergyModel`) computes nuclear binding energy $B(A, Z)$.

### Citations

- BibTeX keys: [`misner1973gravitation`](../../papers/references.bib#misner1973gravitation), [`schwarzschild1916`](../../papers/references.bib#schwarzschild1916), [`orszag1972numerical`](../../papers/references.bib#orszag1972numerical) (DOI: `10.1103/PhysRevLett.28.76`), [`Rosner`](../../papers/references.bib#Rosner).
- Manuscripts: [`papers/quantum_mechanics.tex`](../../papers/quantum_mechanics.tex), [`papers/dwarf_galaxy_empirical_dependencies.tex`](../../papers/dwarf_galaxy_empirical_dependencies.tex), [`papers/general_relativity_geodesics.tex`](../../papers/general_relativity_geodesics.tex), [`papers/chaos_theory.tex`](../../papers/chaos_theory.tex).

### Stability

Linear mixing $\alpha \in (0, 1]$ damps fixed-point oscillations in `BcsGapSolver`; D2Q9 BGK relaxation requires $\tau > 0.5$ and low Mach number $|\mathbf{u}| \ll c_s = 1/\sqrt{3}$ for positivity and $L^2$ stability.

### Domain of Validity

Requires attractive pairing potential $V > 0$, Debye cutoff $\hbar\omega_D > 0$, tolerance $> 0$, relaxation time $\tau > 0.5$, positive galactic mass/scale radii ($R > 0$), and nucleon counts $A \ge 1$, $0 \le Z \le A$ (`NuclearError`).

```rust
use math_explorer::physics::nuclear::models::{BindingEnergyModel, LiquidDropModel};
use math_explorer::physics::nuclear::types::{AtomicNumber, MassNumber};

let model = LiquidDropModel::new();
let fe56 = model
    .binding_energy_per_nucleon(AtomicNumber::new(26), MassNumber::new(56).unwrap())
    .unwrap();
assert!(fe56 > 8.0 && fe56 < 9.5);
```

## 2. FMCW Radar Gating, Bluestein Chirp Z-Transform & Dose Kernels (`medical`)

### Governing Equations

Millimeter-wave radiotherapy gating (`FmcwConfig::range_from_beat_frequency`, `chirp_z_transform`, `ConstantVelocityModel`) relates beat frequency $\hat{f}_{\text{FFT}}$ to target range $R$ and evaluates narrowband zoom spectra via Bluestein convolution:

$$
R = \frac{c\,T_c\,\hat{f}_{\text{FFT}}}{2 B}, \qquad X_k = \sum_{n=0}^{N-1} x_n \exp\!\left(-j 2\pi \frac{f_0 + k\,\Delta f}{f_s} n\right)
$$

### Citations

- BibTeX keys: [`kammler2007first`](../../papers/references.bib#kammler2007first), [`parseval1806memoire`](../../papers/references.bib#parseval1806memoire), [`pushpa2022eisenstein`](../../papers/references.bib#pushpa2022eisenstein) (DOI: `10.1007/s40065-022-00360-6`).
- Manuscript: [`papers/mmwave_radiotherapy_setup.tex`](../../papers/mmwave_radiotherapy_setup.tex).

### Stability

Power-of-two zero-padded FFT convolution (`conv_len >= N + K - 1`) eliminates circular aliasing and executes in $O((N + K)\log(N + K))$ operations with bounded floating-point phase drift, falling back to exact Horner/phasor summation for $N \le 4$ or $K \le 4$.

### Domain of Validity

Requires sweep bandwidth $B > 0$, chirp duration $T_c > 0$, sampling rate $f_s > 0$ satisfying Nyquist bounds ($|f_0| + |B| \le f_s / 2$ for unaliased real targets), and finite time step $\Delta t > 0$ for `ConstantVelocityModel`.

## 3. Unitary 1D Quantum Cayley Propagation & Continuum Statistical Mechanics SPDEs (`quantum`, `stat_mech`)

### Governing Equations

1D Schrödinger propagation (`CrankNicolson1D`, `CrankNicolsonPropagator1D`) advances wavefunctions via the $O(N)$ tridiagonal Cayley transform, while non-equilibrium statistical mechanics solvers evolve Model A (`AllenCahn2D`), conserved Model B (`CahnHilliard2D`), linear `EdwardsWilkinson1D`/`EdwardsWilkinson2D`, and nonlinear `KpzModel1D`/`KpzModel2D` fields:

$$
\left(I + \frac{i\Delta t}{2\hbar}\hat{H}\right)\psi^{n+1} = \left(I - \frac{i\Delta t}{2\hbar}\hat{H}\right)\psi^n, \qquad \partial_t h = \nu \nabla^2 h + \frac{\lambda}{2}|\nabla h|^2 + \eta, \qquad \partial_t \phi = M \nabla^2(\phi^3 - \phi - \kappa \nabla^2 \phi) + \nabla \cdot \vec{\eta}
$$

### Citations

- BibTeX keys: [`orszag1972numerical`](../../papers/references.bib#orszag1972numerical) (DOI: `10.1103/PhysRevLett.28.76`), [`orszag1979spectral`](../../papers/references.bib#orszag1979spectral), [`Rosner`](../../papers/references.bib#Rosner).
- Literature: Crank & Nicolson (1947), DOI: `10.1017/S0305004100023197`; Hohenberg & Halperin (1977), DOI: `10.1103/RevModPhys.49.435`; Kardar, Parisi, & Zhang (1986), DOI: `10.1103/PhysRevLett.56.889`.
- Manuscripts: [`papers/quantum_mechanics.tex`](../../papers/quantum_mechanics.tex), [`papers/generative_turbulence.tex`](../../papers/generative_turbulence.tex).

### Stability

Cayley propagation is unconditionally unitary ($\|\psi^{n+1}\|_2 = \|\psi^n\|_2$) for real symmetric $\hat{H}$ and $\Delta t > 0$; radix-2 `step_fourier_dealiased` treats $+\nu|\mathbf{k}|^2$ and $+M\kappa|\mathbf{k}|^4$ implicitly with $2/3$-rule dealiasing, while explicit stencil stepping obeys parabolic/biharmonic CFL bounds.

### Domain of Validity

Requires finite potentials $V(x_i)$ and strictly positive `(dx, dt, mass, h_bar)` for `CrankNicolson1D`; `AllenCahn2D` requires `width, height >= 3`, $M > 0$, $\kappa \ge 0$; `CahnHilliard2D`, `EdwardsWilkinson*`, and `KpzModel*` require grid sizes $\ge 4$ (power-of-two with `BoundaryCondition::Periodic` for Fourier stepping), positive spacings, and $M, \kappa, \nu > 0$.

