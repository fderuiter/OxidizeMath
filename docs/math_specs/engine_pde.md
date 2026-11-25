# Canonical Mathematical Specification: `engine_pde`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/engine_pde`](../../crates/engine_pde/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and workspace continuum PDE/SPDE manuscripts.

## 1. Finite-Difference & Operator-Split SPDE Stencil Steppers (`fused_stepper`, `spde_stencil`)

### Governing Equations

`FusedStencilStepper`, `ExplicitSpdeStencilStepper`, and Strang `SplitStepSpdeStencilStepper` evolve 1D/2D/3D reaction-diffusion, Allen-Cahn, and fourth-order conserved Cahn-Hilliard fields with additive white noise or staggered-face conserved-flux noise ($\nabla_h \cdot \vec{\eta}$):

$$
\partial_t u = D \nabla_h^2 u + R(u, \nabla_h u) + \xi, \qquad \partial_t \phi = M\bigl[\nabla_h^2(\phi^3 - \phi) - \kappa \nabla_h^4 \phi\bigr] + \nabla_h \cdot \vec{\eta}
$$

### Citations

- BibTeX keys: [`orszag1972numerical`](../../papers/references.bib#orszag1972numerical) (DOI: `10.1103/PhysRevLett.28.76`), [`karniadakis1993nodes`](../../papers/references.bib#karniadakis1993nodes).
- Literature: LeVeque (2007), *Finite Difference Methods for Ordinary and Partial Differential Equations*, DOI: `10.1137/1.9780898717839`; Strang (1968), *SIAM J. Numer. Anal.* 5(3), DOI: `10.1137/0705041`.
- Manuscript: [`papers/generative_turbulence.tex`](../../papers/generative_turbulence.tex).

### Stability

Explicit $d$-dimensional second-order diffusion requires $D\,\Delta t \sum_{k=1}^d \Delta x_k^{-2} \le \frac{1}{2}$ (partitioned across `diffusion_substeps` in `SplitStepSpdeStencilStepper`), while explicit fourth-order Biharmonic Cahn-Hilliard stepping requires $M \kappa\,\Delta t (\Delta x^{-2} + \Delta y^{-2})^2 \le \frac{1}{8}$.

### Domain of Validity

Requires strictly positive finite grid spacings $\Delta x, \Delta y, \Delta z > 0$, non-negative step $\Delta t \ge 0$, non-negative diffusivity/mobility/interfacial stiffness ($D, M, \kappa \ge 0$), and preallocated `SolverWorkspace` capacity matching `scratch_len_1d` / `scratch_len_2d`.

## 2. Radix-2 Orszag 2/3-Dealiased Semi-Implicit Fourier SPDE Stepper (`fourier_stepper`)

### Governing Equations

`SemiImplicitFourierStepper` treats stiff linear surface-tension/biharmonic dispersion implicitly in Fourier space while evaluating nonlinear Cahn-Hilliard ($\phi^3 - \phi$) and Kardar-Parisi-Zhang ($\frac{\lambda}{2}|\nabla h|^2$) terms pseudospectrally with Orszag $2/3$-rule spectral truncation:

$$
\hat{\phi}_{\mathbf{k}}^{n+1} = \frac{\hat{\phi}_{\mathbf{k}}^n - \Delta t\,M |\mathbf{k}|^2 \widehat{(\phi^3 - \phi)}_{\mathbf{k}}^{(2/3)} + \widehat{(\nabla \cdot \vec{\eta})}_{\mathbf{k}}}{1 + \Delta t\,M \kappa |\mathbf{k}|^4}, \qquad \hat{h}_{\mathbf{k}}^{n+1} = \frac{\hat{h}_{\mathbf{k}}^n + \frac{\lambda \Delta t}{2}\widehat{|\nabla h|^2}_{\mathbf{k}}^{(2/3)} + \hat{\eta}_{\mathbf{k}}}{1 + \Delta t\,\nu |\mathbf{k}|^2}
$$

### Citations

- BibTeX keys: [`orszag1972numerical`](../../papers/references.bib#orszag1972numerical) (DOI: `10.1103/PhysRevLett.28.76`), [`orszag1979spectral`](../../papers/references.bib#orszag1979spectral), [`li2020fourier`](../../papers/references.bib#li2020fourier).
- Literature: Kardar, Parisi, & Zhang (1986), *Phys. Rev. Lett.* 56(9), DOI: `10.1103/PhysRevLett.56.889`; Orszag (1971), *J. Atmos. Sci.* 28(6), DOI: `10.1175/1520-0469(1971)028<1074:OTEOAI>2.0.CO;2`.
- Manuscript: [`papers/generative_turbulence.tex`](../../papers/generative_turbulence.tex).

### Stability

Implicit treatment of $+M\kappa |\mathbf{k}|^4$ and $+\nu |\mathbf{k}|^2$ removes the severe $O(\Delta x^4)$ and $O(\Delta x^2)$ linear parabolic stiffness barriers; Orszag $2/3$-rule dealiasing zeroes modes $|k_i| \ge \frac{2}{3} k_{\text{Nyq}}$ to prevent quadratic/cubic aliasing blow-up, and $\mathbf{k} = \mathbf{0}$ flux projection preserves spatial mean mass to machine precision.

### Domain of Validity

Valid on periodic domains with power-of-two grid dimensions $N_x, N_y \ge 4$ (`is_power_of_two()`), positive spacings $\Delta x, \Delta y > 0$, non-negative parameters $M, \kappa, \nu, \Delta t \ge 0$, and `SolverWorkspace` capacity `scratch_len_1d(N)` / `scratch_len_2d(W, H)`.

```rust
use math_explorer::math_memory::ByteArena;
use math_explorer::pure_math::analysis::pde::fused_stepper::SemiImplicitFourierStepper;
use math_explorer::rng::OxidizeRng;

let stepper = SemiImplicitFourierStepper::new_1d(0.5);
let mut phi = [0.1_f64, -0.1, 0.05, -0.05, 0.08, -0.08, 0.02, -0.02];
let mut scratch = vec![0.0_f64; SemiImplicitFourierStepper::scratch_len_1d(phi.len())];
let mut ws = ByteArena::from_slice(&mut scratch);
let mut rng = OxidizeRng::new(11);
stepper.step_cahn_hilliard_1d(&mut phi, 1.0, 0.5, 0.0, 0.05, &mut ws, &mut rng);
assert!(phi.iter().sum::<f64>().abs() < 1e-12);
```
