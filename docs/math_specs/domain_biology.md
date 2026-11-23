# Canonical Mathematical Specification: `domain_biology`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/domain_biology`](../../crates/domain_biology/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and workspace manuscripts.

## 1. Reaction-Diffusion Morphogenesis, 2D Finite-Difference Diffusion & FFT Lenia (`morphogenesis`, `reaction_diffusion`, `diffusion`, `lenia`)

### Governing Equations

Coupled two-species activator-inhibitor PDEs (`ReactionDiffusionModel`, `DiffusionModel`, `ReactionModel`, `FusedEulerSolver`, `StandardSolverAdapter`, `TuringSolverStrategy`) evolve concentrations $(u, v)$ under discrete Laplacian stencils:

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u + R_u(u, v), \qquad \frac{\partial v}{\partial t} = D_v \nabla^2 v + R_v(u, v)
$$

For Gray-Scott kinetics, $R_u(u, v) = -u v^2 + F(1 - u)$ and $R_v(u, v) = u v^2 - (F + k)v$.

### Citations

- BibTeX keys: [`raissi2019physics`](../../papers/references.bib#raissi2019physics) (DOI: `10.1016/j.jcp.2018.10.045`), [`orszag1972numerical`](../../papers/references.bib#orszag1972numerical), [`Beucler2024Climate-invariantLearning`](../../papers/references.bib#Beucler2024Climate-invariantLearning) (DOI: `10.1126/sciadv.adj7250`).
- Manuscripts: [`papers/favorite_child.tex`](../../papers/favorite_child.tex), [`papers/chaos_theory.tex`](../../papers/chaos_theory.tex).

### Stability

Explicit forward-Euler 5-point and 7-point Laplacian stencils require the parabolic CFL bound $\Delta t \le \frac{h^2}{2 d \max(D_u, D_v)}$ in $d \in \{2, 3\}$ spatial dimensions to prevent checkerboard high-wavenumber amplification.

### Domain of Validity

Requires non-negative diffusion coefficients $D_u, D_v \ge 0$, strictly positive grid spacing $h > 0$ and time step $\Delta t > 0$, positive grid dimensions, and finite initial concentrations.

```rust
# use domain_biology::__doctest_facade as math_explorer;
use math_explorer::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize};

let mut sys = TuringSystem::new(
    Dimension(16),
    DiffusionCoeff(1.0),
    DiffusionCoeff(40.0),
    StepSize(1.0),
);
sys.step(0.01);
assert_eq!(sys.u().len(), 16);
```

## 2. Conductance Neuroscience, Enzyme Kinetics & Evolutionary Dynamics (`neuroscience`, `kinetics`, `evolution`)

### Governing Equations

`HodgkinHuxleyModel` integrates membrane potential $V$ coupled to potassium ($n$) and sodium ($m, h$) gating probabilities, while `KineticsModel` (Michaelis-Menten / Hill) characterizes enzymatic reaction velocity $v([S])$:

$$
C_m \frac{dV}{dt} = I_{\text{ext}} - \bar{g}_{Na} m^3 h (V - E_{Na}) - \bar{g}_K n^4 (V - E_K) - \bar{g}_L (V - E_L), \qquad v([S]) = \frac{V_{\max} [S]^n}{K_m^n + [S]^n}
$$

### Citations

- BibTeX keys: [`Yuval2021UsePrecision`](../../papers/references.bib#Yuval2021UsePrecision) (DOI: `10.1029/2020GL091363`), [`chen1995universal`](../../papers/references.bib#chen1995universal).
- Manuscripts: [`papers/favorite_child.tex`](../../papers/favorite_child.tex), [`papers/graph_parameters_rust.tex`](../../papers/graph_parameters_rust.tex).

### Stability

Voltage-rate singularities ($\frac{x}{e^x - 1}$ near $x = 0$) use local L'Hopital limits to maintain smooth gating transitions, and gating state variables remain invariant on $[0, 1]^3$ for $\Delta t \lesssim 0.05\,\text{ms}$ under RK4 integration.

### Domain of Validity

Requires membrane capacitance $C_m > 0$, non-negative maximal conductances $\bar{g}_i \ge 0$, substrate concentration $[S] \ge 0$, half-saturation constant $K_m > 0$, and Hill cooperativity $n > 0$.
