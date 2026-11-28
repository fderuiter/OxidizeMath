# Canonical Mathematical Specification: `engine_diffeq`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/engine_diffeq`](../../crates/02_engines/engine_diffeq/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and workspace stochastic/dynamical manuscripts.

## 1. Multivariate Itô & Stratonovich SDE Solvers (`euler_maruyama`, `milstein`, `sriw1`)

### Governing Equations

`EulerMaruyamaSolver` (strong order $0.5$ Itô / Euler-Heun Stratonovich), `MilsteinSolver` (strong order $1.0$ with optional `p`-mode Karhunen-Loève Lévy area $A_{j_1, j_2}$), and derivative-free Rössler `Sriw1Solver` (strong order $1.5$ with correlated $(\Delta W_j, \Delta Z_j)$) integrate $d$-dimensional stochastic differential equations driven by $m$-dimensional Wiener noise:

$$
dX_t = f(t, X_t)\,dt + G(t, X_t)\,dW_t, \qquad X_{n+1}^{\text{Mil}} = X_n + f_n \Delta t + G_n \Delta W_n + \sum_{j_1, j_2 = 1}^m (D g_{j_2}) g_{j_1} I_{(j_1, j_2)}
$$

### Citations

- BibTeX keys: [`norris1997markov`](../../papers/references.bib#norris1997markov), [`rabiner1989tutorial`](../../papers/references.bib#rabiner1989tutorial) (DOI: `10.1109/5.18626`).
- Literature: Kloeden & Platen (1992), *Numerical Solution of Stochastic Differential Equations*, DOI: `10.1007/978-3-662-12616-5`; Rössler (2010), *SIAM J. Numer. Anal.* 48(3), DOI: `10.1137/09076636X`.
- Manuscript: [`papers/chaos_theory.tex`](../../papers/chaos_theory.tex).

### Stability

Mean-square linear stability on test equation $dX_t = \lambda X_t\,dt + \mu X_t\,dW_t$ ($\Re\lambda < 0$) requires $|1 + \lambda \Delta t|^2 + |\mu|^2 \Delta t < 1$; all workspace buffers are borrowed via zero-allocation `SolverWorkspace` slices.

### Domain of Validity

Requires globally Lipschitz drift $f$ and diffusion $G$ (plus bounded $C^2$/$C^3$ derivatives for Milstein and `SRIW1` strong convergence orders), non-negative step $\Delta t \ge 0$, state length $\ge d$, and `SolverWorkspace` capacity at least `scratch_len(d, m)`.

```rust
use math_explorer::pure_math::statistics::ou_process::{EulerMaruyama, OuParams, TimeStep};
use math_explorer::rng::OxidizeRng;

let params = OuParams::from_values(0.5, 2.0, 0.25).unwrap();
let solver = EulerMaruyama::new(params, TimeStep::new(0.01).unwrap());
let mut rng = OxidizeRng::new(7);
let x1 = solver.step(1.0, &mut rng);
assert!(x1.is_finite());
```

```rust
use engine_diffeq::{EulerMaruyamaSolver, SdeSystem, StochasticCalculus};
use math_commons::math_kernel::types::ScalarWorkspace;
use math_rng::OxidizeRng;

struct LinearOu;
impl SdeSystem<f64> for LinearOu {
    fn state_dim(&self) -> usize { 1 }
    fn wiener_dim(&self) -> usize { 1 }
    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) { out[0] = -2.0 * x[0]; }
    fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) { out[0] = 0.25; }
}

let solver = EulerMaruyamaSolver::new(StochasticCalculus::Ito);
let mut x = [1.0_f64];
let mut buf = [0.0_f64; 8];
let mut ws = ScalarWorkspace::new(&mut buf);
let mut rng = OxidizeRng::new(7);
solver.step(&LinearOu, 0.0, 0.01, &mut x, &mut ws, &mut rng);
assert!(x[0].is_finite());
```

## 2. Geometric Underdamped Langevin Splitting (`baoab`)

### Governing Equations

`BaoabLangevinSolver` integrates underdamped Hamiltonian Langevin dynamics via symmetric Leimkuhler-Matthews `B-A-O-A-B` operator splitting with exact Ornstein-Uhlenbeck momentum thermostatting ($c_1 = e^{-\gamma \Delta t}$):

$$
dq = M^{-1} p\,dt, \quad dp = -\nabla V(q)\,dt - \gamma p\,dt + \sqrt{2\gamma m k_B T}\,dW_t, \quad p' = c_1 p + \sqrt{m k_B T (1 - c_1^2)}\,\xi
$$

### Citations

- BibTeX keys: [`norris1997markov`](../../papers/references.bib#norris1997markov), [`kingma2014adam`](../../papers/references.bib#kingma2014adam) (`arXiv:1412.6980`).
- Literature: Leimkuhler & Matthews (2013), *Appl. Math. Res. Express* 2013(1), DOI: `10.1093/amrx/abs010`.
- Manuscript: [`papers/advanced_linear_algebra.tex`](../../papers/advanced_linear_algebra.tex).

### Stability

Reduces to symplectic Velocity-Verlet when $\gamma = 0$; for harmonic potentials $V(q) = \frac{1}{2} m \omega^2 q^2$, stable for $0 < \omega \Delta t < 2$ and reproduces the exact Gibbs-Boltzmann configurational variance $k_B T / (m \omega^2)$ independent of $\Delta t$.

### Domain of Validity

Requires strictly positive mass $m > 0$, step size $\Delta t > 0$, non-negative friction $\gamma \ge 0$ and thermal energy $k_B T \ge 0$, `q.len() >= dim`, `p.len() >= dim`, and `SolverWorkspace` length $\ge 2\,\text{dim}$.

```rust
use math_explorer::nalgebra::Vector2;
use math_explorer::pure_math::analysis::evolution::EvolutionEngine;
use math_explorer::pure_math::analysis::momentum_dynamics::{LangevinBaoab2D, PhasePoint2D};
use math_explorer::rng::OxidizeRng;

let mut state = PhasePoint2D::new(Vector2::new(1.0, -0.5), Vector2::zeros());
let mut solver = LangevinBaoab2D::new(1.0, 1.0, 0.5);
let mut rng = OxidizeRng::new(42);
let mut grad = |q: Vector2<f64>| q;
solver.step(&mut state, &mut grad, &mut rng, 0.01).unwrap();
assert!(state.is_finite());
```
