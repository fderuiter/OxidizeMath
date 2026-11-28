# Canonical Mathematical Specification: `domain_applied`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/domain_applied`](../../crates/03_domains/domain_applied/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and workspace manuscripts.

## 1. Markowitz Portfolio QP, Mean-Field & Spatial Games, FreeSurfer MRI/GLM, GRPO & LoRAHub (`algorithms`, `portfolio`, `freesurfer`, `game_theory`, `grpo`, `lorahub`)

### Governing Equations

`PortfolioSpec` solves analytical Merton two-fund Lagrangian and primal-dual active-set box-simplex Markowitz quadratic programs (`long_only_tangency`, `constrained_frontier_in_place`):

$$
\min_{\mathbf{0} \le \mathbf{w} \le \mathbf{u}} \frac{1}{2}\mathbf{w}^T \Sigma \mathbf{w} - \lambda \boldsymbol{\mu}^T \mathbf{w} \quad \text{subject to} \quad \mathbf{1}^T \mathbf{w} = 1
$$

`FixedPointSolver` (`MFGSolver`) and CSR-backed `GraphReplicatorSystem` / `SpatialReplicatorGrid` couple backward Hamilton-Jacobi-Bellman or Fermi pairwise imitation ($P_{i \to j} = [1 + \exp((f_i - f_j)/\kappa)]^{-1}$) with forward density transport (building on `FitnessStrategy`, `MatrixPayoff`, and `ReplicatorDynamics` promoted to [`pure_math`](pure_math.md), with compatibility re-exports in `applied::game_theory::evolutionary` and `applied::algorithms::kalman`). FreeSurfer deformable cortical reconstruction (`evolve_surface_deformable`), Potts Markov Random Field segmentation, and `CorticalGlm` evaluate surface-based neuroimaging statistics (re-exporting triangulated `Surface` and `cortical_thickness_batch` from [`pure_math`](pure_math.md)), while LoRAHub and GRPO normalize group advantages and convex adapter mixtures. Rechargeable battery aging and CNC/UART calculators reside in [`domain_engineering`](domain_engineering.md).

### Citations

- BibTeX keys: [`huang2023lorahub`](../../papers/references.bib#huang2023lorahub) (`arXiv:2307.13269`), [`R-Zero-paper`](../../papers/references.bib#R-Zero-paper), [`Rubner2000TheRetrieval`](../../papers/references.bib#Rubner2000TheRetrieval) (DOI: `10.1023/A:1026543900054`), [`boisvert1997`](../../papers/references.bib#boisvert1997).
- Literature: Markowitz (1952), DOI: `10.2307/2975974`; Merton (1972), DOI: `10.2307/2329621`; Fischl & Dale (2000), `freesurfer.tex`.
- Manuscripts: [`papers/grpo.tex`](../../papers/grpo.tex), [`papers/lorahub.tex`](../../papers/lorahub.tex), [`papers/dijkstra.tex`](../../papers/dijkstra.tex), [`papers/freesurfer.tex`](../../papers/freesurfer.tex), [`papers/self_calibration_paper.tex`](../../papers/self_calibration_paper.tex).

### Stability

Active-set box-simplex projection enforces exact simplex normalization ($\sum_i w_i = 1$) with Cholesky/SVD conditioning checks; mean-field fixed-point damping $\alpha \in (0, 1]$ prevents policy-density oscillation; `CorticalGlm` uses pivoted Householder QR (`engine_linalg`) for rank-safe contrast estimation.

### Domain of Validity

Requires matching return/covariance dimensions with symmetric positive definite $\Sigma \succ 0$ and feasible upper bounds $\sum_i u_i \ge 1$, positive Fermi noise temperature $\kappa > 0$, non-empty 3D MRI voxel volumes, and full-column-rank GLM design matrices.

```rust
use math_explorer::applied::pharmacokinetics::{
    BatemanModel, PKParameters, PharmacokineticModel,
};

let params = PKParameters::new(1.0, 100.0, 1.2, 0.3, 10.0).unwrap();
let model = BatemanModel::new(params);
let conc = model.concentration(1.0);
assert!(conc > 0.0 && conc < 10.0);
```

## 2. Pharmacokinetics, Clinical Win-Ratio, Cannibalism PDEs, Favoritism & Isosurfaces (`pharmacokinetics`, `clinical_trials`, `win_ratio`, `cannibalism`, `favoritism`, `isosurface`)

### Governing Equations

One-compartment oral absorption (`BatemanModel`, `EnantiomerModel`, `TwoPulseModel`, `SuperpositionModel`, `PharmacokineticModel`, `RegimenModel`) and age-structured McKendrick-von Foerster cannibalism (`McKendrickSolver`, `JuvenileAdultSolver`, `CannibalismModel`) evolve concentration and age-density profiles:

$$
C(t) = \frac{F D k_a}{V_d (k_a - k_e)}\bigl(e^{-k_e t} - e^{-k_a t}\bigr), \qquad \frac{\partial n}{\partial t} + \frac{\partial n}{\partial a} = -\bigl(\mu_0 + c\,C(t)\bigr) n(a, t)
$$

Hierarchical clinical pair comparisons (`BivariateWeibullModel`) estimate the Pocock win ratio $W_R = N_{\text{win}} / N_{\text{loss}}$, `UnifiedFavoritismModel` and `FavoritismInputs` evaluate quadrature-normalized asymmetric resource allocation, and Marching Cubes extracts watertight level sets $\{x \in \Omega : \phi(x) = \phi_0\}$.

### Citations

- BibTeX keys: [`tillaboev2025`](../../papers/references.bib#tillaboev2025), [`chen2019med3d`](../../papers/references.bib#chen2019med3d), [`pushpa2022eisenstein`](../../papers/references.bib#pushpa2022eisenstein) (DOI: `10.1007/s40065-022-00360-6`).
- Manuscripts: [`papers/clinical_trials_statistics.tex`](../../papers/clinical_trials_statistics.tex), [`papers/win_ratio.tex`](../../papers/win_ratio.tex), [`papers/cannibalism.tex`](../../papers/cannibalism.tex), [`papers/favorite_child.tex`](../../papers/favorite_child.tex), [`papers/isosurface_extraction.tex`](../../papers/isosurface_extraction.tex), [`papers/freesurfer.tex`](../../papers/freesurfer.tex).

### Stability

When $|k_a - k_e| \to 0$, Bateman evaluation switches smoothly to the secular limit $\frac{F D k_e t}{V_d} e^{-k_e t}$ to avoid catastrophic cancellation, while upwind characteristics $\Delta t \le \Delta a$ preserve positivity and $L^1$ stability in McKendrick age transport.

### Domain of Validity

Requires strictly positive pharmacokinetic clearance/absorption rates $k_a, k_e, V_d > 0$, bioavailability $F \in (0, 1]$, non-negative time $t \ge 0$, Weibull shape/scale parameters $> 0$ with Gumbel-Hougaard dependence $\theta \ge 1$, and non-empty 3D voxel grids ($N_x, N_y, N_z \ge 2$).
