# Canonical Mathematical Specification: `domain_applied`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/domain_applied`](../../crates/03_domains/domain_applied/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and workspace manuscripts.

## 1. Kalman Filtering, Markowitz Portfolio QP, Battery Aging, Mean-Field Games, GRPO & LoRAHub (`algorithms`, `portfolio`, `battery_degradation`, `game_theory`, `grpo`, `lorahub`)

### Governing Equations

Discrete-time `KalmanFilter` (`KalmanModel`) propagates state mean and Joseph-stabilized covariance:

$$
\hat{x}_{k|k-1} = F \hat{x}_{k-1|k-1} + B u_k, \qquad K_k = P_{k|k-1} H^T (H P_{k|k-1} H^T + R)^{-1}, \qquad \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1})
$$

Electrochemical SEI/plating degradation (`PowerLawModel`, `ElectrochemicalAgingModel`, `DegradationModel`) tracks Arrhenius capacity fade $Q_{\text{loss}}(N) = a N^b$, `PortfolioSpec` solves analytical Merton two-fund Lagrangian and primal-dual active-set box-simplex Markowitz quadratic programs $\min_{0 \le w_i \le u_i} \frac{1}{2} w^T \Sigma w$ subject to $\mathbf{1}^T w = 1$ (`long_only_tangency`, `constrained_frontier_in_place`, DOI: `10.2307/2975974`, DOI: `10.2307/2329621`), `FixedPointSolver` (`MFGSolver`) and CSR-backed `GraphReplicatorSystem` couple backward Hamilton-Jacobi-Bellman / spatial replicator imitation with forward density transport, and LoRAHub/GRPO normalize group advantages and convex adapter mixtures.

### Citations

- BibTeX keys: [`huang2023lorahub`](../../papers/references.bib#huang2023lorahub) (`arXiv:2307.13269`), [`R-Zero-paper`](../../papers/references.bib#R-Zero-paper), [`Rubner2000TheRetrieval`](../../papers/references.bib#Rubner2000TheRetrieval) (DOI: `10.1023/A:1026543900054`), [`boisvert1997`](../../papers/references.bib#boisvert1997).
- Manuscripts: [`papers/grpo.tex`](../../papers/grpo.tex), [`papers/lorahub.tex`](../../papers/lorahub.tex), [`papers/dijkstra.tex`](../../papers/dijkstra.tex), [`papers/self_calibration_paper.tex`](../../papers/self_calibration_paper.tex).

### Stability

Kalman innovation inversion checks matrix invertibility (`KalmanError::SingularMatrix`) before updating covariance; mean-field fixed-point damping $\alpha \in (0, 1]$ prevents policy-density oscillation; electrochemical aging clamps remaining capacity to non-negative physical bounds.

### Domain of Validity

Requires matching state/observation matrix dimensions, symmetric positive semi-definite process covariance $Q \succeq 0$, symmetric positive definite measurement noise $R \succ 0$, positive absolute temperature $T > 0\,\text{K}$ for Arrhenius kinetics, and finite cycle counts $N \ge 0$.

```rust
use math_explorer::applied::battery_degradation::{
    Cycles, DegradationModel, DepthOfDischarge, PowerLawModel,
};

let model = PowerLawModel::standard();
let soh = f64::from(model.capacity(
    Cycles::new_clamped(100.0),
    DepthOfDischarge::new_clamped(80.0),
));
assert!(soh > 0.7 && soh < 1.0);
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
