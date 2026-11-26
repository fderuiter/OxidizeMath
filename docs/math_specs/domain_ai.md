# Canonical Mathematical Specification: `domain_ai`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/domain_ai`](../../crates/03_domains/domain_ai/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and workspace manuscripts.

## 1. Deep Learning MLP, Transformer Attention & Self-Calibration (`deep_learning_theory`, `transformer`, `self_calibration`)

### Governing Equations

Two-layer perceptions (`TwoLayerMLP`) and scaled dot-product multi-head Transformer architectures (`Transformer`) compute contextual representations:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V, \qquad h(x) = W_2\,\sigma(W_1 x + b_1) + b_2
$$

### Citations

- BibTeX keys: [`vaswani2017attention`](../../papers/references.bib#vaswani2017attention) (`arXiv:1706.03762`), [`wang2022self`](../../papers/references.bib#wang2022self), [`Yuval2021UsePrecision`](../../papers/references.bib#Yuval2021UsePrecision) (DOI: `10.1029/2020GL091363`).
- Manuscripts: [`papers/attention_is_all_you_need_rust.tex`](../../papers/attention_is_all_you_need_rust.tex), [`papers/self_calibration_paper.tex`](../../papers/self_calibration_paper.tex).

### Stability

Row-wise max-shifted $\operatorname{softmax}(z_i - \max_j z_j)$ prevents exponential overflow across arbitrary logit magnitudes, $1/\sqrt{d_k}$ scaling preserves unit variance prior to normalization, and `LayerNorm` guards variance normalization with $\epsilon > 0$.

### Domain of Validity

Requires $d_{\text{model}} > 0$ divisible by `num_heads`, non-empty token vocabularies, finite weight/input matrices with compatible inner dimensions, and positive learning rates $\eta > 0$.

```rust
use math_explorer::ai::deep_learning_theory::model::{Trainable, TwoLayerMLP};
use math_explorer::nalgebra::DVector;

let mlp = TwoLayerMLP::new(2, 4, 1);
let x = DVector::from_vec(vec![0.5, -0.25]);
let out = mlp.forward(&x);
assert_eq!(out.len(), 1);
```

## 2. 3D Gaussian Splatting, Score Distillation & Tabular RL (`gaussian_splatting`, `sds`, `reinforcement_learning`)

### Governing Equations

Anisotropic 3D Gaussians project world covariance $\Sigma = R S S^T R^T$ through affine camera Jacobian $J$ onto screen space $\Sigma' = J W \Sigma W^T J^T$, while Score Distillation Sampling (`DiffusionModel`, `NeRFModel`, `SgdOptimizer`, `AdamOptimizer`) and Bellman Q-learning evolve parameters and action values:

$$
\nabla_\theta \mathcal{L}_{\text{SDS}}(\theta) = \mathbb{E}_{t, \epsilon}\!\left[w(t)\bigl(\hat{\epsilon}_\phi(z_t; y, t) - \epsilon\bigr)\frac{\partial x}{\partial \theta}\right], \qquad Q(s, a) \leftarrow Q(s, a) + \alpha\bigl(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr)
$$

### Citations

- BibTeX keys: [`kerbl20233dgs`](../../papers/references.bib#kerbl20233dgs) (DOI: `10.1145/3592433`), [`ho2020denoising`](../../papers/references.bib#ho2020denoising) (`arXiv:2006.11239`), [`kingma2014adam`](../../papers/references.bib#kingma2014adam) (`arXiv:1412.6980`), [`Rubner2000TheRetrieval`](../../papers/references.bib#Rubner2000TheRetrieval).
- Manuscripts: [`papers/gaussian_splatting.tex`](../../papers/gaussian_splatting.tex), [`papers/generative_turbulence.tex`](../../papers/generative_turbulence.tex).

### Stability

Front-to-back $\alpha$-compositing clamps transmittance $T_i \in [0, 1]$ with early-ray termination, SDS noise schedules bound $\bar{\alpha}_t \in (0, 1)$, and Bellman iterations contract in $\ell_\infty$ norm for discount factor $\gamma \in [0, 1)$.

### Domain of Validity

Requires positive focal lengths and positive camera-space depth $z_{\text{cam}} > 0$, noise schedule variances $\beta_t \in (0, 1)$, learning rates $\alpha \in (0, 1]$, and exploration probabilities $\epsilon \in [0, 1]$.
