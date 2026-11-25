# Canonical Mathematical Specification: `domain_climate`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/domain_climate`](../../crates/domain_climate/README.md), linked to [`papers/references.bib`](../../papers/references.bib) (`L23-L58`) and [`papers/cera_framework.tex`](../../papers/cera_framework.tex).

## 1. Climate-Invariant Encoding & Representation Alignment (`Cera`, `AutoencoderModel`, `PredictorModel`)

### Governing Equations

`Cera` maps vertical atmospheric profiles $X \in \mathbb{R}^{(L \cdot C_{\text{in}}) \times B}$ through a 1D convolutional encoder-decoder (`AutoencoderModel`) into latent representations $Z = E_\theta(X)$, aligns a climate-invariant subspace $Z_{\text{align}} = Z_{1:C_{\text{align}}}$ across reference and warmed climate regimes via 1D Earth Mover's Distance (1-Wasserstein distance), and predicts subgrid tendencies $\hat{Y} = P_\psi(Z_{\text{align}})$ (`PredictorModel`):

$$
\mathcal{L}_{\text{CERA}}(\theta, \psi) = \|X - D_\theta(E_\theta(X))\|_F^2 + \lambda_{\text{pred}} \|Y - P_\psi(Z_{\text{align}})\|_F^2 + \lambda_{\text{emd}} W_1\bigl(p_{\text{ref}}(Z_{\text{align}}),\, p_{\text{warm}}(Z_{\text{align}})\bigr)
$$

### Citations

- BibTeX keys: [`Liu2025CERA`](../../papers/references.bib#Liu2025CERA) (`arXiv:2509.00010`), [`Beucler2024Climate-invariantLearning`](../../papers/references.bib#Beucler2024Climate-invariantLearning) (DOI: `10.1126/sciadv.adj7250`), [`Yuval2021UsePrecision`](../../papers/references.bib#Yuval2021UsePrecision) (DOI: `10.1029/2020GL091363`), [`Rubner2000TheRetrieval`](../../papers/references.bib#Rubner2000TheRetrieval) (DOI: `10.1023/A:1026543900054`).
- Manuscript: [`papers/cera_framework.tex`](../../papers/cera_framework.tex).
- Architecture guide: [`docs/architecture/cera_framework.md`](../architecture/cera_framework.md).

### Stability

Quantile-sorted 1D Wasserstein loss computes exact $O(B \log B)$ subgradients without adversarial discriminator mode collapse, while ReLU hidden activations and batch-normalized MSE scaling bound forward/backward activations under `f32` precision (`Yuval2021UsePrecision`).

### Domain of Validity

Requires positive channel/level counts ($C_{\text{in}}, C_{\text{latent}}, C_{\text{align}}, L, d_{\text{out}} \ge 1$), subspace inclusion $C_{\text{align}} \le C_{\text{latent}}$ (`ClimateError::ChannelAlignmentExceeded`), input matrix row count $L \cdot C_{\text{in}}$, and finite learning rate $\eta > 0$ with non-negative regularization weights $\lambda_{\text{pred}}, \lambda_{\text{emd}} \ge 0$.

```rust
use math_explorer::climate::cera::Cera;
use math_explorer::climate::config::CeraConfig;
use math_explorer::nalgebra::DMatrix;

let cfg = CeraConfig {
    learning_rate: 1e-3,
    lambda_pred: 0.5,
    lambda_emd: 0.1,
    epochs: 1,
    batch_size: 2,
    in_channels: 2,
    latent_channels: 4,
    aligned_channels: 2,
    num_levels: 4,
    output_size: 2,
};
let cera = Cera::new(cfg).unwrap();
let input = DMatrix::from_element(8, 2, 0.25_f32);
assert_eq!(cera.predict(&input).shape(), (2, 2));
```
