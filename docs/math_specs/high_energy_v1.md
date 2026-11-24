# High-Energy Particle Kinematics Specification (`spec:high_energy_v1`)

- **Spec ID**: `spec:high_energy_v1`
- **Traceability Module Key**: `high_energy`

## Governing Equations

Relativistic $2 \to 2$ particle scattering with four-momenta $p_1 + p_2 = p_3 + p_4$ in Minkowski metric $\eta = \mathrm{diag}(1, -1, -1, -1)$ is parameterized by Lorentz-invariant Mandelstam variables:

$$
s = (p_1 + p_2)^2, \qquad t = (p_1 - p_3)^2, \qquad u = (p_1 - p_4)^2, \qquad s + t + u = \sum_{a=1}^4 m_a^2
$$

## Numerical Stability & Error Bounds

- On-shell energy $E = \sqrt{\|\mathbf{p}\|^2 + m^2}$ uses `hypot`-equivalent scaling to avoid overflow at ultra-relativistic momenta $\|\mathbf{p}\| \gg m$, while Lorentz boost factors $\gamma = (1 - \beta^2)^{-1/2}$ factor $(1 - \beta)(1 + \beta)$ near $\beta \to 1^-$.

## Domain of Validity

- Sub-luminal boost speed $\|\boldsymbol{\beta}\| \in [0, 1)$, non-negative rest masses $m_a \ge 0$, and center-of-mass threshold $\sqrt{s} \ge \max(m_1 + m_2, m_3 + m_4)$.

## References

- `@weinberg1967model` — Weinberg (1967), *A Model of Leptons*, `DOI:10.1103/PhysRevLett.19.1264`.
