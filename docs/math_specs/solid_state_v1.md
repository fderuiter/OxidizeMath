# Solid-State & Condensed Matter Specification (`spec:solid_state_v1`)

- **Spec ID**: `spec:solid_state_v1`
- **Traceability Module Key**: `solid_state`

## Governing Equations

Bloch electrons on a periodic Bravais lattice with nearest-neighbor hopping $t$ and lattice constant $a$ obey tight-binding dispersion and Fermi-Dirac occupancy:

$$
\varepsilon(\mathbf{k}) = \varepsilon_0 - 2t \sum_{\alpha=1}^d \cos(k_\alpha a), \qquad f(\varepsilon; \mu, T) = \frac{1}{\exp\!\bigl((\varepsilon - \mu)/(k_B T)\bigr) + 1}
$$

## Numerical Stability & Error Bounds

- Fermi-Dirac occupancy evaluates $1 / (1 + e^x)$ for $x \le 0$ and $e^{-x} / (1 + e^{-x})$ for $x > 0$ where $x = (\varepsilon - \mu)/(k_B T)$, eliminating `exp` overflow as $T \to 0^+$ with full IEEE-754 `f64` accuracy.

## Domain of Validity

- Positive lattice constant $a > 0$, positive temperature $T > 0$, wavevector $\mathbf{k}$ in the first Brillouin zone $[-\pi/a, \pi/a]^d$, and finite hopping $t \in \mathbb{R}$.

## References

- `@ashcroft1976solid` — Ashcroft & Mermin (1977), *Solid State Physics*, `DOI:10.1063/1.3037551`.
