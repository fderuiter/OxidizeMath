# Continuum PDE & SPDE Engine Specification (`spec:continuum_pde_spde_v1`)

- **Spec ID**: `spec:continuum_pde_spde_v1`
- **Traceability Module Key**: `engine_pde`

## Governing Equations

Semi-implicit spectral and finite-difference SPDE steppers integrate second- and fourth-order continuum field equations on 1D/2D/3D grids:

$$
\partial_t u = \mathcal{L} u + \mathcal{N}(u) + \sigma \xi(\mathbf{x}, t), \qquad \widehat{u}_{\mathbf{k}}^{n+1} = \frac{\widehat{u}_{\mathbf{k}}^n + \Delta t\,\widehat{\mathcal{N}(u)}_{\mathbf{k}} + \sqrt{\Delta t}\,\widehat{\xi}_{\mathbf{k}}}{1 - \Delta t\,\lambda(\mathbf{k})}
$$

## Numerical Stability & Error Bounds

- Semi-implicit Fourier stepping stabilizes stiff linear symbols $\lambda(\mathbf{k}) = -D|\mathbf{k}|^2 - \kappa|\mathbf{k}|^4 \le 0$ unconditionally in the linear part while Orszag $2/3$-rule spectral truncation suppresses quadratic/cubic aliasing errors.
- Staggered-face conserved-flux stochastic noise $\nabla \cdot \boldsymbol{\xi}$ preserves discrete spatial integral $\sum_{\mathbf{i}} u_{\mathbf{i}} \Delta V$ up to IEEE-754 rounding error $\mathcal{O}(N \varepsilon_{\text{mach}})$.

## Domain of Validity

- Power-of-two grid dimensions $N_d = 2^m \ge 4$ for radix-2 Cooley-Tukey FFT kernels, strictly positive grid spacings $\Delta x_d > 0$, non-negative damping coefficients $D, \kappa \ge 0$, and explicit reaction CFL step $\Delta t > 0$.

## References

- `@cahn1958free` — Cahn & Hilliard (1958), *Free Energy of a Nonuniform System. I. Interfacial Free Energy*, `DOI:10.1063/1.1744102`.
- `@kardar1986dynamic` — Kardar, Parisi & Zhang (1986), *Dynamic Scaling of Growing Interfaces*, `DOI:10.1103/PhysRevLett.56.889`.
- `@kloeden1992numerical` — Kloeden & Platen (1992), *Numerical Solution of Stochastic Differential Equations*, `DOI:10.1007/978-3-662-12616-5`.
