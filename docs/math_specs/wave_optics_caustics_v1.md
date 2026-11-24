# Wave Optics & Caustics Specification (`spec:wave_optics_caustics_v1`)

- **Spec ID**: `spec:wave_optics_caustics_v1`
- **Traceability Module Key**: `optics`

## Governing Equations

High-frequency geometric wavefronts obey the eikonal equation in inhomogeneous refractive index $n(\mathbf{r})$:

$$
|\nabla S(\mathbf{r})|^2 = n(\mathbf{r})^2, \qquad \frac{d}{ds}\!\left(n(\mathbf{r})\frac{d\mathbf{r}}{ds}\right) = \nabla n(\mathbf{r})
$$

Near caustics where ray Jacobians vanish, scalar wave intensity is governed by the Helmholtz / Fresnel diffraction integral:

$$
\nabla^2 U(\mathbf{r}) + k_0^2 n(\mathbf{r})^2 U(\mathbf{r}) = 0
$$

## Numerical Stability & Error Bounds

- Symplectic / Runge-Kutta ray integration preserves optical Hamiltonian invariants up to $\mathcal{O}(\Delta s^4)$ local truncation error per step $\Delta s$.
- Paraxial and Fresnel grid evaluations enforce Nyquist phase sampling $\Delta x \le \lambda / (2\,\mathrm{NA})$ to prevent spatial aliasing.

## Domain of Validity

- Positive refractive index $n(\mathbf{r}) \ge 1.0$, positive vacuum wavelength $\lambda > 0$, and non-degenerate ray step $\Delta s > 0$.
- Geometric ray intensity diverges on caustic envelopes $\det(\partial \mathbf{r}/\partial \boldsymbol{\alpha}) = 0$, where diffraction wave optics must be used.

## References

- `@born1999principles` — Born & Wolf (1999), *Principles of Optics* (7th ed.), `DOI:10.1017/CBO9781139644181`.
