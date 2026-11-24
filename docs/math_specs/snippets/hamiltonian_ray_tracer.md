Zero-allocation RK4 Hamiltonian ray bundle and caustic envelope tracer integrating canonical optical path flow
$$\frac{d\mathbf{r}}{d\tau} = \mathbf{p}, \qquad \frac{d\mathbf{p}}{d\tau} = \frac{1}{2}\nabla\bigl(n^2(\mathbf{r})\bigr)$$
on pre-allocated [`pure_math::pure_math::analysis::ode::ArrayState<4>`] buffers with exact vector Snell interface refraction and ray-tube Jacobian sign-flip caustic detection.

Reference: Luneburg (1964), *Mathematical Theory of Optics*, [`luneburg1964mathematical`](../../../papers/references.bib#luneburg1964mathematical);
Berry & Upstill (1980), [`berry1980catastrophe`](../../../papers/references.bib#berry1980catastrophe) (DOI: 10.1016/S0079-6638(08)70215-4); [`docs/math_specs/domain_physics.md`](../domain_physics.md).

# Stability
Fourth-order Runge-Kutta achieves $O(\Delta\tau^4)$ canonical optical-momentum invariant drift ($|\mathbf{p}|^2 - n^2(\mathbf{r})$) across smooth graded-index media when $\Delta\tau \ll R / \sup |\nabla(n^2)|$.

# Domain of Validity
Valid in the short-wavelength geometric-optics regime ($\lambda \to 0$) across piecewise-smooth isotropic media with bounded refractive index gradients and non-grazing discrete dielectric interfaces.
