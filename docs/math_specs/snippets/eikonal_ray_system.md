Canonical 2D graded-index (GRIN) eikonal ray ODE system
$$\frac{d\mathbf{r}}{d\tau} = \mathbf{p}, \qquad \frac{d\mathbf{p}}{d\tau} = \frac{1}{2}\nabla\bigl(n^2(x, y)\bigr) = n(x, y)\nabla n(x, y)$$
reparameterized by differential optical path $d\tau = ds / n(x, y)$ so square-root denominators vanish identically across Luneburg, Maxwell fisheye, regularized Eaton, and parabolic waveguide media.

Reference: Luneburg (1964), *Mathematical Theory of Optics*, [`luneburg1964mathematical`](../../../papers/references.bib#luneburg1964mathematical);
Born & Wolf (1999), [`born1999principles`](../../../papers/references.bib#born1999principles); [`docs/math_specs/domain_physics.md`](../domain_physics.md).

# Stability
Analytic core regularization ($\epsilon \in [0.03, 0.5]$ for Eaton media and clamped denominators $r_{\max}^2 \ge 10^{-12}$) bounds Jacobian Lipschitz constants so explicit RK4 stepping stays well-conditioned.

# Domain of Validity
Valid for monochromatic paraxial and wide-angle geometric rays in isotropic 2D graded-index profiles with strictly positive radius scales $R > 0$ and refractive indices $n(x, y) > 0$.
