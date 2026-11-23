Zero-allocation 1D/2D finite-difference stencil stepper evaluating $\partial_x u \approx \frac{u_{i+1}-u_{i-1}}{2\Delta x}$ and $\partial_{xx} u \approx \frac{u_{i+1}-2u_i+u_{i-1}}{\Delta x^2}$.

Reference: LeVeque (2007), *Finite Difference Methods for Ordinary and Partial Differential Equations*,
DOI: 10.1137/1.9780898717839; [`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Second-order central diffusion requires $\nu \Delta t / \Delta x^2 \le 1/2$ in 1D ($\le 1/4$ in 2D), while first-order upwind advection requires Courant number $|v|\Delta t / \Delta x \le 1$.

# Domain of Validity
Valid on uniform grids with strictly positive spacing $\Delta x > 0$ and at least $3$ grid points per spatial dimension.
