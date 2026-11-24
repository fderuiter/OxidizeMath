Explicit 2D damped wave equation solver for $\partial_{tt} u = c^2(\partial_{xx} u + \partial_{yy} u) - \gamma \partial_t u$.

Reference: LeVeque (2007), DOI: 10.1137/1.9780898717839;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
Three-level second-order leapfrog stepping is conditionally stable under the 2D Courant-Friedrichs-Lewy condition $c \Delta t \sqrt{\Delta x^{-2} + \Delta y^{-2}} \le 1$ with non-negative damping $\gamma \ge 0$.

# Domain of Validity
Valid on rectangular uniform grids ($N_x, N_y \ge 3$) with wave speed $c \ge 0$, damping $\gamma \ge 0$, and positive spacings $\Delta x, \Delta y > 0$.
