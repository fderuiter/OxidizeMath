Zero-allocation 2D conformal symplectic Leapfrog (Störmer-Verlet) momentum integrator evolving
$p_{n+1/2} = e^{-\gamma h/2} p_n - \frac{h}{2}\nabla V(q_n)$, $q_{n+1} = q_n + h\,\partial_p K(p_{n+1/2})$,
$p_{n+1} = e^{-\gamma h/2}\bigl(p_{n+1/2} - \frac{h}{2}\nabla V(q_{n+1})\bigr)$.

Reference: Hairer, Lubich, & Wanner (2006), *Geometric Numerical Integration*, DOI: 10.1007/3-540-30666-8;
[`docs/math_specs/pure_math.md`](../pure_math.md).

# Stability
When `damping == 0.0`, reduces to volume-preserving second-order Velocity-Verlet, conserving shadow Hamiltonians for step sizes $0 < h \omega_{\max} < 2$; positive `damping` applies exact exponential momentum contraction $e^{-\gamma h / 2} \in (0, 1)$.

# Domain of Validity
Requires finite phase-space coordinates `(q, p)`, finite potential gradients $\nabla V(q)$, strictly positive step `dt > 0` (clamped to `[1e-4, 0.1]`), mass clamped to `[0.05, 50.0]`, and damping clamped to `[0.0, 20.0]` (`EvolutionError::InvalidState` otherwise).
