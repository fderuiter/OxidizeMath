# Magnetic Resonance Imaging Specification (`spec:mri_v1`)

- **Spec ID**: `spec:mri_v1`
- **Traceability Module Key**: `mri`

## Governing Equations

Macroscopic nuclear spin magnetization $\mathbf{M}(t) = (M_x, M_y, M_z)$ under field $\mathbf{B}(t)$ and longitudinal/transverse relaxation times $(T_1, T_2)$ satisfies the Bloch equation:

$$
\frac{d\mathbf{M}}{dt} = \gamma \, \mathbf{M} \times \mathbf{B} - \frac{M_x \hat{\mathbf{x}} + M_y \hat{\mathbf{y}}}{T_2} - \frac{M_z - M_0}{T_1}\hat{\mathbf{z}}
$$

## Numerical Stability & Error Bounds

- Operator-split rotation-relaxation propagation applies exact $SO(3)$ Rodrigues rotations followed by exact exponential decay factors $e^{-\Delta t / T_1}, e^{-\Delta t / T_2} \in (0, 1]$, guaranteeing $\|\mathbf{M}(t)\| \le M_0$ unconditionally for any $\Delta t > 0$.

## Domain of Validity

- Relaxation times $0 < T_2 \le 2 T_1 < \infty$, equilibrium magnetization $M_0 \ge 0$, finite magnetic field $\mathbf{B} \in \mathbb{R}^3$, and time step $\Delta t \ge 0$.

## References

- `@bloch1946nuclear` — Bloch (1946), *Nuclear Induction*, `DOI:10.1103/PhysRev.70.460`.
