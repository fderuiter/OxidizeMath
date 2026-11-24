# Fluid Dynamics & Lattice Boltzmann Specification (`spec:fluid_dynamics_v1`)

- **Spec ID**: `spec:fluid_dynamics_v1`
- **Traceability Module Key**: `fluid_dynamics`

## Governing Equations

Discrete D2Q9 Lattice Boltzmann distributions $f_i(\mathbf{x}, t)$ evolve under Bhatnagar-Gross-Krook (BGK) single-relaxation collision and streaming:

$$
f_i(\mathbf{x} + \mathbf{c}_i \Delta t, t + \Delta t) - f_i(\mathbf{x}, t) = -\frac{1}{\tau}\bigl(f_i(\mathbf{x}, t) - f_i^{\mathrm{eq}}(\rho, \mathbf{u})\bigr)
$$

recovering the weakly compressible Navier-Stokes equations with kinematic viscosity $\nu = c_s^2 (\tau - \tfrac{1}{2}) \Delta t$ and $c_s^2 = \tfrac{1}{3}$.

## Numerical Stability & Error Bounds

- Linear LBGK stability requires relaxation time $\tau > 0.5$; over-relaxation instabilities grow rapidly as $\tau \to 0.5^+$.
- Compressibility error scales as $\mathcal{O}(\mathrm{Ma}^2)$ where $\mathrm{Ma} = \|\mathbf{u}\| / c_s$, with second-order spatial/temporal accuracy $\mathcal{O}(\Delta x^2 + \Delta t^2)$ under diffusive scaling.

## Domain of Validity

- Relaxation time $\tau \in (0.5, 5.0]$, low Mach number $\mathrm{Ma} = \|\mathbf{u}\| / c_s < 0.3$, positive macroscopic density $\rho > 0$, and grid dimensions $N_x, N_y \ge 1$.

## References

- `@bhatnagar1954model` — Bhatnagar, Gross & Krook (1954), *A Model for Collision Processes in Gases. I*, `DOI:10.1103/PhysRev.94.511`.
- `@qian1992lattice` — Qian, d'Humières & Lallemand (1992), *Lattice BGK Models for Navier-Stokes Equation*, `DOI:10.1209/0295-5075/17/6/001`.
