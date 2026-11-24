# Continuum SPDE & Statistical Mechanics Specification (`spec:continuum_spde_stat_mech_v1`)

- **Spec ID**: `spec:continuum_spde_stat_mech_v1`
- **Traceability Module Key**: `stat_mech`

## Governing Equations

Discrete Ising spin systems $\sigma_i \in \{-1, +1\}$ at inverse temperature $\beta = 1/(k_B T)$ and continuum non-equilibrium SPDE fields (Edwards-Wilkinson, Kardar-Parisi-Zhang, Allen-Cahn, and conserved Cahn-Hilliard) satisfy:

$$
\mathcal{H}(\boldsymbol{\sigma}) = -\frac{1}{2} J \sum_{i,j} w_{ij}\sigma_i\sigma_j - h \sum_i \sigma_i, \qquad \partial_t \phi = M \nabla^2\!\left(\phi^3 - \phi - \kappa \nabla^2 \phi\right) + \nabla \cdot \boldsymbol{\xi}(\mathbf{x}, t)
$$

$$
\partial_t h = \nu \nabla^2 h + \frac{\lambda}{2}|\nabla h|^2 + \eta(\mathbf{x}, t), \qquad \langle \eta(\mathbf{x}, t)\eta(\mathbf{x}', t')\rangle = 2D\,\delta(\mathbf{x} - \mathbf{x}')\delta(t - t')
$$

## Numerical Stability & Error Bounds

- Metropolis-Hastings and pairwise Fermi exponents clamp $-\beta \Delta E \in [-500, 0]$ and $(\pi_i - \pi_j)/T \in [-500, 500]$ to prevent IEEE-754 `f64::exp` overflow while preserving detailed balance.
- Conserved Cahn-Hilliard and KPZ Fourier steppers treat stiff biharmonic $-M\kappa k^4$ and diffusive $-\nu k^2$ operators semi-implicitly with Orszag $2/3$-rule dealiasing, conserving spatial mean $\bar{\phi}$ to machine precision ($\mathcal{O}(\varepsilon_{\text{mach}})$).

## Domain of Validity

- Strictly positive absolute temperature $T > 0$, finite coupling weights $w_{ij} \in \mathbb{R}$, positive grid spacings $\Delta x, \Delta y > 0$, positive mobility/viscosity $M, \kappa, \nu > 0$, and positive time step $\Delta t > 0$ within CFL/semi-implicit bounds.

## References

- `@onsager1944crystal` — Onsager (1944), *Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition*, `DOI:10.1103/PhysRev.65.117`.
- `@metropolis1953equation` — Metropolis et al. (1953), *Equation of State Calculations by Fast Computing Machines*, `DOI:10.1063/1.1699114`.
- `@cahn1958free` — Cahn & Hilliard (1958), *Free Energy of a Nonuniform System. I. Interfacial Free Energy*, `DOI:10.1063/1.1744102`.
- `@allen1979microscopic` — Allen & Cahn (1979), *A microscopic theory for antiphase boundary motion and its application to antiphase domain coarsening*, `DOI:10.1016/0001-6160(79)90196-2`.
- `@edwards1982surface` — Edwards & Wilkinson (1982), *The surface statistics of a granular aggregate*, `DOI:10.1098/rspa.1982.0056`.
- `@kardar1986dynamic` — Kardar, Parisi & Zhang (1986), *Dynamic Scaling of Growing Interfaces*, `DOI:10.1103/PhysRevLett.56.889`.
