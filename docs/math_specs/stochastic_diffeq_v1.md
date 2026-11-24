# Stochastic Differential Equation Engine Specification (`spec:stochastic_diffeq_v1`)

- **Spec ID**: `spec:stochastic_diffeq_v1`
- **Traceability Module Key**: `engine_diffeq`

## Governing Equations

Multivariate Itô/Stratonovich stochastic differential equations and underdamped Langevin Hamiltonian systems evolve according to:

$$
\mathrm{d}\mathbf{X}_t = \mathbf{a}(t, \mathbf{X}_t)\,\mathrm{d}t + \mathbf{B}(t, \mathbf{X}_t)\,\mathrm{d}\mathbf{W}_t, \qquad \mathrm{d}\mathbf{q} = \mathbf{M}^{-1}\mathbf{p}\,\mathrm{d}t, \quad \mathrm{d}\mathbf{p} = -\nabla V(\mathbf{q})\,\mathrm{d}t - \gamma \mathbf{p}\,\mathrm{d}t + \sqrt{2\gamma k_B T \mathbf{M}}\,\mathrm{d}\mathbf{W}_t
$$

## Numerical Stability & Error Bounds

- Euler-Maruyama achieves strong convergence order $\mathcal{O}(\Delta t^{1/2})$ and weak order $\mathcal{O}(\Delta t)$; Milstein achieves strong order $\mathcal{O}(\Delta t^{1.0})$; derivative-free `SRIW1` achieves strong order $\mathcal{O}(\Delta t^{1.5})$ for scalar/commutative noise.
- Symmetric `B-A-O-A-B` Langevin splitting preserves exact Ornstein-Uhlenbeck velocity variance and achieves $\mathcal{O}(\Delta t^2)$ configurational Boltzmann invariant-measure accuracy in `#![no_std]` preallocated workspaces.

## Domain of Validity

- Globally Lipschitz drift $\mathbf{a}$ and diffusion $\mathbf{B}$ with linear growth bounds, finite state vectors, strictly positive mass $M_{ii} > 0$, friction $\gamma \ge 0$, temperature $T \ge 0$, and step size $0 < \Delta t < 2/\lambda_{\max}$.

## References

- `@kloeden1992numerical` — Kloeden & Platen (1992), *Numerical Solution of Stochastic Differential Equations*, `DOI:10.1007/978-3-662-12616-5`.
- `@leimkuhler2013rational` — Leimkuhler & Matthews (2013), *Rational Construction of Stochastic Numerical Methods for Molecular Sampling*, `DOI:10.1093/amrx/abs010`.
