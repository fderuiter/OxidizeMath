# Statistical Mechanics Specification (`spec:stat_mech_v1`)

- **Spec ID**: `spec:stat_mech_v1`
- **Traceability Module Key**: `stat_mech`

## Governing Equations

The canonical partition function $Z(\beta)$ and Ising lattice Hamiltonian for spins $\sigma_i \in \{-1, +1\}$ at inverse temperature $\beta = 1/(k_B T)$ satisfy:

$$
\mathcal{H}(\boldsymbol{\sigma}) = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i, \qquad Z(\beta) = \sum_{\boldsymbol{\sigma}} e^{-\beta \mathcal{H}(\boldsymbol{\sigma})}
$$

## Numerical Stability & Error Bounds

- Log-partition functions $\ln Z(\beta)$ use log-sum-exp shifting $\ln \sum_k e^{a_k} = a_{\max} + \ln \sum_k e^{a_k - a_{\max}}$ to prevent exponential overflow/underflow at low temperature ($\beta \gg 1$).
- Metropolis acceptance probabilities $\min(1, e^{-\beta \Delta E})$ satisfy exact detailed balance in `f64` for $\Delta E > 0$.

## Domain of Validity

- Strictly positive absolute temperature $T > 0$ ($\beta \in (0, \infty)$), finite coupling $J \in \mathbb{R}$, and finite external field $h \in \mathbb{R}$ on finite lattices $N \ge 1$.

## References

- `@onsager1944crystal` — Onsager (1944), *Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition*, `DOI:10.1103/PhysRev.65.117`.
