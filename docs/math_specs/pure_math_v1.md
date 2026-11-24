# Pure Mathematics: ODEs, Markov Processes & Algebra Specification (`spec:pure_math_v1`)

- **Spec ID**: `spec:pure_math_v1`
- **Traceability Module Key**: `pure_math`

## Governing Equations

1. **Initial-value ordinary differential equations** on $\mathbf{y}(t) \in \mathbb{R}^d$:

$$
\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}), \qquad \mathbf{y}(t_0) = \mathbf{y}_0
$$

2. **Discrete-time Hidden Markov Models** $\lambda = (\boldsymbol{\pi}, A, B)$ over $N$ hidden states and $M$ observation symbols:

$$
\alpha_t(j) = \left(\sum_{i=1}^N \alpha_{t-1}(i) A_{ij}\right) B_{j, o_t}, \qquad P(\mathbf{O} \mid \lambda) = \sum_{i=1}^N \alpha_T(i)
$$

## Numerical Stability & Error Bounds

- Embedded Dormand-Prince RK45 / fixed-step RK4 solvers achieve $\mathcal{O}(h^5)$ / $\mathcal{O}(h^4)$ global truncation accuracy under Lipschitz continuity $\|\mathbf{f}(t, \mathbf{u}) - \mathbf{f}(t, \mathbf{v})\| \le L \|\mathbf{u} - \mathbf{v}\|$ when $h L$ lies inside the explicit Runge-Kutta stability region.
- Forward-backward and log-Viterbi recurrences scale or accumulate in log-probability space to avoid $\mathcal{O}(c^T)$ floating-point underflow on long observation sequences $T \gg 1$.

## Domain of Validity

- Locally Lipschitz vector field $\mathbf{f}$, positive step size $h > 0$, row-stochastic matrices $\sum_j A_{ij} = 1$, $\sum_k B_{ik} = 1$, $\sum_i \pi_i = 1$ with non-negative probabilities, and $N, M \ge 1$.

## References

- `@dormand1980family` — Dormand & Prince (1980), *A family of embedded Runge-Kutta formulae*, `DOI:10.1016/0771-050X(80)90013-3`.
- `@rabiner1989tutorial` — Rabiner (1989), *A tutorial on hidden Markov models and selected applications in speech recognition*, `DOI:10.1109/5.18626`.
- `@norris1997markov` — Norris (1997), *Markov Chains*, `DOI:10.1017/CBO9780511810633`.
