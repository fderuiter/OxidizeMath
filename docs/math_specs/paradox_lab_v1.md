# Decision, Routing & Probability Paradoxes Specification (`spec:paradox_lab_v1`)

- **Spec ID**: `spec:paradox_lab_v1`
- **Traceability Module Key**: `paradox_lab`

## Governing Equations

Wardrop user-equilibrium routing on congested networks minimizes Rosenthal's potential $\Phi(\mathbf{f})$ across path flows $\mathbf{f}$ with edge latencies $\ell_e(f_e)$:

$$
\Phi(\mathbf{f}) = \sum_{e \in E} \int_0^{f_e} \ell_e(u)\,du, \qquad \mathrm{PoA} = \frac{C(\mathbf{f}^{\mathrm{UE}})}{C(\mathbf{f}^{\mathrm{SO}})} \ge 1
$$

exhibiting Braess's paradox when adding a zero-cost link increases equilibrium latency $C(\mathbf{f}^{\mathrm{UE}})$.

## Numerical Stability & Error Bounds

- Convex Frank-Wolfe / projected gradient updates on $\Phi(\mathbf{f})$ converge monotonically at $\mathcal{O}(1/k)$ for non-decreasing affine edge latencies, preserving demand conservation $\sum_p f_p = D$ to $\mathcal{O}(\varepsilon_{\mathrm{mach}} D)$.

## Domain of Validity

- Non-negative total demand $D \ge 0$, continuous non-decreasing latency functions $\ell_e(f_e) \ge 0$ on $f_e \ge 0$, and valid probability distributions $p_i \in [0, 1]$ with $\sum_i p_i = 1$.

## References

- `@braess1968paradoxon` — Braess (1968), *Über ein Paradoxon aus der Verkehrsplanung*, `DOI:10.1007/BF01918335`.
- `@nash1950equilibrium` — Nash (1950), *Equilibrium points in $n$-person games*, `DOI:10.1073/pnas.36.1.48`.
