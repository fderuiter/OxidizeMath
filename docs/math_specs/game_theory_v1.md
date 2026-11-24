# Game Theory & Strategic Equilibria Specification (`spec:game_theory_v1`)

- **Spec ID**: `spec:game_theory_v1`
- **Traceability Module Key**: `game_theory`

## Governing Equations

Mixed-strategy profiles $\mathbf{x} \in \Delta^{m-1}, \mathbf{y} \in \Delta^{n-1}$ in bimatrix games $(A, B)$ and evolutionary replicator dynamics satisfy:

$$
\mathbf{x}^\top A \mathbf{y} \ge \mathbf{x}'^\top A \mathbf{y} \;\;\forall \mathbf{x}' \in \Delta^{m-1}, \qquad \dot{x}_i = x_i \bigl((A \mathbf{x})_i - \mathbf{x}^\top A \mathbf{x}\bigr)
$$

## Numerical Stability & Error Bounds

- Simplex projection projects drifted iterate vectors back onto $\Delta^{m-1} = \{\mathbf{x} \ge 0 : \sum_i x_i = 1\}$ with $\mathcal{O}(m \log m)$ exact sorting and $\mathcal{O}(m \varepsilon_{\mathrm{mach}})$ sum residual, preventing negative probability drift during replicator integration.

## Domain of Validity

- Finite payoff matrices $A, B \in \mathbb{R}^{m \times n}$ with $m, n \ge 1$, probability vectors $\mathbf{x} \in \Delta^{m-1}$, and positive integration step $\Delta t > 0$.

## References

- `@nash1950equilibrium` — Nash (1950), *Equilibrium points in $n$-person games*, `DOI:10.1073/pnas.36.1.48`.
