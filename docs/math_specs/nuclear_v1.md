# Nuclear Physics & Decay Kinetics Specification (`spec:nuclear_v1`)

- **Spec ID**: `spec:nuclear_v1`
- **Traceability Module Key**: `nuclear`

## Governing Equations

Serial radioactive decay chains $N_1 \xrightarrow{\lambda_1} N_2 \xrightarrow{\lambda_2} \cdots \xrightarrow{\lambda_{n-1}} N_n$ obey the Bateman linear system and Bethe-Weizsäcker binding energy $B(A, Z)$:

$$
\frac{dN_i}{dt} = \lambda_{i-1} N_{i-1}(t) - \lambda_i N_i(t), \qquad B(A, Z) = a_V A - a_S A^{2/3} - a_C \frac{Z(Z-1)}{A^{1/3}} - a_A \frac{(A - 2Z)^2}{A} + \delta(A, Z)
$$

## Numerical Stability & Error Bounds

- Closed-form Bateman coefficients suffer catastrophic cancellation when $|\lambda_i - \lambda_j| \to 0$; matrix exponential / Padé or implicit backward-Euler stepping guarantees unconditional $L$-stability and non-negative nuclide populations $N_i(t) \ge 0$.

## Domain of Validity

- Mass number $A \ge 1$, proton number $0 \le Z \le A$, non-negative decay constants $\lambda_i \ge 0$, and elapsed time $t \ge 0$.

## References

- `@bateman1910solution` — Bateman (1910), *The solution of a system of differential equations occurring in the theory of radioactive transformations*, `DOI:10.1017/S0305004100019988`.
- `@bethe1936nuclear` — Bethe & Bacher (1936), *Nuclear Physics A. Stationary States of Nuclei*, `DOI:10.1103/RevModPhys.8.82`.
