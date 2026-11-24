# Pharmacokinetics & Compartmental Clearance Specification (`spec:pharmacokinetics_v1`)

- **Spec ID**: `spec:pharmacokinetics_v1`
- **Traceability Module Key**: `pharmacokinetics`

## Governing Equations

Extravascular gut-to-central absorption ($A_g \to A_c$) with distribution volume $V_d$, absorption rate $k_a$, and elimination rate $k_e = \mathrm{CL} / V_d$ obeys:

$$
\frac{dA_g}{dt} = -k_a A_g, \qquad \frac{dA_c}{dt} = k_a A_g - k_e A_c, \qquad C(t) = \frac{F D k_a}{V_d (k_a - k_e)}\bigl(e^{-k_e t} - e^{-k_a t}\bigr)
$$

## Numerical Stability & Error Bounds

- In the flip-flop / confluent limit $|k_a - k_e| \to 0$, the Bateman ratio $(e^{-k_e t} - e^{-k_a t}) / (k_a - k_e)$ switches smoothly to the confluent limit $t e^{-k_e t}$ (or `expm1` formulation) to eliminate $0/0$ catastrophic cancellation.

## Domain of Validity

- Positive volume of distribution $V_d > 0$, bioavailability $F \in (0, 1]$, non-negative dose $D \ge 0$, positive rate constants $k_a, k_e > 0$, and time $t \ge 0$.

## References

- `@bateman1910solution` — Bateman (1910), *The solution of a system of differential equations occurring in the theory of radioactive transformations*, `DOI:10.1017/S0305004100019988`.
