# Battery Degradation & SEI Kinetics Specification (`spec:battery_degradation_v1`)

- **Spec ID**: `spec:battery_degradation_v1`
- **Traceability Module Key**: `battery_degradation`

## Governing Equations

Lithium-ion capacity fade $Q_{\mathrm{loss}}$ under calendar/cycle stress combines power-law SEI diffusion kinetics and Arrhenius thermal activation:

$$
Q_{\mathrm{loss}}(N) = B \, N^z, \qquad k(T, I) = A \exp\!\left(-\frac{E_a}{R T}\right) \bigl(1 + \alpha_{\mathrm{DoD}}\,\mathrm{DoD}\bigr), \qquad \mathrm{SOH}(N) = \max\bigl(0, 1 - Q_{\mathrm{loss}}(N)\bigr)
$$

## Numerical Stability & Error Bounds

- Arrhenius factors evaluate $-E_a / (R T)$ for $T \ge 200\,\mathrm{K}$ where $-E_a / (R T) < 0$, keeping $\exp(-E_a / (R T)) \in (0, 1)$ free of overflow; remaining capacity and State of Health (SOH) are clamped to $[0, Q_0]$ to prevent unphysical negative capacities beyond End-of-Life.

## Domain of Validity

- Absolute temperature $T \in [233.15, 373.15]\,\mathrm{K}$, activation energy $E_a > 0$, power-law exponent $z \in (0, 1.5]$, Depth of Discharge $\mathrm{DoD} \in [0, 1]$, and cycle count $N \ge 0$.

## References

- `@bloom2001accelerated` — Bloom et al. (2001), *An accelerated calendar and cycle life study of Li-ion cells*, `DOI:10.1016/S0378-7753(01)00783-2`.
- `@ning2004cycle` — Ning & Popov (2004), *Cycle Life Modeling of Lithium-Ion Batteries*, `DOI:10.1149/1.1787631`.
