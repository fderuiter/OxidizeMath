# Compartmental Epidemiology Specification (`spec:epidemiology_v1`)

- **Spec ID**: `spec:epidemiology_v1`
- **Traceability Module Key**: `epidemiology`

## Governing Equations

Closed-population SIR and SEIR compartmental transmission models with total population $N = S + E + I + R$ evolve according to:

$$
\frac{dS}{dt} = -\beta \frac{S I}{N}, \qquad \frac{dE}{dt} = \beta \frac{S I}{N} - \sigma E, \qquad \frac{dI}{dt} = \sigma E - \gamma I, \qquad \frac{dR}{dt} = \gamma I
$$

(reducing to standard Kermack-McKendrick SIR as $\sigma \to \infty$ with basic reproduction number $\mathcal{R}_0 = \beta / \gamma$).

## Numerical Stability & Error Bounds

- Fourth-order Runge-Kutta (RK4) integration preserves total population mass $\sum_c y_c(t) = N$ up to $\mathcal{O}(\Delta t^5)$ local truncation error and $\mathcal{O}(N \varepsilon_{\mathrm{mach}})$ rounding drift when $\Delta t \cdot \max(\beta, \sigma, \gamma) < 1$.

## Domain of Validity

- Strictly positive population $N > 0$, non-negative compartments $S, E, I, R \ge 0$, non-negative rates $\beta, \gamma \ge 0$ (and $\sigma > 0$ for SEIR), and positive step $\Delta t > 0$.

## References

- `@kermack1927contribution` — Kermack & McKendrick (1927), *A contribution to the mathematical theory of epidemics*, `DOI:10.1098/rspa.1927.0118`.
- `@anderson1991infectious` — Anderson & May (1991), *Infectious Diseases of Humans: Dynamics and Control*, `DOI:10.1093/oso/9780198545996.001.0001`.
