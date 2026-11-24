# Mathematical Biology, Excitable Media & Morphogenesis Specification (`spec:biology_v1`)

- **Spec ID**: `spec:biology_v1`
- **Traceability Module Key**: `biology`

## Governing Equations

1. **Hodgkin-Huxley membrane electrophysiology**:

$$
C_m \frac{dV}{dt} = I_{\mathrm{ext}} - \bar{g}_{Na} m^3 h (V - E_{Na}) - \bar{g}_K n^4 (V - E_K) - \bar{g}_L (V - E_L)
$$

2. **Turing reaction-diffusion morphogenesis** (Schnakenberg & Gray-Scott):

$$
\partial_t u = D_u \nabla^2 u + R_u(u, v), \qquad \partial_t v = D_v \nabla^2 v + R_v(u, v)
$$

3. **Continuous cellular automata (Lenia)** with Gaussian growth mapping $G(U; \mu, \sigma) = 2 \exp\!\bigl(-(U - \mu)^2 / (2\sigma^2)\bigr) - 1$:

$$
A^{t + \Delta t} = \mathrm{clamp}_{[0, 1]}\!\bigl(A^t + \Delta t \, G(K * A^t)\bigr)
$$

## Numerical Stability & Error Bounds

- Explicit 2D Laplacian finite differences require parabolic CFL step restriction $\Delta t \le \Delta x^2 / (4 \max(D_u, D_v))$ for von Neumann $L^2$ stability.
- Hodgkin-Huxley voltage-gated rate ratios $(V + V_0) / (1 - e^{-(V + V_0)/s})$ apply L'Hôpital smooth continuation near removable singularities, while gating variables $(m, h, n)$ and Lenia states $A$ remain bounded on $[0, 1]$.

## Domain of Validity

- Membrane capacitance $C_m > 0$, conductances $\bar{g}_i \ge 0$, diffusivities $D_u, D_v \ge 0$, Lenia growth width $\sigma > 0$, positive grid spacing $\Delta x > 0$, and CFL-compliant $\Delta t > 0$.

## References

- `@hodgkin1952quantitative` — Hodgkin & Huxley (1952), *A quantitative description of membrane current and its application to conduction and excitation in nerve*, `DOI:10.1113/jphysiol.1952.sp004764`.
- `@turing1952chemical` — Turing (1952), *The chemical basis of morphogenesis*, `DOI:10.1098/rstb.1952.0012`.
- `@schnakenberg1979simple` — Schnakenberg (1979), *Simple chemical reaction systems with limit cycle behaviour*, `DOI:10.1016/0022-5193(79)90042-0`.
- `@pearson1993complex` — Pearson (1993), *Complex patterns in a simple system*, `DOI:10.1126/science.261.5118.189`.
- `@chan2019lenia` — Chan (2019), *Lenia: Biology of Artificial Life*, `DOI:10.25088/ComplexSystems.28.3.251`, `arXiv:1812.05433`.
