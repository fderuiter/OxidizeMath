# Standard Model & Electroweak Physics Specification (`spec:standard_model_v1`)

- **Spec ID**: `spec:standard_model_v1`
- **Traceability Module Key**: `standard_model`

## Governing Equations

Tree-level electroweak gauge boson masses and Weinberg mixing angle $\theta_W$ follow from spontaneous $SU(2)_L \times U(1)_Y$ symmetry breaking with vacuum expectation value $v$:

$$
m_W = \frac{1}{2} g v, \qquad m_Z = \frac{m_W}{\cos\theta_W}, \qquad \sin^2\theta_W = 1 - \frac{m_W^2}{m_Z^2}
$$

## Numerical Stability & Error Bounds

- Unitarity checks on $3 \times 3$ Cabibbo-Kobayashi-Maskawa (CKM) matrices compute $|V V^\dagger - I|_F$ in `f64` with rounding error bounded by $\mathcal{O}(\varepsilon_{\mathrm{mach}})$.
- Relativistic invariant mass squared $s = (E_1 + E_2)^2 - \|\mathbf{p}_1 + \mathbf{p}_2\|^2$ is clamped against catastrophic cancellation near the massless collinear limit.

## Domain of Validity

- Coupling constants $g, g' > 0$ in the perturbative regime $\alpha_i = g_i^2 / (4\pi) \ll 1$, Higgs VEV $v > 0$, and physical rest masses $m \ge 0$.

## References

- `@weinberg1967model` — Weinberg (1967), *A Model of Leptons*, `DOI:10.1103/PhysRevLett.19.1264`.
