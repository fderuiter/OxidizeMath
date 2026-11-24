# Canonical Mathematical Specification: `domain_epidemiology`

Canonical governing equations, bibliography references, stability bounds, and validity domains for [`crates/domain_epidemiology`](../../crates/domain_epidemiology/README.md), linked to [`papers/references.bib`](../../papers/references.bib) and [`papers/graph_parameters_rust.tex`](../../papers/graph_parameters_rust.tex).

## 1. Compartmental & Contact-Network Epidemic Models (`SIRModel`, `SEIRModel`, `NetworkEpidemicModel`)

### Governing Equations

Kermack-McKendrick compartmental models (`SIRModel`, `SEIRModel`) evolve susceptible $S$, exposed $E$, infectious $I$, and recovered $R$ populations with constant total size $N = S + E + I + R$ and basic reproduction number $R_0 = \beta / \gamma$:

$$
\frac{dS}{dt} = -\frac{\beta S I}{N}, \qquad \frac{dE}{dt} = \frac{\beta S I}{N} - \sigma E, \qquad \frac{dI}{dt} = \sigma E - \gamma I, \qquad \frac{dR}{dt} = \gamma I
$$

On heterogeneous contact graphs $G = (V, E)$, `NetworkEpidemicModel` advances per-node discrete-time contagion with avoidance probability $\prod_{u \in \mathcal{N}_I(v)} (1 - p_{uv})$ and heterogeneity threshold $R_0^{\text{net}} = \frac{\beta}{\gamma} \frac{\langle k^2 \rangle - \langle k \rangle}{\langle k \rangle}$.

### Citations

- BibTeX keys: [`barefoot1987`](../../papers/references.bib#barefoot1987), [`bafna1999`](../../papers/references.bib#bafna1999) (DOI: `10.1137/S0895480196305124`), [`komusiewicz2025parameter`](../../papers/references.bib#komusiewicz2025parameter), [`norris1997markov`](../../papers/references.bib#norris1997markov).
- Manuscript: [`papers/graph_parameters_rust.tex`](../../papers/graph_parameters_rust.tex).

### Stability

Fourth-order Runge-Kutta (`RK4`) integration conserves $S(t) + E(t) + I(t) + R(t) = N$ to machine precision $O(\Delta t^4)$ when $\max(\beta, \sigma, \gamma)\,\Delta t < 1$, while log-complement product accumulation guards per-edge transmission probabilities in $[0, 1]$ without heap allocation during stepping.

### Domain of Validity

Requires strictly positive total population $N > 0$, non-negative compartment counts $S, E, I, R \ge 0$, finite non-negative rates $\beta, \gamma \ge 0$ (and incubation rate $\sigma > 0$ for `SEIRModel`), per-step probabilities $p \in [0, 1]$ for `NetworkEpidemicModel`, and positive integration horizon/step count.

```rust
use math_explorer::epidemiology::compartmental::SIRModel;

let model = SIRModel::builder()
    .n(1000.0)
    .i0(10.0)
    .beta(0.3)
    .gamma(0.1)
    .build()
    .unwrap();
assert_eq!(model.state().s, 990.0);
assert!((model.beta() / model.gamma() - 3.0).abs() < 1e-12);
```
