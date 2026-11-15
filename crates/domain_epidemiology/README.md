# Domain Epidemiology

Deterministic and stochastic disease transmission modeling across compartmental and network systems.

## 1. Overview & Architecture

The `domain_epidemiology` crate provides mathematical and computational models for tracking and analyzing the transmission dynamics of infectious diseases.

By formalizing disease spread through ordinary differential equations, stochastic jump processes, and network graph structures, the crate facilitates outbreak modeling, reproduction number ($R_0$) estimation, and intervention efficacy analysis.

### Module Structure

- **`compartmental`**: Classical continuous-time compartmental ODE formulations:
  - `SIRModel`: Susceptible-Infectious-Recovered dynamics.
  - `SEIRModel`: Susceptible-Exposed-Infectious-Recovered models incorporating incubation latencies.
- **`analytics`**: Analytical epidemiology calculators:
  - Basic reproduction number ($R_0$) and effective reproduction number ($R_t$).
  - Epidemic final size transcendental equation solvers and peak infection timing.
- **`stochastic`**: Discrete stochastic event simulation:
  - Gillespie algorithm for demographic stochasticity and extinction probabilities in small cohorts.
- **`networks`**: Network epidemiology on graph topologies (scale-free, small-world contact graphs).
- **`matrix_dynamics`**: Next-generation matrix methods for stratified contact patterns across demographic groups.

---

## 2. Developer Instructions

### Simulating Disease Outbreak Trajectories

Instantiate and simulate an SIR compartmental epidemic model:

```rust
use domain_epidemiology::epidemiology::SIRModel;

// Population: 1000, Initial infected: 10, Transmission rate: 0.5, Recovery rate: 0.1
let mut model = SIRModel::new(1000.0, 10.0, 0.5, 0.1).expect("Valid epidemic parameters");

let dt = 0.1;
let mut peak_infected = 0.0;

// Advance simulation through the epidemic wave
for _ in 0..500 {
    model.step(dt);
    let infected = model.state().i;
    if infected > peak_infected {
        peak_infected = infected;
    }
}

// In an unmitigated outbreak with R0 = 5.0, peak infection exceeds initial infected count
assert!(peak_infected > 10.0);
assert!(model.state().r > 0.0, "Recovered population must accumulate");
```

---

## 3. Ecosystem Integration

Models from `domain_epidemiology` power the epidemiological exploration tools in `math_explorer_gui`, allowing real-time parameter tweaking of contact rates, quarantine delays, and vaccination coverage.
