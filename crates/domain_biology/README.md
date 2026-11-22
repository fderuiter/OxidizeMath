# Domain Biology

Computational biology models from molecular kinetics and neural firing to morphogenesis and population evolution.

## 1. Overview & Architecture

The `domain_biology` crate implements mathematical models of biological phenomena across hierarchical scales. The crate models complex living systems through ordinary and partial differential equations, stochastic processes, and spatial pattern formation.

By formalizing biological dynamics into clean, verified Rust algorithms, the crate supports reproducible biological simulation from single-cell dynamics to population ecology.

### Module Structure

- **`neuroscience`**: Biophysical neuron models (Hodgkin-Huxley action potential generation, Izhikevich spiking patterns).
- **`kinetics`**: Enzyme reaction kinetics, Michaelis-Menten equations, and cooperativity models (Hill equation).
- **`reaction_diffusion`**: Spatial pattern generation including Gray-Scott and Turing reaction-diffusion systems.
- **`morphogenesis`**: Mathematical models of biological tissue growth, morphogen gradients, and cellular patterning.
- **`evolution`**: Replicator dynamics, selection equations, and evolutionary game theory.
- **`diffusion`**: Molecular concentration gradient solvers and Fickian diffusion mechanics.

---

## 2. Developer Instructions

### Simulating Action Potentials with Hodgkin-Huxley

Simulate electrophysiological voltage spikes in excitable cell membranes:

```rust
# use domain_biology::__doctest_facade as math_explorer;
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;

// Initialize neuron at resting membrane potential (-65.0 mV)
let mut neuron = HodgkinHuxleyNeuron::new(-65.0);

let dt = 0.01; // Integration timestep (ms)
let current_injection = 20.0; // Injected stimulating current (uA/cm^2)
let mut spiked = false;

// Integrate membrane dynamics over time
for _ in 0..500 {
    neuron.update(dt, current_injection);
    if neuron.v() > 0.0 {
        spiked = true;
        break;
    }
}

assert!(spiked, "Neuron should depolarize and fire an action potential under continuous current injection");
```

### Toroidal Gray-Scott Reaction-Diffusion

Simulate seamless 2D morphogenesis with `GrayScottKinetics` and `PeriodicBoundary`:

```rust
# use domain_biology::__doctest_facade as math_explorer;
use math_explorer::biology::diffusion::{BoundaryMode, FiniteDifference2D, PeriodicBoundary};
use math_explorer::biology::morphogenesis::{DiffusionCoeff, GrayScottKinetics, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize, flatten_2d_index};

let (width, height) = (16, 16);
let diffusion = FiniteDifference2D::new_periodic(
    Dimension(width),
    Dimension(height),
    StepSize(1.0),
    StepSize(1.0),
);
assert_eq!(diffusion.boundary_mode, BoundaryMode::Periodic);
let _ = PeriodicBoundary;

let kinetics = GrayScottKinetics::new(0.055, 0.062);
let mut system = TuringSystem::new_with_kinetics(
    Dimension(width * height),
    DiffusionCoeff(0.16),
    DiffusionCoeff(0.08),
    kinetics,
    diffusion,
);

system.u_mut().fill(1.0);
let center = flatten_2d_index(8, 8, width);
system.u_mut()[center] = 0.50;
system.v_mut()[center] = 0.25;

for _ in 0..20 {
    system.step(1.0);
}
assert!(system.v()[center].is_finite() && system.v()[center] > 0.0);
```

---

## 3. Ecosystem Integration

`domain_biology` models provide computational backends for interactive simulation panels in `math_explorer_gui`, allowing users to adjust ionic conductances, morphogen diffusion rates, and population parameters dynamically.
