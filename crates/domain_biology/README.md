# Domain Biology

Computational biology models from molecular kinetics and neural firing to morphogenesis and population evolution.

## 1. Overview & Architecture

The `domain_biology` crate implements mathematical models of biological phenomena across hierarchical scales. The crate models complex living systems through ordinary and partial differential equations, stochastic processes, and spatial pattern formation.

By formalizing biological dynamics into clean, verified Rust algorithms, the crate supports reproducible biological simulation from single-cell dynamics to population ecology.

### Module Structure

- **`neuroscience`**: Biophysical neuron models (`HodgkinHuxleyNeuron` action potential generation, leaky integrate-and-fire spiking patterns, and `KuramotoNetwork` coupled phase-oscillator synchronization).
- **`kinetics`**: Enzyme reaction kinetics (`MichaelisMenten`) and allosteric cooperativity models (`HillKinetics`).
- **`reaction_diffusion`**: Zero-allocation generic $N$-species reaction-diffusion engine (`ReactionDiffusionSystem`, `HillSubstrateReaction`) decoupled over `ReactionModel` and `DiffusionModel` traits.
- **`morphogenesis`**: Mathematical models of biological tissue patterning, `SchnakenbergKinetics` / `GrayScottKinetics` morphogen gradients, and Turing instabilities.
- **`lsystem`**: Zero-allocation deterministic D0L and parametric Lindenmayer vascular branching grammars (`LSystemEngine`, double-buffered $65{,}536$-symbol byte tape, and 2D bracketed turtle geometry).
- **`phyllotaxis`**: Pre-allocated Vogel planar phyllotaxis (`VogelPhyllotaxisEngine`, $r_n = c\sqrt{n}, \theta_n = n\alpha$ up to $4{,}096$ primordia) and opposing Fibonacci $(F_k, F_{k+1})$ parastichy spiral analysis.
- **`lenia`**: Spectral 2D Lenia continuous cellular automata (`LeniaEngine`, concentric-ring `KernelShells`, Gaussian growth curves, and `Orbium` / `Geminium` / `Hydrogeminium` / `Scutium` solitons).
- **`evolution`**: Evolutionary game theory (`HawkDovePopulation`), replicator dynamics, and Evolutionarily Stable Strategy ($p^* = V / C$) equilibria.
- **`diffusion`**: 1D, 2D, and 3D finite-difference Fickian diffusion operators (`FiniteDifference1D`, `FiniteDifference2D` with `NeumannBoundary` / `PeriodicBoundary`, `FiniteDifference3D`) and curved-surface Riemannian diffusion (`LaplaceBeltramiDiffusion2D`).

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
