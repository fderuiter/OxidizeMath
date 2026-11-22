# Domain Biology

Computational biology models spanning molecular enzyme kinetics, neural electrophysiology, spectral continuous cellular automata (Lenia), golden-angle botanical phyllotaxis, nutrient-coupled L-systems, Turing morphogenesis, and evolutionary replicator dynamics.

## 1. Overview & Architecture

The `domain_biology` crate formalizes living systems across hierarchical scales through ordinary and partial differential equations, spectral circular convolutions, parallel rewriting grammars, and evolutionary game theory. Hot simulation loops are engineered with preallocated buffers so time-stepping executes with **zero heap allocations** under `VerifiedAllocator`.

### Module Structure

- **`neuroscience`**: Biophysical neuron models (`HodgkinHuxleyNeuron` action potential generation and ion-channel gating kinetics).
- **`kinetics`**: Biochemical saturation models including `MichaelisMenten` ($n = 1$) and cooperative sigmoidal `HillKinetics` ($n > 0$) unified under `KineticsModel`.
- **`reaction_diffusion`**: Generic $N$-species reaction-diffusion engine (`ReactionDiffusionSystem`, `ChemicalState`, `ReactionModel`, `DiffusionModel`).
- **`morphogenesis`**: Two-species activator-inhibitor Turing pattern solvers (`TuringSystem`, `FusedEulerSolver`) with `SchnakenbergKinetics` and `GrayScottKinetics`.
- **`lenia`**: Spectral continuous cellular automata (`LeniaSystem`) using preplanned transposed 2D FFT circular convolution (`rustfft` + `num-complex`), `Orbium` and `Hydrogeminium` presets, and zero-allocation toroidal velocity and rotational symmetry telemetry.
- **`phyllotaxis`**: Vogel golden-angle primordia packing (`PhyllotaxisState`), consecutive Fibonacci parastichy spiral detection (`detect_parastichy_spirals`), and bounded iterative L-system 2D turtle morphogenesis (`LSystemEngine`) coupled to `MichaelisMenten` and `HillKinetics`.
- **`evolution`**: Replicator ordinary differential equations and Evolutionarily Stable Strategy (ESS) dynamics (`HawkDovePopulation`, `PayoffMatrix2x2`).
- **`diffusion`**: Cache-tiled 1D and 2D finite-difference Laplacian operators (`FiniteDifference1D`, `FiniteDifference2D`) with periodic and zero-flux boundaries.

---

## 2. Developer Instructions

### Simulating Action Potentials with Hodgkin-Huxley

```rust
# use domain_biology::__doctest_facade as math_explorer;
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;

let mut neuron = HodgkinHuxleyNeuron::new(-65.0);
let dt = 0.01;
let current_injection = 20.0;
let mut spiked = false;

for _ in 0..500 {
    neuron.update(dt, current_injection);
    if neuron.v() > 0.0 {
        spiked = true;
        break;
    }
}

assert!(spiked, "Neuron should depolarize and fire an action potential");
```

### Spectral Lenia (`Orbium` Glider) & Golden-Angle Phyllotaxis

```rust
# use domain_biology::__doctest_facade as math_explorer;
use math_explorer::biology::lenia::{LeniaPreset, LeniaSystem};
use math_explorer::biology::phyllotaxis::{
    is_consecutive_fibonacci_pair, NutrientCoupling, PhyllotaxisConfig, PhyllotaxisState,
};

// 1. Evolve an Orbium continuous cellular automaton soliton via 2D FFT convolution
let mut lenia = LeniaSystem::from_preset(64, 64, LeniaPreset::Orbium).unwrap();
lenia.step();
assert!(lenia.telemetry().total_mass > 10.0);

// 2. Pack florets at the golden angle (137.50776°) and verify Fibonacci parastichies
let phyllo = PhyllotaxisState::new(
    512,
    200,
    PhyllotaxisConfig::default(),
    NutrientCoupling::default(),
);
let (m, n) = phyllo.detect_parastichy_pair();
assert!(is_consecutive_fibonacci_pair(m, n));
```

---

## 3. Ecosystem Integration

`domain_biology` backs both the `NeuroscienceTab` and the multi-tool `MorphogenesisTab` (`MorphogenesisTool`, `LeniaTool`, `PhyllotaxisLSystemTool`, and `EvolutionKineticsTool`) in `math_explorer_gui`, running identically on native multi-threaded hosts and `wasm32-unknown-unknown` browsers.
