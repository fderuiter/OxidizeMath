# Domain Biology

Computational biology models spanning molecular enzyme kinetics, neural electrophysiology, spectral continuous cellular automata (Lenia), golden-angle botanical phyllotaxis, parametric L-systems, Turing morphogenesis, and evolutionary replicator dynamics.

## 1. Overview & Architecture

The `domain_biology` crate formalizes living systems across hierarchical scales through ordinary and partial differential equations, spectral circular convolutions, parallel rewriting grammars, and evolutionary game theory. Hot simulation loops execute inside pre-allocated buffers with **zero heap allocations** under `VerifiedAllocator`.

### Module Structure

- **`neuroscience`**: Biophysical neuron models (`HodgkinHuxleyNeuron` action potential generation, CSR synaptic networks via `step_csr_hodgkin_huxley_network`, leaky integrate-and-fire spiking patterns, and `KuramotoNetwork` coupled phase-oscillator synchronization).
- **`kinetics`**: Biochemical saturation models including `MichaelisMenten` ($n = 1$) and cooperative sigmoidal `HillKinetics` ($n > 0$) unified under `KineticsModel`.
- **`reaction_diffusion`**: Zero-allocation generic $N$-species reaction-diffusion engine (`ReactionDiffusionSystem`, `HillSubstrateReaction`, `ChemicalState`) decoupled over `ReactionModel` and `DiffusionModel` traits.
- **`morphogenesis`**: Two-species activator-inhibitor pattern solvers (`TuringSystem`, `FusedEulerSolver`) with `SchnakenbergKinetics` and `GrayScottKinetics`.
- **`lsystem`**: Zero-allocation deterministic D0L and parametric Lindenmayer vascular branching grammars (`LSystemEngine`, double-buffered $65{,}536$-symbol byte tape, and 2D bracketed turtle geometry).
- **`phyllotaxis`**: Pre-allocated Vogel planar phyllotaxis (`VogelPhyllotaxisEngine`, $r_n = c\sqrt{n}, \theta_n = n\alpha$ up to $4{,}096$ primordia), opposing Fibonacci $(F_k, F_{k+1})$ parastichy spiral analysis, and biochemical nutrient-coupled plastochron stepping (`NutrientCoupling`, `PhyllotaxisState`).
- **`lenia`**: Spectral 2D Lenia continuous cellular automata (`LeniaEngine`, concentric-ring `KernelShells`, Gaussian growth curves, and `Orbium` / `Geminium` / `Hydrogeminium` / `Scutium` solitons).
- **`evolution`**: Evolutionary game theory (`HawkDovePopulation`, `PayoffMatrix2x2`), replicator dynamics, and Evolutionarily Stable Strategy ($p^* = \min(V / C, 1)$) equilibria.
- **`diffusion`**: 1D, 2D, and 3D finite-difference Fickian diffusion operators (`FiniteDifference1D`, `FiniteDifference2D`, `FiniteDifference3D`) and curved-surface Riemannian diffusion (`LaplaceBeltramiDiffusion2D`).

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
use math_explorer::biology::lenia::{LeniaEngine, LeniaSpecies};
use math_explorer::biology::phyllotaxis::{
    is_consecutive_fibonacci_pair, NutrientCoupling, PhyllotaxisConfig, PhyllotaxisState,
    VogelPhyllotaxisEngine,
};
use math_explorer::rng::OxidizeRng;

// 1. Evolve an Orbium continuous cellular automaton soliton via 2D FFT convolution
let mut rng = OxidizeRng::new(42);
let mut lenia = LeniaEngine::from_species(64, 64, LeniaSpecies::Orbium, 0.0, &mut rng);
lenia.step();
assert!(lenia.telemetry().total_mass > 10.0);

// 2. Pack florets at the golden angle (137.5078°) and step nutrient-coupled plastochrons
let vogel = VogelPhyllotaxisEngine::golden(200, 1.0).unwrap();
assert!(vogel.analysis().is_fibonacci_pair);

let mut phyllo = PhyllotaxisState::new(
    512,
    200,
    PhyllotaxisConfig::default(),
    NutrientCoupling::default(),
);
phyllo.step(0.25);
let (m, n) = phyllo.detect_parastichy_pair();
assert!(is_consecutive_fibonacci_pair(m, n));
```

---

## 3. Ecosystem Integration

`domain_biology` models provide computational backends for `NeuroscienceTab`, `BiochemicalTab`, and the multi-tool `MorphogenesisTab` (`MorphogenesisTool`, `LeniaTool`, `PhyllotaxisLSystemTool`, and `EvolutionKineticsTool`) in `math_explorer_gui`, running identically on native hosts and `wasm32-unknown-unknown` browsers.
