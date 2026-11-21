# Domain Physics

Multi-scale physical simulation models from quantum states to astrophysical dynamics.

## 1. Overview & Architecture

The `domain_physics` crate provides computational physics simulation kernels spanning multiple spatial and temporal scales. It translates theoretical formulations—from quantum mechanics and electrodynamics to non-linear chaos and galactic kinematics—into verified numerical algorithms.

Simulation models implement standardized time-stepping interfaces and energy conservation checks, ensuring that numerical approximations maintain physical fidelity throughout simulation runs.

### Module Structure

- **`chaos`**: Deterministic chaotic systems, including the Lorenz attractor, double pendulum, and Lyapunov exponent analyzers.
- **`quantum`**: Quantum state evolution, Clebsch-Gordan coupling coefficients, and Schrödinger wave equation solvers.
- **`astrophysics`**: N-body gravitational simulations, Keplerian orbital mechanics, and stellar structure models.
- **`fluid_dynamics`**: Lattice Boltzmann methods and Navier-Stokes velocity field solvers.
- **`mri`**: Bloch equation magnetic resonance spin simulators and RF pulse excitations.
- **`nuclear`**: Liquid drop model nuclear binding energies and radioactive decay kinetics.
- **`solid_state`**: Crystal lattice geometries, reciprocal space calculations, and electronic band structures.
- **`stat_mech`**: Ising spin lattices and canonical ensemble thermodynamic calculations.
- **`optics`**: Geometric ray tracing, refractive boundary equations, and wave diffraction.

---

## 2. Developer Instructions

### Simulating Chaotic Dynamics with the Lorenz Attractor

Initialize and advance non-linear dynamical systems:

```rust
# use domain_physics::__doctest_facade as math_explorer;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};

// Initialize system state near the attractor coordinates
let state = LorenzState::new(10.0, 10.0, 10.0);

// Build the Lorenz system with standard chaotic parameters
let mut lorenz = LorenzBuilder::new()
    .sigma(10.0)
    .rho(28.0)
    .beta(8.0 / 3.0)
    .build(state);

// Advance the simulation by timestep dt
lorenz.step(0.01);

let updated_vec = lorenz.state.vec;
assert!(updated_vec.x != 10.0 || updated_vec.y != 10.0);
```

---

## 3. Ecosystem Integration

Models from `domain_physics` are integrated into `math_explorer` and visualized interactively in `math_explorer_gui`. They utilize runtime safety gates from `verified_engine` to enforce zero-allocation stepping during continuous simulation rendering.
