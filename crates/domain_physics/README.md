# Domain Physics

Multi-scale physical simulation models from quantum states to astrophysical dynamics.

## 1. Overview & Architecture

The `domain_physics` crate provides computational physics simulation kernels spanning multiple spatial and temporal scales. It translates theoretical formulations—from quantum mechanics and electrodynamics to non-linear chaos and galactic kinematics—into verified numerical algorithms.

Simulation models implement standardized time-stepping interfaces—including symplectic integrators (`ImplicitMidpoint`, `GaussLegendre4`, `StormerVerlet`, `Yoshida4`)—and energy/momentum conservation checks, ensuring that numerical approximations maintain physical fidelity throughout long-horizon simulation runs.

### Module Structure

- **`chaos`**: Deterministic chaotic systems, including conservative Hamiltonian models (`DoublePendulum`, `HenonHeilesSystem`, `ThreeBodyChoreography`), cubic-Hermite `PoincareSection` and `PoincareRingBuffer` detectors, the Lorenz attractor, and 3D/4D/6D+ Lyapunov exponent analyzers.
- **`quantum`**: Quantum state evolution, Clebsch-Gordan coupling coefficients, and Schrödinger wave equation solvers.
- **`astrophysics`**: Tillaboev (2025) irregular dwarf-galaxy scaling regressions (`GalaxyModel`), baryonic vs. NFW/Burkert dark-matter rotation curves (`decompose_rotation_curve`), zero-allocation symplectic leapfrog disk & tidal-stream orbits (`GalacticDiskSystem`), and planar 3-body gravitational choreographies (`FigureEightThreeBody`).
- **`fluid_dynamics`**: Lattice Boltzmann methods and Navier-Stokes velocity field solvers.
- **`mri`**: Bloch equation magnetic resonance spin simulators and RF pulse excitations.
- **`nuclear`**: Liquid drop model nuclear binding energies and radioactive decay kinetics.
- **`solid_state`**: Crystal lattice geometries, reciprocal space calculations, and electronic band structures.
- **`stat_mech`**: Ising spin lattices, canonical ensemble thermodynamic calculations, and zero-allocation 2D Bak-Tang-Wiesenfeld Abelian sandpile self-organized criticality & recurrent group engines (`sandpile::SandpileGrid`, `sandpile::SandpileElement`).
- **`optics`**: Geometric ray tracing, refractive boundary equations, and closed-form Fraunhofer wave diffraction (complemented by full-field 2D time-domain multi-slit and refractive-lens wave tank simulation in `pure_math::analysis::pde::wave_2d`).

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

### Symplectic Integration of Conservative Hamiltonian Chaos

Integrate non-separable (`DoublePendulum`) and separable (`HenonHeilesSystem`, `ThreeBodyChoreography`) Hamiltonians with bounded shadow-energy drift:

```rust
# use domain_physics::__doctest_facade as math_explorer;
use math_explorer::physics::chaos::{DoublePendulum, HenonHeilesSystem, ThreeBodyChoreography};
use math_explorer::pure_math::analysis::ode::{
    ArrayState, GaussLegendre4, HamiltonianSystem, RungeKutta4, Solver, Yoshida4,
};

let pendulum = DoublePendulum::default();
let mut s_gl4 = DoublePendulum::default_chaotic_state();
let mut s_rk4 = s_gl4;
let e0 = pendulum.hamiltonian(&s_gl4);
let mut gl4 = GaussLegendre4::new(&s_gl4);
let mut rk4 = RungeKutta4::new(&s_rk4);
for step in 0..400 {
    let t = step as f64 * 0.02;
    gl4.step(&pendulum, t, &mut s_gl4, 0.02);
    rk4.step(&pendulum, t, &mut s_rk4, 0.02);
}
assert!(pendulum.relative_energy_error(&s_gl4, e0) < pendulum.relative_energy_error(&s_rk4, e0));

let hh = HenonHeilesSystem::default();
let mut hh_state = hh.seed_on_poincare_section(0.125, 0.1, 0.0).unwrap();
let hh_e0 = hh.hamiltonian(&hh_state);
let mut yoshida = Yoshida4::new(&ArrayState::<2>::zeros());
yoshida.step_separable(&hh, &mut hh_state, 0.02);
assert!(hh.relative_energy_error(&hh_state, hh_e0) < 1e-8);

let choreo = ThreeBodyChoreography::default();
let nb_state = ThreeBodyChoreography::figure_eight_initial_state();
assert!(choreo.angular_momentum_z(&nb_state).abs() < 1e-12);
```

### Decomposing Dwarf-Galaxy Rotation Curves and Symplectic Disk Orbits

Evaluate Tillaboev (2025) scaling regressions, compare cored (`BurkertCore`) and cuspy (`NfwCusp`) dark-matter halos against Keplerian falloff, and advance zero-allocation leapfrog orbits:

```rust
# use domain_physics::__doctest_facade as math_explorer;
use math_explorer::physics::astrophysics::{
    GalacticDiskSystem, HaloProfile, Mpc, OrbitPopulationConfig, TypeCode10,
    baryons_from_distance, decompose_rotation_curve,
};

let baryons = baryons_from_distance(&TypeCode10, Mpc(12.0), 0.6, 0.8);
let keplerian = decompose_rotation_curve(8.0, &baryons, &HaloProfile::BaryonicOnly);
let burkert = HaloProfile::BurkertCore {
    virial_mass_solar: 1.0e10,
    core_radius_kpc: 2.2,
    concentration: 12.0,
};
let dark_halo = decompose_rotation_curve(8.0, &baryons, &burkert);
assert!(dark_halo.v_circ_kms > keplerian.v_circ_kms);

let mut disk = GalacticDiskSystem::from_galaxy_model(
    &TypeCode10,
    Mpc(12.0),
    0.6,
    0.8,
    burkert,
    OrbitPopulationConfig::default(),
);
disk.step_n_leapfrog(2.0, 20);
assert!(disk.relative_energy_drift() < 1.0e-3);
assert!(disk.relative_angular_momentum_drift() < 1.0e-11);
```

---

## 3. Ecosystem Integration

Models from `domain_physics` are integrated into `math_explorer` and visualized interactively in `math_explorer_gui`. They utilize runtime safety gates from `verified_engine` to enforce zero-allocation stepping during continuous simulation rendering.
