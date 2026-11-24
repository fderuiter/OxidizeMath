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
- **`solid_state`**: Crystal lattice geometries, reciprocal space, electronic band structures, CODATA SI unit bridges (`ElectronVolts`, `Kelvin`), finite-temperature `IsotropicBCSModel` & `BcsGapSolver` superconductivity, Bogoliubov quasiparticle coherence factors, Dynes-broadened NIS tunneling conductance, Debye phonons, Thomas-Fermi/Yukawa screening, Fröhlich vertices, and Heisenberg magnons.
- **`stat_mech`**: Ising spin lattices, SI-bridged `FermiDirac` / `BoseEinstein` / `MaxwellBoltzmann` quantum occupancy (`calculate_occupancy_ev`), log-sum-exp stabilized canonical ensembles (`canonical_thermodynamics_ev`), zero-allocation 2D Bak-Tang-Wiesenfeld Abelian sandpile self-organized criticality & recurrent group engines (`sandpile::SandpileGrid`, `sandpile::SandpileElement`), seeded lattice diffusion, and zero-allocation continuum SPDE engines for Edwards-Wilkinson (`EdwardsWilkinson1D`, `EdwardsWilkinson2D`), Kardar-Parisi-Zhang (`KpzModel1D`, `KpzModel2D`), Allen-Cahn (`AllenCahn2D`), and conserved Cahn-Hilliard (`CahnHilliard2D`) phase-field dynamics.
- **`optics`**: Graded-index and negative-index refractive media (`refraction`), RK4 Hamiltonian ray tracing and caustic envelope detection (`caustics`), and calibrated optical detector models (`sensing`), complemented by full-field 2D time-domain FDTD and Verlet wave solvers in `pure_math::analysis::pde::{wave, wave_2d}`.

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

### Solving Finite-Temperature BCS Gaps and SI Quantum Occupancy

Evaluate finite-temperature Cooper-pair energy gaps, universal weak-coupling $2\Delta(0)/(k_B T_c) \approx 3.528$ scaling, Dynes-broadened NIS tunneling conductance, and CODATA SI-bridged Fermi-Dirac occupancy:

```rust
# use domain_physics::__doctest_facade as math_explorer;
use math_explorer::physics::solid_state::bcs::{
    BcsGapSolver, IsotropicBCSModel, nis_tunneling_conductance,
};
use math_explorer::physics::solid_state::types::{ElectronVolts, Kelvin};
use math_explorer::physics::stat_mech::quantum_stats::{FermiDirac, calculate_occupancy_ev};

let cutoff = 0.024;
let d_xi = (2.0 * cutoff) / 160.0;
let levels: Vec<f64> = (0..160).map(|i| -cutoff + (i as f64 + 0.5) * d_xi).collect();
let model = IsotropicBCSModel::new(levels, 0.30 * d_xi, cutoff).unwrap();
let solver = BcsGapSolver::new().with_tolerance(1e-11).with_max_iterations(250);

let tc = model.critical_temperature().unwrap();
let delta_0 = solver.solve(&model, ElectronVolts::new(0.0)).unwrap();
let ratio = (2.0 * delta_0.as_f64()) / tc.thermal_energy_ev().as_f64();
assert!((ratio - 3.528).abs() < 0.06);

let normal = model.with_temperature(Kelvin::new(tc.as_f64() + 1.0));
let delta_above = solver.solve(&normal, ElectronVolts::new(0.0)).unwrap();
assert_eq!(delta_above.as_f64(), 0.0);
let g_normal = nis_tunneling_conductance(
    ElectronVolts::new(0.0),
    delta_above,
    ElectronVolts::new(4e-5),
    Kelvin::new(tc.as_f64() + 1.0),
)
.unwrap();
assert!((g_normal - 1.0).abs() < 1e-12);

// Physical eV/Kelvin bridge: E - mu = 0.10 eV at 300 K yields ~0.0205, avoiding raw-SI 0.49 clamping
let f_fd = calculate_occupancy_ev(
    FermiDirac,
    ElectronVolts::new(0.10),
    ElectronVolts::new(0.0),
    Kelvin::new(300.0),
)
.unwrap();
assert!((f_fd - 0.020_47).abs() < 1e-3);
```

---

## 3. Ecosystem Integration

Models from `domain_physics` are integrated into `math_explorer` and visualized interactively in `math_explorer_gui`. They utilize runtime safety gates from `verified_engine` to enforce zero-allocation stepping during continuous simulation rendering.
