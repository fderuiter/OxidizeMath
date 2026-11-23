# Domain Physics

Multi-scale physical simulation models from quantum states to astrophysical dynamics.

## 1. Overview & Architecture

The `domain_physics` crate provides computational physics simulation kernels spanning multiple spatial and temporal scales. It translates theoretical formulations—from quantum mechanics and electrodynamics to non-linear chaos and galactic kinematics—into verified numerical algorithms.

Simulation models implement standardized time-stepping interfaces and energy conservation checks, ensuring that numerical approximations maintain physical fidelity throughout simulation runs.

### Module Structure

- **`chaos`**: Deterministic chaotic systems, including conservative Hamiltonian models (`DoublePendulum`, `HenonHeiles`), cubic-Hermite `PoincareSection` detectors, the Lorenz attractor, and Lyapunov exponent analyzers.
- **`quantum`**: Quantum state evolution, Clebsch-Gordan coupling coefficients, and Schrödinger wave equation solvers.
- **`astrophysics`**: Tillaboev (2025) irregular dwarf-galaxy scaling regressions (`GalaxyModel`), baryonic vs. NFW/Burkert dark-matter rotation curves (`decompose_rotation_curve`), zero-allocation symplectic leapfrog disk & tidal-stream orbits (`GalacticDiskSystem`), and planar 3-body gravitational choreographies (`FigureEightThreeBody`).
- **`fluid_dynamics`**: Lattice Boltzmann methods and Navier-Stokes velocity field solvers.
- **`mri`**: Bloch equation magnetic resonance spin simulators and RF pulse excitations.
- **`nuclear`**: Weizsäcker liquid-drop binding-energy breakdowns (`LiquidDropModel::binding_energy_breakdown`), shell-model spin-orbit splittings, Gamow tunneling factors, Breit-Wigner resonances, and stack-allocated `BatemanChain` radioactive decay-chain solvers.
- **`standard_model`**: Electroweak gauge couplings and Weinberg mixing (`GaugeCouplings`), Higgs spontaneous symmetry breaking and $W^\pm/Z^0$ masses (`HiggsPotential`, `boson_masses`), CKM quark mixing and Jarlskog invariants (`construct_ckm`, `jarlskog_invariant`), 1-loop QCD asymptotic freedom (`running_coupling`), and three-flavor PMNS + constant-density MSW matter neutrino oscillations (`PmnsParameters`, `msw_oscillation_probabilities`).
- **`high_energy`**: General-relativistic black-hole and wormhole metrics (`SchwarzschildBlackHole`, `KerrEquatorialMetric`, `MorrisThorneWormhole`), special-relativistic `FourVector` Lorentz boosts and Doppler beaming factors (`relativistic_doppler_factor`, `apparent_transverse_beta`), synchrotron and inverse-Compton radiation (`synchrotron_power`), relativistic shock hydrodynamics, and Li-Ma photon counting significance (`li_ma_significance`).
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

### Subatomic Decay Chains, MSW Neutrino Oscillations, and Relativistic Jets

Evaluate stack-allocated Bateman decay chains, constant-density MSW three-flavor neutrino oscillations, and relativistic Doppler beaming without heap allocations:

```rust
# use domain_physics::__doctest_facade as math_explorer;
use math_explorer::physics::nuclear::{AtomicNumber, BatemanChain, LiquidDropModel, MassNumber};
use math_explorer::physics::standard_model::{
    NeutrinoFlavor, PmnsParameters, msw_oscillation_probabilities,
};
use math_explorer::physics::high_energy::{
    C, FourVector, apparent_transverse_beta, relativistic_doppler_factor,
};

let fe56 = LiquidDropModel::new()
    .binding_energy_breakdown(AtomicNumber::new(26), MassNumber::new(56).unwrap())
    .unwrap();
assert!(fe56.per_nucleon > 8.5);

let chain = BatemanChain::from_half_lives([Some(10.0), Some(10.0), None], [100.0, 0.0, 0.0]).unwrap();
let pops = chain.populations_at(15.0).unwrap();
assert!(((pops[0] + pops[1] + pops[2]) - 100.0).abs() < 1.0e-10);

let pmns = PmnsParameters::pdg_normal_ordering();
let probs = msw_oscillation_probabilities(&pmns, NeutrinoFlavor::Muon, 1300.0, 2.5, 2.8, 0.5, false).unwrap();
assert!(((probs[0] + probs[1] + probs[2]) - 1.0).abs() < 1.0e-12);

let rest_p = FourVector::new(2.0 * C, 0.0, 0.0, 0.0);
let boosted = rest_p.lorentz_boost(-0.6, 0.0, 0.0).unwrap();
assert!(boosted.is_valid_momentum(2.0).unwrap());
assert!((relativistic_doppler_factor(0.6, 0.0).unwrap() - 2.0).abs() < 1.0e-12);
assert!(apparent_transverse_beta(0.98, 0.98_f64.acos()).unwrap() > 4.9);
```

---

## 3. Ecosystem Integration

Models from `domain_physics` are integrated into `math_explorer` and visualized interactively in `math_explorer_gui`. They utilize runtime safety gates from `verified_engine` to enforce zero-allocation stepping during continuous simulation rendering.
