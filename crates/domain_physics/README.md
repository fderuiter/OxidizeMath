# Domain Physics

Multi-scale physical simulation models from quantum states to astrophysical dynamics.

## 1. Overview & Architecture

The `domain_physics` crate provides computational physics simulation kernels spanning multiple spatial and temporal scales. It translates theoretical formulations—from quantum mechanics and electrodynamics to non-linear chaos and galactic kinematics—into verified numerical algorithms.

Simulation models implement standardized time-stepping interfaces and energy conservation checks, ensuring that numerical approximations maintain physical fidelity throughout simulation runs.

### Module Structure

- **`chaos`**: Deterministic chaotic systems, including conservative Hamiltonian models (`DoublePendulum`, `HenonHeiles`), cubic-Hermite `PoincareSection` detectors, the Lorenz attractor, and Lyapunov exponent analyzers.
- **`quantum`**: Quantum state evolution, Clebsch-Gordan coupling coefficients, and Schrödinger wave equation solvers.
- **`astrophysics`**: Tillaboev (2025) irregular dwarf-galaxy scaling regressions (`GalaxyModel`), baryonic vs. NFW/Burkert dark-matter rotation curves (`decompose_rotation_curve`), zero-allocation symplectic leapfrog disk & tidal-stream orbits (`GalacticDiskSystem`), and planar 3-body gravitational choreographies (`FigureEightThreeBody`).
- **`high_energy`**: Schwarzschild, equatorial Kerr, and Morris-Thorne spacetime metrics, zero-allocation stack-based geodesic ray/plunge bundles (`SchwarzschildGeodesicBundle`), Lorentz boosts, relativistic twin-jet Doppler beaming, shock compression, and Li & Ma gamma-ray detection significance (`evaluate_jet_knot_observables`).
- **`fluid_dynamics`**: Lattice Boltzmann methods and Navier-Stokes velocity field solvers.
- **`mri`**: Bloch equation magnetic resonance spin simulators and RF pulse excitations.
- **`nuclear`**: Liquid drop model nuclear binding energies and radioactive decay kinetics.
- **`solid_state`**: Crystal lattice geometries, reciprocal space calculations, and electronic band structures.
- **`stat_mech`**: Ising spin lattices and canonical ensemble thermodynamic calculations.
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

### Tracing Schwarzschild Geodesics and Relativistic Jet Doppler Beaming

Advance pre-allocated null lensing fans and sub-ISCO plunges alongside coupled shock-compressed jet beaming and Li & Ma significance:

```rust
# use domain_physics::__doctest_facade as math_explorer;
use math_explorer::physics::high_energy::{
    JetKnotConfig, SchwarzschildGeodesicBundle, evaluate_jet_knot_observables,
};

let mut bundle = SchwarzschildGeodesicBundle::new(1.0, 8, 128)?;
bundle.seed_photon_lensing_fan(8, 2.60, 17.45, 0.8);
let _ = bundle.step_n_verified(0.01, 4_000);
assert!(bundle.max_killing_drift() < 1.0e-6);

let obs = evaluate_jet_knot_observables(&JetKnotConfig {
    bulk_beta: 0.9,
    viewing_angle_rad: 15.0_f64.to_radians(),
    disk_r_over_rs: 3.0,
    disk_inclination_rad: 90.0_f64.to_radians(),
    electron_index_p: 2.4,
    electron_gamma: 1.0e3,
    upstream_u_b: 1.0e-3,
    adiabatic_index: 4.0 / 3.0,
    intrinsic_signal_counts: 1.0,
    background_counts_off: 40.0,
    exposure_ratio: 1.0,
})?;
assert!((obs.jet_flux_asymmetry - 18_855.15).abs() < 0.1);
# Ok::<(), math_explorer::error::HighEnergyError>(())
```

---

## 3. Ecosystem Integration

Models from `domain_physics` are integrated into `math_explorer` and visualized interactively in `math_explorer_gui`. They utilize runtime safety gates from `verified_engine` to enforce zero-allocation stepping during continuous simulation rendering.
