# Domain Physics

Multi-scale physical simulation models from quantum states to astrophysical dynamics.

## 1. Overview & Architecture

The `domain_physics` crate provides computational physics simulation kernels spanning multiple spatial and temporal scales. It translates theoretical formulations—from quantum mechanics and electrodynamics to non-linear chaos and galactic kinematics—into verified numerical algorithms.

Simulation models implement standardized time-stepping interfaces—including symplectic integrators (`ImplicitMidpoint`, `GaussLegendre4`, `StormerVerlet`, `Yoshida4`)—and energy/momentum conservation checks, ensuring that numerical approximations maintain physical fidelity throughout long-horizon simulation runs.

### Module Structure

- **`chaos`**: Deterministic chaotic systems, including conservative Hamiltonian models (`DoublePendulum`, `HenonHeilesSystem`, `ThreeBodyChoreography`), cubic-Hermite `PoincareSection` and `PoincareRingBuffer` detectors, the Lorenz attractor, and 3D/4D/6D+ Lyapunov exponent analyzers.
- **`quantum`**: Quantum state evolution, Clebsch-Gordan coupling coefficients, and Schrödinger wave equation solvers.
- **`astrophysics`**: Tillaboev (2025) irregular dwarf-galaxy scaling regressions (`GalaxyModel`), baryonic vs. NFW/Burkert dark-matter rotation curves (`decompose_rotation_curve`), zero-allocation symplectic leapfrog disk & tidal-stream orbits (`GalacticDiskSystem`), and planar 3-body gravitational choreographies (`FigureEightThreeBody`).
- **`high_energy`**: Schwarzschild, equatorial Kerr, and Morris-Thorne spacetime metrics, zero-allocation stack-based geodesic ray/plunge bundles (`SchwarzschildGeodesicBundle`), Lorentz boosts, relativistic twin-jet Doppler beaming, shock compression, and Li & Ma gamma-ray detection significance (`evaluate_jet_knot_observables`).
- **`fluid_dynamics`**: Lattice Boltzmann methods and Navier-Stokes velocity field solvers.
- **`mri`**: Bloch equation magnetic resonance spin simulators and RF pulse excitations.
- **`nuclear`**: Liquid drop model nuclear binding energies and radioactive decay kinetics.
- **`solid_state`**: Periodic Bravais lattices (`SimpleCubic`, `BodyCenteredCubic`, `FaceCenteredCubic`), cut-and-project quasicrystals (`Fibonacci1D`, `Penrose5Fold`, `AmmannBeenker8Fold`, `Penrose10Fold`, `Stampfli12Fold`) with phason flip tracking, reciprocal-lattice bases, Brillouin zones, zero-allocation kinematic Bragg structure factors (`BraggWorkspace`), 1D Fibonacci and Aubry-André quasiperiodic eigensolvers (`QuasiperiodicWorkspace`), isotropic s-wave BCS pairing (`BcsGapSolver`, `IsotropicBCSModel`, `coherence_factors_strong`), Debye/Einstein phonons, Fröhlich coupling, and Thomas-Fermi/Yukawa screening.
- **`stat_mech`**: Ising spin lattices, canonical ensemble thermodynamic calculations, zero-allocation 2D Bak-Tang-Wiesenfeld Abelian sandpile self-organized criticality & recurrent group engines (`sandpile::SandpileGrid`, `sandpile::SandpileElement`), and zero-allocation continuum SPDE engines for Edwards-Wilkinson (`EdwardsWilkinson1D`, `EdwardsWilkinson2D`), Kardar-Parisi-Zhang (`KpzModel1D`, `KpzModel2D`), Allen-Cahn (`AllenCahn2D`), and conserved Cahn-Hilliard (`CahnHilliard2D`) phase-field dynamics.
- **`optics`**: Zero-allocation geometric & physical optics (`refraction`, `ray_tracing`, `caustics`, `diffraction`, `sensing`), covering Snell/Fresnel polarization, Cauchy dispersion, negative-index Veselago metamaterials, RK4 Hamiltonian & eikonal GRIN lenses (Luneburg, Maxwell fish-eye, Eaton, parabolic waveguide), coffee-cup nephroid & spherical-aberration caustics, Fraunhofer single-slit, grating, and circular Airy diffraction, and calibrated optical detector models (complemented by full-field 2D time-domain FDTD and Verlet wave solvers in `pure_math::analysis::pde::{wave, wave_2d}`).

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

### Tracing Caustic Envelopes and Fraunhofer Diffraction Profiles

Evaluate Brewster polarization extinction, extract coffee-cup nephroid caustics, and sample circular Airy diffraction into pre-allocated buffers:

```rust
# use domain_physics::__doctest_facade as math_explorer;
use math_explorer::physics::optics::{
    AIRY_FIRST_ZERO_J1, CausticEnvelopeBuffer, OpticalBenchScene, PolarizationMode, RayFanConfig,
    RayFanWorkspace, airy_normalized_intensity, brewster_angle, coffee_cup_cusp_x,
    extract_caustic_envelope, fresnel_polarized, trace_ray_fan,
};

let theta_b = brewster_angle(1.0, 1.5168);
let fresnel = fresnel_polarized(1.0, 1.5168, theta_b);
assert!(fresnel.reflectance_p < 1.0e-24);
assert!((fresnel.reflectance_s + fresnel.transmittance_s - 1.0).abs() < 1.0e-12);
assert!(airy_normalized_intensity(AIRY_FIRST_ZERO_J1) < 1.0e-12);

let mut ws = RayFanWorkspace::new(64, 8);
let cfg = RayFanConfig {
    scene: OpticalBenchScene::CoffeeCupMirror { radius: 2.0 },
    ray_count: 51,
    beam_half_width: 1.9,
    source_x: -2.5,
    source_y: 0.0,
    launch_angle_rad: 0.0,
    wavelength_nm: 550.0,
    polychromatic: false,
    polarization: PolarizationMode::Unpolarized,
    bench_half_extent: 3.0,
};
trace_ray_fan(&cfg, &mut ws);
let mut caustics = CausticEnvelopeBuffer::new(64, 32, -2.0, 2.0);
assert!(extract_caustic_envelope(&ws, 1.0, 3.0, &mut caustics) > 30);
assert_eq!(coffee_cup_cusp_x(2.0), 1.0);
```

### Cut-and-Project Quasicrystals and Zero-Allocation Bragg Diffraction

Project higher-dimensional $\mathbb{Z}^D$ lattices onto physical and perpendicular subspaces, track discrete phason vertex flips, and evaluate kinematic Bragg structure factors $S(\mathbf{q})$ in preallocated workspaces:

```rust
# use domain_physics::__doctest_facade as math_explorer;
use math_explorer::physics::solid_state::quasicrystal::{
    QuasicrystalPatchConfig, QuasicrystalSymmetry, QuasicrystalWorkspace,
};
use math_explorer::physics::solid_state::reciprocal::BraggWorkspace;

let mut qc = QuasicrystalWorkspace::new();
let cfg = QuasicrystalPatchConfig {
    symmetry: QuasicrystalSymmetry::Penrose5Fold,
    patch_radius: 3.5,
    phason_shift: [0.2, 0.1],
    ..QuasicrystalPatchConfig::default()
};
qc.generate_into(&cfg);
assert!(qc.sites().len() > 20 && qc.flipped_in_count > 0);

let mut bragg = BraggWorkspace::new(64);
bragg.evaluate_grid_verified(qc.parallel_points(), 64, 14.0, true);
assert!(bragg.peak_intensity() > 1.0);
```

---

## 3. Ecosystem Integration

Models from `domain_physics` are integrated into `math_explorer` and visualized interactively in `math_explorer_gui`. They utilize runtime safety gates from `verified_engine` to enforce zero-allocation stepping during continuous simulation rendering.
