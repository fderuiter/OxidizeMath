# Domain Applied

Applied mathematical modeling for clinical trials, engineering systems, game theory, and surface extraction.

## 1. Overview & Architecture

The `domain_applied` crate translates theoretical mathematics—including calculus, probability, differential geometry, and optimization—into practical models for complex domains.

The crate spans life sciences, material engineering, distributed systems, and quantitative decision theory, encapsulating specialized domain algorithms inside clean and modular Rust interfaces.

### Module Structure

- **`win_ratio`**: Prioritized composite endpoint evaluation for clinical research:
  - `WinRatioAnalysis`: Configurable hierarchical comparison engine for patient pairs.
  - Comparison strategies (`HigherIsBetter`, `LowerIsBetter`, `UnmatchedPairing`).
- **`clinical_trials`**: Biostatistical trial design, sample size determination, and survival curves.
- **`pharmacokinetics`**: Bateman function models for drug absorption, distribution, metabolism, and elimination (ADME).
- **`battery_degradation`**: Lithium-ion capacity fade and solid electrolyte interphase (SEI) growth models.
- **`isosurface`**: 3D volumetric surface extraction algorithms, including Marching Cubes with rolling slice edge caching, dual contouring, and box-SDF padded watertight slab/solid sampling (`sample_watertight_grid`, `IsosurfaceMode`, `analyze_mesh_manifold`).
- **`freesurfer`**: Cortical surface reconstruction pipelines and geodesic cortical thickness calculators.
- **`game_theory`**: Mean Field Games, evolutionary matrix dynamics, and mechanism design.
- **`lorahub` & `grpo`**: Machine learning parameter merging and group relative policy optimization.
- **`cannibalism` & `favoritism`**: Specialized population dynamics models and socio-mathematical scoring functions.

---

## 2. Developer Instructions

### Evaluating Clinical Outcomes with the Win Ratio

Analyze hierarchical composite endpoints using pair-wise comparison strategies:

```rust
# use domain_applied::__doctest_facade as math_explorer;
use math_explorer::applied::win_ratio::pair_comparison::{
    calculate_statistics, HigherIsBetter, UnmatchedPairing, WinRatioAnalysis,
};

// Define patient outcomes: [Survival Time, Hospitalization-Free Days]
let treatment_patients = vec![
    vec![900.0, 800.0],
    vec![900.0, 500.0],
];

let control_patients = vec![
    vec![200.0, 100.0],
    vec![300.0, 150.0],
];

// Configure hierarchical evaluation (Survival evaluated before Hospitalization)
let analysis = WinRatioAnalysis::new()
    .add_strategy(Box::new(HigherIsBetter))
    .add_strategy(Box::new(HigherIsBetter));

// Perform pairwise comparison across cohorts
let (wins, losses) = analysis.evaluate_pairs(&treatment_patients, &control_patients, &UnmatchedPairing);
let statistics = calculate_statistics(wins, losses).expect("Valid statistics");

assert_eq!(wins, 4);
assert_eq!(losses, 0);
assert!(statistics.win_ratio.is_infinite() || statistics.win_ratio > 1.0);
```

### Sampling a Watertight Volumetric Isosurface

Thicken a periodic Gyroid field into a closed 2-manifold and export binary STL bytes:

```rust
# use domain_applied::__doctest_facade as math_explorer;
use math_explorer::applied::isosurface::{
    analyze_mesh_manifold, extract_isosurface, sample_watertight_grid, IsosurfaceMode,
    WatertightGridConfig,
};
use math_explorer::mesh::export_mesh_to_binary_stl;

let config = WatertightGridConfig::cubic(
    std::f64::consts::PI,
    16,
    0.0,
    IsosurfaceMode::Sheet { thickness: 0.5 },
);

let grid = sample_watertight_grid(&config, |x, y, z| {
    x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos()
})
.expect("Valid watertight voxel grid");

let mesh = extract_isosurface(&grid, 0.0).expect("Valid isosurface mesh");
let stats = analyze_mesh_manifold(&mesh);
assert!(stats.is_watertight_two_manifold());

let stl_bytes = export_mesh_to_binary_stl(&mesh).expect("Valid binary STL");
assert_eq!(stl_bytes.len(), 84 + 50 * stats.triangle_count);
```

---

## 3. Ecosystem Integration

Algorithms from `domain_applied` feed interactive simulation calculators in `math_explorer_gui`—including the `Volumetric Isosurfaces` (`IsosurfaceTab` / `IsosurfaceTool`) workbench for implicit TPMS lattices and watertight `.stl`/`.obj` export—enabling visual scenario exploration for clinical biostatisticians, battery engineers, CAD fabrication workflows, and data scientists.
