# Domain Applied

Applied mathematical modeling for clinical trials, engineering systems, game theory, and surface extraction.

## 1. Overview & Architecture

The `domain_applied` crate translates theoretical mathematics—including calculus, probability, differential geometry, and optimization—into practical models for complex domains.

The crate spans life sciences, material engineering, distributed systems, and quantitative decision theory, encapsulating specialized domain algorithms inside clean and modular Rust interfaces.

### Module Structure

- **`win_ratio`**: Prioritized composite endpoint evaluation for clinical research:
  - `WinRatioAnalysis`: Configurable hierarchical comparison engine for patient pairs, including per-tier waterfall counts (`TierOutcomeCounts`, `evaluate_tiers`, `evaluate_tiers_into`).
  - Comparison and pairing strategies (`HigherIsBetter`, `LowerIsBetter`, `ThresholdComparator`, `UnmatchedPairing`, `MatchedPairing`).
  - Analytical bivariate survival simulation (`BivariateWeibullModel`, `ProbabilityWinRatioContext`, `WinLossCounts`).
- **`clinical_trials`**: Biostatistical trial design (`SimpleRandomizer`, `BlockRandomizer`, `StratifiedRandomizer`), sample size and power determination, hypothesis testing (`TTestIndependent`, `ChiSquare2x2`, `calculate_risk_metrics`), and survival analysis (`kaplan_meier`, `try_estimate_hazard_ratio`, `log_rank_test`).
- **`pharmacokinetics`**: Bateman function models for drug absorption, distribution, metabolism, and elimination (ADME).
- **`battery_degradation`**: Lithium-ion capacity fade and solid electrolyte interphase (SEI) growth models.
- **`isosurface`**: 3D volumetric surface extraction algorithms, including Marching Cubes and dual contouring.
- **`freesurfer`**: Cortical surface reconstruction pipelines and geodesic cortical thickness calculators.
- **`game_theory`**: Mean Field Games, evolutionary matrix dynamics, and mechanism design.
- **`lorahub` & `grpo`**: Machine learning parameter merging and group relative policy optimization.
- **`cannibalism` & `favoritism`**: Specialized population dynamics models and socio-mathematical scoring functions.

---

## 2. Developer Instructions

### Evaluating Clinical Outcomes with the Win Ratio

Analyze hierarchical composite endpoints and per-tier waterfall counts using pair-wise comparison strategies:

```rust
# use domain_applied::__doctest_facade as math_explorer;
use math_explorer::applied::win_ratio::pair_comparison::{
    calculate_statistics, HigherIsBetter, LowerIsBetter, UnmatchedPairing, WinRatioAnalysis,
};

// Define patient outcomes: [Survival Days (HigherIsBetter), Hospitalizations (LowerIsBetter)]
let treatment_patients = vec![
    vec![900.0, 0.0],
    vec![300.0, 1.0],
];

let control_patients = vec![
    vec![300.0, 4.0],
    vec![200.0, 2.0],
];

// Configure hierarchical evaluation (Survival evaluated before Hospitalizations)
let analysis = WinRatioAnalysis::new()
    .add_strategy(Box::new(HigherIsBetter))
    .add_strategy(Box::new(LowerIsBetter));

// Perform pairwise and per-tier waterfall comparison across cohorts
let tiers = analysis
    .evaluate_tiers(&treatment_patients, &control_patients, &UnmatchedPairing)
    .expect("Valid tier evaluation");
let (wins, losses) = analysis.evaluate_pairs(&treatment_patients, &control_patients, &UnmatchedPairing);
let statistics = calculate_statistics(wins, losses).expect("Valid statistics");

assert_eq!(tiers[0].wins, 3);
assert_eq!(tiers[1].wins, 1);
assert_eq!(wins, 4);
assert_eq!(losses, 0);
assert!(statistics.win_ratio.is_infinite() || statistics.win_ratio > 1.0);
```

---

## 3. Ecosystem Integration

Algorithms from `domain_applied` power interactive simulation workbenches in `math_explorer_gui`—including the Win Ratio Workbench, two-arm Kaplan-Meier survival curves with log-rank testing and risk tables, seeded stratified randomization schedules, and power-curve sample size calculators—enabling local scenario exploration for clinical biostatisticians, battery engineers, and data scientists.
