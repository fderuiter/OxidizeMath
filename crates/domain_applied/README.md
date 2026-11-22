# Domain Applied

Applied mathematical modeling for clinical trials, engineering systems, game theory, and surface extraction.

## 1. Overview & Architecture

The `domain_applied` crate translates theoretical mathematics—including calculus, probability, differential geometry, and optimization—into practical models for complex domains.

The crate spans life sciences, material engineering, distributed systems, and quantitative decision theory, encapsulating specialized domain algorithms inside clean and modular Rust interfaces.

### Module Structure

- **`win_ratio`**: Prioritized composite endpoint evaluation for clinical research:
  - `WinRatioAnalysis`: Configurable hierarchical comparison engine for patient pairs with matched and unmatched (`evaluate_unmatched`, `evaluate_unmatched_into`) evaluation.
  - `censoring`: Right-censored time-to-event outcomes (`CensoredOutcome`), Pocock/Finkelstein-Schoenfeld comparators (`RightCensoredComparator`), and hierarchical tier waterfall counters (`TierCount`).
  - `u_statistic`: Pocock/Bebu-Lachin two-sample U-statistic variance projection (`UnmatchedWinRatioSummary`, `UnmatchedScratch`, `compute_unmatched_u_statistics`) calibrating confidence intervals and Type I error across dependent all-pairs comparisons.
  - `simulation`, `weibull_sampler`, & `power_sweep`: Exact two-step inverse-CDF `BivariateWeibullModel` sampling driven by `OxidizeRng` and zero-allocation Monte Carlo power sweeps (`run_power_sweep_into`, `run_weibull_power_sweep`).
- **`clinical_trials`**: Biostatistical trial design, analytical sample-size and power curves (`calculate_power_means`, `calculate_power_proportions`), binary risk metrics (`calculate_risk_metrics`), and Kaplan-Meier / hazard-ratio survival analysis (`try_estimate_hazard_ratio`).
- **`pharmacokinetics`**: Bateman function models for drug absorption, distribution, metabolism, and elimination (ADME).
- **`battery_degradation`**: Lithium-ion capacity fade and solid electrolyte interphase (SEI) growth models.
- **`isosurface`**: 3D volumetric surface extraction algorithms, including Marching Cubes and dual contouring.
- **`freesurfer`**: Cortical surface reconstruction pipelines and geodesic cortical thickness calculators.
- **`game_theory`**: Mean Field Games, evolutionary matrix dynamics, and mechanism design.
- **`lorahub` & `grpo`**: Machine learning parameter merging and group relative policy optimization.
- **`cannibalism` & `favoritism`**: Specialized population dynamics models and socio-mathematical scoring functions.

---

## 2. Developer Instructions

### Evaluating Right-Censored Composite Outcomes with U-Statistics

Analyze prioritized fatal and non-fatal time-to-event endpoints under right censoring with two-sample U-statistic inference:

```rust
# use domain_applied::__doctest_facade as math_explorer;
use math_explorer::applied::win_ratio::pair_comparison::{
    CensoredOutcome, RightCensoredComparator, WinRatioAnalysis,
};

// Hierarchy: [Tier 1: Fatal Event, Tier 2: Non-Fatal Hospitalization]
let treatment = vec![
    vec![
        CensoredOutcome::new(5.0, false).unwrap(),
        CensoredOutcome::new(5.0, false).unwrap(),
    ],
    vec![
        CensoredOutcome::new(4.8, false).unwrap(),
        CensoredOutcome::new(3.2, true).unwrap(),
    ],
];

let control = vec![
    vec![
        CensoredOutcome::new(1.5, true).unwrap(),
        CensoredOutcome::new(0.8, true).unwrap(),
    ],
    vec![
        CensoredOutcome::new(4.5, false).unwrap(),
        CensoredOutcome::new(1.9, true).unwrap(),
    ],
];

let analysis = WinRatioAnalysis::new()
    .add_strategy(Box::new(RightCensoredComparator::new()))
    .add_strategy(Box::new(RightCensoredComparator::new()));

let (summary, tiers) = analysis
    .evaluate_unmatched(&treatment, &control, 0.05)
    .expect("Valid unmatched U-statistic summary");

assert_eq!(summary.total_pairs, 4);
assert_eq!(tiers[0].wins + tiers[1].wins, 4);
assert!(summary.win_ratio > 1.0);
```

---

## 3. Ecosystem Integration

Algorithms from `domain_applied` feed interactive simulation calculators in `math_explorer_gui`—including `WinRatioStressTesterTool`, `SampleSizeCalculatorTool`, and `SurvivalAnalysisTool`—enabling visual scenario exploration and reproducible protocol exports for clinical biostatisticians, battery engineers, and data scientists.
