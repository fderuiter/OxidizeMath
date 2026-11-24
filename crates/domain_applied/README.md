# Domain Applied

Applied mathematical modeling for clinical trials, engineering systems, game theory, and surface extraction.

## 1. Overview & Architecture

The `domain_applied` crate translates theoretical mathematics—including calculus, probability, differential geometry, and optimization—into practical models for complex domains.

The crate spans life sciences, material engineering, distributed systems, and quantitative decision theory, encapsulating specialized domain algorithms inside clean and modular Rust interfaces.

### Module Structure

- **`win_ratio`**: Prioritized composite endpoint evaluation for clinical research:
  - `WinRatioAnalysis`: Configurable hierarchical comparison engine for patient pairs with matched and unmatched (`evaluate_unmatched`, `evaluate_unmatched_into`, `evaluate_tiers`, `evaluate_tiers_into`, `TierOutcomeCounts`) evaluation.
  - Comparison and pairing strategies (`HigherIsBetter`, `LowerIsBetter`, `ThresholdComparator`, `UnmatchedPairing`, `MatchedPairing`).
  - `censoring`: Right-censored time-to-event outcomes (`CensoredOutcome`), Pocock/Finkelstein-Schoenfeld comparators (`RightCensoredComparator`), and hierarchical tier waterfall counters (`TierCount`).
  - `u_statistic`: Pocock/Bebu-Lachin two-sample U-statistic variance projection (`UnmatchedWinRatioSummary`, `UnmatchedScratch`, `compute_unmatched_u_statistics`) calibrating confidence intervals and Type I error across dependent all-pairs comparisons.
  - `probability_win_ratio`, `sample_win_ratio`, `simulation`, `weibull_sampler`, & `power_sweep`: Analytical Clenshaw-Curtis probability win ratios (`ProbabilityWinRatioContext`, `WinLossCounts`), exact two-step inverse-CDF `BivariateWeibullModel` sampling driven by `OxidizeRng`, and zero-allocation Monte Carlo power sweeps (`run_power_sweep_into`, `run_weibull_power_sweep`).
- **`clinical_trials`**: Biostatistical trial design (`SimpleRandomizer`, `BlockRandomizer`, `StratifiedRandomizer`), analytical sample-size and power curves (`calculate_power_means`, `calculate_power_proportions`), hypothesis testing (`TTestIndependent`, `ChiSquare2x2`), binary risk metrics (`calculate_risk_metrics`), and Kaplan-Meier / log-rank / hazard-ratio survival analysis (`kaplan_meier`, `log_rank_test`, `try_estimate_hazard_ratio`).
- **`pharmacokinetics`**: One-compartment oral Bateman ADME kinetics (`PKParameters`, `BatemanModel`, `half_life`, `t_max`, `solve_ka`, `calibrate_ka_from_half_life`), uniform (`SuperpositionModel`) and variable per-event multi-dose superposition (`DoseEvent`, `RegimenSuperposition`), biphasic extended-release formulations (`TwoPulseModel`), stereoselective chiral mixtures (`EnantiomerModel`), and zero-allocation therapeutic-window exposure analytics (`TherapeuticWindow`, `ExposureSummary`).
- **`battery_degradation`**: Coupled lithium-ion electrochemical aging (`ChemistryPreset`, `ElectrochemicalAgingModel`, `PowerLawModel`) combining DoD stress, stretched exponent $z$, Arrhenius thermal activation, C-rate stress, parabolic $\sqrt{t}$ SEI calendar fade, stack-allocated Levenberg-Marquardt CSV calibration (`calibrate_electrochemical_model`), and multi-segment duty warranty solvers (`DutySegment`, `estimate_duty_lifetime`).
- **`isosurface`**: 3D volumetric surface and watertight scalar-collar solid extraction (`MarchingCubes`, `IsosurfaceWorkspace`, `SolidClosureConfig`), including cutting-plane cap closure and dual contouring.
- **`freesurfer`**: Triangulated cortical surface reconstruction (`Surface`, `MriVolume`, thin-plate `internal_energy`, Gaussian-smoothed `external_energy_volume`, `evolve_surface_deformable`), 6-connected Potts MRF Bayesian segmentation (`TissueClass`, `normalized_posteriors`, `mrf_segment_volume`), zero-allocation CSR spatial-grid cortical thickness maps (`FlatSpatialGrid`, `ThicknessWorkspace`, `cortical_thickness`, `cortical_thickness_indexed`, `cortical_thickness_map`, `cortical_thickness_map_into`), and vertex-wise OLS GLM contrasts (`estimate_beta`, `t_statistic`).
- **`game_theory`**: Normal-form matrix/bimatrix solvers (`iterated_dominance`, zero-sum Simplex minimax, `8x8` support enumeration, lexicographically perturbed Lemke-Howson), bitmask cooperative Shapley values (`cooperative`), Mean Field Games (`mean_field`), evolutionary matrix dynamics (`evolutionary`), and Myerson/VCG mechanism design (`mechanism_design`).
- **`decision_theory`**: Arena-indexed finite decision trees with backward induction, `CARA`/`CRRA` certainty equivalents, `EVPI`/`EVSI` value-of-information solvers, and deterministic `OxidizeRng`-seeded `UCB1` and Thompson sampling bandits.
- **`lorahub` & `grpo`**: Machine learning parameter merging and group relative policy optimization.
- **`cannibalism`**: Zero-allocation upwind solvers for McKendrick-von Foerster age-structured PDEs (`McKendrickSolver`), Diekmann mortality and Ricker saturation kernels (`CannibalismKernel`), van den Bosch juvenile-adult delay dynamics (`JuvenileAdultSolver`), and lumped 2D ODE models (`CannibalismModel`).
- **`favoritism`**: Socio-mathematical quadrature scoring functions (`UnifiedFavoritismModel`).

---

## 2. Developer Instructions

### Evaluating Right-Censored Composite Outcomes with U-Statistics

Analyze prioritized fatal and non-fatal time-to-event endpoints under right censoring with two-sample U-statistic inference, or evaluate multi-tier continuous/count composite waterfalls:

```rust
# use domain_applied::__doctest_facade as math_explorer;
use math_explorer::applied::win_ratio::pair_comparison::{
    calculate_statistics, CensoredOutcome, HigherIsBetter, LowerIsBetter,
    RightCensoredComparator, UnmatchedPairing, WinRatioAnalysis,
};

// 1. Right-censored hierarchy: [Tier 1: Fatal Event, Tier 2: Non-Fatal Hospitalization]
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

let censored_analysis = WinRatioAnalysis::new()
    .add_strategy(Box::new(RightCensoredComparator::new()))
    .add_strategy(Box::new(RightCensoredComparator::new()));

let (summary, censored_tiers) = censored_analysis
    .evaluate_unmatched(&treatment, &control, 0.05)
    .expect("Valid unmatched U-statistic summary");

assert_eq!(summary.total_pairs, 4);
assert_eq!(censored_tiers[0].wins + censored_tiers[1].wins, 4);
assert!(summary.win_ratio > 1.0);

// 2. Continuous/count waterfall: [Survival Days (HigherIsBetter), Hospitalizations (LowerIsBetter)]
let trt_numeric = vec![vec![900.0, 0.0], vec![300.0, 1.0]];
let ctrl_numeric = vec![vec![300.0, 4.0], vec![200.0, 2.0]];
let waterfall_analysis = WinRatioAnalysis::new()
    .add_strategy(Box::new(HigherIsBetter))
    .add_strategy(Box::new(LowerIsBetter));
let tiers = waterfall_analysis
    .evaluate_tiers(&trt_numeric, &ctrl_numeric, &UnmatchedPairing)
    .expect("Valid tier evaluation");
let (wins, losses) = waterfall_analysis.evaluate_pairs(&trt_numeric, &ctrl_numeric, &UnmatchedPairing);
let statistics = calculate_statistics(wins, losses).expect("Valid statistics");
assert_eq!(tiers[0].wins, 3);
assert_eq!(tiers[1].wins, 1);
assert!(statistics.win_ratio.is_infinite() || statistics.win_ratio > 1.0);
```

### Modeling Multi-Dose ADME Pharmacokinetics

Calibrate oral absorption rates from target peak times and evaluate multi-dose plasma curves across immediate-release, biphasic two-pulse, or chiral enantiomer regimens:

```rust
# use domain_applied::__doctest_facade as math_explorer;
use math_explorer::applied::pharmacokinetics::{
    half_life, solve_ka, t_max, BatemanModel, PKParameters, PharmacokineticModel,
    SuperpositionModel, TwoPulseModel,
};

let ke = std::f64::consts::LN_2 / 8.0;
let ka = solve_ka(2.0, ke, 1.0, 100, 1.0e-7).expect("Convergent absorption rate");
assert!((half_life(ke) - 8.0).abs() < 1.0e-9);
assert!((t_max(ka, ke) - 2.0).abs() < 1.0e-6);

let params = PKParameters::new(0.85, 100.0, ka, ke, 35.0).expect("Valid PK parameters");
let xr_pulse = TwoPulseModel::new(BatemanModel::new(params), 4.0, 0.5, 0.5);
let regimen = SuperpositionModel::new(xr_pulse, vec![0.0, 12.0, 24.0]);

assert!(regimen.concentration(26.0) > regimen.concentration(2.0));
```

### Age-Structured Cannibalism PDE (`McKendrickSolver`)

Advance the McKendrick-von Foerster equation with Diekmann mortality and Ricker prey saturation using pre-allocated upwind buffers:

```rust
# use domain_applied::__doctest_facade as math_explorer;
use math_explorer::applied::cannibalism::{CannibalismKernel, McKendrickSolver};

let kernel = CannibalismKernel::new(0.12, 0.02, 0.35, 3.5, 0.02, 4.0, 3.5, 80.0);
let mut solver = McKendrickSolver::new(64, 10.0, kernel);
solver.set_initial_density(|age| 15.0 * (-0.25 * age).exp());

solver.step_upwind(0.2);

assert!(solver.total_population() > 0.0);
assert!(solver.cannibal_load() >= 0.0);
```

---

## 3. Ecosystem Integration

Algorithms from `domain_applied` power interactive simulation workbenches in `math_explorer_gui`—including `WinRatioWorkbenchTool`, `WinRatioStressTesterTool`, the `Pharmacokinetics (ADME)` multi-dose regimen planner, two-arm `SurvivalAnalysisTool` with log-rank testing and risk tables, seeded stratified `RandomizationTool` with Simpson's paradox labs, and power-curve `SampleSizeCalculatorTool` in `ClinicalTrialsTab`, the `Pharmacokinetic Regimen` planner in `MedicalTab`, and age-structured cannibalism PDE simulations in `PopulationStructureTab`—enabling local scenario exploration and reproducible protocol exports for clinical biostatisticians, pharmacometricians, ecologists, battery engineers, and data scientists.
