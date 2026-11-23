# Domain Applied

Applied mathematical modeling for clinical trials, engineering systems, game theory, and surface extraction.

## 1. Overview & Architecture

The `domain_applied` crate translates theoretical mathematics—including calculus, probability, differential geometry, and optimization—into practical models for complex domains.

The crate spans life sciences, material engineering, distributed systems, and quantitative decision theory, encapsulating specialized domain algorithms inside clean and modular Rust interfaces.

### Module Structure

- **`win_ratio`**: Prioritized composite endpoint, right-censored U-statistic, and bivariate Weibull probability win ratio evaluation for clinical research:
  - `WinRatioAnalysis` & `WinRatioStats`: Configurable hierarchical comparison engine (`HigherIsBetter`, `LowerIsBetter`, `ThresholdComparator`) across `UnmatchedPairing` (`evaluate_unmatched`, `evaluate_unmatched_into`) and `MatchedPairing` cohorts with 95% confidence intervals and p-values (`calculate_statistics`).
  - `censoring` & `u_statistic`: Right-censored time-to-event outcomes (`CensoredOutcome`), Pocock/Finkelstein-Schoenfeld comparators (`RightCensoredComparator`), hierarchical tier waterfall counters (`TierCount`), and Pocock/Bebu-Lachin two-sample U-statistic variance projection (`UnmatchedWinRatioSummary`, `UnmatchedScratch`, `compute_unmatched_u_statistics`) calibrating confidence intervals and Type I error across dependent all-pairs comparisons.
  - `sample_win_ratio`: Pocock et al. (2012) two-tier matched-pair `WinLossCounts` (`calculate_sample_win_ratio`, `calculate_win_proportion`, `calculate_confidence_interval`, `calculate_significance_test_statistic`).
  - `probability_win_ratio`, `simulation`, `weibull_sampler`, & `power_sweep`: Parametric `BivariateWeibullModel` (`SimulationParams`, `win_ratio_parameter`), Clenshaw-Curtis numerical integration (`ProbabilityWinRatioContext`, `calculate_probability_win_ratio`) over censoring horizons $c$, exact inverse-CDF sampling via `OxidizeRng`, and zero-allocation Monte Carlo power sweeps (`run_power_sweep_into`, `run_weibull_power_sweep`).
  - `bmi`: Anthropometric `calculate_bmi` covariate derivation ($\text{kg}/\text{m}^2$).
- **`clinical_trials`**: Biostatistical trial design (`SimpleRandomizer`, `BlockRandomizer`, `StratifiedRandomizer`), Neyman-Pearson sample size and analytical power curves (`calculate_sample_size_means`, `calculate_sample_size_proportions`, `calculate_power_means`, `calculate_power_proportions`), hypothesis testing (`TTestIndependent`, `ChiSquare2x2`), binary risk metrics (`calculate_risk_metrics`), and right-censored survival analysis (`kaplan_meier`, `try_estimate_hazard_ratio`).
- **`pharmacokinetics`**: One-compartment oral Bateman ADME kinetics (`PKParameters`, `BatemanModel`, `half_life`, `t_max`, `solve_ka`, `calibrate_ka_from_half_life`), uniform (`SuperpositionModel`) and variable per-event multi-dose superposition (`DoseEvent`, `RegimenSuperposition`), biphasic extended-release formulations (`TwoPulseModel`), stereoselective chiral mixtures (`EnantiomerModel`), and zero-allocation therapeutic-window exposure analytics (`TherapeuticWindow`, `ExposureSummary`).
- **`battery_degradation`**: Lithium-ion capacity fade and solid electrolyte interphase (SEI) growth models.
- **`isosurface`**: 3D volumetric surface extraction algorithms, including Marching Cubes and dual contouring.
- **`freesurfer`**: Triangulated cortical surface reconstruction (`Surface`, `MriVolume`, thin-plate `internal_energy`, Gaussian-smoothed `external_energy_volume`, `evolve_surface_deformable`), 6-connected Potts MRF Bayesian segmentation (`TissueClass`, `normalized_posteriors`, `mrf_segment_volume`), uniform-grid cortical thickness maps (`cortical_thickness`, `cortical_thickness_map`), and vertex-wise OLS GLM contrasts (`estimate_beta`, `t_statistic`).
- **`game_theory`**: Mean Field Games, evolutionary matrix dynamics, and mechanism design.
- **`lorahub` & `grpo`**: Machine learning parameter merging and group relative policy optimization.
- **`cannibalism`**: Zero-allocation upwind solvers for McKendrick-von Foerster age-structured PDEs (`McKendrickSolver`), Diekmann mortality and Ricker saturation kernels (`CannibalismKernel`), van den Bosch juvenile-adult delay dynamics (`JuvenileAdultSolver`), and lumped 2D ODE models (`CannibalismModel`).
- **`favoritism`**: Socio-mathematical quadrature scoring functions (`UnifiedFavoritismModel`).

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

### Evaluating Clinical Outcomes with Threshold Hierarchies & Bivariate Weibull `PR(c)`

Analyze hierarchical composite endpoints across fatal, hospitalization, and thresholded continuous biomarker tiers alongside Clenshaw-Curtis `PR(c)` integrals:

```rust
# use domain_applied::__doctest_facade as math_explorer;
use math_explorer::applied::win_ratio::{
    bmi::calculate_bmi,
    pair_comparison::{
        calculate_statistics, HigherIsBetter, LowerIsBetter, ThresholdComparator,
        UnmatchedPairing, WinRatioAnalysis,
    },
    probability_win_ratio::{calculate_probability_win_ratio, ProbabilityWinRatioContext},
    simulation::{
        conditional_survival_x_given_t_control, conditional_survival_x_given_t_treatment,
        marginal_survival_t_control, marginal_survival_t_treatment, pdf_t_control,
        pdf_t_treatment, pdf_x_given_t_control, pdf_x_given_t_treatment, win_ratio_parameter,
        SimulationParams,
    },
};

// Derive baseline BMI covariate and define [Survival Days, Hospitalizations, Biomarker]
let bmi = calculate_bmi(74.0, 1.75);
assert!(bmi > 24.0 && bmi < 25.0);

let treatment_patients = vec![vec![900.0, 0.0, 78.0], vec![860.0, 1.0, 72.0]];
let control_patients = vec![vec![320.0, 3.0, 55.0], vec![900.0, 2.0, 60.0]];

// Configure 3-tier hierarchy: Survival (higher), Hospitalizations (lower), Biomarker (threshold >= 5.0)
let analysis = WinRatioAnalysis::new()
    .add_strategy(Box::new(HigherIsBetter))
    .add_strategy(Box::new(LowerIsBetter))
    .add_strategy(Box::new(ThresholdComparator { threshold: 5.0 }));

let (wins, losses) =
    analysis.evaluate_pairs(&treatment_patients, &control_patients, &UnmatchedPairing);
let statistics = calculate_statistics(wins, losses).expect("Valid statistics");
assert_eq!((wins, losses), (3, 1));
assert!((statistics.win_ratio - 3.0).abs() < 1.0e-12);

// Evaluate Bivariate Weibull probability win ratio PR(c) at follow-up horizon c = 10.0
let params = SimulationParams::new(0.10, 0.20, 0.80, 0.70);
let c = 10.0;
let pr_ctx = ProbabilityWinRatioContext::new(
    marginal_survival_t_control(c, &params),
    marginal_survival_t_treatment(c, &params),
    c,
    1.0e-6,
);
let w_c = pr_ctx.calculate_win_probability(
    |t| marginal_survival_t_treatment(t, &params),
    |t| pdf_t_control(t, &params),
    |x| conditional_survival_x_given_t_treatment(x, c, &params),
    |x| pdf_x_given_t_control(x, c, &params),
);
let l_c = pr_ctx.calculate_loss_probability(
    |t| marginal_survival_t_control(t, &params),
    |t| pdf_t_treatment(t, &params),
    |x| conditional_survival_x_given_t_control(x, c, &params),
    |x| pdf_x_given_t_treatment(x, c, &params),
);
let pr_c = calculate_probability_win_ratio(w_c, l_c);
assert!(pr_c > 1.0 && win_ratio_parameter(&params) > 1.0);
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

Algorithms from `domain_applied` feed interactive, HIPAA-safe client-side simulation workbenches in `math_explorer_gui`—enabling visual scenario exploration and reproducible protocol exports for clinical biostatisticians, pharmacometricians, ecologists, neuroscientists, battery engineers, and data scientists:

- **`Composite Win Ratio` (`WinRatioTool` in `ClinicalTrialsTab`)**: Evaluates prioritized multi-tier `WinRatioAnalysis` (`HigherIsBetter`, `LowerIsBetter`, `ThresholdComparator`) across `UnmatchedPairing` and `MatchedPairing`, Pocock `WinLossCounts`, secondary continuous `TTestIndependent` (`GroupData`), `calculate_bmi`, and Clenshaw-Curtis `ProbabilityWinRatioContext` `BivariateWeibullModel` `PR(c)` curves with local CSV/TSV cohort import and `.md`/`.csv` report downloads.
- **`Composite Win Ratio Stress Tester` (`WinRatioStressTesterTool` in `ClinicalTrialsTab`)**: Right-censored fatal/non-fatal U-statistic inference, hierarchical tier waterfalls, Monte Carlo bivariate Weibull Power-by-$N$ sweeps, and `.csv`/`.json` protocol exports.
- **`Survival Curves (Kaplan-Meier)` (`SurvivalAnalysisTool` in `ClinicalTrialsTab`)**: Multi-arm CSV/TSV right-censored `kaplan_meier` estimation, `try_estimate_hazard_ratio`, `TimePoint` risk-set tables, and `.csv`/`.md`/`.svg` exports.
- **`Sample Size Calculator` (`SampleSizeCalculatorTool` in `ClinicalTrialsTab`)**: Interactive $1 - \beta$ vs $N$ analytical statistical power curves for `Means` and `Proportions` with `.csv`/`.md` protocol sizing cards.
- **`Randomization` (`RandomizationTool` & `SimpsonParadoxLab` in `ClinicalTrialsTab`)**: Retained per-patient allocation schedules (`Simple`, `Block`, and `Stratified`), cumulative arm-balance plots, and `.csv`/`.json` schedule exports.
- **`Pharmacokinetics (ADME)` (`PharmacokineticsTool` in `ClinicalTrialsTab`) & `Pharmacokinetic Regimen` (`PharmacokineticRegimenTool` in `MedicalTab`)**: Multi-dose immediate-release, variable per-event superposition (`RegimenSuperposition`), biphasic extended-release, chiral ADME regimen planning, and therapeutic-window exposure analytics (`TherapeuticWindow`, `ExposureSummary`).
- **`3D Cortical Morphometry` (`CorticalMorphometryTool` in `NeuroscienceTab`) & `Population Structure` (`PopulationStructureTab`)**: FreeSurfer deformable cortical surface physics / Potts MRF segmentation and age-structured cannibalism PDE simulations (`McKendrickSolver`, `CannibalismKernel`).
