//! # Clinical Trials
//!
//! A comprehensive framework for **Evidence-Based Medicine** research design and analysis.
//!
//! This module provides tools for the entire lifecycle of a clinical trial:
//! 1.  **Design**: Randomization strategies (Simple, Block, Stratified).
//! 2.  **Planning**: Sample size and power calculations to ensure statistical validity.
//! 3.  **Analysis**: Hypothesis testing (T-tests, Chi-Square), Risk Metrics (RR, OR), and Survival Analysis (Kaplan-Meier).
//!
//! ## Workflow
//!
//! ```mermaid
//! graph LR
//!     Start([Start]) --> Plan[Sample Size Calculation]
//!     Plan --> Design[Randomization]
//!     Design --> Conduct[Conduct Trial]
//!     Conduct --> Analyze[Statistical Analysis]
//!     Analyze --> Report([Report Results])
//!
//!     subgraph Planning
//!     Plan
//!     end
//!
//!     subgraph Execution
//!     Design
//!     Conduct
//!     end
//!
//!     subgraph Analysis
//!     Analyze
//!     end
//! ```
//!
//! ## Example: Full Trial Simulation
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::clinical_trials::design::{SimpleRandomizer, AllocationStrategy, Group};
//! use math_explorer::rng::OxidizeRng;
//! use math_explorer::applied::clinical_trials::sample_size::calculate_sample_size_means;
//! use math_explorer::applied::clinical_trials::analysis::calculate_risk_metrics;
//! use math_explorer::applied::clinical_trials::types::ContingencyTable;
//!
//! fn main() {
//!     // 1. Plan: We want to detect a mean difference of 5.0 with SD=10.0
//!     // Power=80%, Alpha=0.05
//!     let n_per_group = calculate_sample_size_means(0.05, 0.80, 5.0, 10.0).unwrap();
//!     println!("Required sample size per group: {}", n_per_group);
//!
//!     // 2. Design: Randomize patients
//!     // We need 2 * n_per_group total patients
//!     let total_patients = n_per_group * 2;
//!     let mut rng = OxidizeRng::default();
//!     let assignments = SimpleRandomizer.assign(&mut rng, total_patients).unwrap();
//!
//!     // ... Conduct trial ...
//!
//!     // 3. Analyze: Calculate Relative Risk (RR)
//!     // Scenario: Treatment group had fewer adverse events (20 vs 40) out of 100 each.
//!     let table = ContingencyTable::new(
//!         20, // Treatment: Event
//!         80, // Treatment: No Event
//!         40, // Control: Event
//!         60, // Control: No Event
//!     ).unwrap();
//!     let metrics = calculate_risk_metrics(&table, 0.05).unwrap();
//!
//!     println!("Relative Risk: {:.2}", metrics.relative_risk); // Should be 0.5
//!     println!("Odds Ratio: {:.2}", metrics.odds_ratio);
//! }
//! ```

/// Relative risk and odds ratio estimators with log-Wald confidence intervals.
pub mod analysis;
/// Simple, permuted-block, and covariate-stratified randomization strategies.
pub mod design;
/// Pooled two-sample Student's $t$-tests and $2 \times 2$ Pearson $\chi^2$ tests.
pub mod hypothesis_testing;
/// Power and minimum per-arm sample-size formulas for means and proportions.
pub mod sample_size;
/// Non-parametric Kaplan-Meier survival curves and Greenwood standard errors.
pub mod survival_analysis;
pub mod types;

pub use types::ClinicalTrialError;

// [cite:clinical_trials_statistics]

use pure_math::theory_verification;

theory_verification!(
    module = "clinical_trials",
    constants = {
        ALPHA = 0.05;
    },
    test = {
        use approx::assert_relative_eq;
        // Just verify it compiles and runs.
        assert_relative_eq!(
            ALPHA,
            0.05,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);

pure_math::property_invariant_verification!(
    module = "clinical_trials",
    test = {
        let obs = [survival_analysis::Observation {
            time: types::SurvivalTime::new(2.0).unwrap(),
            event_occurred: false,
        }];
        let curve = survival_analysis::kaplan_meier(&obs);
        assert_eq!(curve.len(), 1);
        assert!((curve[0].survival_probability - 1.0).abs() < 1e-15);
    }
);

pure_math::formal_proof_verification!(
    module = "clinical_trials",
    proof = "SurvivalBounds.lean",
    test = {
        let obs = [survival_analysis::Observation {
            time: types::SurvivalTime::new(3.0).unwrap(),
            event_occurred: true,
        }];
        let curve = survival_analysis::kaplan_meier(&obs);
        assert!(curve[0].survival_probability >= 0.0 && curve[0].survival_probability <= 1.0);
    }
);
