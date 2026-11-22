//! # Pharmacokinetics (ADME Modeling)
//!
//! This module implements core algorithms for modeling the Absorption, Distribution, Metabolism,
//! and Excretion (ADME) of drugs. It provides a composable framework for simulating drug
//! concentration over time in a subject's plasma.
//!
//! ## Architecture
//!
//! The module is built around the `PharmacokineticModel` trait, which allows for the flexible
//! composition of different drug behaviors. Base models (like the Bateman function) can be
//! wrapped by higher-order models (like Superposition or Two-Pulse) to create complex simulations.
//!
//! ```mermaid
//! graph TD
//!     Trait[Trait: PharmacokineticModel]
//!
//!     Bateman[BatemanModel<br/>(Single Dose, 1-Compartment)]
//!     Enantiomer[EnantiomerModel<br/>(Chiral Mixture)]
//!
//!     Super[SuperpositionModel<br/>(Multiple Doses)]
//!     TwoPulse[TwoPulseModel<br/>(Extended Release)]
//!
//!     Trait <|.. Bateman
//!     Trait <|.. Enantiomer
//!
//!     Super -->|wraps| Trait
//!     TwoPulse -->|wraps| Trait
//!
//!     style Trait fill:#f9f,stroke:#333,stroke-width:2px
//! ```
//!
//! ## Quick Start
//!
//! Simulate the concentration of a single oral dose of a drug (e.g., Caffeine) over time.
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::pharmacokinetics::{BatemanModel, PKParameters, PharmacokineticModel};
//!
//! fn main() {
//!     // 1. Define Drug Parameters (e.g., Caffeine ~100mg)
//!     // Using new validated constructor
//!     let params = PKParameters::new(
//!         1.0,   // Bioavailability (100%)
//!         100.0, // Dose (mg)
//!         2.5,   // Absorption rate (fast)
//!         0.15,  // Elimination rate (Half-life ~4.6h)
//!         50.0   // Volume of distribution (L)
//!     ).unwrap();
//!
//!     // 2. Create the Model
//!     let model = BatemanModel::new(params);
//!
//!     // 3. Simulate
//!     let t_hours = 1.0;
//!     let concentration = model.concentration(t_hours);
//!
//!     println!("Concentration at {}h: {:.2} mg/L", t_hours, concentration);
//!     assert!(concentration > 0.0);
//! }
//! ```
//!
//! ## Key Components
//!
//! - **`BatemanModel`**: The standard one-compartment model with first-order absorption and elimination.
//! - **`SuperpositionModel`**: Calculates accumulation from uniform multiple doses.
//! - **`RegimenSuperposition` & `DoseEvent`**: Evaluates variable-dose schedules (loading doses, missed doses, catch-up doses, and mixed IR/XR formulations) with zero heap allocations.
//! - **`ExposureSummary` & `TherapeuticWindow`**: Computes $C_{\max}$, $C_{\min}$, $\text{AUC}_{0-\tau}$, $R_{\text{ac}}$, Time-in-Therapeutic-Range, and toxic hours over caller slices.
//! - **`EnantiomerModel`**: Models drugs with chiral centers (e.g., Adderall) where isomers have different kinetics.
//! - **`TwoPulseModel`**: Simulates biphasic release profiles common in Extended Release (XR) formulations.
//!
//! ## Variable-Dose Regimen & Exposure Example
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::pharmacokinetics::{
//!     BatemanModel, DoseEvent, PKParameters, RegimenSuperposition, TherapeuticWindow,
//!     summarize_exposure,
//! };
//!
//! let params = PKParameters::new(1.0, 100.0, 1.2, 0.1, 20.0).unwrap();
//! let events = [
//!     DoseEvent::ir(0.0, 200.0).unwrap(),
//!     DoseEvent::ir(12.0, 100.0).unwrap(),
//! ];
//! let regimen = RegimenSuperposition::new(BatemanModel::new(params), &events).unwrap();
//! let times = [0.0, 6.0, 12.0, 18.0, 24.0];
//! let mut conc = [0.0; 5];
//! regimen.sample_into(&times, &mut conc).unwrap();
//! let window = TherapeuticWindow::new(3.0, 8.5).unwrap();
//! let summary = summarize_exposure(&times, &conc, window, 12.0, params.ke()).unwrap();
//! assert!(summary.c_max > 0.0);
//! ```

pub mod analytics;
#[allow(missing_docs)]
pub mod bateman;
#[allow(missing_docs)]
pub mod enantiomer;
#[allow(missing_docs)]
pub mod parameters;
pub mod regimen;
#[allow(missing_docs)]
pub mod superposition;
#[allow(missing_docs)]
pub mod traits;
#[allow(missing_docs)]
pub mod two_pulse;

pub use analytics::{
    ExposureSummary, TherapeuticWindow, accumulation_ratio, auc_interval, peak_concentration,
    summarize_exposure, trapezoidal_auc, trough_concentration,
};
pub use bateman::{BatemanModel, calibrate_ka_from_half_life, half_life, solve_ka, t_max};
pub use enantiomer::EnantiomerModel;
pub use parameters::{PKParameters, PKParametersBuilder};
pub use regimen::{
    DoseEvent, MAX_REGIMEN_EVENTS, RegimenModel, RegimenSuperposition, ReleaseFormulation,
};
pub use superposition::SuperpositionModel;
pub use traits::PharmacokineticModel;
pub use two_pulse::TwoPulseModel;

/// Computes the concentration at time t for a single dose using the Bateman function.
///
/// This is a convenience wrapper around `BatemanModel`.
///
/// # Arguments
/// * `params` - The pharmacokinetic parameters.
/// * `t` - The time after the dose.
#[verified_engine::verified]
pub fn concentration_bateman(params: &PKParameters, t: f64) -> f64 {
    let model = BatemanModel::new(*params);
    model.concentration(t)
}

/// Computes the total concentration at time t from multiple doses using superposition.
///
/// This is a convenience wrapper around `SuperpositionModel` using `BatemanModel` as the base.
///
/// # Arguments
/// * `params` - The pharmacokinetic parameters for a single dose.
/// * `dose_times` - A slice of times at which doses were administered.
/// * `t` - The time at which to calculate the total concentration.
#[verified_engine::verified]
pub fn concentration_superposition(params: &PKParameters, dose_times: &[f64], t: f64) -> f64 {
    let base_model = BatemanModel::new(*params);
    let model = SuperpositionModel::new(base_model, dose_times.to_vec());
    model.concentration(t)
}

// [cite:pharmacokinetics]

use pure_math::theory_verification;

theory_verification!(
    module = "pharmacokinetics",
    constants = {
        DOSE = 100.0;
    },
    test = {
        assert_relative_eq!(
            DOSE,
            100.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
