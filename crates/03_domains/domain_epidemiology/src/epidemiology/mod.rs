//! Epidemiology module for modeling disease spread.
//!
//! This module covers:
//! 1. **Deterministic Compartmental Models**: Standard SIR and SEIR equations.
//! 2. **Analytical Solutions**: Final Size equations and R0 calculations.
//! 3. **Stochastic Dynamics**: Gillespie algorithms for small populations.
//! 4. **Network Epidemiology**: Disease spread on graph structures.
//!
//! ##  Quick Start: Simulating an Outbreak
//!
//! ```rust
//! # use domain_epidemiology::__doctest_facade as math_explorer;
//! use math_explorer::epidemiology::{SIRModel, SIRState};
//!
//! // 1. Initialize the model
//! // Population: 1000, Initial Infected: 10
//! // Beta (Infection Rate): 0.5, Gamma (Recovery Rate): 0.1
//! let n = 1000.0;
//! let i0 = 10.0;
//! let beta = 0.5;
//! let gamma = 0.1;
//! let mut model = SIRModel::new(n, i0, beta, gamma).expect("Valid parameters");
//!
//! // 2. Run the simulation
//! let dt = 0.1;
//! let mut peak_infected = 0.0;
//!
//! for _ in 0..1000 {
//!     model.step(dt);
//!     if model.state().i > peak_infected {
//!         peak_infected = model.state().i;
//!     }
//! }
//!
//! println!("Peak infected individuals: {:.0}", peak_infected);
//! assert!(peak_infected > i0);
//! ```
//!
//! ## Compartmental Models
//!
//! We support both standard SIR and SEIR models using deterministic ODEs.
//!
//! ```mermaid
//! graph LR
//!     S[Susceptible] -->|βSI/N| I[Infected]
//!     I -->|γI| R[Recovered]
//!
//!     style S fill:#dfd,stroke:#333
//!     style I fill:#fdd,stroke:#333
//!     style R fill:#ddf,stroke:#333
//! ```
//!
//! For models with an incubation period (SEIR):
//!
//! ```mermaid
//! graph LR
//!     S[Susceptible] -->|βSI/N| E[Exposed]
//!     E -->|σE| I[Infected]
//!     I -->|γI| R[Recovered]
//! ```
//!
//! # Mathematical Background
//!
//! The **Threshold Theorem** states that an epidemic occurs if and only if the basic reproduction
//! number $R_0 > 1$.
//!
//! $R_0$ is defined as the expected number of secondary infections produced by a single infected
//! individual in a completely susceptible population.

/// Implicit epidemic final-size solvers via Newton-Raphson iteration.
pub mod analytics;
/// Deterministic SIR and SEIR compartmental ODE systems and builders.
pub mod compartmental;
/// Next-generation matrix $K = F V^{-1}$ spectral-radius $R_0$ computation.
pub mod matrix_dynamics;
/// Discrete-time stochastic contagion and quarantine dynamics on contact graphs.
pub mod networks;
/// Continuous-time stochastic SIR Gillespie propensities and branching extinction probabilities.
pub mod stochastic;

// Re-exports for easier access
pub use compartmental::{SEIRModel, SEIRState, SIRModel, SIRState};
pub use networks::NetworkModelError;

/// Error types for epidemiology modules.
pub use crate::error;

// [cite:graph_parameters_rust]

use pure_math::theory_verification;
theory_verification!(
    module = "epidemiology",
    constants = {
        POPULATION_N = 1000.0;
        INITIAL_I0 = 10.0;
        BETA = 0.5;
        GAMMA = 0.1;
    },
    test = {
        let mut model =
            SIRModel::new(POPULATION_N, INITIAL_I0, BETA, GAMMA).expect("valid SIR parameters");
        let r0 = compartmental::basic_reproduction_number(BETA, GAMMA);
        assert!((r0 - 5.0).abs() < 1e-12);
        for _ in 0..100 {
            model.step(0.05);
        }
        let st = model.state();
        assert!((st.s + st.i + st.r - POPULATION_N).abs() < 1e-9);
        assert!(st.i > INITIAL_I0);
        assert!(st.r > 0.0);
        assert!((model.time() - 5.0).abs() < 1e-12);
    }
);

pure_math::mms_convergence_verification!(
    module = "epidemiology",
    test = {
        let mut model = SIRModel::new(1000.0, 20.0, 0.3, 0.1).unwrap();
        model.step(0.25);
        assert!((model.time() - 0.25).abs() < 1e-15);
    }
);

pure_math::formal_proof_verification!(
    module = "epidemiology",
    proof = "CompartmentalConservation.lean",
    theorems = [
        "forced_seir_exact_conservation",
        "forced_seir_rounding_drift_bound",
        "forced_seir_nonneg",
    ],
    test = {
        let mut model = SEIRModel::new(1000.0, 15.0, 0.4, 0.2, 0.1).unwrap();
        model.step(0.1);
        let st = model.state();
        assert!((st.s + st.e + st.i + st.r - 1000.0).abs() < 1e-9);
    }
);
