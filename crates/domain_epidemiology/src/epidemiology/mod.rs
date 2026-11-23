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

#[allow(missing_docs)]
pub mod analytics;
#[allow(missing_docs)]
pub mod compartmental;
#[allow(missing_docs)]
pub mod matrix_dynamics;
#[allow(missing_docs)]
pub mod networks;
#[allow(missing_docs)]
pub mod stochastic;

// Re-exports for easier access
pub use compartmental::{SEIRModel, SEIRState, SIRModel, SIRState};
pub use networks::NetworkModelError;

/// Error types for epidemiology modules.
pub use crate::error;

// [cite:graph_parameters_rust]

use pure_math::theory_verification;
theory_verification!(module = "epidemiology", constants = {}, test = {});
