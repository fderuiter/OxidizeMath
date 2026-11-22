//! # Computational Biology
//!
//! This module applies mathematical modeling to biological systems, spanning from
//! molecular interactions to population dynamics.
//!
//! ##  Quick Start: Neural Dynamics
//!
//! Simulate a single neuron firing an action potential using the Hodgkin-Huxley model.
//!
//! ```rust
//! # use domain_biology::__doctest_facade as math_explorer;
//! use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
//!
//! // 1. Initialize neuron at resting potential (-65.0 mV)
//! let mut neuron = HodgkinHuxleyNeuron::new(-65.0);
//!
//! // 2. Apply a current injection to trigger a spike
//! let dt = 0.01;            // 0.01 ms time step
//! let current_inj = 20.0;   // strong current injection
//!
//! // 3. Run simulation
//! let mut spiked = false;
//! for _ in 0..500 {
//!     neuron.update(dt, current_inj);
//!     if neuron.v() > 0.0 {
//!         spiked = true;
//!         break;
//!     }
//! }
//!
//! assert!(spiked, "Neuron should have spiked given sufficient current");
//! println!("Peak Voltage: {:.2} mV", neuron.v());
//! ```
//!
//! ## Hierarchy of Scales
//!
//! The module is organized by biological scale, implementing classic models at each level.
//!
//! ```mermaid
//! graph TD
//!     Pop[Population Scale] -->|Evolutionary Dynamics| Evolution[evolution]
//!     Tissue[Tissue Scale] --> Morph[morphogenesis]
//!     Cell[Cellular Scale] -->|Action Potentials| Neuro[neuroscience]
//!     Mol[Molecular Scale] -->|Enzyme Kinetics| Kinetics[kinetics]
//!     Generic[Generic Framework] -->|Separation of Concerns| RD[reaction_diffusion]
//!     RD --> Morph
//!
//!     style Pop fill:#ff9,stroke:#333,stroke-width:2px
//!     style Tissue fill:#bbf,stroke:#333,stroke-width:2px
//!     style Cell fill:#dfd,stroke:#333,stroke-width:2px
//!     style Mol fill:#fdd,stroke:#333,stroke-width:2px
//!     style Generic fill:#ddd,stroke:#333,stroke-width:2px
//! ```
//!
//! ## Submodules
//!
//! - **Molecular**: [`kinetics`] - Michaelis-Menten and Hill cooperative enzyme kinetics.
//! - **Cellular**: [`neuroscience`] - Hodgkin-Huxley model for action potentials.
//! - **Artificial Life**: [`lenia`] - Spectral continuous cellular automata (`Orbium`, `Hydrogeminium`).
//! - **Botanical**: [`phyllotaxis`] - Vogel golden-angle packing, Fibonacci parastichies, and nutrient-coupled L-systems.
//! - **Tissue**: [`morphogenesis`] - Turing reaction-diffusion systems (`SchnakenbergKinetics`, `GrayScottKinetics`).
//! - **Spatial**: [`diffusion`] - 1D and 2D finite-difference Laplacian operators.
//! - **Population**: [`evolution`] - Evolutionary game theory (`HawkDovePopulation`).
//! - **Generic**: [`reaction_diffusion`] - Generic N-species reaction-diffusion framework.

/// Michaelis-Menten and Hill enzyme kinetics.
pub mod kinetics;

/// Hodgkin-Huxley neuron model.
pub mod neuroscience;

/// Turing reaction-diffusion systems.
pub mod morphogenesis;

/// Spatial diffusion strategies.
pub mod diffusion;

/// Evolutionary game theory models.
pub mod evolution;

/// Generic reaction-diffusion framework.
pub mod reaction_diffusion;

/// Spectral continuous cellular automata (Lenia).
pub mod lenia;

/// Vogel golden-angle phyllotaxis and nutrient-coupled L-systems.
pub mod phyllotaxis;

/// Error types for computational biology modules.
pub use crate::error;

// [cite:biology]

use pure_math::theory_verification;
theory_verification!(module = "biology", constants = {}, test = {});
