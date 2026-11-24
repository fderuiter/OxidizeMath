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
//! - **Molecular**: [`kinetics`] - Michaelis-Menten and cooperative Hill enzyme kinetics.
//! - **Cellular**: [`neuroscience`] - Hodgkin-Huxley model for action potentials.
//! - **Tissue**: [`morphogenesis`] - Turing and Gray-Scott reaction-diffusion systems for pattern formation.
//! - **Botanical**: [`lsystem`] and [`phyllotaxis`] - Zero-allocation L-system vascular grammars and Vogel golden-angle phyllotaxis.
//! - **Artificial Life**: [`lenia`] - Spectral 2D Lenia continuous cellular automata and soliton lifeforms.
//! - **Population**: [`evolution`] - Evolutionary game theory (Hawk-Dove).
//! - **Generic**: [`reaction_diffusion`] - Generic N-species reaction-diffusion framework.

/// Michaelis-Menten and Hill enzyme kinetics.
pub mod kinetics;

/// Hodgkin-Huxley neuron model.
pub mod neuroscience;

/// Turing and Gray-Scott reaction-diffusion systems.
pub mod morphogenesis;

/// Continuous 2D Lenia spectral cellular automata.
pub mod lenia;

/// Deterministic D0L and parametric L-system vascular grammars.
pub mod lsystem;

/// Vogel planar phyllotaxis and Fibonacci parastichy analysis.
pub mod phyllotaxis;

/// Spatial diffusion strategies.
pub mod diffusion;

/// Evolutionary game theory models.
pub mod evolution;

/// Generic reaction-diffusion framework.
pub mod reaction_diffusion;

/// Error types for computational biology modules.
pub use crate::error;
pub use crate::error::BotanicalError;
pub use kinetics::KineticsError;
pub use reaction_diffusion::ReactionDiffusionError;

// [cite:biology]

use pure_math::theory_verification;
theory_verification!(module = "biology", constants = {}, test = {});
pure_math::mms_convergence_verification!(
    module = "biology",
    test = {
        let mut sys = morphogenesis::TuringSystem::new(
            math_commons::math_kernel::types::Dimension(8),
            morphogenesis::DiffusionCoeff(0.16),
            morphogenesis::DiffusionCoeff(0.08),
            math_commons::math_kernel::types::StepSize(1.0),
        );
        sys.step(0.1);
        assert!((sys.time() - 0.1).abs() < 1e-12);
    }
);
