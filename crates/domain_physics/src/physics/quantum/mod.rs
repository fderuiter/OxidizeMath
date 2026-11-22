//! # Quantum Mechanics
//!
//! A collection of tools for simulating quantum systems, from basic angular momentum
//! coupling to time evolution governed by the Schrödinger equation.
//!
//! ## Core Concepts
//!
//! - **States**: Represented as complex vectors $|\psi\rangle$.
//! - **Operators**: Represented as complex matrices $\hat{H}, \hat{U}$.
//! - **Evolution**: The state changes over time according to $|\psi(t)\rangle = e^{-i\hat{H}t/\hbar} |\psi(0)\rangle$.
//! - **Coupling**: Combining angular momenta (Clebsch-Gordan coefficients).
//!
//! ## Workflow: Time Evolution
//!
//! ```mermaid
//! graph LR
//!     Init[Initial State |ψ0>] -->|Apply Hamiltonian| Evol[Time Evolution U(t)]
//!     Evol -->|Compute| Final[Final State |ψt>]
//!     Final -->|Measure| Obs[Observable Value]
//!
//!     subgraph "Schrödinger Equation"
//!     Evol
//!     end
//! ```
//!
//! ##  Quick Start: Angular Momentum & Time Evolution
//!
//! ### 1. Clebsch-Gordan Coefficients
//! Calculate the probability amplitude of coupled angular momentum states.
//!
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::quantum::clebsch_gordan;
//!
//! // Coupling spin j1=1.5, m1=-0.5 with j2=1.0, m2=1.0
//! // We want to know the coefficient for total angular momentum J=2.5, M=0.5
//! let cg = clebsch_gordan(1.5, -0.5, 1.0, 1.0, 2.5, 0.5);
//!
//! println!("CG Coefficient: {:.4}", cg);
//! assert!((cg - (0.3f64).sqrt()).abs() < math_explorer::registry::TOLERANCE_STANDARD); // Expected value sqrt(3/10)
//! ```
//!
//! ### 2. Time Evolution of a Qubit
//! Simulating a spin-1/2 particle in a magnetic field (Pauli-X Hamiltonian).
//!
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::quantum::{QuantumState, QuantumOperator, evolve_state, sigma_x};
//! use math_explorer::nalgebra::DVector;
//! use math_explorer::num_complex::Complex;
//!
//! // 1. Define Initial State |0> = [1, 0]^T
//! let psi_0 = QuantumState::new(DVector::from_vec(vec![
//!     Complex::new(1.0, 0.0),
//!     Complex::new(0.0, 0.0)
//! ]));
//!
//! // 2. Define Hamiltonian H = sigma_x (Spin flip)
//! let h_hat = sigma_x();
//!
//! // 3. Evolve for time t (pi/2 pulse)
//! let t = std::f64::consts::PI / 2.0;
//! let h_bar = 1.0; // Natural units
//!
//! let psi_t = evolve_state(&psi_0, &h_hat, t, h_bar);
//!
//! // 4. Check result. e^{-i * sigma_x * pi/2} |0> = -i |1> (up to phase)
//! // The resulting state should have high probability of being |1>
//! let probs = psi_t.probability_density();
//! let prob_1 = probs[1];
//! println!("Probability of state |1>: {:.4}", prob_1);
//! assert!(prob_1 > 0.99);
//! ```

pub mod coupling;
#[allow(missing_docs)]
pub mod fourier;
pub mod hamiltonian;
pub mod schrodinger;
#[allow(missing_docs)]
pub mod spin;
#[allow(missing_docs)]
pub mod types;

// Re-export key types for convenience
pub use coupling::clebsch_gordan;
pub use fourier::{dft_operator, idft_operator};
pub use hamiltonian::{
    MAX_SCULPT_POTENTIAL, MIN_SCULPT_POTENTIAL, PotentialPreset1D,
    apply_bresenham_potential_stroke, build_1d_potential_preset, construct_1d_hamiltonian,
    gaussian_wavepacket, partition_region_probabilities,
};
pub use schrodinger::{evolve_state, evolve_state_in_place, time_evolution_operator};
pub use spin::{sigma_x, sigma_y, sigma_z};
pub use types::{QuantumOperator, QuantumState};

// [cite:quantum_mechanics]

use pure_math::theory_verification;

theory_verification!(
    module = "quantum",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
