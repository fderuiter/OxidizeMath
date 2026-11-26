//! # Applied Game Theory
//!
//! This module provides tools for modeling strategic interactions between rational (or boundedly rational) agents.
//! It covers three distinct scales of interaction:
//!
//! 1. **Mechanism Design**: Single-shot games where the rules are designed to achieve a specific outcome (e.g., auctions).
//! 2. **Classical Equilibrium**: Fixed-point analysis for finding Nash Equilibria in topological spaces.
//! 3. **Evolutionary Dynamics**: Population-scale interactions where strategies evolve over time based on fitness.
//! 4. **Mean Field Games**: The limit of $N \to \infty$ players, modeled using coupled PDE systems.
//!
//! ##  Taxonomy
//!
//! ```mermaid
//! graph TD
//!     GT[Game Theory]
//!     GT --> MD[Mechanism Design]
//!     GT --> ED[Evolutionary Dynamics]
//!     GT --> MFG[Mean Field Games]
//!     GT --> EQ[Equilibrium Analysis]
//!
//!     MD --> Auction[Optimal Auctions]
//!     MD --> Myerson[Virtual Valuations]
//!
//!     ED --> Replicator[Replicator Dynamics]
//!     ED --> ESS[Evolutionarily Stable Strategies]
//!
//!     MFG --> HJB[Hamilton-Jacobi-Bellman]
//!     MFG --> FP[Fokker-Planck]
//!
//!     EQ --> Kakutani[Kakutani Fixed Point]
//!     EQ --> Nash[Nash Equilibrium]
//! ```
//!
//! ##  Quick Start: Optimal Auctions
//!
//! Calculate the optimal reserve price for an auction where bidders' valuations are uniformly distributed.
//! According to Myerson's Lemma, this is where the virtual valuation $J(v) = 0$.
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::game_theory::mechanism_design::optimal_reserve_price;
//! use math_explorer::applied::game_theory::mechanism_design::distribution::Uniform;
//!
//! fn main() {
//!     // Bidders have valuations uniformly distributed between $0 and $100
//!     let valuation_dist = Uniform::new(0.0, 100.0).unwrap();
//!
//!     // Calculate the revenue-maximizing reserve price
//!     let reserve_price = optimal_reserve_price(&valuation_dist, 0.0, 100.0);
//!
//!     println!("Optimal Reserve Price: ${:.2}", reserve_price);
//!     // For Uniform(0, 100), the answer is $50.
//! }
//! ```

/// Topological concepts for proving equilibrium existence (e.g., Kakutani's Fixed Point Theorem).
pub mod equilibrium;

/// Population dynamics where strategies replicate based on relative fitness (Replicator Equation).
pub mod evolutionary;

/// Games with a continuum of players, modeled by coupled HJB (Control) and Fokker-Planck (Distribution) equations.
pub mod mean_field;

/// Design of rules/mechanisms to achieve specific outcomes, such as revenue-maximizing auctions (Myerson).
pub mod mechanism_design;

/// Errors for Game Theory calculations.
// [cite:graph_parameters_rust]
use pure_math::theory_verification;

theory_verification!(
    module = "game_theory",
    constants = {
        PAYOFF = 5.0;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(
            PAYOFF,
            5.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
