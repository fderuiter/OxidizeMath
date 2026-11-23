//! # Favoritism
//!
//! A **satirical** yet mathematically rigorous framework for quantifying parental affection.
//!
//! This module implements the "Unified Favoritism Theory" (UFT), which postulates that
//! a child's standing in the family hierarchy is a deterministic function of financial
//! contribution, proximity, and social signaling, subject to stochastic perturbations
//! (parental mood swings).
//!
//! > **Warning:** This model is for educational and entertainment purposes only.
//! > Do not use this to confront your parents during Thanksgiving dinner.
//!
//! ## The Algorithm
//!
//! The Favoritism Score $F$ is calculated as:
//!
//! $$ F = \frac{ \mathcal{N}(t) \cdot \mathcal{G} \cdot \mathcal{P} \cdot \mathcal{S} \cdot e^{-\lambda \Delta t} \cdot \xi }{ \int_0^t \sum_{s \in \text{siblings}} \frac{1}{d_s(\tau)} d\tau } $$
//!
//! Where:
//! - $\mathcal{N}(t)$: **Proximity Integral** (Time spent close to home).
//! - $\mathcal{G}$: **Gift Determinant** (Determinant of the Gift Matrix).
//! - $\mathcal{P}$: **Personality Linear Combination** (Wealth, Talent, etc.).
//! - $\mathcal{S}$: **Social Multipliers** (Crisis help, Social Media visibility).
//! - $e^{-\lambda \Delta t}$: **Contact Decay** (Memory fading since last call).
//! - $\xi$: **Stochastic Perturbation** ($\xi \sim U(0.9, 1.1)$).
//!
//! ## Flowchart
//!
//! ```mermaid
//! graph TD
//!     subgraph Inputs
//!     I_Time[Time & Proximity]
//!     I_Gifts[Gifts (Emotional & Practical)]
//!     I_Pers[Personality & Wealth]
//!     I_Soc[Social & Crisis]
//!     end
//!
//!     subgraph "The Black Box"
//!     Int[Clenshaw-Curtis Integration]
//!     Det[Gift Matrix Determinant]
//!     WSum[Weighted Sum]
//!     Decay[Exponential Decay]
//!     end
//!
//!     I_Time --> Int
//!     I_Gifts --> Det
//!     I_Pers --> WSum
//!     I_Soc --> Decay
//!
//!     Int --> Numerator
//!     Det --> Numerator
//!     WSum --> Numerator
//!     Decay --> Numerator
//!
//!     Numerator -->|Normalized by Sibling Effort| Score((Favoritism Score))
//! ```
//!
//! ## Example
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::favoritism::{FavoritismInputs, calculate_favoritism_score};
//!
//! // 1. Configure the child's strategy
//! let mut inputs = FavoritismInputs::default();
//!
//! // Explicitly set strategy parameters
//! inputs.personality.wealth = 9.5;            // High financial success
//! inputs.social.helped_during_crisis = true;  // Was there when needed
//! inputs.contact.time_since_last_contact = 2.0; // Called 2 days ago
//!
//! // The "Buying Love" strategy
//! inputs.gifts.g_practical = 10.0; // High value gifts
//! inputs.gifts.g_emotional = 2.0;  // Low sentimental value
//!
//! // 2. Calculate the score
//! let score = calculate_favoritism_score(&inputs);
//!
//! println!("Your Favoritism Score: {:.2}", score);
//! ```

#[allow(missing_docs)]
pub mod favorite_child;
#[allow(missing_docs)]
pub mod scoring;
#[allow(missing_docs)]
pub mod strategies;
#[allow(missing_docs)]
pub mod types;

pub use scoring::{
    calculate_favoritism_score, calculate_favoritism_score_full,
    calculate_favoritism_score_with_rng,
};
pub use types::{
    ComplimentParams, ContactParams, FamilyParams, FavoritismInputs, GiftParams, PersonalityParams,
    SocialParams, TimeParams,
};

// [cite:favorite_child]

use pure_math::theory_verification;

theory_verification!(
    module = "favoritism",
    constants = {
        BASE_SCORE = 1.0;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(
            BASE_SCORE,
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
