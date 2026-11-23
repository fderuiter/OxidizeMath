//! # Developmental Phyllotaxis & Nutrient-Coupled L-Systems
//!
//! Mathematical models of botanical morphogenesis linking shoot apical meristem packing
//! (Vogel's golden-angle spiral), continued-fraction Fibonacci parastichy emergence, and
//! parallel Lindenmayer rewriting systems modulated by biochemical enzyme saturation
//! ([`crate::biology::kinetics::MichaelisMenten`] and [`crate::biology::kinetics::HillKinetics`]).
//!
//! ## Quick Start
//!
//! ```rust
//! # use domain_biology::__doctest_facade as math_explorer;
//! use math_explorer::biology::phyllotaxis::{
//!     is_consecutive_fibonacci_pair, NutrientCoupling, PhyllotaxisConfig, PhyllotaxisState,
//! };
//!
//! let state = PhyllotaxisState::new(
//!     512,
//!     180,
//!     PhyllotaxisConfig::default(),
//!     NutrientCoupling::default(),
//! );
//! let (m, n) = state.detect_parastichy_pair();
//! assert!(is_consecutive_fibonacci_pair(m, n));
//! ```

/// Bounded iterative L-system grammar and 2D turtle segment interpreter.
pub mod lsystem;
/// Vogel golden-angle primordia packing, parastichy spiral detection, and nutrient kinetics.
pub mod vogel;

#[cfg(test)]
mod tests;

pub use lsystem::{LSystemConfig, LSystemEngine, LSystemPreset};
pub use vogel::{
    GOLDEN_ANGLE_DEG, GOLDEN_ANGLE_RAD, LUCAS_ANGLE_DEG, NutrientCoupling, NutrientKineticsKind,
    PhyllotaxisConfig, PhyllotaxisState, Primordium, detect_parastichy_spirals,
    is_consecutive_fibonacci_pair,
};

use std::collections::HashMap;
use verified_engine::scientific_metadata::theory::TheoryDescribable;

impl TheoryDescribable for PhyllotaxisConfig {
    fn theory_description(&self) -> String {
        "Vogel golden-angle phyllotaxis r_k = c sqrt(k + 0.5), theta_k = k * alpha coupled with iterative L-system turtle morphogenesis and Michaelis-Menten / Hill nutrient uptake.".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Vogel golden-angle phyllotaxis spiral r sub k equals c square root of k plus 0.5 and theta sub k equals k times alpha with nutrient-coupled L-systems.".to_string()
    }

    fn theory_citation(&self) -> String {
        "Vogel, H. (1979). A better way to construct the sunflower head. Mathematical Biosciences, 44(3-4), 179-189; Lindenmayer, A. (1968). Mathematical models for cellular interactions in development. Journal of Theoretical Biology, 18(3), 280-299.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "en".to_string(),
            "Vogel golden-angle phyllotaxis spiral and nutrient-coupled L-system morphogenesis."
                .to_string(),
        );
        map
    }
}

// [cite:biology]

use pure_math::theory_verification;
theory_verification!(module = "biology", constants = {}, test = {});
