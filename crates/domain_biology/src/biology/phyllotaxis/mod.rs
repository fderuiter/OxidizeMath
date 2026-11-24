//! # Botanical Phyllotaxis & Fibonacci Parastichy Mechanics
//!
//! Implements Vogel's planar phyllotaxis model ($r_n = c\sqrt{n}$, $\theta_n = n\alpha$) capped at
//! `MAX_PRIMORDIA` $= 4{,}096$ primordia points alongside zero-allocation opposing parastichy
//! spiral detection $(F_k, F_{k+1})$, annular packing uniformity analysis, and biochemical
//! nutrient-coupled plastochron stepping (`NutrientCoupling`, `PhyllotaxisState`).

// [cite:biology]

/// Biochemical Michaelis-Menten and Hill nutrient coupling for plastochron and L-system growth.
pub mod nutrient;
/// Opposing $(F_k, F_{k+1})$ parastichy spiral pair detection and pre-allocated spiral chain tracing.
pub mod parastichy;
/// Pre-allocated Vogel primordia coordinate generator and divergence angle presets.
pub mod vogel;

pub use nutrient::{
    LUCAS_ANGLE_DEG, NutrientCoupling, NutrientKineticsKind, PhyllotaxisConfig, PhyllotaxisState,
};
pub use parastichy::{ParastichyAnalysis, is_consecutive_fibonacci_pair};
pub use vogel::{
    DivergenceAnglePreset, GOLDEN_ANGLE_DEG, GOLDEN_ANGLE_RAD, GOLDEN_RATIO, MAX_PRIMORDIA,
    PrimordiumPoint, VogelPhyllotaxisEngine,
};

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

impl TheoryDescribable for VogelPhyllotaxisEngine {
    #[verified_engine::verified]
    fn theory_description(&self) -> String {
        let analysis = self.analysis();
        format!(
            "Vogel's phyllotaxis model positions {} botanical primordia on a Fermat spiral \
             r_n = c*sqrt(n), theta_n = n*alpha so each primordium occupies equal area pi*c^2. \
             Current opposing parastichy counts: ({}, {}), packing uniformity {:.3}.",
            self.count(),
            analysis.primary_count,
            analysis.secondary_count,
            analysis.packing_uniformity
        )
    }

    #[verified_engine::verified]
    fn phonetic_description(&self) -> String {
        "Vogel Fermat spiral where radius r sub n equals c times square root of n and polar angle \
         theta sub n equals n times divergence angle alpha, forming opposing Fibonacci parastichies."
            .to_string()
    }

    #[verified_engine::verified]
    fn theory_citation(&self) -> String {
        "Vogel, H. (1979). A better way to construct the sunflower head. Mathematical Biosciences, \
         44(3-4), 179-189."
            .to_string()
    }

    #[verified_engine::verified]
    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Fibonacci Parastichies".to_string(),
            "Irrational golden-angle divergence prevents radial spoke alignment and yields consecutive Fibonacci spirals.".to_string(),
        );
        map
    }
}

#[cfg(test)]
mod tests;
