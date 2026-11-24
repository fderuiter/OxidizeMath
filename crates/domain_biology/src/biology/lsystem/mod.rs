//! # Botanical L-System Vascular & Shoot Morphogenesis
//!
//! Deterministic context-free (**D0L**) and **parametric** Lindenmayer systems backed by a
//! pre-allocated double-buffered byte and parameter tape (capped at `MAX_LSYSTEM_SYMBOLS`
//! $= 65{,}536$ symbols) and a pre-allocated 2D turtle graphics segment buffer.
//!
//! Once constructed via `LSystemEngine::new`, derivation steps (`step`, `derive_bounded`) and
//! 2D vascular turtle interpretation (`interpret_turtle`) execute with **zero heap allocations**
//! under `VerifiedAllocator`.

// [cite:biology]

/// Grammar definitions, fixed-capacity production rules, and double-buffered derivation engine.
pub mod grammar;
/// Canonical botanical D0L and parametric shoot/venation presets.
pub mod presets;
#[cfg(feature = "serde")]
mod serde_support;
/// Zero-allocation 2D bracketed turtle geometry interpreter and bounding-box metrics.
pub mod turtle;

pub use grammar::{
    ExpansionToken, LSystemEngine, LSystemGrammar, MAX_LSYSTEM_RULES, MAX_LSYSTEM_SYMBOLS,
    MAX_RULE_EXPANSION, MAX_TURTLE_STACK_DEPTH, ProductionRule,
};
pub use presets::BotanicalLSystemPreset;
pub use turtle::{TurtleBounds, TurtleConfig, TurtleSegment, TurtleState};

use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

impl TheoryDescribable for LSystemEngine {
    #[verified_engine::verified]
    fn theory_description(&self) -> String {
        format!(
            "Lindenmayer systems (L-systems) model apical meristem elongation and vascular branching \
             through parallel rewriting of symbol strings (generation {}, {} symbols, {} segments). \
             In deterministic D0L and parametric grammars, every module simultaneously expands \
             according to production rules and a bracketed 2D turtle interprets internodes with \
             zero heap allocation via double-buffered symbol tapes.",
            self.generation(),
            self.symbols().len(),
            self.segments().len()
        )
    }

    #[verified_engine::verified]
    fn phonetic_description(&self) -> String {
        "Lindenmayer parallel string rewriting grammar where symbols F move and draw vascular \
         internodes, plus and minus rotate by branching angle delta, and square brackets push and \
         pop apical meristem states."
            .to_string()
    }

    #[verified_engine::verified]
    fn theory_citation(&self) -> String {
        "Lindenmayer, A. (1968). Mathematical models for cellular interactions in development. \
         J. Theor. Biol., 18(3), 280-315; Prusinkiewicz, P. & Lindenmayer, A. (1990). \
         The Algorithmic Beauty of Plants."
            .to_string()
    }

    #[verified_engine::verified]
    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Parallel Rewriting".to_string(),
            "All symbols in the active byte tape rewrite simultaneously into the secondary buffer."
                .to_string(),
        );
        map
    }
}

#[cfg(test)]
mod tests;
