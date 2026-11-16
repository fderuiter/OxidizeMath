# Scientific Metadata

Standardized metadata contracts, parameter constraints, and academic citation registries for the OxidizeMath ecosystem.

## 1. Overview & Architecture

The `scientific_metadata` crate defines decoupled contracts for mathematical model introspection, accessibility representations, literature citations, and parameter boundary constraints.

By isolating these contracts into a lightweight, standalone library, mathematical simulation kernels remain completely stateless during calculation routines, while graphical user interfaces and verification test suites can dynamically query model properties without introducing global runtime locks or circular dependencies.

### Module Structure

- **`theory`**: Defines the `TheoryDescribable` trait and `ParameterConstraint` struct. These contracts allow models to expose:
  - Verified human-readable descriptions of states and operational regimes.
  - Screen-reader phonetic strings for accessible scientific computing.
  - Literature citations for underlying mathematical formulations.
  - Dynamic parameter querying and modification (`get_parameter`, `set_parameter`).
  - Parameter boundary constraints (`min`, `max`, `step`) for UI controls and numerical validity guards.
  * The `theory_description!` macro for fetching theory descriptions in UI contexts.
- **`citation_registry`**: Provides `CitationRegistry`, a thread-safe global catalog for storing and looking up academic citations across the workspace.

---

## 2. Developer Instructions

### Implementing `TheoryDescribable`

To expose metadata for a simulation model or mathematical structure, implement the `TheoryDescribable` trait:

```rust
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;

/// A simple harmonic oscillator with adjustable frequency.
pub struct SimpleOscillator {
    pub frequency: f64,
}

impl TheoryDescribable for SimpleOscillator {
    fn theory_description(&self) -> String {
        format!("Harmonic oscillator with frequency {:.2} Hz", self.frequency)
    }

    fn phonetic_description(&self) -> String {
        "Harmonic oscillator with frequency in hertz".to_string()
    }

    fn theory_citation(&self) -> String {
        "Landau, L. D., & Lifshitz, E. M. (1976). Mechanics. Pergamon Press.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "frequency".to_string(),
            "Oscillation frequency in Hertz".to_string(),
        );
        map
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let mut params = HashMap::new();
        params.insert(
            "frequency".to_string(),
            ParameterConstraint {
                min: 0.1,
                max: 1000.0,
                step: 0.5,
            },
        );
        params
    }

    fn get_parameter(&self, name: &str) -> Option<f64> {
        match name {
            "frequency" => Some(self.frequency),
            _ => None,
        }
    }

    fn set_parameter(&mut self, name: &str, value: f64) {
        if name == "frequency" {
            self.frequency = value;
        }
    }
}

let mut oscillator = SimpleOscillator { frequency: 440.0 };
assert_eq!(oscillator.get_parameter("frequency"), Some(440.0));

oscillator.set_parameter("frequency", 880.0);
assert_eq!(oscillator.get_parameter("frequency"), Some(880.0));
```

### Using `CitationRegistry`

The `CitationRegistry` provides centralized, thread-safe access to bibliographic references without requiring calculation kernels to maintain persistent state:

```rust
use scientific_metadata::citation_registry::CitationRegistry;

// Register a theoretical citation
CitationRegistry::register(
    "simple_oscillator".to_string(),
    "Landau, L. D., & Lifshitz, E. M. (1976). Mechanics. Pergamon Press.".to_string(),
);

// Retrieve the citation
let citation = CitationRegistry::get("simple_oscillator");
assert_eq!(
    citation.as_deref(),
    Some("Landau, L. D., & Lifshitz, E. M. (1976). Mechanics. Pergamon Press.")
);
```

---

## 3. Integration with Macro Derivations

In domain crates, mathematical models typically derive `Theory` via `verified_engine_macros`, which automatically generates `TheoryDescribable` implementations and populates parameter constraints from struct field attributes.
