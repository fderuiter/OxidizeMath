#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::AcademicDomain;
use crate::framework::{InteractiveTool, ToolMetadata};
use scientific_metadata::theory::TheoryDescribable;

struct DummyUnregisteredTheory;

impl TheoryDescribable for DummyUnregisteredTheory {
    fn theory_description(&self) -> String {
        "Synthetic fallback tool theory.".to_string()
    }
    fn phonetic_description(&self) -> String {
        "Synthetic fallback tool theory.".to_string()
    }
    fn theory_citation(&self) -> String {
        "Fallback Citation (2026)".to_string()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

struct DummyUnregisteredTool {
    theory: DummyUnregisteredTheory,
}

impl InteractiveTool for DummyUnregisteredTool {
    fn name(&self) -> &'static str {
        "Synthetic Fallback Solver"
    }
    fn domain(&self) -> &'static str {
        "analysis"
    }
    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }
}

fn build_dummy_tool() -> Box<dyn InteractiveTool> {
    Box::new(DummyUnregisteredTool {
        theory: DummyUnregisteredTheory,
    })
}

static SYNTHETIC_TOOL_WITHOUT_PRESET: ToolMetadata = ToolMetadata {
    id: "SyntheticFallbackSolver",
    display_name: "Synthetic Fallback Solver",
    domain: "analysis",
    tags: &["synthetic", "fallback"],
    build: build_dummy_tool,
};

#[test]
fn test_per_tool_fallback_generated_even_when_sibling_tools_have_presets() {
    let mut tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>
        .into_iter()
        .filter(|m| m.domain == "analysis")
        .collect();
    tools.push(&SYNTHETIC_TOOL_WITHOUT_PRESET);

    let cards = get_starter_cards_for_domain("analysis", &tools);
    assert!(cards
        .iter()
        .any(|c| c.title == "Harmonic Oscillator" && c.preset_json.is_some()));
    let fallback = cards
        .iter()
        .find(|c| c.id == "SyntheticFallbackSolver")
        .expect("Per-tool fallback card must be generated for tool lacking preset");
    assert!(fallback.preset_json.is_none());
    assert_eq!(fallback.tool_name, Some("SyntheticFallbackSolver"));
}

#[test]
fn test_global_starter_cards_search_and_domain_pill_filtering() {
    let all_cards = get_all_starter_cards();
    assert!(all_cards.len() >= 67);

    let physics_cards = filter_starter_cards(&all_cards, "", Some(AcademicDomain::Physics));
    assert!(!physics_cards.is_empty());
    assert!(physics_cards
        .iter()
        .all(|c| card_academic_domain(c.domain) == AcademicDomain::Physics));

    let lorenz = filter_starter_cards(&all_cards, "lorenz", None);
    assert!(lorenz.iter().any(|c| c.title.contains("Attractor")));

    let turing = filter_starter_cards(&all_cards, "turing", Some(AcademicDomain::Biology));
    assert!(turing.iter().any(|c| c.title.contains("Turing")));

    let disjoint = filter_starter_cards(&all_cards, "turing", Some(AcademicDomain::Climate));
    assert!(disjoint.is_empty());
}
