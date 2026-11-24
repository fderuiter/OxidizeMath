#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
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
    #[cfg(feature = "pure_math")]
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

#[cfg(all(
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics",
    feature = "pure_math"
))]
#[test]
fn test_global_starter_cards_search_and_domain_pill_filtering() {
    use crate::app::AcademicDomain;
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
    assert_eq!(card_academic_domain("biochemical"), AcademicDomain::Biology);
    let hill = filter_starter_cards(&all_cards, "hill", Some(AcademicDomain::Biology));
    assert!(hill.iter().any(|c| c.domain == "biochemical"));

    let disjoint = filter_starter_cards(&all_cards, "turing", Some(AcademicDomain::Climate));
    assert!(disjoint.is_empty());

    let all_challenges = get_all_challenge_cards();
    assert!(all_challenges.len() >= 5);
    let physics_challenges =
        filter_challenge_cards(&all_challenges, "", Some(AcademicDomain::Physics));
    assert!(!physics_challenges.is_empty());
    assert!(physics_challenges
        .iter()
        .all(|c| card_academic_domain(c.domain) == AcademicDomain::Physics));
    let bloch = filter_challenge_cards(&all_challenges, "bloch", Some(AcademicDomain::Physics));
    assert!(bloch.iter().any(|c| c.id.contains("bloch")));
    let no_ai_challenges = filter_challenge_cards(&all_challenges, "", Some(AcademicDomain::Ai));
    assert!(no_ai_challenges.is_empty());

    assert_eq!(card_academic_domain("relativity"), AcademicDomain::Physics);
    assert_eq!(
        card_academic_domain("elliptic_curves"),
        AcademicDomain::PureMath
    );
    let all_stories = get_all_stories();
    assert!(!all_stories.is_empty());
    let filtered_stories = filter_stories(&all_stories, "", Some(AcademicDomain::PureMath));
    assert!(filtered_stories
        .iter()
        .all(|s| card_academic_domain(s.domain) == AcademicDomain::PureMath));

    let all_goals = get_all_goal_challenges();
    assert!(!all_goals.is_empty());
    let filtered_goals = filter_goal_challenges(&all_goals, "", Some(AcademicDomain::AppliedMath));
    assert!(filtered_goals
        .iter()
        .all(|g| card_academic_domain(g.domain) == AcademicDomain::AppliedMath));
}

#[test]
fn test_all_28_tab_slugs_map_to_matching_academic_domains_and_titles() {
    use crate::app::AcademicDomain;
    assert_eq!(
        card_academic_domain("optics_waves"),
        AcademicDomain::Physics
    );
    assert_eq!(
        crate::command_palette::format_domain_title("optics_waves").as_ref(),
        "Optics & Waves"
    );

    assert_eq!(
        card_academic_domain("hyperbolic_geometry"),
        AcademicDomain::PureMath
    );
    assert_eq!(
        crate::command_palette::format_domain_title("hyperbolic_geometry").as_ref(),
        "Hyperbolic Geometry"
    );
    assert_eq!(
        card_academic_domain("stochastic_processes"),
        AcademicDomain::PureMath
    );

    let tabs = crate::tabs::instantiate_tabs();
    let expected_tabs = 1
        + usize::from(cfg!(feature = "ai"))
        + usize::from(cfg!(feature = "applied")) * 6
        + usize::from(cfg!(feature = "biology")) * 3
        + usize::from(cfg!(feature = "climate"))
        + usize::from(cfg!(feature = "epidemiology"))
        + usize::from(cfg!(feature = "physics")) * 9
        + usize::from(cfg!(feature = "pure_math")) * 9;
    assert_eq!(tabs.len(), expected_tabs);
    for tab in &tabs {
        let slug = tab.url_slug();
        if slug == "traceability" {
            continue;
        }
        let expected = AcademicDomain::from_str(tab.academic_domain())
            .unwrap_or_else(|| panic!("Tab '{}' has invalid academic_domain", tab.name()));
        assert_eq!(
            card_academic_domain(slug),
            expected,
            "card_academic_domain('{slug}') did not match tab '{}' domain {:?}",
            tab.name(),
            expected
        );
    }
}
