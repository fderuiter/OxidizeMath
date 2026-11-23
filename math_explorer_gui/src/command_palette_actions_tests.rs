#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::ToolMetadata;
use crate::starter_gallery::StarterPresetMetadata;

#[test]
fn test_all_registered_tools_have_lowercase_curated_tags() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    assert_eq!(tools.len(), 87, "Expected all 87 registered ToolMetadata");

    for meta in tools {
        assert!(
            meta.tags.len() >= 2,
            "ToolMetadata '{}' in domain '{}' must define >= 2 curated tags",
            meta.id,
            meta.domain
        );
        for &tag in meta.tags {
            let trimmed = tag.trim();
            assert!(
                !trimmed.is_empty(),
                "ToolMetadata '{}' has blank tag",
                meta.id
            );
            assert_eq!(
                trimmed,
                trimmed.to_lowercase(),
                "ToolMetadata '{}' tag '{}' should be lowercase",
                meta.id,
                tag
            );
        }
    }
}

#[test]
fn test_command_palette_indexes_all_tabs_tools_starter_presets_and_commands() {
    let tabs = crate::tabs::instantiate_tabs();
    let palette = CommandPalette::default();
    let all_items = palette.filter_items(&tabs);

    let tab_count = all_items
        .iter()
        .filter(|i| matches!(i, PaletteItem::Tab { .. }))
        .count();
    let tool_count = all_items
        .iter()
        .filter(|i| matches!(i, PaletteItem::Tool { .. }))
        .count();
    let preset_count = all_items
        .iter()
        .filter(|i| matches!(i, PaletteItem::Preset { .. }))
        .count();
    let command_count = all_items
        .iter()
        .filter(|i| matches!(i, PaletteItem::Command { .. }))
        .count();

    assert_eq!(tab_count, 28, "Expected 28 ExplorerTab entries");
    assert_eq!(tool_count, 87, "Expected 87 ToolMetadata entries");
    assert_eq!(
        preset_count, 148,
        "Expected 148 StarterPresetMetadata cards"
    );
    assert_eq!(
        command_count,
        actions::all_builtin_commands().count(),
        "Expected all built-in CommandPalette actions"
    );

    verify_every_preset_title_matches(&tabs);
}

fn verify_every_preset_title_matches(tabs: &[Box<dyn crate::tabs::ExplorerTab>]) {
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();
    assert_eq!(presets.len(), 148);

    for target in presets {
        let palette = CommandPalette {
            query: target.title.to_string(),
            ..Default::default()
        };
        let matches = palette.filter_items(tabs);
        let found = matches.iter().any(|item| match item {
            PaletteItem::Preset { preset, .. } => {
                preset.id == target.id && preset.title == target.title
            }
            _ => false,
        });
        assert!(
            found,
            "CommandPalette::filter_items failed to match preset '{}' (id '{}')",
            target.title, target.id
        );
    }
}

#[test]
fn test_word_boundary_scoring_rejects_scattered_subsequence_noise() {
    let tabs = crate::tabs::instantiate_tabs();

    for unmatched in ["lsphz", "qspin", "zzzxq"] {
        let palette = CommandPalette {
            query: unmatched.to_string(),
            ..Default::default()
        };
        let results = palette.filter_items(&tabs);
        assert!(
            results.is_empty(),
            "Query '{unmatched}' must return zero irrelevant subsequence matches, got: {results:?}"
        );

        for meta in inventory::iter::<ToolMetadata> {
            assert_eq!(
                score_tool_metadata(meta, unmatched),
                None,
                "SimulationFramework sidebar filter matched '{unmatched}' on tool '{}'",
                meta.id
            );
        }
    }
}

#[test]
fn test_synonym_and_command_queries_match_expected_models_and_actions() {
    let tabs = crate::tabs::instantiate_tabs();

    let expected_queries: &[(&str, &str)] = &[
        ("kermack", "SIR"),
        ("seir", "SIR"),
        ("klein bottle", "Surface Viewer"),
        ("mandelbrot", "Mandelbrot"),
        ("harmonic oscillator", "Harmonic Oscillator"),
        ("turing", "Turing"),
        ("parrondo", "Parrondo"),
        ("braess", "Braess"),
        ("schwarzschild", "Geodesic Playground"),
        ("weierstrass", "Weierstrass"),
        ("symplectic", "Hamiltonian"),
        ("henon", "Hamiltonian Chaos Lab"),
        ("copula", "Correlated Risk"),
        ("win ratio", "Win Ratio"),
        ("freesurfer", "Cortical Morphometry"),
        ("sandpile", "Sandpile"),
        ("grpo", "Closed-Loop"),
        ("baum-welch", "Markov"),
        ("devig", "Devig"),
        ("arbitrage", "Arbitrage, Dutching & Hedge"),
        ("fdtd", "2D Wave & Caustics Playground"),
        ("explore catalog", "Explore Catalog"),
        ("2x2 quad", "2x2 Quad Grid"),
        ("high contrast", "High Contrast"),
        ("shareable link", "Copy Share Link"),
        ("save parameter preset json", "Export Preset JSON"),
    ];

    for &(query, expected_substr) in expected_queries {
        let palette = CommandPalette {
            query: query.to_string(),
            ..Default::default()
        };
        let results = palette.filter_items(&tabs);
        assert!(
            results.iter().any(|i| {
                i.title()
                    .to_lowercase()
                    .contains(&expected_substr.to_lowercase())
            }),
            "Expected query '{query}' to match an item containing '{expected_substr}', got: {results:?}"
        );
    }
}
