#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::ToolMetadata;
use crate::starter_gallery::StarterPresetMetadata;

#[test]
fn test_all_60_tools_have_non_empty_curated_tags() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    assert_eq!(tools.len(), 60, "Expected all 60 registered ToolMetadata");

    for meta in tools {
        assert!(
            !meta.tags.is_empty(),
            "ToolMetadata '{}' in domain '{}' must define curated tags",
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
fn test_command_palette_indexes_all_71_starter_presets_and_16_workspace_actions() {
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
    let action_count = all_items
        .iter()
        .filter(|i| matches!(i, PaletteItem::Action { .. }))
        .count();

    assert_eq!(tab_count, 21, "Expected 21 ExplorerTab entries");
    assert_eq!(tool_count, 60, "Expected 60 ToolMetadata entries");
    assert_eq!(preset_count, 71, "Expected 71 StarterPresetMetadata cards");
    assert_eq!(
        action_count,
        WorkspaceAction::ALL.len(),
        "Expected all WorkspaceAction entries"
    );

    verify_every_preset_title_matches(&tabs);
}

fn verify_every_preset_title_matches(tabs: &[Box<dyn crate::tabs::ExplorerTab>]) {
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .collect();
    assert_eq!(presets.len(), 71);

    for preset in presets {
        let palette = CommandPalette {
            query: preset.title.to_string(),
            ..Default::default()
        };
        let matches = palette.filter_items(tabs);
        let found = matches.iter().any(|item| match item {
            PaletteItem::Preset { card, .. } => card.id == preset.id && card.title == preset.title,
            _ => false,
        });
        assert!(
            found,
            "CommandPalette::filter_items failed to match preset '{}' (id '{}')",
            preset.title, preset.id
        );
    }
}

#[test]
fn test_word_prefix_and_bounded_gap_scoring_rejects_klein_and_seir() {
    let tabs = crate::tabs::instantiate_tabs();

    for unmatched in ["klein", "seir", "zzzxq"] {
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
                score_tool_in_domain(meta, unmatched),
                None,
                "SimulationFramework sidebar filter matched '{unmatched}' on tool '{}'",
                meta.id
            );
        }
    }
}

#[test]
fn test_synonym_and_bounded_gap_queries_match_expected_models() {
    let tabs = crate::tabs::instantiate_tabs();

    let expected_queries: &[(&str, &str)] = &[
        ("qspin", "Bloch Sphere"),
        ("kermack", "SIR"),
        ("mandelbrot", "Mandelbrot"),
        ("harmonic oscillator", "Harmonic Oscillator"),
        ("turing", "Turing"),
        ("explore catalog", "Explore Catalog"),
        ("2x2 quad", "2x2 Quad Grid"),
        ("high contrast", "High Contrast"),
        ("shareable link", "Copy Shareable Simulation Link"),
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
