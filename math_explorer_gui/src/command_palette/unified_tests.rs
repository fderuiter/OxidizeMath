#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::ToolMetadata;

#[test]
fn test_zero_prefix_palette_unifies_tabs_tools_presets_and_commands() {
    let tabs = crate::tabs::instantiate_tabs();
    let palette = CommandPalette::default();
    let items = palette.filter_items(&tabs);

    assert!(items.iter().any(|i| matches!(i, PaletteItem::Tab { .. })));
    assert!(items.iter().any(|i| matches!(i, PaletteItem::Tool { .. })));
    assert!(items
        .iter()
        .any(|i| matches!(i, PaletteItem::Preset { .. })));
    assert!(items
        .iter()
        .any(|i| matches!(i, PaletteItem::Command { .. })));
}

#[test]
fn test_all_registered_tools_have_rich_descriptive_tags() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    assert!(
        tools.len() >= 56,
        "Expected at least 56 registered tools, found {}",
        tools.len()
    );

    for meta in tools {
        assert!(
            meta.tags.len() >= 2,
            "Tool '{}' in '{}' must declare >= 2 descriptive tags, got {:?}",
            meta.id,
            meta.domain,
            meta.tags
        );
        for &tag in meta.tags {
            assert!(!tag.trim().is_empty(), "Empty tag in '{}'", meta.id);
        }
    }
}

#[test]
fn test_word_boundary_search_matches_scientific_terms_and_rejects_noise() {
    let tabs = crate::tabs::instantiate_tabs();
    let mut palette = CommandPalette::default();

    for (query, expected_substr) in [
        ("mandelbrot", "Fractal"),
        ("kelly", "Bet Size"),
        ("schrodinger", "Wave Simulator"),
        ("rk4", "ODE Solver"),
        ("seir", "SIR Model"),
        ("turing spots", "Turing Spots"),
        ("split horizontal", "Layout: Split Horizontal"),
        ("high contrast", "Theme: High Contrast"),
        ("starter gallery", "Open Starter Gallery"),
    ] {
        palette.query = query.to_string();
        let items = palette.filter_items(&tabs);
        assert!(
            items
                .iter()
                .take(8)
                .any(|item| item.title().contains(expected_substr)),
            "Query '{query}' failed to surface '{expected_substr}' in top results: {:?}",
            items
                .iter()
                .take(6)
                .map(PaletteItem::title)
                .collect::<Vec<_>>()
        );
    }

    assert!(fuzzy_match_score("Quantum Spin Visualizer", "qspin").is_none());
    assert!(fuzzy_match_score("Layout: Split Horizontal", "lsphz").is_none());
}
