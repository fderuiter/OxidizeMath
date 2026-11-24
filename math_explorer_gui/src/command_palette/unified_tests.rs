#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::ToolMetadata;

#[test]
fn test_zero_prefix_palette_unifies_tabs_tools_presets_and_commands() {
    let tabs = crate::tabs::instantiate_tabs();
    let palette = CommandPalette::default();
    let items = palette.filter_items(&tabs);

    assert!(items.iter().any(|i| matches!(i, PaletteItem::Tab { .. })));
    #[cfg(any(
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "climate",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    {
        assert!(items.iter().any(|i| matches!(i, PaletteItem::Tool { .. })));
        assert!(items
            .iter()
            .any(|i| matches!(i, PaletteItem::Preset { .. })));
    }
    assert!(items
        .iter()
        .any(|i| matches!(i, PaletteItem::Command { .. })));
}

#[test]
fn test_all_registered_tools_have_rich_descriptive_tags() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    #[cfg(all(
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "climate",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
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

fn assert_top_palette_match(
    palette: &mut CommandPalette,
    tabs: &[Box<dyn crate::tabs::ExplorerTab>],
    query: &str,
    expected_substr: &str,
) {
    palette.query = query.to_string();
    let items = palette.filter_items(tabs);
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

#[test]
fn test_word_boundary_search_matches_scientific_terms_and_rejects_noise() {
    let tabs = crate::tabs::instantiate_tabs();
    let mut palette = CommandPalette::default();

    for (query, expected_substr) in [
        ("split horizontal", "Layout: Split Horizontal"),
        ("high contrast", "Theme: High Contrast"),
        ("starter gallery", "Open Starter Gallery"),
        ("step back", "Step Back"),
        ("previous keyframe", "Previous Keyframe"),
    ] {
        assert_top_palette_match(&mut palette, &tabs, query, expected_substr);
    }

    #[cfg(feature = "physics")]
    for (query, expected_substr) in [("mandelbrot", "Fractal"), ("schrodinger", "Wave Simulator")] {
        assert_top_palette_match(&mut palette, &tabs, query, expected_substr);
    }

    #[cfg(feature = "pure_math")]
    for (query, expected_substr) in [
        ("kelly", "Bet Size"),
        ("rk4", "ODE Solver"),
        ("correlated risk", "Correlated Risk"),
    ] {
        assert_top_palette_match(&mut palette, &tabs, query, expected_substr);
    }

    #[cfg(feature = "epidemiology")]
    assert_top_palette_match(&mut palette, &tabs, "seir", "SIR Model");

    #[cfg(feature = "biology")]
    assert_top_palette_match(&mut palette, &tabs, "turing spots", "Turing Spots");

    assert!(fuzzy_match_score("Quantum Spin Visualizer", "qspin").is_none());
    assert!(fuzzy_match_score("Layout: Split Horizontal", "lsphz").is_none());
}

#[test]
fn test_explore_catalog_alias_shortcut_and_preset_tag_discovery() {
    let tabs = crate::tabs::instantiate_tabs();
    let palette = CommandPalette {
        query: "explore catalog".to_string(),
        ..Default::default()
    };
    let catalog_items = palette.filter_items(&tabs);
    let expected_shortcut = if cfg!(target_os = "macos") {
        "Cmd+Shift+G"
    } else {
        "Ctrl+Shift+G"
    };
    assert!(catalog_items.iter().take(4).any(|item| matches!(
        item,
        PaletteItem::Command {
            action: PaletteAction::OpenStarterGallery,
            shortcut: Some(sc),
            ..
        } if sc == expected_shortcut
    )));

    #[cfg(feature = "epidemiology")]
    {
        let mut palette = palette;
        palette.query = "kermack".to_string();
        let kermack_items = palette.filter_items(&tabs);
        assert!(kermack_items
            .iter()
            .any(|item| matches!(item, PaletteItem::Tool { name, .. } if *name == "SIR Model")));
        assert!(kermack_items.iter().any(|item| matches!(
            item,
            PaletteItem::Preset { preset, .. } if preset.domain == "epidemiology"
        )));
    }
}
