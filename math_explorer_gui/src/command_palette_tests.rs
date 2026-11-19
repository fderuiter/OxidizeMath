use super::*;

struct MockTab(&'static str);

impl crate::tabs::ExplorerTab for MockTab {
    fn name(&self) -> &'static str {
        self.0
    }
    fn show(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {}
}

fn mock_tabs() -> Vec<Box<dyn crate::tabs::ExplorerTab>> {
    vec![
        Box::new(MockTab("Quantum / Spin Visualization")),
        Box::new(MockTab("Fluid Dynamics / Lattice Boltzmann")),
        Box::new(MockTab("Analysis / ODE Solver")),
        Box::new(MockTab("AI / Grid World")),
    ]
}

#[test]
fn test_fuzzy_match_score() {
    assert!(fuzzy_match_score("Quantum / Spin Visualization", "quantum").is_some());
    assert!(fuzzy_match_score("Quantum / Spin Visualization", "spin").is_some());
    assert!(fuzzy_match_score("Quantum / Spin Visualization", "qspin").is_some());
    assert!(fuzzy_match_score("Quantum / Spin Visualization", "nonexistent").is_none());
}

#[test]
fn test_palette_open_close_toggle() {
    let mut palette = CommandPalette::default();
    assert!(!palette.is_open);

    palette.open();
    assert!(palette.is_open);
    assert!(palette.just_opened);

    palette.close();
    assert!(!palette.is_open);

    palette.toggle();
    assert!(palette.is_open);
}

#[test]
fn test_palette_filtering_and_selection() {
    let ctx = egui::Context::default();
    let tabs = mock_tabs();
    let mut palette = CommandPalette::default();

    palette.open();
    palette.query = "quantum".to_string();

    let mut selected = None;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        selected = palette.show(ctx, &tabs);
    });

    assert_eq!(selected, None);
    assert!(palette.is_open);

    // Press enter to select top filtered item
    let raw_input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };

    let _ = ctx.run(raw_input, |ctx| {
        selected = palette.show(ctx, &tabs);
    });

    assert_eq!(selected, Some(PaletteSelection::tab(0))); // Quantum is at index 0
    assert!(!palette.is_open);
}

#[test]
fn test_palette_arrow_navigation() {
    let ctx = egui::Context::default();
    let tabs = mock_tabs();
    let mut palette = CommandPalette::default();

    palette.open();

    // Arrow down
    let raw_input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::ArrowDown,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };

    let _ = ctx.run(raw_input, |ctx| {
        let _ = palette.show(ctx, &tabs);
    });

    assert_eq!(palette.selected_index, 1);

    // Arrow up wraps around or goes back
    let raw_input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::ArrowUp,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };

    let _ = ctx.run(raw_input, |ctx| {
        let _ = palette.show(ctx, &tabs);
    });

    assert_eq!(palette.selected_index, 0);
}

#[test]
fn test_palette_escape_key_closes() {
    crate::accessibility::clear_test_announcements();
    let ctx = egui::Context::default();
    let tabs = mock_tabs();
    let mut palette = CommandPalette::default();

    palette.open();

    let raw_input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };

    let mut selected = None;
    let _ = ctx.run(raw_input, |ctx| {
        selected = palette.show(ctx, &tabs);
    });

    assert_eq!(selected, None);
    assert!(!palette.is_open);
    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg, Some("Command palette closed".to_string()));
    assert_eq!(
        crate::accessibility::all_announcements(),
        vec![("Command palette closed".to_string(), "polite".to_string())]
    );
}

#[test]
fn test_query_change_resets_selection() {
    crate::accessibility::clear_test_announcements();
    let ctx = egui::Context::default();
    let tabs = mock_tabs();
    let mut palette = CommandPalette::default();

    palette.open();
    palette.selected_index = 2;

    palette.query = "fluid".to_string();
    let expected_fluid_count = palette.filter_items(&tabs).len();
    assert!(expected_fluid_count > 0);

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let _ = palette.show(ctx, &tabs);
    });

    assert_eq!(palette.selected_index, 0);
    let expected_msg = if expected_fluid_count == 1 {
        "1 matching result found".to_string()
    } else {
        format!("{expected_fluid_count} matching results found")
    };
    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg.as_deref(), Some(expected_msg.as_str()));
    assert_eq!(
        crate::accessibility::all_announcements(),
        vec![(expected_msg, "polite".to_string())]
    );
}

#[test]
fn test_search_input_widget_info_label_descriptor() {
    let ctx = egui::Context::default();
    let tabs = mock_tabs();
    let mut palette = CommandPalette::default();
    palette.open();

    // Frame 1 (modal sizing pass) + Frame 2 (settled layout): search input is enabled & focused
    for _ in 0..2 {
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            let _ = palette.show(ctx, &tabs);
        });
    }

    let info: egui::WidgetInfo = ctx
        .data(|d| d.get_temp(egui::Id::new(SEARCH_WIDGET_INFO_ID)))
        .expect("search input WidgetInfo descriptor should be stored in temp data");
    assert_eq!(info.typ, egui::WidgetType::TextEdit);
    assert!(info.enabled);
    assert_eq!(info.label.as_deref(), Some("Search tabs and tools"));
    assert_eq!(info.current_text_value.as_deref(), Some(""));
    assert!(info.description().contains("Search tabs and tools"));

    // Next frame: type query via egui::Event::Text and verify updated WidgetInfo & live announcement
    crate::accessibility::clear_test_announcements();
    let raw_input = egui::RawInput {
        events: vec![egui::Event::Text("dijkstra".to_string())],
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        let _ = palette.show(ctx, &tabs);
    });

    let updated_info: egui::WidgetInfo = ctx
        .data(|d| d.get_temp(egui::Id::new(SEARCH_WIDGET_INFO_ID)))
        .expect("updated WidgetInfo descriptor expected");
    assert_eq!(updated_info.label.as_deref(), Some("Search tabs and tools"));
    assert_eq!(updated_info.current_text_value.as_deref(), Some("dijkstra"));
    assert_eq!(updated_info.prev_text_value.as_deref(), Some(""));
    let dijkstra_count = palette.filter_items(&tabs).len();
    let expected = match dijkstra_count {
        0 => "No matching tools found".to_string(),
        1 => "1 matching result found".to_string(),
        n => format!("{n} matching results found"),
    };
    assert_eq!(
        crate::accessibility::all_announcements(),
        vec![(expected.clone(), "polite".to_string())]
    );
    assert_eq!(
        ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message"))),
        Some(expected)
    );
}

#[test]
fn test_query_empty_state_and_clear_back_to_unfiltered_announcements() {
    let ctx = egui::Context::default();
    let tabs = mock_tabs();
    let mut palette = CommandPalette::default();
    palette.open();

    let total_unfiltered = palette.filter_items(&tabs).len();
    assert!(total_unfiltered >= tabs.len());

    // 1. Zero-match query announces "No matching tools found"
    crate::accessibility::clear_test_announcements();
    palette.query = "zzzz_unmatched_query_999".to_string();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let _ = palette.show(ctx, &tabs);
    });
    assert_eq!(
        ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")))
            .as_deref(),
        Some("No matching tools found")
    );
    assert_eq!(
        crate::accessibility::all_announcements(),
        vec![("No matching tools found".to_string(), "polite".to_string())]
    );

    // 2. Unchanged query on subsequent frame does not emit duplicate announcements
    crate::accessibility::clear_test_announcements();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let _ = palette.show(ctx, &tabs);
    });
    assert!(crate::accessibility::all_announcements().is_empty());

    // 3. Clearing query back to empty announces full unfiltered count
    palette.query.clear();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let _ = palette.show(ctx, &tabs);
    });
    let expected_full = format!("{total_unfiltered} matching results found");
    assert_eq!(
        ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message"))),
        Some(expected_full.clone())
    );
    assert_eq!(
        crate::accessibility::all_announcements(),
        vec![(expected_full, "polite".to_string())]
    );
}

#[test]
#[cfg(all(feature = "pure_math", feature = "physics", feature = "climate"))]
fn test_palette_tool_indexing_and_fuzzy_match() {
    let tabs = mock_tabs();
    // 1. Dijkstra (via tag on AlgorithmVisualizerTool)
    let mut palette = CommandPalette {
        query: "dijkstra".to_string(),
        ..Default::default()
    };
    let items = palette.filter_items(&tabs);
    assert!(
        !items.is_empty(),
        "Dijkstra tool should be matched via tags"
    );
    let first = &items[0];
    match first {
        PaletteItem::Tool { name, .. } => {
            assert_eq!(*name, "Algorithm Visualizer");
        }
        _ => panic!(
            "Expected Dijkstra query to match Algorithm Visualizer, found {:?}",
            first
        ),
    }

    // 2. Lorenz (via tag on Attractor Plotter)
    palette.query = "lorenz".to_string();
    let items = palette.filter_items(&tabs);
    assert!(!items.is_empty(), "Lorenz tool should be matched via tags");
    match &items[0] {
        PaletteItem::Tool { name, .. } => {
            assert_eq!(*name, "Attractor Plotter");
        }
        _ => panic!(
            "Expected Lorenz query to match Attractor Plotter, found {:?}",
            items[0]
        ),
    }

    // 3. CERA (via name CERA Model / id CeraTool)
    palette.query = "cera".to_string();
    let items = palette.filter_items(&tabs);
    assert!(
        !items.is_empty(),
        "Cera tool should be matched via name/tags"
    );
    match &items[0] {
        PaletteItem::Tool { name, .. } => {
            assert_eq!(*name, "CERA Model");
        }
        _ => panic!(
            "Expected CERA query to match CERA Model, found {:?}",
            items[0]
        ),
    }
}

#[test]
#[cfg(feature = "pure_math")]
fn test_palette_keyboard_tool_selection() {
    let ctx = egui::Context::default();
    let tabs = mock_tabs();
    let mut palette = CommandPalette::default();

    palette.open();
    palette.query = "dijkstra".to_string();

    let raw_input = egui::RawInput {
        events: vec![egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };

    let mut selected = None;
    let _ = ctx.run(raw_input, |ctx| {
        selected = palette.show(ctx, &tabs);
    });

    assert_eq!(
        selected,
        Some(PaletteSelection::tool(0, "Algorithm Visualizer"))
    );
    assert!(!palette.is_open);
}

#[test]
#[cfg(feature = "pure_math")]
fn test_palette_score_ranking_tabs_vs_tools() {
    let tabs = mock_tabs();
    // Query that matches both tab name and tool
    // "ode" matches tab "Analysis / ODE Solver" (substring pos) and "OdeSolverTool" (prefix)
    let palette = CommandPalette {
        query: "ode".to_string(),
        ..Default::default()
    };
    let items = palette.filter_items(&tabs);
    assert!(!items.is_empty());
    // Exact prefix match on OdeSolverTool has score 1, while "Analysis / ODE Solver" substring has score > 10.
    // Therefore OdeSolverTool should be ranked higher!
    match &items[0] {
        PaletteItem::Tool { name, .. } => {
            assert_eq!(*name, "ODE Solvers");
        }
        _ => panic!("Expected ODE Solvers to rank higher due to prefix match score"),
    }
}

#[test]
#[cfg(feature = "pure_math")]
fn test_palette_domain_fallback_safely_defaults_to_zero() {
    // Create tabs without matching domains
    let tabs: Vec<Box<dyn crate::tabs::ExplorerTab>> =
        vec![Box::new(MockTab("General Simulation"))];
    let palette = CommandPalette {
        query: "dijkstra".to_string(),
        ..Default::default()
    };
    let items = palette.filter_items(&tabs);
    assert!(!items.is_empty());
    match &items[0] {
        PaletteItem::Tool {
            tab_index, name, ..
        } => {
            assert_eq!(
                *tab_index, 0,
                "Tool with unmatched domain should fall back to tab 0"
            );
            assert_eq!(*name, "Algorithm Visualizer");
        }
        _ => panic!("Expected Tool item"),
    }
}

#[test]
fn test_palette_domain_title_formatting_and_dual_key_search() {
    assert_eq!(format_domain_title("clinical_trials"), "Clinical Trials");
    assert_eq!(format_domain_title("number_theory"), "Number Theory");
    assert_eq!(
        format_domain_title("geometry_topology"),
        "Geometry & Topology"
    );

    #[cfg(feature = "applied")]
    {
        let tabs = mock_tabs();
        for query in [
            "clinical_trials",
            "Clinical Trials",
            "Sample Size Calculator (Clinical Trials)",
        ] {
            let palette = CommandPalette {
                query: query.to_string(),
                ..Default::default()
            };
            let items = palette.filter_items(&tabs);
            assert!(
                items.iter().any(|item| matches!(
                    item,
                    PaletteItem::Tool {
                        name: "Sample Size Calculator",
                        domain: "clinical_trials",
                        ..
                    }
                )),
                "Expected Sample Size Calculator to match query '{query}'"
            );
        }
    }
}
