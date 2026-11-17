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
}

#[test]
fn test_query_change_resets_selection() {
    let ctx = egui::Context::default();
    let tabs = mock_tabs();
    let mut palette = CommandPalette::default();

    palette.open();
    palette.selected_index = 2;

    palette.query = "fluid".to_string();

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let _ = palette.show(ctx, &tabs);
    });

    assert_eq!(palette.selected_index, 0);
}

#[test]
fn test_palette_tool_indexing_and_fuzzy_match() {
    let tabs = mock_tabs();
    let mut palette = CommandPalette {
        query: "dijkstra".to_string(),
        ..Default::default()
    };

    // 1. Dijkstra (via tag on AlgorithmVisualizerTool)
    let items = palette.filter_items(&tabs);
    assert!(
        !items.is_empty(),
        "Dijkstra tool should be matched via tags"
    );
    let first = &items[0];
    match first {
        PaletteItem::Tool { name, .. } => {
            assert_eq!(*name, "AlgorithmVisualizerTool");
        }
        _ => panic!(
            "Expected Dijkstra query to match AlgorithmVisualizerTool, found {:?}",
            first
        ),
    }

    // 2. Lorenz (via tag on AttractorPlotter)
    palette.query = "lorenz".to_string();
    let items = palette.filter_items(&tabs);
    assert!(!items.is_empty(), "Lorenz tool should be matched via tags");
    match &items[0] {
        PaletteItem::Tool { name, .. } => {
            assert_eq!(*name, "AttractorPlotter");
        }
        _ => panic!(
            "Expected Lorenz query to match AttractorPlotter, found {:?}",
            items[0]
        ),
    }

    // 3. CERA (via name CeraTool)
    palette.query = "cera".to_string();
    let items = palette.filter_items(&tabs);
    assert!(
        !items.is_empty(),
        "Cera tool should be matched via name/tags"
    );
    match &items[0] {
        PaletteItem::Tool { name, .. } => {
            assert_eq!(*name, "CeraTool");
        }
        _ => panic!(
            "Expected CERA query to match CeraTool, found {:?}",
            items[0]
        ),
    }
}

#[test]
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
        Some(PaletteSelection::tool(0, "AlgorithmVisualizerTool"))
    );
    assert!(!palette.is_open);
}

#[test]
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
            assert_eq!(*name, "OdeSolverTool");
        }
        _ => panic!("Expected OdeSolverTool to rank higher due to prefix match score"),
    }
}

#[test]
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
            assert_eq!(*name, "AlgorithmVisualizerTool");
        }
        _ => panic!("Expected Tool item"),
    }
}
