#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{AcademicDomain, MathExplorerApp, ThemeMode, ViewportLayout};
use crate::command_palette::{PaletteAction, PaletteSelection};
use eframe::egui;

#[test]
fn test_curated_cold_start_tab_and_flagship_tool_ordering() {
    let app = MathExplorerApp::default();
    #[cfg(all(
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "climate",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    assert!(app.tabs.len() >= 21);

    #[cfg(feature = "physics")]
    {
        assert_eq!(app.tabs[0].name(), "Quantum Mechanics");
        assert_eq!(app.tabs[0].active_tool_id(), Some("WaveSimulator"));
        assert_eq!(app.tabs[1].name(), "MRI Physics");
        assert_eq!(app.tabs[2].name(), "Fluid Dynamics");
        assert_eq!(app.tabs[3].name(), "Chaos Theory");
        assert_eq!(app.tabs[4].name(), "Solid State Physics");
        assert_eq!(app.tabs[5].name(), "Medical Physics");
    }

    #[cfg(feature = "ai")]
    assert_eq!(
        app.tabs.last().map(|t| t.name()),
        Some("Artificial Intelligence")
    );

    let domain_ranks: Vec<usize> = app
        .tabs
        .iter()
        .map(|t| {
            let dom = AcademicDomain::from_str(t.academic_domain()).unwrap();
            AcademicDomain::ALL.iter().position(|&d| d == dom).unwrap()
        })
        .collect();
    let mut sorted = domain_ranks.clone();
    sorted.sort_unstable();
    assert_eq!(
        domain_ranks, sorted,
        "Cold-start tabs must follow Physics -> Pure Math -> Applied Math -> Biology -> Climate -> AI"
    );
}

#[cfg(all(feature = "ai", feature = "physics", feature = "pure_math"))]
#[test]
fn test_legacy_and_named_session_restoration_across_single_split_and_quad() {
    let mut legacy_single = MathExplorerApp::default();
    let legacy_env = r#"{
        "selected_tab": 0,
        "selected_tab_name": null,
        "show_info": true,
        "show_warnings": true,
        "show_errors": true,
        "tab_states": { "Artificial Intelligence": "{}" }
    }"#;
    legacy_single.load_state_from_json(legacy_env);
    assert_eq!(
        legacy_single.tabs[legacy_single.selected_tab].name(),
        "Artificial Intelligence"
    );

    let mut split_src = MathExplorerApp::default();
    let chaos_idx = split_src
        .tabs
        .iter()
        .position(|t| t.name() == "Chaos Theory")
        .unwrap();
    let num_idx = split_src
        .tabs
        .iter()
        .position(|t| t.name() == "Number Theory")
        .unwrap();
    split_src.set_layout(ViewportLayout::SplitHorizontal);
    split_src.set_pane_tab(0, chaos_idx);
    split_src.set_pane_tab(1, num_idx);
    let mut saved_split = split_src.get_app_state();
    saved_split.selected_tab = 0;
    saved_split.secondary_tab = 1;
    saved_split.pane_tabs = vec![0, 1, 2, 3];
    let split_json = split_src.session_manager.create_envelope(saved_split);

    let mut split_dst = MathExplorerApp::default();
    split_dst.load_state_from_json(&split_json);
    assert_eq!(split_dst.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(
        split_dst.tabs[split_dst.selected_tab].name(),
        "Chaos Theory"
    );
    assert_eq!(
        split_dst.tabs[split_dst.secondary_tab].name(),
        "Number Theory"
    );
}

#[cfg(all(
    feature = "ai",
    feature = "climate",
    feature = "physics",
    feature = "pure_math"
))]
#[test]
fn test_quad_grid_session_restoration_by_tab_name() {
    let mut quad_src = MathExplorerApp::default();
    quad_src.set_layout(ViewportLayout::QuadGrid);
    let target_names = [
        "MRI Physics",
        "Graph Theory",
        "Climate",
        "Artificial Intelligence",
    ];
    for (p, name) in target_names.iter().enumerate() {
        let idx = quad_src
            .tabs
            .iter()
            .position(|t| t.name() == *name)
            .unwrap();
        quad_src.set_pane_tab(p, idx);
    }
    let mut saved_quad = quad_src.get_app_state();
    saved_quad.pane_tabs = vec![0, 1, 2, 3];
    let quad_json = quad_src.session_manager.create_envelope(saved_quad);

    let mut quad_dst = MathExplorerApp::default();
    quad_dst.load_state_from_json(&quad_json);
    assert_eq!(quad_dst.viewport_layout, ViewportLayout::QuadGrid);
    for (p, expected) in target_names.iter().enumerate() {
        assert_eq!(quad_dst.tabs[quad_dst.pane_tabs[p]].name(), *expected);
    }
}

#[test]
fn test_palette_executes_actions_and_presets_directly() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    app.execute_palette_selection(
        &ctx,
        PaletteSelection::command(PaletteAction::SetThemeHighContrast),
    );
    assert_eq!(app.theme, ThemeMode::HighContrast);

    app.execute_palette_selection(
        &ctx,
        PaletteSelection::command(PaletteAction::SetLayoutSplitHorizontal),
    );
    assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);

    app.execute_palette_selection(
        &ctx,
        PaletteSelection::command(PaletteAction::OpenStarterGallery),
    );
    assert!(app.show_starter_gallery);
    app.show_starter_gallery = false;

    app.execute_palette_selection(
        &ctx,
        PaletteSelection::command(PaletteAction::ToggleLinkPlotAxes),
    );
    assert!(app.compare_sync.link_axes);

    #[cfg(feature = "pure_math")]
    {
        let preset = inventory::iter::<crate::starter_gallery::StarterPresetMetadata>
            .into_iter()
            .find(|p| p.title == "Logistic Growth")
            .expect("Logistic Growth preset registered");
        let analysis_idx = app
            .tabs
            .iter()
            .position(|t| t.name() == "Analysis & Calculus")
            .unwrap();
        app.focus_pane(1);
        app.execute_palette_selection(&ctx, PaletteSelection::preset(analysis_idx, preset));
        assert_eq!(app.tabs[app.pane_tabs[1]].name(), "Analysis & Calculus");
        let pane1_state = app.pane_tab(1).and_then(|t| t.save_state()).unwrap();
        assert!(pane1_state.contains("LogisticGrowth"));
    }
}

#[cfg(feature = "physics")]
#[derive(Default)]
struct MemoryTestStorage {
    kv: std::collections::HashMap<String, String>,
}

#[cfg(feature = "physics")]
impl eframe::Storage for MemoryTestStorage {
    fn get_string(&self, key: &str) -> Option<String> {
        self.kv.get(key).cloned()
    }
    fn set_string(&mut self, key: &str, value: String) {
        self.kv.insert(key.to_string(), value);
    }
    fn flush(&mut self) {}
}

fn run_ui_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    use eframe::App as _;
    let t = ctx.data_mut(|d| {
        let id = egui::Id::new("cold_start_test_clock");
        let next = d.get_temp::<f64>(id).unwrap_or(0.0) + 0.1;
        d.insert_temp(id, next);
        next
    });
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        time: Some(t),
        events,
        ..Default::default()
    };
    ctx.run(raw, |ctx| app.update(ctx, frame))
}

#[derive(Default)]
struct AkTree {
    nodes: std::collections::HashMap<egui::accesskit::NodeId, egui::accesskit::Node>,
    root: Option<egui::accesskit::NodeId>,
}

impl AkTree {
    fn step(
        &mut self,
        ctx: &egui::Context,
        app: &mut MathExplorerApp,
        frame: &mut eframe::Frame,
        events: Vec<egui::Event>,
    ) {
        let out = run_ui_frame(ctx, app, frame, events);
        if let Some(ak) = &out.platform_output.accesskit_update {
            if let Some(tree) = &ak.tree {
                self.root = Some(tree.root);
            }
            for (id, node) in &ak.nodes {
                self.nodes.insert(*id, node.clone());
            }
        }
    }

    fn find_center(&self, needle: &str) -> Option<egui::Pos2> {
        let mut stack = vec![self.root?];
        while let Some(id) = stack.pop() {
            let Some(node) = self.nodes.get(&id) else {
                continue;
            };
            if let Some(label) = node.label() {
                if label.contains(needle) {
                    if let Some(b) = node.bounds() {
                        return Some(egui::pos2(
                            ((b.x0 + b.x1) * 0.5) as f32,
                            ((b.y0 + b.y1) * 0.5) as f32,
                        ));
                    }
                }
            }
            stack.extend(node.children().iter().copied());
        }
        None
    }
}

fn click_at(pos: egui::Pos2) -> Vec<egui::Event> {
    vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]
}

#[cfg(feature = "physics")]
#[test]
fn test_cold_boot_opens_starter_gallery_and_bypasses_on_session_or_url() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let cc_cold = eframe::CreationContext::_new_kittest(ctx.clone());
    let mut cold_app = MathExplorerApp::new(&cc_cold);
    assert!(
        cold_app.show_starter_gallery,
        "Cold launch without saved session or URL must open global Starter Gallery"
    );

    // Hydrating a shared simulation link dismisses the cold-boot gallery
    assert!(cold_app
        .apply_share_location_strings("#tab=mri&tool=MriTool", "", Some(&ctx))
        .unwrap());
    assert!(!cold_app.show_starter_gallery);

    // Warm launch with saved eframe::Storage bypasses cold-boot gallery
    let mut storage = MemoryTestStorage::default();
    {
        use eframe::App as _;
        cold_app.save(&mut storage);
    }
    let ctx_warm = egui::Context::default();
    let mut cc_warm = eframe::CreationContext::_new_kittest(ctx_warm);
    cc_warm.storage = Some(&storage);
    let warm_app = MathExplorerApp::new(&cc_warm);
    assert!(
        !warm_app.show_starter_gallery,
        "Restored storage session must bypass cold-boot Starter Gallery"
    );
}

fn key_event(key: egui::Key, modifiers: egui::Modifiers) -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }]
}

#[test]
fn test_explore_catalog_button_and_shortcut_on_standalone_traceability_tab() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut ak = AkTree::default();

    let trace_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "traceability")
        .expect("Traceability tab must exist");
    app.selected_tab = trace_idx;
    app.pane_tabs[0] = trace_idx;

    ak.step(&ctx, &mut app, &mut frame, Vec::new());
    let explore_pos = ak
        .find_center("Explore Catalog")
        .expect("Explore Catalog button in top bar");
    ak.step(&ctx, &mut app, &mut frame, click_at(explore_pos));
    assert!(app.show_starter_gallery);
    let live_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(live_msg.as_deref(), Some("Starter template gallery opened"));

    let cmd = if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    };
    let shift_cmd = cmd | egui::Modifiers::SHIFT;

    // Toggle closed via Ctrl+Shift+G / Cmd+Shift+G
    ak.step(
        &ctx,
        &mut app,
        &mut frame,
        key_event(egui::Key::G, shift_cmd),
    );
    assert!(!app.show_starter_gallery);
    let closed_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(
        closed_msg.as_deref(),
        Some("Starter template gallery closed")
    );

    // Re-open via Ctrl+Shift+G, then press Ctrl+K to transition into CommandPalette
    ak.step(
        &ctx,
        &mut app,
        &mut frame,
        key_event(egui::Key::G, shift_cmd),
    );
    assert!(app.show_starter_gallery);
    ak.step(&ctx, &mut app, &mut frame, key_event(egui::Key::K, cmd));
    assert!(!app.show_starter_gallery);
    assert!(app.command_palette.is_open);
}

#[cfg(all(feature = "physics", feature = "biology"))]
#[test]
fn test_per_tab_starter_gallery_prefilters_domain_and_expands_all_domains_with_announcements() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut ak = AkTree::default();

    let mri_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "mri")
        .expect("MRI tab must exist");
    app.selected_tab = mri_idx;
    app.pane_tabs[0] = mri_idx;
    app.tabs[mri_idx]
        .framework_mut()
        .expect("MRI framework")
        .show_starter_gallery = true;

    ak.step(&ctx, &mut app, &mut frame, Vec::new());
    ak.step(&ctx, &mut app, &mut frame, Vec::new());
    assert!(
        ak.find_center("Off-Resonance Bloch Precession").is_some(),
        "MRI Physics preset should be visible first when opened from MRI tab"
    );
    assert!(
        ak.find_center("Turing Spots").is_none(),
        "Biology preset must be filtered out while Physics pill is pre-selected"
    );

    // Click 'All Domains' pill -> reveals cross-domain presets and emits polite live status
    let all_dom_pos = ak.find_center("All Domains").expect("All Domains pill");
    ak.step(&ctx, &mut app, &mut frame, click_at(all_dom_pos));
    let live_msg = ctx
        .data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")))
        .unwrap_or_default();
    assert!(
        live_msg.contains("starter templates shown across all domains"),
        "Expected All Domains count announcement, got '{live_msg}'"
    );

    // Click Biology pill -> focus search -> filter 'Turing Spots' into view -> launch cross-domain
    let bio_pos = ak.find_center("Biology").expect("Biology pill");
    ak.step(&ctx, &mut app, &mut frame, click_at(bio_pos));
    let search_pos = ak
        .find_center("Search starter gallery")
        .expect("Search starter gallery input");
    ak.step(&ctx, &mut app, &mut frame, click_at(search_pos));
    ak.step(
        &ctx,
        &mut app,
        &mut frame,
        vec![egui::Event::Text("Turing Spots".to_string())],
    );
    ak.step(&ctx, &mut app, &mut frame, Vec::new());
    let launch_pos = ak
        .find_center("▶ Launch Turing Spots")
        .expect("Launch Turing Spots button");
    ak.step(&ctx, &mut app, &mut frame, click_at(launch_pos));
    assert_eq!(app.tabs[app.selected_tab].name(), "Morphogenesis");
}
