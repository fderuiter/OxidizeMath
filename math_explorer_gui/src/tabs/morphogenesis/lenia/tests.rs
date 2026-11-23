#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::PRESET_LENIA_GEMINIUM;
use super::*;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool, SimulationFramework};
use crate::presets::MorphogenesisKineticsMode;
use crate::tabs::morphogenesis::{MorphogenesisTab, MorphogenesisToolState};
use crate::tabs::ExplorerTab;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;

fn screen_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        events,
        ..Default::default()
    }
}

fn read_morpho_state(tab: &MorphogenesisTab) -> MorphogenesisToolState {
    let raw = tab
        .framework
        .active_tool
        .as_ref()
        .and_then(|t| t.save_state())
        .expect("MorphogenesisTool state JSON");
    serde_json::from_str(&raw).expect("valid MorphogenesisToolState")
}

#[test]
fn test_lenia_species_switching_and_gray_scott_modes_in_morphogenesis_tab() {
    let mut tab = MorphogenesisTab::default();
    assert_eq!(tab.active_tool_id(), Some("MorphogenesisTool"));

    // Verify Gray-Scott Mitosis and Coral modes in MorphogenesisTool without breaking Spots/Stripes
    let gs_mitosis = crate::tabs::morphogenesis::presets::PRESET_GRAY_SCOTT_MITOSIS;
    tab.framework
        .apply_starter_card(
            "Gray-Scott Mitosis",
            Some("MorphogenesisTool"),
            Some(gs_mitosis),
        )
        .expect("Gray-Scott Mitosis starter must apply");
    let gs_state = read_morpho_state(&tab);
    assert_eq!(gs_state.kinetics_mode, MorphogenesisKineticsMode::GrayScott);
    assert!((gs_state.f - 0.0367).abs() < 1e-9);
    assert!((gs_state.k - 0.0649).abs() < 1e-9);

    tab.framework
        .apply_starter_card(
            "Turing Spots",
            Some("MorphogenesisTool"),
            Some(crate::starter_gallery::PRESET_TURING_SPOTS),
        )
        .expect("Turing Spots starter must remain intact");
    let spots_state = read_morpho_state(&tab);
    assert_eq!(
        spots_state.kinetics_mode,
        MorphogenesisKineticsMode::Schnakenberg
    );
    assert_eq!(tab.export_config().a, 0.1);

    // Switch cleanly to LeniaTool and cycle Orbium, Geminium, Hydrogeminium, Scutium
    assert!(tab.select_tool_by_name("LeniaTool"));
    assert_eq!(tab.active_tool_id(), Some("LeniaTool"));

    let lenia = tab
        .framework
        .active_tool
        .as_mut()
        .and_then(|t| t.as_any_mut())
        .and_then(|a| a.downcast_mut::<LeniaTool>())
        .expect("active tool should be LeniaTool");

    for sp in LeniaSpeciesPreset::ALL {
        lenia.apply_species_preset(sp);
        lenia.step_once();
        assert!(FiniteStateGuard::is_finite_state(lenia));
        assert!(lenia.engine.telemetry().total_mass > 10.0);
    }
}

fn key_press(key: egui::Key) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
}

#[test]
fn test_lenia_mouse_brushing_and_keyboard_stamping() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = LeniaTool {
        brush_mode: LeniaBrushMode::Slice,
        ..Default::default()
    };
    let initial_mass = tool.engine.telemetry().total_mass;

    let out = ctx.run(screen_input(Vec::new()), |ctx| tool.show(ctx));
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit update");
    let canvas_bounds = ak
        .nodes
        .iter()
        .find(|(_, n)| {
            n.label()
                .is_some_and(|l| l.contains("Lenia Studio Continuous Automaton Canvas"))
        })
        .and_then(|(_, n)| n.bounds())
        .expect("Lenia canvas bounds");
    let center = egui::pos2(
        ((canvas_bounds.x0 + canvas_bounds.x1) * 0.5) as f32,
        ((canvas_bounds.y0 + canvas_bounds.y1) * 0.5) as f32,
    );

    let click_events = vec![
        egui::Event::PointerMoved(center),
        egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(screen_input(click_events), |ctx| tool.show(ctx));
    let sliced_mass = tool.engine.telemetry().total_mass;
    assert!(
        sliced_mass < initial_mass,
        "Mouse slice should reduce mass ({sliced_mass} vs {initial_mass})"
    );
    assert!(tool.focused_canvas_id().is_some());

    let key_events = vec![
        key_press(egui::Key::ArrowRight),
        key_press(egui::Key::Space),
    ];
    let _ = ctx.run(screen_input(key_events), |ctx| tool.show(ctx));
    assert!(tool.engine.telemetry().total_mass > sliced_mass);
}

#[test]
fn test_lenia_finite_state_guard_divergence_and_recovery() {
    let mut tool = LeniaTool {
        dt: 2.5,
        sigma: -0.05,
        playing: true,
        ..Default::default()
    };
    tool.step_once();
    assert!(FiniteStateGuard::diverged(&tool));
    assert!(!tool.playing);
    assert!(tool.diagnostic_report().is_some());

    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert!((0.005..=0.25).contains(&tool.dt));
    assert!((0.002..=0.15).contains(&tool.sigma));
}

#[test]
fn test_lenia_exports_png_gif_csv_json() {
    let ctx = egui::Context::default();
    let mut framework = SimulationFramework::new("biology");
    framework.export_options.width = 320;
    framework.export_options.height = 240;
    framework.export_gif_frames = 4;
    assert!(framework.select_tool_by_name("LeniaTool"));
    let fast_geminium = PRESET_LENIA_GEMINIUM.replace("\"grid_size\": 128", "\"grid_size\": 64");
    framework
        .apply_starter_card(
            "Lenia Geminium Twin",
            Some("LeniaTool"),
            Some(&fast_geminium),
        )
        .expect("Geminium 64x64 starter");

    for fmt in [
        ExportFormat::Png,
        ExportFormat::Gif,
        ExportFormat::Csv,
        ExportFormat::Json,
    ] {
        let _ = ctx.run(screen_input(Vec::new()), |ctx| {
            framework.show_export_modal = true;
            framework.queue_export(fmt);
            framework.show(ctx, "biology");
        });
        assert_eq!(
            framework.export_status_severity,
            ExportStatusSeverity::Success,
            "Export {fmt:?} failed: {:?}",
            framework.export_status_message
        );
    }
}

#[test]
fn test_lenia_compact_url_share_roundtrip() {
    let mut framework = SimulationFramework::new("biology");
    assert!(framework.select_tool_by_name("LeniaTool"));
    framework
        .apply_starter_card(
            "Lenia Geminium Twin",
            Some("LeniaTool"),
            Some(PRESET_LENIA_GEMINIUM),
        )
        .expect("Geminium starter");

    let preset_json = framework
        .export_url_preset_json()
        .expect("URL preset export")
        .expect("non-empty URL preset");
    assert!(
        preset_json.len() < MAX_DECODED_URL_STATE_BYTES,
        "URL genome JSON {} bytes must stay < 4 KiB",
        preset_json.len()
    );
    let share_url = framework.encode_share_url().expect("encode share URL");
    let hash_idx = share_url.find('#').expect("fragment separator");
    let fragment = &share_url[hash_idx..];

    let mut app = crate::app::MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings(fragment, "", None)
        .unwrap());
    let bio_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .unwrap();
    assert_eq!(app.selected_tab, bio_idx);
    assert_eq!(app.tabs[bio_idx].active_tool_id(), Some("LeniaTool"));
    assert_eq!(
        app.tabs[bio_idx].export_url_preset_json().unwrap(),
        Some(preset_json)
    );
}
