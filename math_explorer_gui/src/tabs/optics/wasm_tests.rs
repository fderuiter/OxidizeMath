#![cfg_attr(any(), verified(opt_out = "tests"))]

use wasm_bindgen_test::wasm_bindgen_test;

use super::diffraction_tool::FraunhoferDiffractionTool;
use super::ray_bench_tool::CausticsRayBenchTool;
use crate::app::MathExplorerApp;
use crate::framework::{ExportFormat, FiniteStateGuard, InteractiveTool, SimulationFramework};
use crate::starter_gallery::StarterPresetMetadata;

#[wasm_bindgen_test]
fn wasm_optics_starter_presets_url_roundtrip_and_exports() {
    for preset in inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| p.domain == "optics")
    {
        let mut fw = SimulationFramework::new("optics");
        fw.apply_starter_card(
            preset.title,
            preset.tool_name,
            Some(preset.generate_preset()),
        )
        .expect("apply optics starter preset on wasm");
    }

    let mut bench = CausticsRayBenchTool::default();
    bench.state.launch_angle_rad = f64::NAN;
    assert!(!bench.has_finite_parameters());
    FiniteStateGuard::clamp_parameters(&mut bench);
    assert!(bench.has_finite_parameters());

    let mut diff = FraunhoferDiffractionTool::default();
    diff.state.wavelength_nm = f64::NEG_INFINITY;
    assert!(!diff.has_finite_parameters());
    FiniteStateGuard::clamp_parameters(&mut diff);
    assert!(diff.has_finite_parameters());

    let mut app = MathExplorerApp::default();
    let idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "optics")
        .expect("optics tab present on wasm");
    app.selected_tab = idx;
    let url = app.encode_current_share_url().expect("encode url");
    let hash = url.split('#').nth(1).unwrap_or(&url);
    let mut restored = MathExplorerApp::default();
    assert!(restored
        .apply_share_location_strings(hash, "", None)
        .unwrap());
    assert_eq!(restored.tabs[restored.selected_tab].url_slug(), "optics");

    let mut fw = SimulationFramework::new("optics");
    assert!(fw.select_tool_by_name("FraunhoferDiffractionTool"));
    for fmt in [
        ExportFormat::Png,
        ExportFormat::Svg,
        ExportFormat::Csv,
        ExportFormat::Json,
    ] {
        let ctx = eframe::egui::Context::default();
        let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
            fw.show_export_modal = true;
            fw.queue_export(fmt);
            fw.show(ctx, "wasm_optics_export");
        });
        assert_eq!(
            fw.export_status_severity,
            crate::framework::ExportStatusSeverity::Success
        );
    }
}

#[wasm_bindgen_test]
fn wasm_optics_keyboard_canvas_controls() {
    let ctx = eframe::egui::Context::default();
    ctx.enable_accesskit();
    let mut kb_tool = CausticsRayBenchTool::default();
    let initial_angle = kb_tool.state.launch_angle_rad;
    let rect = eframe::egui::Rect::from_min_size(
        eframe::egui::Pos2::ZERO,
        eframe::egui::vec2(960.0, 720.0),
    );
    let make_key = |key: eframe::egui::Key| eframe::egui::RawInput {
        screen_rect: Some(rect),
        events: vec![eframe::egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: eframe::egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let out = ctx.run(
        eframe::egui::RawInput {
            screen_rect: Some(rect),
            ..Default::default()
        },
        |ctx| kb_tool.show(ctx),
    );
    let nid = out
        .platform_output
        .accesskit_update
        .expect("wasm accesskit")
        .nodes
        .iter()
        .find(|(_, n)| {
            n.role() == eframe::egui::accesskit::Role::GraphicsDocument
                && n.label().unwrap_or("").contains("Optical Ray Bench")
        })
        .map(|(id, _)| *id)
        .expect("wasm canvas node");
    let focused = (0..80).any(|_| {
        ctx.run(make_key(eframe::egui::Key::Tab), |ctx| kb_tool.show(ctx))
            .platform_output
            .accesskit_update
            .is_some_and(|u| u.focus == nid)
    });
    assert!(focused, "Optical Ray Bench reachable via Tab on WASM");
    let _ = ctx.run(make_key(eframe::egui::Key::ArrowRight), |ctx| {
        kb_tool.show(ctx)
    });
    assert!(kb_tool.state.launch_angle_rad > initial_angle);
}
