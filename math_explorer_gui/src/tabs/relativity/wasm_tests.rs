#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Headless browser `wasm32` tests for [`RelativityTab`], `#tab=relativity` hydration,
//! `Camera3D` interaction, custom AST metrics, 2D lensing atlas `1x` convergence, and challenges.

use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

use super::*;
use crate::app::MathExplorerApp;
use crate::async_sim::atlas::AtlasTeleportEvent;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::starter_gallery::{get_goal_challenges_for_domain, get_stories_for_domain};
use crate::tabs::ExplorerTab;
use crate::url_state::{encode_share_fragment, parse_share_location};

fn wasm_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        events,
        ..Default::default()
    }
}

#[wasm_bindgen_test]
fn wasm_relativity_tab_hydrates_renders_gestures_presets_and_exports() {
    let mut app = MathExplorerApp::default();
    let hydrated = app
        .apply_share_location_strings("#tab=relativity&tool=GeodesicPlaygroundTool", "", None)
        .expect("relativity share URL must parse");
    assert!(hydrated);
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "relativity");

    let mut tab = RelativityTab::default();
    assert!(tab.select_tool_by_name("GeodesicPlaygroundTool"));
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(wasm_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });

    // Verify wheel gesture inside 3D viewport and CSV/SVG/PNG exports on wasm32
    let wheel = vec![
        egui::Event::PointerMoved(egui::pos2(680.0, 280.0)),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, 1.5),
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(wasm_input(wheel), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });

    let mut inspector = ChristoffelInspectorTool::default();
    inspector
        .import_preset_json(presets::PRESET_CHRISTOFFEL_CROSS_CHECK)
        .expect("preset must load on wasm32");
    assert!(!inspector.export_datasets().is_empty());

    let fw = tab.framework_mut().expect("framework");
    fw.export_options.width = 320;
    fw.export_options.height = 200;
    for fmt in [ExportFormat::Png, ExportFormat::Svg, ExportFormat::Csv] {
        let _ = ctx.run(wasm_input(Vec::new()), |ctx| {
            egui_plot::clear_last_plot_datasets();
            fw.show_export_modal = true;
            fw.queue_export(fmt);
            fw.show(ctx, "wasm_relativity_export");
        });
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    }
}

async fn yield_wasm_tick() {
    use wasm_bindgen::JsCast;
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        if let Ok(win) = js_sys::global().dyn_into::<web_sys::Window>() {
            let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0);
        } else {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[wasm_bindgen_test]
async fn wasm_relativity_ast_sandbox_atlas_convergence_teleport_and_challenges() {
    assert!(!get_stories_for_domain("relativity").is_empty());
    assert!(get_goal_challenges_for_domain("relativity").len() >= 2);

    // Verify custom AST Christoffel parity on wasm32
    let mut inspector = ChristoffelInspectorTool::default();
    inspector.state.manifold = InspectorManifold::CustomRadialLapse4D;
    inspector.state.scale_radius = 2.0;
    inspector.set_custom_lapse_expression("1 - 2 / r");
    let ev = inspector
        .evaluate_current()
        .expect("wasm custom Christoffel");
    assert!(ev.max_abs_diff < 1e-6);

    // Verify 2D atlas converges to 1x via cooperative frame budget on wasm32
    let mut tool = GeodesicPlaygroundTool::default();
    tool.import_preset_json(presets::PRESET_PHOTON_RING_LENSING)
        .expect("photon ring preset");
    assert!((tool.launch_energy - 1.0).abs() < 1e-12);
    tool.atlas.set_visible(true);
    tool.state.atlas_visible = true;

    let ctx = egui::Context::default();
    for _ in 0..600 {
        let _ = ctx.run(wasm_input(Vec::new()), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
        });
        if tool.atlas.is_converged() {
            break;
        }
        yield_wasm_tick().await;
    }
    assert_eq!(tool.atlas.current_stride(), 1);

    // Teleport to photon-ring boundary and verify URL fragment round-trip
    let b_target = playground::kerr_prograde_critical_b(2.0, 0.0) + 0.02;
    tool.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "impact_or_momentum",
        x_val: b_target,
        y_key: "initial_radius",
        y_val: 11.5,
        regime_label: "Photon-Ring Winding / Rosette",
    });
    let url_json = tool.save_url_state().expect("url state");
    let frag = encode_share_fragment(
        "relativity",
        Some("GeodesicPlaygroundTool"),
        Some(&url_json),
    )
    .expect("encode share fragment");
    let decoded = parse_share_location(&frag, "")
        .expect("decode")
        .expect("payload");
    let mut hydrated = GeodesicPlaygroundTool::default();
    hydrated
        .try_load_state(decoded.state_json.as_deref().unwrap())
        .expect("load state");
    assert!((hydrated.state.impact_or_momentum - b_target).abs() < 1e-9);
}
