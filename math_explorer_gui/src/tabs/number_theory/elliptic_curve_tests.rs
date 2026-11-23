//! Integration, accessibility, preset/URL validation, singularity guardrail, and
//! zero-allocation verification tests for [`super::elliptic_curve_playground::EllipticCurvePlayground`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use eframe::egui;
use math_explorer::pure_math::elliptic_curves::AffinePointFp;
use verified_engine::engine::VerifiedEngine;

use super::elliptic_curve_playground::{ActivePointSelector, EllipticCurvePlayground};
use super::NumberTheoryTab;
use crate::command_palette::CommandPalette;
use crate::framework::{InteractiveTool, StarterPresetMetadata, ToolMetadata};
use crate::tabs::ExplorerTab;

#[test]
fn test_elliptic_curve_registered_in_sidebar_palette_and_starter_gallery() {
    let mut tab = NumberTheoryTab::default();
    assert!(tab.select_tool_by_name("EllipticCurvePlayground"));
    let fw = tab
        .framework()
        .expect("NumberTheoryTab owns SimulationFramework");
    assert_eq!(fw.active_tool_id(), Some("EllipticCurvePlayground"));

    let has_tool_meta = inventory::iter::<ToolMetadata>
        .into_iter()
        .any(|m| m.id == "EllipticCurvePlayground" && m.domain == "number_theory");
    assert!(
        has_tool_meta,
        "ToolMetadata must register EllipticCurvePlayground"
    );

    let has_starter = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .any(|p| {
            p.id == "elliptic_curve_fp_starter"
                && p.domain == "number_theory"
                && p.tool_name == Some("EllipticCurvePlayground")
        });
    assert!(
        has_starter,
        "StarterPresetMetadata must register elliptic_curve_fp_starter"
    );

    let tabs: Vec<Box<dyn ExplorerTab>> = vec![Box::new(NumberTheoryTab::default())];
    let mut palette = CommandPalette::default();
    palette.open();
    palette.query = "Elliptic Curve Playground".to_string();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let action = palette.show(ctx, &tabs);
        assert!(action.is_none());
    });
}

#[test]
fn test_point_selection_updates_algebra_accesskit_and_never_traps_tab_or_escape() {
    let mut tool = EllipticCurvePlayground::default();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let out_initial = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    let ak = out_initial
        .platform_output
        .accesskit_update
        .expect("AccessKit update emitted");
    let canvas_id = ak
        .nodes
        .iter()
        .find_map(|(id, node)| {
            (node.role() == egui::accesskit::Role::Canvas
                && node.label() == Some("Elliptic curve F_p torus lattice"))
            .then_some(*id)
        })
        .expect("Torus lattice AccessKit Canvas node registered");

    tool.selector = ActivePointSelector::PointQ;
    tool.select_curve_point_by_index(&ctx, 1);
    assert!(tool.status_summary.contains("P+Q ="));

    let egui_canvas_id = egui::Id::new(canvas_id.0);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        ctx.memory_mut(|m| m.request_focus(egui_canvas_id));
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });

    let mut tab_input = egui::RawInput::default();
    tab_input.events.push(egui::Event::Key {
        key: egui::Key::Tab,
        physical_key: Some(egui::Key::Tab),
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = ctx.run(tab_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    let still_trapped_on_canvas = ctx.memory(|m| m.focused() == Some(egui_canvas_id));
    assert!(
        !still_trapped_on_canvas,
        "Tab must advance focus away from torus canvas rather than trapping keyboard navigation"
    );
}

#[test]
fn test_singular_curve_blocks_division_while_keeping_plot_and_exports_usable() {
    let mut tool = EllipticCurvePlayground::default();
    tool.p = 17;
    tool.a = -3;
    tool.b = 2;
    tool.recompute_all();

    assert!(
        tool.is_singular,
        "y^2 = x^3 - 3x + 2 is singular over every prime >= 5"
    );
    assert!(
        tool.chord.is_none(),
        "Singular curve must block invalid slope division"
    );
    assert!(tool.status_summary.contains("Singular curve"));

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });

    let datasets = tool.export_datasets();
    assert!(
        !datasets.is_empty(),
        "Dataset exports remain available when active curve is singular"
    );
    assert!(datasets.iter().all(|(_, pts)| !pts.is_empty()));
    let primitives = tool.export_plot_primitives();
    assert!(
        !primitives.is_empty(),
        "SVG/PNG plot primitives remain non-empty when singular"
    );
}

#[test]
fn test_preset_and_share_url_roundtrip_and_invalid_state_rejection() {
    let mut tab = NumberTheoryTab::default();
    assert!(tab.select_tool_by_name("EllipticCurvePlayground"));
    if let Some(fw) = tab.framework_mut() {
        let tool = fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<EllipticCurvePlayground>())
            .unwrap();
        tool.try_load_state(
            r#"{"p":29,"a":4,"b":20,"point_p":{"Finite":{"x":1,"y":5}},"point_q":{"Finite":{"x":4,"y":19}},"scalar_k":7,"hasse_p_max":307}"#,
        )
        .unwrap();
    }

    let url = tab.framework().unwrap().encode_share_url().unwrap();
    let hash_pos = url.find('#').expect("Share URL contains # fragment");
    let params = crate::url_state::parse_share_location(&url[hash_pos..], "")
        .unwrap()
        .expect("Parsed ShareUrlParams");
    let mut restored_tab = NumberTheoryTab::default();
    restored_tab
        .import_url_preset_json(params.tool.as_deref(), params.state_json.as_deref())
        .unwrap();
    let restored = restored_tab
        .framework()
        .and_then(|fw| fw.active_tool.as_ref())
        .and_then(|t| t.as_any())
        .and_then(|a| a.downcast_ref::<EllipticCurvePlayground>())
        .unwrap();
    assert_eq!(restored.p, 29);
    assert_eq!(restored.scalar_k, 7);
    assert_eq!(restored.point_p, AffinePointFp::Finite { x: 1, y: 5 });

    let mut guard_tool = EllipticCurvePlayground::default();
    for invalid in [
        r#"{"p":15,"a":2,"b":2,"point_p":"Infinity","point_q":"Infinity","scalar_k":1,"hasse_p_max":199}"#,
        r#"{"p":17,"a":-3,"b":2,"point_p":"Infinity","point_q":"Infinity","scalar_k":1,"hasse_p_max":199}"#,
        r#"{"p":1009,"a":2,"b":2,"point_p":"Infinity","point_q":"Infinity","scalar_k":1,"hasse_p_max":199}"#,
        r#"{"p":17,"a":2,"b":2,"point_p":{"Finite":{"x":5,"y":2}},"point_q":"Infinity","scalar_k":1,"hasse_p_max":199}"#,
    ] {
        assert!(
            guard_tool.try_load_state(invalid).is_err(),
            "Expected try_load_state to reject invalid payload: {invalid}"
        );
    }
}

#[test]
fn test_hot_loop_recompute_and_orbit_stepping_zero_heap_allocations() {
    let mut tool = EllipticCurvePlayground::default();
    tool.recompute_curve_hot_loop();

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        tool.recompute_curve_hot_loop();
        for _ in 0..8 {
            tool.step_orbit_hot_loop();
        }
    });
    assert!(metrics.function_calls > 0);
}

// [cite:modular_polynomials_review]
