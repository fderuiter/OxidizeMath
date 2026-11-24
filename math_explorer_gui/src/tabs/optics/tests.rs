#![cfg_attr(any(), verified(opt_out = "tests"))]

use eframe::egui;
use math_explorer::verified_engine::engine::VerifiedEngine;

use super::diffraction_tool::FraunhoferDiffractionTool;
use super::ray_bench_tool::CausticsRayBenchTool;
use super::OpticsTab;
use crate::app::MathExplorerApp;
use crate::framework::{ExportFormat, FiniteStateGuard, InteractiveTool, SimulationFramework};
use crate::starter_gallery::StarterPresetMetadata;
use crate::tabs::ExplorerTab;

#[test]
fn optics_tab_metadata_and_four_starter_presets_roundtrip() {
    let tab = OpticsTab::default();
    assert_eq!(tab.name(), "Optics & Caustics");
    assert_eq!(tab.academic_domain(), "Physics");
    assert_eq!(tab.url_slug(), "optics");

    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| p.domain == "optics")
        .collect();
    assert_eq!(presets.len(), 4);
    for expected_title in [
        "Coffee-Cup Nephroid",
        "Luneburg Metamaterial",
        "Dispersive Prism",
        "Fraunhofer Grating",
    ] {
        assert!(
            presets.iter().any(|p| p.title == expected_title),
            "Missing optics preset '{expected_title}'"
        );
    }
}

#[test]
fn optics_url_hash_roundtrip_and_state_bounds_guard() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut src = MathExplorerApp::default();
    let idx = src
        .tabs
        .iter()
        .position(|t| t.url_slug() == "optics")
        .expect("optics tab auto-discovered");
    src.selected_tab = idx;
    assert!(src.tabs[idx].select_tool_by_name("FraunhoferDiffractionTool"));
    let share_url = src.encode_current_share_url().expect("encode #tab=optics");
    assert!(share_url.contains("tab=optics"));

    let hash = share_url.split('#').nth(1).unwrap_or(&share_url);
    let mut dst = MathExplorerApp::default();
    assert!(dst.apply_share_location_strings(hash, "", None).unwrap());
    assert_eq!(dst.tabs[dst.selected_tab].url_slug(), "optics");
    assert_eq!(
        dst.tabs[dst.selected_tab].active_tool_id(),
        Some("FraunhoferDiffractionTool")
    );

    let mut bench = CausticsRayBenchTool::default();
    assert!(bench.try_load_state(r#"{"unknown":1}"#).is_err());
    bench.state.wavelength_nm = f64::NAN;
    assert!(!bench.has_finite_parameters());
    assert!(InteractiveTool::clamp_parameters(&mut bench));
    assert!(bench.has_finite_parameters());

    let mut diff = FraunhoferDiffractionTool::default();
    assert!(diff.try_load_state(r#"{"wavelength_nm":99999.0}"#).is_err());
    diff.state.slit_width_um = f64::INFINITY;
    assert!(!diff.has_finite_parameters());
    FiniteStateGuard::clamp_parameters(&mut diff);
    assert!(diff.has_finite_parameters());
}

#[test]
fn optics_tools_export_svg_png_csv_json_and_zero_alloc_stepping() {
    for tool_id in ["CausticsRayBenchTool", "FraunhoferDiffractionTool"] {
        let mut fw = SimulationFramework::new("optics");
        assert!(fw.select_tool_by_name(tool_id));
        fw.export_options.width = 240;
        fw.export_options.height = 180;

        for fmt in [
            ExportFormat::Png,
            ExportFormat::Svg,
            ExportFormat::Csv,
            ExportFormat::Json,
        ] {
            let ctx = egui::Context::default();
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(960.0, 720.0),
                )),
                ..Default::default()
            };
            let _ = ctx.run(raw, |ctx| {
                fw.show_export_modal = true;
                fw.queue_export(fmt);
                fw.show(ctx, "optics_test_export");
            });
            assert_eq!(
                fw.export_status_severity,
                crate::framework::ExportStatusSeverity::Success,
                "{tool_id} export {fmt:?} failed: {:?}",
                fw.export_status_message
            );
        }
    }

    let mut bench = CausticsRayBenchTool::default();
    let cfg = bench.ray_fan_config();
    let mut diff = FraunhoferDiffractionTool::default();
    let dcfg = diff.fraunhofer_config();
    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (_, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..100 {
            math_explorer::physics::optics::trace_ray_fan(&cfg, &mut bench.workspace);
            math_explorer::physics::optics::extract_caustic_envelope(
                &bench.workspace,
                bench.state.detector_x,
                2.6,
                &mut bench.caustics,
            );
            math_explorer::physics::optics::sample_fraunhofer_profile(
                &dcfg,
                241,
                &mut diff.profile,
            );
        }
    });
    assert!(metrics.function_calls >= 300);
}

#[test]
fn optics_canvas_registers_accesskit_graphics_document() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tab = OpticsTab::default();
    assert!(tab.select_tool_by_name("CausticsRayBenchTool"));
    let mut frame = eframe::Frame::_new_kittest();
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1024.0, 768.0),
            )),
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
        },
    );
    let ak = out
        .platform_output
        .accesskit_update
        .expect("accesskit tree");
    assert!(ak.nodes.iter().any(|(_, n)| {
        n.role() == egui::accesskit::Role::GraphicsDocument
            && n.label().unwrap_or("").contains("Optical Ray Bench")
    }));
}

fn raw_key(rect: egui::Rect, key: egui::Key) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(rect),
        events: vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    }
}

#[test]
fn optics_canvases_handle_keyboard_controls_without_focus_trap() {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1024.0, 768.0));
    let mut bench = CausticsRayBenchTool::default();
    let mut diff = FraunhoferDiffractionTool::default();
    let tools: [(&mut dyn InteractiveTool, &str); 2] = [
        (&mut bench, "Optical Ray Bench"),
        (&mut diff, "Fraunhofer 2D Far-Field"),
    ];
    for (tool, expected_label) in tools {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let before = tool.save_state().expect("initial state");
        let out1 = ctx.run(
            egui::RawInput {
                screen_rect: Some(rect),
                ..Default::default()
            },
            |ctx| tool.show(ctx),
        );
        let canvas_nid = out1
            .platform_output
            .accesskit_update
            .expect("ak tree")
            .nodes
            .iter()
            .find(|(_, n)| {
                n.role() == egui::accesskit::Role::GraphicsDocument
                    && n.label().unwrap_or("").contains(expected_label)
            })
            .map(|(id, _)| *id)
            .expect("canvas node");

        let mut reached = false;
        for _ in 0..80 {
            let step = ctx.run(raw_key(rect, egui::Key::Tab), |ctx| tool.show(ctx));
            if step
                .platform_output
                .accesskit_update
                .is_some_and(|u| u.focus == canvas_nid)
            {
                reached = true;
                break;
            }
        }
        assert!(reached, "{expected_label} must be reachable via Key::Tab");
        let _ = ctx.run(raw_key(rect, egui::Key::ArrowRight), |ctx| tool.show(ctx));
        let _ = ctx.run(raw_key(rect, egui::Key::ArrowUp), |ctx| tool.show(ctx));
        let _ = ctx.run(raw_key(rect, egui::Key::Enter), |ctx| tool.show(ctx));
        assert_ne!(before, tool.save_state().expect("updated state"));

        let exit_out = ctx.run(raw_key(rect, egui::Key::Tab), |ctx| tool.show(ctx));
        if let Some(u) = exit_out.platform_output.accesskit_update {
            assert_ne!(u.focus, canvas_nid, "Key::Tab must not trap focus");
        }
    }
}

fn apply_optics_preset_and_save(
    tab: &mut dyn ExplorerTab,
    tool_id: &str,
    preset_title: &str,
) -> String {
    assert!(tab.select_tool_by_name(tool_id));
    let preset = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .find(|p| p.domain == "optics" && p.title == preset_title)
        .unwrap_or_else(|| panic!("Missing optics preset {preset_title}"));
    tab.framework_mut()
        .expect("optics framework")
        .apply_starter_card(
            preset.title,
            preset.tool_name,
            Some(preset.generate_preset()),
        )
        .expect("apply optics preset");
    tab.framework()
        .and_then(|fw| fw.active_tool.as_ref())
        .and_then(|t| t.save_state())
        .expect("optics tool state")
}

#[test]
fn optics_multi_pane_workspace_url_roundtrip_preserves_split_tools() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut src = MathExplorerApp::default();
    let idx = src
        .tabs
        .iter()
        .position(|t| t.url_slug() == "optics")
        .expect("optics tab registered");

    src.set_layout(crate::app::ViewportLayout::SplitHorizontal);
    src.set_pane_tab(0, idx);
    src.set_pane_tab(1, idx);
    src.split_ratio = 0.58;
    src.focused_pane = 1;

    let expected_p0 = apply_optics_preset_and_save(
        src.tabs[idx].as_mut(),
        "CausticsRayBenchTool",
        "Dispersive Prism",
    );
    let expected_p1 = apply_optics_preset_and_save(
        src.pane_tab_mut(1).expect("pane 1 active").as_mut(),
        "FraunhoferDiffractionTool",
        "Fraunhofer Grating",
    );
    assert_ne!(expected_p0, expected_p1);

    let frag = src.encode_current_share_fragment().expect("encode #ws=");
    assert!(frag.starts_with("#ws=") && frag.len() <= crate::url_state::MAX_URL_FRAGMENT_BYTES);

    let mut dst = MathExplorerApp::default();
    assert!(dst.apply_share_location_strings(&frag, "", None).unwrap());
    assert_eq!(
        dst.viewport_layout,
        crate::app::ViewportLayout::SplitHorizontal
    );
    assert_eq!(dst.focused_pane, 1);
    assert_eq!(
        dst.tabs[dst.selected_tab].active_tool_id(),
        Some("CausticsRayBenchTool")
    );
    assert_eq!(
        dst.pane_tab(1).and_then(|t| t.active_tool_id()),
        Some("FraunhoferDiffractionTool")
    );
    let actual_p0 = dst.tabs[dst.selected_tab]
        .framework()
        .and_then(|fw| fw.active_tool.as_ref())
        .and_then(|t| t.save_state());
    let actual_p1 = dst
        .pane_tab(1)
        .and_then(|t| t.framework())
        .and_then(|fw| fw.active_tool.as_ref())
        .and_then(|t| t.save_state());
    assert_eq!(actual_p0.as_deref(), Some(expected_p0.as_str()));
    assert_eq!(actual_p1.as_deref(), Some(expected_p1.as_str()));
}
