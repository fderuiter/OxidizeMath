#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Desktop unit tests for [`RelativityTab`], [`GeodesicPlaygroundTool`], [`ChristoffelInspectorTool`],
//! `Orbit`/`Slingshot`/`PhotonFan` modes, exact Kerr observables, 2D cross-domain lensing,
//! 1,000-step zero-allocation execution, `.wav`/`.frag` exports, and `#tab=relativity` URL sharing.

use std::collections::HashSet;

use eframe::egui;
use verified_engine::engine::VerifiedEngine;

use super::*;
use crate::app::{MathExplorerApp, ThemeMode};
use crate::audio::wav::synthesize_frames_to_wav;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool, ShaderDialect};
use crate::tabs::ExplorerTab;
use crate::url_state::URL_JSON_BOUNDS;

fn test_raw_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        events,
        ..Default::default()
    }
}

#[test]
fn relativity_tab_renders_both_tools_and_2d_lensing_across_all_three_themes() {
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut tab = RelativityTab::default();
    assert_eq!(tab.name(), "General Relativity");
    assert_eq!(tab.academic_domain(), "Physics");
    assert_eq!(tab.url_slug(), "relativity");

    for tool_id in ["GeodesicPlaygroundTool", "ChristoffelInspectorTool"] {
        assert!(tab.select_tool_by_name(tool_id));
        for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
            theme.apply_to_context(&ctx);
            let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    tab.show_inside(ui, &mut frame);
                });
            });
        }
    }

    let mut lens_tool = GeodesicPlaygroundTool::default();
    lens_tool.state.view_mode = ViewportViewMode::LensingCamera2D;
    for bg in [
        LensingBackgroundKind::GalacticDisk,
        LensingBackgroundKind::LeniaPortal,
        LensingBackgroundKind::Starfield,
    ] {
        lens_tool.state.lensing_background = bg;
        lens_tool.lensing_dirty = true;
        let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| lens_tool.show_inside(ui));
        });
    }
}

#[test]
fn geodesic_playground_camera3d_orbit_vs_slingshot_rotation_lock() {
    let ctx = egui::Context::default();
    let mut tool = GeodesicPlaygroundTool::default();
    let initial_cam = tool.state.camera;

    let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });

    let center = egui::pos2(780.0, 320.0);
    let zoom_events = vec![
        egui::Event::PointerMoved(center),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, 2.0),
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(test_raw_input(zoom_events), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    assert!((tool.state.camera.zoom - initial_cam.zoom).abs() > 1e-4);

    tool.state.interaction_mode = GeodesicInteractionMode::PhotonFan;
    tool.state.fan_count = MAX_BUNDLE_RAYS;
    tool.rebuild_trajectory(180);
    assert_eq!(tool.active_ray_count, 16);
    assert!(tool.locks_camera_orbit());
    let locked_cam = tool.state.camera;

    let drag_events = vec![
        egui::Event::PointerMoved(center),
        egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerMoved(egui::pos2(830.0, 355.0)),
    ];
    let _ = ctx.run(test_raw_input(drag_events), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    assert_eq!(tool.state.camera, locked_cam);
}

#[test]
fn geodesic_playground_keyboard_arrow_and_enter_slingshot_launch() {
    let ctx = egui::Context::default();
    let mut tool = GeodesicPlaygroundTool::default();
    let b_before = tool.state.impact_or_momentum;
    let key_events = vec![
        egui::Event::Key {
            key: egui::Key::ArrowRight,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
        assert!(!tool.handle_keyboard_slingshot(ctx, false));
    });
    let _ = ctx.run(test_raw_input(key_events), |ctx| {
        assert!(tool.handle_keyboard_slingshot(ctx, true));
    });
    assert!((tool.state.impact_or_momentum - b_before).abs() > 1e-3);
    assert!(tool.state.is_playing);
}

#[test]
fn geodesic_playground_accesskit_meter_image_and_canvas_focus() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = GeodesicPlaygroundTool::default();
    tool.state.view_mode = ViewportViewMode::LensingCamera2D;
    let out = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit tree");
    assert!(ak
        .nodes
        .iter()
        .any(|(_, n)| n.role() == egui::accesskit::Role::Meter));
    assert!(ak
        .nodes
        .iter()
        .any(|(_, n)| n.role() == egui::accesskit::Role::Image));

    let center = egui::pos2(780.0, 320.0);
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
    let _ = ctx.run(test_raw_input(click_events), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    assert!(tool.focused_canvas_id().is_some());
}

#[test]
fn exact_kerr_observables_lensing_shaders_and_1000_step_zero_alloc() {
    let mut tool = GeodesicPlaygroundTool::default();
    tool.state.spacetime = SpacetimePreset::KerrEquatorial;
    tool.state.mode = GeodesicMode::TimelikeParticle;
    tool.state.spin_chi = 0.85;
    tool.state.initial_radius = 4.2;
    tool.state.impact_or_momentum = 3.6;
    tool.lensing_dirty = true;
    tool.rebuild_trajectory(120);
    tool.refresh_lensing_pixels_in_place();

    let (b_neg, b_pos) = tool.deflection_table.critical_impact_bounds();
    assert!(
        b_pos < b_neg.abs() - 0.5,
        "Spinning Kerr shadow must be asymmetric: prograde {b_pos} < |retrograde| {}",
        b_neg.abs()
    );

    let img = tool.export_raster_image().expect("lensing raster image");
    let unique_px: HashSet<egui::Color32> = img.pixels.iter().copied().collect();
    assert!(unique_px.len() > 24, "Lensed galaxy image must not be flat");

    let webgl2 = tool.export_glsl_shader(ShaderDialect::WebGl2Es300).unwrap();
    let stoy = tool.export_glsl_shader(ShaderDialect::Shadertoy).unwrap();
    assert!(webgl2.starts_with("#version 300 es") && webgl2.contains("void main()"));
    assert!(!stoy.contains("#version") && stoy.contains("void mainImage("));

    let wav = synthesize_frames_to_wav(&[tool.build_audio_frame()], 44_100, 128, 0.8);
    assert_eq!(&wav[0..4], b"RIFF");

    // 1,000-step VerifiedEngine zero-allocation verification over ray bundle + lensing lookup + audio
    tool.state.interaction_mode = GeodesicInteractionMode::PhotonFan;
    tool.state.fan_count = 4;
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (acc, _) = VerifiedEngine::run_verified(|| {
            let mut sum = 0.0_f64;
            for step in 0..1_000usize {
                tool.rebuild_trajectory(6);
                let hit = tool
                    .deflection_table
                    .map_image_uv((step % 20) as f64 * 0.04, 0.25, 10.0);
                let af = tool.build_audio_frame();
                sum += hit.source_u + f64::from(af.voices[0].freq_hz);
            }
            sum
        });
        assert!(acc.is_finite());
    }
}

#[test]
fn relativity_presets_url_bounds_hydration_and_finite_state_guard_rollback() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings("#tab=relativity&tool=GeodesicPlaygroundTool", "", None)
        .unwrap());
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "relativity");

    let mut playground = GeodesicPlaygroundTool::default();
    for payload in [
        presets::PRESET_PHOTON_RING_LENSING,
        presets::PRESET_ISCO_ROSETTE_CLOCKS,
        presets::PRESET_WORMHOLE_TRANSIT,
        presets::PRESET_KERR_PHOTON_FAN_SHADOW,
        presets::PRESET_EINSTEIN_RING_GALAXY_LENSING,
        presets::PRESET_WORMHOLE_LENIA_PORTAL,
    ] {
        playground.import_preset_json(payload).unwrap();
        assert!(!playground.samples.is_empty());
        let saved = playground.save_state().unwrap();
        URL_JSON_BOUNDS.validate_json(&saved).unwrap();
        playground.try_load_state(&saved).unwrap();
    }
    assert_eq!(
        playground.state.spacetime,
        SpacetimePreset::MorrisThorneWormhole
    );

    playground.state.scale_radius = f64::NAN;
    assert!(!playground.has_finite_parameters());
    assert!(playground.clamp_parameters());
    assert!(playground.has_finite_parameters());

    let mut inspector = ChristoffelInspectorTool::default();
    inspector
        .import_preset_json(presets::PRESET_CHRISTOFFEL_CROSS_CHECK)
        .unwrap();
    let eval = inspector
        .evaluate_current()
        .expect("valid Kerr Christoffel");
    assert!(eval.max_abs_diff < 1e-6);
}

#[test]
fn relativity_tools_export_valid_png_svg_csv_and_glsl_artifacts() {
    for tool_id in ["GeodesicPlaygroundTool", "ChristoffelInspectorTool"] {
        let mut tab = RelativityTab::default();
        assert!(tab.select_tool_by_name(tool_id));
        let fw = tab.framework_mut().unwrap();
        fw.export_options.width = 360;
        fw.export_options.height = 240;

        for format in [ExportFormat::Png, ExportFormat::Svg, ExportFormat::Csv] {
            let ctx = egui::Context::default();
            let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
                egui_plot::clear_last_plot_datasets();
                fw.show_export_modal = true;
                fw.queue_export(format);
                fw.show(ctx, "relativity_export_test");
            });
            assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
            if format == ExportFormat::Png {
                let bytes = fw.export_queue.last_exported_bytes.as_deref().unwrap();
                let px =
                    crate::framework::deferred_export_tests::decode_rgba_pixels(bytes).unwrap();
                let colors: HashSet<[u8; 4]> = px.into_iter().collect();
                assert!(colors.len() >= 2);
            }
        }
    }
}

#[test]
fn flagship_lensing_and_slingshot_png_exports_have_distinct_palettes() {
    for (title, preset, out_path) in [
        (
            "Einstein-Ring Galaxy Lensing",
            presets::PRESET_EINSTEIN_RING_GALAXY_LENSING,
            "/tmp/einstein_ring_galaxy_lensing.png",
        ),
        (
            "Wormhole Lenia Soliton Portal",
            presets::PRESET_WORMHOLE_LENIA_PORTAL,
            "/tmp/wormhole_lenia_portal.png",
        ),
        (
            "Kerr Photon Fan & Frame-Dragged Shadow",
            presets::PRESET_KERR_PHOTON_FAN_SHADOW,
            "/tmp/kerr_photon_fan_3d.png",
        ),
    ] {
        let mut tab = RelativityTab::default();
        let fw = tab.framework_mut().unwrap();
        fw.apply_starter_card(title, Some("GeodesicPlaygroundTool"), Some(preset))
            .unwrap();
        fw.export_options.width = 960;
        fw.export_options.height = 600;
        let ctx = egui::Context::default();
        let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
            egui_plot::clear_last_plot_datasets();
            fw.show_export_modal = true;
            fw.queue_export(ExportFormat::Png);
            fw.show(ctx, "flagship_relativity_png");
        });
        let png = fw.export_queue.last_exported_bytes.as_deref().unwrap();
        let px = crate::framework::deferred_export_tests::decode_rgba_pixels(png).unwrap();
        let colors: HashSet<[u8; 4]> = px.into_iter().collect();
        assert!(
            colors.len() >= 8,
            "Expected rich PNG palette for {out_path}"
        );
        let _ = std::fs::write(out_path, png);
    }
}

#[test]
fn flagship_galactic_slingshot_png_export_has_distinct_palette() {
    let mut gal =
        crate::tabs::astrophysics::galactic_orbits_tool::GalacticOrbitSimulator::default();
    gal.launch_slingshot_probe(-2.2, -1.4, 125.0, 175.0);
    gal.launch_slingshot_probe(2.4, 1.1, -140.0, 155.0);
    for _ in 0..24 {
        gal.advance_burst();
    }
    let state_json = gal.save_state().unwrap();
    let mut astro = crate::tabs::astrophysics::AstrophysicsTab::default();
    assert!(astro.select_tool_by_name("GalacticOrbitSimulator"));
    let fw = astro.framework_mut().unwrap();
    fw.active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&state_json)
        .unwrap();
    fw.export_options.width = 960;
    fw.export_options.height = 600;
    let ctx = egui::Context::default();
    let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui_plot::clear_last_plot_datasets();
        fw.show_export_modal = true;
        fw.queue_export(ExportFormat::Png);
        fw.show(ctx, "flagship_galactic_png");
    });
    let gal_png = fw.export_queue.last_exported_bytes.as_deref().unwrap();
    let gal_px = crate::framework::deferred_export_tests::decode_rgba_pixels(gal_png).unwrap();
    let gal_colors: HashSet<[u8; 4]> = gal_px.into_iter().collect();
    assert!(gal_colors.len() >= 4);
    let _ = std::fs::write("/tmp/galactic_slingshot_orbits.png", gal_png);
}
