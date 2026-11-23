#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Desktop unit tests for 2D gravitational lensing, Kerr shadow asymmetry, 1,000-step
//! zero-allocation ray bundles + audio frames, `.wav`/`.frag` exports, and flagship PNGs.

use std::collections::HashSet;

use eframe::egui;
use verified_engine::engine::VerifiedEngine;

use super::tests::test_raw_input;
use super::*;
use crate::audio::wav::synthesize_frames_to_wav;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool, ShaderDialect};
use crate::tabs::ExplorerTab;

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
