//! Unit tests for [`super::QuasicrystalStudio`]: presets, `#[serde(deny_unknown_fields)]`,
//! URL bounds, `FiniteStateGuard` rollback, zero-allocation stepping, AccessKit keyboard navigation,
//! and `.svg`, `.png`, `.csv`, and `.json` exports without modifying `CrystalViewer`.

// [cite:solid_state]

use std::collections::HashSet;

use eframe::egui;
use verified_engine::engine::VerifiedEngine;

use super::*;
use crate::app::{MathExplorerApp, ThemeMode};
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::solid_state::SolidStateTab;
use crate::tabs::ExplorerTab;

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
fn solid_state_tab_renders_quasicrystal_studio_across_themes_alongside_crystal_viewer() {
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut tab = SolidStateTab::default();

    for tool_id in ["CrystalViewer", "QuasicrystalStudio"] {
        assert!(
            tab.select_tool_by_name(tool_id),
            "SolidStateTab must discover {tool_id}"
        );
        for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
            theme.apply_to_context(&ctx);
            let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    tab.show_inside(ui, &mut frame);
                });
            });
        }
    }
}

#[test]
fn starter_presets_deny_unknown_fields_url_bounds_and_rollback_guard() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut studio = QuasicrystalStudio::default();

    for payload in [
        presets::PRESET_PENROSE_10FOLD,
        presets::PRESET_AMMANN_BEENKER_8FOLD,
        presets::PRESET_STAMPFLI_12FOLD,
        presets::PRESET_HEXAGONAL_BRAVAIS_BZ,
    ] {
        studio
            .import_preset_json(payload)
            .expect("starter preset must load cleanly");
    }
    assert_eq!(studio.state.mode, StudioPatternMode::PeriodicBravais);
    assert_eq!(studio.state.bravais, BravaisLattice2D::Hexagonal);

    let unknown_field_json = r#"{"mode":"Quasicrystal","unexpected_key":42}"#;
    assert!(
        studio.try_load_state(unknown_field_json).is_err(),
        "#[serde(deny_unknown_fields)] must reject unknown JSON keys"
    );

    let saved = studio.save_state().expect("serializable state");
    assert!(
        saved.len() <= crate::url_state::MAX_DECODED_URL_STATE_BYTES,
        "State JSON ({} bytes) must fit inside URL_JSON_BOUNDS",
        saved.len()
    );
    assert!(studio.state_bounds().validate_json(&saved).is_ok());

    let mut app = MathExplorerApp::default();
    assert!(app
        .apply_share_location_strings(
            "#tab=solid-state-physics&tool=QuasicrystalStudio&preset=solid_state_penrose_10fold",
            "",
            None,
        )
        .unwrap());

    studio.state.edge_length = f64::NAN;
    studio.state.phason_x = f64::INFINITY;
    assert!(!studio.has_finite_parameters());
    assert!(studio.clamp_parameters());
    assert!(studio.has_finite_parameters());
}

#[test]
fn studio_phason_step_runs_with_zero_heap_allocations_under_verified_allocator() {
    let mut studio = QuasicrystalStudio::default();
    let _alloc_lock = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let ((flipped, c10), metrics) = VerifiedEngine::run_verified(|| {
        studio.step_phason_zero_alloc([0.025, -0.015]);
        (
            studio.qc_workspace.stats().flipped_tile_count,
            studio.fft_workspace.report().correlation(10),
        )
    });

    assert!(metrics.function_calls > 0);
    assert!(flipped > 0, "Phason step must flip tiles");
    assert!(c10 >= 0.99, "Penrose C_10 = {c10:.5} must remain >= 0.99");
}

fn collect_direct_canvas_nodes(
    ctx: &egui::Context,
    studio: &mut QuasicrystalStudio,
) -> Vec<egui::accesskit::NodeId> {
    let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| studio.show_inside(ui));
    });
    let out = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| studio.show_inside(ui));
    });
    out.platform_output
        .accesskit_update
        .expect("AccessKit update must be emitted")
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            (node.role() == egui::accesskit::Role::GraphicsDocument
                && node
                    .label()
                    .unwrap_or_default()
                    .contains("Direct-Space 2D Tiling and Phason Flip Canvas")
                && node.supports_action(egui::accesskit::Action::Focus))
            .then_some(*id)
        })
        .collect()
}

#[test]
fn direct_space_canvas_supports_arrow_key_phason_shifts_without_trapping_tab() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut studio = QuasicrystalStudio::default();
    let canvas_nodes = collect_direct_canvas_nodes(&ctx, &mut studio);
    assert!(
        !canvas_nodes.is_empty(),
        "Focusable AccessKit canvas node missing"
    );

    let key_frame = |key: egui::Key| {
        test_raw_input(vec![egui::Event::Key {
            key,
            physical_key: Some(key),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }])
    };

    let mut reached_canvas = false;
    for _ in 0..80 {
        let step = ctx.run(key_frame(egui::Key::Tab), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| studio.show_inside(ui));
        });
        if step
            .platform_output
            .accesskit_update
            .is_some_and(|u| canvas_nodes.contains(&u.focus))
        {
            reached_canvas = true;
            break;
        }
    }
    assert!(
        reached_canvas,
        "Direct-space canvas must be reachable via Tab"
    );
    let focused_id = ctx.memory(|m| m.focused());

    let (wx0, wy0) = (studio.state.phason_x, studio.state.phason_y);
    let _ = ctx.run(key_frame(egui::Key::ArrowRight), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| studio.show_inside(ui));
    });
    let _ = ctx.run(key_frame(egui::Key::ArrowUp), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| studio.show_inside(ui));
    });
    assert!(studio.state.phason_x > wx0 && studio.state.phason_y > wy0);

    let _ = ctx.run(key_frame(egui::Key::Tab), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| studio.show_inside(ui));
    });
    assert_ne!(ctx.memory(|m| m.focused()), focused_id);
}

#[test]
fn quasicrystal_studio_exports_valid_png_svg_csv_and_json_artifacts() {
    let mut tab = SolidStateTab::default();
    assert!(tab.select_tool_by_name("QuasicrystalStudio"));
    let fw = tab.framework_mut().unwrap();
    fw.export_options.width = 256;
    fw.export_options.height = 256;

    for format in [
        ExportFormat::Png,
        ExportFormat::Svg,
        ExportFormat::Csv,
        ExportFormat::Json,
    ] {
        let ctx = egui::Context::default();
        let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
            egui_plot::clear_last_plot_datasets();
            fw.show_export_modal = true;
            fw.queue_export(format);
            fw.show(ctx, "quasicrystal_export_test");
        });
        assert_eq!(
            fw.export_status_severity,
            ExportStatusSeverity::Success,
            "{format:?} export failed: {:?}",
            fw.export_status_message
        );
        match format {
            ExportFormat::Png => {
                let bytes = fw.export_queue.last_exported_bytes.as_deref().unwrap();
                let px =
                    crate::framework::deferred_export_tests::decode_rgba_pixels(bytes).unwrap();
                let unique: HashSet<[u8; 4]> = px.into_iter().collect();
                assert!(unique.len() >= 8);
            }
            ExportFormat::Svg => {
                let svg = fw.export_queue.last_exported_content.as_deref().unwrap();
                assert!(svg.contains("<svg") && svg.len() > 250);
            }
            ExportFormat::Csv => {
                let csv = fw.export_queue.last_exported_content.as_deref().unwrap();
                assert!(csv.lines().count() > 50);
            }
            ExportFormat::Json => {
                let json = fw.export_queue.last_exported_content.as_deref().unwrap();
                let val: serde_json::Value = serde_json::from_str(json).unwrap();
                assert!(val.is_object() || val.is_array());
            }
            _ => {}
        }
    }
}

#[test]
fn diffraction_colormaps_and_overlays_satisfy_wcag_3_to_1_contrast_on_black() {
    let srgb_to_lin = |c: u8| -> f64 {
        let s = f64::from(c) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    let contrast_on_black = |col: egui::Color32| -> f64 {
        let lum = 0.2126 * srgb_to_lin(col.r())
            + 0.7152 * srgb_to_lin(col.g())
            + 0.0722 * srgb_to_lin(col.b());
        (lum + 0.05) / 0.05
    };

    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        for cm in [
            DiffractionColormap::Inferno,
            DiffractionColormap::Viridis,
            DiffractionColormap::Monochrome,
        ] {
            for step in 5..=100 {
                let t = f64::from(step) / 100.0;
                let px = canvas::map_diffraction_color(t, cm, theme);
                assert!(
                    contrast_on_black(px) >= 3.0,
                    "{cm:?} ({theme:?}) at t={t:.2} yielded contrast < 3.0:1"
                );
            }
        }
    }
    let lifted = canvas::lift_dark_overlay(egui::Color32::from_rgb(0, 0, 180));
    assert!(contrast_on_black(lifted) >= 3.0);
}

#[test]
fn export_capabilities_and_code_artifact_contract() {
    let studio = QuasicrystalStudio::default();
    let caps = studio.export_capabilities();
    assert!(caps.dataset);
    assert!(caps.figure);
    assert!(!caps.code);
    assert!(!caps.code_disabled_reason.is_empty());
    assert!(!caps.shader);
    assert!(!caps.shader_disabled_reason.is_empty());
    assert!(studio.export_code_artifact().is_none());
}

fn collect_shape_texts(shape: &egui::epaint::Shape, out: &mut Vec<String>) {
    match shape {
        egui::epaint::Shape::Text(t) => out.push(t.galley.text().to_string()),
        egui::epaint::Shape::Vec(items) => {
            for s in items {
                collect_shape_texts(s, out);
            }
        }
        _ => {}
    }
}

#[test]
fn quasicrystal_studio_resolves_maturity_tier_and_renders_accessible_badges() {
    use crate::framework::maturity_badge::resolve_tool_maturity_tier;
    use scientific_metadata::theory::MaturityTier;

    let studio = QuasicrystalStudio::default();
    assert_eq!(studio.theory().maturity_tier(), MaturityTier::Experimental);
    assert_eq!(
        resolve_tool_maturity_tier(
            "QuasicrystalStudio",
            "solid_state",
            Some(studio.theory().maturity_tier()),
        ),
        MaturityTier::Experimental
    );

    let mut tab = SolidStateTab::default();
    assert!(tab.select_tool_by_name("QuasicrystalStudio"));
    tab.framework_mut().unwrap().show_theory_portal = true;

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut frame = eframe::Frame::_new_kittest();
    let out = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });

    let mut texts = Vec::new();
    for cs in &out.shapes {
        collect_shape_texts(&cs.shape, &mut texts);
    }
    assert!(texts.iter().any(|t| t.contains("[EXP]")));
    assert!(texts.iter().any(|t| t.contains("[EXPERIMENTAL]")));
}
