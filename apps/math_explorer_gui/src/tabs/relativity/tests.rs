#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Desktop unit tests for [`RelativityTab`], [`GeodesicPlaygroundTool`], [`ChristoffelInspectorTool`],
//! `Orbit`/`Slingshot`/`PhotonFan` modes, photon-ring normalization, Kerr/ISCO geometry,
//! custom AST metrics, AccessKit semantics, and `#tab=relativity` URL sharing.

use std::collections::HashMap;

use eframe::egui;

use super::*;
use crate::app::{MathExplorerApp, ThemeMode};
use crate::framework::InteractiveTool;
use crate::tabs::ExplorerTab;
use crate::url_state::URL_JSON_BOUNDS;

pub(super) fn test_raw_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        events,
        ..Default::default()
    }
}

fn obs_map(tool: &dyn InteractiveTool) -> HashMap<&'static str, f64> {
    tool.observables().into_iter().collect()
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
fn photon_ring_preset_normalizes_energy_grazes_photon_sphere_and_captures_twin() {
    let mut tool = GeodesicPlaygroundTool::default();
    tool.import_preset_json(presets::PRESET_PHOTON_RING_LENSING)
        .expect("photon ring preset must load");
    assert!((tool.launch_energy - 1.0).abs() < 1e-12);
    let obs = obs_map(&tool);
    assert!((obs["launch_energy"] - 1.0).abs() < 1e-12);
    assert!((obs["effective_energy"] - 0.5).abs() < 1e-12);
    assert!(
        (3.0..=3.3).contains(&obs["r_min"]),
        "expected r_min in [3.0, 3.3], got {}",
        obs["r_min"]
    );
    assert!(obs["captured"] < 0.5, "primary ray should escape");
    assert!(
        obs["twin_captured"] > 0.5,
        "twin ray with negative delta_b must capture"
    );
    assert!(
        obs["twin_r_min"] <= 2.05,
        "captured twin must reach r <= 2.05, got {}",
        obs["twin_r_min"]
    );

    let datasets = tool.export_datasets();
    let e_line = datasets
        .iter()
        .find(|(n, _)| n == "0.5 * e^2")
        .expect("0.5 * e^2 dataset");
    assert!((e_line.1[0][1] - 0.5).abs() < 1e-12);
}

#[test]
fn kerr_equatorial_spin_09_surfaces_exact_horizons_frame_dragging_and_isco() {
    let mut tool = GeodesicPlaygroundTool::default();
    tool.state.spacetime = SpacetimePreset::KerrEquatorial;
    tool.state.scale_radius = 2.0;
    tool.state.spin_chi = 0.9;
    tool.state.initial_radius = 9.0;
    tool.state.impact_or_momentum = 3.1;
    tool.clamp_parameters();

    let obs = obs_map(&tool);
    assert!(
        (obs["outer_horizon_radius"] - 1.43589).abs() < 1e-5,
        "expected r_+ ≈ 1.43589, got {}",
        obs["outer_horizon_radius"]
    );
    assert!((obs["ergosphere_radius"] - 2.0).abs() < 1e-12);
    assert!(obs["frame_dragging_omega"] > 0.0 && obs["frame_dragging_omega"].is_finite());
    assert!(obs["isco"] > 2.0 && obs["isco"] < 6.0);

    tool.state.mode = playground::GeodesicMode::TimelikeParticle;
    tool.state.initial_radius = 1.8; // Inside the ergosphere (r_+ = 1.436 < 1.8 < r_ergo = 2.0)
    tool.state.impact_or_momentum = 3.6;
    tool.rebuild_trajectory(120);
    assert!((tool.samples[0].metric_norm + 1.0).abs() < 1e-9);

    let ctx = egui::Context::default();
    let _ = ctx.run(test_raw_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
}

#[test]
fn custom_ast_lapse_matches_schwarzschild_christoffel_and_steps_720_zero_alloc() {
    let mut inspector = ChristoffelInspectorTool::default();
    inspector.state.manifold = InspectorManifold::CustomRadialLapse4D;
    inspector.state.scale_radius = 2.0;
    inspector.state.radial_coord = 4.5;
    inspector.set_custom_lapse_expression("1 - 2 / r");
    let eval = inspector
        .evaluate_current()
        .expect("custom 1 - 2 / r Christoffel");
    assert!(
        eval.max_abs_diff < 1e-6,
        "1 - 2/r central diff must match Schwarzschild within 1e-6, got {}",
        eval.max_abs_diff
    );

    inspector.set_custom_lapse_expression("1 - Rs / r + q^2 / r^2");
    assert_eq!(
        inspector.sandbox.lapse_bar.active_param_names(),
        &["q".to_string()]
    );

    let mut playground = GeodesicPlaygroundTool::default();
    playground.state.spacetime = SpacetimePreset::CustomRadialLapse;
    playground.state.scale_radius = 2.0;
    playground.state.initial_radius = 12.0;
    playground.state.impact_or_momentum = 5.25;
    playground.state.step_size = 0.015;
    playground.sandbox.lapse_bar.set_expression("1 - 2 / r");
    playground.clamp_parameters();

    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            playground.rebuild_trajectory(720);
        });
    }
    assert_eq!(playground.samples.len(), 721);
    assert!(playground
        .samples
        .iter()
        .all(|s| s.radial_coord.is_finite() && s.metric_norm.is_finite()));

    // Plunging custom Reissner-Nordstrom ray halts cleanly when f(r) <= 0.02 without NaNs
    playground
        .sandbox
        .lapse_bar
        .set_expression("1 - Rs / r + q^2 / r^2");
    playground.state.impact_or_momentum = 0.5;
    playground.state.step_size = 0.05;
    playground.rebuild_trajectory(600);
    assert!(playground.halted_near_singularity);
    assert!(playground
        .samples
        .iter()
        .all(|s| s.radial_coord.is_finite() && s.embed_xyz[2].is_finite()));
}

#[test]
fn atomic_ast_hydration_rollback_and_maturity_tiers_align_with_upstream_contracts() {
    use crate::framework::maturity_badge::resolve_tool_maturity_tier;
    use crate::tabs::astrophysics::galactic_orbits_tool::GalacticOrbitSimulator;
    use scientific_metadata::theory::MaturityTier;

    let h5 = (0..5).fold("r*r".to_string(), |acc, _| format!("({acc})*({acc})"));
    let over_256_nodes = format!("{h5}+{h5}+r");
    assert!(over_256_nodes.len() <= 512);
    let over_512_bytes = format!("{}1", "r+".repeat(260));

    let mut tool = GeodesicPlaygroundTool::default();
    let baseline_rev = tool.state_revision();
    let baseline_axes = tool.atlas.axis_keys();
    let baseline_expr = tool.state.custom_lapse_expr.clone();

    let mut dormant_oversized = tool.state.clone();
    dormant_oversized.custom_lapse_expr = over_512_bytes;
    let mut active_ast_overflow = tool.state.clone();
    active_ast_overflow.spacetime = SpacetimePreset::CustomRadialLapse;
    active_ast_overflow.scale_radius = 3.5;
    active_ast_overflow
        .custom_lapse_expr
        .clone_from(&over_256_nodes);
    let mut bad_axes = tool.state.clone();
    bad_axes.scale_radius = 4.0;
    bad_axes.atlas_axes = ["scale_radius".into(), "unknown_axis".into()];

    for bad_state in [&dormant_oversized, &active_ast_overflow, &bad_axes] {
        let raw = serde_json::to_string(bad_state).expect("serialize bad state");
        assert!(tool.try_load_state(&raw).is_err());
        assert_eq!(tool.state_revision(), baseline_rev);
        assert_eq!(tool.state.spacetime, SpacetimePreset::Schwarzschild);
        assert!((tool.state.scale_radius - 2.0).abs() < 1e-12);
        assert_eq!(tool.atlas.axis_keys(), baseline_axes);
        assert_eq!(tool.state.custom_lapse_expr, baseline_expr);
        assert!(tool.sandbox.lapse_bar.is_valid());
    }

    let mut inspector = ChristoffelInspectorTool::default();
    let mut bad_insp = inspector.state.clone();
    bad_insp.manifold = InspectorManifold::CustomRadialLapse4D;
    bad_insp.custom_lapse_expr = over_256_nodes;
    assert!(inspector
        .try_load_state(&serde_json::to_string(&bad_insp).expect("inspector json"))
        .is_err());
    assert_eq!(inspector.state.manifold, InspectorManifold::Schwarzschild4D);

    let mut orb = GalacticOrbitSimulator::default();
    assert!(orb.try_load_state(&" ".repeat(70_000)).is_err());
    let g_tier = tool.theory().maturity_tier();
    let o_tier = orb.theory().maturity_tier();
    assert_eq!(
        resolve_tool_maturity_tier("GeodesicPlaygroundTool", "relativity", Some(g_tier)),
        MaturityTier::Validated
    );
    assert_eq!(
        resolve_tool_maturity_tier("GalacticOrbitSimulator", "astrophysics", Some(o_tier)),
        MaturityTier::Validated
    );
}
