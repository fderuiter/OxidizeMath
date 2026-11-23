#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::galactic_orbits_tool::GalacticOrbitSimulator;
use super::rotation_curve_tool::RotationCurveExplorer;
use super::*;
use crate::audio::wav::synthesize_frames_to_wav;
use crate::framework::{
    BrushInteractionMode, BrushMode, BrushStroke, FiniteStateGuard, InteractionContext,
    InteractiveTool, ShaderDialect,
};
use crate::url_state::URL_JSON_BOUNDS;
use egui_plot::export::{color32_image_to_png, format_csv, ExportOptions};
use egui_plot::export_primitives::{
    datasets_to_primitives, export_primitives_svg, render_primitives_image,
};
use math_explorer::physics::astrophysics::{DwarfMorphology, SLINGSHOT_RING_CAPACITY};
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::engine::VerifiedEngine;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[test]
fn test_astrophysics_tab_domain_slug_and_tools() {
    let mut tab = AstrophysicsTab::default();
    assert_eq!(tab.name(), "Astrophysics");
    assert_eq!(tab.academic_domain(), "Physics");
    assert_eq!(tab.url_slug(), "astrophysics");
    assert!(tab.select_tool_by_name("RotationCurveExplorer"));
    assert_eq!(tab.active_tool_id(), Some("RotationCurveExplorer"));
    assert!(tab.select_tool_by_name("GalacticOrbitSimulator"));
    assert_eq!(tab.active_tool_id(), Some("GalacticOrbitSimulator"));
}

#[test]
fn test_type_code_95_to_99_renders_unestablished_badge_without_panic() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = RotationCurveExplorer::default();
    tool.cfg.morphology = DwarfMorphology::TypeCode95To99;
    tool.cfg.use_absolute_magnitude = false;
    tool.recalculate();
    assert!(tool
        .current_baryons()
        .scaling
        .apparent_magnitude_b
        .is_none());

    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let out = ctx.run(raw, |ctx| tool.show(ctx));
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit output");
    let found_badge = ak.nodes.iter().any(|(_, n)| {
        n.label()
            .or_else(|| n.value())
            .is_some_and(|l| l.contains("Unestablished in Tillaboev (2025)"))
    });
    assert!(
        found_badge,
        "Expected explicit unestablished-formula badge for TypeCode95To99"
    );
}

#[test]
fn test_galactic_orbits_slingshot_1000_step_zero_alloc_audio_and_shaders() {
    let mut sim = GalacticOrbitSimulator::default();
    sim.brush_kit.interaction_mode = BrushInteractionMode::Sculpt;

    // Launch 10 probes (exceeding SLINGSHOT_RING_CAPACITY = 8) via cursor & brush strokes
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp = ui.allocate_response(egui::vec2(64.0, 64.0), egui::Sense::click_and_drag());
            let stroke = BrushStroke {
                start: [1.5, 0.4],
                end: [2.1, 1.2],
                radius: 0.6,
                mode: BrushMode::VelocityJet,
                drag_vector: [0.6, 0.8],
            };
            let ictx = InteractionContext::from_brush_stroke(&resp, stroke);
            sim.on_brush(&ictx);
        });
    });
    for k in 0..9 {
        sim.launch_probe_at_cursor(1.2 + 0.1 * f64::from(k), -0.8);
    }
    assert_eq!(sim.system.slingshot_probe_count(), SLINGSHOT_RING_CAPACITY);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (calls, ops) = sim.step_inside_verified_engine();
        assert!(calls > 0 && ops > 100);
        let (acc, _) = VerifiedEngine::run_verified(|| {
            let mut sum = 0.0_f64;
            for step in 0..1_000usize {
                if step % 125 == 0 {
                    let _ = sim.system.launch_tracer_in_place(1.8, 0.5, -22.0, 58.0);
                }
                sim.system.step_n_leapfrog(sim.cfg.dt_myr, 1);
                let sb = sim.system.sample_surface_brightness(1.1, -0.7);
                let af = sim.build_audio_frame();
                sum += sb[0] + sb[1] + sb[2] + f64::from(af.voices[0].freq_hz);
            }
            sum
        });
        assert!(acc.is_finite());
    }

    let wav = synthesize_frames_to_wav(&[sim.build_audio_frame()], 44_100, 128, 0.8);
    assert_eq!(&wav[0..4], b"RIFF");
    let glsl_web = sim.export_glsl_shader(ShaderDialect::WebGl2Es300).unwrap();
    let glsl_toy = sim.export_glsl_shader(ShaderDialect::Shadertoy).unwrap();
    assert!(glsl_web.starts_with("#version 300 es") && glsl_web.contains("void main()"));
    assert!(!glsl_toy.contains("#version") && glsl_toy.contains("void mainImage("));

    let saved = sim.save_state().unwrap();
    URL_JSON_BOUNDS.validate_json(&saved).unwrap();
    sim.try_load_state(&saved).unwrap();

    sim.cfg.dt_myr = f64::NAN;
    assert!(!sim.has_finite_parameters());
    FiniteStateGuard::clamp_parameters(&mut sim);
    assert!(sim.has_finite_parameters());
}

#[test]
fn test_galactic_orbits_keyboard_arrow_and_enter_slingshot_launch() {
    let ctx = egui::Context::default();
    let mut sim = GalacticOrbitSimulator::default();
    sim.brush_kit.interaction_mode = BrushInteractionMode::Sculpt;
    assert_eq!(sim.system.slingshot_probe_count(), 0);

    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| sim.show_inside(ui));
        },
    );
    let plot_id = sim.last_plot_id.expect("galactic orbit plot rendered");
    ctx.memory_mut(|m| m.request_focus(plot_id));

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
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            events: key_events,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| sim.show_inside(ui));
        },
    );
    assert_eq!(sim.system.slingshot_probe_count(), 1);
}

#[test]
fn test_paczynski_wiita_isco_capture_rosetta_zero_alloc_and_doppler() {
    let mut sim = GalacticOrbitSimulator::default();
    sim.cfg.central_bh.enabled = true;
    sim.cfg.central_bh.mass_solar = 2.5e9;
    sim.cfg.central_bh.effective_rs_kpc = 0.28;
    sim.cfg.central_bh.relativistic_beta_scale = 0.008;
    sim.cfg.tidal_tracers = 12;
    sim.cfg.tidal_tangential_ratio = 0.20;
    sim.cfg.substeps_per_frame = 8;
    sim.rebuild_system(0);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for _ in 0..8 {
            let (calls, _ops) = sim.step_inside_verified_engine();
            assert!(calls > 0);
        }
    }
    #[cfg(target_arch = "wasm32")]
    for _ in 0..8 {
        sim.advance_burst();
    }
    assert!(
        sim.system.captured_particle_count() > 0,
        "Expected plunging stream tracers to capture cleanly at ISCO (3 R_s)"
    );
    assert!(sim.has_finite_parameters());

    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |sub| sim.show(sub));

    let saved = sim.save_state().expect("save BH orbit state");
    let mut restored = GalacticOrbitSimulator::default();
    restored
        .try_load_state(&saved)
        .expect("restore BH orbit state");
    assert!(restored.cfg.central_bh.enabled);
    assert!((restored.cfg.central_bh.effective_rs_kpc - 0.28).abs() < 1.0e-9);
}

#[test]
fn test_interactive_orbit_caustic_drag_and_nfw_vs_burkert_exports() {
    let mut sim = GalacticOrbitSimulator::default();
    sim.cfg.lens_overlay.enabled = true;
    sim.cfg.lens_overlay.log10_sigma_crit = 7.35;
    sim.drag_orbit_lens_to([2.25, -1.40]);
    assert!((sim.cfg.lens_overlay.lens_center_kpc[0] - 2.25).abs() < 1.0e-9);
    assert!((sim.cfg.lens_overlay.lens_center_kpc[1] + 1.40).abs() < 1.0e-9);

    let (r_nfw, r_burk) = sim.compute_caustic_radii_kpc();
    assert!(r_nfw.is_some() && r_burk.is_some());
    assert!((r_nfw.unwrap_or(0.0) - r_burk.unwrap_or(0.0)).abs() > 1.0e-3);

    egui_plot::clear_last_plot_datasets();
    let orbit_ds = sim.export_datasets();
    assert!(orbit_ds
        .iter()
        .any(|(n, _)| n.contains("NFW Cusp Critical")));
    assert!(orbit_ds
        .iter()
        .any(|(n, _)| n.contains("Burkert Core Critical")));
    assert_valid_dataset_and_figure_exports(&orbit_ds);

    let mut rc = RotationCurveExplorer::default();
    rc.cfg.relativity.enabled = true;
    rc.cfg.relativity.central_bh_rs_kpc = 0.20;
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            rc.recalculate_zero_alloc();
        });
    }
    assert_ne!(
        rc.nfw_lens_samples[12].shear,
        rc.burkert_lens_samples[12].shear
    );
    assert!(rc.dilation_samples[0][1] < rc.dilation_samples[95][1]);

    let rc_saved = rc.save_state().expect("save rotation relativity state");
    let mut rc_reloaded = RotationCurveExplorer::default();
    rc_reloaded
        .try_load_state(&rc_saved)
        .expect("reload rotation relativity state");
    assert!(rc_reloaded.cfg.relativity.enabled);

    egui_plot::clear_last_plot_datasets();
    let rc_ds = rc.export_datasets();
    assert!(rc_ds
        .iter()
        .any(|(n, _)| n.contains("Schwarzschild Time Dilation")));
    assert_valid_dataset_and_figure_exports(&rc_ds);
}

fn assert_valid_dataset_and_figure_exports(datasets: &[(String, Vec<[f64; 2]>)]) {
    let csv = format_csv(datasets);
    assert!(csv.lines().count() > 20);
    let opts = ExportOptions::default();
    let prims = datasets_to_primitives(datasets);
    let svg = export_primitives_svg(&prims, &opts).expect("SVG export");
    assert!(svg.contains("<svg"));
    let img = render_primitives_image(&prims, &opts).expect("raster figure");
    let png = color32_image_to_png(&img).expect("PNG encode");
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n") && png.len() > 256);
}

#[test]
fn test_astrophysics_mystery_halo_fit_zero_alloc_residuals_and_clamping() {
    use super::challenge::{
        CHALLENGE_ASTRO_MYSTERY_HALO_FIT, LOCKED_DISTANCE_MPC, MYSTERY_DWARF_TARGETS,
    };
    use crate::starter_gallery::find_challenge_metadata;

    assert_eq!(MYSTERY_DWARF_TARGETS.len(), 16);
    let meta = find_challenge_metadata(CHALLENGE_ASTRO_MYSTERY_HALO_FIT).expect("registered");
    let baseline = meta.generate_baseline_state();
    let mut tool = RotationCurveExplorer::default();
    tool.apply_challenge(Some(CHALLENGE_ASTRO_MYSTERY_HALO_FIT))
        .expect("challenge applies");
    tool.try_load_state(&baseline).expect("baseline loads");

    let base_eval = tool.evaluate_challenge().expect("eval");
    assert!(!base_eval.passed, "Baseline NFW halo must fail fit");
    assert!(tool.rmse_kms > 10.0 && tool.reduced_chi_sq > 10.0);

    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            tool.cfg.halo_kind = HaloKindChoice::BurkertCore;
            tool.cfg.halo_log10_mass = 10.15;
            tool.cfg.halo_scale_kpc = 2.40;
            tool.cfg.halo_concentration = 13.0;
            tool.recalculate_zero_alloc();
        });
    }

    let solved_eval = tool.evaluate_challenge().expect("eval");
    assert!(
        solved_eval.passed,
        "Cored Burkert profile must pass mystery dwarf fit: {solved_eval:?}"
    );
    assert!(tool.rmse_kms <= 2.40 && tool.reduced_chi_sq <= 1.35);
    assert!(
        tool.counterpart_reduced_chi_sq > 5.0,
        "NFW counterpart at identical parameters must fail cored inner slope"
    );

    tool.sync_rotation_callouts();
    assert!(tool.callout_manager.callouts().len() >= 2);

    let tampered = r#"{"distance_mpc":88.0,"abs_magnitude_v":-19.0,"use_absolute_magnitude":true,"morphology":"GeneralIrregular","gas_fraction":0.10,"disk_scale_kpc":3.0,"halo_kind":"BurkertCore","halo_log10_mass":10.15,"halo_scale_kpc":2.40,"halo_concentration":13.0,"max_radius_kpc":22.0,"compare_counterpart":true}"#;
    let clamped = meta.clamp_state_json(tampered).expect("clamps cleanly");
    let parsed: super::rotation_curve_tool::RotationCurveState =
        serde_json::from_str(&clamped).unwrap();
    assert!((parsed.distance_mpc - LOCKED_DISTANCE_MPC).abs() < 1e-9);
    assert!(!parsed.use_absolute_magnitude);
    assert_eq!(parsed.morphology, DwarfMorphology::TypeCode10);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn wasm_astrophysics_tab_url_roundtrip_and_symplectic_stepping() {
    use super::challenge::CHALLENGE_ASTRO_MYSTERY_HALO_FIT;
    use crate::starter_gallery::find_challenge_metadata;

    let mut tab = AstrophysicsTab::default();
    assert_eq!(tab.url_slug(), "astrophysics");
    assert_eq!(tab.academic_domain(), "Physics");

    assert!(tab.select_tool_by_name("RotationCurveExplorer"));
    let mut rc = RotationCurveExplorer::default();
    rc.cfg.morphology = DwarfMorphology::TypeCode95To99;
    rc.cfg.halo_kind = HaloKindChoice::NfwCusp;
    rc.cfg.distance_mpc = 19.5;
    rc.cfg.relativity.enabled = true;
    rc.recalculate();
    let exported = rc.export_preset_json().expect("export preset json");
    tab.import_url_preset_json(Some("RotationCurveExplorer"), Some(&exported))
        .expect("import url preset");
    assert!(tab.export_url_preset_json().unwrap().is_some());

    let meta = find_challenge_metadata(CHALLENGE_ASTRO_MYSTERY_HALO_FIT).expect("registered");
    rc.apply_challenge(Some(CHALLENGE_ASTRO_MYSTERY_HALO_FIT))
        .expect("challenge enabled");
    rc.try_load_state(&meta.generate_baseline_state())
        .expect("baseline loaded");
    assert!(!rc.evaluate_challenge().expect("eval").passed);

    let mut orbits = GalacticOrbitSimulator::default();
    orbits.cfg.central_bh.enabled = true;
    orbits.cfg.lens_overlay.enabled = true;
    orbits.rebuild_system(4);
    orbits.drag_orbit_lens_to([1.5, -0.8]);
    orbits.launch_probe_at_cursor(1.6, -0.9);
    assert_eq!(orbits.system.slingshot_probe_count(), 1);
    let initial_t = orbits.system.elapsed_myr();
    orbits.step_simulation();
    assert!(orbits.system.elapsed_myr() > initial_t);
    assert!(orbits.has_finite_parameters());
    let wav = synthesize_frames_to_wav(&[orbits.build_audio_frame()], 44_100, 128, 0.8);
    assert_eq!(&wav[0..4], b"RIFF");
    assert!(orbits
        .export_glsl_shader(ShaderDialect::WebGl2Es300)
        .unwrap()
        .starts_with("#version 300 es"));
    egui_plot::clear_last_plot_datasets();
    let ds = orbits.export_datasets();
    assert_valid_dataset_and_figure_exports(&ds);
}
