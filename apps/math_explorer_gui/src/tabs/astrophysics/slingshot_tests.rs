#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::galactic_orbits_tool::GalacticOrbitSimulator;
use crate::audio::wav::synthesize_frames_to_wav;
use crate::framework::{
    BrushInteractionMode, BrushMode, BrushStroke, FiniteStateGuard, InteractionContext,
    InteractiveTool, ShaderDialect,
};
use crate::url_state::URL_JSON_BOUNDS;
use eframe::egui;
use math_explorer::physics::astrophysics::SLINGSHOT_RING_CAPACITY;
#[cfg(not(target_arch = "wasm32"))]
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_galactic_orbits_slingshot_1000_step_zero_alloc_audio_and_shaders() {
    let mut sim = GalacticOrbitSimulator::default();
    sim.brush_kit.interaction_mode = BrushInteractionMode::Sculpt;

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
}

#[test]
fn test_galactic_orbits_wgsl_glsl_shaders_and_finite_state_guard() {
    let mut sim = GalacticOrbitSimulator::default();
    sim.launch_probe_at_cursor(1.4, -0.7);
    let wgsl = sim
        .export_shader_source(crate::framework::ShaderExportFormat::Wgsl)
        .unwrap();
    let glsl = sim
        .export_shader_source(crate::framework::ShaderExportFormat::Glsl)
        .unwrap();
    let glsl_web = sim.export_glsl_shader(ShaderDialect::WebGl2Es300).unwrap();
    let glsl_toy = sim.export_glsl_shader(ShaderDialect::Shadertoy).unwrap();
    assert!(wgsl.contains("@fragment") && wgsl.contains("fn fs_main("));
    assert!(glsl.starts_with("#version 300 es"));
    assert!(glsl_web.starts_with("#version 300 es") && glsl_web.contains("void main()"));
    assert!(!glsl_toy.contains("#version") && glsl_toy.contains("void mainImage("));

    let saved = sim.save_state().unwrap();
    URL_JSON_BOUNDS.validate_json(&saved).unwrap();
    sim.try_load_state(&saved).unwrap();

    sim.cfg.dt_myr = f64::NAN;
    assert!(!sim.has_finite_parameters());
    assert!(sim
        .export_shader_source(crate::framework::ShaderExportFormat::Wgsl)
        .is_err());
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
