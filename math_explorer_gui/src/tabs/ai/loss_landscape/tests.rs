#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::arena::RacerKind;
use super::data::{LossMetric, TargetDataMode};
use super::presets::{PredictorMode, NONCONVEX_SINUSOIDAL_PRESET_JSON, STARTER_PRESET_JSON};
use super::surfaces::{LandscapeSurface, SculptBump, SurfaceEvaluator, MAX_SCULPT_BUMPS};
use super::{CanvasInteractionMode, LandscapeViewMode, LossLandscapeTool};
use crate::framework::{InputMode, InteractionContext, InteractiveTool};
use eframe::egui;

fn arena_raw_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        events,
        ..Default::default()
    }
}

#[test]
fn test_default_state_and_legacy_preset_backwards_compatibility() {
    let mut tool = LossLandscapeTool::default();
    assert_eq!(tool.w_min, 0.0);
    assert_eq!(tool.w_max, 4.0);
    assert_eq!(tool.b_min, -1.0);
    assert_eq!(tool.b_max, 3.0);
    assert_eq!(tool.resolution, 20);
    assert!(tool.has_finite_parameters());

    let legacy_json = r#"{"w_min":-1.0,"w_max":5.0,"b_min":-2.0,"b_max":4.0,"resolution":30}"#;
    tool.try_load_state(legacy_json)
        .expect("legacy 5-field JSON state must load cleanly");
    assert_eq!(tool.w_min, -1.0);
    assert_eq!(tool.w_max, 5.0);
    assert_eq!(tool.resolution, 30);
    assert!(tool.has_finite_parameters());
}

#[test]
fn test_analytical_gradients_match_central_finite_differences() {
    let bumps = [
        SculptBump::new(0.4, -0.3, 3.2, 0.55),
        SculptBump::new(-0.7, 0.8, -2.1, 0.4),
    ];
    let eps = 1e-6;
    for surface in LandscapeSurface::ALL {
        for (w, b) in [(-0.85, 0.65), (0.35, -0.45), (1.15, 0.9)] {
            let grad = surface.gradient(&bumps, w, b);
            let fd_w = (surface.evaluate(&bumps, w + eps, b)
                - surface.evaluate(&bumps, w - eps, b))
                / (2.0 * eps);
            let fd_b = (surface.evaluate(&bumps, w, b + eps)
                - surface.evaluate(&bumps, w, b - eps))
                / (2.0 * eps);
            assert!(
                (grad.x - fd_w).abs() < 1e-4 && (grad.y - fd_b).abs() < 1e-4,
                "Gradient mismatch on {:?} at ({w}, {b}): analytic={grad:?}, fd=({fd_w}, {fd_b})",
                surface
            );
        }
    }
}

#[test]
fn test_on_brush_sculpts_gaussian_barriers_wells_and_caps_at_32() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    let base_center = tool.surface.evaluate(&tool.sculpt_bumps, 2.0, 1.0);
    tool.view_mode = LandscapeViewMode::Contour2D;
    tool.sculpt_amplitude = 4.0;
    tool.sculpt_sigma = 0.5;

    let canvas_rect = egui::Rect::from_min_max(egui::pos2(300.0, 80.0), egui::pos2(1100.0, 720.0));
    let _ = ctx.run(arena_raw_input(Vec::new()), |egui_ctx| {
        egui::CentralPanel::default().show(egui_ctx, |ui| {
            let response = ui.allocate_rect(canvas_rect, egui::Sense::click_and_drag());
            let center_pos = canvas_rect.center();
            let brush_ctx = InteractionContext {
                pointer_pos: Some(center_pos),
                delta: egui::vec2(4.0, 0.0),
                is_dragging: true,
                is_clicked: false,
                response: &response,
                input_mode: InputMode::Mouse,
                multi_touch: None,
                keys_down: std::collections::HashSet::new(),
                modifiers: egui::Modifiers::NONE,
            };
            tool.on_brush(&brush_ctx);
            assert_eq!(tool.sculpt_bumps.len(), 1);
            assert!(tool.surface.evaluate(&tool.sculpt_bumps, 2.0, 1.0) > base_center + 3.5);

            for step in 1..45 {
                let t = step as f32 / 45.0;
                let pos = egui::pos2(
                    canvas_rect.min.x + 20.0 + t * 740.0,
                    canvas_rect.min.y + 30.0 + t * 560.0,
                );
                let alt_mods = egui::Modifiers {
                    alt: step % 2 == 0,
                    ..egui::Modifiers::NONE
                };
                let stroke_ctx = InteractionContext {
                    pointer_pos: Some(pos),
                    delta: egui::vec2(8.0, 6.0),
                    is_dragging: true,
                    is_clicked: false,
                    response: &response,
                    input_mode: InputMode::Mouse,
                    multi_touch: None,
                    keys_down: std::collections::HashSet::new(),
                    modifiers: alt_mods,
                };
                tool.on_brush(&stroke_ctx);
            }
        });
    });

    assert_eq!(tool.sculpt_bumps.len(), MAX_SCULPT_BUMPS);
    assert!(tool.sculpt_bumps.iter().any(|b| b.amplitude < 0.0));
    assert!(tool.sculpt_bumps.iter().any(|b| b.amplitude > 0.0));
}

fn simulate_canvas_drag(
    ctx: &egui::Context,
    tool: &mut LossLandscapeTool,
    start: egui::Pos2,
    end: egui::Pos2,
) {
    let _ = ctx.run(arena_raw_input(Vec::new()), |c| tool.show(c));
    let _ = ctx.run(
        arena_raw_input(vec![
            egui::Event::PointerMoved(start),
            egui::Event::PointerButton {
                pos: start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |c| tool.show(c),
    );
    let mid = egui::pos2((start.x + end.x) * 0.5, (start.y + end.y) * 0.5);
    let _ = ctx.run(arena_raw_input(vec![egui::Event::PointerMoved(mid)]), |c| {
        tool.show(c)
    });
    let _ = ctx.run(arena_raw_input(vec![egui::Event::PointerMoved(end)]), |c| {
        tool.show(c)
    });
    let _ = ctx.run(
        arena_raw_input(vec![egui::Event::PointerButton {
            pos: end,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]),
        |c| tool.show(c),
    );
}

#[test]
fn test_slingshot_and_sculpt_modes_suppress_camera_orbit() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    let initial_yaw = tool.camera.yaw;
    let initial_pitch = tool.camera.pitch;
    let start = egui::pos2(720.0, 360.0);
    let end = egui::pos2(860.0, 470.0);

    tool.interaction_mode = CanvasInteractionMode::Slingshot;
    simulate_canvas_drag(&ctx, &mut tool, start, end);
    assert_eq!(tool.camera.yaw, initial_yaw);
    assert_eq!(tool.camera.pitch, initial_pitch);
    assert!(
        tool.arena_config.slingshot_velocity[0].abs() > 1e-3
            || tool.arena_config.slingshot_velocity[1].abs() > 1e-3
    );

    tool.interaction_mode = CanvasInteractionMode::SculptBrush;
    simulate_canvas_drag(&ctx, &mut tool, start, end);
    assert_eq!(tool.camera.yaw, initial_yaw);
    assert_eq!(tool.camera.pitch, initial_pitch);
    assert!(!tool.sculpt_bumps.is_empty());

    tool.interaction_mode = CanvasInteractionMode::Orbit;
    simulate_canvas_drag(&ctx, &mut tool, start, end);
    assert!(
        (tool.camera.yaw - initial_yaw).abs() > 1e-3
            || (tool.camera.pitch - initial_pitch).abs() > 1e-3
    );
}

#[test]
fn test_3d_and_2d_views_step_and_reflect_racers_within_bounds() {
    let ctx = egui::Context::default();
    let mut tool = LossLandscapeTool::default();
    tool.apply_surface_preset(LandscapeSurface::Rosenbrock);
    let bounds = tool.domain_bounds();
    let eval = SurfaceEvaluator::new(tool.surface, &tool.sculpt_bumps, None);
    tool.arena.launch_slingshot(
        &mut tool.arena_config,
        [-1.8, 2.8],
        [-25.0, 25.0],
        eval,
        bounds,
    );
    for _ in 0..80 {
        tool.step_simulation();
    }
    for (idx, racer) in tool.arena.racers.iter().enumerate() {
        if !RacerKind::ALL[idx].is_available_in_build() {
            continue;
        }
        assert!(
            (tool.w_min..=tool.w_max).contains(&racer.phase.q.x)
                && (tool.b_min..=tool.b_max).contains(&racer.phase.q.y),
            "Racer {:?} escaped bounds: {:?}",
            racer.kind,
            racer.phase.q
        );
        assert!(racer.potential_energy.is_finite() && racer.kinetic_energy.is_finite());
        assert!(racer.trail.len() > 2);
    }

    let _ = ctx.run(arena_raw_input(Vec::new()), |c| tool.show(c));
    assert!(!tool.export_datasets().is_empty());
    tool.view_mode = LandscapeViewMode::Contour2D;
    let _ = ctx.run(arena_raw_input(Vec::new()), |c| tool.show(c));
    assert!(!tool.export_datasets().is_empty());
}

#[test]
fn test_max_32_sculpt_bumps_preset_fits_under_max_decoded_url_state_bytes() {
    let mut tool = LossLandscapeTool::default();
    tool.apply_surface_preset(LandscapeSurface::Rastrigin);
    for i in 0..MAX_SCULPT_BUMPS {
        let angle = (i as f64) * 0.37;
        super::surfaces::push_sculpt_bump(
            &mut tool.sculpt_bumps,
            SculptBump::new(
                2.5 * angle.cos(),
                2.5 * angle.sin(),
                if i % 2 == 0 { 3.25 } else { -2.75 },
                0.45,
            ),
        );
    }
    for _ in 0..25 {
        tool.step_simulation();
    }
    assert_eq!(tool.sculpt_bumps.len(), MAX_SCULPT_BUMPS);

    let state_json = tool.save_state().expect("save_state must succeed");
    assert!(
        !state_json.contains("trail"),
        "Transient trails must be omitted from LossLandscapeState"
    );
    let envelope = tool
        .export_preset_json()
        .expect("export_preset_json must succeed");
    assert!(
        envelope.len() <= crate::url_state::MAX_DECODED_URL_STATE_BYTES,
        "32-bump preset envelope ({} bytes) exceeded MAX_DECODED_URL_STATE_BYTES ({})",
        envelope.len(),
        crate::url_state::MAX_DECODED_URL_STATE_BYTES
    );
    crate::url_state::validate_url_preset_json(&envelope)
        .expect("URL preset validator must accept 32-bump LossLandscapeTool state");

    let mut restored = LossLandscapeTool::default();
    restored
        .import_preset_json(&envelope)
        .expect("import_preset_json must succeed");
    assert_eq!(restored.save_state().as_deref(), Some(state_json.as_str()));
    assert_eq!(restored.arena.racers[0].trail.len(), 1);
}

#[test]
fn test_zero_allocation_arena_stepping_under_verified_allocator() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut tool = LossLandscapeTool::default();
    tool.apply_surface_preset(LandscapeSurface::Rastrigin);
    super::surfaces::push_sculpt_bump(&mut tool.sculpt_bumps, SculptBump::new(-1.0, 1.0, 2.5, 0.5));
    tool.reset_arena();

    let (_res, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..64 {
            tool.step_simulation();
        }
    });
    assert_eq!(tool.arena.step_count, 64);
    assert_eq!(metrics.memory_loads, 0);
}

#[test]
fn test_lorenz_chaotic_preset_graceful_feature_gating() {
    let mut tool = LossLandscapeTool::default();
    let preset = r#"{"w_min":-2.0,"w_max":2.0,"b_min":-1.0,"b_max":3.0,"resolution":20,"surface":"Rosenbrock","arena":{"enabled_racers":[true,true,true,true,true]}}"#;
    tool.try_load_state(preset)
        .expect("preset enabling LorenzChaotic must load in all feature combinations");
    for _ in 0..20 {
        tool.step_simulation();
    }
    assert_eq!(
        RacerKind::LorenzChaotic.is_available_in_build(),
        cfg!(feature = "physics")
    );
    for (idx, racer) in tool.arena.racers.iter().enumerate().take(4) {
        assert!(
            racer.trail.len() > 2,
            "Optimizer slot {idx} must advance regardless of physics feature"
        );
    }
    if cfg!(feature = "physics") {
        assert!(tool.arena.racers[4].trail.len() > 2);
    } else {
        assert_eq!(tool.arena.racers[4].trail.len(), 1);
    }
}

#[test]
fn test_racer_semantic_palette_colors_distinct_and_wcag_aa_across_themes() {
    use crate::app::{contrast_ratio, ThemeMode};

    for mode in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let visuals = mode.visuals();
        let palette = mode.palette();
        let colors: Vec<egui::Color32> = RacerKind::ALL.iter().map(|k| k.color(&palette)).collect();
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(
                    colors[i],
                    colors[j],
                    "{mode:?} assigned duplicate colors to {:?} and {:?}",
                    RacerKind::ALL[i],
                    RacerKind::ALL[j]
                );
            }
            for bg in [
                visuals.panel_fill,
                visuals.window_fill,
                visuals.extreme_bg_color,
            ] {
                let ratio = contrast_ratio(colors[i], bg);
                assert!(
                    ratio >= 4.5,
                    "{mode:?} {:?} color achieved {ratio:.2}:1 (< 4.5:1)",
                    RacerKind::ALL[i]
                );
            }
        }
    }
}

#[test]
fn test_custom_loss_landscape_mse_and_preset_roundtrip() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut tool = LossLandscapeTool {
        predictor_mode: PredictorMode::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("w * x + b");
    // At true solution w = 2, b = 1 for y = 2x + 1, MSE must be 0
    assert!((tool.evaluate_mse_at(2.0, 1.0).unwrap() - 0.0).abs() < 1e-12);

    tool.import_preset_json(NONCONVEX_SINUSOIDAL_PRESET_JSON)
        .expect("non-convex preset should load");
    assert_eq!(tool.predictor_mode, PredictorMode::Custom);
    assert!(tool.evaluate_mse_at(1.5, 0.8).unwrap().is_finite());

    let (_res, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..16 {
            tool.step_simulation();
        }
    });
    assert_eq!(tool.arena.step_count, 16);
    assert_eq!(metrics.memory_loads, 0);
    assert!(tool.import_preset_json(STARTER_PRESET_JSON).is_ok());
}

#[test]
fn test_loss_landscape_table_targets_mse_mae_and_huber() {
    let mut tool = LossLandscapeTool::default();
    tool.set_target_table_text("0.0, 1.0\n1.0, 3.0\n2.0, 15.0\n3.0, 7.0");
    assert!(tool.table_parse_error.is_none());

    tool.loss_metric = LossMetric::Mse;
    let mse = tool.evaluate_loss_at(2.0, 1.0).expect("mse");
    assert!((mse - 25.0).abs() < 1e-12);

    tool.loss_metric = LossMetric::Mae;
    let mae = tool.evaluate_loss_at(2.0, 1.0).expect("mae");
    assert!((mae - 2.5).abs() < 1e-12);

    tool.loss_metric = LossMetric::Huber;
    tool.huber_delta = 2.0;
    let huber = tool.evaluate_loss_at(2.0, 1.0).expect("huber");
    assert!((huber - 4.5).abs() < 1e-12);
    assert!(huber < mse && huber > mae);

    tool.set_target_table_text("0.0, 1.0\nbad_row");
    assert!(tool.table_parse_error.is_some());
    assert!((tool.evaluate_loss_at(2.0, 1.0).expect("retained") - 4.5).abs() < 1e-12);
}

#[test]
fn test_loss_landscape_formula_targets_zero_alloc_and_roundtrip() {
    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut tool = LossLandscapeTool {
        loss_metric: LossMetric::Huber,
        huber_delta: 2.0,
        target_mode: TargetDataMode::Formula,
        target_x_bounds: [-2.0, 2.0],
        target_sample_count: 11,
        predictor_mode: PredictorMode::Custom,
        ..Default::default()
    };
    tool.target_formula_bar.set_expression("3 * x - 2");
    tool.refresh_target_samples();
    tool.math_input.set_expression("w * x + b");
    tool.reset_arena();
    assert!(tool.evaluate_loss_at(3.0, -2.0).expect("exact match") < 1e-12);

    let (val, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let mut acc = 0.0;
        for i in 0..16 {
            acc += tool
                .evaluate_loss_at(2.5 + i as f64 * 0.01, -1.5)
                .unwrap_or(0.0);
            tool.step_simulation();
        }
        acc
    });
    assert!(val.is_finite());
    assert_eq!(metrics.memory_loads, 0);

    let json = tool.save_state().expect("save formula state");
    assert!(json.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);
    let mut restored = LossLandscapeTool::default();
    restored
        .try_load_state(&json)
        .expect("restore formula state");
    assert_eq!(restored.target_mode, TargetDataMode::Formula);
    assert_eq!(restored.loss_metric, LossMetric::Huber);
    assert!((restored.huber_delta - 2.0).abs() < 1e-12);
    assert!(restored.evaluate_loss_at(3.0, -2.0).expect("restored") < 1e-12);
}
