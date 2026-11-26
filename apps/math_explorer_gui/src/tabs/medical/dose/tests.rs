//! Verification tests for the live 4D radar-gated IMRT contouring workbench:
//! zero-allocation `VerifiedEngine::run_verified` coupled loops, `+10 mm` cough OAR sparing,
//! all 6 `RadarGatingTool` link-bus call sites, `< 4 KiB` RLE URL/`.mws` roundtrips,
//! keyboard gantry + brush controls, and `<= 30 ms` interlock audio suspension.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::canvas::gantry_handle_pos_cm;
use super::{CanvasKeyTarget, DoseCalculationTool, DoseDisplayMode};
use crate::app::{MathExplorerApp, ViewportLayout, WorkspaceSessionManager};
use crate::audio::backend::{MuteReason, SonificationBackend};
use crate::audio::mappings::{imrt_delivery_frame, radar_linac_frame};
use crate::audio::synth::OfflinePcmSynthesizer;
use crate::framework::{BrushMode, BrushStroke, InteractionContext, InteractiveTool};
use crate::tabs::medical::imrt::{CHALLENGE_IMRT_OAR_LIMIT_GY, CHALLENGE_IMRT_OAR_SPARING};
use crate::tabs::medical::link_bus::{clear_link_bus, current_write_seq, latest_gating_telemetry};
use crate::tabs::medical::radar_gating::RadarGatingTool;
use crate::url_state::{encode_share_fragment, parse_share_location, URL_JSON_BOUNDS};
use eframe::egui::{self, Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect};
use math_explorer::physics::medical::accelerator::beam_loading_energy;
use math_explorer::physics::medical::radar_gating::BeamState;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_all_six_radar_gating_sites_publish_link_bus_telemetry() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_link_bus();
    let mut radar = RadarGatingTool::default();
    let _ = radar.arm_gating();
    let seq0 = current_write_seq();

    // Site 1: InteractiveTool::step_simulation
    radar.step_simulation();
    let seq1 = current_write_seq();
    assert!(seq1 > seq0 && latest_gating_telemetry().map(|t| t.step_event) == Some(true));

    // Site 2: continuous playing tick inside show_inside
    let ctx = egui::Context::default();
    radar.playing = true;
    let _ = ctx.run(RawInput::default(), |ctx| radar.show(ctx));
    assert!(current_write_seq() > seq1);
    radar.playing = false;

    // Sites 3, 4, 5: "Inject Cough Spike", "Inject Static Reflector", "Inject NaN Fault"
    for label in [
        "Inject Cough Spike",
        "Inject Static Reflector",
        "Inject NaN Fault",
    ] {
        let before = current_write_seq();
        click_button_by_label(&ctx, &mut radar, label);
        assert!(
            current_write_seq() > before,
            "'{label}' must publish telemetry"
        );
    }
    assert_eq!(
        latest_gating_telemetry().map(|t| t.beam_state),
        Some(BeamState::SafetyInterlock)
    );

    // Site 6: PlaybackAction::Step button after Reset Interlock + Arm
    let _ = radar.reset_interlock();
    let _ = radar.arm_gating();
    let before_step = current_write_seq();
    click_button_by_label(&ctx, &mut radar, "Step");
    assert!(current_write_seq() > before_step);
}

fn click_button_by_label(ctx: &egui::Context, tool: &mut dyn InteractiveTool, needle: &str) {
    ctx.enable_accesskit();
    let rect = Rect::from_min_size(Pos2::ZERO, egui::vec2(1280.0, 900.0));
    let out = ctx.run(
        RawInput {
            screen_rect: Some(rect),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let pos = out
        .platform_output
        .accesskit_update
        .and_then(|ak| {
            ak.nodes.into_iter().find_map(|(_, node)| {
                if node.label().is_some_and(|l| l.contains(needle)) {
                    node.bounds()
                        .map(|b| Pos2::new(0.5 * (b.x0 + b.x1) as f32, 0.5 * (b.y0 + b.y1) as f32))
                } else {
                    None
                }
            })
        })
        .unwrap_or_else(|| panic!("Button '{needle}' not found in AccessKit"));
    let _ = ctx.run(
        RawInput {
            screen_rect: Some(rect),
            events: vec![
                Event::PointerMoved(pos),
                Event::PointerButton {
                    pos,
                    button: PointerButton::Primary,
                    pressed: true,
                    modifiers: Modifiers::NONE,
                },
                Event::PointerButton {
                    pos,
                    button: PointerButton::Primary,
                    pressed: false,
                    modifiers: Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
}

#[test]
fn test_coupled_4d_cough_sparing_and_zero_heap_allocations() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_link_bus();
    let mut radar = RadarGatingTool::default();
    let _ = radar.arm_gating();
    let mut dose = DoseCalculationTool::default();
    dose.apply_oar_sparing_4d_preset();
    dose.beam_current_ma = 65.0;
    dose.bus_cursor_seq = current_write_seq();
    let _ = radar.step_gating();
    let _ = dose.poll_link_bus_in_place();

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        for step in 0..160 {
            if step == 24 || step == 72 || step == 120 {
                radar.inject_cough();
            }
            let bs = radar.step_gating();
            assert_eq!(dose.poll_link_bus_in_place(), 1);
            let t = dose.latest_telemetry;
            let m = &dose.imrt_workspace.motion;
            let rf = radar_linac_frame(
                bs,
                t.measured_disp_mm,
                t.velocity_mm_s,
                t.duty_cycle_pct,
                t.sim_time_s,
            );
            let df = imrt_delivery_frame(
                bs,
                t.true_disp_mm,
                m.gated_ptv_mean_gy,
                dose.prescription_gy,
                m.ungated_oar_max_gy.0,
                dose.oar_limit_gy,
                m.beam_energy_mev,
            );
            std::hint::black_box((rf, df));
        }
    });
    assert_coupled_4d_metrics(&dose);
}

fn assert_coupled_4d_metrics(dose: &DoseCalculationTool) {
    let m = &dose.imrt_workspace.motion;
    assert!(m.gated_steps > 0 && m.ungated_steps > m.gated_steps);
    assert!(m.gated_oar_max_gy.0 <= CHALLENGE_IMRT_OAR_LIMIT_GY);
    assert!(m.ungated_oar_max_gy.0 > CHALLENGE_IMRT_OAR_LIMIT_GY);
    assert!(m.eval_pearson_r.is_finite() && m.eval_pearson_r > 0.5);
    assert!(m.eval_rmse_mm.is_finite() && m.eval_spatial_err_pct.is_finite());
    assert!((m.beam_energy_mev - beam_loading_energy(65.0)).abs() < 1.0e-9);
    assert!(m.static_dice > 0.5 && m.gated_dice > 0.5);
}

#[test]
fn test_single_pane_4d_cough_sparing_and_interlock_halt() {
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    clear_link_bus();
    let mut dose = DoseCalculationTool::default();
    dose.apply_oar_sparing_4d_preset();

    let ((), _metrics) = VerifiedEngine::run_verified(|| {
        for step in 0..160 {
            if step == 20 || step == 68 || step == 116 {
                dose.standalone_4d.inject_cough();
            }
            let _ = dose.step_4d_in_place();
        }
    });

    assert!(!dose.linked_to_radar_bus);
    assert!(dose.imrt_workspace.motion.gated_oar_max_gy.0 <= dose.oar_limit_gy);
    assert!(dose.imrt_workspace.motion.ungated_oar_max_gy.0 > dose.oar_limit_gy);

    let (g0, u0) = (
        dose.imrt_workspace.motion.gated_steps,
        dose.imrt_workspace.motion.ungated_steps,
    );
    dose.standalone_4d.inject_nan_fault();
    assert_eq!(
        dose.step_4d_in_place().beam_state,
        BeamState::SafetyInterlock
    );
    assert_eq!(dose.imrt_workspace.motion.gated_steps, g0);
    assert_eq!(dose.imrt_workspace.motion.ungated_steps, u0);
}

#[test]
fn test_brush_sculpt_rle_url_bounds_and_mws_split_session_roundtrip() {
    let mut tool = DoseCalculationTool::default();
    assert_eq!(
        tool.save_state().as_deref(),
        Some(r#"{"amplitude":1.0,"beta":1.0,"width":200,"height":200}"#)
    );
    tool.apply_oar_sparing_4d_preset();
    sculpt_sample_contours(&mut tool);
    assert!(tool.imrt_workspace.custom_contours);

    let saved = tool.save_state().expect("sculpted state");
    assert!(saved.len() < 2_048);
    URL_JSON_BOUNDS
        .validate_json(&saved)
        .expect("must pass URL_JSON_BOUNDS");
    let frag = encode_share_fragment("medical", Some("DoseCalculationTool"), Some(&saved))
        .expect("encode share URL fragment");
    let parsed = parse_share_location(&frag, "")
        .expect("parse share fragment")
        .expect("share url params");
    let mut restored = DoseCalculationTool::default();
    restored
        .try_load_state(parsed.state_json.as_deref().expect("decoded json"))
        .expect("restore sculpted state");
    assert_eq!(
        restored.imrt_workspace.tumor_indices,
        tool.imrt_workspace.tumor_indices
    );
    assert_eq!(
        restored.imrt_workspace.organ_indices,
        tool.imrt_workspace.organ_indices
    );

    verify_split_horizontal_mws_roundtrip(&saved);
    tool.apply_challenge(Some(CHALLENGE_IMRT_OAR_SPARING))
        .expect("apply challenge");
    assert!(!tool.imrt_workspace.custom_contours);
}

fn sculpt_sample_contours(tool: &mut DoseCalculationTool) {
    let ctx = egui::Context::default();
    for (start, end, radius, mode) in [
        ([-1.5, -1.0], [1.2, 1.0], 1.1, BrushMode::PtvTarget),
        ([3.8, -1.2], [5.2, 1.2], 0.9, BrushMode::OrganAtRisk),
        ([-0.2, 0.0], [0.0, 0.0], 0.4, BrushMode::Erase),
    ] {
        let stroke = BrushStroke {
            start,
            end,
            radius,
            mode,
            drag_vector: [end[0] - start[0], end[1] - start[1]],
        };
        let _ = ctx.run(RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let (_, r) = ui.allocate_exact_size(egui::vec2(120.0, 120.0), egui::Sense::click());
                tool.on_brush(&InteractionContext::from_brush_stroke(&r, stroke));
            });
        });
    }
}

fn verify_split_horizontal_mws_roundtrip(sculpted_dose_json: &str) {
    let mut app = MathExplorerApp::default();
    let med_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Medical Physics")
        .expect("Medical Physics tab");
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, med_idx);
    app.set_pane_tab(1, med_idx);
    assert!(app
        .pane_tab_mut(0)
        .unwrap()
        .framework_mut()
        .unwrap()
        .select_tool_by_name("RadarGatingTool"));
    let fw1 = app.pane_tab_mut(1).unwrap().framework_mut().unwrap();
    assert!(fw1.select_tool_by_name("DoseCalculationTool"));
    fw1.active_tool
        .as_mut()
        .unwrap()
        .load_state(sculpted_dose_json);

    let mgr = WorkspaceSessionManager::default();
    let mws = mgr.create_envelope(app.get_app_state());
    let mut restored_app = MathExplorerApp::default();
    restored_app.load_state_from_json(&mws);
    assert_eq!(
        restored_app.viewport_layout,
        ViewportLayout::SplitHorizontal
    );
    let fw1_after = restored_app
        .pane_tab_mut(1)
        .unwrap()
        .framework_mut()
        .unwrap();
    let dose_json_after = fw1_after
        .active_tool
        .as_ref()
        .unwrap()
        .save_state()
        .expect("restored DoseCalculationTool state");
    assert_eq!(dose_json_after, sculpted_dose_json);
}

#[test]
fn test_gantry_handle_keyboard_nudge_aria_and_non_trapping_tab() {
    let mut tool = DoseCalculationTool {
        imrt_mode: true,
        canvas_key_target: CanvasKeyTarget::Gantry,
        selected_beam: 0,
        ..Default::default()
    };
    tool.refresh_model();
    let p0 = gantry_handle_pos_cm(0.0);
    assert!((p0[0] - 8.6).abs() < 1e-9 && p0[1].abs() < 1e-9);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen = Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1200.0, 820.0)));
    let canvas_focus_id = tab_to_dose_canvas(&ctx, &mut tool, screen);

    let angle_before = tool.beam_angles_deg[0];
    send_key_frame(&ctx, &mut tool, screen, Key::ArrowRight, Modifiers::SHIFT);
    assert!((tool.beam_angles_deg[0] - (angle_before + 5.0)).abs() < 1e-6);
    let live = ctx
        .data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")))
        .unwrap_or_default();
    assert!(live.contains("Beam 1") && live.contains("degrees"));

    // Switch to keyboard Brush mode with 'B', move reticle, stamp with Space, then Tab out
    send_key_frame(&ctx, &mut tool, screen, Key::B, Modifiers::NONE);
    send_key_frame(&ctx, &mut tool, screen, Key::Space, Modifiers::NONE);
    assert_eq!(tool.canvas_key_target, CanvasKeyTarget::Brush);

    send_key_frame(&ctx, &mut tool, screen, Key::Tab, Modifiers::NONE);
    assert_ne!(
        ctx.memory(|m| m.focused()),
        Some(canvas_focus_id),
        "Tab must advance focus out of the dose plot canvas without trapping"
    );
}

fn tab_to_dose_canvas(
    ctx: &egui::Context,
    tool: &mut DoseCalculationTool,
    screen: Option<Rect>,
) -> egui::Id {
    let _ = ctx.run(
        RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    for _ in 0..90 {
        send_key_frame(ctx, tool, screen, Key::Tab, Modifiers::NONE);
        if let Some(id) = tool.focused_canvas_id() {
            return id;
        }
    }
    panic!("Tab traversal must reach the dose plot canvas");
}

fn send_key_frame(
    ctx: &egui::Context,
    tool: &mut DoseCalculationTool,
    screen: Option<Rect>,
    key: Key,
    modifiers: Modifiers,
) {
    let _ = ctx.run(
        RawInput {
            screen_rect: screen,
            events: vec![Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers,
            }],
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
}

#[test]
fn test_sonification_meter_and_sub_30ms_safety_interlock_mute() {
    let mut tool = DoseCalculationTool::default();
    tool.apply_oar_sparing_4d_preset();
    tool.display_mode = DoseDisplayMode::Gated4d;
    let _ = tool.step_4d_in_place();

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let out = ctx.run(
        RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1200.0, 820.0))),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let has_meter = out.platform_output.accesskit_update.is_some_and(|ak| {
        ak.nodes
            .iter()
            .any(|(_, n)| n.role() == egui::accesskit::Role::Meter)
    });
    assert!(
        has_meter,
        "DoseCalculationTool must expose AccessKit Role::Meter"
    );

    let mut synth = OfflinePcmSynthesizer::new(48_000, 2_400);
    let _ = synth.set_muted_from_user_gesture(false);
    synth.set_master_volume(0.8);
    let active_frame =
        imrt_delivery_frame(BeamState::BeamActive, -2.5, 59.0, 60.0, 20.0, 26.0, 5.521);
    synth.apply_frame(&active_frame);
    let active_peak = synth
        .render_frames(2_400)
        .iter()
        .fold(0.0_f32, |a, &s| a.max(s.abs()));
    assert!(active_peak > 0.05);

    let interlock_frame = imrt_delivery_frame(
        BeamState::SafetyInterlock,
        f64::NAN,
        59.0,
        60.0,
        20.0,
        26.0,
        5.521,
    );
    assert_eq!(interlock_frame.master_gain, 0.0);
    synth.mute_and_suspend(MuteReason::SolverDiverged);
    synth.apply_frame(&interlock_frame);
    let _ = synth.render_frames(1_440);
    let residual_peak = synth
        .render_frames(480)
        .iter()
        .fold(0.0_f32, |a, &s| a.max(s.abs()));
    assert!(residual_peak < 1.0e-3);
}
