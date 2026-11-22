#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::state::{
    frame_sweep_mutation_count, load_slider_settings, load_slider_state, reset_frame_sweep_flags,
    save_slider_settings, save_slider_state, AnimatedSliderSettings, AnimatedSliderState,
    SweepMode,
};
use super::AnimatedSlider;
use eframe::egui;

const SCREEN_RECT: Option<egui::Rect> = Some(egui::Rect {
    min: egui::Pos2::ZERO,
    max: egui::pos2(800.0, 600.0),
});

fn dt_input(dt: f32) -> egui::RawInput {
    egui::RawInput {
        predicted_dt: dt,
        screen_rect: SCREEN_RECT,
        ..Default::default()
    }
}

#[test]
fn test_unclamped_soft_entry_and_finite_hard_bounds() {
    let ctx = egui::Context::default();

    // 1. Value outside soft bounds [0.0, 10.0] but within hard_bounds [-50.0, 50.0] is NOT clamped to 10.0
    let mut val = 27.5;
    let _ = ctx.run(dt_input(0.016), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp = ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .hard_bounds(-50.0..=50.0)
                    .text("Gain")
                    .id_salt("unclamped_soft_test"),
            );
            assert!(!resp.changed());
        });
    });
    assert!((val - 27.5).abs() < 1e-12);

    // 2. Value exceeding upper hard bound clamps strictly to hard_max (50.0)
    let mut over_val = 125.0;
    let _ = ctx.run(dt_input(0.016), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp = ui.add(
                AnimatedSlider::new(&mut over_val, 0.0..=10.0)
                    .hard_bounds(-50.0..=50.0)
                    .text("Gain")
                    .id_salt("hard_upper_clamp_test"),
            );
            assert!(resp.changed());
        });
    });
    assert_eq!(over_val, 50.0);

    // 3. Value below lower hard bound clamps strictly to hard_min (-50.0)
    let mut under_val = -999.0;
    let _ = ctx.run(dt_input(0.016), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut under_val, 0.0..=10.0)
                    .hard_bounds(-50.0..=50.0)
                    .text("Gain")
                    .id_salt("hard_lower_clamp_test"),
            );
        });
    });
    assert_eq!(under_val, -50.0);

    // 4. Non-finite inputs (NaN / Inf) sanitize to finite fallback inside hard_bounds
    let mut nan_val = f64::NAN;
    let _ = ctx.run(dt_input(0.016), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut nan_val, 2.0..=8.0)
                    .hard_bounds(0.5..=20.0)
                    .text("Safe")
                    .id_salt("nan_guard_test"),
            );
        });
    });
    assert!(nan_val.is_finite());
    assert_eq!(nan_val, 2.0);
}

fn run_single_sweep_step(
    ctx: &egui::Context,
    salt: &'static str,
    mode: SweepMode,
    start_val: f64,
) -> (f64, bool) {
    let id = egui::Id::new(salt);
    save_slider_state(
        ctx,
        id,
        AnimatedSliderState {
            playing: true,
            direction: if matches!(mode, SweepMode::Decreasing) {
                -1.0
            } else {
                1.0
            },
            sweep_mode: mode,
        },
    );
    let mut val = start_val;
    let _ = ctx.run(dt_input(0.1), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .speed(10.0)
                    .sweep_mode(mode)
                    .id_salt(salt),
            );
        });
    });
    (val, load_slider_state(ctx, id, mode).playing)
}

#[test]
fn test_sweep_modes_increasing_decreasing_and_once() {
    let ctx = egui::Context::default();
    let (inc_val, inc_playing) =
        run_single_sweep_step(&ctx, "sweep_increasing", SweepMode::Increasing, 9.5);
    assert_eq!(inc_val, 0.0);
    assert!(inc_playing);

    let (dec_val, dec_playing) =
        run_single_sweep_step(&ctx, "sweep_decreasing", SweepMode::Decreasing, 0.4);
    assert_eq!(dec_val, 10.0);
    assert!(dec_playing);

    reset_frame_sweep_flags(&ctx);
    let (once_val, once_playing) = run_single_sweep_step(&ctx, "sweep_once", SweepMode::Once, 9.5);
    assert_eq!(once_val, 10.0);
    assert!(!once_playing);
    assert_eq!(frame_sweep_mutation_count(&ctx), 1);
}

#[test]
fn test_persisted_state_and_soft_bounds_scoped_by_pane_and_owner() {
    let ctx = egui::Context::default();
    let (mut v_pane0, mut v_pane1) = (1.0, 1.0);
    let (mut id_p0, mut id_p1) = (egui::Id::NULL, egui::Id::NULL);

    let _ = ctx.run(dt_input(0.016), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.ctx()
                .data_mut(|d| d.insert_temp(egui::Id::new("CURRENT_RENDERING_PANE"), 0_usize));
            id_p0 = super::resolve_scoped_slider_id(ui, Some(egui::Id::new("shared_k")), "k");
            save_slider_settings(
                ui.ctx(),
                id_p0,
                AnimatedSliderSettings {
                    min: Some(-2.0),
                    max: Some(4.0),
                    step: Some(0.25),
                    speed: Some(3.0),
                },
            );
            save_slider_state(
                ui.ctx(),
                id_p0,
                AnimatedSliderState {
                    playing: true,
                    direction: 1.0,
                    sweep_mode: SweepMode::Increasing,
                },
            );
            ui.add(
                AnimatedSlider::new(&mut v_pane0, -5.0..=5.0)
                    .text("k")
                    .id_salt("shared_k"),
            );

            ui.ctx()
                .data_mut(|d| d.insert_temp(egui::Id::new("CURRENT_RENDERING_PANE"), 1_usize));
            id_p1 = super::resolve_scoped_slider_id(ui, Some(egui::Id::new("shared_k")), "k");
            ui.add(
                AnimatedSlider::new(&mut v_pane1, -5.0..=5.0)
                    .text("k")
                    .id_salt("shared_k"),
            );
            ui.ctx()
                .data_mut(|d| d.remove::<usize>(egui::Id::new("CURRENT_RENDERING_PANE")));
        });
    });

    assert_ne!(id_p0, id_p1);
    assert_eq!(load_slider_settings(&ctx, id_p0).max, Some(4.0));
    assert_eq!(load_slider_settings(&ctx, id_p1).max, None);
    assert!(v_pane0 > 1.0);
    assert_eq!(v_pane1, 1.0);
    verify_persisted_serde_roundtrip(&ctx, id_p0);
}

fn verify_persisted_serde_roundtrip(ctx: &egui::Context, id_p0: egui::Id) {
    let cloned_data = ctx.memory(|m| m.data.clone());
    let ctx2 = egui::Context::default();
    ctx2.memory_mut(|m| m.data = cloned_data);

    let restored_settings: AnimatedSliderSettings = serde_json::from_str(
        &serde_json::to_string(&load_slider_settings(&ctx2, id_p0)).expect("serialize settings"),
    )
    .expect("deserialize settings");
    let restored_state: AnimatedSliderState = serde_json::from_str(
        &serde_json::to_string(&load_slider_state(&ctx2, id_p0, SweepMode::Oscillate))
            .expect("serialize state"),
    )
    .expect("deserialize state");
    assert_eq!(restored_settings.min, Some(-2.0));
    assert_eq!(restored_settings.max, Some(4.0));
    assert_eq!(restored_settings.step, Some(0.25));
    assert_eq!(restored_settings.speed, Some(3.0));
    assert_eq!(restored_state.sweep_mode, SweepMode::Increasing);
    assert!(restored_state.playing);
}

#[test]
fn test_settings_popover_accessibility_open_and_narrow_sidebar() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let id = egui::Id::new("narrow_gear_slider");
    let mut val = 2.0;

    let _ = ctx.run(dt_input(0.016), |ctx| {
        egui::SidePanel::left("narrow_test_panel")
            .exact_width(176.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add(
                    AnimatedSlider::new(&mut val, 0.0..=10.0)
                        .hard_bounds(0.0..=20.0)
                        .step_by(0.1)
                        .text("Sodium Conductance (mS/cm²)")
                        .id_salt("narrow_gear_slider"),
                );
            });
    });

    let gear_info: egui::WidgetInfo = ctx
        .data(|d| d.get_temp(id.with("gear_button_info")))
        .expect("gear_button_info must be registered");
    assert_eq!(gear_info.typ, egui::WidgetType::Button);
    assert_eq!(
        gear_info.label.as_deref(),
        Some("Configure range and animation for Sodium Conductance (mS/cm²)")
    );

    save_slider_settings(
        &ctx,
        id,
        AnimatedSliderSettings {
            min: Some(1.0),
            max: Some(6.0),
            step: Some(0.5),
            speed: Some(2.5),
        },
    );
    egui::Popup::open_id(&ctx, id.with("settings_popup"));

    let out = ctx.run(dt_input(0.016), |ctx| {
        egui::SidePanel::left("narrow_test_panel")
            .exact_width(176.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add(
                    AnimatedSlider::new(&mut val, 0.0..=20.0)
                        .hard_bounds(0.0..=20.0)
                        .text("Sodium Conductance (mS/cm²)")
                        .id_salt("narrow_gear_slider"),
                );
            });
    });
    assert!(egui::Popup::is_id_open(&ctx, id.with("settings_popup")));
    assert!(!out.shapes.is_empty());
}

#[test]
fn test_settings_popover_gear_click_announces_aria_status() {
    let ctx = egui::Context::default();
    let id2 = egui::Id::new("click_gear_slider");
    let mut val = 2.0;

    let _ = ctx.run(dt_input(0.016), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .text("Omega")
                    .id_salt("click_gear_slider"),
            );
        });
    });

    let gear_center = egui::pos2(36.0, 16.0);
    for pressed in [true, false] {
        let ev = egui::Event::PointerButton {
            pos: gear_center,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        let _ = ctx.run(
            egui::RawInput {
                events: vec![ev],
                screen_rect: SCREEN_RECT,
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.add(
                        AnimatedSlider::new(&mut val, 0.0..=10.0)
                            .text("Omega")
                            .id_salt("click_gear_slider"),
                    );
                });
            },
        );
    }
    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(
        aria_msg.as_deref(),
        Some("Opened slider settings for Omega")
    );
    assert!(egui::Popup::is_id_open(&ctx, id2.with("settings_popup")));
}
