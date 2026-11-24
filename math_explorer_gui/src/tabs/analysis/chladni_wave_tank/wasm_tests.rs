#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:graph_parameters_rust]

use eframe::egui::{self, vec2, Context, Event, Key, Modifiers, Pos2, RawInput, Rect};
use egui_plot::export::{color32_image_to_png, format_csv, format_json, ExportOptions};
use egui_plot::export_primitives::export_primitives_svg;
use wasm_bindgen_test::*;

use super::*;
use crate::audio::{web_audio::WebAudioNodeBackend, MuteReason, SonificationBackend};
use crate::framework::{BrushInteractionMode, BrushMode, InteractiveTool};

fn wasm_key(key: Key) -> RawInput {
    let mut raw = RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    };
    raw.events.push(Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    raw
}

#[wasm_bindgen_test]
fn wasm_smoke_chladni_wave_tank_presets_stepping_and_exports() {
    let mut tool = ChladniWaveTankTool::default();
    for preset in [
        TankPreset::SquareChladniPlate,
        TankPreset::CircularDrumhead,
        TankPreset::DoubleSlitInterference,
        TankPreset::AcousticLens,
    ] {
        tool.apply_preset(preset);
        tool.state.custom_clamps.push([0.35, 0.65]);
        tool.rebuild_masks();
        tool.step_simulation();
        assert!(tool.has_finite_parameters());
        assert!(tool.audio.latest_frame.is_audible());

        let saved = tool.save_state().expect("WASM save_state");
        tool.try_load_state(&saved).expect("WASM try_load_state");
        assert_eq!(tool.state.custom_clamps, vec![[0.35, 0.65]]);

        let img = tool.export_raster_image().expect("WASM raster image");
        let png = color32_image_to_png(&img).expect("WASM PNG encode");
        assert!(!png.is_empty());

        let prims = tool.export_plot_primitives();
        let svg = export_primitives_svg(&prims, &ExportOptions::default()).expect("WASM SVG");
        assert!(svg.contains("<svg"));

        let datasets = tool.export_datasets();
        let csv = format_csv(&datasets);
        assert!(csv.starts_with("Dataset,X,Y"));
        let json = format_json(&datasets);
        assert!(json.starts_with('['));
    }

    tool.apply_preset(TankPreset::SquareChladniPlate);
    tool.state.pickup_left = [0.30, 0.62];
    tool.state.pickup_right = [0.71, 0.39];
    let ir_wav = tool.render_complete_ir_blocking(0.5);
    let summary = crate::audio::wav::inspect_wav_riff(&ir_wav).expect("WASM IR RIFF valid");
    assert_eq!(summary.channels, 2);
    assert_eq!(summary.sample_rate_hz, 48_000);
    tool.audition_strike_from_user_gesture();
    assert!(crate::audio::studio_preview::snapshot_studio_preview().last_strike_frames > 0);

    let theory = super::atlas::chladni_parameter_constraints();
    let cart = std::sync::Arc::new(super::atlas::ChladniModalCartographer::new(
        &theory,
        ("driver_frequency", "plate_stiffness"),
        &tool.state,
        tool.domain_physical_length(),
    ));
    let cell = crate::async_sim::atlas::RegimeCartographer::evaluate_cell(&*cart, 0.565, 0.0004);
    assert!(cell.primary.is_finite());
}

#[wasm_bindgen_test]
fn wasm_chladni_sculpt_accesskit_meter_and_tab_traversal() {
    let mut tool = ChladniWaveTankTool {
        paused: true,
        ..Default::default()
    };
    tool.brush_kit.interaction_mode = BrushInteractionMode::Sculpt;
    tool.brush_kit.mode = BrushMode::PotentialBarrier;
    tool.step_simulation();

    let ctx = Context::default();
    ctx.enable_accesskit();
    let frame0 = RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    };
    let _ = ctx.run(frame0.clone(), |ctx| tool.show(ctx));
    let ak = ctx
        .run(frame0, |ctx| tool.show(ctx))
        .platform_output
        .accesskit_update
        .expect("AccessKit tree");
    assert!(ak
        .nodes
        .iter()
        .any(|(_, n)| n.role() == egui::accesskit::Role::Meter));

    let canvas_ids: Vec<egui::accesskit::NodeId> = ak
        .nodes
        .iter()
        .filter_map(|(id, n)| {
            (n.role() == egui::accesskit::Role::GraphicsDocument
                && n.label()
                    .unwrap_or_default()
                    .contains("2D Chladni Plate & Wave Tank Visualization"))
            .then_some(*id)
        })
        .collect();
    let mut focused = false;
    for _ in 0..80 {
        if ctx
            .run(wasm_key(Key::Tab), |ctx| tool.show(ctx))
            .platform_output
            .accesskit_update
            .is_some_and(|u| canvas_ids.contains(&u.focus))
        {
            focused = true;
            break;
        }
    }
    assert!(focused);
    let _ = ctx.run(wasm_key(Key::ArrowRight), |ctx| tool.show(ctx));
    let _ = ctx.run(wasm_key(Key::Space), |ctx| tool.show(ctx));
    assert!(!tool.state.sculpt_rle.is_empty());
    let saved = tool.save_state().unwrap();
    assert!(saved.len() < 4_096);
    tool.try_load_state(&saved).unwrap();

    let next_tab = ctx.run(wasm_key(Key::Tab), |ctx| tool.show(ctx));
    if let Some(upd) = next_tab.platform_output.accesskit_update {
        assert!(!canvas_ids.contains(&upd.focus));
    }
}

#[wasm_bindgen_test]
fn wasm_chladni_web_audio_backend_mute_and_pause_lifecycle() {
    let mut tool = ChladniWaveTankTool::default();
    tool.step_simulation();
    assert!(tool.audio.latest_frame.is_audible());

    let mut backend = WebAudioNodeBackend::new();
    assert!(backend.is_muted() && !backend.has_audio_context());
    backend.set_muted_from_user_gesture(false).unwrap();
    assert!(!backend.is_muted() && backend.has_audio_context());
    backend.apply_frame(&tool.audio.latest_frame);
    assert!((20.0..=8_000.0).contains(&backend.meter_reading().dominant_hz));
    backend.mute_and_suspend(MuteReason::Paused);
    assert!(backend.is_muted() && backend.last_mute_reason() == MuteReason::Paused);
    assert!(backend.meter_reading().peak_dbfs <= crate::audio::frame::FLOOR_DBFS + 1.0e-4);
}
