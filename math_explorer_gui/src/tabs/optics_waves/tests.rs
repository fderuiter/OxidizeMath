#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::overlays::optics_overlay_palette;
use super::presets::{PRESET_OPTICS_DOUBLE_SLIT, PRESET_OPTICS_LUNEBURG, PRESET_OPTICS_VESELAGO};
use super::runner::{OpticsSharedParams, OpticsWaveRunner};
use super::state::OpticsWaveToolState;
use super::tool::OpticsWaveTool;
use super::OpticsWaveTab;
use crate::app::theme::contrast_ratio;
use crate::app::ThemeMode;
use crate::async_sim::SimulationRunner;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
use crate::tabs::ExplorerTab;
use eframe::egui::{self, vec2, Color32, Context, Event, Key, Modifiers, Pos2, RawInput, Rect};
use math_explorer::physics::optics::refraction::OpticsPreset;
use std::sync::Arc;

fn raw_frame() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

fn key_frame(key: Key) -> RawInput {
    let mut raw = raw_frame();
    raw.events.push(Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    raw
}

#[test]
fn test_optics_runner_step_zero_heap_allocation() {
    let params = Arc::new(std::sync::RwLock::new(OpticsSharedParams::default()));
    let mut runner = OpticsWaveRunner::new(params, 80);
    // Warm up once before locking allocations.
    runner.step();

    let (ok, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        for _ in 0..8 {
            runner.step();
        }
        SimulationRunner::is_finite_state(&runner)
    });
    assert!(ok);
    assert!(runner.last_readout.lambertian_intensity > 0.0);
    assert!(runner.last_readout.lws_range_resolution > 0.0);
}

#[test]
fn test_optics_tab_renders_across_themes_with_wcag_aa_contrast() {
    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let ctx = Context::default();
        theme.apply_to_context(&ctx);
        let mut tool = OpticsWaveTool::default();
        let _ = ctx.run(raw_frame(), |ctx| tool.show(ctx));
        let _ = ctx.run(raw_frame(), |ctx| tool.show(ctx));

        let pal = optics_overlay_palette(theme);
        let panel_bg = ctx.style().visuals.panel_fill;
        for (name, fg) in [
            ("ray", pal.ray),
            ("caustic", pal.caustic),
            ("boundary", pal.boundary),
            ("detector", pal.detector),
        ] {
            let ratio_halo = contrast_ratio(fg, pal.halo);
            let ratio_panel = contrast_ratio(fg, panel_bg);
            assert!(
                ratio_halo >= 3.0,
                "{theme:?} overlay '{name}' vs halo contrast {ratio_halo:.2} must be >= 3.0:1"
            );
            assert!(
                ratio_panel >= 3.0,
                "{theme:?} overlay '{name}' vs panel contrast {ratio_panel:.2} must be >= 3.0:1"
            );
        }

        if theme == ThemeMode::HighContrast {
            let img = tool.export_raster_image().expect("raster image present");
            for px in &img.pixels {
                let ratio = contrast_ratio(*px, Color32::BLACK);
                assert!(
                    ratio >= 3.0,
                    "HighContrast pixel {px:?} must exceed 3.0:1 contrast on #000000 (got {ratio:.2})"
                );
            }
        }
    }
}

#[test]
fn test_optics_canvas_focus_arrow_nudge_and_no_tab_trap() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let mut tool = OpticsWaveTool::default();
    let _ = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let out = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit update emitted");

    let image_nodes: Vec<egui::accesskit::NodeId> = ak
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            (node.role() == egui::accesskit::Role::Image
                && node
                    .label()
                    .unwrap_or_default()
                    .contains("2D Wave & Caustics Playground")
                && node.supports_action(egui::accesskit::Action::Focus))
            .then_some(*id)
        })
        .collect();
    assert!(
        !image_nodes.is_empty(),
        "Role::Image node must be focusable"
    );

    let mut focused = false;
    for _ in 0..80 {
        let step = ctx.run(key_frame(Key::Tab), |ctx| tool.show(ctx));
        if let Some(upd) = step.platform_output.accesskit_update {
            if image_nodes.contains(&upd.focus) {
                focused = true;
                break;
            }
        }
    }
    assert!(focused, "Wave canvas must be reachable via Key::Tab");

    let before_x = tool.params.read().unwrap().detector_x;
    let _ = ctx.run(key_frame(Key::ArrowRight), |ctx| tool.show(ctx));
    let after_x = tool.params.read().unwrap().detector_x;
    assert!(
        (after_x - (before_x + 0.02)).abs() < 1e-6,
        "ArrowRight should nudge detector_x from {before_x} to {}",
        before_x + 0.02
    );

    // Pressing Tab again must advance focus away from the canvas (no focus trap).
    let next_tab = ctx.run(key_frame(Key::Tab), |ctx| tool.show(ctx));
    if let Some(upd) = next_tab.platform_output.accesskit_update {
        assert!(
            !image_nodes.contains(&upd.focus),
            "Tab must leave the wave canvas without trapping keyboard focus"
        );
    }
}

#[test]
fn test_presets_validation_and_finite_state_guard_recovery() {
    let mut tab = OpticsWaveTab::default();
    assert_eq!(tab.name(), "Optics & Waves");
    assert_eq!(tab.academic_domain(), "Physics");
    assert_eq!(tab.url_slug(), "optics_waves");

    let mut tool = OpticsWaveTool::default();
    for (preset_json, expected_preset) in [
        (PRESET_OPTICS_LUNEBURG, OpticsPreset::LuneburgLens),
        (PRESET_OPTICS_DOUBLE_SLIT, OpticsPreset::DoubleSlit),
        (PRESET_OPTICS_VESELAGO, OpticsPreset::VeselagoLens),
    ] {
        tool.import_preset_json(preset_json)
            .expect("starter preset must load cleanly");
        assert_eq!(tool.params.read().unwrap().preset, expected_preset);
        let exported = tool.export_preset_json().expect("export succeeds");
        let mut copy = OpticsWaveTool::default();
        copy.import_preset_json(&exported)
            .expect("re-import succeeds");
        assert_eq!(copy.save_state(), tool.save_state());
    }

    // Out-of-bounds grid sizes must be rejected by OpticsWaveToolState::validate().
    let mut bad_state: OpticsWaveToolState =
        serde_json::from_str(&tool.save_state().unwrap()).unwrap();
    bad_state.grid_size = 32;
    assert!(bad_state.validate().is_err());
    bad_state.grid_size = 512;
    assert!(bad_state.validate().is_err());

    // Simulate non-finite fault and verify FiniteStateGuard recovery restores finiteness.
    {
        let mut p = tool.params.write().unwrap();
        p.frequency = f64::NAN;
        p.force_non_finite = true;
    }
    assert!(!tool.has_finite_parameters());
    assert!(FiniteStateGuard::diverged(&tool));
    FiniteStateGuard::clamp_parameters(&mut tool);
    assert!(tool.has_finite_parameters());
    assert!(FiniteStateGuard::is_finite_state(&tool));

    FiniteStateGuard::auto_scale_step_size(&mut tab);
    FiniteStateGuard::reset_parameters(&mut tab);
    assert!(FiniteStateGuard::is_finite_state(&tab));
}
