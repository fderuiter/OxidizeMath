#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::interaction::compute_split_layout;
use super::state::{BrushMode, ColormapMode, PropagationMode, ScenarioPreset};
use super::ExcitableSheetTool;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui::{self, Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect};

fn screen_rect() -> Rect {
    Rect::from_min_size(Pos2::ZERO, egui::vec2(1280.0, 800.0))
}

#[test]
fn test_excitable_sheet_on_brush_stimulates_and_paints_scars() {
    let mut tool = ExcitableSheetTool::default();
    tool.cfg.running = false;
    tool.apply_scenario(ScenarioPreset::WholeSheetDefibrillation);
    let baseline_max = tool.sheet.max_tissue_voltage();

    let ctx = egui::Context::default();
    let _ = ctx.run(
        RawInput {
            screen_rect: Some(screen_rect()),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );

    let center = Pos2::new(760.0, 260.0);
    let mut drag_start = RawInput {
        screen_rect: Some(screen_rect()),
        ..Default::default()
    };
    drag_start.events.push(Event::PointerMoved(center));
    drag_start.events.push(Event::PointerButton {
        pos: center,
        button: PointerButton::Primary,
        pressed: true,
        modifiers: Modifiers::NONE,
    });
    let _ = ctx.run(drag_start, |ctx| tool.show(ctx));

    let next = Pos2::new(790.0, 275.0);
    let mut drag_move = RawInput {
        screen_rect: Some(screen_rect()),
        ..Default::default()
    };
    drag_move.events.push(Event::PointerMoved(next));
    let _ = ctx.run(drag_move, |ctx| tool.show(ctx));

    assert!(
        tool.sheet.max_tissue_voltage() > baseline_max + 1.0,
        "on_brush electrode stroke must depolarize the 2D sheet"
    );

    tool.cfg.brush_mode = BrushMode::PaintScar;
    let mut drag_scar = RawInput {
        screen_rect: Some(screen_rect()),
        ..Default::default()
    };
    drag_scar
        .events
        .push(Event::PointerMoved(Pos2::new(805.0, 285.0)));
    let _ = ctx.run(drag_scar, |ctx| tool.show(ctx));
    assert!(
        tool.sheet.scar_mask().iter().any(|&s| s != 0),
        "PaintScar brush must populate zero-flux scar_mask"
    );
}

#[test]
fn test_excitable_sheet_lead_movement_and_dual_exports() {
    let mut tool = ExcitableSheetTool::default();
    tool.apply_scenario(ScenarioPreset::DualEdgeCollision);
    for _ in 0..10 {
        tool.step_simulation();
    }
    let before = tool.export_datasets();
    assert_eq!(tool.export_capabilities(), ExportCapabilities::plot());
    assert_eq!(before.len(), 3);
    assert!(!before[0].1.is_empty());

    tool.move_lead_to(0, 82.0, 48.0);
    let after = tool.export_datasets();
    assert_ne!(
        before[0].1.last().unwrap()[1],
        after[0].1.last().unwrap()[1],
        "Moving virtual lead 1 must update live dipole trace readout"
    );

    tool.cfg.colormap = ColormapMode::Perceptual;
    let img_perceptual = tool.export_raster_image().expect("raster export");
    tool.cfg.colormap = ColormapMode::HighContrast;
    let img_hc = tool.export_raster_image().expect("high-contrast raster export");
    assert_eq!(img_perceptual.size, [384, 384]);
    assert_ne!(img_perceptual.pixels, img_hc.pixels);
}

#[test]
fn test_excitable_sheet_keyboard_hh_cable_mode_and_finite_guard() {
    let mut tool = ExcitableSheetTool::default();
    tool.cfg.propagation_mode = PropagationMode::HodgkinHuxleyCable1D;
    for _ in 0..5 {
        tool.step_simulation();
    }
    assert!(FiniteStateGuard::is_finite_state(&tool));

    tool.sheet.voltage_mut()[0] = f64::NAN;
    tool.step_once_guarded();
    assert!(
        FiniteStateGuard::is_finite_state(&tool),
        "FiniteStateGuard must roll back non-finite voltage states automatically"
    );

    let layout = compute_split_layout(screen_rect());
    assert!(layout.sheet_rect.height() > 50.0 && layout.ecg_rect.height() > 40.0);

    let ctx = egui::Context::default();
    let _ = ctx.run(
        RawInput {
            screen_rect: Some(screen_rect()),
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    if let Some(id) = tool.focused_canvas.or_else(|| {
        ctx.memory(|m| m.focused())
    }) {
        ctx.memory_mut(|m| m.request_focus(id));
    }
    let mut key_in = RawInput {
        screen_rect: Some(screen_rect()),
        ..Default::default()
    };
    key_in.events.push(Event::Key {
        key: Key::D,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    let _ = ctx.run(key_in, |ctx| tool.show(ctx));
    assert!(tool.try_load_state(r#"{"bad": 1}"#).is_err());
}
