//! Unit and headless accessibility/export tests for [`HamiltonianChaosLab`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::preset::{
    PRESET_DOUBLE_PENDULUM_DRIFT, PRESET_FIGURE_EIGHT_CHOREOGRAPHY, PRESET_HENON_HEILES_KAM,
};
use super::{
    HamiltonianChaosLab, HamiltonianModelKind, MAX_ORBITS, MAX_SECTION_POINTS,
    SymplecticSolverChoice,
};
use crate::framework::InteractiveTool;
use eframe::egui::{self, Context, Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect};

fn raw_viewport() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

fn push_key_event(key: Key) -> RawInput {
    let mut raw = raw_viewport();
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
fn test_hamiltonian_lab_keyboard_seeding_with_focus_ring() {
    let mut lab = HamiltonianChaosLab::default();
    lab.clear_orbits();
    let ctx = Context::default();
    ctx.enable_accesskit();
    let _ = ctx.run(raw_viewport(), |ctx| lab.show(ctx));

    let mut canvas_focused = false;
    for _ in 0..60 {
        let _ = ctx.run(push_key_event(Key::Tab), |ctx| lab.show(ctx));
        if lab.focused_canvas_id().is_some() {
            canvas_focused = true;
            break;
        }
    }
    assert!(canvas_focused, "Poincare canvas must receive Tab focus");

    let prev_cursor = lab.cursor_qp;
    let _ = ctx.run(push_key_event(Key::ArrowRight), |ctx| lab.show(ctx));
    assert!(lab.cursor_qp[0] > prev_cursor[0]);

    let _ = ctx.run(push_key_event(Key::Enter), |ctx| lab.show(ctx));
    assert_eq!(lab.orbits.len(), 1, "Enter on focused canvas must seed orbit");
}

#[test]
fn test_hamiltonian_lab_pointer_click_seeding() {
    let mut lab = HamiltonianChaosLab::default();
    lab.clear_orbits();
    let ctx = Context::default();
    let _ = ctx.run(raw_viewport(), |ctx| lab.show(ctx));

    let click_pos = Pos2::new(620.0, 400.0);
    let mut press = raw_viewport();
    press.events.push(Event::PointerMoved(click_pos));
    press.events.push(Event::PointerButton {
        pos: click_pos,
        button: PointerButton::Primary,
        pressed: true,
        modifiers: Modifiers::NONE,
    });
    let _ = ctx.run(press, |ctx| lab.show(ctx));

    let mut release = raw_viewport();
    release.events.push(Event::PointerButton {
        pos: click_pos,
        button: PointerButton::Primary,
        pressed: false,
        modifiers: Modifiers::NONE,
    });
    let _ = ctx.run(release, |ctx| lab.show(ctx));
    assert_eq!(lab.orbits.len(), 1, "Pointer click on canvas must seed orbit");
}

#[test]
fn test_ensemble_and_section_caps_and_state_bounds() {
    let mut lab = HamiltonianChaosLab::default();
    lab.clear_orbits();
    for i in 0..24 {
        let q = -0.2 + (i as f64) * 0.015;
        let _ = lab.seed_orbit_no_warmup([q, 0.0]);
    }
    assert_eq!(
        lab.orbits.len(),
        MAX_ORBITS,
        "Seeded ensemble must cap at {MAX_ORBITS} orbits"
    );

    let saved = lab.save_state().expect("save_state");
    assert!(
        saved.len() <= crate::url_state::MAX_DECODED_URL_STATE_BYTES,
        "16-orbit snapshot ({}) must fit URL_JSON_BOUNDS",
        saved.len()
    );
    lab.state_bounds()
        .validate_json(&saved)
        .expect("ToolStateBounds must accept 16-orbit snapshot");

    // Verify per-orbit section buffer never exceeds MAX_SECTION_POINTS.
    if let Some(first) = lab.orbits.first_mut() {
        first.section_points.resize(MAX_SECTION_POINTS, [0.1, 0.0]);
    }
    lab.step_batch(600);
    assert!(lab.orbits[0].section_points.len() <= MAX_SECTION_POINTS);
}

#[test]
fn test_presets_csv_svg_and_url_roundtrip() {
    let mut lab = HamiltonianChaosLab::default();
    for preset in [
        PRESET_HENON_HEILES_KAM,
        PRESET_DOUBLE_PENDULUM_DRIFT,
        PRESET_FIGURE_EIGHT_CHOREOGRAPHY,
    ] {
        lab.import_preset_json(preset)
            .expect("starter preset must load cleanly");
        let datasets = lab.export_datasets();
        assert!(!datasets.is_empty(), "Preset must yield non-empty datasets");
        let csv_text = egui_plot::export::format_csv(&datasets);
        assert!(csv_text.contains("Poincare Orbit 1"));
        let parsed_csv = egui_plot::parse_csv(&csv_text);
        assert!(!parsed_csv.is_empty());

        let prims = lab.export_plot_primitives();
        let svg = egui_plot::export_primitives_svg(&prims, &egui_plot::ExportOptions::default())
            .expect("SVG export");
        assert!(svg.contains("<svg") && svg.contains("</svg>"));
    }

    // Verify DoublePendulum comparator shows GaussLegendre4 beating RK4 drift.
    lab.switch_model(HamiltonianModelKind::DoublePendulum);
    assert_eq!(lab.solver, SymplecticSolverChoice::GaussLegendre4);
    lab.energy = 36.0;
    lab.clear_orbits();
    lab.seed_orbit_at([1.2, 0.0]).expect("seed chaotic pendulum");
    lab.step_batch(2_500);
    assert!(
        lab.ghost.symplectic_err_pct < lab.ghost.rk4_err_pct,
        "GaussLegendre4 error ({:.6}%) must stay below RK4 drift ({:.4}%)",
        lab.ghost.symplectic_err_pct,
        lab.ghost.rk4_err_pct
    );
}
