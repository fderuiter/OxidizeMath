//! Unit and headless accessibility/export tests for [`HamiltonianChaosTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::preset::{
    PRESET_CHIRIKOV_STANDARD_MAP, PRESET_DOUBLE_PENDULUM_DRIFT, PRESET_FIGURE_EIGHT_CHOREOGRAPHY,
    PRESET_HENON_HEILES_KAM,
};
use super::{
    HamiltonianChaosLab, HamiltonianChaosTool, HamiltonianModelKind, SymplecticSolverChoice,
    MAX_ORBITS, MAX_SECTION_POINTS,
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
    assert_eq!(
        lab.orbits.len(),
        1,
        "Enter on focused canvas must seed orbit"
    );
}

#[test]
fn test_hamiltonian_lab_pointer_click_seeding_and_out_of_shell_rejection() {
    let mut lab = HamiltonianChaosTool::default();
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
    assert_eq!(
        lab.orbits.len(),
        1,
        "Pointer click inside Hill shell must seed orbit"
    );

    // Clicking / seeding outside the active Hill energy shell must NOT mutate orbits
    // and must emit an inline "Cannot seed..." notice.
    let prev_points = lab.orbits[0].section_points.clone();
    lab.seed_and_announce(&ctx, [0.85, 0.58]);
    assert_eq!(lab.orbits.len(), 1, "Out-of-shell click must not add orbit");
    assert_eq!(lab.orbits[0].section_points, prev_points);
    assert!(
        lab.last_status_note
            .as_deref()
            .unwrap_or("")
            .starts_with("Cannot seed"),
        "Out-of-shell click must set inline rejection notice: {:?}",
        lab.last_status_note
    );
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

    if let Some(first) = lab.orbits.first_mut() {
        first.section_points.resize(MAX_SECTION_POINTS, [0.1, 0.0]);
    }
    lab.step_batch(600);
    assert!(lab.orbits[0].section_points.len() <= MAX_SECTION_POINTS);
}

#[test]
fn test_presets_csv_svg_and_implicit_midpoint_fan_and_chirikov() {
    let mut lab = HamiltonianChaosTool::default();
    for preset in [
        PRESET_HENON_HEILES_KAM,
        PRESET_DOUBLE_PENDULUM_DRIFT,
        PRESET_FIGURE_EIGHT_CHOREOGRAPHY,
        PRESET_CHIRIKOV_STANDARD_MAP,
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

    // Verify DoublePendulum ImplicitMidpoint ensemble fan & GaussLegendre4 vs RK4 drift.
    lab.switch_model(HamiltonianModelKind::DoublePendulum);
    assert_eq!(lab.solver, SymplecticSolverChoice::GaussLegendre4);
    lab.solver = SymplecticSolverChoice::ImplicitMidpoint;
    lab.energy = 34.0;
    lab.cursor_qp = [0.85, 0.0];
    let fan_count = lab.seed_pendulum_fan().expect("seed pendulum fan");
    assert_eq!(fan_count, 8);
    lab.step_batch(2_000);
    assert!(
        lab.ghost.symplectic_err_pct < 0.5,
        "ImplicitMidpoint pendulum fan energy oscillation must stay bounded: {:.4}%",
        lab.ghost.symplectic_err_pct
    );

    // Verify Figure-8 momentum perturbation slider updates ensemble cleanly.
    lab.switch_model(HamiltonianModelKind::FigureEightThreeBody);
    lab.three_body_perturbation = 0.04;
    lab.apply_figure_eight_perturbation();
    assert_eq!(lab.orbits.len(), 2);
}

#[test]
fn test_implicit_midpoint_perturbation_lyapunov_and_nan_recovery() {
    let mut lab = HamiltonianChaosLab::default();
    lab.switch_model(HamiltonianModelKind::DoublePendulum);
    lab.solver = SymplecticSolverChoice::ImplicitMidpoint;
    lab.clear_orbits();
    lab.seed_orbit_at([1.1, 0.0]).expect("seed midpoint orbit");
    lab.step_batch(1_200);
    assert!(lab.lyapunov_estimate.is_finite());
    assert!(lab.ghost.symplectic_err_pct < lab.ghost.rk4_err_pct);

    let encoded = lab.save_state().expect("save midpoint state");
    let mut restored = HamiltonianChaosLab::default();
    restored
        .try_load_state(&encoded)
        .expect("load midpoint state");
    assert_eq!(restored.solver, SymplecticSolverChoice::ImplicitMidpoint);

    lab.switch_model(HamiltonianModelKind::FigureEightThreeBody);
    lab.three_body_perturbation = 0.04;
    lab.rebuild_existing_seeds();
    lab.step_batch(400);
    let lead = lab.orbits.last().expect("3-body orbit");
    let p_tot = lab.three_body.total_linear_momentum(&lead.state12);
    assert!(p_tot[0].hypot(p_tot[1]) < 1e-11);

    lab.dt = f64::NAN;
    lab.three_body_perturbation = f64::INFINITY;
    lab.orbits[0].state12.0[0] = f64::NAN;
    assert!(!lab.has_finite_parameters());
    assert!(lab.clamp_parameters());
    assert!(lab.has_finite_parameters());
}

fn pump_gif_export(ctx: &Context, fw: &mut crate::framework::SimulationFramework) {
    for _ in 0..250 {
        if fw.active_gif_progress().is_none() {
            break;
        }
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(std::time::Duration::from_millis(10));
        let _ = ctx.run(raw_viewport(), |ctx| {
            fw.show(ctx, "hamiltonian_export_fw");
        });
    }
}

#[test]
fn test_simulation_framework_exports_csv_json_png_svg_gif() {
    use crate::framework::{ExportFormat, ExportStatusSeverity, SimulationFramework};

    let mut fw = SimulationFramework::new("__hamiltonian_export_test__");
    fw.export_options.width = 240;
    fw.export_options.height = 180;
    fw.export_gif_frames = 5;
    fw.active_tool = Some(Box::new(HamiltonianChaosLab::default()));

    let ctx = Context::default();
    for fmt in [
        ExportFormat::Csv,
        ExportFormat::Json,
        ExportFormat::Png,
        ExportFormat::Svg,
        ExportFormat::Gif,
    ] {
        let _ = ctx.run(raw_viewport(), |ctx| {
            egui_plot::clear_last_plot_datasets();
            fw.queue_export(fmt);
            fw.show(ctx, "hamiltonian_export_fw");
        });
        if fmt == ExportFormat::Gif {
            pump_gif_export(&ctx, &mut fw);
        }
        assert_eq!(
            fw.export_status_severity,
            ExportStatusSeverity::Success,
            "Export {fmt:?} failed: {:?}",
            fw.export_status_message
        );
        let bytes = fw
            .export_queue
            .last_exported_bytes
            .as_deref()
            .expect("exported bytes");
        assert!(!bytes.is_empty());
        match fmt {
            ExportFormat::Png => assert_eq!(&bytes[..4], b"\x89PNG"),
            ExportFormat::Gif => assert_eq!(&bytes[..4], b"GIF8"),
            ExportFormat::Svg => assert!(std::str::from_utf8(bytes).unwrap().contains("<svg")),
            ExportFormat::Json => assert!(std::str::from_utf8(bytes).unwrap().contains("Poincare")),
            ExportFormat::Csv => assert!(std::str::from_utf8(bytes).unwrap().contains("Poincare")),
            _ => unreachable!(),
        }
    }
}
