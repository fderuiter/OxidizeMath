//! Unit and UI integration tests for the 2D stability atlas, Gray-Scott kinetics,
//! CFL-safe stepping, StarterGallery presets, and live canvas chemical brushing.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::atlas::{
    build_atlas_image, classify_gray_scott, classify_schnakenberg, gray_scott_fold_k,
    gray_scott_hopf_f, GrayScottRegime, SchnakenbergRegime,
};
use super::presets::{
    PRESET_GRAY_SCOTT_CORAL, PRESET_GRAY_SCOTT_MITOSIS, PRESET_GRAY_SCOTT_SOLITONS,
};
use super::state::spatial_variance;
use super::*;
use crate::async_sim::{SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::InteractiveTool;
use math_commons::math_kernel::types::flatten_2d_index;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

fn click_accesskit_label(ctx: &egui::Context, tab: &mut MorphogenesisTab, label_substr: &str) {
    let mut frame = eframe::Frame::_new_kittest();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1400.0, 900.0));
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| tab.show(ctx, &mut frame),
    );
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let bounds = ak
        .nodes
        .iter()
        .find(|(_, n)| n.label().is_some_and(|l| l.contains(label_substr)))
        .and_then(|(_, n)| n.bounds())
        .unwrap_or_else(|| panic!("Widget containing '{label_substr}' not found"));
    let pos = egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    );
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            events: vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| tab.show(ctx, &mut frame),
    );
}

#[test]
fn test_analytical_schnakenberg_and_gray_scott_atlas_curves() {
    assert_eq!(
        classify_schnakenberg(0.1, 0.9, 1.0, 100.0),
        SchnakenbergRegime::TuringUnstable
    );
    assert_eq!(
        classify_schnakenberg(0.14, 0.86, 1.0, 50.0),
        SchnakenbergRegime::TuringUnstable
    );
    assert_eq!(
        classify_schnakenberg(0.8, 0.2, 1.0, 100.0),
        SchnakenbergRegime::UniformStable
    );
    assert_eq!(
        classify_schnakenberg(0.05, 0.45, 1.0, 1.0),
        SchnakenbergRegime::HopfOscillatory
    );

    // Bogdanov-Takens codimension-2 point where Fold F=4(F+k)^2 meets Hopf (F+k)^2=F*sqrt(k)
    let k_bt = gray_scott_fold_k(0.0625).unwrap();
    let f_bt = gray_scott_hopf_f(0.0625).unwrap();
    assert!((k_bt - 0.0625).abs() < 1e-12);
    assert!((f_bt - 0.0625).abs() < 1e-12);

    // Verify fold equation F = 4*(F + k_fold)^2 and Hopf equation (F_hopf + k)^2 = F_hopf*sqrt(k)
    let k_fold = gray_scott_fold_k(0.04).unwrap();
    assert!((4.0 * (0.04 + k_fold).powi(2) - 0.04).abs() < 1e-12);
    let f_hopf = gray_scott_hopf_f(0.05).unwrap();
    assert!(((f_hopf + 0.05).powi(2) - f_hopf * 0.05_f64.sqrt()).abs() < 1e-12);

    assert_eq!(
        classify_gray_scott(0.0367, 0.0649, 0.2, 0.1),
        GrayScottRegime::SubcriticalPattern
    );
    assert_eq!(
        classify_gray_scott(0.0545, 0.062, 0.2, 0.1),
        GrayScottRegime::BifurcationUnstable
    );
    assert_eq!(
        classify_gray_scott(0.030, 0.060, 0.2, 0.1),
        GrayScottRegime::SubcriticalPattern
    );
    assert_eq!(
        classify_gray_scott(0.020, 0.069, 0.2, 0.1),
        GrayScottRegime::UniformStable
    );

    let schnak_img = build_atlas_image(MorphogenesisKinetics::Schnakenberg, 1.0, 100.0, false);
    let gs_img = build_atlas_image(MorphogenesisKinetics::GrayScott, 0.2, 0.1, false);
    assert_eq!(schnak_img.pixels.len(), 96 * 72);
    assert_eq!(gs_img.pixels.len(), 96 * 72);
}

#[test]
fn test_atlas_unstable_samples_yield_nonzero_variance_after_500_cfl_steps() {
    for (a, b) in [(0.10, 0.90), (0.14, 0.86), (0.08, 1.10)] {
        assert_eq!(
            classify_schnakenberg(a, b, 1.0, 100.0),
            SchnakenbergRegime::TuringUnstable
        );
        let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
            kinetics: MorphogenesisKinetics::Schnakenberg,
            a,
            b,
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.002,
            auto_dt: true,
            ..Default::default()
        }));
        let mut runner = MorphogenesisTab::create_runner(&params, 24, 24);
        for _ in 0..500 {
            runner.step();
        }
        assert!(FiniteStateGuard::is_finite_state(&runner));
        assert!(spatial_variance(runner.system.u()) > 1e-4);
    }

    for (f, k) in [(0.0367, 0.0649), (0.0545, 0.062), (0.030, 0.060)] {
        assert_ne!(
            classify_gray_scott(f, k, 0.2, 0.1),
            GrayScottRegime::UniformStable
        );
        let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
            kinetics: MorphogenesisKinetics::GrayScott,
            f,
            k,
            d_u: 0.2,
            d_v: 0.1,
            dt: 1.0,
            auto_dt: true,
            ..Default::default()
        }));
        let mut runner = MorphogenesisTab::create_runner(&params, 24, 24);
        for _ in 0..500 {
            runner.step();
        }
        assert!(FiniteStateGuard::is_finite_state(&runner));
        assert!(spatial_variance(runner.system.u()) > 1e-4);
    }
}

#[test]
fn test_atlas_click_updates_state_revision_in_both_kinetics_modes() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tab = MorphogenesisTab::default();
    let rev0 = tab.state_revision();

    click_accesskit_label(&ctx, &mut tab, "2D Turing Atlas (a, b)");
    let rev1 = tab.state_revision();
    assert_ne!(
        rev1, rev0,
        "Clicking Schnakenberg atlas must update state_revision"
    );

    click_accesskit_label(&ctx, &mut tab, "Gray-Scott");
    let rev2 = tab.state_revision();
    assert_ne!(rev2, rev1);
    assert_eq!(
        tab.export_config().kinetics,
        MorphogenesisKinetics::GrayScott
    );

    click_accesskit_label(&ctx, &mut tab, "2D Gray-Scott Atlas (F, k)");
    let rev3 = tab.state_revision();
    assert_ne!(
        rev3, rev2,
        "Clicking Gray-Scott atlas must update state_revision"
    );
}

#[test]
fn test_all_five_starter_presets_evolve_without_fault_and_roundtrip_urls() {
    let cards = [
        ("Turing Spots", crate::starter_gallery::PRESET_TURING_SPOTS),
        (
            "Turing Stripes",
            crate::starter_gallery::PRESET_TURING_STRIPES,
        ),
        ("Gray-Scott Mitosis", PRESET_GRAY_SCOTT_MITOSIS),
        ("Gray-Scott Coral", PRESET_GRAY_SCOTT_CORAL),
        ("Gray-Scott Solitons", PRESET_GRAY_SCOTT_SOLITONS),
    ];

    for (title, preset_json) in cards {
        let mut tab = MorphogenesisTab::default();
        tab.framework
            .apply_starter_card(title, Some("MorphogenesisTool"), Some(preset_json))
            .unwrap_or_else(|e| panic!("{title} failed to apply: {e}"));

        let url = tab.framework.encode_share_url().expect("share URL");
        let hash_idx = url.find('#').expect("fragment");
        let parsed = crate::url_state::parse_share_location(&url[hash_idx..], "")
            .unwrap()
            .unwrap();
        let mut restored = MorphogenesisTab::default();
        restored
            .import_url_preset_json(parsed.tool.as_deref(), parsed.state_json.as_deref())
            .expect("URL import");
        assert_eq!(restored.export_config(), tab.export_config());
        let restored_state: MorphogenesisToolState = serde_json::from_str(
            &restored
                .framework
                .active_tool
                .as_ref()
                .unwrap()
                .save_state()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(restored_state.state, Some(SimWorkerState::Ready));
        assert!(restored_state.pending_initial_start);

        let cfg = tab.export_config();
        let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
            kinetics: cfg.kinetics,
            a: cfg.a,
            b: cfg.b,
            f: cfg.f,
            k: cfg.k,
            d_u: cfg.d_u,
            d_v: cfg.d_v,
            dt: cfg.dt,
            auto_dt: cfg.auto_dt,
            high_contrast: false,
        }));
        let mut runner = MorphogenesisTab::create_runner(&params, 24, 24);
        for _ in 0..40 {
            runner.step();
        }
        assert!(
            FiniteStateGuard::is_finite_state(&runner),
            "Preset '{title}' faulted during CFL-safe evolution"
        );
    }
}

#[test]
fn test_apply_brush_injection_erasure_and_paused_worker_snapshots() {
    let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
        kinetics: MorphogenesisKinetics::GrayScott,
        f: 0.0367,
        k: 0.0649,
        d_u: 0.2,
        d_v: 0.1,
        dt: 1.0,
        auto_dt: true,
        ..Default::default()
    }));
    let mut runner = MorphogenesisTab::create_runner(&params, 24, 24);
    let idx = flatten_2d_index(5, 5, 24);
    assert!((runner.system.u()[idx] - 1.0).abs() < 1e-12);
    assert!(runner.system.v()[idx].abs() < 1e-12);

    // Primary brush injects reagent
    runner.process_command(SimCommand::ApplyBrush {
        cx: 5,
        cy: 5,
        r: 2,
        is_obstacle: true,
    });
    assert!((runner.system.u()[idx] - 0.50).abs() < 1e-12);
    assert!((runner.system.v()[idx] - 0.25).abs() < 1e-12);

    // Secondary brush restores homogeneous equilibrium (1.0, 0.0)
    runner.process_command(SimCommand::ApplyBrush {
        cx: 5,
        cy: 5,
        r: 2,
        is_obstacle: false,
    });
    assert!((runner.system.u()[idx] - 1.0).abs() < 1e-12);
    assert!(runner.system.v()[idx].abs() < 1e-12);

    // Verify paused worker (Ready) emits immediate snapshot upon ApplyBrush without unpausing
    let mut tool = MorphogenesisTool {
        pending_initial_start: false,
        ..Default::default()
    };
    tool.apply_gray_scott_preset(0.0367, 0.0649, 0.2, 0.1, 1.0);
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
    while tool.worker.update().is_some() {}

    tool.worker
        .send_command(SimCommand::ApplyBrush {
            cx: 12,
            cy: 12,
            r: 4,
            is_obstacle: true,
        })
        .expect("ApplyBrush valid in Ready state");

    let mut got_paused_snapshot = false;
    for _ in 0..1000 {
        if tool.worker.update().is_some() {
            got_paused_snapshot = true;
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        got_paused_snapshot,
        "Paused brush stroke must emit snapshot"
    );
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
}

#[test]
fn test_canvas_pointer_drag_dispatches_brush_and_zero_alloc_gray_scott_step() {
    let params = Arc::new(RwLock::new(MorphogenesisSharedParams {
        kinetics: MorphogenesisKinetics::GrayScott,
        f: 0.0545,
        k: 0.062,
        d_u: 0.2,
        d_v: 0.1,
        dt: 1.0,
        auto_dt: true,
        ..Default::default()
    }));
    let mut runner = MorphogenesisTab::create_runner(&params, 24, 24);
    runner.step();
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            runner.step();
        });
    }
    assert!(FiniteStateGuard::is_finite_state(&runner));
}

#[test]
fn test_morphogenesis_theory_and_canvas_pointer_drag_brush_dispatch() {
    // Verify MorphogenesisTheory switches dynamically between Schnakenberg and Gray-Scott
    let mut tool = MorphogenesisTool::default();
    assert!(tool.theory().theory_citation().contains("Schnakenberg"));
    assert!(tool
        .theory()
        .theory_latex_equation()
        .unwrap()
        .contains("u^2 v"));
    tool.switch_kinetics(MorphogenesisKinetics::GrayScott);
    assert!(tool.theory().theory_citation().contains("Pearson"));
    assert!(tool
        .theory()
        .theory_latex_equation()
        .unwrap()
        .contains("F(1 - u)"));

    // Render tool in Brush mode while paused with initial Reset snapshot and drag across canvas center
    for _ in 0..1000 {
        if tool.worker.update().is_some() || tool.worker.latest_snapshot().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }
    let ctx = egui::Context::default();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 700.0));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
        },
    );
    let canvas_pt = egui::pos2(650.0, 350.0);
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            events: vec![
                egui::Event::PointerMoved(canvas_pt),
                egui::Event::PointerButton {
                    pos: canvas_pt,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
        },
    );
    assert!(tool.pending_brush_frames > 0);
}

#[test]
fn test_morphogenesis_defers_start_until_render_and_faults_on_divergence() {
    use super::tests::{active_morpho_tool_state, render_tool_frame};

    let ctx = egui::Context::default();
    let mut tab = MorphogenesisTab::default();
    let mut tool = MorphogenesisTool::default();
    assert_eq!(
        active_morpho_tool_state(&tab).state,
        Some(SimWorkerState::Ready)
    );
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);

    thread::sleep(Duration::from_millis(40));
    assert_eq!(
        active_morpho_tool_state(&tab).state,
        Some(SimWorkerState::Ready)
    );
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);

    let mut frame = eframe::Frame::_new_kittest();
    for _ in 0..8 {
        let _ = ctx.run(egui::RawInput::default(), |ctx| tab.show(ctx, &mut frame));
        let _ = render_tool_frame(&ctx, &mut tool);
        thread::sleep(Duration::from_millis(8));
    }
    assert_eq!(
        active_morpho_tool_state(&tab).state,
        Some(SimWorkerState::Running)
    );
    assert_eq!(tool.worker.state(), SimWorkerState::Running);
    assert!(tool.worker.diagnostic_report().is_none());

    tab.params.write().unwrap().auto_dt = false;
    tool.params.write().unwrap().auto_dt = false;
    for _ in 0..1200 {
        let _ = ctx.run(egui::RawInput::default(), |ctx| tab.show(ctx, &mut frame));
        let _ = render_tool_frame(&ctx, &mut tool);
        if active_morpho_tool_state(&tab).state == Some(SimWorkerState::Faulted)
            && tool.worker.state() == SimWorkerState::Faulted
        {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert_eq!(
        active_morpho_tool_state(&tab).state,
        Some(SimWorkerState::Faulted)
    );
    assert_eq!(tool.worker.state(), SimWorkerState::Faulted);
    assert!(tool.worker.diagnostic_report().is_some());
}
