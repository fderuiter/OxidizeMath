//! Headless `wasm32-unknown-unknown` verification for ChaosTab Hamiltonian, Billiards, and Bifurcation tools.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::bifurcation::BifurcationDiagram;
use super::billiards::{BilliardGeometryKind, DynamicalBilliardsTool};
use super::hamiltonian::{HamiltonianChaosTool, HamiltonianModelKind, SymplecticSolverChoice};
use crate::framework::InteractiveTool;
use eframe::egui::{self, Context, Pos2, RawInput, Rect};
use wasm_bindgen_test::wasm_bindgen_test;

fn wasm_viewport() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

#[wasm_bindgen_test]
fn wasm_chaos_hamiltonian_billiards_and_lyapunov_smoke() {
    let ctx = Context::default();

    // 1. HamiltonianChaosTool: Yoshida4 Henon-Heiles, ImplicitMidpoint Pendulum Fan, Chirikov Map
    let mut ham = HamiltonianChaosTool::default();
    let _ = ctx.run(wasm_viewport(), |ctx| ham.show(ctx));
    ham.step_batch(120);
    assert!(ham.has_finite_parameters());

    ham.switch_model(HamiltonianModelKind::DoublePendulum);
    ham.solver = SymplecticSolverChoice::ImplicitMidpoint;
    ham.cursor_qp = [0.75, 0.0];
    assert_eq!(ham.seed_pendulum_fan().expect("fan"), 8);
    ham.step_batch(80);

    ham.switch_model(HamiltonianModelKind::ChirikovStandardMap);
    ham.step_batch(100);
    assert!(!ham.export_datasets().is_empty());

    // 2. DynamicalBilliardsTool: Ellipse, Stadium, and Sinai Birkhoff portraits
    let mut bil = DynamicalBilliardsTool::default();
    for geom in [
        BilliardGeometryKind::Ellipse,
        BilliardGeometryKind::Stadium,
        BilliardGeometryKind::Sinai,
    ] {
        bil.switch_geometry(geom);
        bil.advance_bounces(60);
        let _ = ctx.run(wasm_viewport(), |ctx| bil.show(ctx));
        assert!(!bil.export_datasets().is_empty());
    }

    // 3. BifurcationDiagram: Lyapunov overlay
    let mut bif = BifurcationDiagram::default();
    bif.show_lyapunov = true;
    bif.probe_r = 4.0;
    bif.recompute();
    let _ = ctx.run(wasm_viewport(), |ctx| bif.show(ctx));
    assert!(bif.probe_lyapunov().unwrap_or(0.0) > 0.5);
}
