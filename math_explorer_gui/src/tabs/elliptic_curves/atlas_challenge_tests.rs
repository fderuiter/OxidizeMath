#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::atlas::{WeierstrassAtlasRegime, WeierstrassRegimeCartographer};
use super::chord_tangent_tool::{CurveFieldMode, WeierstrassPlaygroundTool};
use super::handoff::{request_elliptic_torus_handoff, EllipticHandoffTarget, EllipticTorusHandoff};
use super::modular_invariants_tool::{ModularInvariantsTool, ModularPlotMode};
use super::observables::CHALLENGE_HASSE_WEIL_BOUNDARY_P23;
use super::EllipticCurvesTab;
use crate::app::domain_bar::poll_cross_tab_tool_handoffs;
use crate::async_sim::atlas::RegimeCartographer;
use crate::framework::{CompiledGoalPredicate, InteractiveTool};
use crate::starter_gallery::{
    find_challenge_metadata, find_goal_challenge_by_id, find_story_by_id,
};
use crate::tabs::number_theory::elliptic_curve_playground::EllipticCurvePlayground;
use crate::tabs::ExplorerTab;
use eframe::egui;
use math_explorer::pure_math::elliptic_curves::{AffinePointFp, CurvePoint, PrimeWeierstrassCurve};
use verified_engine::engine::VerifiedEngine;

const EXPECTED_OBSERVABLE_KEYS: [&str; 9] = [
    "discriminant",
    "real_components",
    "j_invariant",
    "group_order",
    "frobenius_trace",
    "hasse_gap",
    "point_order",
    "cofactor",
    "is_singular",
];

fn assert_nine_finite_observables(obs: &[(&'static str, f64)]) {
    assert_eq!(obs.len(), 9);
    for (idx, &expected) in EXPECTED_OBSERVABLE_KEYS.iter().enumerate() {
        assert_eq!(obs[idx].0, expected);
        assert!(obs[idx].1.is_finite(), "Non-finite observable {expected}");
    }
}

#[test]
fn test_atlas_cartographer_zero_alloc_real_and_fp_regimes() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let real_carto = WeierstrassRegimeCartographer::new(("real_a", "real_b"), 23, 0.05);
    let fp_p5 = WeierstrassRegimeCartographer::new(("mod_a", "mod_b"), 5, 1.0);
    let fp_p23 = WeierstrassRegimeCartographer::new(("mod_a", "mod_b"), 23, 1.0);

    let ((regimes, ss_primary, sing_p5, boundary_p23), metrics) =
        VerifiedEngine::run_verified(|| {
            let regimes = [
                real_carto.evaluate_cell(-3.0, 1.0).regime,
                real_carto.evaluate_cell(1.0, 1.0).regime,
                real_carto.evaluate_cell(0.0, 1.0).regime,
                real_carto.evaluate_cell(-2.0, 0.0).regime,
                real_carto.evaluate_cell(-3.0, 2.0).regime,
                real_carto.evaluate_cell(-3.0, -2.0).regime,
                real_carto.evaluate_cell(0.0, 0.0).regime,
                fp_p23.evaluate_cell(0.0, 1.0).regime,
            ];
            let ss_primary = fp_p23.evaluate_cell(0.0, 1.0).primary;
            let (mut sing_p5, mut boundary_p23) = (0_usize, 0_usize);
            for a in -15..=15 {
                for b in -15..=15 {
                    if fp_p5.evaluate_cell(a as f64, b as f64).regime
                        == WeierstrassAtlasRegime::FpSingular as u8
                    {
                        sing_p5 += 1;
                    }
                    if fp_p23.evaluate_cell(a as f64, b as f64).regime
                        == WeierstrassAtlasRegime::FpHasseBoundary as u8
                    {
                        boundary_p23 += 1;
                    }
                }
            }
            (regimes, ss_primary, sing_p5, boundary_p23)
        });

    assert_eq!(
        regimes,
        [
            WeierstrassAtlasRegime::RealTwoComponent as u8,
            WeierstrassAtlasRegime::RealSingleBranch as u8,
            WeierstrassAtlasRegime::RealCmJ0 as u8,
            WeierstrassAtlasRegime::RealCmJ1728 as u8,
            WeierstrassAtlasRegime::RealSingularCrunode as u8,
            WeierstrassAtlasRegime::RealSingularAcnode as u8,
            WeierstrassAtlasRegime::RealSingularCusp as u8,
            WeierstrassAtlasRegime::FpSupersingular as u8,
        ]
    );
    assert!((ss_primary - 24.0).abs() < 1e-9);
    assert_eq!(sing_p5, 193);
    assert!(boundary_p23 > 0 && metrics.function_calls > 0);
}

#[test]
fn test_real_singular_crunode_acnode_cusp_rendering_and_exports() {
    let ctx = egui::Context::default();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1180.0, 740.0));
    let mut tool = WeierstrassPlaygroundTool::default();
    tool.state.field_mode = CurveFieldMode::Real;
    tool.atlas.set_visible(true);

    for (a, b, px, qx, expected_label) in [
        (-3.0, 2.0, -1.5, 2.0, "Crunode Tangent"),
        (-3.0, -2.0, 2.3, 3.2, "Singular Acnode Vertex (x_s, 0)"),
        (0.0, 0.0, 0.8, 1.8, "Singular Cusp Vertex (x_s, 0)"),
    ] {
        tool.state.real_a = a;
        tool.state.real_b = b;
        tool.state.real_px = px;
        tool.state.real_qx = qx;
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ctx| tool.show(ctx),
        );
        let ds = tool.export_datasets();
        assert!(
            !ds.is_empty()
                && ds
                    .iter()
                    .any(|(n, pts)| n.contains(expected_label) && !pts.is_empty())
        );
        assert_nine_finite_observables(&tool.observables());
    }

    tool.state.real_a = -3.0;
    tool.state.real_b = 2.0;
    tool.state.real_px = 1.0;
    tool.state.real_qx = 2.0;
    assert!(!tool
        .export_datasets()
        .iter()
        .any(|(name, _)| name.contains("Secant / Tangent")));
}

#[test]
fn test_fp_singular_torus_rendering_across_playgrounds() {
    let ctx = egui::Context::default();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 760.0));

    let mut play = WeierstrassPlaygroundTool::default();
    play.state.field_mode = CurveFieldMode::FiniteField;
    play.state.prime_p = 17;
    play.state.mod_a = -3;
    play.state.mod_b = 2;
    play.atlas.set_visible(true);
    let _ = play.atlas.set_axis_keys("mod_a", "mod_b");
    play.rebuild_lattice();
    assert_eq!(play.lattice_points.len(), 18);
    assert!(play
        .lattice_points
        .contains(&CurvePoint::Affine { x: 1, y: 0 }));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| play.show(ctx),
    );
    assert!(play
        .export_datasets()
        .iter()
        .any(|(n, pts)| n.contains("Singular Vertex") && pts == &vec![[1.0, 0.0]]));
    assert_nine_finite_observables(&play.observables());

    let mut nt = EllipticCurvePlayground::default();
    nt.p = 17;
    nt.a = -3;
    nt.b = 2;
    nt.atlas.set_visible(true);
    nt.recompute_all();
    assert!(nt.is_singular && nt.chord.is_none());
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |ctx| nt.show(ctx),
    );
    assert!(nt
        .export_datasets()
        .iter()
        .any(|(n, pts)| n.contains("Singular") && !pts.is_empty()));
    assert_nine_finite_observables(&nt.observables());
}

#[test]
fn test_modular_invariants_singular_state_preserves_sweeps_and_exports() {
    let ctx = egui::Context::default();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 760.0));
    let mut mod_tool = ModularInvariantsTool::default();
    mod_tool.state.coeff_a = 0;
    mod_tool.state.coeff_b = 0;
    mod_tool.state.inspect_prime = 23;
    mod_tool.atlas.set_visible(true);
    mod_tool.recompute();
    let saved = mod_tool.save_state().unwrap();
    mod_tool.try_load_state(&saved).unwrap();

    for mode in [
        ModularPlotMode::HasseWeilTraces,
        ModularPlotMode::HurwitzKroneckerSums,
        ModularPlotMode::BreuerValuationBounds,
        ModularPlotMode::ModularQSeriesBars,
    ] {
        mod_tool.state.view_mode = mode;
        mod_tool.recompute();
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |ctx| mod_tool.show(ctx),
        );
        let ds = mod_tool.export_datasets();
        assert!(!ds.is_empty() && ds.iter().any(|(_, pts)| !pts.is_empty()));
    }
    assert_nine_finite_observables(&mod_tool.observables());
}

fn sample_handoff_cases() -> [(EllipticTorusHandoff, &'static str, &'static str); 3] {
    [
        (
            EllipticTorusHandoff {
                target: EllipticHandoffTarget::NumberTheoryTorus,
                p: 23,
                a: 1,
                b: 1,
                point_p: AffinePointFp::Finite { x: 0, y: 1 },
                point_q: AffinePointFp::Finite { x: 6, y: 4 },
                scalar_k: 7,
            },
            "Number Theory",
            "EllipticCurvePlayground",
        ),
        (
            EllipticTorusHandoff {
                target: EllipticHandoffTarget::ModularInvariants,
                p: 29,
                a: -2,
                b: 3,
                point_p: AffinePointFp::Infinity,
                point_q: AffinePointFp::Infinity,
                scalar_k: 4,
            },
            "Elliptic Curves",
            "ModularInvariantsTool",
        ),
        (
            EllipticTorusHandoff {
                target: EllipticHandoffTarget::WeierstrassPlayground,
                p: 23,
                a: 0,
                b: 1,
                point_p: AffinePointFp::Finite { x: 0, y: 1 },
                point_q: AffinePointFp::Finite { x: 2, y: 3 },
                scalar_k: 5,
            },
            "Elliptic Curves",
            "WeierstrassPlaygroundTool",
        ),
    ]
}

#[test]
fn test_cross_tab_torus_handoffs_and_url_bookmark_preservation() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ctx = egui::Context::default();
    let mut app = crate::app::MathExplorerApp::default();

    for (payload, expected_tab, expected_tool) in sample_handoff_cases() {
        request_elliptic_torus_handoff(&ctx, payload);
        poll_cross_tab_tool_handoffs(&mut app, &ctx);
        assert_eq!(app.tabs[app.selected_tab].name(), expected_tab);
        assert_eq!(
            app.tabs[app.selected_tab].active_tool_id(),
            Some(expected_tool)
        );
    }

    let ec_bookmark = app.tabs[app.selected_tab]
        .export_url_preset_json()
        .unwrap()
        .unwrap();
    let mut restored = EllipticCurvesTab::default();
    restored
        .import_url_preset_json(Some("WeierstrassPlaygroundTool"), Some(&ec_bookmark))
        .unwrap();
    assert_eq!(
        restored.export_url_preset_json().unwrap(),
        Some(ec_bookmark)
    );
}

#[test]
fn test_guided_stories_and_ast_goal_challenges() {
    for (story_id, domain) in [
        ("weierstrass_discriminant_cusp_tour", "elliptic_curves"),
        ("modular_cm_and_hurwitz_tour", "elliptic_curves"),
        ("fp_torus_hasse_arch_tour", "number_theory"),
    ] {
        let story = find_story_by_id(story_id, Some(domain)).expect("story registered");
        assert!(story.steps.len() >= 3);
        let meta = inventory::iter::<crate::framework::ToolMetadata>
            .into_iter()
            .find(|t| t.id == story.tool_name && t.domain == story.domain)
            .unwrap();
        let mut tool = (meta.build)();
        for step in story.steps {
            tool.import_preset_json((step.preset_generator)()).unwrap();
            assert_nine_finite_observables(&tool.observables());
        }
    }

    for (goal_id, domain) in [
        ("elliptic_pinch_crunode", "elliptic_curves"),
        ("elliptic_two_component_cm", "elliptic_curves"),
        ("modular_supersingular_j0", "elliptic_curves"),
        ("torus_prime_order_generator", "number_theory"),
    ] {
        let goal = find_goal_challenge_by_id(goal_id, Some(domain)).expect("goal registered");
        let pred = CompiledGoalPredicate::compile(goal.predicate_expr).expect("valid AST");
        let meta = inventory::iter::<crate::framework::ToolMetadata>
            .into_iter()
            .find(|t| t.id == goal.tool_name && t.domain == goal.domain)
            .unwrap();
        let mut tool = (meta.build)();
        tool.import_preset_json((goal.initial_preset_generator)())
            .unwrap();
        let mut scratch = std::collections::HashMap::new();
        assert!(!pred
            .eval_observables(&tool.observables(), &mut scratch)
            .unwrap());
    }
}

#[test]
fn test_locked_prime_hasse_weil_boundary_challenge() {
    let meta = find_challenge_metadata(CHALLENGE_HASSE_WEIL_BOUNDARY_P23).expect("registered");
    let baseline = meta.generate_baseline_state();
    let mut tampered: serde_json::Value = serde_json::from_str(&baseline).unwrap();
    tampered["prime_p"] = serde_json::json!(17);
    tampered["field_mode"] = serde_json::json!("Real");
    let clamped = meta.clamp_state_json(&tampered.to_string()).unwrap();
    assert!(clamped.contains("\"prime_p\":23") && clamped.contains("\"FiniteField\""));

    let mut tool = WeierstrassPlaygroundTool::default();
    tool.apply_challenge(Some(CHALLENGE_HASSE_WEIL_BOUNDARY_P23))
        .unwrap();
    tool.try_load_state(&baseline).unwrap();
    assert!(!tool.evaluate_challenge().unwrap().passed);

    let (win_a, win_b) = (-15..=15)
        .flat_map(|a| (-15..=15).map(move |b| (a, b)))
        .find(|&(a, b)| {
            PrimeWeierstrassCurve::new(a, b, 23)
                .ok()
                .and_then(|c| c.frobenius_trace().ok())
                .is_some_and(|ap| ap.abs() == 9)
        })
        .expect("Extremal Hasse-Weil curve exists in [-15, 15]^2 over F_23");
    tool.state.mod_a = win_a;
    tool.state.mod_b = win_b;
    tool.rebuild_lattice();
    let eval = tool.evaluate_challenge().unwrap();
    assert!(
        eval.passed,
        "Expected win at ({win_a}, {win_b}): {}",
        eval.detail
    );
    export_singular_and_challenge_pngs(&tool);
}

fn export_singular_and_challenge_pngs(fp_tool: &WeierstrassPlaygroundTool) {
    let opts = egui_plot::export::ExportOptions::default();
    if let Ok(png) = egui_plot::export::export_custom_png(&fp_tool.export_datasets(), &opts) {
        let _ = std::fs::write("/tmp/elliptic_fp23_hasse_boundary_preview.png", png);
    }
    let mut real_tool = WeierstrassPlaygroundTool::default();
    real_tool.state.field_mode = CurveFieldMode::Real;
    for (a, b, px, qx, path) in [
        (
            -3.0,
            2.0,
            -1.5,
            2.0,
            "/tmp/elliptic_singular_crunode_preview.png",
        ),
        (
            -3.0,
            -2.0,
            2.3,
            3.4,
            "/tmp/elliptic_singular_acnode_preview.png",
        ),
    ] {
        real_tool.state.real_a = a;
        real_tool.state.real_b = b;
        real_tool.state.real_px = px;
        real_tool.state.real_qx = qx;
        if let Ok(png) = egui_plot::export::export_custom_png(&real_tool.export_datasets(), &opts) {
            let _ = std::fs::write(path, png);
        }
    }
}
