#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:graph_parameters_rust]

use std::f64::consts::TAU;

use eframe::egui::{vec2, Context, Pos2, RawInput, Rect};
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::wave_2d::masks::clamped_beam_eigenfunction;
use math_explorer::pure_math::analysis::pde::wave_2d::{
    analytic_clamped_square_omega, BoundaryMode2D,
};

use super::eigenmodes::{
    eigenmode_sample, mode_frequency_hz, mode_metric_label, seed_eigenmode_field,
};
use super::refractive::{
    detector_focal_metrics, DEFAULT_BICONVEX_EXPR, DEFAULT_GRIN_EXPR, DEFAULT_LUNEBURG_EXPR,
};
use super::*;
use crate::app::MathExplorerApp;
use crate::framework::SimulationFramework;
use crate::starter_gallery::{find_challenge_by_id, find_story_by_id, StarterPresetMetadata};
use crate::url_state::URL_JSON_BOUNDS;

fn raw_frame() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

#[test]
fn test_square_chladni_plate_preset_antinode_and_60s_modal_correlation() {
    let tool = ChladniWaveTankTool::default();
    assert_eq!(tool.state.preset, TankPreset::SquareChladniPlate);
    assert!((tool.state.driver_x - 0.35).abs() < 1.0e-9);
    assert!((tool.state.driver_y - 0.08).abs() < 1.0e-9);
    assert!((tool.state.driver_frequency - 0.3142).abs() < 1.0e-9);

    let starter = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .find(|p| p.id == "chladni_square_plate_starter")
        .expect("Square Chladni Plate starter");
    let raw: serde_json::Value = serde_json::from_str(starter.generate_preset()).unwrap();
    assert!((raw["config"]["driver_x"].as_f64().unwrap() - 0.35).abs() < 1.0e-9);
    assert!((raw["config"]["driver_y"].as_f64().unwrap() - 0.08).abs() < 1.0e-9);
    assert!((raw["config"]["driver_frequency"].as_f64().unwrap() - 0.3142).abs() < 1.0e-9);

    let params = tool.pde_params();
    let mut resonant = Wave2DSolver::new_unit_square(48);
    math_explorer::pure_math::analysis::pde::wave_2d::apply_point_clamp(
        &mut resonant.wall_mask,
        48,
        48,
        0.5,
        0.5,
        1.0,
    );
    seed_eigenmode_field(&mut resonant, &tool.state, 1, 3);
    let mut res_drv = tool.driver_config();
    res_drv.radius_cells = 1.2;
    while resonant.time() < 60.0 {
        resonant.step_driven(&params, &res_drv, 0.02, None);
    }
    let corr = modal_correlation_on_solver(&resonant, &tool.state, 1, 3);
    assert!(
        corr >= 0.60,
        "Expected mode (1, 3) modal_field correlation >= 0.60 at 60 s, got {corr:.4}"
    );
}

fn modal_correlation_on_solver(
    solver: &Wave2DSolver,
    state: &ChladniWaveTankState,
    m: usize,
    n: usize,
) -> f64 {
    let w = solver.width();
    let h = solver.height();
    let denom_x = (w - 1) as f64;
    let denom_y = (h - 1) as f64;
    let modal = solver.modal_field();
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for y in 0..h {
        let yn = (y as f64) / denom_y;
        for x in 0..w {
            let idx = flatten_2d_index(x, y, w);
            if solver.wall_mask[idx] > 0.5 {
                let ref_val = eigenmode_sample(state, m, n, (x as f64) / denom_x, yn);
                dot += modal[idx] * ref_val;
                norm_a += modal[idx] * modal[idx];
                norm_b += ref_val * ref_val;
            }
        }
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt()).max(1.0e-12)).abs()
}

#[test]
fn test_eigenmode_lattice_dispersion_antinode_and_clamped_beam_seeding() {
    let mut tool = ChladniWaveTankTool::default();
    assert_eq!(mode_metric_label(&tool.state, 1, 3), "Neumann f(1,3)");
    assert!((mode_frequency_hz(&tool.state, tool.solver.dx(), 1, 3) - 0.31416).abs() < 5.0e-4);

    tool.select_eigenmode(2, 3);
    assert_eq!(tool.active_mode, (2, 3));
    let expected_23 = mode_frequency_hz(&tool.state, tool.solver.dx(), 2, 3);
    assert!((tool.state.driver_frequency - expected_23).abs() < 1.0e-3);
    let sample_at_driver =
        eigenmode_sample(&tool.state, 2, 3, tool.state.driver_x, tool.state.driver_y).abs();
    assert!(
        sample_at_driver > 0.25,
        "Driver must be placed near an antinode off nodal lines, got |Phi| = {sample_at_driver}"
    );

    // Clamped boundary uses analytic_clamped_square_omega and clamped_beam_eigenfunction
    tool.state.boundary = BoundaryMode2D::Clamped;
    tool.select_eigenmode(1, 2);
    assert_eq!(mode_metric_label(&tool.state, 1, 2), "Clamped f(1,2)");
    let expected_clamped =
        analytic_clamped_square_omega(1, 2, 1.0, tool.state.plate_stiffness) / TAU;
    assert!((tool.state.driver_frequency - expected_clamped).abs() < 1.0e-3);
    let test_idx = flatten_2d_index(32, 20, GRID_SIZE);
    let xn = 32.0 / 127.0;
    let yn = 20.0 / 127.0;
    let expected_seed = 0.24
        * (clamped_beam_eigenfunction(2, xn) * clamped_beam_eigenfunction(1, yn)
            - clamped_beam_eigenfunction(1, xn) * clamped_beam_eigenfunction(2, yn));
    assert!((tool.solver.displacement()[test_idx] - expected_seed).abs() < 1.0e-9);

    // CircularDrumhead uses Bessel roots j_{m,n}
    tool.apply_preset(TankPreset::CircularDrumhead);
    assert_eq!(mode_metric_label(&tool.state, 2, 2), "Bessel f(2,2)");
    assert!((mode_frequency_hz(&tool.state, tool.solver.dx(), 2, 2) - 2.9770).abs() < 2.0e-3);
    tool.select_eigenmode(0, 1);
    assert!((tool.state.driver_frequency - 0.8505).abs() < 2.0e-3);
}

#[test]
fn test_refractive_luneburg_and_grin_gain_vs_biconvex_and_courant_scaling() {
    let mut biconvex = ChladniWaveTankTool::default();
    biconvex.apply_preset(TankPreset::AcousticLens);
    for _ in 0..340 {
        biconvex.advance_physics_step();
    }
    biconvex.refresh_analysis_outputs();
    let (bi_gain, bi_fwhm) = detector_focal_metrics(&biconvex.detector_buf);
    assert!(
        (2.2..=2.8).contains(&bi_gain),
        "Expected biconvex focal gain ~2.49x, got {bi_gain:.3}"
    );
    assert!(
        bi_fwhm >= 24.0,
        "Expected biconvex FWHM ~28 cells, got {bi_fwhm}"
    );

    for expr in [DEFAULT_LUNEBURG_EXPR, DEFAULT_GRIN_EXPR] {
        let mut lens = ChladniWaveTankTool::default();
        lens.apply_preset(TankPreset::AcousticLens);
        lens.set_refractive_formula(expr);
        for _ in 0..340 {
            lens.advance_physics_step();
        }
        lens.refresh_analysis_outputs();
        let (gain, fwhm) = detector_focal_metrics(&lens.detector_buf);
        assert!(
            gain >= 3.3,
            "Expected formula '{expr}' focal gain >= 3.3x at x = 92, got {gain:.3}"
        );
        assert!(
            (12.0..=20.0).contains(&fwhm),
            "Expected formula '{expr}' FWHM in 12..=20 cells, got {fwhm}"
        );
    }

    verify_refractive_syntax_guard_and_sub_unity_courant();
}

fn verify_refractive_syntax_guard_and_sub_unity_courant() {
    let ctx = Context::default();
    let mut fw = SimulationFramework::new("analysis");
    assert!(fw.select_tool_by_name("ChladniWaveTankTool"));
    {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<ChladniWaveTankTool>()
            .unwrap();
        tool.apply_preset(TankPreset::AcousticLens);
        tool.set_refractive_formula(DEFAULT_BICONVEX_EXPR);
        // Simulate mid-keystroke syntax error: last_valid_compiled stays active, no FiniteStateGuard rollback
        tool.refractive.input_bar.set_expression("sqrt(1 + ");
        assert!(!tool.refractive.input_bar.is_valid());
        assert!(tool.refractive.input_bar.active_compiled().is_some());
        assert!(tool.has_finite_parameters());
    }
    let _ = ctx.run(raw_frame(), |ctx| fw.show(ctx, "analysis"));
    assert!(
        !fw.is_diverged,
        "Transient syntax edit must not trigger FiniteStateGuard rollback"
    );

    let mut fast = ChladniWaveTankTool::default();
    fast.apply_preset(TankPreset::AcousticLens);
    fast.set_refractive_formula("0.1 + 0 * (x + y)");
    let cfl = fast
        .solver
        .effective_courant(&fast.pde_params(), fast.state.time_step);
    assert!(
        (cfl - (1.0 / 3.0)).abs() < 1.0e-6,
        "Expected n clamped to 0.6 -> Courant 0.3333, got {cfl:.6}"
    );
}

#[test]
fn test_sixteen_custom_clamps_byte_budget_and_formula_url_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut tool = ChladniWaveTankTool::default();
    tool.state.custom_clamps = (0..16)
        .map(|i| [0.1 + 0.05 * ((i % 8) as f64), 0.2 + 0.1 * ((i / 2) as f64)])
        .map(|[x, y]| [(x * 10.0).round() / 10.0, (y * 10.0).round() / 10.0])
        .collect();
    let compact_json = tool.save_state().expect("save_state");
    assert!(
        compact_json.len() <= 813,
        "Expected 16-pin compact JSON <= 813 bytes, got {} bytes",
        compact_json.len()
    );
    assert!(URL_JSON_BOUNDS.validate_json(&compact_json).is_ok());

    let mut source_app = MathExplorerApp::default();
    let idx = source_app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "analysis")
        .unwrap();
    source_app.selected_tab = idx;
    assert!(source_app.tabs[idx].select_tool_by_name("ChladniWaveTankTool"));
    if let Some(fw) = source_app.tabs[idx].framework_mut() {
        let active = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<ChladniWaveTankTool>()
            .unwrap();
        active.state.custom_clamps = tool.state.custom_clamps.clone();
        active.set_refractive_formula(DEFAULT_LUNEBURG_EXPR);
        active.refractive.input_bar.set_param("a", 1.25);
        active.refractive.sync_to_state(&mut active.state);
        active.rebuild_masks();
    }
    let url = source_app.encode_current_share_url().unwrap();
    let hash = &url[url.find('#').unwrap()..];

    let mut restored_app = MathExplorerApp::default();
    assert!(restored_app
        .apply_share_location_strings(hash, "", None)
        .unwrap());
    let restored = restored_app.tabs[idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .as_any()
        .unwrap()
        .downcast_ref::<ChladniWaveTankTool>()
        .unwrap();
    assert_eq!(restored.state.custom_clamps.len(), 16);
    assert_eq!(restored.state.refractive_mode, RefractiveMode::Formula);
    assert_eq!(restored.state.refractive_expr, DEFAULT_LUNEBURG_EXPR);
    assert_eq!(restored.state.refractive_params.get("a"), Some(&1.25));
}

#[test]
fn test_observables_story_tour_and_goal_challenges_complete_in_hud() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = Context::default();

    let story = find_story_by_id("chladni_resonance_and_metamaterial_tour", Some("analysis"))
        .expect("Chladni story pack registered");
    let mut fw = SimulationFramework::new("analysis");
    fw.launch_story(story, 0).expect("launch step 0");
    fw.step_story(1).expect("step 1");
    fw.step_story(2).expect("step 2");
    let obs = fw.active_tool.as_ref().unwrap().observables();
    assert!(obs.iter().all(|(_, v)| v.is_finite()));
    assert!(obs.iter().any(|(k, _)| *k == "sand_nodal_energy"));
    assert!(obs.iter().any(|(k, _)| *k == "focal_gain"));

    // Challenge 1: Pin & Lock a Chladni Eigenmode
    let ch1 = find_challenge_by_id("chladni_nodal_resonance", Some("analysis")).unwrap();
    fw.launch_challenge(ch1).expect("launch ch1");
    assert_eq!(fw.evaluate_challenge_predicates(Some(&ctx)), Some(false));
    {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<ChladniWaveTankTool>()
            .unwrap();
        tool.select_eigenmode(1, 3);
        tool.state.custom_clamps.push([0.5, 0.25]);
        tool.rebuild_masks();
        for _ in 0..80 {
            tool.step_simulation();
        }
    }
    assert_eq!(fw.evaluate_challenge_predicates(Some(&ctx)), Some(true));

    // Challenge 2: Sculpt a 3.3x Metamaterial Focus
    let ch2 = find_challenge_by_id("metamaterial_luneburg_focus", Some("analysis")).unwrap();
    fw.launch_challenge(ch2).expect("launch ch2");
    assert_eq!(fw.evaluate_challenge_predicates(Some(&ctx)), Some(false));
    {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<ChladniWaveTankTool>()
            .unwrap();
        tool.set_refractive_formula(DEFAULT_LUNEBURG_EXPR);
        for _ in 0..340 {
            tool.advance_physics_step();
        }
        tool.refresh_analysis_outputs();
    }
    assert_eq!(fw.evaluate_challenge_predicates(Some(&ctx)), Some(true));
}
