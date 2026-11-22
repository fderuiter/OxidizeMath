#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::lenia_tool::LeniaRunner;
use super::*;
use crate::async_sim::SimulationRunner;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::InteractiveTool;
use math_explorer::biology::phyllotaxis::is_consecutive_fibonacci_pair;

fn extract_hash(url: &str) -> &str {
    let idx = url.find('#').expect("share URL must include '#' fragment");
    &url[idx..]
}

#[test]
fn test_morphogenesis_tab_exposes_all_four_biology_tools_in_side_panel() {
    let mut tab = MorphogenesisTab::default();
    let ids: Vec<&'static str> = tab
        .framework
        .available_tools
        .iter()
        .map(|m| m.id)
        .collect();
    assert_eq!(ids.len(), 4);
    assert!(ids.contains(&"MorphogenesisTool"));
    assert!(ids.contains(&"LeniaTool"));
    assert!(ids.contains(&"PhyllotaxisLSystemTool"));
    assert!(ids.contains(&"EvolutionKineticsTool"));
    assert_eq!(tab.active_tool_id(), Some("MorphogenesisTool"));

    for id in [
        "LeniaTool",
        "PhyllotaxisLSystemTool",
        "EvolutionKineticsTool",
        "MorphogenesisTool",
    ] {
        assert!(tab.select_tool_by_name(id));
        assert_eq!(tab.active_tool_id(), Some(id));
    }
}

#[test]
fn test_morphogenesis_gray_scott_and_generic_reaction_diffusion_zero_allocation_steps() {
    let mut tool = MorphogenesisTool::default();
    tool.apply_model_preset(
        MorphogenesisModelKind::GrayScott,
        (0.1, 0.9),
        (0.037, 0.060),
        (0.2, 0.1, 0.5),
    );
    let mut gs_runner =
        MorphogenesisTool::create_multi_runner(&tool.params, &tool.model_params, 32, 32);
    gs_runner.step();
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            gs_runner.step();
        });
    }
    assert!(FiniteStateGuard::is_finite_state(&gs_runner));

    tool.apply_model_preset(
        MorphogenesisModelKind::GenericReactionDiffusion,
        (0.1, 0.9),
        (0.028, 0.062),
        (0.2, 0.1, 0.5),
    );
    let mut rd_runner =
        MorphogenesisTool::create_multi_runner(&tool.params, &tool.model_params, 32, 32);
    rd_runner.step();
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            rd_runner.step();
        });
    }
    assert!(FiniteStateGuard::is_finite_state(&rd_runner));
}

#[test]
fn test_lenia_runner_zero_allocation_and_tool_lifecycle() {
    let mut tool = LeniaTool::default();
    let mut runner = LeniaRunner::new(&tool.params, 128, 128);
    runner.step();
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            runner.step();
        });
    }
    assert!(FiniteStateGuard::is_finite_state(&runner));
    assert!(runner.system.telemetry().total_mass > 20.0);

    tool.apply_preset_choice(LeniaPresetChoice::Hydrogeminium);
    let saved = tool.save_state().expect("LeniaTool state");
    assert!(saved.contains("Hydrogeminium"));

    tool.step_simulation();
    assert!(tool.export_raster_image().is_some());

    // FiniteStateGuard recovery
    tool.params.write().unwrap().dt = 0.4;
    FiniteStateGuard::auto_scale_step_size(&mut tool);
    assert!((tool.params.read().unwrap().dt - 0.2).abs() < 1e-12);
}

#[test]
fn test_phyllotaxis_and_evolution_kinetics_tools_invariants_and_recovery() {
    let mut phyllo = PhyllotaxisLSystemTool::default();
    let (m, n) = phyllo.phyllotaxis.detect_parastichy_pair();
    assert!(is_consecutive_fibonacci_pair(m, n));
    assert!(!phyllo.lsystem.segments().is_empty());
    assert_eq!(phyllo.export_datasets().len(), 2);

    let before_count = phyllo.state.primordia_count;
    phyllo.advance_plastochrons(0.5);
    assert!(phyllo.state.primordia_count >= before_count);

    phyllo.state.divergence_angle_deg = f64::NAN;
    assert!(!phyllo.has_finite_parameters());
    assert!(InteractiveTool::clamp_parameters(&mut phyllo));
    assert!(phyllo.has_finite_parameters());

    let mut evo = EvolutionKineticsTool::default();
    assert!((evo.state.analytical_ess() - 0.5).abs() < 1e-12);
    let terminal_hawk = evo.hawk_curve.last().unwrap()[1];
    assert!(
        (terminal_hawk - 0.5).abs() < 0.05,
        "Hawk-Dove RK4 trajectory should approach ESS=0.5, got {terminal_hawk}"
    );
    let (v_mm, v_hill) = evo.probe_velocities();
    assert!(v_mm > 0.0 && v_hill > 0.0);
    assert_eq!(evo.export_datasets().len(), 5);

    evo.state.resource_value = f64::INFINITY;
    assert!(!evo.has_finite_parameters());
    assert!(InteractiveTool::clamp_parameters(&mut evo));
    assert!(evo.has_finite_parameters());
}

#[test]
fn test_biology_new_tools_url_fragment_roundtrips() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = crate::app::MathExplorerApp::default();
    let tab_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .expect("Morphogenesis tab");
    app.selected_tab = tab_idx;

    for (tool_id, preset_json) in [
        ("MorphogenesisTool", presets::PRESET_GRAY_SCOTT_CORAL),
        ("LeniaTool", presets::PRESET_LENIA_HYDROGEMINIUM),
        ("PhyllotaxisLSystemTool", presets::PRESET_PHYLLOTAXIS_HILL_SHOOT),
        ("EvolutionKineticsTool", presets::PRESET_EVOLUTION_KINETICS_ESS),
    ] {
        app.tabs[tab_idx]
            .framework_mut()
            .unwrap()
            .apply_starter_card("Test Card", Some(tool_id), Some(preset_json))
            .expect("starter card should load");
        let expected_state = app.tabs[tab_idx]
            .framework()
            .unwrap()
            .active_tool
            .as_ref()
            .unwrap()
            .save_state();

        let share_url = app.encode_current_share_url().expect("encode share url");
        assert!(share_url.contains("tab=morphogenesis"));
        let hash = extract_hash(&share_url);

        let mut restored = crate::app::MathExplorerApp::default();
        assert!(restored.apply_share_location_strings(hash, "", None).unwrap());
        assert_eq!(restored.tabs[tab_idx].active_tool_id(), Some(tool_id));
        let actual_state = restored.tabs[tab_idx]
            .framework()
            .unwrap()
            .active_tool
            .as_ref()
            .unwrap()
            .save_state();
        assert_eq!(actual_state, expected_state, "URL mismatch for {tool_id}");
    }
}
