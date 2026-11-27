use super::atlas::{OdeAtlasRegime, OdeRegimeCartographer};
use super::phase::PlotViewMode;
use super::*;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{InteractiveTool, SimulationFramework};
use math_explorer::pure_math::analysis::ode::{OdeModel, RungeKutta4, TimeStepper, VecState};

fn collect_ui_texts(shapes: &[egui::epaint::Shape], out: &mut Vec<String>) {
    for shape in shapes {
        match shape {
            egui::epaint::Shape::Text(t) => out.push(t.galley.job.text.clone()),
            egui::epaint::Shape::Vec(inner) => collect_ui_texts(inner, out),
            _ => {}
        }
    }
}

fn build_200_inst_planar_expr() -> String {
    let mut terms = vec!["-0.1 * x".to_string()];
    for idx in 1..=18 {
        terms.push(format!("0.002 * sin({idx} * t + 0.05 * y)"));
    }
    terms.join(" + ")
}

#[test]
fn test_planar_custom_synthesizes_shared_sliders_and_updates_dynamics_and_ui() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let mut tool = OdeSolverTool {
        preset: OdePreset::PlanarCustom,
        phase_plane_mode: true,
        show_nullclines: true,
        comparison_mode: true,
        ic_y0: 1.2,
        ic_v0: -0.4,
        dt: 0.05,
        total_time: 6.0,
        ..Default::default()
    };
    tool.planar_x_input
        .set_expression("mu * (x - x^3 / 3 - y) + a");
    tool.planar_y_input.set_expression("x / mu - b * y");
    tool.seed_orbit([0.6, 0.8]);
    tool.recalculate();

    assert!(!tool.diverged());
    assert!(FiniteStateGuard::is_finite_state(&tool));
    assert_eq!(tool.planar_bank.param_names(), &["mu", "a", "b"]);

    tool.planar_bank.set_param("mu", 0.5);
    tool.planar_bank.set_param("a", 0.1);
    tool.planar_bank.set_param("b", 0.2);
    tool.recalculate();
    let y_mu_low = tool.y_series.clone();
    let v_mu_low = tool.v_series.clone();
    let seeded_low = tool.seeded_trajectories[0].y_series.clone();
    let quiver_low = tool.compute_quiver_arrows().1;
    let null_x_low = tool.compute_nullcline_segments().0;

    tool.planar_bank.set_param("mu", 2.6);
    tool.planar_bank.set_param("a", 0.9);
    tool.planar_bank.set_param("b", 0.7);
    tool.recalculate();
    assert_ne!(tool.y_series, y_mu_low);
    assert_ne!(tool.seeded_trajectories[0].y_series, seeded_low);
    assert_ne!(tool.compute_quiver_arrows().1, quiver_low);
    assert_ne!(tool.compute_nullcline_segments().0, null_x_low);
    assert_eq!(tool.comparison_manager().len(), 3);
    assert!(tool.discrepancy_euler_rk4().max_error.is_finite());

    export_planar_sweep_png(&y_mu_low, &v_mu_low, &tool.y_series, &tool.v_series);
    assert_planar_ui_renders_single_synthesized_bank(&mut tool);
}

#[cfg(not(target_arch = "wasm32"))]
fn export_planar_sweep_png(x_low: &[f64], y_low: &[f64], x_high: &[f64], y_high: &[f64]) {
    let zip_pairs = |xs: &[f64], ys: &[f64]| -> Vec<[f64; 2]> {
        xs.iter()
            .copied()
            .zip(ys.iter().copied())
            .map(|(x, y)| [x, y])
            .collect()
    };
    let datasets = vec![
        (
            "mu=0.5, a=0.1, b=0.2 (x, y)".to_string(),
            zip_pairs(x_low, y_low),
        ),
        (
            "mu=2.6, a=0.9, b=0.7 (x, y)".to_string(),
            zip_pairs(x_high, y_high),
        ),
    ];
    let png = egui_plot::export::export_custom_png(&datasets, &Default::default())
        .expect("export planar custom sweep PNG");
    let _ = std::fs::write("/tmp/planar_custom_fhn_phase_sweep.png", png);
}

#[cfg(target_arch = "wasm32")]
fn export_planar_sweep_png(_: &[f64], _: &[f64], _: &[f64], _: &[f64]) {}

fn assert_planar_ui_renders_single_synthesized_bank(tool: &mut OdeSolverTool) {
    let ctx = egui::Context::default();
    let out = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    let shapes: Vec<_> = out.shapes.into_iter().map(|cs| cs.shape).collect();
    let mut texts = Vec::new();
    collect_ui_texts(&shapes, &mut texts);

    assert!(
        !texts
            .iter()
            .any(|t| t.contains("Solver Divergence Warning")),
        "unexpected divergence banner for custom planar parameter: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.contains("k (Coupling / Damping)")),
        "legacy k slider must not appear when k is absent from planar formulas: {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.contains("r (Growth / Forcing)")),
        "legacy r slider must not appear when r is absent from planar formulas: {texts:?}"
    );
    let bank_headers = texts.iter().filter(|t| t.as_str() == "Parameters:").count();
    assert_eq!(
        bank_headers, 1,
        "expected exactly one 'Parameters:' slider bank in sidebar without inline bar duplicates, got {bank_headers}"
    );
    let mu_labels = texts.iter().filter(|t| t.as_str() == "mu").count();
    assert_eq!(
        mu_labels, 2,
        "expected 'mu' once in ParameterSliderBank and once in Sensitivity panel, got {mu_labels}"
    );
}

#[test]
fn test_planar_custom_scope_rules_and_caps_preserve_last_valid_bytecode() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::PlanarCustom,
        ic_y0: 1.0,
        ic_v0: 0.5,
        dt: 0.05,
        total_time: 3.0,
        ..Default::default()
    };
    tool.planar_x_input
        .set_expression("r * x * (1 - x / cap_k) - k * x * y");
    tool.planar_y_input.set_expression("k * x * y - y");
    tool.recalculate();
    assert!(!tool.has_custom_input_error());
    assert_eq!(tool.planar_bank.param_names(), &["r", "cap_k", "k"]);
    assert_eq!(tool.planar_bank.active_value("cap_k"), Some(10.0));
    tool.planar_bank.set_param("cap_k", 24.0);
    tool.recalculate();
    assert_eq!(tool.param_cap_k, 24.0);
    let valid_end_y = *tool.y_series.last().expect("valid final x(t)");

    // Disallowed z variable
    tool.planar_x_input.set_expression("x + z");
    let z_err = tool.planar_x_input.error_message().expect("z disallowed");
    assert!(z_err.contains("allowed variables: t, x, y"), "got: {z_err}");
    assert!(tool.has_custom_input_error());

    // Disallowed v variable
    tool.planar_x_input.set_expression("y");
    tool.planar_y_input.set_expression("-x - 0.2 * v");
    let v_err = tool.planar_y_input.error_message().expect("v disallowed");
    assert!(
        v_err.contains("allowed variables: t, x, y") && v_err.contains("Coupled"),
        "got: {v_err}"
    );

    // > 6 combined free parameters across both bars
    tool.planar_x_input
        .set_expression("p1*x + p2*y + p3*t + p4");
    tool.planar_y_input.set_expression("p5*x + p6*y + p7");
    tool.recalculate();
    let param_err = tool
        .planar_bank
        .first_error(&tool.planar_x_input, &tool.planar_y_input)
        .expect(">6 params error");
    assert!(param_err.contains("free parameters") && param_err.contains('6'));
    assert!(!tool.diverged());
    assert_eq!(*tool.y_series.last().expect("preserved y"), valid_end_y);

    // > 384 combined compiled instructions across two individually valid (~201 inst) bars
    let heavy = build_200_inst_planar_expr();
    tool.planar_x_input.set_expression(&heavy);
    tool.planar_y_input.set_expression(&heavy);
    assert!(tool.planar_x_input.is_valid() && tool.planar_y_input.is_valid());
    tool.recalculate();
    let inst_err = tool
        .planar_bank
        .first_error(&tool.planar_x_input, &tool.planar_y_input)
        .expect(">384 instructions error");
    assert!(inst_err.contains("384"), "got: {inst_err}");
}

#[test]
fn test_planar_custom_url_state_roundtrip_legacy_fallback_and_atomic_rejection() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::PlanarCustom,
        phase_plane_mode: true,
        ic_y0: 1.25,
        ic_v0: -0.35,
        dt: 0.04,
        total_time: 4.0,
        ..Default::default()
    };
    tool.planar_x_input.set_expression("mu * (x - x^3 / 3 - y)");
    tool.planar_y_input.set_expression("x / mu + a");
    tool.recalculate();
    let rev_before = tool.url_state_revision();
    tool.planar_bank.set_param("mu", 2.35);
    tool.planar_bank.set_param("a", -0.15);
    tool.recalculate();
    assert_ne!(tool.url_state_revision(), rev_before);

    let saved = tool.save_state().expect("serialize planar state");
    assert!(saved.contains(r#""planar_params""#) && saved.contains("2.35"));
    let mut restored = OdeSolverTool::default();
    restored
        .try_load_state(&saved)
        .expect("restore planar state");
    assert_eq!(restored.preset, OdePreset::PlanarCustom);
    assert_eq!(restored.planar_bank.active_value("mu"), Some(2.35));
    assert_eq!(restored.planar_bank.active_value("a"), Some(-0.15));
    assert_eq!(restored.y_series, tool.y_series);
    assert_eq!(restored.v_series, tool.v_series);

    assert_legacy_planar_fallback_and_rejections(&mut restored, &tool.y_series);
}

fn assert_legacy_planar_fallback_and_rejections(restored: &mut OdeSolverTool, expected_y: &[f64]) {
    let heavy = build_200_inst_planar_expr();
    for bad in [
        r#"{"preset":"PlanarCustom","dt":0.05,"total_time":4.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"planar_x_expr":"x + z","planar_y_expr":"-x"}"#.to_string(),
        r#"{"preset":"PlanarCustom","dt":0.05,"total_time":4.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"planar_x_expr":"y","planar_y_expr":"-v"}"#.to_string(),
        r#"{"preset":"PlanarCustom","dt":0.05,"total_time":4.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"planar_x_expr":"p1*x+p2*y+p3*t+p4","planar_y_expr":"p5*x+p6*y+p7"}"#.to_string(),
        r#"{"preset":"PlanarCustom","dt":0.05,"total_time":4.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"planar_x_expr":"mu*x","planar_y_expr":"-y","planar_params":{"mu":100000.0}}"#.to_string(),
        r#"{"preset":"PlanarCustom","dt":0.05,"total_time":4.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"planar_x_expr":"mu*x","planar_y_expr":"-y","planar_params":{"p1":1.0,"p2":1.0,"p3":1.0,"p4":1.0,"p5":1.0,"p6":1.0,"p7":1.0}}"#.to_string(),
        format!(
            r#"{{"preset":"PlanarCustom","dt":0.05,"total_time":4.0,"param_k":1.0,"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"planar_x_expr":"{heavy}","planar_y_expr":"{heavy}"}}"#
        ),
    ] {
        assert!(restored.try_load_state(&bad).is_err(), "expected rejection: {bad}");
        assert_eq!(restored.y_series, expected_y);
        assert_eq!(restored.planar_bank.active_value("mu"), Some(2.35));
    }

    // Legacy URL JSON without planar_params populates k, r, K from scalar fields
    let legacy = r#"{"preset":"PlanarCustom","dt":0.05,"total_time":4.0,"param_k":2.2,"param_r":1.4,"param_cap_k":25.0,"ic_y0":1.0,"ic_v0":0.5,"planar_x_expr":"r*x*(1 - x/K) - k*x*y","planar_y_expr":"k*x*y - y"}"#;
    let mut fw = SimulationFramework::new("analysis");
    let wrapped = format!(
        r#"{{"metadata":{{"schema_version":"1.0","title":"LegacyPlanar","domain":"analysis","timestamp":"2026-11-25T00:00:00Z"}},"config":{legacy}}}"#
    );
    fw.import_url_preset_json(Some("OdeSolverTool"), Some(&wrapped))
        .expect("legacy planar URL preset should hydrate");
    let mut legacy_tool = OdeSolverTool::default();
    legacy_tool.try_load_state(legacy).expect("legacy load");
    assert_eq!(legacy_tool.planar_bank.active_value("k"), Some(2.2));
    assert_eq!(legacy_tool.planar_bank.active_value("r"), Some(1.4));
    assert_eq!(legacy_tool.planar_bank.active_value("K"), Some(25.0));
}

#[test]
fn test_planar_custom_sensitivity_recovery_and_calibrated_code_export() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::PlanarCustom,
        view_mode: PlotViewMode::PhasePortrait,
        phase_plane_mode: false,
        ic_y0: 1.5,
        ic_v0: 0.5,
        dt: 0.04,
        total_time: 3.0,
        ..Default::default()
    };
    tool.planar_x_input.set_expression("-a * x + b * y");
    tool.planar_y_input.set_expression("-b * x - a * y");
    tool.recalculate();
    tool.planar_bank.set_param("a", 0.45);
    tool.planar_bank.set_param("b", 1.15);
    tool.recalculate();

    let sens_names: Vec<_> = tool
        .collect_sensitivity_params()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(sens_names, vec!["a", "b", "y0", "v0"]);
    tool.run_sensitivity_analysis(false);
    tool.run_sensitivity_analysis(false);
    assert!(tool.sensitivity_engine.is_converged());
    assert!(tool
        .sensitivity_engine
        .metrics
        .iter()
        .any(|m| m.param_name == "a" && m.partial_derivative.abs() > 1e-4));

    let (_, tips_a) = tool.quiver_vectors();
    tool.planar_bank.set_param("b", 2.4);
    tool.recalculate();
    let (_, tips_b) = tool.quiver_vectors();
    assert_ne!(tips_a, tips_b);

    assert_planar_export_and_recovery(&mut tool);
}

fn assert_planar_export_and_recovery(tool: &mut OdeSolverTool) {
    let art = tool.export_code_artifact().expect("planar code artifact");
    assert!(art.validate().is_ok());
    assert!(art.calibrated_latex.contains("0.45") && art.calibrated_latex.contains("2.4"));
    assert!(art
        .python_source
        .contains("x, y = float(state[0]), float(state[1])"));
    assert!(
        art.rust_source.contains("pub fn dx_rhs(") && art.rust_source.contains("pub fn dy_rhs(")
    );

    let src_path = std::env::temp_dir().join("test_planar_custom_ode_export.rs");
    let bin_path = std::env::temp_dir().join(format!(
        "test_planar_custom_ode_export_bin{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&src_path, &art.rust_source).expect("write exported rust source");
    let status = std::process::Command::new("rustc")
        .arg("--edition=2024")
        .arg(&src_path)
        .arg("-o")
        .arg(&bin_path)
        .status()
        .expect("invoke rustc on exported planar solver");
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&bin_path);
    let _ = std::fs::remove_file(bin_path.with_extension("pdb"));
    assert!(
        status.success(),
        "exported planar Rust code must compile cleanly"
    );

    // Recovery clamping & reset on synthesized planar parameters
    tool.planar_bank.set_param("a", 250.0);
    tool.clamp_instability_parameters();
    assert_eq!(tool.planar_bank.active_value("a"), Some(10.0));
    assert!(FiniteStateGuard::is_finite_state(tool));

    tool.reset_parameters();
    assert_eq!(tool.planar_bank.param_names(), &["k"]);
    assert_eq!(tool.planar_bank.active_value("k"), Some(1.0));
    assert!(FiniteStateGuard::is_finite_state(tool));
}

#[test]
fn test_planar_custom_zero_alloc_solver_and_regime_atlas_with_teleport() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::PlanarCustom,
        ic_y0: 2.0,
        ic_v0: 0.0,
        dt: 0.06,
        total_time: 6.0,
        ..Default::default()
    };
    tool.planar_x_input.set_expression("y");
    tool.planar_y_input.set_expression("mu * (1 - x^2) * y - x");
    tool.recalculate();
    tool.planar_bank.set_param("mu", 0.8);
    tool.recalculate();

    let spec = tool.build_custom_ode_spec().expect("planar coupled spec");
    let init = VecState(vec![2.0, 0.0]);
    let mut model = OdeModel::new(init.clone(), spec, RungeKutta4::new(&init));
    let cart_mu = OdeRegimeCartographer::from_tool(&tool, ("mu", "dt"));

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((damped, cycle, stepped), _) =
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..64 {
                model.step(0.02);
            }
            let c_neg = cart_mu.evaluate_cell(-0.6, 0.06);
            let c_pos = cart_mu.evaluate_cell(0.8, 0.08);
            (c_neg, c_pos, model.state.0[0] + model.state.0[1])
        });
    drop(_alloc_guard);

    assert!(stepped.is_finite());
    assert_eq!(
        OdeAtlasRegime::from_u8(damped.regime),
        OdeAtlasRegime::UnderdampedSpiral
    );
    assert_eq!(
        OdeAtlasRegime::from_u8(cycle.regime),
        OdeAtlasRegime::SustainedCycle
    );
    assert_planar_ab_atlas_and_teleport(&mut tool);
}

fn assert_planar_ab_atlas_and_teleport(tool: &mut OdeSolverTool) {
    tool.planar_x_input.set_expression("a * x - y");
    tool.planar_y_input.set_expression("x + b * y");
    tool.recalculate();
    let cart_ab = OdeRegimeCartographer::from_tool(tool, ("a", "b"));
    let c_stable = cart_ab.evaluate_cell(-0.5, -0.5);
    let c_unstable = cart_ab.evaluate_cell(0.8, 0.8);
    assert_eq!(
        OdeAtlasRegime::from_u8(c_stable.regime),
        OdeAtlasRegime::UnderdampedSpiral
    );
    assert_eq!(
        OdeAtlasRegime::from_u8(c_unstable.regime),
        OdeAtlasRegime::UnstableDivergence
    );

    tool.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "a",
        x_val: -0.35,
        y_key: "b",
        y_val: -0.45,
        regime_label: "Underdamped Spiral (c² < 4k)",
    });
    assert_eq!(tool.planar_bank.active_value("a"), Some(-0.35));
    assert_eq!(tool.planar_bank.active_value("b"), Some(-0.45));
    assert!((tool.atlas_param_value("a") - (-0.35)).abs() < 1e-12);
    assert_eq!(tool.snapshot_manager.history_len(), 1);
}
