#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::wcag_contrast_on_black;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::framework::state_bounds::ToolStateBounds;
use crate::framework::InteractiveTool;
use scientific_metadata::theory::TheoryDescribable;

#[cfg(feature = "biology")]
#[test]
fn test_morphogenesis_theory_bounds_and_four_regimes() {
    use crate::tabs::morphogenesis::{
        classify_schnakenberg_regime, MorphogenesisCartographer, MorphogenesisRegime,
        MorphogenesisTheory,
    };

    let theory = MorphogenesisTheory::default().theory_parameters();
    for key in ["a", "b", "d_u", "d_v", "dt"] {
        let c = theory.get(key).expect("theory parameter present");
        assert!(c.min.is_finite() && c.max.is_finite() && c.max > c.min);
    }

    assert_eq!(
        MorphogenesisRegime::from_u8(
            classify_schnakenberg_regime(0.6, 0.2, 1.0, 40.0, 0.002).regime
        ),
        MorphogenesisRegime::StableWashout
    );
    assert_eq!(
        MorphogenesisRegime::from_u8(
            classify_schnakenberg_regime(0.1, 0.9, 1.0, 40.0, 0.002).regime
        ),
        MorphogenesisRegime::TuringPattern
    );
    assert_eq!(
        MorphogenesisRegime::from_u8(
            classify_schnakenberg_regime(0.02, 0.4, 1.0, 40.0, 0.002).regime
        ),
        MorphogenesisRegime::BulkHopfOscillation
    );
    assert_eq!(
        MorphogenesisRegime::from_u8(
            classify_schnakenberg_regime(0.1, 0.9, 1.0, 100.0, 0.05).regime
        ),
        MorphogenesisRegime::EulerCflUnstable
    );

    let cart = MorphogenesisCartographer::new(&theory, ("a", "b"), 0.1, 0.9, 1.0, 40.0, 0.002);
    let axes = cart.axes();
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for sy in 0..32 {
            for sx in 0..32 {
                let (p1, p2) = axes.pixel_to_coords(sx, sy, 32, 32);
                let cell = cart.evaluate_cell(p1, p2);
                let hc = cart.color_for_cell(cell, true);
                assert!(wcag_contrast_on_black(hc) >= 3.0);
            }
        }
    });
}

#[cfg(feature = "biology")]
#[test]
fn test_morphogenesis_washout_auto_reseeds_into_turing_patterns_without_reset() {
    use crate::async_sim::SimulationRunner;
    use crate::tabs::morphogenesis::{grid_u_variance, MorphogenesisTool};

    let tool = MorphogenesisTool::default();
    {
        let mut p = tool.params.write().unwrap();
        p.a = 0.6;
        p.b = 0.2;
        p.d_u = 1.0;
        p.d_v = 40.0;
        p.dt = 0.002;
    }
    let mut runner = MorphogenesisTool::create_runner(&tool.params, 24, 24);
    let eq_u = 0.8;
    let eq_v = 0.2 / (0.8 * 0.8);
    runner.system.u_mut().fill(eq_u);
    runner.system.v_mut().fill(eq_v);
    runner.step();
    assert!(
        grid_u_variance(runner.system.u()) < 1e-8,
        "washout regime remains spatially uniform"
    );

    {
        let mut p = tool.params.write().unwrap();
        p.a = 0.1;
        p.b = 0.9;
    }
    for _ in 0..80 {
        runner.step();
    }
    let post_var = grid_u_variance(runner.system.u());
    assert!(
        post_var > 1e-4,
        "entering Turing island from washed-out grid must auto-reseed and grow patterns, got var={post_var}"
    );
}

#[cfg(feature = "biology")]
#[test]
fn test_morphogenesis_state_bounds_and_teleport_reseed() {
    use crate::tabs::morphogenesis::{grid_u_variance, MorphogenesisTool};

    let mut tool = MorphogenesisTool::default();
    {
        let mut p = tool.params.write().unwrap();
        p.a = 0.6;
        p.b = 0.2;
        p.d_v = 40.0;
        p.dt = 0.002;
    }
    let mut sr = MorphogenesisTool::create_runner(&tool.params, tool.width, tool.height);
    sr.system.u_mut().fill(0.8);
    sr.system.v_mut().fill(0.2 / 0.64);
    tool.step_runner = Some(sr);
    assert!(tool.is_active_grid_washed_out());

    tool.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "a",
        x_val: 0.1,
        y_key: "b",
        y_val: 0.9,
        regime_label: "Turing Pattern",
    });
    let reseeded_var = grid_u_variance(tool.step_runner.as_ref().unwrap().system.u());
    assert!(
        reseeded_var > 1e-5,
        "teleport into Turing regime reseeds washed-out step_runner"
    );

    tool.atlas.set_visible(true);
    tool.atlas.set_axis_keys("d_v", "dt").unwrap();
    let saved = tool.save_state().expect("morpho state json");
    ToolStateBounds::DEFAULT
        .validate_json(&saved)
        .expect("within ToolStateBounds");
    assert!(!saved.contains("pixels") && saved.len() < 1024);

    let mut restored = MorphogenesisTool::default();
    restored.try_load_state(&saved).unwrap();
    assert!(restored.atlas.is_visible());
    assert_eq!(restored.atlas.axis_keys(), ("d_v", "dt"));
}

#[cfg(feature = "biology")]
#[test]
fn test_hodgkin_huxley_regimes_and_zero_alloc() {
    use crate::async_sim::unified::UnifiedModel;
    use crate::tabs::neuroscience::hodgkin_huxley::{
        HodgkinHuxleyCartographer, HodgkinHuxleyRegime, HodgkinHuxleyUnified,
    };

    let hh = HodgkinHuxleyUnified::new(&std::collections::HashMap::new());
    assert!(hh.theory_parameters().contains_key("i_ext"));
    assert!(hh.theory_parameters().contains_key("g_na"));

    let mut base = std::collections::HashMap::new();
    base.insert("g_na".to_string(), 120.0);
    base.insert("g_k".to_string(), 36.0);
    base.insert("g_l".to_string(), 0.3);
    base.insert("i_ext".to_string(), 10.0);
    let cart = HodgkinHuxleyCartographer::from_params(&base, ("i_ext", "g_na"));

    assert_eq!(
        HodgkinHuxleyRegime::from_u8(cart.evaluate_cell(0.0, 120.0).regime),
        HodgkinHuxleyRegime::Quiescent
    );
    assert_eq!(
        HodgkinHuxleyRegime::from_u8(cart.evaluate_cell(15.0, 120.0).regime),
        HodgkinHuxleyRegime::TonicFiring
    );
    let found_single = (20..=60).any(|tenths| {
        let cur = f64::from(tenths) * 0.1;
        HodgkinHuxleyRegime::from_u8(cart.evaluate_cell(cur, 120.0).regime)
            == HodgkinHuxleyRegime::SingleSpike
    });
    assert!(found_single, "phasic SingleSpike onset exists");

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        let c = cart.evaluate_cell(12.0, 120.0);
        assert!(wcag_contrast_on_black(cart.color_for_cell(c, true)) >= 3.0);
    });
}

#[cfg(feature = "biology")]
#[test]
fn test_hodgkin_huxley_teleport_and_state_bounds_roundtrip() {
    use crate::async_sim::unified::UnifiedSimTool;
    use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;

    let mut tool = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    tool.apply_atlas_teleport(
        AtlasTeleportEvent {
            x_key: "i_ext",
            x_val: 18.5,
            y_key: "g_na",
            y_val: 135.0,
            regime_label: "Tonic Firing",
        },
        None,
    );
    assert!((tool.params.read().unwrap()["i_ext"] - 18.5).abs() < 1e-12);
    if let Some(atlas) = &mut tool.atlas {
        atlas.set_visible(true);
        atlas.set_axis_keys("i_ext", "g_k").unwrap();
    }
    let json = tool.save_state().expect("hh state");
    ToolStateBounds::DEFAULT.validate_json(&json).unwrap();
    let mut restored = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    restored.try_load_state(&json).unwrap();
    assert_eq!(
        restored.atlas.as_ref().unwrap().axis_keys(),
        ("i_ext", "g_k")
    );
}

#[cfg(feature = "epidemiology")]
#[test]
fn test_sir_final_size_regimes_contour_and_zero_alloc_128x128() {
    use crate::async_sim::atlas::DEFAULT_ATLAS_SIDE;
    use crate::tabs::epidemiology::sir::{SirCartographer, SirRegime, SirTool};

    let tool = SirTool::default();
    let theory = tool.theory_parameters();
    assert!(theory.contains_key("beta") && theory.contains_key("gamma"));

    let cart = SirCartographer::new(&theory, ("beta", "gamma"), 1000.0, 10.0, 0.3, 0.1);
    let sub = cart.evaluate_cell(0.05, 0.25);
    let crit = cart.evaluate_cell(0.20, 0.20);
    let epi = cart.evaluate_cell(0.80, 0.20);
    assert_eq!(SirRegime::from_u8(sub.regime), SirRegime::SubthresholdDecay);
    assert_eq!(
        SirRegime::from_u8(crit.regime),
        SirRegime::CriticalThreshold
    );
    assert_eq!(SirRegime::from_u8(epi.regime), SirRegime::EpidemicOutbreak);
    assert!(sub.primary < 0.03 && epi.primary > 0.95);
    let (contour_label, pts) = cart.overlay_contour().expect("R0 = 1 contour");
    assert!(contour_label.contains("R₀ = 1") && pts.len() == 2);

    let axes = cart.axes();
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for sy in 0..DEFAULT_ATLAS_SIDE {
            for sx in 0..DEFAULT_ATLAS_SIDE {
                let (p1, p2) = axes.pixel_to_coords(sx, sy, DEFAULT_ATLAS_SIDE, DEFAULT_ATLAS_SIDE);
                let cell = cart.evaluate_cell(p1, p2);
                assert!(cell.primary.is_finite());
                assert!(wcag_contrast_on_black(cart.color_for_cell(cell, true)) >= 3.0);
            }
        }
    });
}

#[cfg(feature = "epidemiology")]
#[test]
fn test_sir_atlas_teleport_updates_trajectories_and_state_bounds() {
    use crate::tabs::epidemiology::sir::SirTool;

    let mut tool = SirTool::default();
    let sub_ev = AtlasTeleportEvent {
        x_key: "beta",
        x_val: 0.04,
        y_key: "gamma",
        y_val: 0.20,
        regime_label: "Sub-Threshold Decay (R₀ < 1)",
    };
    tool.apply_atlas_teleport(sub_ev);
    let (_, sub_final) = tool.final_outbreak_metrics();
    assert!(sub_final < 0.02);

    tool.apply_atlas_teleport(AtlasTeleportEvent {
        x_key: "beta",
        x_val: 0.90,
        y_key: "gamma",
        y_val: 0.15,
        regime_label: "Epidemic Outbreak (R₀ > 1)",
    });
    assert!(tool.final_outbreak_metrics().1 > 0.98);
    assert_eq!(tool.snapshot_manager.history_len(), 2);

    tool.atlas.set_visible(true);
    tool.atlas.set_axis_keys("beta", "I0").unwrap();
    let json = tool.save_state().expect("sir json");
    ToolStateBounds::DEFAULT.validate_json(&json).unwrap();
    let mut restored = SirTool::default();
    restored.try_load_state(&json).unwrap();
    assert!(restored.atlas.is_visible());
    assert_eq!(restored.atlas.axis_keys(), ("beta", "I0"));
}

#[cfg(any(
    all(feature = "biology", feature = "epidemiology"),
    feature = "physics"
))]
fn scan_accesskit_atlas_button(out: &eframe::egui::FullOutput) -> bool {
    let Some(ak) = &out.platform_output.accesskit_update else {
        return false;
    };
    ak.nodes
        .iter()
        .filter_map(|(_, node)| node.label())
        .any(|lbl| lbl.contains("2D Parameter Atlas"))
}

#[cfg(all(feature = "biology", feature = "epidemiology"))]
#[test]
fn test_flagship_tools_expose_atlas_toggle_in_accesskit_tree() {
    use eframe::{egui, App};

    let mut app = crate::app::MathExplorerApp::default();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut frame = eframe::Frame::_new_kittest();
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0));
    let targets = [
        ("Morphogenesis", "MorphogenesisTool"),
        ("Epidemiology", "SirTool"),
        ("Neuroscience", "HodgkinHuxleyTool"),
    ];

    for (tab_name, tool_id) in targets {
        let tab_idx = app
            .tabs
            .iter()
            .position(|t| t.name().contains(tab_name))
            .unwrap_or_else(|| panic!("missing tab {tab_name}"));
        app.set_pane_tab(0, tab_idx);
        assert!(app.tabs[tab_idx].select_tool_by_name(tool_id));
        let mut found = false;
        for _ in 0..2 {
            let raw = egui::RawInput {
                screen_rect: Some(rect),
                ..Default::default()
            };
            let out = ctx.run(raw, |egui_ctx| app.update(egui_ctx, &mut frame));
            found |= scan_accesskit_atlas_button(&out);
        }
        assert!(found, "{tool_id} must expose 2D Parameter Atlas button");
    }
}

#[cfg(feature = "physics")]
fn tp(x_key: &'static str, x_val: f64, y_key: &'static str, y_val: f64) -> AtlasTeleportEvent {
    AtlasTeleportEvent {
        x_key,
        x_val,
        y_key,
        y_val,
        regime_label: "chaos",
    }
}

#[cfg(feature = "physics")]
#[test]
fn test_chaos_fractal_and_bifurcation_zero_alloc_wcag() {
    use crate::tabs::chaos::bifurcation as bif;
    use crate::tabs::chaos::fractals as frac;

    let f_th = frac::FractalViewer::default().theory_parameters();
    let f_pair = ("julia_re", "julia_im");
    let c0 = num_complex::Complex::new(0.0, 0.0);
    let f_c = frac::FractalCartographer::new(
        &f_th,
        f_pair,
        frac::CompiledFractalJob::quadratic(),
        c0,
        1.0,
    );
    for ((re, im), exp) in [((0.0, 0.0), 0u8), ((0.0, 1.0), 1), ((0.8, 0.6), 2)] {
        assert_eq!(f_c.evaluate_cell(re, im).regime, exp);
    }

    let b_th = bif::BifurcationDiagram::default().theory_parameters();
    let b_mode = bif::BifurcationMapMode::Logistic;
    let b_c = bif::BifurcationCartographer::new(&b_th, ("r", "x0"), b_mode, None, 3.5, 0.5, 0.0);
    for ((r, x0), exp) in [((2.5, 0.4), 0u8), ((3.9, 0.4), 2), ((3.8, 1.2), 3)] {
        assert_eq!(b_c.evaluate_cell(r, x0).regime, exp);
    }

    let (fa, ba) = (f_c.axes(), b_c.axes());
    let _g = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for i in 0..16 {
            let (fx, fy) = fa.pixel_to_coords(i, i, 16, 16);
            let fc = f_c.color_for_cell(f_c.evaluate_cell(fx, fy), true);
            assert!(wcag_contrast_on_black(fc) >= 3.0);
            let (bx, by) = ba.pixel_to_coords(i, i, 16, 16);
            let bc = b_c.color_for_cell(b_c.evaluate_cell(bx, by), true);
            assert!(wcag_contrast_on_black(bc) >= 3.0);
        }
    });
}

#[cfg(feature = "physics")]
#[test]
fn test_chaos_attractor_hopf_and_zero_alloc() {
    use crate::tabs::chaos::attractors as att;

    let pl = att::AttractorPlotter::default();
    let th = pl.theory_parameters();
    let job = att::CompiledFlowJob::from_bank(&pl.flow_bank);
    let base = (10.0, 28.0, 8.0 / 3.0);
    let a_c = att::AttractorCartographer::new(
        &th,
        ("sigma", "rho"),
        att::AttractorMode::Lorenz,
        job.clone(),
        base,
        (0.2, 0.2),
    );
    let c_c = att::AttractorCartographer::new(
        &th,
        ("param_0", "param_1"),
        att::AttractorMode::Custom,
        job,
        base,
        (0.2, 0.2),
    );
    assert!((att::lorenz_hopf_threshold(10.0, 8.0 / 3.0).unwrap() - 470.0 / 19.0).abs() < 1e-9);
    assert!(a_c.overlay_contour().expect("Hopf").1.len() >= 10);
    assert_eq!(a_c.evaluate_cell(10.0, 0.5).regime, 0);
    assert_eq!(a_c.evaluate_cell(10.0, 28.0).regime, 2);
    let _g = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        let l_col = a_c.color_for_cell(a_c.evaluate_cell(10.0, 28.0), true);
        let c_col = c_c.color_for_cell(c_c.evaluate_cell(0.2, 0.2), true);
        assert!(wcag_contrast_on_black(l_col) >= 3.0 && wcag_contrast_on_black(c_col) >= 3.0);
    });
}

#[cfg(feature = "physics")]
#[test]
fn test_chaos_teleport_bounds_and_a11y() {
    use crate::tabs::chaos::{
        attractors::AttractorPlotter, bifurcation::BifurcationDiagram, fractals::FractalViewer,
    };
    use eframe::egui;

    let mut f = FractalViewer::default();
    f.apply_atlas_teleport(tp("julia_re", -0.12, "julia_im", 0.75));
    f.atlas.set_visible(true);
    f.atlas.set_axis_keys("julia_re", "param_0").unwrap();
    let mut b = BifurcationDiagram::default();
    b.apply_atlas_teleport(None, tp("r", 3.85, "x0", 0.35));
    b.atlas.set_visible(true);
    b.atlas.set_axis_keys("r", "param_0").unwrap();
    let mut p = AttractorPlotter::default();
    p.apply_atlas_teleport(tp("sigma", 12.0, "rho", 32.0));
    p.atlas.set_visible(true);
    p.atlas.set_axis_keys("rho", "beta").unwrap();
    assert!((f.julia_c.re + 0.12).abs() < 1e-9 && (b.probe_r - 3.85).abs() < 1e-9);
    assert!((p.system.sigma - 12.0).abs() < 1e-9 && (p.system.rho - 32.0).abs() < 1e-9);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0));
    let raw = egui::RawInput {
        screen_rect: Some(rect),
        ..Default::default()
    };
    for tool in [&mut f as &mut dyn InteractiveTool, &mut b, &mut p] {
        let json = tool.save_state().unwrap();
        ToolStateBounds::DEFAULT.validate_json(&json).unwrap();
        tool.try_load_state(&json).unwrap();
        let out = ctx.run(raw.clone(), |c| tool.show(c));
        assert!(scan_accesskit_atlas_button(&out));
    }
}
