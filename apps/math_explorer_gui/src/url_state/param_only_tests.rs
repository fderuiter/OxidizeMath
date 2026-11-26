#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, SimulationFramework};

#[cfg(not(target_arch = "wasm32"))]
fn write_restored_plot_png(path: &str, datasets: &[(String, Vec<[f64; 2]>)]) {
    let decimated: Vec<(String, Vec<[f64; 2]>)> = datasets
        .iter()
        .map(|(name, pts)| {
            let step = (pts.len() / 500).max(1);
            (name.clone(), pts.iter().step_by(step).copied().collect())
        })
        .collect();
    if let Ok(png) = egui_plot::export::export_custom_png(&decimated, &Default::default()) {
        let _ = std::fs::write(path, png);
    }
}

#[cfg(feature = "pure_math")]
fn build_populated_ode_200k_with_15_runs() -> crate::tabs::analysis::ode::OdeSolverTool {
    use crate::framework::TraceSeries;
    use crate::presets::{OdePreset, MAX_ODE_STEPS};
    use crate::tabs::analysis::ode::OdeSolverTool;
    use math_explorer::pure_math::analysis::ode::IntegrationMethod;

    let mut ode = OdeSolverTool::default();
    ode.preset = OdePreset::LogisticGrowth;
    ode.method = IntegrationMethod::Heun;
    ode.total_time = 100.0;
    ode.dt = ode.total_time / (MAX_ODE_STEPS as f64);
    ode.param_r = 2.4;
    ode.param_cap_k = 42.0;
    ode.ic_y0 = 3.5;
    ode.recalculate();
    assert_eq!(ode.y_series.len(), MAX_ODE_STEPS + 1);

    let full_series = ode.build_current_trace_series();
    for i in 0..crate::framework::MAX_HISTORY_RUNS {
        ode.snapshot_manager.push_history_run(
            format!("r={:.2}", 1.5 + i as f64 * 0.2),
            full_series.clone(),
        );
    }
    for i in 0..crate::framework::MAX_SNAPSHOTS {
        assert!(ode.snapshot_manager.add_snapshot(
            format!("Pinned {}", i + 1),
            format!("K={}", 20 + i * 5),
            vec![TraceSeries::new("y(t)", full_series[0].points.clone())],
        ));
    }
    ode.bump_revision();
    ode
}

#[cfg(feature = "pure_math")]
fn assert_ode_url_restore_and_snapshot_revision_isolation(fw: &mut SimulationFramework) {
    use crate::tabs::analysis::ode::OdeSolverTool;

    let share_url = fw.encode_share_url().unwrap();
    let hash = &share_url[share_url.find('#').unwrap()..];
    let parsed = parse_share_location(hash, "").unwrap().unwrap();
    let mut restored_fw = SimulationFramework::new("analysis");
    restored_fw.import_url_share_params(&parsed, None).unwrap();
    let restored_ode = restored_fw
        .active_tool
        .as_ref()
        .unwrap()
        .as_any()
        .unwrap()
        .downcast_ref::<OdeSolverTool>()
        .unwrap();
    assert!(restored_ode.snapshot_manager.is_empty());
    assert_eq!(restored_ode.param_r, 2.4);
    assert_eq!(restored_ode.param_cap_k, 42.0);
    assert_eq!(restored_ode.ic_y0, 3.5);
    assert_eq!(restored_ode.total_time, 100.0);
    assert_eq!(restored_fw.url_state_revision(), fw.url_state_revision());
    #[cfg(not(target_arch = "wasm32"))]
    write_restored_plot_png(
        "/tmp/ode_param_only_url_roundtrip.png",
        &restored_ode.export_datasets(),
    );

    let url_rev_before = fw.url_state_revision();
    let session_rev_before = fw.state_revision();
    {
        let active_ode = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<OdeSolverTool>()
            .unwrap();
        active_ode.snapshot_manager.toggle_visibility(0);
        active_ode.snapshot_manager.remove_snapshot(9);
        active_ode.bump_revision();
    }
    assert_ne!(fw.state_revision(), session_rev_before);
    assert_eq!(fw.url_state_revision(), url_rev_before);
}

#[cfg(feature = "pure_math")]
#[test]
fn test_ode_200k_steps_15_runs_exports_url_under_4kib_and_1ms_and_skips_pin_sync() {
    use crate::web_io::PlatformInstant;

    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ode = build_populated_ode_200k_with_15_runs();
    assert_eq!(ode.snapshot_manager.history_len(), 5);
    assert_eq!(ode.snapshot_manager.len(), 10);
    assert!(ode.save_state().unwrap().len() > MAX_DECODED_URL_STATE_BYTES);
    assert!(ode
        .export_preset_json()
        .unwrap()
        .contains("snapshot_manager"));

    let mut fw = SimulationFramework::new("analysis");
    assert!(fw.select_tool_by_name("OdeSolverTool"));
    fw.active_tool = Some(Box::new(ode));

    let session_json = fw.save_state().unwrap();
    assert!(session_json.contains("snapshot_manager"));

    let preset_json = fw.export_url_preset_json().unwrap().unwrap();
    assert!(preset_json.len() <= MAX_DECODED_URL_STATE_BYTES);
    assert!(!preset_json.contains("snapshot_manager"));

    let t0 = PlatformInstant::now();
    for _ in 0..10 {
        let url = fw.encode_share_url().expect("valid ODE share URL");
        assert!(url.len() <= MAX_URL_FRAGMENT_BYTES + 64);
    }
    let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    assert!(
        avg_ms < 1.0,
        "ODE URL export + validation took {avg_ms:.4} ms (>= 1.0 ms) at MAX_ODE_STEPS = 200,000"
    );

    assert_ode_url_restore_and_snapshot_revision_isolation(&mut fw);
}

#[cfg(feature = "epidemiology")]
fn build_populated_sir_with_15_runs() -> crate::tabs::epidemiology::sir::SirTool {
    use crate::framework::TraceSeries;
    use crate::tabs::epidemiology::sir::SirTool;

    let mut sir = SirTool::default();
    sir.n = 25_000.0;
    sir.i0 = 150.0;
    sir.beta = 0.65;
    sir.gamma = 0.14;
    sir.duration = 365.0;
    sir.atlas.set_axis_keys("beta", "I0").unwrap();
    sir.atlas.set_visible(true);
    sir.recalculate();

    let to_pairs =
        |pts: &[egui_plot::PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
    let series = vec![
        TraceSeries::new("Susceptible", to_pairs(&sir.s_points)),
        TraceSeries::new("Infected", to_pairs(&sir.i_points)),
        TraceSeries::new("Recovered", to_pairs(&sir.r_points)),
    ];
    for i in 0..crate::framework::MAX_HISTORY_RUNS {
        sir.snapshot_manager
            .push_history_run(format!("beta={:.2}", 0.2 + i as f64 * 0.1), series.clone());
    }
    for i in 0..crate::framework::MAX_SNAPSHOTS {
        assert!(sir.snapshot_manager.add_snapshot(
            format!("Baseline {}", i + 1),
            sir.param_summary(),
            series.clone(),
        ));
    }
    sir
}

#[cfg(feature = "epidemiology")]
fn assert_sir_url_restore_and_snapshot_revision_isolation(fw: &mut SimulationFramework) {
    use crate::tabs::epidemiology::sir::SirTool;

    let share_url = fw.encode_share_url().unwrap();
    let hash = &share_url[share_url.find('#').unwrap()..];
    let parsed = parse_share_location(hash, "").unwrap().unwrap();
    let mut restored_fw = SimulationFramework::new("epidemiology");
    restored_fw.import_url_share_params(&parsed, None).unwrap();
    let restored_sir = restored_fw
        .active_tool
        .as_ref()
        .unwrap()
        .as_any()
        .unwrap()
        .downcast_ref::<SirTool>()
        .unwrap();
    assert!(restored_sir.snapshot_manager.is_empty());
    assert_eq!(restored_sir.n, 25_000.0);
    assert_eq!(restored_sir.i0, 150.0);
    assert_eq!(restored_sir.beta, 0.65);
    assert_eq!(restored_sir.gamma, 0.14);
    assert_eq!(restored_sir.duration, 365.0);
    assert!(restored_sir.atlas.is_visible());
    assert_eq!(restored_sir.atlas.axis_keys(), ("beta", "I0"));
    #[cfg(not(target_arch = "wasm32"))]
    write_restored_plot_png(
        "/tmp/sir_param_only_url_roundtrip.png",
        &restored_sir.export_datasets(),
    );

    let url_rev_before = fw.url_state_revision();
    let session_rev_before = fw.state_revision();
    {
        let active_sir = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<SirTool>()
            .unwrap();
        active_sir.snapshot_manager.toggle_visibility(0);
        active_sir.snapshot_manager.remove_snapshot(9);
    }
    assert_ne!(fw.state_revision(), session_rev_before);
    assert_eq!(fw.url_state_revision(), url_rev_before);
}

#[cfg(feature = "epidemiology")]
#[test]
fn test_sir_15_runs_exports_url_under_4kib_and_1ms_and_skips_pin_sync() {
    use crate::web_io::PlatformInstant;

    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let sir = build_populated_sir_with_15_runs();
    assert_eq!(sir.snapshot_manager.history_len(), 5);
    assert_eq!(sir.snapshot_manager.len(), 10);
    assert!(sir.save_state().unwrap().contains("snapshot_manager"));
    assert!(sir
        .export_preset_json()
        .unwrap()
        .contains("snapshot_manager"));

    let mut fw = SimulationFramework::new("epidemiology");
    assert!(fw.select_tool_by_name("SirTool"));
    fw.active_tool = Some(Box::new(sir));

    let session_json = fw.save_state().unwrap();
    assert!(session_json.contains("snapshot_manager"));

    let preset_json = fw.export_url_preset_json().unwrap().unwrap();
    assert!(preset_json.len() <= MAX_DECODED_URL_STATE_BYTES);
    assert!(!preset_json.contains("snapshot_manager"));

    let t0 = PlatformInstant::now();
    for _ in 0..10 {
        let url = fw.encode_share_url().expect("valid SIR share URL");
        assert!(url.len() <= MAX_URL_FRAGMENT_BYTES + 64);
    }
    let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    assert!(
        avg_ms < 1.0,
        "SIR URL export took {avg_ms:.4} ms (>= 1.0 ms)"
    );

    assert_sir_url_restore_and_snapshot_revision_isolation(&mut fw);
}

#[cfg(all(
    feature = "ai",
    feature = "applied",
    feature = "biology",
    feature = "climate",
    feature = "epidemiology",
    feature = "physics",
    feature = "pure_math"
))]
mod full_app_tests {
    use super::*;
    use crate::app::MathExplorerApp;
    use eframe::{egui, App};

    fn build_untracked_nav_inputs(screen: egui::Rect) -> [egui::RawInput; 3] {
        let hover = egui::pos2(800.0, 400.0);
        [
            egui::RawInput {
                screen_rect: Some(screen),
                events: vec![
                    egui::Event::PointerMoved(hover),
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: egui::vec2(0.0, -40.0),
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                ..Default::default()
            },
            egui::RawInput {
                screen_rect: Some(screen),
                events: vec![egui::Event::PointerMoved(hover), egui::Event::Zoom(1.2)],
                ..Default::default()
            },
            egui::RawInput {
                screen_rect: Some(screen),
                events: vec![
                    egui::Event::Key {
                        key: egui::Key::Tab,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                    egui::Event::Key {
                        key: egui::Key::F6,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                ..Default::default()
            },
        ]
    }

    fn select_chaos_tool(app: &mut MathExplorerApp, tool_name: &str) -> usize {
        let idx = app
            .tabs
            .iter()
            .position(|t| t.url_slug() == "chaos")
            .unwrap();
        app.selected_tab = idx;
        assert!(app.tabs[idx].select_tool_by_name(tool_name));
        idx
    }

    #[test]
    fn test_untracked_navigation_skips_url_sync_and_schedules_no_repaints() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let ctx = egui::Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let mut app = MathExplorerApp::default();
        let chaos_idx = select_chaos_tool(&mut app, "BifurcationDiagram");
        let active = app.tabs[chaos_idx]
            .framework()
            .unwrap()
            .active_tool
            .as_ref();
        assert_eq!(active.unwrap().url_state_revision(), 0);

        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
        let init_in = egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        };
        let _ = ctx.run(init_in, |c| app.update(c, &mut frame));
        let baseline_sig = app.current_url_signature();
        let mut sync = UrlSyncController::default();
        let mut encode_calls = 0_usize;
        assert_eq!(
            sync.poll_at_millis(baseline_sig, false, 0, URL_SYNC_DEBOUNCE_MS, || {
                encode_calls += 1;
                app.encode_current_share_fragment()
            }),
            None
        );

        for (step, input) in build_untracked_nav_inputs(screen).into_iter().enumerate() {
            let _ = ctx.run(input, |c| app.update(c, &mut frame));
            let sig = app.current_url_signature();
            assert_eq!(sig, baseline_sig, "Navigation step {step} mutated URL sig");
            let now_ms = (step as u64 + 1) * 400;
            let pending = sync.poll_at_millis(sig, false, now_ms, URL_SYNC_DEBOUNCE_MS, || {
                encode_calls += 1;
                app.encode_current_share_fragment()
            });
            assert_eq!(pending, None);
        }
        assert_eq!(encode_calls, 0);
    }

    fn find_fractal_canvas_center(out: &egui::FullOutput) -> egui::Pos2 {
        if let Some(update) = &out.platform_output.accesskit_update {
            for (_id, node) in &update.nodes {
                if node.role() == egui::accesskit::Role::Image {
                    if let Some(b) = node.bounds() {
                        return egui::pos2(
                            ((b.x0 + b.x1) * 0.5) as f32,
                            ((b.y0 + b.y1) * 0.5) as f32,
                        );
                    }
                }
            }
        }
        egui::pos2(850.0, 420.0)
    }

    #[test]
    fn test_fractal_viewer_scroll_zoom_flushes_after_300ms_debounce() {
        let _guard = crate::reflective_ui::tests::lock_test_mutex();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut frame = eframe::Frame::_new_kittest();
        let mut app = MathExplorerApp::default();
        let _ = select_chaos_tool(&mut app, "FractalViewer");

        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
        let init_in = egui::RawInput {
            screen_rect: Some(screen),
            max_texture_side: Some(2048),
            ..Default::default()
        };
        let out0 = ctx.run(init_in, |c| app.update(c, &mut frame));
        let hover_pos = find_fractal_canvas_center(&out0);
        let mut sync = UrlSyncController::default();
        let mut encode_calls = 0_usize;
        let base_sig = app.current_url_signature();
        sync.mark_synced(base_sig, Some(app.encode_current_share_fragment().unwrap()));

        let wheel = egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, 24.0),
            modifiers: egui::Modifiers::NONE,
        };
        let zoom_in = egui::RawInput {
            screen_rect: Some(screen),
            max_texture_side: Some(2048),
            events: vec![egui::Event::PointerMoved(hover_pos), wheel],
            ..Default::default()
        };
        let _ = ctx.run(zoom_in, |c| app.update(c, &mut frame));
        let zoomed_sig = app.current_url_signature();
        assert_ne!(zoomed_sig, base_sig);
        assert!(sync
            .poll_at_millis(zoomed_sig, false, 100, URL_SYNC_DEBOUNCE_MS, || {
                encode_calls += 1;
                app.encode_current_share_fragment()
            })
            .is_some());
        assert_eq!(encode_calls, 0);

        let flush_ms = 100 + URL_SYNC_DEBOUNCE_MS + 10;
        let after = sync.poll_at_millis(zoomed_sig, false, flush_ms, URL_SYNC_DEBOUNCE_MS, || {
            encode_calls += 1;
            app.encode_current_share_fragment()
        });
        assert_eq!(after, None);
        assert_eq!(encode_calls, 1);
        assert!(sync
            .last_synced_fragment()
            .unwrap()
            .starts_with("#tab=chaos&tool=FractalViewer&state="));
    }
}
