#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, TraceSeries};
use crate::tabs::analysis::ode::OdeSolverTool;
use crate::tabs::epidemiology::sir::SirTool;
use crate::web_io::PlatformInstant;
use eframe::App;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

type TabCounter = (String, Arc<AtomicUsize>, Arc<AtomicU64>);

struct CountingTab {
    inner: Box<dyn ExplorerTab>,
    save_calls: Arc<AtomicUsize>,
    extra_rev: Arc<AtomicU64>,
}

impl ExplorerTab for CountingTab {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn show(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.inner.show(ctx, frame);
    }
    fn state_revision(&self) -> u64 {
        let base = self.inner.state_revision();
        let bump = self.extra_rev.load(Ordering::Relaxed);
        base.wrapping_add(bump).max(1)
    }
    fn save_state(&self) -> Option<String> {
        self.save_calls.fetch_add(1, Ordering::Relaxed);
        let bump = self.extra_rev.load(Ordering::Relaxed);
        let base = self.inner.save_state();
        if bump == 0 {
            base
        } else {
            Some(format!(
                r#"{{"_drag_rev":{bump},"inner":{}}}"#,
                base.as_deref().unwrap_or("null")
            ))
        }
    }
    fn load_state(&mut self, state: &str) {
        self.inner.load_state(state);
    }
    fn academic_domain(&self) -> &'static str {
        self.inner.academic_domain()
    }
}

fn build_heavy_ode_framework_json() -> String {
    let mut ode = OdeSolverTool::default();
    let curve: Vec<[f64; 2]> = (0..1_000)
        .map(|i| [i as f64 * 0.01, (i as f64 * 0.02).cos()])
        .collect();
    for idx in 0..10 {
        ode.snapshot_manager.add_snapshot(
            format!("ODE Run {}", idx + 1),
            format!("k={:.2}", 0.5 + idx as f64 * 0.2),
            vec![TraceSeries::new("y(t)", curve.clone())],
        );
    }
    ode.callout_manager
        .add_callout([2.0, 0.5], "Pinned equilibrium", [18.0, -18.0]);
    let mut fw = crate::framework::SimulationFramework::new("analysis");
    if let Some(pos) = fw
        .available_tools
        .iter()
        .position(|t| t.id == "OdeSolverTool")
    {
        fw.select_tool(pos);
    }
    fw.active_tool
        .as_mut()
        .unwrap()
        .load_state(&ode.save_state().unwrap());
    fw.save_state().unwrap()
}

fn build_heavy_sir_framework_json() -> String {
    let mut sir = SirTool::default();
    let curve: Vec<[f64; 2]> = (0..1_000)
        .map(|i| [i as f64 * 0.1, 800.0 - i as f64 * 0.5])
        .collect();
    for idx in 0..10 {
        sir.snapshot_manager.add_snapshot(
            format!("SIR Run {}", idx + 1),
            format!("beta={:.2}", 0.4 + idx as f64 * 0.05),
            vec![
                TraceSeries::new("Susceptible", curve.clone()),
                TraceSeries::new("Infected", curve.clone()),
                TraceSeries::new("Recovered", curve.clone()),
            ],
        );
    }
    let mut fw = crate::framework::SimulationFramework::new("epidemiology");
    if let Some(pos) = fw.available_tools.iter().position(|t| t.id == "SirTool") {
        fw.select_tool(pos);
    }
    fw.active_tool
        .as_mut()
        .unwrap()
        .load_state(&sir.save_state().unwrap());
    fw.save_state().unwrap()
}

fn instrument_app_with_heavy_hidden_tabs() -> (MathExplorerApp, Vec<TabCounter>) {
    let mut app = MathExplorerApp::default();
    assert_eq!(app.tabs.len(), 22);

    let ode_fw_json = build_heavy_ode_framework_json();
    let sir_fw_json = build_heavy_sir_framework_json();
    let mut loaded_ode = false;
    let mut loaded_sir = false;
    for tab in &mut app.tabs {
        if tab.name() == "Analysis & Calculus" {
            tab.load_state(&ode_fw_json);
            loaded_ode = true;
        } else if tab.name() == "Epidemiology" {
            tab.load_state(&sir_fw_json);
            loaded_sir = true;
        }
    }
    assert!(loaded_ode && loaded_sir);

    let nt_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Number Theory")
        .expect("Number Theory tab");
    app.selected_tab = nt_idx;

    let mut counters = Vec::with_capacity(app.tabs.len());
    app.tabs = app
        .tabs
        .into_iter()
        .map(|inner| {
            let calls = Arc::new(AtomicUsize::new(0));
            let extra = Arc::new(AtomicU64::new(0));
            counters.push((
                inner.name().to_string(),
                Arc::clone(&calls),
                Arc::clone(&extra),
            ));
            Box::new(CountingTab {
                inner,
                save_calls: calls,
                extra_rev: extra,
            }) as Box<dyn ExplorerTab>
        })
        .collect();
    (app, counters)
}

#[test]
fn test_idle_21_tab_extraction_under_half_ms_with_zero_save_state_calls() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let (mut app, counters) = instrument_app_with_heavy_hidden_tabs();

    // Prime initial workspace state on frame 0
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });
    for (_, calls, _) in &counters {
        calls.store(0, Ordering::Relaxed);
    }

    // Run 5 consecutive idle frames: zero ExplorerTab::save_state calls across all 21 tabs
    for _ in 0..5 {
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            app.update(ctx, &mut frame);
        });
    }
    let idle_frame_calls: usize = counters
        .iter()
        .map(|(_, c, _)| c.load(Ordering::Relaxed))
        .sum();
    assert_eq!(
        idle_frame_calls, 0,
        "Idle frames must perform zero ExplorerTab::save_state calls across hidden and clean visible tabs"
    );

    // Measure direct 21-tab AppState::extract(&app) latency and verify zero tab save_state calls
    let start = PlatformInstant::now();
    let iterations = 20;
    for _ in 0..iterations {
        let extracted = AppState::extract(&app);
        assert_eq!(extracted.tab_states.len(), 22);
    }
    let avg_ms = (start.elapsed().as_secs_f64() * 1000.0) / f64::from(iterations);
    let extract_calls: usize = counters
        .iter()
        .map(|(_, c, _)| c.load(Ordering::Relaxed))
        .sum();
    assert_eq!(extract_calls, 0);
    assert!(
        avg_ms < 0.5,
        "Idle 21-tab AppState::extract took {avg_ms:.4} ms (>= 0.5 ms budget)"
    );
}

#[test]
fn test_drag_editing_skips_extraction_and_commits_once_on_pointer_release() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let (mut app, counters) = instrument_app_with_heavy_hidden_tabs();
    let active_idx = app.selected_tab;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        app.update(ctx, &mut frame);
    });
    for (_, calls, _) in &counters {
        calls.store(0, Ordering::Relaxed);
    }

    let pointer_pos = egui::pos2(240.0, 240.0);
    for step in 1..=4_u64 {
        counters[active_idx].2.store(step, Ordering::Relaxed);
        let mut raw = egui::RawInput::default();
        raw.events.push(egui::Event::PointerButton {
            pos: pointer_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(raw, |ctx| {
            app.update(ctx, &mut frame);
        });
        assert_eq!(
            app.session_manager.lifecycle_state(),
            SessionLifecycleState::DragEditing
        );
    }

    let drag_calls: usize = counters
        .iter()
        .map(|(_, c, _)| c.load(Ordering::Relaxed))
        .sum();
    assert_eq!(
        drag_calls, 0,
        "Active DragEditing frames must skip full tab extraction"
    );

    // Pointer release commits once for the dirty active tab and zero times for hidden tabs
    let mut release_input = egui::RawInput::default();
    release_input.events.push(egui::Event::PointerButton {
        pos: pointer_pos,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = ctx.run(release_input, |ctx| {
        app.update(ctx, &mut frame);
    });
    assert_eq!(
        app.session_manager.lifecycle_state(),
        SessionLifecycleState::Idle
    );
    assert_eq!(counters[active_idx].1.load(Ordering::Relaxed), 1);
    let hidden_calls: usize = counters
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != active_idx)
        .map(|(_, (_, c, _))| c.load(Ordering::Relaxed))
        .sum();
    assert_eq!(hidden_calls, 0);
    assert!(app.session_manager.can_undo());
}

#[test]
fn test_mws_envelope_and_layouts_preserve_baselines_and_callouts() {
    let mut app = MathExplorerApp::default();
    let ode_fw_json = build_heavy_ode_framework_json();
    let sir_fw_json = build_heavy_sir_framework_json();
    for tab in &mut app.tabs {
        if tab.name() == "Analysis & Calculus" {
            tab.load_state(&ode_fw_json);
        } else if tab.name() == "Epidemiology" {
            tab.load_state(&sir_fw_json);
        }
    }

    for layout in [
        ViewportLayout::Single,
        ViewportLayout::SplitHorizontal,
        ViewportLayout::QuadGrid,
    ] {
        app.set_layout(layout);
        let envelope_json = app.session_manager.create_envelope(app.get_app_state());
        let raw_env: WorkspaceEnvelope =
            serde_json::from_str(&envelope_json).expect("deserialize WorkspaceEnvelope");
        assert_eq!(raw_env.schema_version, "1.0");

        let mut parser_mgr = WorkspaceSessionManager::default();
        let parsed_state = parser_mgr
            .parse_envelope(&envelope_json)
            .expect("parse WorkspaceEnvelope 1.0");

        let mut restored_app = MathExplorerApp::default();
        restored_app.apply_app_state(parsed_state);
        let analysis_tab = restored_app
            .tabs
            .iter()
            .find(|t| t.name() == "Analysis & Calculus")
            .unwrap();
        let saved_analysis = analysis_tab.save_state().unwrap();
        assert!(saved_analysis.contains("ODE Run 10"));
        assert!(saved_analysis.contains("Pinned equilibrium"));
    }

    // Verify eframe::Storage autosave JSON roundtrip and RestoringHistory two-frame quiescence
    let autosave_json = app.save_state_to_json();
    let mut autosaved_app = MathExplorerApp::default();
    autosaved_app.load_state_from_json(&autosave_json);
    let ctx = egui::Context::default();
    let mut sm = std::mem::take(&mut autosaved_app.session_manager);
    sm.track_frame_changes(&ctx, &autosaved_app);
    sm.set_is_restoring(true);
    assert!(sm.is_restoring());
    sm.track_frame_changes(&ctx, &autosaved_app);
    assert!(sm.is_restoring());
    sm.track_frame_changes(&ctx, &autosaved_app);
    assert!(!sm.is_restoring());
    autosaved_app.session_manager = sm;
}
