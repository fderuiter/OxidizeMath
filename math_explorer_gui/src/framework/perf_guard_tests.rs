#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::tabs::analysis::ode::OdeSolverTool;
use crate::tabs::climate::temperature::TemperatureAnomaliesTool;
use crate::tabs::epidemiology::sir::SirTool;
use crate::tabs::geometry_topology::vietoris_rips::VietorisRipsTool;
use crate::tabs::graph_theory::graph_editor::GraphEditorTool;
use crate::web_io::PlatformInstant;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct SaveStateTrapTool {
    inner: OdeSolverTool,
    save_calls: Arc<AtomicUsize>,
    finite_override: Option<bool>,
}

impl InteractiveTool for SaveStateTrapTool {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn domain(&self) -> &'static str {
        self.inner.domain()
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self.inner.theory()
    }
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        self.inner.show_ui(ui);
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.inner.export_datasets()
    }
    fn state_revision(&self) -> u64 {
        self.inner.state_revision()
    }
    fn url_state_revision(&self) -> u64 {
        self.inner.url_state_revision()
    }
    fn has_finite_parameters(&self) -> bool {
        self.finite_override
            .unwrap_or_else(|| self.inner.has_finite_parameters())
    }
    fn save_state(&self) -> Option<String> {
        self.save_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.save_state()
    }
    fn save_recovery_state(&self) -> Option<String> {
        self.inner.save_recovery_state()
    }
    fn save_url_state(&self) -> Option<String> {
        self.inner.save_url_state()
    }
    fn load_state(&mut self, state: &str) {
        self.inner.load_state(state);
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.inner.try_load_state(state)
    }
    fn try_load_recovery_state(&mut self, state: &str) -> Result<(), String> {
        self.inner.try_load_recovery_state(state)
    }
    fn load_recovery_state(&mut self, state: &str) {
        self.inner.load_recovery_state(state);
    }
}

fn populate_ode_with_runs(pinned: usize, history: usize, points_per_run: usize) -> OdeSolverTool {
    let mut tool = OdeSolverTool::default();
    let curve: Vec<[f64; 2]> = (0..points_per_run)
        .map(|i| {
            let t = i as f64 * 0.01;
            [t, (-0.2 * t).exp() * t.cos()]
        })
        .collect();
    for idx in 0..pinned {
        tool.snapshot_manager.add_snapshot(
            format!("Pinned {}", idx + 1),
            format!("k={:.2}", 1.0 + idx as f64 * 0.1),
            vec![TraceSeries::new("y(t)", curve.clone())],
        );
    }
    for idx in 0..history {
        tool.snapshot_manager.push_history_run(
            format!("hist={idx}"),
            vec![TraceSeries::new("y(t)", curve.clone())],
        );
    }
    tool.callout_manager
        .add_callout([1.25, 0.75], "Inflection baseline", [20.0, -20.0]);
    tool.bump_revision();
    tool
}

fn populate_sir_with_runs(pinned: usize, history: usize, points_per_run: usize) -> SirTool {
    let mut tool = SirTool::default();
    let curve: Vec<[f64; 2]> = (0..points_per_run)
        .map(|i| [i as f64 * 0.1, 500.0 - i as f64 * 0.2])
        .collect();
    for idx in 0..pinned {
        tool.snapshot_manager.add_snapshot(
            format!("SIR Pinned {}", idx + 1),
            format!("beta={:.2}", 0.3 + idx as f64 * 0.05),
            vec![
                TraceSeries::new("Susceptible", curve.clone()),
                TraceSeries::new("Infected", curve.clone()),
                TraceSeries::new("Recovered", curve.clone()),
            ],
        );
    }
    for idx in 0..history {
        tool.snapshot_manager.push_history_run(
            format!("sir-hist={idx}"),
            vec![TraceSeries::new("Infected", curve.clone())],
        );
    }
    tool
}

#[test]
fn test_trace_series_arc_clone_is_constant_time() {
    let raw_pts: Vec<[f64; 2]> = (0..10_000).map(|i| [i as f64, (i as f64).sin()]).collect();
    let series = TraceSeries::new("y(t)", raw_pts);
    let cloned = series.clone();
    assert!(
        series.points.ptr_eq(&cloned.points),
        "Cloning TraceSeries must share Arc-backed coordinates without copying points"
    );

    let mut mgr = TraceSnapshotManager::new();
    mgr.add_snapshot("Run 1", "k=1.0", vec![series]);
    let mgr_clone = mgr.clone();
    assert!(mgr.snapshots()[0].series[0]
        .points
        .ptr_eq(&mgr_clone.snapshots()[0].series[0].points));

    let encoded = serde_json::to_string(&mgr).expect("serialize TraceSnapshotManager");
    let decoded: TraceSnapshotManager =
        serde_json::from_str(&encoded).expect("deserialize TraceSnapshotManager");
    assert_eq!(decoded.snapshots()[0].series[0].points.len(), 10_000);
}

#[test]
fn test_framework_save_state_embeds_raw_value_without_inner_escaping() {
    let mut framework = SimulationFramework::new("analysis");
    framework.saved_tool_states.insert(
        "Inactive Custom Tool".to_string(),
        r#"{"alpha":1.5,"label":"wave \"A\""}"#.to_string(),
    );

    let saved = framework.save_state().expect("framework save_state");
    assert!(
        saved.contains(r#""Inactive Custom Tool":{"alpha":1.5,"label":"wave \"A\""}"#),
        "Inactive saved_tool_states must be embedded via RawValue without second-pass string escaping: {saved}"
    );

    let mut restored = SimulationFramework::new("analysis");
    restored.load_state(&saved);
    assert_eq!(
        restored
            .saved_tool_states
            .get("Inactive Custom Tool")
            .map(String::as_str),
        Some(r#"{"alpha":1.5,"label":"wave \"A\""}"#)
    );

    // Verify backward compatibility with legacy stringified JSON tool_states
    let legacy_json = r#"{"selected_tool_index":0,"selected_tool_name":"OdeSolverTool","show_theory_portal":false,"show_starter_gallery":false,"tool_states":{"LegacyTool":"{\"beta\":2.5}"}}"#;
    let mut legacy_restored = SimulationFramework::new("analysis");
    legacy_restored.load_state(legacy_json);
    assert_eq!(
        legacy_restored
            .saved_tool_states
            .get("LegacyTool")
            .map(String::as_str),
        Some(r#"{"beta":2.5}"#)
    );
}

#[test]
fn test_inspect_runtime_state_under_1ms_with_15_runs_without_save_state() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ode = populate_ode_with_runs(10, 5, 1_000);
    assert_eq!(ode.snapshot_manager.len(), 10);
    assert_eq!(ode.snapshot_manager.history_len(), 5);

    let save_calls = Arc::new(AtomicUsize::new(0));
    let mut framework = SimulationFramework::new("analysis");
    framework.active_tool = Some(Box::new(SaveStateTrapTool {
        inner: ode,
        save_calls: Arc::clone(&save_calls),
        finite_override: None,
    }));

    egui_plot::clear_last_plot_datasets();
    egui_plot::append_last_plot_datasets(&[(
        "y(t)".to_string(),
        (0..250).map(|i| [i as f64 * 0.04, 1.0]).collect(),
    )]);

    assert!(framework.inspect_runtime_state().is_ok());
    let start = PlatformInstant::now();
    for _ in 0..10 {
        assert!(framework.inspect_runtime_state().is_ok());
    }
    let avg_secs = start.elapsed().as_secs_f64() / 10.0;
    assert!(
        avg_secs < 0.001,
        "inspect_runtime_state took {:.4} ms (>= 1.0 ms)",
        avg_secs * 1000.0
    );
    assert_eq!(
        save_calls.load(Ordering::Relaxed),
        0,
        "inspect_runtime_state must never call InteractiveTool::save_state"
    );

    // Non-finite parameter triggers divergence without calling save_state
    if let Some(active) = framework.active_tool.as_mut() {
        let nan_ode = OdeSolverTool {
            param_k: f64::NAN,
            ..Default::default()
        };
        *active = Box::new(SaveStateTrapTool {
            inner: nan_ode,
            save_calls: Arc::clone(&save_calls),
            finite_override: None,
        });
    }
    assert!(framework.inspect_runtime_state().is_err());
    assert!(framework.is_diverged());
    assert_eq!(save_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn test_snapshot_under_2mb_and_pinned_baselines_survive_rollbacks() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ode = populate_ode_with_runs(10, 5, 2_500);
    let expected_callouts = ode.callout_manager.len();
    let ode_json = ode.save_state().expect("full ode save_state");
    let mut framework = SimulationFramework::new("analysis");
    if let Some(pos) = framework
        .available_tools
        .iter()
        .position(|t| t.id == "OdeSolverTool")
    {
        framework.select_tool(pos);
    }
    framework
        .active_tool
        .as_mut()
        .unwrap()
        .load_state(&ode_json);
    egui_plot::clear_last_plot_datasets();
    framework.cache_valid_snapshot();

    let snap = framework.last_valid_snapshot().expect("cached snapshot");
    assert!(
        snap.memory_size() <= MAX_SNAPSHOT_BYTES,
        "ODE recovery snapshot {} exceeded 2 MB",
        snap.memory_size()
    );

    // Diverge, then verify Step Back, Auto-scale dt, and Clamp preserve all 10 pinned runs
    for mode in 0..3 {
        egui_plot::clear_last_plot_datasets();
        egui_plot::append_last_plot_datasets(&[(
            "y(t)".to_string(),
            vec![[0.0, 1.0], [0.05, f64::INFINITY]],
        )]);
        assert!(framework.inspect_runtime_state().is_err());
        match mode {
            0 => assert!(framework.rollback_step_back()),
            1 => assert!(framework.rollback_auto_scale_step_size()),
            _ => {
                framework.clamp_parameters();
                assert!(!framework.is_diverged());
            }
        }
        let full_after = framework
            .active_tool
            .as_ref()
            .unwrap()
            .save_state()
            .unwrap();
        assert!(full_after.contains("Inflection baseline"));
        let mut check_ode = OdeSolverTool::default();
        check_ode.load_state(&full_after);
        assert_eq!(check_ode.snapshot_manager.len(), 10);
        assert_eq!(check_ode.callout_manager.len(), expected_callouts);
    }

    // Verify SirTool with 15 runs also stays well below 2 MB
    let sir = populate_sir_with_runs(10, 5, 2_000);
    let mut epi_fw = SimulationFramework::new("epidemiology");
    epi_fw.active_tool = Some(Box::new(sir));
    epi_fw.cache_valid_snapshot();
    assert!(epi_fw.last_valid_snapshot().unwrap().memory_size() <= MAX_SNAPSHOT_BYTES);
}

#[test]
fn test_temperature_omits_static_series_and_heavy_tools_roundtrip() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut temp = TemperatureAnomaliesTool::default();
    let default_json = temp.save_state().expect("temperature save_state");
    assert_eq!(
        default_json, "{}",
        "Default TemperatureAnomaliesTool must omit static climate series from save_state"
    );
    egui_plot::clear_last_plot_datasets();
    let _ = ctx.run(egui::RawInput::default(), |ctx| temp.show(ctx));
    let rendered = egui_plot::get_last_plot_datasets();
    assert!(!rendered.is_empty() && rendered[0].1.len() > 50);

    let mut custom_temp = TemperatureAnomaliesTool::default();
    custom_temp.load_state(r#"{"time_series":[[2020.0,1.25],[2021.0,1.31]]}"#);
    egui_plot::clear_last_plot_datasets();
    let _ = ctx.run(egui::RawInput::default(), |ctx| custom_temp.show(ctx));
    assert_eq!(egui_plot::get_last_plot_datasets()[0].1.len(), 2);

    let mut vr = VietorisRipsTool::default();
    let vr_input = r#"{"radius":1.5,"points":[[0.0,1.0],[2.0,3.5],[4.25,1.75]]}"#;
    vr.load_state(vr_input);
    assert!(vr.has_finite_parameters() && vr.state_revision() > 0);
    let vr_json = vr.save_state().expect("vr save_state");
    assert_eq!(vr_json, vr_input);
    let mut vr2 = VietorisRipsTool::default();
    vr2.load_state(&vr_json);
    assert_eq!(vr_json, vr2.save_state().unwrap());

    let mut ge = GraphEditorTool::default();
    ge.load_state(
        r#"{"node_positions":{"0":[10.0,20.0],"1":[30.0,40.0]},"edges":[[0,1,1.5]],"selected_node":0,"next_node_id":2}"#,
    );
    assert!(ge.has_finite_parameters() && ge.state_revision() > 0);
    assert_eq!(
        ge.save_state().as_deref(),
        Some(
            r#"{"node_positions":{"0":[10.0,20.0],"1":[30.0,40.0]},"edges":[[0,1,1.5]],"selected_node":0,"next_node_id":2}"#
        )
    );
}

#[test]
fn test_all_registered_tools_snapshot_memory_within_2mb_and_export_pngs() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut checked = 0_usize;
    for meta in inventory::iter::<ToolMetadata> {
        let mut fw = SimulationFramework::new(meta.domain);
        if let Some(pos) = fw.available_tools.iter().position(|t| t.id == meta.id) {
            fw.select_tool(pos);
        }
        fw.cache_valid_snapshot();
        if let Some(snap) = fw.last_valid_snapshot() {
            assert!(
                snap.memory_size() <= MAX_SNAPSHOT_BYTES,
                "Tool {} exceeded 2 MB snapshot ceiling: {} bytes",
                meta.id,
                snap.memory_size()
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 50,
        "Expected >= 50 registered tools, checked {checked}"
    );

    // Export visual verification PNGs for ODE pinned baseline rollback and Temperature static plot
    let ode = populate_ode_with_runs(3, 0, 250);
    let opts = egui_plot::export::ExportOptions::default();
    if let Ok(png) = egui_plot::export::export_custom_png(&ode.export_datasets(), &opts) {
        let _ = std::fs::write("/tmp/ode_pinned_baseline_preserved_rollback.png", png);
    }

    let ctx = egui::Context::default();
    let mut temp = TemperatureAnomaliesTool::default();
    egui_plot::clear_last_plot_datasets();
    let _ = ctx.run(egui::RawInput::default(), |ctx| temp.show(ctx));
    if let Ok(png) =
        egui_plot::export::export_custom_png(&egui_plot::get_last_plot_datasets(), &opts)
    {
        let _ = std::fs::write("/tmp/temperature_anomalies_static_series_plot.png", png);
    }
}
