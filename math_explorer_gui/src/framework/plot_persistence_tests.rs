#![cfg_attr(any(), verified(opt_out = "tests"))]

use std::collections::{BTreeMap, HashMap};

use eframe::egui;
use egui_plot::{
    snapshot::OrderedId, FitType, ImportedDataset, Legend, Line, Plot, PlotBounds, PlotMemory,
    PlotOverlaySnapshot, SavedCallout,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

use crate::framework::{
    clamp::clamp_state_json, Camera3D, InteractiveTool, Plot3DViewport, SimulationFramework,
    ToolMetadata,
};

struct MockPlotTool {
    name: &'static str,
    plot_key: &'static str,
    scale: f64,
}

impl MockPlotTool {
    fn new(name: &'static str, plot_key: &'static str) -> Self {
        Self {
            name,
            plot_key,
            scale: 1.0,
        }
    }
}

impl TheoryDescribable for MockPlotTool {
    fn theory_description(&self) -> String {
        format!("Mock plot tool ({})", self.name)
    }

    fn phonetic_description(&self) -> String {
        "Mock plot tool".to_string()
    }

    fn theory_citation(&self) -> String {
        "Test fixture citation (2026)".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::from([("default".to_string(), "Mock plot tool".to_string())])
    }
}

impl InteractiveTool for MockPlotTool {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn domain(&self) -> &'static str {
        "test_plot_persistence"
    }

    fn state_revision(&self) -> u64 {
        self.scale.to_bits().max(1)
    }

    fn save_state(&self) -> Option<String> {
        Some(serde_json::json!({ "scale": self.scale }).to_string())
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let v: serde_json::Value = serde_json::from_str(state).map_err(|e| e.to_string())?;
        let s = v
            .get("scale")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| "missing scale".to_string())?;
        if !s.is_finite() {
            return Err("non-finite scale".to_string());
        }
        self.scale = s;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let s = self.scale;
        egui::CentralPanel::default().show_inside(ui, |ui| {
            Plot::new(self.plot_key)
                .height(180.0)
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(
                        "Primary Series",
                        vec![[1.0, s], [10.0, 10.0 * s], [100.0, 100.0 * s]],
                    ));
                    plot_ui.line(Line::new(
                        "Secondary Series",
                        vec![[1.0, 2.0 * s], [10.0, 20.0 * s], [100.0, 200.0 * s]],
                    ));
                });
        });
    }
}

fn build_mock_tool_a() -> Box<dyn InteractiveTool> {
    Box::new(MockPlotTool::new("Mock Tool A", "shared_canvas"))
}

fn build_mock_tool_b() -> Box<dyn InteractiveTool> {
    Box::new(MockPlotTool::new("Mock Tool B", "shared_canvas"))
}

static META_A: ToolMetadata = ToolMetadata {
    id: "MockToolA",
    display_name: "Mock Tool A",
    domain: "test_plot_persistence",
    tags: &["mock_a"],
    build: build_mock_tool_a,
};

static META_B: ToolMetadata = ToolMetadata {
    id: "MockToolB",
    display_name: "Mock Tool B",
    domain: "test_plot_persistence",
    tags: &["mock_b"],
    build: build_mock_tool_b,
};

fn build_test_framework() -> SimulationFramework {
    let mut fw = SimulationFramework::new("__empty_test_domain__");
    fw.domain = "test_plot_persistence".to_string();
    fw.available_tools = vec![&META_A, &META_B];
    fw.select_tool(0);
    fw
}

pub(super) fn rich_test_snapshot() -> PlotOverlaySnapshot {
    let mut hidden = std::collections::BTreeSet::new();
    hidden.insert(OrderedId(egui::Id::new("Secondary Series")));
    PlotOverlaySnapshot {
        auto_bounds: Some([false, false]),
        bounds: Some(PlotBounds::from_min_max([1.5, 2.5], [80.0, 160.0])),
        hidden_items: hidden,
        imported_datasets: vec![ImportedDataset::new(
            "Empirical CSV",
            vec![[2.0, 4.0], [20.0, 42.0], [50.0, 98.0]],
        )],
        callouts: vec![SavedCallout::new([10.0, 10.0], "Inflection", [28.0, -22.0])],
        log_x: true,
        log_y: true,
        grid_density: 20.0,
        show_x_labels: false,
        show_y_labels: true,
        clamp_grid: true,
        show_grid: [true, false],
        show_table: true,
        fit_model: Some(FitType::PowerLaw),
        fit_degree: None,
        fit_log_mode: true,
        show_fit_residuals: true,
        fit_dataset_idx: 0,
        show_fit_overlay: true,
        custom_fit_formula: "a * x^b".to_string(),
        custom_fit_seeds_text: "a=1, b=1".to_string(),
    }
}

#[test]
fn test_framework_roundtrip_all_overlays_and_tool_switch_isolation() {
    let mut fw = build_test_framework();
    let expected = rich_test_snapshot();
    fw.saved_plot_states
        .entry("Mock Tool A".to_string())
        .or_default()
        .insert("shared_canvas".to_string(), expected.clone());
    fw.bump_plot_sync_generation();

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| fw.show_inside(ui, "pane_0"));
    });
    let saved_json = fw
        .save_state()
        .expect("serialize framework with plot_states");
    assert!(saved_json.contains("\"plot_states\""));

    fw.select_tool(1);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| fw.show_inside(ui, "pane_0"));
    });
    assert!(!fw.saved_plot_states.contains_key("Mock Tool B"));

    fw.select_tool(0);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| fw.show_inside(ui, "pane_0"));
    });
    assert_eq!(
        fw.saved_plot_states["Mock Tool A"]["shared_canvas"],
        expected
    );

    let mut restored_fw = build_test_framework();
    restored_fw.load_state(&saved_json);
    let fresh_ctx = egui::Context::default();
    let _ = fresh_ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| restored_fw.show_inside(ui, "pane_reloaded"));
    });
    assert_eq!(
        restored_fw.saved_plot_states["Mock Tool A"]["shared_canvas"],
        expected
    );
}

#[test]
fn test_multi_pane_plot_state_isolation() {
    let mut pane_left = build_test_framework();
    let mut pane_right = build_test_framework();
    let left_snap = rich_test_snapshot();
    let mut right_snap = PlotOverlaySnapshot::default();
    right_snap
        .callouts
        .push(SavedCallout::new([5.0, 5.0], "Right Only", [-15.0, 15.0]));

    pane_left
        .saved_plot_states
        .entry("Mock Tool A".to_string())
        .or_default()
        .insert("shared_canvas".to_string(), left_snap.clone());
    pane_left.bump_plot_sync_generation();

    pane_right
        .saved_plot_states
        .entry("Mock Tool A".to_string())
        .or_default()
        .insert("shared_canvas".to_string(), right_snap.clone());
    pane_right.bump_plot_sync_generation();

    let shared_ctx = egui::Context::default();
    let _ = shared_ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.push_id("left_pane", |ui| pane_left.show_inside(ui, "left_pane"));
            ui.push_id("right_pane", |ui| pane_right.show_inside(ui, "right_pane"));
        });
    });

    assert_eq!(
        pane_left.saved_plot_states["Mock Tool A"]["shared_canvas"],
        left_snap
    );
    assert_eq!(
        pane_right.saved_plot_states["Mock Tool A"]["shared_canvas"],
        right_snap
    );
}

#[test]
fn test_preset_starter_and_reset_clear_manual_bounds() {
    let mut fw = build_test_framework();
    fw.saved_plot_states
        .entry("Mock Tool A".to_string())
        .or_default()
        .insert("shared_canvas".to_string(), rich_test_snapshot());
    fw.bump_plot_sync_generation();

    let preset_env = fw
        .active_tool
        .as_ref()
        .unwrap()
        .export_preset_json()
        .expect("export preset");
    fw.apply_preset_json(&preset_env)
        .expect("apply preset resets manual bounds");
    let after_preset = &fw.saved_plot_states["Mock Tool A"]["shared_canvas"];
    assert_eq!(after_preset.bounds, None);
    assert_eq!(after_preset.auto_bounds, None);
    assert_eq!(after_preset.imported_datasets.len(), 1);
    assert_eq!(after_preset.callouts.len(), 1);

    fw.saved_plot_states
        .get_mut("Mock Tool A")
        .unwrap()
        .get_mut("shared_canvas")
        .unwrap()
        .bounds = Some(PlotBounds::from_min_max([2.0, 2.0], [20.0, 20.0]));
    fw.apply_starter_card("Starter", Some("MockToolA"), Some(&preset_env))
        .expect("starter card");
    assert_eq!(
        fw.saved_plot_states["Mock Tool A"]["shared_canvas"].bounds,
        None
    );

    fw.saved_plot_states
        .get_mut("Mock Tool A")
        .unwrap()
        .get_mut("shared_canvas")
        .unwrap()
        .bounds = Some(PlotBounds::from_min_max([3.0, 3.0], [30.0, 30.0]));
    fw.reset_parameters();
    assert_eq!(
        fw.saved_plot_states["Mock Tool A"]["shared_canvas"].bounds,
        None
    );
}

#[test]
fn test_challenge_commit_clears_manual_bounds_and_preserves_overlays() {
    static MOCK_CHALLENGE: crate::starter_gallery::ChallengeMetadata =
        crate::starter_gallery::ChallengeMetadata {
            id: "mock_plot_challenge",
            title: "Mock Plot Challenge",
            description: "Match target scale",
            category: "Test",
            icon: "🎯",
            domain: "test_plot_persistence",
            tool_name: "MockToolA",
            locked_inputs: "none",
            tunable_budget: "scale",
            target_summary: "scale == 2.5",
            baseline_state_generator: || r#"{"scale":2.5}"#.to_string(),
            clamp_locked_state: |json| Ok(json.to_string()),
        };

    let mut fw = build_test_framework();
    fw.saved_plot_states
        .entry("Mock Tool A".to_string())
        .or_default()
        .insert("shared_canvas".to_string(), rich_test_snapshot());
    fw.bump_plot_sync_generation();

    fw.commit_challenge_instance(0, &MOCK_CHALLENGE, r#"{"scale":2.5}"#)
        .expect("challenge commit resets manual bounds");
    let after_challenge = &fw.saved_plot_states["Mock Tool A"]["shared_canvas"];
    assert_eq!(after_challenge.bounds, None);
    assert_eq!(after_challenge.auto_bounds, None);
    assert_eq!(after_challenge.callouts.len(), 1);
}

#[test]
fn test_untrusted_json_rejects_non_finite_and_clamps_fit_degree() {
    let mut fw = build_test_framework();
    let baseline_json = fw.save_state().unwrap();
    let untrusted_degree = serde_json::json!({
        "selected_tool_index": 0,
        "selected_tool_name": "MockToolA",
        "show_theory_portal": false,
        "show_starter_gallery": false,
        "tool_states": {},
        "plot_states": {
            "Mock Tool A": { "shared_canvas": { "fit_degree": 42 } }
        }
    })
    .to_string();
    fw.load_state(&untrusted_degree);
    assert_eq!(
        fw.saved_plot_states["Mock Tool A"]["shared_canvas"].fit_model,
        Some(FitType::Quartic)
    );

    fw.load_state(&baseline_json);
    let corrupted_bounds = serde_json::json!({
        "selected_tool_index": 0,
        "selected_tool_name": "MockToolA",
        "show_theory_portal": false,
        "show_starter_gallery": false,
        "tool_states": {},
        "plot_states": {
            "Mock Tool A": {
                "shared_canvas": {
                    "bounds": { "min": [50.0, 0.0], "max": [10.0, 10.0] },
                    "auto_bounds": [false, false]
                }
            }
        }
    })
    .to_string();
    fw.load_state(&corrupted_bounds);
    assert!(fw.saved_plot_states.is_empty());
}

#[test]
fn test_clamp_state_json_preserves_plot_states_and_fixed_callout_offsets() {
    let pts: Vec<[f64; 2]> = (0..900)
        .map(|i| [i as f64 * 10.0, i as f64 * 25.0])
        .collect();
    let mut snap = PlotOverlaySnapshot::default();
    snap.imported_datasets
        .push(ImportedDataset::new("HighCoordCSV", pts));
    snap.bounds = Some(PlotBounds::from_min_max([100.0, 200.0], [9000.0, 22500.0]));
    snap.auto_bounds = Some([false, false]);

    let plot_states = BTreeMap::from([(
        "Tool".to_string(),
        BTreeMap::from([("canvas".to_string(), snap.clone())]),
    )]);
    let baseline = serde_json::json!({
        "dt": 0.01,
        "fixed_callout_offsets": [[35.0, -25.0], [-35.0, 25.0]],
        "plot_states": {}
    })
    .to_string();
    let diverged = serde_json::json!({
        "dt": 0.5,
        "fixed_callout_offsets": [[180.0, -140.0], [-210.0, 165.0]],
        "plot_states": plot_states
    })
    .to_string();

    let constraints = HashMap::from([(
        "dt".to_string(),
        ParameterConstraint {
            min: 0.001,
            max: 0.05,
            step: 0.001,
        },
    )]);
    let clamped = clamp_state_json(&diverged, Some(&baseline), &constraints)
        .expect("dt must be clamped while UI_ONLY_KEYS remain untouched");
    let parsed: serde_json::Value = serde_json::from_str(&clamped).unwrap();
    assert_eq!(parsed["dt"].as_f64(), Some(0.01));
    assert_eq!(
        parsed["fixed_callout_offsets"],
        serde_json::json!([[180.0, -140.0], [-210.0, 165.0]])
    );
    let restored_snap: PlotOverlaySnapshot =
        serde_json::from_value(parsed["plot_states"]["Tool"]["canvas"].clone()).unwrap();
    assert_eq!(restored_snap.bounds, snap.bounds);
    assert_eq!(restored_snap.imported_datasets[0].points.len(), 900);
}

#[test]
fn test_plot_3d_viewport_locks_bounds_against_injected_snapshots() {
    let seed = BTreeMap::from([(
        "viewport_3d".to_string(),
        PlotOverlaySnapshot {
            auto_bounds: Some([false, false]),
            bounds: Some(PlotBounds::from_min_max([900.0, 900.0], [950.0, 950.0])),
            show_table: true,
            ..Default::default()
        },
    )]);
    let ctx = egui::Context::default();
    let mut cam = Camera3D::default();
    let mut plot_id = egui::Id::NULL;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui_plot::begin_plot_persistence_scope(ctx, 77, 1, &seed);
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot3DViewport::new("viewport_3d", 32.0).show(ui, &mut cam, |plot_ui| {
                plot_id = plot_ui.plot_id();
            });
        });
        let out = egui_plot::end_plot_persistence_scope(ctx, 77);
        let snap = out.get("viewport_3d").expect("show_table retained");
        assert_eq!(snap.bounds, None);
        assert!(snap.show_table);
    });
    let mem = PlotMemory::load(&ctx, plot_id).expect("3D PlotMemory");
    assert!(mem.bounds().min()[0] <= -32.0 + 1e-3);
    assert!(mem.bounds().max()[1] >= 32.0 - 1e-3);
}
