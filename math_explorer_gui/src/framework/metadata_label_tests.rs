#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{SimulationFramework, ToolMetadata};
use crate::async_sim::unified::{UnifiedModel, UnifiedSimTool};
use crate::tabs::fluid_dynamics::lattice_boltzmann::LbmUnified;
use crate::tabs::neuroscience::hodgkin_huxley::HodgkinHuxleyUnified;
use crate::tabs::quantum::wave_sim::{
    wave_parameter_constraints, wave_parameter_label, WavePotentialPreset, WaveSimulatorTool,
};
use crate::tabs::solid_state::crystal_viewer::CrystalSystemEnum;
use eframe::egui;
use scientific_metadata::theory::TheoryDescribable;

fn has_raw_struct_suffix(label: &str) -> bool {
    for suffix in ["Tool", "Widget", "Plotter", "Viewer", "Simulator"] {
        if let Some(prefix) = label.strip_suffix(suffix) {
            if let Some(last_char) = prefix.chars().last() {
                if !last_char.is_whitespace() {
                    return true;
                }
            }
        }
    }
    false
}

fn collect_accesskit_labels(output: &egui::FullOutput) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(u) = output.platform_output.accesskit_update.as_ref() {
        for (_, n) in &u.nodes {
            if let Some(lbl) = n.label() {
                out.push(lbl.to_string());
            }
            if let Some(val) = n.value() {
                out.push(val.to_string());
            }
        }
    }
    out
}

fn compact_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(960.0, 640.0),
        )),
        ..Default::default()
    }
}

#[test]
fn test_all_registered_tools_separate_id_from_plain_display_name() {
    let tools: Vec<&'static ToolMetadata> = inventory::iter::<ToolMetadata>.into_iter().collect();
    assert!(
        tools.len() >= 52,
        "Expected at least 52 registered tools, found {}",
        tools.len()
    );

    for meta in tools {
        assert!(
            !meta.id.trim().is_empty(),
            "ToolMetadata.id must not be empty"
        );
        assert!(
            !meta.display_name.trim().is_empty(),
            "ToolMetadata.display_name must not be empty for id '{}'",
            meta.id
        );
        assert!(
            !has_raw_struct_suffix(meta.display_name),
            "ToolMetadata.display_name '{}' for id '{}' leaked a raw struct suffix",
            meta.display_name,
            meta.id
        );
        let built = (meta.build)();
        assert_eq!(
            meta.display_name,
            built.name(),
            "ToolMetadata.display_name must match InteractiveTool::name() for id '{}'",
            meta.id
        );
    }
}

#[test]
fn test_sidebar_and_launchpad_render_display_names_without_struct_ids() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    for (domain, expected_label, forbidden_id) in [
        (
            "clinical_trials",
            "Sample Size Calculator",
            "SampleSizeCalculatorTool",
        ),
        ("number_theory", "Partition Function", "PartitionsWidget"),
        ("chaos", "Attractor Plotter", "AttractorPlotter"),
        ("quantum", "Wave Simulator", "WaveSimulator"),
        ("solid_state", "Crystal Lattice Viewer", "CrystalViewer"),
    ] {
        let mut framework = SimulationFramework::new(domain);
        let out = ctx.run(egui::RawInput::default(), |ctx| {
            framework.show(ctx, domain);
        });
        let labels = collect_accesskit_labels(&out);
        assert!(
            labels.iter().any(|l| l.contains(expected_label)),
            "Expected sidebar in '{domain}' to render '{expected_label}', got {labels:?}"
        );
        assert!(
            !labels.iter().any(|l| l.contains(forbidden_id)),
            "Sidebar in '{domain}' leaked raw ID '{forbidden_id}': {labels:?}"
        );

        framework.active_tool = None;
        framework.selected_tool_index = None;
        let launchpad_out = ctx.run(egui::RawInput::default(), |ctx| {
            framework.show(ctx, domain);
        });
        let launch_labels = collect_accesskit_labels(&launchpad_out);
        assert!(
            launch_labels.iter().any(|l| l.contains(expected_label)),
            "Expected launchpad in '{domain}' to render '{expected_label}'"
        );
        assert!(
            !launch_labels.iter().any(|l| l.contains(forbidden_id)),
            "Launchpad in '{domain}' leaked raw ID '{forbidden_id}'"
        );
    }
}

#[test]
fn test_legacy_id_and_display_name_workspace_deserialization() {
    let mut framework = SimulationFramework::new("clinical_trials");
    assert!(framework.select_tool_by_name("SampleSizeCalculatorTool"));
    assert_eq!(
        framework.active_tool.as_ref().map(|t| t.name()),
        Some("Sample Size Calculator")
    );

    let saved = framework.save_state().expect("must serialize state");
    assert!(
        saved.contains("\"SampleSizeCalculatorTool\""),
        "Persisted selected_tool_name should use stable ToolMetadata.id: {saved}"
    );

    let legacy_json = r#"{
        "selected_tool_index": 0,
        "selected_tool_name": "SampleSizeCalculatorTool",
        "show_theory_portal": false,
        "show_starter_gallery": false,
        "tool_states": {}
    }"#;
    let mut restored_legacy = SimulationFramework::new("clinical_trials");
    restored_legacy.select_tool(0);
    restored_legacy.load_state(legacy_json);
    assert_eq!(
        restored_legacy.active_tool.as_ref().map(|t| t.name()),
        Some("Sample Size Calculator")
    );

    let display_json = r#"{
        "selected_tool_index": 0,
        "selected_tool_name": "Sample Size Calculator",
        "show_theory_portal": false,
        "show_starter_gallery": false,
        "tool_states": {}
    }"#;
    let mut restored_display = SimulationFramework::new("clinical_trials");
    restored_display.select_tool(0);
    restored_display.load_state(display_json);
    assert_eq!(
        restored_display.active_tool.as_ref().map(|t| t.name()),
        Some("Sample Size Calculator")
    );
}

#[test]
fn test_theory_citations_and_portal_have_no_raw_cite_tags_or_default_keys() {
    use math_explorer::physics::chaos::lorenz::{LorenzState, LorenzSystem};
    let lorenz = LorenzSystem::default_chaotic(LorenzState::new(1.0, 1.0, 1.0));
    assert!(!lorenz.theory_citation().contains("[cite:"));

    let qstate = math_explorer::physics::quantum::QuantumState::new(
        nalgebra::DVector::from_element(2, num_complex::Complex::new(1.0, 0.0)),
    );
    assert!(!qstate.theory_citation().contains("[cite:"));

    for meta in inventory::iter::<ToolMetadata> {
        let tool = (meta.build)();
        let desc = tool.theory().theory_description();
        let cite = tool.theory().theory_citation();
        assert!(
            !desc.contains("[cite:"),
            "Tool '{}' leaked [cite:...] in theory_description: {desc}",
            meta.id
        );
        assert!(
            !cite.contains("[cite:"),
            "Tool '{}' leaked [cite:...] in theory_citation: {cite}",
            meta.id
        );
    }

    for (domain, tool_id) in [
        ("chaos", "BifurcationDiagram"),
        ("chaos", "FractalViewer"),
        ("quantum", "ClebschGordanTool"),
        ("quantum", "WaveSimulator"),
        ("neuroscience", "HodgkinHuxleyTool"),
    ] {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut framework = SimulationFramework::new(domain);
        assert!(framework.select_tool_by_name(tool_id));
        framework.show_theory_portal = true;
        ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), true));
        let small_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(360.0, 240.0),
            )),
            ..Default::default()
        };
        let out = ctx.run(small_input, |ctx| {
            framework.show(ctx, domain);
        });
        for label in collect_accesskit_labels(&out) {
            assert!(
                !label.contains("[cite:"),
                "Portal leaked [cite:...]: {label}"
            );
            assert!(
                !label.starts_with("default:") && !label.contains("for default"),
                "Portal leaked raw 'default' key: {label}"
            );
        }
    }
}

#[test]
fn test_unified_models_expose_readable_parameter_labels_and_discrete_choices() {
    for key in wave_parameter_constraints().keys() {
        let label = wave_parameter_label(key);
        assert!(
            !label.contains('_'),
            "WaveSimulator label '{label}' has underscore"
        );
    }
    for key in HodgkinHuxleyUnified::parameters().keys() {
        let label = HodgkinHuxleyUnified::parameter_label(key);
        assert!(
            !label.contains('_'),
            "HodgkinHuxleyUnified label '{label}' has underscore"
        );
    }
    for key in LbmUnified::parameters().keys() {
        let label = LbmUnified::parameter_label(key);
        assert!(
            !label.contains('_'),
            "LbmUnified label '{label}' has underscore"
        );
    }

    assert_eq!(WavePotentialPreset::ALL.len(), 5);
    assert_eq!(
        WavePotentialPreset::ALL.map(WavePotentialPreset::label),
        [
            "Infinite Potential Well",
            "Harmonic Oscillator",
            "Rectangular Barrier",
            "Double Barrier Resonance",
            "Custom Formula",
        ]
    );

    let wave_ctx = egui::Context::default();
    wave_ctx.enable_accesskit();
    let mut wave_tool = WaveSimulatorTool::new();
    let wave_out = wave_ctx.run(compact_input(), |ctx| {
        use crate::framework::InteractiveTool;
        wave_tool.show(ctx);
    });
    let wave_labels = collect_accesskit_labels(&wave_out);
    assert!(
        wave_labels.iter().any(|l| l.contains("Potential Profile")),
        "wave_labels: {wave_labels:?}"
    );
    assert!(
        wave_labels
            .iter()
            .any(|l| l.contains("Infinite Potential Well")),
        "wave_labels: {wave_labels:?}"
    );
    assert!(!wave_labels.iter().any(|l| l == "potential_type"));
}

#[test]
fn test_hodgkin_huxley_sensitivity_cards_separate_ids_from_display_labels() {
    let hh_ctx = egui::Context::default();
    hh_ctx.enable_accesskit();
    let mut hh_tool = UnifiedSimTool::<HodgkinHuxleyUnified>::new();
    let _ = hh_ctx.run(compact_input(), |ctx| {
        use crate::framework::InteractiveTool;
        hh_tool.show(ctx);
    });
    let hh_out = hh_ctx.run(compact_input(), |ctx| {
        use crate::framework::InteractiveTool;
        hh_tool.show(ctx);
    });
    let hh_labels = collect_accesskit_labels(&hh_out);
    for expected_label in [
        "Baseline Peak Membrane Potential (mV):",
        "Sodium Conductance (mS/cm²)",
        "Potassium Conductance (mS/cm²)",
        "Leak Conductance (mS/cm²)",
        "External Stimulus Current (µA/cm²)",
    ] {
        assert!(
            hh_labels.iter().any(|l| l.contains(expected_label)),
            "Expected '{expected_label}' in hh_labels: {hh_labels:?}"
        );
    }
    assert!(
        !hh_labels
            .iter()
            .any(|l| matches!(l.as_str(), "g_na" | "g_k" | "g_l" | "i_ext")),
        "Sensitivity or slider controls leaked raw snake_case keys: {hh_labels:?}"
    );
    assert!(
        hh_labels
            .iter()
            .any(|l| l.starts_with("dY/dP:") && l != "dY/dP: 0.0000"),
        "Expected non-zero biophysical sensitivity derivatives: {hh_labels:?}"
    );
}

#[test]
fn test_readable_enum_and_error_formatting() {
    assert_eq!(CrystalSystemEnum::SimpleCubic.label(), "Simple Cubic");
    assert_eq!(
        CrystalSystemEnum::BodyCenteredCubic.label(),
        "Body-Centered Cubic (BCC)"
    );
    assert_eq!(
        CrystalSystemEnum::FaceCenteredCubic.label(),
        "Face-Centered Cubic (FCC)"
    );

    use math_explorer::pure_math::statistics::tda::{
        core::{Point2D, PointCloud},
        vietoris_rips_complex,
    };
    let err = PointCloud::new(Vec::new()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("point cloud cannot be empty"),
        "got: {msg}"
    );
    assert!(!msg.contains("EmptyPointCloud"));

    let cloud = PointCloud::new(vec![Point2D::new(0.0, 0.0)]).unwrap();
    let rad_err = vietoris_rips_complex(&cloud, -1.0).unwrap_err();
    let rad_msg = rad_err.to_string();
    assert!(
        rad_msg
            .to_lowercase()
            .contains("filtration radius must be non-negative"),
        "got: {rad_msg}"
    );
    assert!(!rad_msg.contains('{'));
}
