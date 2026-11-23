#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:advanced_linear_algebra]
// [cite:graph_parameters_rust]

use eframe::egui::{self, vec2, Context, Event, Key, Modifiers, Pos2, RawInput, Rect};
use math_explorer::pure_math::analysis::pde::kdv::kdv_two_soliton_phase_shifts;

use super::*;
use crate::app::MathExplorerApp;
use crate::framework::{ExportFormat, SimulationFramework};
use crate::starter_gallery::StarterPresetMetadata;

fn raw_frame() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

fn raw_key(key: Key) -> RawInput {
    let mut raw = raw_frame();
    raw.events.push(Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    raw
}

#[test]
fn test_soliton_state_bounds_validation_and_finite_guard_rollback() {
    let mut tool = SolitonCollisionTool::default();
    assert!(tool.has_finite_parameters());

    let valid_json = tool.save_state().expect("save_state must succeed");
    assert!(tool.state_bounds().validate_json(&valid_json).is_ok());
    assert!(tool.try_load_state(&valid_json).is_ok());

    let mut unknown: serde_json::Value = serde_json::from_str(&valid_json).unwrap();
    unknown["extra_unknown"] = serde_json::json!(true);
    assert!(tool.try_load_state(&unknown.to_string()).is_err());

    let huge = format!("{{\"pad\":\"{}\"}}", "x".repeat(16_384));
    assert!(tool.try_load_state(&huge).is_err());

    let mut out_of_range: serde_json::Value = serde_json::from_str(&valid_json).unwrap();
    out_of_range["fast_speed_c1"] = serde_json::json!(999.0);
    assert!(tool.try_load_state(&out_of_range.to_string()).is_err());

    tool.state.fast_speed_c1 = f64::NAN;
    tool.state.time_step = 99.0;
    assert!(!tool.has_finite_parameters());
    assert!(tool.clamp_parameters());
    assert!(tool.has_finite_parameters());
    assert!(tool.state.time_step <= DT_MAX);
}

#[test]
fn test_soliton_two_soliton_collision_and_airy_toggle_metrics() {
    let mut tool = SolitonCollisionTool::default();
    tool.apply_preset(SolitonPreset::TwoSolitonOvertaking);
    for _ in 0..140 {
        tool.step_simulation();
    }
    let drift = tool
        .current_invariants
        .relative_drift(tool.initial_invariants);
    assert!(drift[0] < 1.0e-5 && drift[1] < 1.0e-5 && drift[2] < 1.0e-5);
    let exact = kdv_two_soliton_phase_shifts(16.0, 4.0);
    assert!((tool.measured_shifts.0 - exact.delta_fast).abs() < 5.0e-3);
    assert!((tool.measured_shifts.1 - exact.delta_slow).abs() < 5.0e-3);

    let mut airy = SolitonCollisionTool::default();
    airy.apply_preset(SolitonPreset::LinearAiryCollapse);
    for _ in 0..140 {
        airy.step_simulation();
    }
    let airy_ratio = airy.top_peaks.0.amplitude / airy.initial_peak;
    assert!(1.0 - airy_ratio > 0.60);
}

#[test]
fn test_soliton_url_hash_hydration_roundtrip() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut source_app = MathExplorerApp::default();
    let analysis_idx = source_app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "analysis")
        .expect("AnalysisTab must exist");
    source_app.selected_tab = analysis_idx;
    assert!(source_app.tabs[analysis_idx].select_tool_by_name("SolitonCollisionTool"));

    if let Some(fw) = source_app.tabs[analysis_idx].framework_mut() {
        let tool = fw
            .active_tool
            .as_mut()
            .unwrap()
            .as_any_mut()
            .unwrap()
            .downcast_mut::<SolitonCollisionTool>()
            .unwrap();
        tool.apply_preset(SolitonPreset::CubicNlsEnvelopeCollision);
        tool.state.nls_coupling = 1.35;
        tool.state_rev = tool.state_rev.wrapping_add(1);
    }
    let expected_state = source_app.tabs[analysis_idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state();

    let share_url = source_app.encode_current_share_url().unwrap();
    let hash_pos = share_url.find('#').expect("URL must contain hash");
    let hash = &share_url[hash_pos..];

    let mut restored_app = MathExplorerApp::default();
    assert!(restored_app
        .apply_share_location_strings(hash, "", None)
        .unwrap());
    assert_eq!(
        restored_app.tabs[analysis_idx].active_tool_id(),
        Some("SolitonCollisionTool")
    );
    let actual_state = restored_app.tabs[analysis_idx]
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state();
    assert_eq!(actual_state, expected_state);
}

#[test]
fn test_soliton_accesskit_focus_arrow_navigation_enter_inject_and_tab_exit() {
    let mut tool = SolitonCollisionTool {
        paused: true,
        ..Default::default()
    };
    let ctx = Context::default();
    ctx.enable_accesskit();

    let _ = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let out = ctx.run(raw_frame(), |ctx| tool.show(ctx));
    let ak = out.platform_output.accesskit_update.expect("AccessKit");
    let canvas_nodes: Vec<egui::accesskit::NodeId> = ak
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            (node.role() == egui::accesskit::Role::GraphicsDocument
                && node
                    .label()
                    .unwrap_or_default()
                    .contains("1D KdV Soliton Space-Time Waterfall Visualization")
                && node.supports_action(egui::accesskit::Action::Focus))
            .then_some(*id)
        })
        .collect();
    assert!(!canvas_nodes.is_empty());

    let mut focused = false;
    for _ in 0..80 {
        let step_out = ctx.run(raw_key(Key::Tab), |ctx| tool.show(ctx));
        if step_out
            .platform_output
            .accesskit_update
            .is_some_and(|upd| canvas_nodes.contains(&upd.focus))
        {
            focused = true;
            break;
        }
    }
    assert!(focused, "Waterfall canvas must receive Tab focus");

    let x_before = tool.state.pulse_x0;
    let m_before = tool.current_invariants.mass;
    let _ = ctx.run(raw_key(Key::ArrowRight), |ctx| tool.show(ctx));
    assert!(tool.state.pulse_x0 > x_before);
    let _ = ctx.run(raw_key(Key::Enter), |ctx| tool.show(ctx));
    assert!(tool.current_invariants.mass > m_before + 0.5);

    let next_tab = ctx.run(raw_key(Key::Tab), |ctx| tool.show(ctx));
    if let Some(upd) = next_tab.platform_output.accesskit_update {
        assert!(!canvas_nodes.contains(&upd.focus));
    }
}

#[test]
fn test_soliton_all_four_starter_presets_export_svg_png_csv_json() {
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| p.tool_name == Some("SolitonCollisionTool"))
        .collect();
    assert_eq!(presets.len(), 4);

    let temp_dir =
        std::env::temp_dir().join(format!("oxidize_soliton_exports_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);
    let ctx = Context::default();

    for preset in presets {
        let mut framework = SimulationFramework::new("analysis");
        framework.export_queue.destination_dir = Some(temp_dir.clone());
        framework
            .apply_starter_card(
                preset.title,
                preset.tool_name,
                Some(preset.generate_preset()),
            )
            .expect("apply_starter_card must succeed");
        let _ = ctx.run(raw_frame(), |ctx| framework.show(ctx, "analysis"));
        for fmt in [
            ExportFormat::Svg,
            ExportFormat::Png,
            ExportFormat::Csv,
            ExportFormat::Json,
        ] {
            framework.queue_export(fmt);
            let _ = ctx.run(raw_frame(), |ctx| framework.show(ctx, "analysis"));
            let filename = framework
                .export_queue
                .last_exported_filename
                .as_ref()
                .expect("Export filename recorded");
            let meta = std::fs::metadata(temp_dir.join(filename)).unwrap();
            assert!(meta.len() > 64);
        }
    }
    let _ = std::fs::remove_dir_all(&temp_dir);
}
