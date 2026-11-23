#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::state::{StudioBedMode, StudioFlowElement, StudioFlowElementKind};
use super::*;
use crate::framework::{ExportFormat, InteractiveTool, SimulationFramework};
use crate::url_state::{parse_share_location, MAX_DECODED_URL_STATE_BYTES};
use eframe::egui;

fn assert_fabrication_svg_contract(svg: &str, expected_w_mm: &str, expected_h_mm: &str) {
    assert!(
        svg.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg "),
        "SVG must start with XML declaration and <svg> root"
    );
    let expected_header = format!(
        "width=\"{expected_w_mm}mm\" height=\"{expected_h_mm}mm\" viewBox=\"0 0 {expected_w_mm} {expected_h_mm}\""
    );
    assert!(
        svg.contains(&expected_header),
        "Expected physical mm header '{expected_header}' in:\n{}",
        svg.lines().take(4).collect::<Vec<_>>().join("\n")
    );
    assert!(svg.contains("<g id=\"layer-cut\""));
    assert!(svg.contains("<g id=\"layer-score\""));
    assert!(svg.contains("<path fill=\"none\""));
    assert!(!svg.contains("<image"), "Fabrication SVG must contain zero <image> tags");
    assert!(!svg.contains("<rect"), "Fabrication SVG must contain zero <rect> backgrounds");
    assert!(svg.trim_end().ends_with("</svg>"));
}

#[test]
fn test_all_three_sources_emit_layered_mm_svg_without_rect_or_image() {
    let mut tool = FabricationStudioTool::default();

    // 1. Aperiodic tiling (Einstein Hat on A4)
    tool.import_preset_json(PRESET_FABRICATION_EINSTEIN_HAT)
        .expect("valid Einstein Hat preset");
    let hat_svg = tool.serialize_fabrication_svg().expect("Hat SVG");
    assert_fabrication_svg_contract(&hat_svg, "210", "297");

    // Verify Spectre & Penrose P2/P3 also serialize cleanly
    for kind in [
        FabricationTilingSelector::EinsteinSpectre,
        FabricationTilingSelector::PenroseP2,
        FabricationTilingSelector::PenroseP3,
    ] {
        tool.state.tiling_kind = kind;
        tool.state.tiling_depth = 2;
        tool.mark_dirty();
        let svg = tool.serialize_fabrication_svg().expect("tiling SVG");
        assert_fabrication_svg_contract(&svg, "210", "297");
    }

    // 2. Turing coaster on A4
    tool.import_preset_json(PRESET_FABRICATION_TURING_COASTER)
        .expect("valid Turing preset");
    let turing_svg = tool.serialize_fabrication_svg().expect("Turing SVG");
    assert_fabrication_svg_contract(&turing_svg, "210", "297");

    // 3. Potential flow streamlines on US Letter and Custom bed
    tool.import_preset_json(PRESET_FABRICATION_FLOW_PLOTTER)
        .expect("valid PotentialFlow preset");
    let letter_svg = tool.serialize_fabrication_svg().expect("Flow SVG");
    assert_fabrication_svg_contract(&letter_svg, "215.9", "279.4");

    tool.state.bed_mode = StudioBedMode::Custom;
    tool.state.custom_width_mm = 300.0;
    tool.state.custom_height_mm = 180.0;
    tool.mark_dirty();
    let custom_svg = tool.serialize_fabrication_svg().expect("Custom bed SVG");
    assert_fabrication_svg_contract(&custom_svg, "300", "180");
}

#[test]
fn test_nearest_neighbor_optimization_reduces_pen_lift_travel_in_studio() {
    let mut tool = FabricationStudioTool::default();
    tool.import_preset_json(PRESET_FABRICATION_TURING_COASTER)
        .unwrap();
    let art = tool.artifact_or_build().expect("artifact built");
    assert!(
        art.active_travel_mm <= art.raw_travel_mm,
        "Optimized travel ({}) must not exceed raw travel ({})",
        art.active_travel_mm,
        art.raw_travel_mm
    );
}

fn sample_custom_studio_state() -> FabricationStudioState {
    FabricationStudioState {
        source: FabricationSourceKind::TuringField,
        bed_mode: StudioBedMode::Custom,
        custom_width_mm: 450.0,
        custom_height_mm: 320.0,
        margin_mm: 14.5,
        optimize_travel: false,
        show_pen_travel: true,
        turing_a: 0.13,
        turing_b: 0.85,
        turing_d_u: 1.15,
        turing_d_v: 52.0,
        turing_dt: 0.035,
        turing_grid_size: 40,
        turing_steps: 90,
        turing_score_levels: 5,
        turing_coaster_border: false,
        flow_elements: vec![
            StudioFlowElement {
                kind: StudioFlowElementKind::Uniform,
                strength: 12.5,
                x: 0.0,
                y: 0.0,
                angle_deg: 15.0,
            },
            StudioFlowElement {
                kind: StudioFlowElementKind::Vortex,
                strength: -24.0,
                x: 1.5,
                y: -2.0,
                angle_deg: 0.0,
            },
        ],
        flow_view_range: 12.0,
        flow_streamline_count: 20,
        tiling_kind: FabricationTilingSelector::EinsteinSpectre,
        tiling_depth: 3,
    }
}

#[test]
fn test_url_share_roundtrip_restores_all_fabrication_studio_parameters() {
    let mut fw = SimulationFramework::new("geometry_topology");
    assert!(fw.select_tool_by_name("FabricationStudioTool"));

    let custom_state = sample_custom_studio_state();
    let expected_json = serde_json::to_string(&custom_state).unwrap();
    fw.active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&expected_json)
        .expect("custom studio state valid");

    let share_url = fw.encode_share_url().expect("share URL encoded");
    let hash_idx = share_url.find('#').expect("fragment present");
    let parsed = parse_share_location(&share_url[hash_idx..], "")
        .expect("valid share URL")
        .expect("share params present");
    assert_eq!(parsed.tab, "geometry_topology");
    assert_eq!(parsed.tool.as_deref(), Some("FabricationStudioTool"));
    let preset_json = parsed.state_json.expect("decoded preset json");
    assert!(preset_json.len() <= MAX_DECODED_URL_STATE_BYTES);

    let mut restored_fw = SimulationFramework::new("geometry_topology");
    restored_fw
        .import_url_preset_json(parsed.tool.as_deref(), Some(&preset_json))
        .expect("import_url_preset_json succeeded");
    assert_eq!(restored_fw.active_tool_id(), Some("FabricationStudioTool"));
    let actual_json = restored_fw
        .active_tool
        .as_ref()
        .unwrap()
        .save_state()
        .expect("restored state");
    let restored_state: FabricationStudioState = serde_json::from_str(&actual_json).unwrap();
    assert_eq!(restored_state, custom_state);
}

fn export_framework_svg(fw: &mut SimulationFramework, id: &str) -> String {
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        fw.show_export_modal = true;
        fw.queue_export(ExportFormat::Svg);
        fw.show(ctx, id);
    });
    fw.export_queue
        .last_exported_content
        .clone()
        .expect("SVG exported")
}

#[test]
fn test_simulation_framework_svg_export_across_fabrication_morphogenesis_and_flow() {
    let mut fab_fw = SimulationFramework::new("geometry_topology");
    assert!(fab_fw.select_tool_by_name("FabricationStudioTool"));
    let fab_svg = export_framework_svg(&mut fab_fw, "fab_svg_test");
    assert_fabrication_svg_contract(&fab_svg, "210", "297");

    #[cfg(feature = "biology")]
    {
        let mut morpho_tool = crate::tabs::morphogenesis::MorphogenesisTool::default();
        morpho_tool.step_simulation();
        let mut morpho_fw = SimulationFramework::new("__morpho_svg_test__");
        morpho_fw.active_tool = Some(Box::new(morpho_tool));
        let morpho_svg = export_framework_svg(&mut morpho_fw, "morpho_svg_test");
        assert!(
            morpho_svg.contains("<polyline") || morpho_svg.contains("<polygon"),
            "MorphogenesisTool SVG must emit vector contours"
        );
        assert!(
            !morpho_svg.contains("<image"),
            "MorphogenesisTool SVG must omit Base64 raster <image>"
        );
    }

    #[cfg(feature = "physics")]
    {
        let mut flow_fw = SimulationFramework::new("fluid_dynamics");
        assert!(flow_fw.select_tool_by_name("PotentialFlowTool"));
        let flow_svg = export_framework_svg(&mut flow_fw, "flow_svg_test");
        assert!(flow_svg.contains("data-name=\"Streamline 1\""));
        assert!(flow_svg.contains("data-name=\"Vector Field\""));
    }
}
