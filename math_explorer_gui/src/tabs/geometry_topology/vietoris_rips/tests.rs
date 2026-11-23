//! Unit and integration tests for the Vietoris-Rips TDA workbench.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::io::{
    format_intervals_csv, format_intervals_json, parse_point_cloud_text, IntervalExportRow,
    MAX_IMPORT_BYTES, MAX_VIETORIS_RIPS_POINTS,
};
use super::plots::PersistencePlotMode;
use super::presets::{ShapePreset, DEFAULT_PRESET_POINTS};
use super::*;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::url_state::{UrlStateSignature, UrlSyncController, MAX_DECODED_URL_STATE_BYTES};
use egui_plot::export::{color32_image_to_png, ExportOptions};
use egui_plot::export_primitives::{export_primitives_svg, render_primitives_image};
use egui_plot::PlotPoints;
use std::f64::consts::{E, PI};

#[test]
fn test_vietoris_rips_default_initialization_and_presets() {
    let mut tool = VietorisRipsTool::default();
    assert_eq!(tool.cached_vertices.len(), DEFAULT_PRESET_POINTS);
    assert_eq!(tool.cached_edges.len(), tool.complex.count_simplices(1));
    assert_eq!(tool.cached_triangles.len(), tool.complex.count_simplices(2));
    assert!(tool.error_msg.is_none());
    assert!(!tool.exceeds_url_bounds);
    assert!(tool.betti.1 >= 1);

    for preset in ShapePreset::ALL {
        tool.select_shape_preset(preset);
        assert!(tool.has_finite_parameters());
        assert_eq!(tool.cached_vertices.len(), DEFAULT_PRESET_POINTS);
    }
}

#[test]
fn test_vietoris_rips_parameter_updates_and_error_clearing() {
    let mut tool = VietorisRipsTool::default();
    tool.set_radius(15.0);
    assert!(!tool.cached_edges.is_empty());
    assert!(!tool.cached_triangles.is_empty());

    {
        let v_slice = &tool.cached_vertices[..];
        let v_plot: PlotPoints<'_> = v_slice.into();
        assert!(matches!(v_plot, PlotPoints::Borrowed(_)));
    }

    tool.set_radius(0.0);
    assert_eq!(tool.cached_edges.len(), 0);
    assert_eq!(tool.cached_triangles.len(), 0);

    tool.set_radius(-1.0);
    assert!(tool.error_msg.is_some());
    assert!(tool.cached_vertices.is_empty());
}

#[test]
fn test_csv_tsv_json_import_bounds_bom_and_state_preservation() {
    let mut tool = VietorisRipsTool::default();
    let baseline_state = tool.save_state().unwrap();
    let builds_before = tool.complex_builds;

    let oversized = "x,y\n".to_string() + &"1.0,2.0\n".repeat(MAX_IMPORT_BYTES / 6);
    assert!(oversized.len() > MAX_IMPORT_BYTES);
    assert!(tool
        .import_point_cloud_text("huge.csv", &oversized)
        .is_err());

    let mut over_rows = String::from("x,y\n");
    for i in 0..=MAX_VIETORIS_RIPS_POINTS {
        over_rows.push_str(&format!("{i}.0,1.0\n"));
    }
    assert!(tool
        .import_point_cloud_text("many.csv", &over_rows)
        .is_err());

    assert!(tool
        .import_point_cloud_text("bad_hdr.csv", "x,1.0\n2.0,3.0\n")
        .is_err());
    assert!(tool
        .import_point_cloud_text("jagged.csv", "x,y,z\n1,2,3\n4,5\n")
        .is_err());
    assert!(tool
        .import_point_cloud_text("nan.json", "[[0.0, 1.0], [2.0, null]]")
        .is_err());
    let wide_json = serde_json::json!([vec![1.0_f64; 33], vec![2.0_f64; 33]]).to_string();
    assert!(tool
        .import_point_cloud_text("wide.json", &wide_json)
        .is_err());

    assert_eq!(tool.save_state().as_deref(), Some(baseline_state.as_str()));
    assert_eq!(tool.complex_builds, builds_before);

    let tsv_4d = "\u{feff}w\tx\ty\tz\n0.0\t0.0\t0.0\t5.0\n0.0\t1.0\t0.0\t5.0\n0.0\t1.0\t1.0\t5.0\n0.0\t0.0\t1.0\t5.0\n";
    tool.import_point_cloud_text("square4d.tsv", tsv_4d)
        .unwrap();
    assert_eq!(tool.ambient_dimension(), 4);
    tool.set_radius(1.1);
    assert_eq!(tool.betti, (1, 1));
    tool.set_projection_axes(1, 2);
    assert_eq!(tool.projection_axes(), (1, 2));
    assert!(parse_point_cloud_text("pts.json", r#"{"points":[[0,0],[1,0],[0,1]]}"#).is_ok());
}

#[test]
fn test_url_4kib_budget_for_generator_and_quantized_64_points() {
    let mut fw = SimulationFramework::new("geometry_topology");
    assert!(fw.select_tool_by_name("VietorisRipsTool"));

    let vr_state = serde_json::json!({
        "radius": 1.4,
        "generator": { "preset": "two_circles", "count": 250, "seed": 7, "noise": 0.05 }
    })
    .to_string();
    fw.active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&vr_state)
        .unwrap();
    let gen_preset = fw.export_url_preset_json().unwrap().unwrap();
    assert!(gen_preset.len() <= MAX_DECODED_URL_STATE_BYTES);
    assert!(fw.encode_share_url().is_ok());

    let noisy_64: Vec<[f64; 2]> = (0..64)
        .map(|i| {
            let a = (i as f64) * 0.09817477042468103;
            [a.cos() * PI, a.sin() * E]
        })
        .collect();
    fw.active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&serde_json::json!({"radius": 0.8, "points": noisy_64}).to_string())
        .unwrap();
    let url_64 = fw.export_url_preset_json().unwrap().unwrap();
    assert!(url_64.len() <= MAX_DECODED_URL_STATE_BYTES);
    assert!(fw.encode_share_url().is_ok());
}

#[test]
fn test_custom_250_points_skips_url_sync_and_roundtrips_session() {
    let mut fw = SimulationFramework::new("geometry_topology");
    assert!(fw.select_tool_by_name("VietorisRipsTool"));

    let custom_250: Vec<[f64; 2]> = (0..250)
        .map(|i| {
            let a = (i as f64) * 0.025132741228718346;
            [5.0 + 3.123456789 * a.cos(), 5.0 + 3.123456789 * a.sin()]
        })
        .collect();
    fw.active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&serde_json::json!({"radius": 0.6, "points": custom_250}).to_string())
        .unwrap();
    assert!(fw.encode_share_url().is_err());

    let mut sync = UrlSyncController::default();
    let sig0 = UrlStateSignature {
        selected_tab: 0,
        active_tool_id: Some("VietorisRipsTool"),
        state_revision: 1,
    };
    sync.mark_synced(
        sig0,
        Some("#tab=geometry_topology&tool=VietorisRipsTool".to_string()),
    );
    let sig1 = UrlStateSignature {
        state_revision: 2,
        ..sig0
    };
    let _ = sync.poll_at_millis(sig1, false, 100, 300, || unreachable!());
    let _ = sync.poll_at_millis(sig1, false, 500, 300, || fw.encode_share_url());
    assert_eq!(
        sync.last_synced_fragment(),
        Some("#tab=geometry_topology&tool=VietorisRipsTool")
    );

    let preset_250 = fw
        .active_tool
        .as_ref()
        .unwrap()
        .export_preset_json()
        .unwrap();
    let session_250 = fw.save_state().unwrap();
    let mut fw_restored = SimulationFramework::new("geometry_topology");
    fw_restored.load_state(&session_250);
    assert_eq!(
        fw_restored.save_state().as_deref(),
        Some(session_250.as_str())
    );
    fw_restored
        .active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(&preset_250)
        .unwrap();
}

#[test]
fn test_interval_csv_json_exports_and_svg_png_figure_primitives() {
    let mut tool = VietorisRipsTool::default();
    tool.try_load_state(r#"{"radius":1.1,"points":[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]}"#)
        .unwrap();

    let csv = format_intervals_csv(&tool.persistence);
    assert!(csv.starts_with("dimension,birth,death,persistence,is_essential\n"));
    assert!(csv.contains(",true\n"));
    assert!(csv.contains("1,1.000000,1.414214,0.414214,false"));

    let json = format_intervals_json(&tool.persistence).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let rows: Vec<IntervalExportRow> = serde_json::from_value(parsed["intervals"].clone()).unwrap();
    assert!(rows
        .iter()
        .any(|r| r.dimension == 0 && r.is_essential && r.death > 1.4));
    assert!(rows.iter().any(|r| r.dimension == 1 && !r.is_essential));

    for mode in [PersistencePlotMode::Diagram, PersistencePlotMode::Barcode] {
        tool.set_plot_mode(mode);
        let prims = tool.export_plot_primitives();
        let opts = ExportOptions::default();
        let svg = export_primitives_svg(&prims, &opts).unwrap();
        assert!(svg.contains("<svg") && svg.len() > 250);
        let img = render_primitives_image(&prims, &opts).unwrap();
        let png = color32_image_to_png(&img).unwrap();
        let decoder = png::Decoder::new(std::io::Cursor::new(&png));
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0_u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        let (w, h) = (info.width as usize, info.height as usize);
        let pixels = buf[..info.buffer_size()].as_chunks::<4>().0;
        assert!(w > 0 && h > 0 && pixels.len() == w * h);
    }
}

// [cite:graph_parameters_rust]
