use super::deferred_export_tests::decode_rgba_pixels;
use super::*;
use eframe::egui;
use std::collections::HashSet;

fn build_tool(id: &str) -> Box<dyn InteractiveTool> {
    let meta = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == id || m.display_name == id)
        .unwrap_or_else(|| panic!("Missing registered tool {id}"));
    (meta.build)()
}

fn run_export_frame(fw: &mut SimulationFramework, format: ExportFormat) {
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw_input.clone(), |ctx| {
        egui_plot::clear_last_plot_datasets();
        fw.show_export_modal = true;
        fw.queue_export(format);
        fw.show(ctx, "adapter_export_test");
    });
    if format == ExportFormat::Gif {
        for _ in 0..250 {
            if fw.active_gif_progress().is_none() {
                break;
            }
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(std::time::Duration::from_millis(10));
            let _ = ctx.run(raw_input.clone(), |ctx| {
                fw.show(ctx, "adapter_export_test");
            });
        }
    }
}

fn verify_visual_tool_figure_exports(id: &str, fw: &mut SimulationFramework, png_out: &str) {
    run_export_frame(fw, ExportFormat::Png);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let png = fw
        .export_queue
        .last_exported_bytes
        .clone()
        .unwrap_or_else(|| panic!("Missing PNG for {id}"));
    let colors: HashSet<[u8; 4]> = decode_rgba_pixels(&png)
        .unwrap_or_else(|e| panic!("Invalid PNG for {id}: {e}"))
        .into_iter()
        .collect();
    assert!(
        colors.len() >= 3,
        "Expected >= 3 distinct colors in {id} PNG, got {}",
        colors.len()
    );
    let _ = std::fs::write(png_out, &png);

    run_export_frame(fw, ExportFormat::Svg);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let svg = fw
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing SVG for {id}"));
    assert!(svg.contains("<svg") && svg.len() > 250);

    run_export_frame(fw, ExportFormat::Gif);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let gif = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .unwrap_or_else(|| panic!("Missing GIF for {id}"));
    assert!(gif.starts_with(b"GIF89a") && gif.len() > 128);
}

fn verify_visual_tool_dataset_exports(
    id: &str,
    fw: &mut SimulationFramework,
    expect_multi_table: bool,
) {
    let ctx = egui::Context::default();
    run_export_frame(fw, ExportFormat::Csv);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let csv = fw
        .export_queue
        .last_exported_content
        .clone()
        .unwrap_or_else(|| panic!("Missing CSV for {id}"));
    if expect_multi_table {
        assert!(
            csv.starts_with("Table,") && csv.lines().count() >= 3,
            "Expected multi-table CSV header with 'Table,' for {id}, got: {}",
            csv.lines().next().unwrap_or("")
        );
    } else {
        assert!(
            !csv.starts_with("Table,") && csv.lines().count() >= 2,
            "Expected single-table CSV header for {id}, got: {}",
            csv.lines().next().unwrap_or("")
        );
    }

    fw.queue_clipboard_export();
    fw.dispatch_pending_exports_with_ctx(&ctx);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert_eq!(
        fw.export_queue.last_clipboard_content.as_deref(),
        Some(csv.as_str())
    );

    run_export_frame(fw, ExportFormat::Json);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let json = fw
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing JSON for {id}"));
    let parsed: serde_json::Value = serde_json::from_str(json).expect("valid dataset JSON");
    if expect_multi_table {
        assert!(
            parsed.get("tables").and_then(|t| t.as_object()).is_some(),
            "Expected {{\"tables\": ...}} JSON object for {id}"
        );
    } else {
        assert!(
            parsed.as_array().is_some_and(|a| !a.is_empty()),
            "Expected non-empty JSON array for single-table tool {id}"
        );
    }
}

#[test]
fn test_visual_non_figure_tools_png_svg_gif_csv_clipboard_json_and_ply() {
    let cases = [
        (
            "Gaussian Splatting",
            "/tmp/gaussian_splatting_export.png",
            true,
        ),
        ("CeraTool", "/tmp/cera_climate_export.png", true),
        ("GridWorldTool", "/tmp/grid_world_export.png", false),
        (
            "QTableInspectorTool",
            "/tmp/q_table_inspector_export.png",
            false,
        ),
        (
            "NeuralNetworkVizTool",
            "/tmp/neural_network_viz_export.png",
            true,
        ),
        (
            "NetworkPropagationTool",
            "/tmp/network_propagation_export.png",
            true,
        ),
    ];

    for (id, png_out, expect_multi_table) in cases {
        let mut tool = build_tool(id);
        for _ in 0..3 {
            tool.step_simulation();
        }
        let caps = tool.export_capabilities();
        assert!(
            caps.figure && caps.dataset,
            "{id} must support figure+dataset"
        );
        let mut fw = SimulationFramework::new("__adapter_visual__");
        fw.export_options.width = 480;
        fw.export_options.height = 320;
        fw.export_gif_frames = 4;
        fw.active_tool = Some(tool);

        verify_visual_tool_figure_exports(id, &mut fw, png_out);
        verify_visual_tool_dataset_exports(id, &mut fw, expect_multi_table);

        if id == "Gaussian Splatting" {
            run_export_frame(&mut fw, ExportFormat::Ply);
            assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
            let ply = fw
                .export_queue
                .last_exported_bytes
                .as_deref()
                .expect("Gaussian Splatting PLY bytes");
            assert!(ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
        }
    }
}

#[test]
fn test_all_eight_raster_tools_bounded_matrix_csv_clipboard_and_json() {
    let raster_ids = [
        "AttentionMapsTool",
        "IsingModelTool",
        "OpticsWaveTool",
        "FractalViewer",
        "DoseCalculationTool",
        "PrimeSpiralWidget",
        "MorphogenesisTool",
        "LatticeBoltzmannTool",
    ];
    let ctx = egui::Context::default();
    for id in raster_ids {
        let mut fw = SimulationFramework::new("__adapter_raster__");
        fw.active_tool = Some(build_tool(id));
        run_export_frame(&mut fw, ExportFormat::Csv);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let csv = fw
            .export_queue
            .last_exported_content
            .clone()
            .unwrap_or_else(|| panic!("Missing CSV for {id}"));
        let data_rows = csv.lines().count().saturating_sub(1);
        assert!(
            (1..=MAX_RASTER_TABLE_ROWS * 2).contains(&data_rows),
            "Unexpected row count {data_rows} for {id}"
        );

        fw.queue_clipboard_export();
        fw.dispatch_pending_exports_with_ctx(&ctx);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        assert!(fw.export_queue.last_clipboard_content.is_some());

        run_export_frame(&mut fw, ExportFormat::Json);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let json = fw
            .export_queue
            .last_exported_content
            .as_deref()
            .expect("raster JSON");
        assert!(serde_json::from_str::<serde_json::Value>(json).is_ok());
    }

    let mut morpho = crate::tabs::morphogenesis::MorphogenesisTool::default();
    morpho.step_lockstep(2);
    assert!(morpho.fork_here());
    morpho.step_lockstep(2);
    let mut forked_fw = SimulationFramework::new("__adapter_morpho_fork__");
    forked_fw.active_tool = Some(Box::new(morpho));
    run_export_frame(&mut forked_fw, ExportFormat::Csv);
    let forked_csv = forked_fw
        .export_queue
        .last_exported_content
        .as_deref()
        .expect("forked Morphogenesis CSV");
    assert!(
        forked_csv.starts_with("Dataset,X,Y\n"),
        "Forked MorphogenesisTool must preserve twin Dataset,X,Y CSV schema"
    );
}

#[test]
fn test_analytical_calculators_dataset_only_starter_and_invalid_input_guards() {
    let ctx = egui::Context::default();
    for id in ["TokenizationTool", "FactorizationTool", "UalbfWidget"] {
        let tool = build_tool(id);
        assert_eq!(
            tool.export_capabilities(),
            ExportCapabilities::dataset_only()
        );
        let mut fw = SimulationFramework::new("__adapter_calc__");
        fw.active_tool = Some(tool);

        run_export_frame(&mut fw, ExportFormat::Csv);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let csv = fw
            .export_queue
            .last_exported_content
            .clone()
            .expect("calculator CSV");
        assert!(csv.lines().count() >= 2);

        fw.queue_clipboard_export();
        fw.dispatch_pending_exports_with_ctx(&ctx);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        assert_eq!(fw.export_queue.last_clipboard_content, Some(csv));

        run_export_frame(&mut fw, ExportFormat::Json);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let json = fw
            .export_queue
            .last_exported_content
            .as_deref()
            .expect("calculator JSON");
        assert!(serde_json::from_str::<serde_json::Value>(json).is_ok());

        for blocked in [ExportFormat::Png, ExportFormat::Svg] {
            run_export_frame(&mut fw, blocked);
            assert_eq!(fw.export_status_severity, ExportStatusSeverity::Warning);
        }
    }

    let temp_dir = std::env::temp_dir().join("oxidize_factorization_invalid_guard");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("temp dir");
    let mut invalid_fact = crate::tabs::number_theory::factorization::FactorizationTool::default();
    invalid_fact.set_test_input("not_an_integer");
    let mut bad_fw = SimulationFramework::new("__adapter_bad_fact__");
    bad_fw.export_queue.destination_dir = Some(temp_dir.clone());
    bad_fw.active_tool = Some(Box::new(invalid_fact));
    run_export_frame(&mut bad_fw, ExportFormat::Csv);
    assert_eq!(bad_fw.export_status_severity, ExportStatusSeverity::Error);
    run_export_frame(&mut bad_fw, ExportFormat::Json);
    assert_eq!(bad_fw.export_status_severity, ExportStatusSeverity::Error);
    assert_eq!(std::fs::read_dir(&temp_dir).unwrap().count(), 0);
    let _ = std::fs::remove_dir_all(&temp_dir);
}
