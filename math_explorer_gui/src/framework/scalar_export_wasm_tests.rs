use super::*;
use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

fn run_wasm_export_frame(fw: &mut SimulationFramework, format: ExportFormat, modal: bool) {
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(800.0, 600.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        egui_plot::clear_last_plot_datasets();
        fw.show_export_modal = modal;
        fw.queue_export(format);
        fw.show(ctx, "wasm_scalar_export_audit");
    });
}

fn verify_wasm_tool_scalar_exports(meta_id: &str, is_continuous: bool) {
    let meta = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == meta_id)
        .unwrap_or_else(|| panic!("Missing WASM tool {meta_id}"));
    let mut tool = (meta.build)();
    tool.step_simulation();
    let native = tool
        .export_scalar_field_2d(None)
        .unwrap_or_else(|| panic!("{meta_id} missing scalar field on wasm32"));
    let expected_dims = (native.width, native.height);

    let mut fw = SimulationFramework::new("__wasm_scalar__");
    fw.export_options.width = 240;
    fw.export_options.height = 180;
    fw.export_options.contour_levels = 4;
    fw.active_tool = Some(tool);

    run_wasm_export_frame(&mut fw, ExportFormat::Svg, true);
    let svg = fw
        .export_queue
        .last_exported_content
        .as_deref()
        .expect("WASM SVG content");
    assert!(!svg.contains("data:image/png;base64,") && !svg.contains("<image"));
    assert!(if is_continuous {
        svg.contains("<path ")
    } else {
        svg.contains("<rect ")
    });

    fw.export_options.seamless_tile = true;
    run_wasm_export_frame(&mut fw, ExportFormat::Png, true);
    let tile_bytes = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("WASM tile PNG");
    let tile_info = png::Decoder::new(std::io::Cursor::new(tile_bytes))
        .read_info()
        .expect("valid WASM tile PNG");
    assert_eq!(
        (
            tile_info.info().width as usize,
            tile_info.info().height as usize
        ),
        expected_dims
    );
    verify_wasm_heightmap_and_normal(&mut fw, expected_dims);
}

fn verify_wasm_heightmap_and_normal(fw: &mut SimulationFramework, expected_dims: (usize, usize)) {
    run_wasm_export_frame(fw, ExportFormat::HeightmapPng, false);
    let h_bytes = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("WASM heightmap PNG");
    let (hw, hh, samples) =
        egui_plot::decode_heightmap_png_16bit(h_bytes).expect("WASM 16-bit PNG decode");
    assert_eq!((hw, hh), expected_dims);
    assert_eq!(samples.len(), expected_dims.0 * expected_dims.1);

    run_wasm_export_frame(fw, ExportFormat::NormalMapPng, false);
    let n_bytes = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("WASM normal-map PNG");
    let n_info = png::Decoder::new(std::io::Cursor::new(n_bytes))
        .read_info()
        .expect("valid WASM normal PNG");
    assert_eq!(
        (n_info.info().width as usize, n_info.info().height as usize),
        expected_dims
    );
}

#[wasm_bindgen_test]
fn wasm_all_ten_raster_tools_scalar_svg_tile_heightmap_and_normal_exports() {
    let cases = [
        ("MorphogenesisTool", true),
        ("LeniaTool", true),
        ("ChladniWaveTankTool", true),
        ("FractalViewer", true),
        ("IsingModelTool", true),
        ("DoseCalculationTool", true),
        ("LatticeBoltzmannTool", true),
        ("MriKSpaceStudio", true),
        ("OpticsWaveTool", true),
        ("PrimeSpiralWidget", false),
        ("AttentionMapsTool", false),
    ];

    for (meta_id, is_continuous) in cases {
        verify_wasm_tool_scalar_exports(meta_id, is_continuous);
    }
}
