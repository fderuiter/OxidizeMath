#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::deferred_export_audit_tests::run_deferred_export_frame;
use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements, with_live_region};

fn decode_rgba_dims_and_pixels(png_bytes: &[u8]) -> (usize, usize, Vec<[u8; 4]>) {
    let mut reader = png::Decoder::new(std::io::Cursor::new(png_bytes))
        .read_info()
        .expect("valid PNG header");
    let mut buf = vec![0_u8; reader.output_buffer_size().expect("PNG buffer size")];
    let info = reader.next_frame(&mut buf).expect("PNG frame decode");
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    let raw = &buf[..info.buffer_size()];
    let pixels: Vec<[u8; 4]> = match info.color_type {
        png::ColorType::Rgba => raw.as_chunks::<4>().0.to_vec(),
        png::ColorType::Rgb => raw
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[r, g, b]| [r, g, b, 255])
            .collect(),
        other => panic!("Unexpected PNG color type {other:?}"),
    };
    (info.width as usize, info.height as usize, pixels)
}

fn verify_scalar_or_lattice_tool_exports(meta_id: &str, expect_contour_paths: bool) {
    let meta = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == meta_id)
        .unwrap_or_else(|| panic!("Missing registered raster tool {meta_id}"));
    let mut tool = (meta.build)();
    tool.step_simulation();
    assert!(
        tool.supports_scalar_field_2d(),
        "{meta_id} must advertise supports_scalar_field_2d()"
    );
    let field = tool
        .export_scalar_field_2d(None)
        .unwrap_or_else(|| panic!("{meta_id} returned None from export_scalar_field_2d"));
    let (native_w, native_h) = (field.width, field.height);

    let mut fw = SimulationFramework::new("__audit_scalar__");
    fw.export_options.width = 320;
    fw.export_options.height = 240;
    fw.export_options.contour_levels = 6;
    fw.active_tool = Some(tool);

    run_deferred_export_frame(&mut fw, ExportFormat::Svg, true);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let svg = fw
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing SVG for {meta_id}"));
    assert!(
        !svg.contains("data:image/png;base64,") && !svg.contains("<image"),
        "{meta_id} SVG must not embed base64 PNG <image>"
    );
    if expect_contour_paths {
        assert!(
            svg.contains("<path ") && svg.contains("data-iso-level="),
            "{meta_id} continuous SVG must emit stitched <path> contours"
        );
    } else {
        assert!(
            svg.contains("<rect ") && svg.contains("crispEdges"),
            "{meta_id} discrete lattice SVG must emit crisp vector <rect> cells"
        );
    }

    fw.export_options.seamless_tile = true;
    run_deferred_export_frame(&mut fw, ExportFormat::Png, true);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let tile_png = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("seamless tile PNG bytes");
    let (tw, th, _) = decode_rgba_dims_and_pixels(tile_png);
    assert_eq!(
        (tw, th),
        (native_w, native_h),
        "{meta_id} seamless tile PNG must match exact unpadded grid dimensions"
    );
    verify_heightmap_and_normal_exports(&mut fw, meta_id, (native_w, native_h));
}

fn verify_heightmap_and_normal_exports(
    fw: &mut SimulationFramework,
    meta_id: &str,
    expected_dims: (usize, usize),
) {
    clear_test_announcements();
    run_deferred_export_frame(fw, ExportFormat::HeightmapPng, false);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert!(with_live_region(|r| {
        r.slots[r.active_slot].contains("16-bit Heightmap PNG saved successfully")
            && r.slots[1 - r.active_slot].is_empty()
    }));
    assert!(all_announcements().iter().any(|(msg, prio)| {
        msg.contains("16-bit Heightmap PNG saved successfully") && prio == "polite"
    }));
    let h_bytes = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("heightmap PNG bytes");
    let (hw, hh, samples16) = egui_plot::decode_heightmap_png_16bit(h_bytes)
        .unwrap_or_else(|e| panic!("{meta_id} 16-bit heightmap decode failed: {e}"));
    assert_eq!((hw, hh), expected_dims);
    assert_eq!(samples16.len(), expected_dims.0 * expected_dims.1);

    clear_test_announcements();
    run_deferred_export_frame(fw, ExportFormat::NormalMapPng, false);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert!(all_announcements()
        .iter()
        .any(|(msg, prio)| msg.contains("Normal Map PNG saved successfully") && prio == "polite"));
    let n_bytes = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("normal-map PNG bytes");
    let (nw, nh, npx) = decode_rgba_dims_and_pixels(n_bytes);
    assert_eq!((nw, nh), expected_dims);
    assert!(
        npx.iter().all(|px| px[2] >= 127 && px[3] == 255),
        "{meta_id} normal-map must encode unit-hemisphere RGB vectors"
    );
}

#[test]
fn test_all_ten_raster_tools_deferred_scalar_svg_tile_heightmap_and_normal_exports() {
    let cases = [
        ("MorphogenesisTool", true),
        ("LeniaTool", true),
        ("ChladniWaveTankTool", true),
        ("FractalViewer", true),
        ("IsingModelTool", true),
        ("DoseCalculationTool", true),
        ("LatticeBoltzmannTool", true),
        ("MriKSpaceStudio", true),
        ("MriKSpaceScannerTool", true),
        ("OpticsWaveTool", true),
        ("PrimeSpiralWidget", false),
        ("AttentionMapsTool", false),
        ("AbelianSandpileStudio", false),
    ];
    for (id, is_continuous) in cases {
        verify_scalar_or_lattice_tool_exports(id, is_continuous);
    }

    let morpho_meta = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == "MorphogenesisTool")
        .expect("MorphogenesisTool registered");
    let mut morpho = (morpho_meta.build)();
    let saved = morpho.save_state().expect("morpho save_state");
    let periodic_json = saved.replacen("{", "{\"periodic_boundary\":true,", 1);
    morpho
        .try_load_state(&periodic_json)
        .expect("load periodic_boundary state");
    morpho.step_simulation();
    let p_field = morpho
        .export_scalar_field_2d(None)
        .expect("periodic morpho field");
    assert!(
        p_field.periodic,
        "MorphogenesisTool must mark periodic=true"
    );

    let fractal_meta = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == "FractalViewer")
        .expect("FractalViewer registered");
    let fractal = (fractal_meta.build)();
    let hi_field = fractal
        .export_scalar_field_2d(Some((640, 480)))
        .expect("640x480 fractal scalar field");
    assert_eq!((hi_field.width, hi_field.height), (640, 480));
    let hi_img = fractal
        .export_raster_image_sized((640, 480), false)
        .expect("640x480 fractal raster");
    assert_eq!((hi_img.width(), hi_img.height()), (640, 480));
}

#[test]
fn test_lexis_heatmap_mode_scalar_field_exports() {
    let pop_meta = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == "PopulationStructureTool")
        .expect("PopulationStructureTool registered");
    let mut pop = (pop_meta.build)();
    assert!(!pop.supports_scalar_field_2d());
    let saved = pop
        .save_state()
        .expect("PopulationStructureTool save_state");
    let lexis_json = saved.replace("\"Dashboard\"", "\"LexisHeatmap\"");
    pop.try_load_state(&lexis_json)
        .expect("load LexisHeatmap view_mode");
    assert!(pop.supports_scalar_field_2d());
    let l_field = pop
        .export_scalar_field_2d(None)
        .expect("LexisHeatmap scalar field");
    let l_dims = (l_field.width, l_field.height);

    let mut fw = SimulationFramework::new("__audit_lexis__");
    fw.export_options.width = 320;
    fw.export_options.height = 240;
    fw.active_tool = Some(pop);
    run_deferred_export_frame(&mut fw, ExportFormat::Svg, true);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let lexis_svg = fw.export_queue.last_exported_content.as_deref().unwrap();
    assert!(
        lexis_svg.contains("<path ")
            && lexis_svg.contains("data-iso-level=")
            && !lexis_svg.contains("<image"),
        "LexisHeatmap SVG must emit stitched contours without base64 <image>"
    );
    verify_heightmap_and_normal_exports(&mut fw, "PopulationStructureTool", l_dims);
}

#[test]
fn test_zero_base64_image_svg_across_all_figure_tools() {
    for meta in inventory::iter::<crate::framework::ToolMetadata> {
        let tool = (meta.build)();
        if !tool.export_capabilities().figure {
            continue;
        }
        let mut tool_fw = SimulationFramework::new("__audit_no_base64__");
        tool_fw.export_options.width = 240;
        tool_fw.export_options.height = 180;
        tool_fw.active_tool = Some(tool);
        run_deferred_export_frame(&mut tool_fw, ExportFormat::Svg, true);
        assert_eq!(
            tool_fw.export_status_severity,
            ExportStatusSeverity::Success,
            "SVG export failed for {}",
            meta.id
        );
        let svg = tool_fw
            .export_queue
            .last_exported_content
            .as_deref()
            .unwrap();
        assert!(
            !svg.contains("data:image/png;base64,") && !svg.contains("<image"),
            "Figure tool {} emitted raster <image> fallback in SVG",
            meta.id
        );
    }
}
