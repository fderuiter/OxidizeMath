#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui;
use std::collections::HashSet;

fn find_registered_tool(id: &str) -> Box<dyn InteractiveTool> {
    inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == id)
        .map(|m| (m.build)())
        .unwrap_or_else(|| panic!("Registered tool {id} not found"))
}

fn run_modal_export_after_clear(framework: &mut SimulationFramework, format: ExportFormat) {
    run_modal_request_after_clear(
        framework,
        ExportRequest {
            format,
            target: ExportTarget::File,
        },
    );
}

fn run_modal_request_after_clear(framework: &mut SimulationFramework, request: ExportRequest) {
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        egui_plot::clear_last_plot_datasets();
        framework.show_export_modal = true;
        framework
            .export_queue
            .push_target(request.format, request.target);
        framework.show(ctx, "audit_spatial_exports");
    });
}

#[test]
fn test_spatial_tools_emit_pure_vector_svg_without_base64_raster() {
    let spatial_tools = [
        ("MorphogenesisTool", "<path"),
        ("FractalViewer", "<path"),
        ("LatticeBoltzmannTool", "<path"),
        ("DoseCalculationTool", "<path"),
        ("IsingModelTool", "<path"),
        ("PrimeSpiralWidget", "<path"),
        ("AttentionMapsTool", "<rect"),
    ];

    let copied_buf = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    for (tool_id, expected_tag) in spatial_tools {
        let mut tool = find_registered_tool(tool_id);
        if tool_id == "MorphogenesisTool"
            || tool_id == "LatticeBoltzmannTool"
            || tool_id == "IsingModelTool"
        {
            for _ in 0..4 {
                tool.step_simulation();
            }
        }
        let mut framework = SimulationFramework::new("__vector_svg_audit__");
        framework.export_contour_levels = 8;
        let sink = std::sync::Arc::clone(&copied_buf);
        framework.export_queue.custom_clipboard_writer = Some(std::sync::Arc::new(move |txt| {
            *sink.lock().unwrap() = txt.to_string();
            Ok(())
        }));
        framework.active_tool = Some(tool);

        run_modal_export_after_clear(&mut framework, ExportFormat::Svg);
        assert_eq!(
            framework.export_status_severity,
            ExportStatusSeverity::Success,
            "SVG export failed for {tool_id}: {:?}",
            framework.export_status_message
        );
        let svg = framework
            .export_queue
            .last_exported_content
            .as_deref()
            .unwrap_or_else(|| panic!("Missing SVG for {tool_id}"));
        assert!(
            svg.contains(expected_tag)
                && !svg.contains("data:image/png")
                && !svg.contains("<image"),
            "{tool_id} SVG must emit {expected_tag} primitives with zero raster <image> tags"
        );

        run_modal_request_after_clear(
            &mut framework,
            ExportRequest {
                format: ExportFormat::Svg,
                target: ExportTarget::Clipboard,
            },
        );
        let clip_svg = copied_buf.lock().unwrap().clone();
        assert!(clip_svg.contains(expected_tag) && !clip_svg.contains("data:image/png"));
    }
}

fn verify_png_16bit_header_and_decode(
    png_bytes: &[u8],
    expected_size: u32,
    expected_color: png::ColorType,
) {
    let decoder = png::Decoder::new(std::io::Cursor::new(png_bytes));
    let reader = decoder.read_info().expect("valid PNG header");
    let info = reader.info();
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
    assert_eq!(info.color_type, expected_color);
    assert_eq!((info.width, info.height), (expected_size, expected_size));
}

fn assert_periodic_normal_seam_continuity(
    heights: &[u16],
    normals: &[[u16; 3]],
    size: usize,
    strength: f32,
) {
    use math_commons::math_kernel::types::flatten_2d_index;
    let min_q = *heights.iter().min().unwrap_or(&0) as f64;
    let max_q = *heights.iter().max().unwrap_or(&0) as f64;
    let span = (max_q - min_q).max(1.0);
    let grad_scale = f64::from(strength) * (size as f64 / 64.0);
    let mut max_err = 0.0_f64;

    for idx in 0..size {
        for &(x, y) in &[
            (0usize, idx),
            (size - 1, idx),
            (idx, 0usize),
            (idx, size - 1),
        ] {
            let xp = if x + 1 < size { x + 1 } else { 0 };
            let xm = if x > 0 { x - 1 } else { size - 1 };
            let yp = if y + 1 < size { y + 1 } else { 0 };
            let ym = if y > 0 { y - 1 } else { size - 1 };
            let dh_dx = 0.5
                * ((heights[flatten_2d_index(xp, y, size)] as f64
                    - heights[flatten_2d_index(xm, y, size)] as f64)
                    / span)
                * grad_scale;
            let dh_dy = 0.5
                * ((heights[flatten_2d_index(x, yp, size)] as f64
                    - heights[flatten_2d_index(x, ym, size)] as f64)
                    / span)
                * grad_scale;
            let inv = 1.0 / (dh_dx * dh_dx + dh_dy * dh_dy + 1.0).sqrt();
            let expected = [
                (-dh_dx * inv) * 0.5 + 0.5,
                (-dh_dy * inv) * 0.5 + 0.5,
                inv * 0.5 + 0.5,
            ];
            let actual = normals[flatten_2d_index(x, y, size)];
            for c in 0..3 {
                let err = ((actual[c] as f64 / 65535.0) - expected[c]).abs();
                max_err = max_err.max(err);
            }
        }
    }
    assert!(
        max_err < 5e-4,
        "Periodic seam normal wrap mismatch {max_err}"
    );
}

#[test]
fn test_16bit_heightmap_and_normal_map_unpadded_resolutions() {
    for &size in &[512_u32, 1024_u32] {
        let mut framework = SimulationFramework::new("__texture16_audit__");
        framework.export_texture_size = size;
        framework.export_normal_strength = 2.0;
        framework.active_tool = Some(find_registered_tool("FractalViewer"));

        run_modal_export_after_clear(&mut framework, ExportFormat::Heightmap16Png);
        assert_eq!(
            framework.export_status_severity,
            ExportStatusSeverity::Success
        );
        let h_bytes = framework
            .export_queue
            .last_exported_bytes
            .clone()
            .expect("heightmap bytes");
        verify_png_16bit_header_and_decode(&h_bytes, size, png::ColorType::Grayscale);
        let (hw, hh, samples) = egui_plot::export_primitives::decode_heightmap_16bit_png(&h_bytes)
            .expect("16-bit gray");
        assert_eq!((hw, hh), (size, size));
        let unique_levels: HashSet<u16> = samples.iter().copied().collect();
        assert!(
            unique_levels.len() > 256,
            "Smooth fractal heightmap at {size}x{size} should exceed 8-bit 256 levels, got {}",
            unique_levels.len()
        );

        run_modal_export_after_clear(&mut framework, ExportFormat::NormalMapPng);
        assert_eq!(
            framework.export_status_severity,
            ExportStatusSeverity::Success
        );
        let n_bytes = framework
            .export_queue
            .last_exported_bytes
            .clone()
            .expect("normal bytes");
        verify_png_16bit_header_and_decode(&n_bytes, size, png::ColorType::Rgb);
        let (nw, nh, normals) = egui_plot::export_primitives::decode_normal_map_16bit_png(&n_bytes)
            .expect("16-bit rgb");
        assert_eq!((nw, nh), (size, size));
        assert_eq!(normals.len(), (size * size) as usize);
    }
}

#[test]
fn test_periodic_gray_scott_normal_map_seam_continuity() {
    let mut morpho = find_registered_tool("MorphogenesisTool");
    morpho
        .try_load_state(
            r#"{"a":0.1,"b":0.9,"d_u":0.16,"d_v":0.08,"dt":1.0,"width":64,"height":64,"kinetics_mode":"GrayScott","boundary":"Periodic","f":0.0367,"k":0.0649,"paused":true}"#,
        )
        .expect("periodic Gray-Scott state");
    for _ in 0..15 {
        morpho.step_simulation();
    }
    let mut fw = SimulationFramework::new("__periodic_seam_audit__");
    fw.export_texture_size = 512;
    fw.export_normal_strength = 2.0;
    fw.active_tool = Some(morpho);
    run_modal_export_after_clear(&mut fw, ExportFormat::Heightmap16Png);
    let (_, _, heights) = egui_plot::export_primitives::decode_heightmap_16bit_png(
        fw.export_queue.last_exported_bytes.as_deref().unwrap(),
    )
    .unwrap();
    run_modal_export_after_clear(&mut fw, ExportFormat::NormalMapPng);
    let (_, _, normals) = egui_plot::export_primitives::decode_normal_map_16bit_png(
        fw.export_queue.last_exported_bytes.as_deref().unwrap(),
    )
    .unwrap();
    assert_periodic_normal_seam_continuity(&heights, &normals, 512, 2.0);
}
