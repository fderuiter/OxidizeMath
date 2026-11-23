#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{ExportFormat, InteractiveTool, ShaderExportFormat, SimulationFramework};
use crate::url_state::{encode_share_fragment, parse_share_location, validate_url_preset_json};
use eframe::egui;
use std::collections::BTreeMap;

#[test]
fn test_full_viewport_raster_export_clamps_up_to_2048_instead_of_480x320() {
    let ctx = egui::Context::default();
    let mut viewer = FractalViewer::default();

    let default_export = viewer.export_raster_image().expect("default raster export");
    assert_eq!(default_export.size, [640, 480]);
    assert_ne!(default_export.size, [480, 320]);

    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(2560.0, 1440.0),
        )),
        max_texture_side: Some(2048),
        ..Default::default()
    };
    let _ = ctx.run(input, |c| viewer.show(c));
    assert!(viewer.viewport_size.0 > 2048 && viewer.viewport_size.1 > 1000);

    viewer.max_iter = 24;
    let hi_res = viewer.export_raster_image().expect("clamped hi-res export");
    assert_eq!(hi_res.size[0], MAX_FRACTAL_TEXTURE_SIDE);
    assert!(hi_res.size[1] > 900 && hi_res.size[1] <= MAX_FRACTAL_TEXTURE_SIDE);
}

#[test]
fn test_custom_recurrence_smooth_coloring_zero_alloc_and_palette_variation() {
    let mut viewer = FractalViewer {
        mode: FractalMode::CustomMandelbrot,
        max_iter: 60,
        smooth_coloring: true,
        palette: ShaderPalette::Heatmap,
        viewport_size: (64, 48),
        ..Default::default()
    };
    viewer.math_input.set_expression("z^3 + c");

    let job = CompiledFractalJob::from_bar(&viewer.math_input);
    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((s0, s1), _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        (
            job.smooth_escape_time(Complex::new(0.0, 0.0), Complex::new(0.1, 0.1), 80),
            job.smooth_escape_time(Complex::new(0.0, 0.0), Complex::new(2.0, 1.5), 80),
        )
    });
    drop(_alloc_guard);
    assert!((s0 - 80.0).abs() < 1.0e-9);
    assert!(s1.is_finite() && s1 < 10.0);

    let smooth_img = viewer.export_raster_image().expect("smooth custom export");
    viewer.smooth_coloring = false;
    let discrete_img = viewer
        .export_raster_image()
        .expect("discrete custom export");
    assert_eq!(smooth_img.size, [64, 48]);
    assert_ne!(smooth_img.pixels, discrete_img.pixels);
}

#[test]
fn test_fractal_wgsl_and_glsl_standalone_exports_and_url_roundtrip() {
    let mut pmap = BTreeMap::new();
    pmap.insert("a".to_string(), 1.25);
    let mut viewer = FractalViewer {
        mode: FractalMode::CustomJulia,
        smooth_coloring: false,
        palette: ShaderPalette::MappedGradient,
        julia_c: Complex::new(-0.35, 0.62),
        ..Default::default()
    };
    viewer.math_input.set_expression("a * sin(z^3) + c");
    viewer.math_input.set_params_map(&pmap);

    let wgsl = viewer
        .export_shader_source(ShaderExportFormat::Wgsl)
        .expect("emit fractal WGSL");
    assert!(wgsl.contains("struct ShaderUniforms"));
    assert!(wgsl.contains("@fragment"));
    assert!(wgsl.contains("c_sin(c_pow(z, vec2<f32>(3.0, 0.0)))"));
    assert!(wgsl.contains("1.25"));

    let glsl = viewer
        .export_shader_source(ShaderExportFormat::Glsl)
        .expect("emit fractal GLSL");
    assert!(glsl.starts_with("#version 300 es"));
    assert!(glsl.contains("void mainImage(out vec4 fragColor, in vec2 fragCoord)"));
    assert!(glsl.contains("c_sin(c_pow(z, vec2(3.0, 0.0)))"));
    assert!(glsl.contains("1.25"));

    let preset = viewer.export_preset_json().expect("export preset JSON");
    validate_url_preset_json(&preset).expect("fits within 4096-byte URL budget");
    let frag = encode_share_fragment("chaos", Some("FractalViewer"), Some(&preset))
        .expect("encode share fragment");
    let parsed = parse_share_location(&frag, "")
        .expect("parse share fragment")
        .expect("present params");
    let mut restored = FractalViewer::default();
    restored
        .import_preset_json(parsed.state_json.as_deref().unwrap())
        .expect("restore fractal state");
    assert_eq!(restored.mode, FractalMode::CustomJulia);
    assert!(!restored.smooth_coloring);
    assert_eq!(restored.palette, ShaderPalette::MappedGradient);
    assert_eq!(restored.math_input.raw_text, "a * sin(z^3) + c");
    assert_eq!(restored.math_input.params_map(), pmap);
}

#[test]
fn test_fractal_framework_shader_file_and_clipboard_queue_dispatch() {
    let temp_dir = std::env::temp_dir().join(format!("om_fractal_shader_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);

    let mut fw = SimulationFramework::new("chaos");
    assert!(fw.select_tool_by_name("FractalViewer"));
    fw.export_queue.destination_dir = Some(temp_dir.clone());
    assert!(
        fw.active_tool
            .as_ref()
            .unwrap()
            .export_capabilities()
            .shader
    );

    let ctx = egui::Context::default();
    fw.queue_export(ExportFormat::Wgsl);
    let _ = ctx.run(egui::RawInput::default(), |c| fw.show(c, "test_wgsl"));
    let fname_w = fw
        .export_queue
        .last_exported_filename
        .clone()
        .expect("saved .wgsl file");
    assert!(fname_w.ends_with(".wgsl"));
    let wgsl_disk = std::fs::read_to_string(temp_dir.join(&fname_w)).expect("read .wgsl");
    assert!(wgsl_disk.contains("fn fs_main"));

    fw.queue_export(ExportFormat::Glsl);
    let _ = ctx.run(egui::RawInput::default(), |c| fw.show(c, "test_glsl"));
    let fname_g = fw
        .export_queue
        .last_exported_filename
        .clone()
        .expect("saved .glsl file");
    assert!(fname_g.ends_with(".glsl"));
    let glsl_disk = std::fs::read_to_string(temp_dir.join(&fname_g)).expect("read .glsl");
    assert!(glsl_disk.starts_with("#version 300 es"));

    let clips = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let clips_clone = Arc::clone(&clips);
    fw.export_queue.custom_clipboard_writer = Some(Arc::new(move |text| {
        clips_clone.lock().unwrap().push(text.to_string());
        Ok(())
    }));
    assert!(fw.queue_shader_clipboard_export(ExportFormat::Wgsl));
    let _ = ctx.run(egui::RawInput::default(), |c| fw.show(c, "test_clip"));
    let captured = clips.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].contains("struct ShaderUniforms"));
    assert_eq!(
        fw.export_queue.last_clipboard_content.as_deref(),
        Some(captured[0].as_str())
    );
    let _ = std::fs::remove_dir_all(&temp_dir);
}
