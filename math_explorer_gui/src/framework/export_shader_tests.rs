#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::tabs::analysis::complex_mapping::{ComplexMappingTool, MappingFunction};
use crate::tabs::chaos::fractals::{FractalMode, FractalViewer};
use crate::tabs::fluid_dynamics::potential_flow::{
    ElementType, PotentialFlowTool, SerializedFlowElement, MAX_SHADER_SINGULARITIES,
};
use eframe::egui;
use std::sync::{Arc, Mutex};

type CapturedFiles = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

fn find_ak_button_center(output: &egui::FullOutput, label_substr: &str) -> Option<egui::Pos2> {
    let ak_update = output.platform_output.accesskit_update.as_ref()?;
    let bounds = ak_update.nodes.iter().find_map(|(_, node)| {
        let label = node.label().unwrap_or("");
        (node.role() == egui::accesskit::Role::Button && label.contains(label_substr))
            .then(|| node.bounds())
            .flatten()
    })?;
    Some(egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    ))
}

fn click_at(
    ctx: &egui::Context,
    framework: &mut SimulationFramework,
    pos: egui::Pos2,
    size: egui::Vec2,
) {
    let screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect,
            events: vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| framework.show(ctx, "shader_test_fw"),
    );
    let _ = ctx.run(
        egui::RawInput {
            screen_rect,
            events: vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ctx| framework.show(ctx, "shader_test_fw"),
    );
}

fn setup_fractal_shader_framework() -> (SimulationFramework, CapturedFiles, Arc<Mutex<Vec<String>>>)
{
    let mut framework = SimulationFramework::new("__shader_test__");
    let fractal = FractalViewer {
        mode: FractalMode::Julia,
        julia_c: num_complex::Complex::new(-0.8, 0.156),
        zoom: 2.5,
        ..Default::default()
    };
    framework.active_tool = Some(Box::new(fractal));

    let captured_files: CapturedFiles = Arc::new(Mutex::new(Vec::new()));
    let files_sink = Arc::clone(&captured_files);
    framework.export_queue.custom_writer = Some(Arc::new(move |name, bytes| {
        files_sink
            .lock()
            .unwrap()
            .push((name.to_string(), bytes.to_vec()));
    }));

    let captured_clip: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let clip_sink = Arc::clone(&captured_clip);
    framework.export_queue.custom_clipboard_writer = Some(Arc::new(move |text| {
        clip_sink.lock().unwrap().push(text.to_string());
        Ok(())
    }));
    (framework, captured_files, captured_clip)
}

#[test]
fn test_action_bar_shader_file_and_clipboard_exports_and_live_region() {
    let (mut framework, captured_files, captured_clip) = setup_fractal_shader_framework();

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let wide = egui::vec2(1400.0, 800.0);
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, wide)),
            ..Default::default()
        },
        |ctx| framework.show(ctx, "shader_test_fw"),
    );

    let webgl_btn = find_ak_button_center(&out, "Export WebGL2 (.frag)")
        .expect("Export WebGL2 (.frag) button must be rendered");
    click_at(&ctx, &mut framework, webgl_btn, wide);

    let files = captured_files.lock().unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].0, "fractal_viewer_webgl2.frag");
    let src = std::str::from_utf8(&files[0].1).unwrap();
    assert!(src.starts_with("#version 300 es"));
    assert!(src.contains("const bool IS_JULIA = true;"));
    assert!(src.contains("const float ZOOM = 2.5;"));
    drop(files);

    let out2 = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, wide)),
            ..Default::default()
        },
        |ctx| framework.show(ctx, "shader_test_fw"),
    );
    let copy_st_btn =
        find_ak_button_center(&out2, "Copy Shadertoy").expect("Copy Shadertoy button must exist");
    click_at(&ctx, &mut framework, copy_st_btn, wide);

    let clips = captured_clip.lock().unwrap();
    assert_eq!(clips.len(), 1);
    assert!(clips[0].contains("void mainImage(out vec4 fragColor, in vec2 fragCoord)"));
    assert_eq!(
        framework.export_status_message.as_deref(),
        Some("Copied Shadertoy GLSL shader to clipboard!")
    );
    let live_msg = crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone());
    assert_eq!(live_msg, "Copied Shadertoy GLSL shader to clipboard!");
}

#[test]
fn test_complex_mapping_custom_ast_high_contrast_and_error_handling() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        high_contrast: true,
        ..Default::default()
    };
    tool.math_input.set_expression("z^3 - 1/z");

    let webgl2 = tool.export_glsl_shader(ShaderDialect::WebGl2Es300).unwrap();
    assert!(webgl2.contains("cpow(z, vec2(3.0, 0.0)) - cdiv(vec2(1.0, 0.0), z)"));
    assert!(webgl2.contains("const bool HIGH_CONTRAST = true;"));
    assert!(webgl2.contains("ensure_dark_canvas_contrast"));

    tool.math_input.set_expression("z + a^2 / z");
    let mut params = std::collections::BTreeMap::new();
    params.insert("a".to_string(), 0.85);
    tool.math_input.set_params_map(&params);
    let joukowsky = tool.export_glsl_shader(ShaderDialect::WebGl2Es300).unwrap();
    assert!(joukowsky.contains("const vec2 a = vec2(0.85, 0.0);"));
    assert!(joukowsky.contains("z + cdiv(cpow(a, vec2(2.0, 0.0)), z)"));

    tool.math_input.set_expression("z + gl_FragCoord");
    let mut framework = SimulationFramework::new("__shader_err__");
    framework.active_tool = Some(Box::new(tool));
    let ctx = egui::Context::default();
    framework.queue_export(ExportFormat::GlslWebGl2);
    framework.dispatch_pending_exports_with_ctx(&ctx);

    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Error
    );
    let err_msg = framework.export_status_message.as_deref().unwrap_or("");
    assert!(
        err_msg.contains("gl_FragCoord"),
        "Expected invalid GLSL identifier diagnostic, got `{err_msg}`"
    );
}

#[test]
fn test_potential_flow_singularity_cap_64_sink_sign_and_non_shader_warning() {
    let mut flow = PotentialFlowTool::default();
    flow.elements.clear();
    flow.elements.push(SerializedFlowElement {
        element_type: ElementType::Sink,
        strength: 12.5,
        x: 1.0,
        y: -2.0,
        angle: 0.0,
    });
    for i in 0..75 {
        flow.elements.push(SerializedFlowElement {
            element_type: ElementType::Vortex,
            strength: 5.0,
            x: (i % 10) as f64,
            y: (i / 10) as f64,
            angle: 0.0,
        });
    }
    flow.high_contrast = true;

    let glsl = flow.export_glsl_shader(ShaderDialect::Shadertoy).unwrap();
    assert!(glsl.contains("64/76 elements (cap 64)"));
    assert!(glsl.contains("flow_source(p, -12.5, vec2(1.0, -2.0))"));
    assert!(glsl.contains("const bool HIGH_CONTRAST = true;"));
    let emitted_calls = glsl.matches("    acc += flow_").count();
    assert_eq!(emitted_calls, MAX_SHADER_SINGULARITIES);
}

#[test]
fn test_compact_overflow_menu_and_non_shader_tool_warning_status() {
    let mut fw = SimulationFramework::new("epidemiology");
    assert!(fw
        .active_tool
        .as_ref()
        .is_some_and(|t| !t.export_capabilities().shader));
    let ctx = egui::Context::default();
    fw.queue_export(ExportFormat::GlslWebGl2);
    fw.dispatch_pending_exports_with_ctx(&ctx);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Warning);
    assert!(fw
        .export_status_message
        .as_deref()
        .unwrap_or("")
        .contains("GLSL shader export is unavailable"));

    let mut shader_fw = SimulationFramework::new("__compact_shader__");
    shader_fw.active_tool = Some(Box::new(FractalViewer::default()));
    ctx.enable_accesskit();
    let compact = egui::vec2(720.0, 600.0);
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, compact)),
            ..Default::default()
        },
        |ctx| shader_fw.show(ctx, "shader_test_fw"),
    );
    let menu_pos =
        find_ak_button_center(&out, "Export ▾").expect("Compact Export menu button must exist");
    click_at(&ctx, &mut shader_fw, menu_pos, compact);
    let menu_out = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, compact)),
            ..Default::default()
        },
        |ctx| shader_fw.show(ctx, "shader_test_fw"),
    );
    assert!(
        find_ak_button_center(&menu_out, "Export WebGL2 (.frag)").is_some(),
        "Compact overflow menu must expose Export WebGL2 (.frag)"
    );
}

#[test]
fn test_fractal_custom_mandelbrot_and_julia_glsl_shader_exports() {
    let mut fractal = FractalViewer {
        mode: FractalMode::CustomMandelbrot,
        high_contrast: true,
        ..Default::default()
    };
    fractal.math_input.set_expression("a * z^3 + c");
    let mut params = std::collections::BTreeMap::new();
    params.insert("a".to_string(), 1.25);
    fractal.math_input.set_params_map(&params);

    let webgl2 = fractal
        .export_glsl_shader(ShaderDialect::WebGl2Es300)
        .unwrap();
    assert!(webgl2.contains("const bool IS_JULIA = false;"));
    assert!(webgl2.contains("const bool CUSTOM_ESCAPE = true;"));
    assert!(webgl2.contains("const vec2 a = vec2(1.25, 0.0);"));
    assert!(webgl2.contains("cmul(a, cpow(z, vec2(3.0, 0.0))) + c"));

    fractal.mode = FractalMode::CustomJulia;
    let shadertoy = fractal
        .export_glsl_shader(ShaderDialect::Shadertoy)
        .unwrap();
    assert!(shadertoy.contains("const bool IS_JULIA = true;"));
    assert!(shadertoy.contains("void mainImage(out vec4 fragColor, in vec2 fragCoord)"));
}
