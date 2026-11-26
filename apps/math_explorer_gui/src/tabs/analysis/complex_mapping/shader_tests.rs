#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{ExportFormat, ShaderExportFormat, SimulationFramework};
use crate::url_state::{encode_share_fragment, parse_share_location, validate_url_preset_json};

#[test]
fn test_domain_coloring_high_contrast_and_shader_exports() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        palette: ShaderPalette::PhaseWheel,
        high_contrast: true,
        ..Default::default()
    };
    tool.math_input.set_expression("sin(z^3) - 1/z");

    let img = tool.render_domain_coloring_image(32);
    assert_eq!(img.size, [32, 32]);
    for px in &img.pixels {
        let ratio = crate::app::contrast_ratio(*px, egui::Color32::BLACK);
        assert!(
            ratio >= 3.0,
            "HighContrast pixel {px:?} fell below 3.0:1 ({ratio:.3})"
        );
    }

    let wgsl = tool
        .export_shader_source(ShaderExportFormat::Wgsl)
        .expect("emit WGSL domain shader");
    assert!(wgsl.contains("om_domain_color("));
    assert!(wgsl.contains("c_sin(c_pow(z, vec2<f32>(3.0, 0.0)))"));

    let glsl = tool
        .export_shader_source(ShaderExportFormat::Glsl)
        .expect("emit GLSL domain shader");
    assert!(glsl.starts_with("#version 300 es"));
    assert!(glsl.contains("om_domain_color("));
    assert!(glsl.contains("c_div(vec2(1.0, 0.0), z)"));
}

#[test]
fn test_complex_mapping_url_bookmark_and_export_queue_roundtrip() {
    let temp_dir = std::env::temp_dir().join(format!("om_complex_shader_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);

    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        domain_coloring: false,
        palette: ShaderPalette::Heatmap,
        grid_min: -3.0,
        grid_max: 3.0,
        ..Default::default()
    };
    tool.math_input.set_expression("cos(z) + z / (z^2 + 1)");

    let preset = tool.export_preset_json().expect("export preset");
    validate_url_preset_json(&preset).expect("within 4096-byte URL budget");
    let frag = encode_share_fragment("analysis", Some("ComplexMappingTool"), Some(&preset))
        .expect("encode share fragment");
    let decoded = parse_share_location(&frag, "")
        .expect("parse share fragment")
        .expect("present share params");

    let mut fw = SimulationFramework::new("analysis");
    assert!(fw.select_tool_by_name("ComplexMappingTool"));
    fw.export_queue.destination_dir = Some(temp_dir.clone());
    fw.active_tool
        .as_mut()
        .unwrap()
        .import_preset_json(decoded.state_json.as_deref().unwrap())
        .expect("import decoded URL state");

    let ctx = egui::Context::default();
    fw.queue_export(ExportFormat::Glsl);
    let _ = ctx.run(egui::RawInput::default(), |c| fw.show(c, "test_cm_glsl"));
    let fname = fw
        .export_queue
        .last_exported_filename
        .clone()
        .expect("saved .glsl file");
    assert!(fname.ends_with(".glsl"));
    let glsl_disk = std::fs::read_to_string(temp_dir.join(&fname)).expect("read .glsl");
    assert!(glsl_disk.contains("c_cos(z)"));
    let _ = std::fs::remove_dir_all(&temp_dir);
}
