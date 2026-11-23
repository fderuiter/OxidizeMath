#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{ExportFormat, ShaderExportFormat, SimulationFramework};
use crate::url_state::{encode_share_fragment, parse_share_location, validate_url_preset_json};
use std::collections::BTreeMap;

#[test]
fn test_custom_complex_mapping_evaluation() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("z^3 - 1/z");

    let z0 = Complex64::new(2.0, 0.0);
    let w0 = tool.eval_z(z0);
    assert!((w0.re - 7.5).abs() < 1e-6);
    assert!(w0.im.abs() < 1e-6);

    let deriv = tool.deriv_z(z0);
    // f'(z) = 3z^2 + 1/z^2 = 3*(4) + 0.25 = 12.25
    assert!((deriv.re - 12.25).abs() < 1e-4);
    assert!(deriv.im.abs() < 1e-4);
}

#[test]
fn test_custom_complex_mapping_division_by_zero_splits_contours_without_fallback_spikes() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("z^3 - 1/z");

    let z_zero = Complex64::new(0.0, 0.0);
    let w_zero = tool.eval_z(z_zero);
    assert!(w_zero.re.is_nan() && w_zero.im.is_nan());

    // Sweep real axis [-2, 2] across z = 0: must split into >= 2 segments and emit 0 (1e6, 1e6) vertices
    let segments = ui::split_contour_segments(161, 4.0, |j| {
        let x = -2.0 + 4.0 * (j as f64) / 160.0;
        tool.eval_z(Complex64::new(x, 0.0))
    });
    assert!(
        segments.len() >= 2,
        "Expected contour across z = 0 pole to split into at least 2 segments, got {}",
        segments.len()
    );
    for seg in &segments {
        for pt in seg {
            assert!(pt[0].is_finite() && pt[1].is_finite());
            assert_ne!(*pt, [1e6, 1e6]);
        }
    }
}

#[test]
fn test_custom_expr_and_params_state_roundtrip_and_legacy_compat() {
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("z + a^2 / z");
    let mut params = BTreeMap::new();
    params.insert("a".to_string(), 0.85);
    tool.math_input.set_params_map(&params);

    let saved = tool.save_state().expect("serialize ComplexMappingTool");
    let mut restored = ComplexMappingTool::default();
    restored.try_load_state(&saved).expect("load saved state");
    assert_eq!(restored.math_input.raw_text, "z + a^2 / z");
    assert_eq!(restored.math_input.params_map().get("a"), Some(&0.85));
    let w = restored.eval_z(Complex64::new(1.0, 0.0));
    assert!((w.re - (1.0 + 0.85 * 0.85)).abs() < 1e-12);
    let wgsl = restored
        .export_shader_source(ShaderExportFormat::Wgsl)
        .expect("export WGSL for parameterized Joukowsky map");
    assert!(wgsl.contains("vec2<f32>(0.85, 0.0)"));
    assert!(restored.import_preset_json(STARTER_PRESET_JSON).is_ok());
}

#[test]
fn test_exponential_and_sine_guards_and_bounds_validation() {
    let exp_tool = ComplexMappingTool {
        function: MappingFunction::Exponential,
        ..Default::default()
    };
    let w_exp = exp_tool.eval_z(Complex64::new(1_000.0, 1.0));
    let dw_exp = exp_tool.deriv_z(Complex64::new(1_000.0, 1.0));
    assert!(w_exp.re.is_finite() && w_exp.im.is_finite());
    assert!(dw_exp.re.is_finite() && dw_exp.im.is_finite());

    let sin_tool = ComplexMappingTool {
        function: MappingFunction::Sine,
        ..Default::default()
    };
    let w_sin = sin_tool.eval_z(Complex64::new(0.5, 1_000.0));
    let dw_sin = sin_tool.deriv_z(Complex64::new(0.5, -1_000.0));
    assert!(w_sin.re.is_finite() && w_sin.im.is_finite());
    assert!(dw_sin.re.is_finite() && dw_sin.im.is_finite());

    let mut check = ComplexMappingTool::default();
    assert!(check.has_finite_parameters());
    assert!(check
        .try_load_state(
            r#"{"function":"Exponential","grid_min":-60.0,"grid_max":2.0,"grid_density":10,"inspection_point":[1.0,1.0]}"#
        )
        .is_err());
    assert!(check
        .try_load_state(
            r#"{"function":"Sine","grid_min":-2.0,"grid_max":2.0,"grid_density":10,"inspection_point":[55.0,1.0]}"#
        )
        .is_err());
}

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

#[test]
fn test_complex_mapping_clamps_grid_density_and_caches_contours_across_probe_moves() {
    let mut tool = ComplexMappingTool::default();
    tool.try_load_state(
        r#"{"function":"ZSquared","grid_min":-2.0,"grid_max":2.0,"grid_density":1000,"inspection_point":[1.0,0.5]}"#,
    )
    .expect("grid_density 1000 should load and clamp");
    assert_eq!(tool.grid_density, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
    assert!(tool.has_finite_parameters());

    let mut live = ComplexMappingTool::default();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| live.show(c));
    let _ = ctx.run(egui::RawInput::default(), |c| live.show(c));
    assert_eq!(live.cache.eval_count, 1);

    live.inspection_point = [-0.75, 1.25];
    let _ = ctx.run(egui::RawInput::default(), |c| live.show(c));
    assert_eq!(live.cache.eval_count, 1);
}
