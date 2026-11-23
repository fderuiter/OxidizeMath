#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

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

fn collect_rendered_texts(shapes: &[egui::epaint::Shape], out: &mut Vec<String>) {
    for shape in shapes {
        match shape {
            egui::epaint::Shape::Text(t) => out.push(t.galley.job.text.clone()),
            egui::epaint::Shape::Vec(inner) => collect_rendered_texts(inner, out),
            _ => {}
        }
    }
}

#[test]
fn test_custom_complex_mapping_unbound_c_renders_scope_error_and_undefined_metrics() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        ..Default::default()
    };
    tool.math_input.set_expression("z^2 + c");
    assert!(!tool.math_input.is_valid());
    let err = tool.math_input.error_message().expect("scope error for c");
    assert!(err.contains("Unknown variable 'c'") && err.contains("z, a"));

    let w_probe = tool.eval_z(Complex64::new(1.0, 1.0));
    assert!(w_probe.re.is_nan() && w_probe.im.is_nan());

    let ctx = egui::Context::default();
    let out = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    assert!(
        tool.cache.w_vertical.is_empty() && tool.cache.w_horizontal.is_empty(),
        "Invalid custom expression must omit mapped w-plane contours"
    );

    let shapes: Vec<_> = out.shapes.into_iter().map(|cs| cs.shape).collect();
    let mut texts = Vec::new();
    collect_rendered_texts(&shapes, &mut texts);
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Syntax Error:") && t.contains("Unknown variable 'c'")),
        "Expected inline scope error in {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("Undefined")),
        "Expected Undefined readout in {texts:?}"
    );
    assert!(
        !texts.iter().any(|t| t.contains("1000000.000")),
        "Must never render clamped 1000000.000 readout in {texts:?}"
    );
}

fn hover_frame_input(pos: egui::Pos2) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        events: vec![egui::Event::PointerMoved(pos)],
        ..Default::default()
    }
}

fn run_tool_frame(ctx: &egui::Context, tool: &mut ComplexMappingTool, raw: egui::RawInput) {
    let _ = ctx.run(raw, |c| {
        egui_plot::clear_last_plot_datasets();
        tool.show(c);
    });
}

#[test]
fn test_conformal_cache_reuse_invalidation_and_invalid_bounds_clearing() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut tool = ComplexMappingTool::default();
    let ctx = egui::Context::default();
    run_tool_frame(&ctx, &mut tool, hover_frame_input(egui::pos2(300.0, 300.0)));
    assert_eq!(tool.cache.eval_count, 1);
    let w_cap = tool.cache.w_vertical.points.capacity();
    assert!(w_cap > 0);

    // Idle, pointer hover, and inspection_point updates reuse cached contours without recomputing
    run_tool_frame(&ctx, &mut tool, egui::RawInput::default());
    run_tool_frame(&ctx, &mut tool, hover_frame_input(egui::pos2(720.0, 340.0)));
    tool.inspection_point = [0.25, -1.5];
    run_tool_frame(&ctx, &mut tool, hover_frame_input(egui::pos2(740.0, 360.0)));
    assert_eq!(tool.cache.eval_count, 1);

    // Grid bounds, density, function, Custom formula, param, and fallback edits invalidate cache
    tool.grid_min = -3.0;
    run_tool_frame(&ctx, &mut tool, egui::RawInput::default());
    assert_eq!(tool.cache.eval_count, 2);
    assert!(tool.cache.w_vertical.points.capacity() >= w_cap);

    tool.grid_max = 3.0;
    tool.grid_density = 8;
    run_tool_frame(&ctx, &mut tool, egui::RawInput::default());
    assert_eq!(tool.cache.eval_count, 3);

    tool.function = MappingFunction::Custom;
    run_tool_frame(&ctx, &mut tool, egui::RawInput::default());
    assert_eq!(tool.cache.eval_count, 4);

    tool.math_input.set_expression("z + a^2 / z");
    run_tool_frame(&ctx, &mut tool, egui::RawInput::default());
    assert_eq!(tool.cache.eval_count, 5);

    tool.math_input.set_param("a", 0.6);
    run_tool_frame(&ctx, &mut tool, egui::RawInput::default());
    assert_eq!(tool.cache.eval_count, 6);

    tool.math_input.set_expression("z +");
    run_tool_frame(&ctx, &mut tool, egui::RawInput::default());
    assert_eq!(tool.cache.eval_count, 7);
    assert!(!tool.cache.z_vertical.is_empty() && tool.cache.w_vertical.is_empty());

    // Invalid grid bounds clear cached contour buffers and recover cleanly once valid
    tool.grid_min = 4.0;
    tool.grid_max = 2.0;
    run_tool_frame(&ctx, &mut tool, egui::RawInput::default());
    assert!(tool.cache.z_vertical.is_empty() && tool.cache.w_vertical.is_empty());

    tool.math_input.set_expression("z + a^2 / z");
    tool.grid_min = -2.0;
    run_tool_frame(&ctx, &mut tool, egui::RawInput::default());
    assert_eq!(tool.cache.eval_count, 8);
    assert!(!tool.cache.w_vertical.is_empty());
}

fn assert_contour_batch_linear_and_log_shapes(
    z_batch: &cache::ContourBatch<'_>,
    w_batch: &cache::ContourBatch<'_>,
    frame: egui::Rect,
    lin_tf: &egui_plot::PlotTransform,
) {
    use egui_plot::{PlotBounds, PlotItem as _, PlotTransform};

    let mut log_tf = PlotTransform::new(
        frame,
        PlotBounds::from_min_max([0.01, 0.01], [10.0, 10.0]),
        true,
    );
    log_tf.set_log_axis(egui::Vec2b::new(true, false));
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            let mut z_shapes = Vec::new();
            z_batch.shapes(ui, lin_tf, &mut z_shapes);
            assert_eq!(z_shapes.len(), 5);
            assert!(z_shapes
                .iter()
                .all(|s| matches!(s, egui::Shape::LineSegment { .. })));

            let mut w_log_shapes = Vec::new();
            w_batch.shapes(ui, &log_tf, &mut w_log_shapes);
            assert!(!w_log_shapes.is_empty());
            for shape in &w_log_shapes {
                if let egui::Shape::Path(path_shape) = shape {
                    assert!(path_shape
                        .points
                        .iter()
                        .all(|p| p.x.is_finite() && p.y.is_finite()));
                }
            }
        });
    });
}

#[test]
fn test_contour_batch_shapes_hover_log_split_and_nan_free_extract_points() {
    use egui_plot::{PlotBounds, PlotItem as _, PlotTransform};

    let mut tool = ComplexMappingTool {
        function: MappingFunction::Inverse,
        grid_density: 5,
        ..Default::default()
    };
    assert!(tool.ensure_cached_contours());

    let z_batch = cache::ContourBatch::new(
        "Re(z) grid",
        &tool.cache.z_vertical,
        egui::Color32::LIGHT_BLUE,
        tool.cache.generation,
    );
    let w_batch = cache::ContourBatch::new(
        "u-contours",
        &tool.cache.w_vertical,
        egui::Color32::LIGHT_BLUE,
        tool.cache.generation,
    );
    assert!(z_batch
        .extract_points()
        .into_iter()
        .chain(w_batch.extract_points())
        .all(|pt| pt[0].is_finite() && pt[1].is_finite()));
    assert!(w_batch.extract_export_primitives().is_empty());

    let frame = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 400.0));
    let lin_tf = PlotTransform::new(
        frame,
        PlotBounds::from_min_max([-5.0, -5.0], [5.0, 5.0]),
        true,
    );
    let mid_pos = lin_tf.position_from_point(&egui_plot::PlotPoint::new(-1.02, 0.35));
    let (snap_pt, _) = z_batch.closest_continuous_point(mid_pos, &lin_tf).unwrap();
    assert!((snap_pt.x - (-1.0)).abs() < 1e-4 && (snap_pt.y - 0.35).abs() < 1e-3);
    assert_contour_batch_linear_and_log_shapes(&z_batch, &w_batch, frame, &lin_tf);
}

#[test]
fn test_state_revision_updates_across_all_tracked_inputs() {
    let mut tool = ComplexMappingTool::default();
    let r0 = tool.state_revision();
    assert!(r0 > 0);

    tool.function = MappingFunction::Sine;
    let r1 = tool.state_revision();
    assert!(r1 > 0 && r1 != r0);

    tool.grid_min = -3.5;
    let r2 = tool.state_revision();
    assert_ne!(r2, r1);

    tool.grid_max = 3.5;
    let r3 = tool.state_revision();
    assert_ne!(r3, r2);

    tool.grid_density = 16;
    let r4 = tool.state_revision();
    assert_ne!(r4, r3);

    tool.inspection_point = [0.25, -0.75];
    let r5 = tool.state_revision();
    assert_ne!(r5, r4);

    tool.math_input.set_expression("sin(z) + 1/z");
    let r6 = tool.state_revision();
    assert_ne!(r6, r5);
}

fn dispatch_framework_export(
    fw: &mut crate::framework::SimulationFramework,
    ctx: &egui::Context,
    fmt: crate::framework::ExportFormat,
) {
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |c| {
        egui_plot::clear_last_plot_datasets();
        fw.show_export_modal = fmt != crate::framework::ExportFormat::Csv;
        fw.queue_export(fmt);
        fw.show(c, "cm_export_fw");
    });
    assert!(
        !fw.is_diverged(),
        "FiniteStateGuard must not diverge on pole contours"
    );
    assert_eq!(
        fw.export_status_severity,
        crate::framework::ExportStatusSeverity::Success
    );
}

#[test]
fn test_finite_state_guard_and_disjoint_contour_exports_png_svg_gif_csv() {
    use crate::framework::{ExportFormat, SimulationFramework};

    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut pole_tool = ComplexMappingTool {
        function: MappingFunction::Custom,
        grid_density: 5,
        inspection_point: [0.0, 0.0],
        ..Default::default()
    };
    pole_tool.math_input.set_expression("z^3 - 1/z");

    // Cold export_plot_primitives builds disjoint polylines before any UI frame runs
    let cold_prims = pole_tool.export_plot_primitives();
    let polyline_count = cold_prims
        .iter()
        .filter(|p| matches!(p, egui_plot::ExportPrimitive::Polyline { .. }))
        .count();
    assert!(polyline_count > 4 * pole_tool.grid_density);

    let mut fw = SimulationFramework::new("__complex_mapping_export_test__");
    fw.export_options.width = 320;
    fw.export_options.height = 240;
    fw.export_gif_frames = 4;
    fw.active_tool = Some(Box::new(pole_tool));

    let ctx = egui::Context::default();
    dispatch_framework_export(&mut fw, &ctx, ExportFormat::Png);
    let png = fw.export_queue.last_exported_bytes.as_deref().unwrap();
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));

    dispatch_framework_export(&mut fw, &ctx, ExportFormat::Svg);
    let svg = fw.export_queue.last_exported_content.as_deref().unwrap();
    assert!(!svg.contains("NaN"));
    assert!(svg.matches("<polyline ").count() >= polyline_count);

    dispatch_framework_export(&mut fw, &ctx, ExportFormat::Gif);
    let gif = fw.export_queue.last_exported_bytes.as_deref().unwrap();
    assert!(gif.starts_with(b"GIF89a"));

    dispatch_framework_export(&mut fw, &ctx, ExportFormat::Csv);
    let csv = fw.export_queue.last_exported_content.as_deref().unwrap();
    assert!(csv.contains("Re(z) grid,") && csv.contains("Im(z) grid,"));
    assert!(csv.contains("\"u-contours\",") && csv.contains("\"v-contours\","));
    assert!(!csv.contains("NaN") && !csv.contains("inf"));
}
